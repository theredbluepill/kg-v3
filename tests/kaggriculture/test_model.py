from __future__ import annotations

import pytest
import torch
from owl.kaggriculture import actor_codec
from owl.kaggriculture.types import (
    MAX_ACTORS,
    MAX_FRAMES,
    KaggricultureActionConfig,
    KaggricultureActionMask,
    KaggricultureActions,
    KaggricultureObsBatch,
    KaggricultureObsConfig,
)
from owl.model.kaggriculture import (
    SLOT_NAMES,
    KaggricultureTransformer,
    KaggricultureTransformerConfig,
    _categorical_choice,
)


@pytest.fixture(autouse=True)
def single_threaded_torch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _observation(actors: int = 1, *, orders: int = 2) -> KaggricultureObsBatch:
    features = torch.zeros(1, 2, 8176)
    features[..., 1024] = 1
    features[..., 1025:1027] = actors
    positions = features[..., 3827:5273].reshape(1, 2, 2, MAX_ACTORS, 3)
    positions[..., :actors, 0] = 1
    positions[..., :actors, 1] = torch.arange(actors) % 10
    positions[..., :actors, 2] = (torch.arange(actors) // 10) % 10
    features[..., 8165:] = torch.tensor([1, 0, 0, 0, 720, 24, 0, 1, 1, 100, orders])
    return KaggricultureObsBatch(
        features=features,
        context=torch.tensor([[0, actors, actors, orders]]),
        entity_mask=(torch.arange(MAX_ACTORS)[None, None, :] < actors).expand(1, 2, -1),
        still_playing=torch.ones(1, 2, dtype=torch.bool),
        action_mask=KaggricultureActionMask(
            can_act=(
                torch.arange(MAX_FRAMES)[None, None, :] < actors + orders + 1
            ).expand(1, 2, -1)
        ),
    )


def _model(**overrides) -> KaggricultureTransformer:
    torch.manual_seed(31)
    return KaggricultureTransformer(
        KaggricultureTransformerConfig(
            embed_dim=16,
            depth=1,
            n_heads=2,
            mlp_ratio=2,
            n_scratch_tokens=1,
            **overrides,
        ),
        obs_spec=KaggricultureObsConfig(),
        action_spec=KaggricultureActionConfig(),
    )


def _pass_program(actors: int, orders: int = 0) -> KaggricultureActions:
    tokens = torch.zeros(1, 2, MAX_FRAMES, 12, dtype=torch.int64)
    tokens[..., :actors, 0] = torch.arange(actors)
    tokens[..., :actors, 1] = 1  # PASS
    tokens[..., actors : actors + orders, 7] = 7  # EMPTY market slots remain explicit
    tokens[..., actors + orders, 11] = 1
    return KaggricultureActions(tokens, torch.full((1, 2), actors + orders + 1))


def test_default_model_obeys_owner_parameter_budget():
    model = KaggricultureTransformer(
        KaggricultureTransformerConfig(),
        obs_spec=KaggricultureObsConfig(),
        action_spec=KaggricultureActionConfig(),
    )
    count = sum(parameter.numel() for parameter in model.parameters())
    assert 6_000_000 <= count <= 10_000_000


@pytest.mark.parametrize("mode", ["win_loss", "win_only", "margin"])
def test_critic_range_matches_reward_objective(mode):
    model = _model(value_mode=mode)
    with torch.no_grad():
        model.critic_head[-1].weight.zero_()
        model.critic_head[-1].bias.fill_(3)
        values = model.compute_value(_observation())
    expected = torch.tensor(3.0)
    if mode == "win_loss":
        expected = expected.tanh()
    elif mode == "win_only":
        expected = expected.sigmoid()
    torch.testing.assert_close(values, expected.expand(1, 2))


def test_sampling_evaluation_density_entropy_and_native_oracle_agree():
    obs, model = _observation(3), _model()
    with torch.no_grad():
        sampled = model(obs)
        evaluated = model.evaluate_actions(obs, sampled.actions)
    assert isinstance(sampled.actions, KaggricultureActions)
    assert sampled.actions.tokens.shape == (1, 2, 252, 12)
    torch.testing.assert_close(
        sampled.log_probs.per_player_entity, evaluated.log_probs.per_player_entity
    )
    torch.testing.assert_close(
        sampled.entropies.per_player_entity, evaluated.entropies.per_player_entity
    )
    torch.testing.assert_close(sampled.values, evaluated.values)
    torch.testing.assert_close(
        sampled.log_probs.event.sum(-1), sampled.log_probs.per_player_entity
    )
    assert torch.isfinite(sampled.log_probs.per_player_entity).all()
    assert torch.isfinite(sampled.entropies.per_player_entity).all()
    # Actor ordinal, target and STOP are forced grammar choices.
    assert torch.count_nonzero(sampled.log_probs.event[..., [0, 2, 11]]) == 0
    legal = {
        "observation": {
            "player": 0,
            "farms": [{"hands": [[], []]}, {"hands": [[], []]}],
        },
        "configuration": {"maxMarketOrdersPerTurn": 2},
    }
    for seat in range(2):
        previous = []
        for row in sampled.actions.tokens[
            0, seat, : sampled.actions.lengths[0, seat]
        ].tolist():
            frame = dict(zip(SLOT_NAMES, row, strict=True))
            actor_codec.frame_masks(legal, previous, frame, hire_limit=241)
            previous.append(frame)
        assert previous[-1]["stop"] == 1


def test_policy_gradient_matches_finite_difference_and_updates_trunk():
    obs, model = _observation(), _model()
    actions = _pass_program(1)
    evaluated = model.evaluate_actions(obs, actions)
    joint = evaluated.log_probs.per_player_entity.sum()
    joint.backward()
    analytic = model.heads[1].bias.grad[1].item()
    epsilon = 0.002
    with torch.no_grad():
        model.heads[1].bias[1] += epsilon
        plus = (
            model.evaluate_actions(obs, actions)
            .log_probs.per_player_entity.sum()
            .item()
        )
        model.heads[1].bias[1] -= 2 * epsilon
        minus = (
            model.evaluate_actions(obs, actions)
            .log_probs.per_player_entity.sum()
            .item()
        )
        model.heads[1].bias[1] += epsilon
    assert analytic == pytest.approx((plus - minus) / (2 * epsilon), rel=2e-3, abs=2e-3)
    model.zero_grad(set_to_none=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.blocks[0].attn.q.weight.detach().clone()
    evaluated = model.evaluate_actions(obs, actions)
    loss = (
        -evaluated.log_probs.per_player_entity.sum()
        + (evaluated.values - 0.7).square().mean()
    )
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert model.critic_head[-1].weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(model.blocks[0].attn.q.weight, before)


@pytest.mark.parametrize(
    "command", [["PICKUP", "WHEAT"], ["PICKUP", "WHEAT", 1], ["PLACE", "COW", 1023]]
)
def test_quantity_omission_zero_market_and_empty_orders_survive(command):
    obs, model = _observation(), _model()
    legal = {
        "observation": {"player": 0, "farms": [{"hands": []}, {"hands": []}]},
        "configuration": {"maxMarketOrdersPerTurn": 2},
    }
    raw = {
        "farmer": command,
        "hands": [],
        "market": [["BUY_SEED", "WHEAT", 0], []],
    }
    encoded = actor_codec.encode_action(raw, legal, hire_limit=241)
    assert actor_codec.decode_action(encoded, legal, hire_limit=241) == raw
    tokens = torch.zeros(1, 2, MAX_FRAMES, 12, dtype=torch.int64)
    rows = torch.tensor(
        [[frame[name] for name in SLOT_NAMES] for frame in encoded["frames"]]
    )
    tokens[..., : len(rows), :] = rows
    actions = KaggricultureActions(tokens, torch.full((1, 2), len(rows)))
    with torch.no_grad():
        evaluated = model.evaluate_actions(obs, actions)
        state = model._encode(obs)
        grammar = model._grammar(state)
        nodes = grammar.starts
        previous = []
        for frame in encoded["frames"]:
            prefix = {}
            for slot, name in enumerate(SLOT_NAMES):
                expected = actor_codec.mask_for_slot(
                    name, legal, previous, prefix, hire_limit=241
                )
                torch.testing.assert_close(
                    grammar.options(nodes, slot)[0], torch.tensor(expected)
                )
                nodes = grammar.advance(nodes, torch.full((2,), frame[name]), slot)
                prefix[name] = frame[name]
            previous.append(frame)
    assert torch.isfinite(evaluated.log_probs.per_player_entity).all()


@pytest.mark.parametrize("actors", [17, 241])
def test_full_actor_and_market_capacity_is_represented(actors):
    obs, model = _observation(actors, orders=10), _model()
    actions = _pass_program(actors, orders=10)
    if actors == 17:
        actions.tokens[..., actors, 7] = 1  # HIRE is supported beyond the old 16 cap.
    with torch.no_grad():
        evaluated = model.evaluate_actions(obs, actions)
        original_value = evaluated.values.clone()
        # The highest own actor's inventory is represented, including slot 240.
        obs.features[..., 5273 + (actors - 1) * 12] = 100
        changed_value = model.compute_value(obs)
    assert actions.tokens[0, 0, actors - 1, 0] == actors - 1
    assert actions.lengths[0, 0] == actors + 11
    assert torch.isfinite(evaluated.log_probs.per_player_entity).all()
    assert not torch.equal(original_value, changed_value)


def test_padding_cannot_enter_product_inventory_totals():
    obs, model = _observation(), _model()
    with torch.no_grad():
        original = model.compute_value(obs)
        obs.features[..., 5273 + 12 : 8165] = 1000
        modified = model.compute_value(obs)
    torch.testing.assert_close(original, modified, rtol=0, atol=0)


def test_private_perspectives_are_isolated_and_checkpoint_reloads():
    obs, model = _observation(), _model()
    actions = _pass_program(1)
    with torch.no_grad():
        original = model.evaluate_actions(obs, actions)
        obs.features[:, 1, 5273:8165] = 1000
        obs.features[:, 1, 872:889] = 123
        modified = model.evaluate_actions(obs, actions)
    torch.testing.assert_close(
        original.values[:, 0], modified.values[:, 0], rtol=0, atol=0
    )
    torch.testing.assert_close(
        original.log_probs.per_player_entity[:, 0],
        modified.log_probs.per_player_entity[:, 0],
        rtol=0,
        atol=0,
    )
    restored = _model()
    restored.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        reloaded = restored.evaluate_actions(obs, actions)
    torch.testing.assert_close(reloaded.values, modified.values, rtol=0, atol=0)
    torch.testing.assert_close(
        reloaded.log_probs.per_player_entity,
        modified.log_probs.per_player_entity,
        rtol=0,
        atol=0,
    )


def test_invalid_program_and_insufficient_capacity_fail_closed():
    obs, model = _observation(), _model()
    actions = _pass_program(1)
    actions.tokens[..., 0, 1] = 0
    with pytest.raises(ValueError, match="violates native grammar"):
        model.evaluate_actions(obs, actions)
    with pytest.raises(ValueError, match="max_decode_frames"):
        _model(max_decode_frames=2)(obs)
    actions = _pass_program(1)
    actions.tokens[..., 10, 3] = 1
    with pytest.raises(ValueError, match="after STOP"):
        model.evaluate_actions(obs, actions)
    with pytest.raises(ValueError, match="no state"):
        model.compute_value(obs, hidden_state=torch.zeros(1))


def test_segment_major_evaluation_preserves_shapes():
    obs, model = _observation(), _model()
    batch = KaggricultureObsBatch(
        features=obs.features[:, None].expand(-1, 2, -1, -1),
        context=obs.context[:, None].expand(-1, 2, -1),
        entity_mask=obs.entity_mask[:, None].expand(-1, 2, -1, -1),
        still_playing=obs.still_playing[:, None].expand(-1, 2, -1),
        action_mask=KaggricultureActionMask(
            can_act=obs.action_mask.can_act[:, None].expand(-1, 2, -1, -1)
        ),
    )
    actions = _pass_program(1)
    sequence = KaggricultureActions(
        tokens=actions.tokens[:, None].expand(-1, 2, -1, -1, -1),
        lengths=actions.lengths[:, None].expand(-1, 2, -1),
    )
    with torch.no_grad():
        evaluated = model.evaluate_actions(batch, sequence)
    assert evaluated.values.shape == (1, 2, 2)
    assert evaluated.log_probs.per_player_entity.shape == (1, 2, 2, 252)


def test_gumbel_sampling_matches_categorical_frequencies_and_exact_support():
    torch.manual_seed(510)
    probabilities = torch.tensor([0.05, 0.15, 0.3, 0.5, 0.0])
    draws = _categorical_choice(probabilities.log().expand(100_000, -1))
    frequencies = torch.bincount(draws, minlength=5) / draws.numel()
    # A fixed-seed, six-standard-error band detects distribution mistakes while
    # avoiding an exact random-stream contract with torch.multinomial.
    tolerance = 6 * (probabilities * (1 - probabilities) / draws.numel()).sqrt()
    assert torch.all((frequencies - probabilities).abs() <= tolerance)
    singleton = torch.tensor([-torch.inf, 0.0, -torch.inf]).expand(256, -1)
    assert torch.equal(
        _categorical_choice(singleton), torch.ones(256, dtype=torch.long)
    )


def test_factored_frame_inputs_and_grammar_binding_preserve_contract():
    model = _model()
    encoded = model._encode(_observation(3))
    actor, market = model._decoder_inputs(encoded)
    for ordinal in range(3):
        original = model.frame_input(
            torch.cat((encoded.plan, encoded.own_actors[:, ordinal]), -1)
        )
        torch.testing.assert_close(actor[:, ordinal], original)
    original_market = model.frame_input(torch.cat((encoded.plan, encoded.plan), -1))
    torch.testing.assert_close(market, original_market)
    first = model._grammar(encoded)
    second = model._grammar(encoded)
    assert first is second
    assert first.starts.data_ptr() == second.starts.data_ptr()


@pytest.mark.parametrize("active", [True, False])
def test_every_parameter_participates_in_active_and_inactive_rank_backward(active):
    model, obs = _model(), _observation()
    actions = _pass_program(1)
    if not active:
        obs.still_playing.zero_()
        actions.tokens.zero_()
        actions.lengths.zero_()
    evaluated = model.evaluate_actions(obs, actions)
    loss = evaluated.log_probs.per_player_entity.sum() + evaluated.values.square().sum()
    loss.backward()
    missing = [
        name for name, parameter in model.named_parameters() if parameter.grad is None
    ]
    assert missing == []
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    for slot in (0, 2, 11):
        assert torch.count_nonzero(model.heads[slot].weight.grad) == 0
