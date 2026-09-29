"""Task 2.3: Kaggriculture grammar action heads (brief §1, §3 to §7, §9.3).

Tiny CPU shapes and the synthetic ``expected_grammar_tables`` until Task 1.2.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest
import torch
from owl.kaggriculture import gpu_grammar as gg
from owl.kaggriculture import types as kt
from owl.model import kaggriculture as km
from owl.model import kaggriculture_actor as ka
from owl.model.actor.common import OutputProjectionMLP
from owl.train.optimizer import MuonConfig, create_optimizer
from torch import nn

from tests.kaggriculture.conftest import make_obs
from tests.kaggriculture.helpers import (
    EMPTY,
    HIRE,
    NONE,
    PASS,
    F,
    K,
    Order,
    _base_case,
    _obs_double,
    _replay_case,
    _seat_program,
    _take_actions,
    _take_obs,
    _tiny,
)

ROOT = Path(__file__).resolve().parents[2]
HEAD_NAMES = {kt.SLOT_NAMES[s] for s in ka.POLICY_SLOTS}
IMPLICIT_SLOTS = [0, 2, 11]
ATOL = 1e-5


# --- helpers --------------------------------------------------------------------------


def _assert_saved(
    evaluation: Any, saved: dict[str, torch.Tensor], index: torch.Tensor | None = None
) -> None:
    def pick(t: torch.Tensor) -> torch.Tensor:
        return t if index is None else t.index_select(0, index)

    got = {
        "event": evaluation.log_probs.event,
        "frame": evaluation.log_probs.per_player_entity,
        "entropy": evaluation.entropies.event,
        "entropy_frame": evaluation.entropies.per_player_entity,
    }
    for name, value in got.items():
        torch.testing.assert_close(
            value, pick(saved[name]), rtol=0, atol=ATOL, msg=name
        )


# --- §7.1 replay invariant ------------------------------------------------------------


@pytest.mark.parametrize(
    "case", ["dense", "hire_capacity", "stop_every_position", "mixed"]
)
def test_replay_matches_saved_sample_whole_split_and_permuted(case: str) -> None:
    model, obs = _replay_case(case)
    model.eval()
    torch.manual_seed(11)
    with torch.no_grad():
        sampled = model(obs)
    actions = kt.KaggricultureActions(
        tokens=sampled.actions.tokens.clone(), lengths=sampled.actions.lengths.clone()
    )
    saved = {
        "event": sampled.log_probs.event.clone(),
        "frame": sampled.log_probs.per_player_entity.clone(),
        "entropy": sampled.entropies.event.clone(),
        "entropy_frame": sampled.entropies.per_player_entity.clone(),
    }
    torch.testing.assert_close(saved["frame"], saved["event"].sum(-1))
    envs = obs.still_playing.shape[0]
    with torch.no_grad():
        # (a) the same batch
        _assert_saved(model.evaluate_actions(obs, actions), saved)
        # (b) minibatches of different sizes
        for sizes in ([1] * envs, [envs - 1, 1] if envs > 1 else [1], [2, envs - 2]):
            if any(size <= 0 for size in sizes):
                continue
            start = 0
            for size in sizes:
                index = torch.arange(start, start + size)
                _assert_saved(
                    model.evaluate_actions(
                        _take_obs(obs, index), _take_actions(actions, index)
                    ),
                    saved,
                    index,
                )
                start += size
        # (c) permuted, re-batched, restored to the original order
        perm = torch.randperm(envs, generator=torch.Generator().manual_seed(3))
        parts: list[Any] = []
        for chunk in perm.split(max(1, (envs + 1) // 2)):
            parts.append(
                model.evaluate_actions(
                    _take_obs(obs, chunk), _take_actions(actions, chunk)
                )
            )
        restore = torch.argsort(perm)
        merged = {
            "event": torch.cat([p.log_probs.event for p in parts])[restore],
            "frame": torch.cat([p.log_probs.per_player_entity for p in parts])[restore],
            "entropy": torch.cat([p.entropies.event for p in parts])[restore],
            "entropy_frame": torch.cat([p.entropies.per_player_entity for p in parts])[
                restore
            ],
        }
        for name, value in merged.items():
            torch.testing.assert_close(value, saved[name], rtol=0, atol=ATOL, msg=name)

    tokens, lengths = actions.tokens, actions.lengths
    live = obs.still_playing
    counts = obs.actor_mask[..., : kt.MAX_ACTORS].sum(-1)
    stops = lengths - counts - 1
    if case == "dense":
        assert bool((counts == 241).all())
    if case == "hire_capacity":
        hires = (tokens[..., 7] == HIRE).sum(-1)
        budget = (241 - counts).clamp_min(0)
        assert bool((hires <= budget).all())
        corrected = (hires == budget) & (stops > budget)
        assert bool(corrected.any()), "no HIRE capacity correction was exercised"
    if case == "stop_every_position":
        assert sorted(stops[live].unique().tolist()) == list(range(11))
    if case == "mixed":
        assert bool((lengths[~live] == 0).all())
        assert int(tokens[~live].count_nonzero()) == 0
        assert int(saved["event"][~live].count_nonzero()) == 0
        assert len(set(stops[live].tolist())) > 1


# --- §7.3 Gumbel exactness ------------------------------------------------------------


def test_gumbel_draws_match_the_masked_softmax() -> None:
    torch.manual_seed(1234)
    logits = torch.tensor([0.3, -1.0, 1.2, 0.0, 2.0, -0.5])
    mask = torch.tensor([True, True, False, True, True, False])
    masked = logits.masked_fill(~mask, -torch.inf)
    draws = ka.gumbel_argmax(masked.expand(20_000, -1))
    counts = torch.bincount(draws, minlength=6).double()
    assert int(counts[~mask].sum()) == 0
    expected = masked.softmax(-1).double()[mask] * draws.numel()
    chi2 = float(((counts[mask] - expected) ** 2 / expected).sum())
    # df = 3; the 99.99% chi-square quantile is 21.11.
    assert chi2 < 21.11
    assert bool((counts[mask] > 0).all())  # not a disguised argmax


# --- §7.4 HIRE coupling by enumeration ------------------------------------------------


def _enumerate_coupled_sampler(
    logits: torch.Tensor, hire_limit: int, actors: int
) -> tuple[dict[tuple[int, ...], float], set[int]]:
    """Joint final-program probabilities of the actual coupled sampler.

    Per available position the Gumbel-max outcome is (raw argmax ``r``, and for
    ``r = HIRE`` the argmax ``k`` over the other kinds), with probability
    ``softmax(r)`` and ``softmax(HIRE) * softmax_without_HIRE(k)``. Perturbed
    scores realizing every outcome combination drive ``couple_market_kinds``.
    """
    positions = logits.shape[0]
    full = logits.softmax(-1)
    hire_column = torch.arange(8) == HIRE
    restricted = logits.masked_fill(hire_column, -torch.inf).softmax(-1)
    outcomes: list[list[tuple[int, int | None, float]]] = []
    for p in range(positions):
        row: list[tuple[int, int | None, float]] = []
        for raw in range(8):
            if raw != HIRE:
                row.append((raw, None, float(full[p, raw])))
            else:
                row.extend(
                    (HIRE, k, float(full[p, HIRE] * restricted[p, k]))
                    for k in range(8)
                    if k != HIRE
                )
        outcomes.append(row)
    combos = list(itertools.product(*outcomes))
    perturbed = torch.full((len(combos), ka.MARKET_POSITIONS, 8), -torch.inf)
    perturbed = perturbed.double()
    perturbed[:, positions:, NONE] = 0.0  # sentinel and beyond: NONE only
    for index, combo in enumerate(combos):
        for p, (raw, k, _) in enumerate(combo):
            perturbed[index, p] = 0.0
            perturbed[index, p, raw] = 10.0
            if k is not None:
                perturbed[index, p, k] = 5.0
    final = ka.couple_market_kinds(
        perturbed, torch.full((len(combos),), actors), hire_limit
    ).tolist()
    probabilities: dict[tuple[int, ...], list[float]] = defaultdict(list)
    corrections: set[int] = set()
    for combo, kinds in zip(combos, final, strict=True):
        stop = kinds.index(NONE)
        probabilities[tuple(kinds[: stop + 1])].append(
            math.prod(prob for _, _, prob in combo)
        )
        for p, (raw, _, _) in enumerate(combo[: stop + 1]):
            if raw == HIRE and kinds[p] != HIRE:
                corrections.add(kinds[p])
    return {k: math.fsum(v) for k, v in probabilities.items()}, corrections


@pytest.mark.parametrize("budget", [0, 1, 2, 3, 10])
def test_hire_coupling_enumeration_equals_replayed_density(budget: int) -> None:
    actors, orders = 1, 3
    hire_limit = actors + budget
    actor = _tiny(seed=9).actor.double()
    with torch.no_grad():
        head = actor.heads["market_kind"].out
        head.weight.normal_(0.0, 0.6, generator=torch.Generator().manual_seed(1))
        head.bias.copy_(torch.tensor([-0.4, 0.9, 0.1, 0.3, -0.2, 0.0, 0.2, 0.5]))
        generator = torch.Generator().manual_seed(2)
        market_input = torch.randn(1, 16, dtype=torch.float64, generator=generator)
        logits = actor.slot_logits(7, actor.market_base(market_input))[0, :orders]
    enumerated, corrections = _enumerate_coupled_sampler(logits, hire_limit, actors)
    assert math.fsum(enumerated.values()) == pytest.approx(1.0, abs=1e-9)
    if budget < orders:
        assert {NONE, EMPTY} <= corrections
    else:
        assert not corrections

    programs = sorted(enumerated)
    tables = gg.expected_grammar_tables()
    rows = []
    for kinds in programs:
        orders_list: list[Order] = []
        for kind in kinds[:-1]:
            item = int(tables.market_item[kind].nonzero()[0])
            orders_list.append((kind, item, 0, 0))
        rows.append(_seat_program([(PASS, 0, 0, 0, 0)], orders_list))
    count = len(rows)
    tokens = torch.stack([t for t, _ in rows])
    lengths = torch.tensor([n for _, n in rows])
    with torch.no_grad():
        result = actor.policy_core(
            torch.randn(count, kt.MAX_ACTORS, 16, dtype=torch.float64),
            market_input.expand(count, -1),
            torch.full((count,), actors),
            torch.full((count,), orders),
            torch.ones(count, dtype=torch.bool),
            hire_limit,
            tokens,
            lengths,
            False,
        )
    assert bool(result.valid.all())
    replayed = result.log_probs[:, :, 7].sum(-1).exp().tolist()
    for kinds, density in zip(programs, replayed, strict=True):
        assert density == pytest.approx(enumerated[kinds], abs=1e-9), kinds
    assert math.fsum(replayed) == pytest.approx(1.0, abs=1e-9)
    # Programs exceeding the budget are unreachable and absent.
    assert all(sum(k == HIRE for k in kinds) <= budget for kinds in programs)


# --- §7.5 STOP and marginalization ----------------------------------------------------


def test_stop_keeps_its_density_the_sentinel_has_none_and_later_frames_are_zero() -> (
    None
):
    model = _tiny().eval()
    obs, actions = _base_case()
    with torch.no_grad():
        evaluation = model.evaluate_actions(obs, actions)
    event, entropy = evaluation.log_probs.event, evaluation.entropies.event
    # seat 0: 2 unit frames, orders at frames 2-3, STOP (first NONE) at frame 4
    assert float(event[0, 0, 4, 7]) < 0.0
    assert float(entropy[0, 0, 4, 7]) > 0.0
    assert int(event[0, 0, 4, 8:].count_nonzero()) == 0  # forced item/digits
    # seat 1: 3 unit frames, 3 orders, STOP at the forced sentinel (frame 6)
    assert float(event[0, 1, 6, 7]) == 0.0
    assert float(entropy[0, 1, 6, 7]) == 0.0
    for seat, length in ((0, 5), (1, 7)):
        assert int(actions.lengths[0, seat]) == length
        assert int(event[0, seat, length:].count_nonzero()) == 0
        assert int(entropy[0, seat, length:].count_nonzero()) == 0
        assert bool((event[0, seat, :length, 7:11].sum(-1) != 0).any())
    assert int(event[..., IMPLICIT_SLOTS].count_nonzero()) == 0
    assert int(entropy[..., IMPLICIT_SLOTS].count_nonzero()) == 0
    torch.testing.assert_close(
        evaluation.log_probs.per_player_entity, event.sum(-1), rtol=0, atol=0
    )
    torch.testing.assert_close(
        evaluation.entropies.per_player_entity, entropy.sum(-1), rtol=0, atol=0
    )
    assert torch.equal(evaluation.log_probs.launch, torch.zeros_like(event[..., 0]))
    for slot, name in enumerate(kt.SLOT_NAMES):
        assert torch.equal(evaluation.entropies.components[name], entropy[..., slot])


def test_sampled_lengths_count_every_frame_including_stop() -> None:
    model = _tiny().eval()
    obs = make_obs(envs=3, own_actors=(1, 4, 9), rival_actors=(2, 3, 1), order_limit=6)
    torch.manual_seed(4)
    with torch.no_grad():
        sampled = model(obs)
    tokens, lengths = sampled.actions.tokens, sampled.actions.lengths
    counts = obs.actor_mask[..., : kt.MAX_ACTORS].sum(-1)
    for env in range(3):
        for seat in range(2):
            n, count = int(lengths[env, seat]), int(counts[env, seat])
            row = tokens[env, seat]
            assert count + 1 <= n <= count + 6 + 1
            assert int(row[n - 1, 11]) == 1
            assert int(row[n - 1, 7]) == NONE
            assert int(row[: n - 1, 11].sum()) == 0
            assert int(row[n:].count_nonzero()) == 0
            assert row[:count, 0].tolist() == list(range(count))
            assert int(sampled.log_probs.event[env, seat, n:].count_nonzero()) == 0


# --- §7.6 malformed-program rejection -------------------------------------------------


def _rejected(
    model: Any, obs: kt.KaggricultureObsBatch, actions: kt.KaggricultureActions
) -> tuple[str, ...]:
    with pytest.raises(ka.GrammarReplayError) as error, torch.no_grad():
        model.evaluate_actions(obs, actions)
    for group in error.value.groups:
        assert group in str(error.value)
    return error.value.groups


def test_the_base_programs_are_admitted() -> None:
    model = _tiny().eval()
    obs, actions = _base_case()
    with torch.no_grad():
        model.evaluate_actions(obs, actions)


@pytest.mark.parametrize(
    ("seat", "frame", "slot", "value", "groups"),
    [
        (0, 1, 3, 3, ("support",)),  # PASS carries an item
        (0, 0, 5, 0, ("support",)),  # explicit unit zero (low set to 0 below)
        (0, 2, 8, 12, ("support",)),  # SELL item outside 1..9
        (0, 0, 1, 0, ("support",)),  # unit NONE for a live actor
        (0, 0, 1, 19, ("support",)),  # always-masked unit kind
        # A non-NONE kind at the forced sentinel also moves the reconstructed
        # STOP, so length and canonical equality fail with it.
        (1, 6, 7, EMPTY, ("support", "length", "canonical")),
    ],
)
def test_out_of_support_values_fail_the_support_group(
    seat: int, frame: int, slot: int, value: int, groups: tuple[str, ...]
) -> None:
    model = _tiny().eval()
    obs, actions = _base_case()
    if (seat, frame, slot) == (0, 0, 5):
        actions.tokens[0, 0, 0, 6] = 0
    actions.tokens[0, seat, frame, slot] = value
    assert _rejected(model, obs, actions) == groups


def test_hire_beyond_capacity_fails_support_only() -> None:
    # Seat 1 has 3 actors and one HIRE; hire_limit 4 leaves no second HIRE.
    model = _tiny(hire_limit=4).eval()
    obs, actions = _base_case()
    with torch.no_grad():
        model.evaluate_actions(obs, actions)
    actions.tokens[0, 1, 4, 7:11] = torch.tensor([HIRE, 0, 0, 0])
    assert _rejected(model, obs, actions) == ("support",)


def test_wrong_length_fails_the_length_group() -> None:
    model = _tiny().eval()
    obs, actions = _base_case()
    actions.lengths[0, 0] += 1
    assert _rejected(model, obs, actions) == ("length",)


@pytest.mark.parametrize(
    ("seat", "frame", "slot", "value"),
    [
        (0, 1, 0, 0),  # non-canonical actor ordinal
        (0, 0, 2, 5),  # reserved unit_target
        (0, 0, 7, 2),  # market field in a unit frame
        (0, 2, 1, 1),  # unit field in a market frame
        (0, 2, 0, 1),  # actor ordinal in a market frame
        (0, 3, 11, 1),  # STOP bit on a non-STOP frame
        (0, 4, 11, 0),  # missing STOP bit
        (0, 10, 3, 1),  # padding after STOP
        (1, 7, 7, EMPTY),  # a market order after the sentinel STOP
    ],
)
def test_non_canonical_tokens_fail_the_canonical_group(
    seat: int, frame: int, slot: int, value: int
) -> None:
    model = _tiny().eval()
    obs, actions = _base_case()
    actions.tokens[0, seat, frame, slot] = value
    assert _rejected(model, obs, actions) == ("canonical",)


def test_inactive_rows_must_be_all_zero() -> None:
    model = _tiny().eval()
    obs, actions = _base_case()
    obs.still_playing[0, 1] = False
    actions.lengths[0, 1] = 0
    assert _rejected(model, obs, actions) == ("canonical",)
    actions.tokens[0, 1] = 0
    with torch.no_grad():
        evaluation = model.evaluate_actions(obs, actions)
    assert int(evaluation.log_probs.event[0, 1].count_nonzero()) == 0
    assert int(evaluation.entropies.event[0, 1].count_nonzero()) == 0
    actions.lengths[0, 1] = 1
    assert _rejected(model, obs, actions) == ("length",)


@pytest.mark.parametrize(
    ("seat", "frame", "slot", "value"),
    [
        (0, 0, 1, 999),
        (0, 0, 3, -7),
        (0, 0, 5, 10**6),
        (0, 2, 7, 10**9),
        (0, 2, 8, -5),
        (1, 3, 7, -1),
        (0, 2, 10, 2**40),
    ],
)
def test_out_of_range_indices_are_rejected_without_a_fault(
    seat: int, frame: int, slot: int, value: int
) -> None:
    model = _tiny().eval()
    obs, actions = _base_case()
    actions.tokens[0, seat, frame, slot] = value
    groups = _rejected(model, obs, actions)
    assert "support" in groups
    assert "canonical" in groups  # a clamped index is never admitted


def test_wrong_dtype_or_shape_fails_before_any_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny().eval()
    obs, actions = _base_case()
    monkeypatch.setattr(
        model, "encode_observations", lambda *_: pytest.fail("encoder ran")
    )
    cases = [
        (actions.tokens.int(), actions.lengths, "int64"),
        (actions.tokens, actions.lengths.int(), "int64"),
        (actions.tokens[:, :, :251], actions.lengths, "shape"),
        (actions.tokens[..., :11], actions.lengths, "shape"),
        (actions.tokens, actions.lengths[:, :1], "shape"),
        (actions.tokens.reshape(2, F, K), actions.lengths, "shape"),
    ]
    for tokens, lengths, message in cases:
        with pytest.raises(ValueError, match=message):
            model.evaluate_actions(
                obs, kt.KaggricultureActions(tokens=tokens, lengths=lengths)
            )


@pytest.mark.parametrize(
    ("present", "value", "admitted"),
    [
        (0, 0, True),  # omitted unit quantity
        (1, 0, False),  # rejected explicit unit zero
        (1, 1, True),
        (1, 31, True),
        (1, 32, True),
        (1, 1023, True),
    ],
)
def test_unit_quantity_cases_through_replay(
    present: int, value: int, admitted: bool
) -> None:
    model = _tiny().eval()
    obs, actions = _base_case()
    high, low = divmod(value, 32)
    actions.tokens[0, 0, 0, 3:7] = torch.tensor([2, present, high, low])
    if admitted:
        with torch.no_grad():
            model.evaluate_actions(obs, actions)
    else:
        assert _rejected(model, obs, actions) == ("support",)


@pytest.mark.parametrize("value", [0, 1, 1023])
def test_market_quantity_zero_is_accepted_through_replay(value: int) -> None:
    model = _tiny().eval()
    obs, actions = _base_case()
    actions.tokens[0, 0, 2, 9:11] = torch.tensor(divmod(value, 32))
    with torch.no_grad():
        model.evaluate_actions(obs, actions)


# --- §7.7 compiled density/gradient and finite differences ----------------------------


def _loss(evaluation: Any) -> torch.Tensor:
    return (
        evaluation.log_probs.per_player_entity.sum()
        + 0.1 * evaluation.entropies.per_player_entity.sum()
        + evaluation.values.square().sum()
    )


def test_fullgraph_captured_core_matches_eager_density_and_gradients() -> None:
    """Capture the head core with ``backend="eager", fullgraph=True``.

    A CPU graph-compatibility check only; production heads stay eager. Host
    validation runs outside the captured core.
    """
    model = _tiny()
    obs, actions = _base_case()
    eager = model.evaluate_actions(obs, actions)
    _loss(eager).backward()
    eager_grads = {
        name: p.grad.clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    assert any(name.startswith("actor.heads.") for name in eager_grads)
    model.zero_grad(set_to_none=True)
    torch._dynamo.reset()
    model._compiled_actor_core = torch.compile(
        model.actor.policy_core, backend="eager", fullgraph=True, dynamic=True
    )
    captured = model.evaluate_actions(obs, actions)
    torch.testing.assert_close(captured.log_probs.event, eager.log_probs.event)
    torch.testing.assert_close(captured.entropies.event, eager.entropies.event)
    _loss(captured).backward()
    for name, parameter in model.named_parameters():
        if name in eager_grads:
            assert parameter.grad is not None, name
            torch.testing.assert_close(parameter.grad, eager_grads[name], msg=name)
    with torch.no_grad():
        sampled = model(obs)
        replayed = model.evaluate_actions(obs, sampled.actions)
    torch.testing.assert_close(sampled.log_probs.event, replayed.log_probs.event)
    actions.lengths[0, 0] += 1
    with pytest.raises(ka.GrammarReplayError, match="length"), torch.no_grad():
        model.evaluate_actions(obs, actions)
    torch._dynamo.reset()


def test_replay_log_prob_matches_finite_differences_of_a_head_weight() -> None:
    model = _tiny().double()
    obs, actions = _base_case()
    obs = _obs_double(obs)
    weight = model.actor.heads["market_item"].out.weight
    item = 3  # seat 0 SELLs item 3 at frame 2
    model.evaluate_actions(obs, actions).log_probs.per_player_entity.sum().backward()
    assert weight.grad is not None
    analytic = float(weight.grad[item, 0])
    assert abs(analytic) > 1e-8
    epsilon = 1e-6

    def total() -> float:
        with torch.no_grad():
            evaluation = model.evaluate_actions(obs, actions)
        return float(evaluation.log_probs.per_player_entity.sum())

    with torch.no_grad():
        weight[item, 0] += epsilon
    plus = total()
    with torch.no_grad():
        weight[item, 0] -= 2 * epsilon
    minus = total()
    with torch.no_grad():
        weight[item, 0] += epsilon
    assert analytic == pytest.approx((plus - minus) / (2 * epsilon), rel=1e-6)


def test_every_parameter_receives_a_finite_gradient() -> None:
    model = _tiny()
    obs, actions = _base_case()
    _loss(model.evaluate_actions(obs, actions)).backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert missing == []
    assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters())


# --- §7.8 seat isolation and statelessness --------------------------------------------


def test_heads_are_seat_isolated_and_stateless() -> None:
    model = _tiny().eval()
    obs = make_obs(envs=2, own_actors=(2, 3), rival_actors=(3, 2), order_limit=4)
    torch.manual_seed(8)
    with torch.no_grad():
        sampled = model(obs)
        base = model.evaluate_actions(obs, sampled.actions)
        changed = make_obs(
            envs=2, own_actors=(2, 3), rival_actors=(3, 2), order_limit=4
        )
        changed.tiles_float[:, 1] += 1.0
        changed.actors_float[:, 1] += 0.5
        other = model.evaluate_actions(changed, sampled.actions)
        torch.testing.assert_close(
            other.log_probs.event[:, 0], base.log_probs.event[:, 0], rtol=0, atol=0
        )
        assert (
            float(
                (other.log_probs.event[:, 1] - base.log_probs.event[:, 1]).abs().max()
            )
            > 1e-6
        )
        # No state carries across calls: unrelated batches in between change nothing.
        first = model(obs, deterministic=True)
        model(make_obs(envs=3, own_actors=7, order_limit=10))
        model.evaluate_actions(changed, sampled.actions)
        second = model(obs, deterministic=True)
        again = model.evaluate_actions(obs, sampled.actions)
    assert torch.equal(first.actions.tokens, second.actions.tokens)
    torch.testing.assert_close(
        first.log_probs.event, second.log_probs.event, rtol=0, atol=0
    )
    torch.testing.assert_close(
        again.log_probs.event, base.log_probs.event, rtol=0, atol=0
    )


def test_forward_serve_and_values_share_one_encode_contract() -> None:
    model = _tiny().eval()
    obs = make_obs(envs=2)
    with torch.no_grad():
        out = model(obs, deterministic=True)
        served = model.serve(obs, deterministic=True)
        evaluation = model.evaluate_actions(obs, out.actions)
        values = model.compute_value(obs)
    assert out.actions.tokens.shape == (2, 2, F, K)
    assert out.actions.lengths.shape == (2, 2)
    assert out.log_probs.event.shape == (2, 2, F, K)
    assert out.entropies.event.shape == (2, 2, F, K)
    assert set(out.entropies.components) == set(kt.SLOT_NAMES)
    assert torch.equal(served.actions.tokens, out.actions.tokens)
    torch.testing.assert_close(out.values, values)
    torch.testing.assert_close(evaluation.values, values)
    assert out.winner_probabilities.shape == (2, 2, 2)
    assert evaluation.winner_log_probabilities is not None
    torch.testing.assert_close(
        evaluation.winner_log_probabilities.exp(), evaluation.winner_probabilities
    )


# --- §7.9 topology and initialization -------------------------------------------------


def _optimizer_groups(model: Any) -> tuple[set[int], set[int]]:
    optimizer = create_optimizer(model, MuonConfig())
    groups: dict[type, set[int]] = {torch.optim.Muon: set(), torch.optim.AdamW: set()}
    for inner in optimizer.optimizers:
        groups[type(inner)].update(
            id(p) for group in inner.param_groups for p in group["params"]
        )
    return groups[torch.optim.Muon], groups[torch.optim.AdamW]


def test_actor_topology_follows_the_brief() -> None:
    model = _tiny()
    proj = model.actor_input_proj
    assert type(proj) is nn.Linear
    assert (proj.in_features, proj.out_features) == (48, 16)
    actor = model.actor
    assert type(actor) is ka.KaggricultureGrammarActor
    assert type(actor.source_norm) is nn.LayerNorm
    assert actor.market_position.num_embeddings == 11
    assert set(actor.heads) == HEAD_NAMES
    for slot in ka.POLICY_SLOTS:
        head = actor.heads[kt.SLOT_NAMES[slot]]
        assert type(head) is OutputProjectionMLP
        assert head.out.out_features == kt.SLOT_WIDTHS[slot]
        assert head.up.in_features == head.up.out_features == 16
    widths = {name: emb.num_embeddings for name, emb in actor.slot_embeddings.items()}
    assert widths == {
        "unit_kind": 20,
        "unit_item": 16,
        "unit_quantity_present": 2,
        "unit_quantity_high": 32,
        "market_kind": 8,
        "market_item": 16,
        "market_quantity_high": 32,
    }
    assert not any(
        isinstance(m, (nn.GRU, nn.GRUCell, nn.LSTM)) for m in model.modules()
    )
    # Tables are grammar data, never checkpoint state.
    assert not any(key.startswith("actor.table_") for key in model.state_dict())


def test_head_and_embedding_optimizer_membership() -> None:
    model = _tiny()
    muon, adamw = _optimizer_groups(model)
    assert not muon & adamw
    assert muon | adamw == {id(p) for p in model.parameters()}
    assert id(model.actor_input_proj.weight) in muon
    assert id(model.actor_input_proj.bias) in adamw
    for head in model.actor.heads.values():
        assert id(head.up.weight) in muon
        assert id(head.out.weight) in adamw
        assert id(head.out.bias) in adamw
    input_ids = {id(layer) for layer in model.get_input_layers()}
    output_ids = {id(layer) for layer in model.get_output_layers()}
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            assert name.startswith("actor."), name
            assert id(module.weight) in adamw, name
            assert id(module.weight) in input_ids, name
    for head in model.actor.heads.values():
        assert id(head.out) in output_ids
    for p in model.actor.source_norm.parameters():
        assert id(p) in adamw


def _singular_values(linear: nn.Linear) -> torch.Tensor:
    return torch.linalg.svdvals(linear.weight.detach())


def test_head_initialization_follows_isaiah() -> None:
    model = _tiny(embed_dim=64, n_heads=4)
    with torch.no_grad():
        for head in model.actor.heads.values():
            head.out.weight.fill_(1.0)
            head.out.bias.fill_(1.0)
        model.actor.market_position.weight.fill_(3.0)
    model.reset_parameters()
    for name, head in model.actor.heads.items():
        singular = _singular_values(head.out)
        torch.testing.assert_close(
            singular, torch.full_like(singular, 0.01), rtol=1e-5, atol=1e-6, msg=name
        )
        assert int(head.out.bias.count_nonzero()) == 0, name
        up = _singular_values(head.up)
        torch.testing.assert_close(up, torch.full_like(up, math.sqrt(2.0)), msg=name)
    proj = _singular_values(model.actor_input_proj)
    torch.testing.assert_close(proj, torch.full_like(proj, math.sqrt(2.0)))
    target = 64**-0.5
    for layer in model.actor.get_input_layers():
        std = float(layer.detach().std())
        assert abs(std - target) < 0.3 * target
    pooled = torch.cat(
        [layer.detach().flatten() for layer in model.actor.get_input_layers()]
    )
    assert abs(float(pooled.std()) - target) < 0.1 * target
    # The critic keeps Isaiah's critic gain.
    critic = _singular_values(model.critic_head.out)
    torch.testing.assert_close(critic, torch.ones_like(critic))


# --- §7.10 budget ---------------------------------------------------------------------


def test_default_preset_meets_the_owner_parameter_budget() -> None:
    config = km.KaggricultureTransformerConfig.from_file(
        ROOT / "configs/model/kaggriculture.yaml"
    )
    assert config == km.KaggricultureTransformerConfig(force_flash_attn=True)
    model = km.KaggricultureTransformer(
        config,
        obs_spec=kt.KaggricultureObsConfig(),
        action_spec=kt.KaggricultureActionConfig(),
    )
    count = sum(p.numel() for p in model.parameters())
    assert 6_000_000 <= count <= 10_000_000
    assert count == 6_252_223
    heads = sum(p.numel() for p in model.actor.parameters()) + sum(
        p.numel() for p in model.actor_input_proj.parameters()
    )
    assert heads == 873_406


# --- §9.3 head-extent guard -----------------------------------------------------------


def test_head_rows_per_chunk_boundary_at_the_preset() -> None:
    config = km.KaggricultureTransformerConfig()
    width = km.head_gemm_width(config)
    assert width == 3 * 256
    rows = km.head_rows_per_chunk(config)
    assert rows == 11_096
    assert rows * F * width < 2**31 <= (rows + 1) * F * width
    small = km.KaggricultureTransformerConfig(embed_dim=4, n_heads=1)
    assert km.head_gemm_width(small) == 32  # the widest head dominates


@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_head_rows_per_chunk_follows_the_limit(
    monkeypatch: pytest.MonkeyPatch, offset: int
) -> None:
    config = _tiny().config
    per_row = F * km.head_gemm_width(config)
    monkeypatch.setattr(km, "_GEMM_ELEMENT_LIMIT", 5 * per_row + offset)
    assert km.head_rows_per_chunk(config) == (5 if offset > 0 else 4)
    monkeypatch.setattr(km, "_GEMM_ELEMENT_LIMIT", per_row)
    with pytest.raises(ValueError, match="compiled-gemm-template-overflows"):
        km.head_rows_per_chunk(config)


@pytest.mark.parametrize(("chunk", "calls"), [(7, [6]), (6, [6]), (5, [5, 1])])
def test_head_chunking_dispatches_ordered_slices_and_matches_unchunked(
    monkeypatch: pytest.MonkeyPatch, chunk: int, calls: list[int]
) -> None:
    """Chunk limit ±1 around a 6-row batch; chunked equals unchunked."""
    model = _tiny().eval()
    obs = make_obs(envs=3, own_actors=(1, 4, 2), rival_actors=(3, 1, 5), order_limit=3)
    torch.manual_seed(2)
    with torch.no_grad():
        sampled = model(obs)
        whole = model.evaluate_actions(obs, sampled.actions)
        encoded = model.encode_observations(obs)
        whole_result = model._policy(
            encoded, model._grammar_context(obs), sampled.actions, deterministic=False
        )
        whole_greedy = model(obs, deterministic=True)
    seen: list[tuple[torch.Tensor, torch.Tensor]] = []

    def spy(*args: Any) -> ka.GrammarPolicyResult:
        seen.append((args[0].clone(), args[2].clone()))
        return model.actor.policy_core(*args)

    monkeypatch.setattr(km, "head_rows_per_chunk", lambda _config: chunk)
    model._compiled_actor_core = spy
    with torch.no_grad():
        chunked = model.evaluate_actions(obs, sampled.actions)
        chunked_result = model._policy(
            encoded, model._grammar_context(obs), sampled.actions, deterministic=False
        )
        chunked_greedy = model(obs, deterministic=True)
    assert [x.shape[0] for x, _ in seen[: len(calls)]] == calls
    counts = model._grammar_context(obs).actor_counts
    start = 0
    for (_, chunk_counts), size in zip(seen[: len(calls)], calls, strict=True):
        assert torch.equal(chunk_counts, counts[start : start + size])
        start += size
    torch.testing.assert_close(chunked.log_probs.event, whole.log_probs.event)
    torch.testing.assert_close(chunked.entropies.event, whole.entropies.event)
    assert torch.equal(chunked_result.valid, whole_result.valid)
    assert torch.equal(chunked_result.tokens, whole_result.tokens)
    assert torch.equal(chunked_greedy.actions.tokens, whole_greedy.actions.tokens)
    # A malformed row is still rejected, with the flags merged across chunks.
    bad = kt.KaggricultureActions(
        tokens=sampled.actions.tokens.clone(), lengths=sampled.actions.lengths.clone()
    )
    bad.lengths[2, 1] += 1
    with pytest.raises(ka.GrammarReplayError) as error, torch.no_grad():
        model.evaluate_actions(obs, bad)
    assert error.value.groups == ("length",)
    assert error.value.first_rows == (5,)


def test_trunk_compile_leaves_the_heads_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _tiny()
    monkeypatch.setattr(torch, "compile", lambda fn, **_: fn)
    assert model.compile_transformer_trunk(mode="max-autotune-no-cudagraphs") == 1
    assert model._compiled_actor_core is None
