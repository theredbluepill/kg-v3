"""Per-seat critic offset head (``model.critic_offset``).

V_seat = 2 p(self) - 1 + o_seat, where o_seat comes from a head on the seat
row's own critic-value token with a zero-initialised output layer. Covers the
default-off identity, the step-0 identity, fitting a common-mode return, the
detached-trunk option, the checkpoint loader rule, row independence and the
trainer telemetry. The ops receipt ``ops/critic-offset-2026-09-30/`` pins the
default-off native, trainer, model and config digests against 3e89425.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import torch
from owl.kaggriculture import types as kt
from owl.model import kaggriculture as km
from owl.model import load_model_state_dict_allowing_lora
from owl.train import FullConfig
from owl.train.logging import config_sha256
from owl.train.ppo import PPOTrainer
from pydantic import ValidationError

from tests.kaggriculture.conftest import make_obs
from tests.kaggriculture.helpers import _tiny
from tests.kaggriculture.test_training_smoke import _native_env, _trainer

_HEAD = "critic_offset_head."


def _obs() -> kt.KaggricultureObsBatch:
    return make_obs(envs=3, own_actors=[2, 3, 4], rival_actors=[1, 2, 3])


def _pair(**overrides: Any) -> tuple[Any, Any]:
    """An off model and an on model holding the off model's weights."""
    off = _tiny(seed=5)
    on = _tiny(seed=6, critic_offset=True, **overrides)
    load_model_state_dict_allowing_lora(on, off.state_dict())
    return off.eval(), on.eval()


def _randomize_head(model: Any, seed: int = 3) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in model.critic_offset_head.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator))


# --- config -----------------------------------------------------------------


def test_default_config_dumps_exactly_as_before_the_fields() -> None:
    plain = km.KaggricultureTransformerConfig()
    dumped = plain.model_dump(mode="json")
    assert "critic_offset" not in dumped
    assert "critic_offset_detach_trunk" not in dumped
    assert km.KaggricultureTransformerConfig.model_validate(dumped) == plain
    on = km.KaggricultureTransformerConfig(
        critic_offset=True, critic_offset_detach_trunk=True
    )
    on_dump = on.model_dump(mode="json")
    assert (on_dump["critic_offset"], on_dump["critic_offset_detach_trunk"]) == (
        True,
        True,
    )
    assert km.KaggricultureTransformerConfig.model_validate(on_dump) == on
    with pytest.raises(ValidationError, match=r"requires model\.critic_offset"):
        km.KaggricultureTransformerConfig(critic_offset_detach_trunk=True)


def test_preset_config_hash_is_unchanged_by_the_default_fields() -> None:
    # config_sha256 of configs/kaggriculture.yaml on the pre-change tree 3e89425
    # (ops/critic-offset-2026-09-30/baseline-digest-3e89425.json).
    cfg = FullConfig.from_file(
        Path(__file__).parents[2] / "configs" / "kaggriculture.yaml"
    )
    assert (
        config_sha256(cfg)
        == "79ab336b7e60bd6284185a34cb025f92efaa611aae129e532f75d1b4eb504f3a"
    )


# --- model ------------------------------------------------------------------


def test_off_model_has_no_head_and_reports_no_offsets() -> None:
    model = _tiny(seed=5).eval()
    assert model.critic_offset_head is None
    assert not any(key.startswith(_HEAD) for key in model.state_dict())
    assert model.optional_state_keys() == frozenset()
    with torch.no_grad():
        assert model(_obs()).value_offsets is None


def test_fresh_head_output_layer_is_zero_and_its_hidden_layer_is_not() -> None:
    model = _tiny(seed=5, critic_offset=True)
    head = model.critic_offset_head
    assert head is not None
    assert head.out.weight.eq(0).all()
    assert head.out.bias.eq(0).all()
    assert head.up.weight.abs().sum() > 0
    # The output layer is a head output (AdamW, not Muon; no int8).
    assert any(layer is head.out for layer in model.get_output_layers())
    assert model.optional_state_keys() == frozenset(
        f"{_HEAD}{key}" for key in head.state_dict()
    )


def test_zero_init_head_keeps_every_output_bit_for_bit() -> None:
    off, on = _pair()
    obs = _obs()
    with torch.no_grad():
        torch.manual_seed(11)
        a = off(obs)
        torch.manual_seed(11)
        b = on(obs)
        assert b.value_offsets is not None
        assert b.value_offsets.eq(0).all()
        assert torch.equal(a.values, b.values)
        assert torch.equal(a.winner_probabilities, b.winner_probabilities)
        assert torch.equal(a.actions.tokens, b.actions.tokens)
        assert torch.equal(a.log_probs.event, b.log_probs.event)
        ea = off.evaluate_actions(obs, a.actions)
        eb = on.evaluate_actions(obs, a.actions)
        assert torch.equal(ea.values, eb.values)
        assert torch.equal(ea.log_probs.event, eb.log_probs.event)
        assert torch.equal(off.compute_value(obs), on.compute_value(obs))
    # The values vary across rows, so the equality is not a constant's.
    assert torch.unique(a.values).numel() > 1


def test_value_is_the_winner_value_plus_the_offset_and_actions_ignore_it() -> None:
    off, on = _pair()
    _randomize_head(on)
    obs = _obs()
    with torch.no_grad():
        torch.manual_seed(11)
        a = off(obs)
        torch.manual_seed(11)
        b = on(obs)
    offsets = b.value_offsets
    assert offsets is not None
    assert offsets.abs().min() > 0
    torch.testing.assert_close(b.values, a.values + offsets, rtol=0, atol=0)
    # The bootstrap (`compute_value`, the horizon `last_values` and truncation)
    # and the update's replay (`evaluate_actions`) see the same sum, at a
    # nonzero head.
    with torch.no_grad():
        bootstrap = on.compute_value(obs)
        replay = on.evaluate_actions(obs, b.actions).values
    torch.testing.assert_close(bootstrap, a.values + offsets, rtol=0, atol=0)
    torch.testing.assert_close(replay, a.values + offsets, rtol=0, atol=0)
    # The policy never reads the head: identical programs and densities.
    assert torch.equal(a.actions.tokens, b.actions.tokens)
    assert torch.equal(a.log_probs.event, b.log_probs.event)
    assert torch.equal(a.winner_probabilities, b.winner_probabilities)


def test_non_live_rows_keep_a_zero_offset() -> None:
    _off, on = _pair()
    _randomize_head(on)
    obs = _obs()
    obs.still_playing[1, 0] = False
    with torch.no_grad():
        offsets = on(obs).value_offsets
    assert offsets is not None
    assert offsets[1, 0] == 0
    assert offsets[obs.still_playing].abs().min() > 0


def test_the_head_reads_only_its_rows_own_critic_token() -> None:
    # No opponent token and no other row: the head's input is exactly token 0
    # of each row's critic-value pair, and a row's offset does not change when
    # the other rows of the batch change.
    _off, on = _pair()
    _randomize_head(on)
    head = on.critic_offset_head
    obs = _obs()
    captured: list[torch.Tensor] = []
    handle = head.up.register_forward_hook(
        lambda _module, inputs, _output: captured.append(inputs[0].detach())
    )
    with torch.no_grad():
        encoded = on.encode_observations(obs)
        full = on._critic_offsets(encoded)
    handle.remove()
    assert torch.equal(captured[-1], encoded.critic_value_hidden[:, 0])
    single = make_obs(envs=1, own_actors=[2], rival_actors=[1])
    with torch.no_grad():
        alone = on._critic_offsets(on.encode_observations(single))
    torch.testing.assert_close(alone, full[:2])


def test_head_fits_a_common_mode_return_the_winner_critic_cannot() -> None:
    # Returns with a common mode of 1.2 and a zero-sum part of +-0.3 (seat 0
    # 1.5, seat 1 0.9). The winner critic's 2p - 1 lies in (-1, 1), so its MSE
    # stays at least (1.5 - 1)^2 / 2 = 0.125; the offset head fits it.
    obs = _obs()
    target = torch.tensor([1.5, 0.9]).expand(3, 2)

    def fit(model: Any) -> float:
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        loss = torch.tensor(0.0)
        for _ in range(250):
            optimizer.zero_grad()
            values, _ = model._values(model.encode_observations(obs), obs)
            loss = (values - target).pow(2).mean()
            loss.backward()
            optimizer.step()
        return float(loss.detach())

    off = _tiny(seed=5).train()
    on = _tiny(seed=5, critic_offset=True).train()
    assert fit(off) >= 0.125
    assert fit(on) < 1e-3


@pytest.mark.parametrize("detach", [False, True])
def test_detach_trunk_blocks_the_heads_trunk_gradient(detach: bool) -> None:
    model = _tiny(seed=5, critic_offset=True, critic_offset_detach_trunk=detach).train()
    _randomize_head(model)
    obs = _obs()
    offsets = model._critic_offsets(model.encode_observations(obs))
    assert offsets is not None
    offsets.sum().backward()
    head = model.critic_offset_head
    for parameter in head.parameters():
        assert parameter.grad is not None
        assert parameter.grad.abs().sum() > 0
    trunk = [
        p
        for name, p in model.named_parameters()
        if not name.startswith(("critic_offset_head.", "critic_head.", "actor"))
    ]
    reached = [p.grad is not None and p.grad.abs().sum() > 0 for p in trunk]
    if detach:
        assert not any(reached)
    else:
        # The control: without the detach the offset trains the trunk.
        assert any(reached)


# --- loader rule ------------------------------------------------------------


def test_a_checkpoint_without_the_head_loads_and_zeroes_its_output() -> None:
    off = _tiny(seed=5)
    on = _tiny(seed=6, critic_offset=True)
    _randomize_head(on)
    up_before = on.critic_offset_head.up.weight.clone()
    load_model_state_dict_allowing_lora(on, off.state_dict())
    head = on.critic_offset_head
    assert head.out.weight.eq(0).all()
    assert head.out.bias.eq(0).all()
    # Only the output layer is reset; the hidden layer keeps its values.
    assert torch.equal(head.up.weight, up_before)
    for key, value in off.state_dict().items():
        assert torch.equal(on.state_dict()[key], value), key


def test_only_the_head_keys_may_be_missing() -> None:
    off = _tiny(seed=5)
    on = _tiny(seed=6, critic_offset=True)
    state = off.state_dict()
    del state["final_norm.weight"]
    with pytest.raises(RuntimeError, match=r"missing non-LoRA .*final_norm\.weight"):
        load_model_state_dict_allowing_lora(on, state)
    partial = on.state_dict()
    del partial[f"{_HEAD}out.bias"]
    with pytest.raises(RuntimeError, match="part of the optional model state"):
        load_model_state_dict_allowing_lora(_tiny(seed=6, critic_offset=True), partial)
    extra = off.state_dict() | {"bogus.weight": torch.zeros(1)}
    with pytest.raises(RuntimeError, match=r"unexpected .*bogus\.weight"):
        load_model_state_dict_allowing_lora(on, extra)


def test_a_head_checkpoint_fails_loudly_on_a_model_without_the_head() -> None:
    on = _tiny(seed=6, critic_offset=True)
    with pytest.raises(RuntimeError, match=r"unexpected .*critic_offset_head"):
        load_model_state_dict_allowing_lora(_tiny(seed=5), on.state_dict())


def test_a_head_checkpoint_round_trips_through_its_own_config() -> None:
    # The rule the Kaggle packaging follows: build the model from the
    # checkpoint's own config (which records critic_offset); the head is then
    # loaded strictly and never changes an action.
    on = _tiny(seed=6, critic_offset=True)
    _randomize_head(on)
    config = km.KaggricultureTransformerConfig.model_validate(
        on.config.model_dump(mode="json")
    )
    rebuilt = km.KaggricultureTransformer(
        config,
        obs_spec=kt.KaggricultureObsConfig(),
        action_spec=kt.KaggricultureActionConfig(),
    )
    rebuilt.load_state_dict(on.state_dict(), strict=True)
    for key, value in on.state_dict().items():
        assert torch.equal(rebuilt.state_dict()[key], value), key


def test_trainer_weight_load_accepts_a_headless_checkpoint_model_only(
    tmp_path: Path,
) -> None:
    # --load-model-weights (model_only) from a checkpoint without the head,
    # e.g. the BC best; model_and_optimizer fails, the optimizer groups differ.
    torch.manual_seed(5)
    off_trainer = _trainer(_native_env(), _tiny(seed=5))
    path = tmp_path / "checkpoint.pt"
    off_trainer.write_checkpoint(path, env_steps=0)
    on_model = _tiny(seed=6, critic_offset=True)
    _randomize_head(on_model)
    on_trainer = _trainer(_native_env(), on_model)
    on_trainer.load_model_weights(path)
    head = on_model.critic_offset_head
    assert head.out.weight.eq(0).all()
    assert head.out.bias.eq(0).all()
    for key, value in off_trainer.model.state_dict().items():
        assert torch.equal(on_model.state_dict()[key], value), key
    with pytest.raises(ValueError, match="param count must match"):
        _trainer(_native_env(), _tiny(seed=6, critic_offset=True)).load_model_weights(
            path, load_optimizer=True
        )


# --- trainer ----------------------------------------------------------------


def _bank_env() -> Any:
    return _native_env(bank_weight=0.25)


def test_off_trainer_stores_and_logs_no_offsets() -> None:
    torch.manual_seed(307)
    trainer = _trainer(_bank_env(), _tiny(seed=5))
    metrics = trainer.train_iteration()
    assert trainer.rollout.value_offsets is None
    assert trainer.rollout.segment_major().value_offsets is None
    assert not any("offset" in key or "ev_common" in key for key in metrics)


def test_two_updates_with_the_head_and_the_own_bank_term() -> None:
    torch.manual_seed(307)
    model = _tiny(seed=5, critic_offset=True)
    trainer: PPOTrainer = _trainer(_bank_env(), model, vf_coef=2.0)
    results = [trainer.train_iteration() for _ in range(2)]
    for metrics in results:
        assert all(math.isfinite(value) for value in metrics.values())
        for key in (
            "train/value_offset_mean",
            "train/value_offset_abs_mean",
            "train/ev_common",
        ):
            assert key in metrics
        assert metrics["train/value_offset_abs_mean"] >= abs(
            metrics["train/value_offset_mean"]
        )
    # Update 1 collects with the zero head; the value loss then moves it.
    assert results[0]["train/value_offset_abs_mean"] == 0.0
    assert results[1]["train/value_offset_abs_mean"] > 0.0
    head = model.critic_offset_head
    assert head.out.weight.abs().sum() > 0
    # A fresh rollout stores the winner value plus the stored offset.
    trainer._collect_rollout()
    rollout = trainer.rollout
    assert rollout.value_offsets is not None
    obs = rollout.obs
    assert isinstance(obs, kt.KaggricultureObsBatch)
    with torch.no_grad():
        winner = model.winner_log_probabilities(obs).exp()[..., 0] * 2.0 - 1.0
    torch.testing.assert_close(
        rollout.values, winner + rollout.value_offsets, rtol=0, atol=1e-6
    )


def test_learner_rows_carry_their_offsets_and_scripted_rows_zero() -> None:
    # Fixed-opponent collection (env.opponent_mix) runs only the learner rows;
    # their offsets are scattered like their values.
    from owl.train import ppo

    from tests.kaggriculture.test_opponent_mix import _env as _mix_env

    torch.manual_seed(13)
    env = _mix_env(n_envs=4, bot="starter", bot_envs=2)
    _off, model = _pair()
    _randomize_head(model)
    obs = env.observations
    learner = env.learner_mask.clone()
    with torch.no_grad():
        full = model(obs, deterministic=True)
        rows = ppo.forward_learner_rows(model, obs, learner, deterministic=True)
    assert full.value_offsets is not None
    assert rows.value_offsets is not None
    torch.testing.assert_close(rows.value_offsets[learner], full.value_offsets[learner])
    assert rows.value_offsets[learner].abs().min() > 0
    assert rows.value_offsets[~learner].eq(0).all()


def test_fixed_opponent_run_omits_ev_common_without_both_seats() -> None:
    from tests.kaggriculture.test_opponent_mix import _env as _mix_env
    from tests.kaggriculture.test_opponent_mix import _trainer as _mix_trainer

    torch.manual_seed(17)
    env = _mix_env(n_envs=2, bot="starter", bot_envs=2, episode_steps=3)
    model = _tiny(seed=5, critic_offset=True)
    trainer = _mix_trainer(env, model)
    metrics = trainer.train_iteration()
    assert trainer.rollout.value_offsets is not None
    assert all(math.isfinite(value) for value in metrics.values())
    assert "train/value_offset_mean" in metrics
    # At fraction 1.0 no step has both seats learned: no common mode to score.
    assert "train/ev_common" not in metrics


def test_offset_telemetry_scores_the_common_mode_by_the_offsets() -> None:
    torch.manual_seed(307)
    trainer = _trainer(_bank_env(), _tiny(seed=5, critic_offset=True))
    generator = torch.Generator().manual_seed(1)
    returns = torch.randn((2, 3, 2), generator=generator)
    common = returns.mean(dim=-1, keepdim=True)
    mask = torch.ones((2, 3, 2), dtype=torch.bool)
    mask[0, 0, 1] = False
    # Offsets equal to the common mode explain all of it.
    offsets = common.expand(2, 3, 2).clone()
    metrics = trainer._value_offset_metrics(offsets, returns, mask)
    assert metrics["train/ev_common"] == pytest.approx(1.0)
    assert metrics["train/value_offset_mean"] == pytest.approx(
        float(offsets[mask].mean())
    )
    assert metrics["train/value_offset_abs_mean"] == pytest.approx(
        float(offsets[mask].abs().mean())
    )
    # Zero offsets explain none of it.
    zero = trainer._value_offset_metrics(torch.zeros_like(offsets), returns, mask)
    assert zero["train/ev_common"] == pytest.approx(0.0, abs=1e-6)
    # Only both-seat steps count: a wrong offset on the half-masked step (0, 0)
    # does not change the score.
    offsets[0, 0] = 100.0
    masked = trainer._value_offset_metrics(offsets, returns, mask)
    assert masked["train/ev_common"] == pytest.approx(1.0)
