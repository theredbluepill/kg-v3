"""Fixed-opponent PPO collection (``env.opponent_mix``) on the native env.

The native tests (``src/kaggriculture/opponent_env_tests.rs``) replay the same
games on an independent kernel + controller reference, proving the scripted
seat's actions come from the bot and the learner's are executed unchanged.
These tests cover the config, the adapter, the trainer's learner mask and the
unchanged self-play default.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from typing import Any

import numpy as np
import pytest
import torch
from owl import rs
from owl.game import create_env
from owl.kaggriculture.config import (
    KaggricultureEnvConfig,
    KaggricultureOpponentMixConfig,
)
from owl.kaggriculture.env import KaggricultureVectorizedEnv
from owl.kaggriculture.rewards import KaggricultureRewardConfig
from owl.kaggriculture.telemetry import (
    fixed_opponent_metrics,
    split_fixed_opponent_games,
)
from owl.kaggriculture.types import (
    KaggricultureActionConfig,
    KaggricultureActions,
    KaggricultureGameConfig,
    KaggricultureObsBatch,
    KaggricultureObsConfig,
)
from owl.model.kaggriculture import (
    KaggricultureTransformer,
    KaggricultureTransformerConfig,
)
from owl.train import ppo
from owl.train.ppo import PPOConfig, PPOTrainer

from .test_native_env import buffers, make_env, pass_actions

_MARGIN_REWARD = KaggricultureRewardConfig(
    econ_shaping=0.0,
    econ_starvation_weight=0.0,
    econ_drought_weight=0.0,
    econ_cap=0.0,
    econ_ineffective_weight=0.0,
    econ_ineffective_cap=0.0,
    econ_bank_weight=0.0,
    econ_bank_scale=100_000.0,
    econ_bank_cap=0.0,
    econ_margin_weight=0.5,
    econ_margin_scale=50_000.0,
    econ_margin_cap=0.5,
)


def _env(
    *,
    n_envs: int = 2,
    bot: str | None = "starter",
    bot_envs: int = 2,
    episode_steps: int = 720,
    seed: int = 509,
) -> KaggricultureVectorizedEnv:
    return KaggricultureVectorizedEnv(
        n_envs=n_envs,
        seed=seed,
        seed_stride=1,
        config=KaggricultureGameConfig(episodeSteps=episode_steps),
        reward_config=_MARGIN_REWARD,
        reward_mode="win_loss",
        native_threads=1,
        pin_memory=False,
        transfer_device=torch.device("cpu"),
        obs_spec=KaggricultureObsConfig(),
        action_spec=KaggricultureActionConfig(),
        opponent_bot=bot,
        opponent_envs=bot_envs if bot is not None else 0,
    )


def _tiny_model() -> KaggricultureTransformer:
    return KaggricultureTransformer(
        KaggricultureTransformerConfig(
            embed_dim=16, depth=1, n_heads=1, mlp_ratio=1, n_scratch_tokens=0
        ),
        obs_spec=KaggricultureObsConfig(),
        action_spec=KaggricultureActionConfig(),
    )


def _trainer(
    env: KaggricultureVectorizedEnv,
    model: KaggricultureTransformer,
    teacher: KaggricultureTransformer | None = None,
    **config: object,
) -> PPOTrainer:
    return PPOTrainer(
        config=PPOConfig.model_validate(
            {
                "horizon": 3,
                "ppo_epochs": 1,
                "segments_per_minibatch": 1,
                "gradient_accumulation_steps": 1,
                "dtype": "float32",
                "model_compile": "none",
                "compile_mode": None,
                "teacher_mode": None if teacher is None else "last_best",
                **config,
            }
        ),
        env=env,
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        device=torch.device("cpu"),
        teacher_model=teacher,
        teacher_active=teacher is not None,
    )


def _env_config(**mix: Any) -> KaggricultureEnvConfig:
    return KaggricultureEnvConfig.model_validate(
        {
            "n_envs": 4,
            "reward_shaping": _MARGIN_REWARD.model_dump(),
            "native_threads": 1,
            **({"opponent_mix": mix} if mix else {}),
        }
    )


# --- config -------------------------------------------------------------------


def test_registry_keys_come_from_the_native_opponent_registry() -> None:
    assert rs.kaggriculture_opponent_bots() == (
        "starter",
        "r04",
        "ecobot",
        "e776",
        "cha22",
    )
    with pytest.raises(ValueError, match="unknown opponent bot 'cha99'"):
        KaggricultureOpponentMixConfig(bot="cha99", fraction=1.0)


@pytest.mark.parametrize(
    ("fraction", "n_envs", "expected"),
    [(1.0, 4, 4), (0.5, 4, 2), (0.25, 4, 1), (0.5, 3, None), (0.1, 4, None)],
)
def test_the_fraction_must_name_whole_envs(
    fraction: float, n_envs: int, expected: int | None
) -> None:
    mix = KaggricultureOpponentMixConfig(bot="r04", fraction=fraction)
    if expected is None:
        with pytest.raises(ValueError, match="whole number"):
            mix.bot_envs(n_envs)
    else:
        assert mix.bot_envs(n_envs) == expected
    for bad in (0.0, 1.5, math.nan):
        with pytest.raises(ValueError, match="fraction"):
            KaggricultureOpponentMixConfig(bot="r04", fraction=bad)


def test_self_play_config_dumps_exactly_as_before_the_mix() -> None:
    plain = _env_config()
    assert plain.opponent_mix is None
    assert "opponent_mix" not in plain.model_dump(mode="json")
    mixed = _env_config(bot="starter", fraction=0.5)
    dumped = mixed.model_dump(mode="json")
    assert dumped["opponent_mix"] == {"bot": "starter", "fraction": 0.5}
    assert KaggricultureEnvConfig.model_validate(dumped) == mixed
    with pytest.raises(ValueError, match="whole number"):
        _env_config(bot="starter", fraction=0.3)


def test_create_env_hosts_the_bot_in_the_first_envs() -> None:
    env = create_env(
        _env_config(bot="e776", fraction=0.5),
        n_envs=4,
        base_seed=3,
        rank=0,
        world_size=1,
        pin_memory=False,
        transfer_device=torch.device("cpu"),
    )
    assert isinstance(env, KaggricultureVectorizedEnv)
    assert (env.opponent_bot, env.opponent_envs) == ("e776", 2)
    assert env.learner_mask.tolist() == [
        [True, False],
        [False, True],
        [True, True],
        [True, True],
    ]


# --- none-mix byte identity ---------------------------------------------------

# Recorded on the pre-mix tree 25412a7 (ops/opponent-mix-2026-09-30/
# baseline_digest.py): every output byte and step-metric dict of two native
# games (episodeSteps 4) through nine PASS steps, one truncation and two
# auto-resets. Native arithmetic is platform independent.
_PRE_MIX_NATIVE_DIGEST = (
    "257eae38864aa2aa26373c7751be97b3b0d3c6d9d81ee80782036df41159a590"
)


def test_self_play_native_env_is_byte_identical_to_the_pre_mix_tree() -> None:
    digest = hashlib.sha256()
    env = make_env(2, seed=11, stride=2, config='{"episodeSteps":4}')
    arrays = buffers(2)
    env.observe(**arrays)
    tokens, lengths = pass_actions(2)
    for name in sorted(arrays):
        digest.update(arrays[name].tobytes())
    for step in range(9):
        info = env.step(tokens, lengths, **arrays)
        digest.update(json.dumps(info, sort_keys=True).encode())
        for name in sorted(arrays):
            digest.update(arrays[name].tobytes())
        if step == 4:
            env.truncate_envs(np.array([True, False]), **arrays)
            for name in sorted(arrays):
                digest.update(arrays[name].tobytes())
    digest.update(json.dumps(env.seed_state()).encode())
    assert digest.hexdigest() == _PRE_MIX_NATIVE_DIGEST


def test_self_play_trainer_never_enters_the_mix_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the self-play path called a mix helper")

    monkeypatch.setattr(ppo, "forward_learner_rows", forbidden)
    monkeypatch.setattr(ppo, "_apply_learner_mask", forbidden)
    torch.manual_seed(5)
    env = _env(bot=None, episode_steps=3)
    assert env.opponent_envs == 0
    assert env.learner_mask.all()
    trainer = _trainer(env, _tiny_model())
    metrics = trainer.train_iteration()
    assert trainer.rollout.learner is None
    assert trainer.rollout.segment_major().learner is None
    assert not any("vs_bot" in key for key in metrics)
    assert metrics["train/bank_games"] == 2.0


# --- adapter ------------------------------------------------------------------


def test_adapter_rejects_inconsistent_opponent_arguments() -> None:
    for bot, envs in (("starter", 0), (None, 1), ("starter", 3)):
        with pytest.raises(ValueError, match="opponent_bot and opponent_envs"):
            _env(bot=bot, bot_envs=envs) if bot else KaggricultureVectorizedEnv(
                n_envs=2,
                seed=1,
                seed_stride=1,
                config=KaggricultureGameConfig(),
                reward_config=_MARGIN_REWARD,
                reward_mode="win_loss",
                native_threads=1,
                pin_memory=False,
                transfer_device=torch.device("cpu"),
                obs_spec=KaggricultureObsConfig(),
                action_spec=KaggricultureActionConfig(),
                opponent_envs=envs,
            )
    with pytest.raises(ValueError, match="unknown opponent"):
        _env(bot="nobody")


def _learner_pass_actions(env: KaggricultureVectorizedEnv) -> KaggricultureActions:
    tokens, lengths = pass_actions(env.n_envs)
    mask = env.learner_mask.numpy()
    tokens[~mask] = 0
    lengths[~mask] = 0
    return KaggricultureActions(
        tokens=torch.from_numpy(tokens), lengths=torch.from_numpy(lengths)
    )


def test_seats_alternate_and_auto_reset_restarts_the_bot() -> None:
    # episodeSteps 3: every second step completes all games and auto-resets.
    env = _env(n_envs=3, bot="r04", bot_envs=2, episode_steps=3)
    # Construction is episode 0: env e learns seat e % 2.
    assert env.learner_mask.tolist() == [[True, False], [False, True], [True, True]]
    env.reset()  # episode 1 everywhere
    assert env.learner_mask.tolist() == [[False, True], [True, False], [True, True]]
    for game in range(3):
        _obs, _rewards, dones, metrics = env.step(_learner_pass_actions(env))
        assert not dones.any()
        assert metrics["_terminal_learner_seat"] == []
        learned = env.learner_mask.clone()
        _obs, _rewards, dones, metrics = env.step(_learner_pass_actions(env))
        assert dones.all()
        # The completed games report the seat the learner played in them.
        assert metrics["_terminal_learner_seat"] == [
            float(learned[0].to(torch.int64).argmax()),
            float(learned[1].to(torch.int64).argmax()),
            -1.0,
        ]
        assert "terminal_learner_seat" not in metrics
        # The new episode flips each bot env's learned seat.
        assert (env.learner_mask[:2] == ~learned[:2]).all(), game
        assert env.learner_mask[2].all()
    # A truncation starts a new episode in the selected env only.
    before = env.learner_mask.clone()
    env.truncate_envs(torch.tensor([False, True, False]))
    assert (env.learner_mask[1] == ~before[1]).all()
    assert (env.learner_mask[0] == before[0]).all()


def test_scripted_rows_must_be_submitted_as_the_absent_program() -> None:
    env = _env(n_envs=2, bot="starter", bot_envs=1)
    actions = _learner_pass_actions(env)
    bad = KaggricultureActions(
        tokens=actions.tokens.clone(), lengths=actions.lengths.clone()
    )
    bad.lengths[0, 1] = 2  # env 0's scripted seat is 1
    with pytest.raises(ValueError, match="played by the fixed opponent"):
        env.step(bad)
    env.step(actions)


def test_observations_carry_no_opponent_identity() -> None:
    # Same seeds, three collection setups: every observation byte at step zero
    # (construction and reset) is identical, whichever bot is hosted.
    snapshots = []
    for bot in (None, "starter", "ecobot"):
        env = _env(n_envs=2, bot=bot)
        first = {k: v.clone() for k, v in _obs_tensors(env.observations).items()}
        env.reset()
        second = {k: v.clone() for k, v in _obs_tensors(env.observations).items()}
        snapshots.append((first, second))
    for other in snapshots[1:]:
        for expected, actual in zip(snapshots[0], other, strict=True):
            assert expected.keys() == actual.keys()
            for name, tensor in expected.items():
                assert torch.equal(tensor, actual[name]), name
    # The model-facing batch has no new field: the mix adds none.
    assert "learner" not in KaggricultureObsBatch.model_fields
    assert not any(
        "opponent" in name or "bot" in name
        for name in KaggricultureObsBatch.model_fields
    )


def _obs_tensors(obs: KaggricultureObsBatch) -> dict[str, torch.Tensor]:
    tensors = {
        name: getattr(obs, name)
        for name in KaggricultureObsBatch.model_fields
        if name != "action_mask"
    }
    tensors["can_act"] = obs.action_mask.can_act
    return tensors


def test_scripted_seat_is_not_idle() -> None:
    # Non-vacuity at the binding: after ten steps the bot's farm differs from a
    # game where that seat submitted PASS.
    mixed = _env(n_envs=1, bot="starter", bot_envs=1)
    idle = _env(n_envs=1, bot=None)
    for _ in range(10):
        mixed.step(_learner_pass_actions(mixed))
        idle.step(_learner_pass_actions(idle))
    bot_seat = int((~mixed.learner_mask[0]).to(torch.int64).argmax())
    assert (
        mixed.state_snapshot(0)["public"]["farms"][bot_seat]  # type: ignore[index]
        != idle.state_snapshot(0)["public"]["farms"][bot_seat]  # type: ignore[index]
    )


# --- trainer ------------------------------------------------------------------


def test_rollout_forward_runs_on_learner_rows_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(11)
    env = _env(n_envs=2, bot="starter", bot_envs=1)
    model = _tiny_model()
    batch_rows: list[int] = []
    real_forward = model.forward

    def counting_forward(obs: KaggricultureObsBatch, **kwargs: Any) -> Any:
        batch_rows.append(obs.still_playing.numel())
        return real_forward(obs, **kwargs)

    monkeypatch.setattr(model, "forward", counting_forward)
    trainer = _trainer(env, model)
    trainer._collect_rollout()
    # Env 0 hosts the bot (one learned row), env 1 is self-play (two rows).
    assert batch_rows == [3] * trainer.config.horizon
    learner = trainer.rollout.learner
    assert learner is not None
    assert learner[:, 1].all()
    assert learner[:, 0].sum(dim=-1).eq(1).all()
    # Scripted rows store the absent program, zero log-probs and zero values.
    scripted = ~learner
    assert trainer.rollout.actions.lengths[scripted].eq(0).all()
    assert trainer.rollout.actions.tokens[scripted].eq(0).all()
    assert trainer.rollout.logp[scripted].eq(0).all()
    assert trainer.rollout.values[scripted].eq(0).all()
    assert trainer.rollout.actions.lengths[learner].gt(0).all()


def test_forward_learner_rows_matches_the_full_batch_rows() -> None:
    # Rows are encoded independently: the selected rows' deterministic actions
    # and values equal the same rows of a full-batch forward. Hosted and
    # self-play envs are mixed, and sampled play first makes every learned
    # row's value distinct (at the reset all rows are identical), so a row
    # scattered to the wrong place cannot match by coincidence.
    torch.manual_seed(13)
    env = _env(n_envs=4, bot="starter", bot_envs=2)
    model = _tiny_model().eval()
    obs = env.observations
    for _ in range(4):
        with torch.no_grad():
            played = ppo.forward_learner_rows(model, obs, env.learner_mask.clone())
        obs, _rewards, dones, _metrics = env.step(
            KaggricultureActions(
                tokens=played.actions.tokens.contiguous(),
                lengths=played.actions.lengths.contiguous(),
            )
        )
        assert not dones.any()
    learner = env.learner_mask.clone()
    assert learner.sum().item() == 6
    with torch.no_grad():
        full = model(obs, deterministic=True)
        rows = ppo.forward_learner_rows(model, obs, learner, deterministic=True)
    learned_values = full.values[learner]
    assert torch.unique(learned_values).numel() == learned_values.numel()
    assert torch.equal(rows.actions.tokens[learner], full.actions.tokens[learner])
    assert torch.equal(rows.actions.lengths[learner], full.actions.lengths[learner])
    torch.testing.assert_close(rows.values[learner], learned_values)
    torch.testing.assert_close(
        rows.logp[learner], full.log_probs.per_player_entity.sum(dim=-1)[learner]
    )
    # Scripted rows are never forwarded: zero actions, log-probs and values.
    assert rows.actions.lengths[~learner].eq(0).all()
    assert rows.logp[~learner].eq(0).all()
    assert rows.values[~learner].eq(0).all()


def _perturb_scripted_rows(trainer: PPOTrainer) -> None:
    """Scramble everything stored for scripted seats after collection."""
    learner = trainer.rollout.learner
    assert learner is not None
    scripted = ~learner
    generator = torch.Generator().manual_seed(99)

    def noise(tensor: torch.Tensor) -> torch.Tensor:
        return torch.randn(tensor.shape, generator=generator, dtype=torch.float32)

    rollout = trainer.rollout
    rollout.rewards[scripted] = 50.0 * noise(rollout.rewards[scripted])
    rollout.values[scripted] = noise(rollout.values[scripted])
    rollout.logp[scripted] = noise(rollout.logp[scripted])
    rollout.entity_logp[scripted] = noise(rollout.entity_logp[scripted])
    obs = rollout.obs
    assert isinstance(obs, KaggricultureObsBatch)
    obs.tiles_float[scripted] += noise(obs.tiles_float[scripted])
    obs.actors_float[scripted] += noise(obs.actors_float[scripted])
    obs.banks[scripted] += 1000.0 * noise(obs.banks[scripted]).to(torch.float64)


def _update_after_perturbation(
    *,
    perturb: bool,
    n_envs: int,
    bot_envs: int,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    torch.manual_seed(21)
    env = _env(n_envs=n_envs, bot="starter", bot_envs=bot_envs, episode_steps=3)
    model = _tiny_model()
    teacher = _tiny_model()
    trainer = _trainer(
        env,
        model,
        teacher,
        normalize_advantages=True,
        ent_coef=0.05,
        vf_coef=0.5,
        teacher_kl_coef=0.2,
        teacher_value_coef=0.2,
        first_minibatch_logratio_limit=None,
        target_kl=None,
    )
    real_collect = trainer._collect_rollout

    def collect_then_perturb() -> torch.Tensor:
        last_values = real_collect()
        if perturb:
            _perturb_scripted_rows(trainer)
        return last_values

    trainer._collect_rollout = collect_then_perturb  # type: ignore[method-assign]
    metrics = trainer.train_iteration()
    kept = {
        key: value
        for key, value in metrics.items()
        if not key.startswith(("time/", "perf/"))
    }
    return kept, {k: v.clone() for k, v in model.state_dict().items()}


@pytest.mark.parametrize(("n_envs", "bot_envs"), [(2, 2), (2, 1)])
def test_learner_mask_excludes_scripted_rows_from_every_loss_term(
    n_envs: int, bot_envs: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    clean_metrics, clean_state = _update_after_perturbation(
        perturb=False, n_envs=n_envs, bot_envs=bot_envs
    )
    dirty_metrics, dirty_state = _update_after_perturbation(
        perturb=True, n_envs=n_envs, bot_envs=bot_envs
    )
    # Every term that exists is active, so each one is covered.
    for term in (
        "loss/policy_loss",
        "loss/value_loss",
        "loss/entropy_loss",
        "loss/teacher_kl_loss",
        "loss/teacher_value_loss",
    ):
        assert clean_metrics[term] != 0.0, term
    assert clean_metrics == dirty_metrics
    assert clean_state.keys() == dirty_state.keys()
    for name, value in clean_state.items():
        assert torch.equal(value, dirty_state[name]), name

    # Mutation check: without the learner mask the scrambled scripted rows
    # reach the losses and the update.
    def no_learner_mask(
        _learner: torch.Tensor, *masks: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        return masks

    monkeypatch.setattr(ppo, "_apply_learner_mask", no_learner_mask)
    mutated_clean, _ = _update_after_perturbation(
        perturb=False, n_envs=n_envs, bot_envs=bot_envs
    )
    mutated_dirty, mutated_state = _update_after_perturbation(
        perturb=True, n_envs=n_envs, bot_envs=bot_envs
    )
    for term in ("loss/policy_loss", "loss/value_loss", "loss/entropy_loss"):
        assert mutated_clean[term] != mutated_dirty[term], term
    assert any(
        not torch.equal(value, mutated_state[name])
        for name, value in clean_state.items()
    )


def test_gae_and_denominators_use_the_learner_seat_rewards_only() -> None:
    # One update's learner-step count is the learner rows alone.
    torch.manual_seed(3)
    env = _env(n_envs=2, bot="starter", bot_envs=2, episode_steps=3)
    trainer = _trainer(env, _tiny_model())
    metrics = trainer.train_iteration()
    assert metrics["train/player_step_total"] == trainer.config.horizon * 2
    assert metrics["train/policy_active_ratio"] == pytest.approx(0.5)


def test_training_telemetry_keys_against_the_bot() -> None:
    torch.manual_seed(17)
    env = _env(n_envs=2, bot="starter", bot_envs=2, episode_steps=3)
    trainer = _trainer(env, _tiny_model(), horizon=2)
    results = [trainer.train_iteration() for _ in range(2)]
    for metrics in results:
        assert all(math.isfinite(value) for value in metrics.values())
        assert metrics["train/bank_games_vs_bot"] == 2.0
        for key in (
            "train/win_rate_vs_bot",
            "train/own_bank_mean_vs_bot",
            "train/opponent_bank_mean_vs_bot",
            "train/margin_mean_vs_bot",
        ):
            assert key in metrics
        # No self-play games in a fraction-1.0 batch.
        assert metrics["train/bank_games"] == 0.0
        assert "train/terminal_learner_seat" not in metrics
        assert metrics["train/margin_mean_vs_bot"] == pytest.approx(
            metrics["train/own_bank_mean_vs_bot"]
            - metrics["train/opponent_bank_mean_vs_bot"]
        )


def test_fixed_opponent_telemetry_helpers() -> None:
    self_play, versus = split_fixed_opponent_games(
        [10.0, 30.0, 50.0, 7.0], [20.0, 30.0, 40.0, 9.0], [0.0, 1.0, -1.0, 1.0]
    )
    assert self_play == ([50.0], [40.0])
    assert versus == ([10.0, 30.0, 9.0], [20.0, 30.0, 7.0])
    metrics = fixed_opponent_metrics(*versus, prefix="train/")
    assert metrics == {
        "train/bank_games_vs_bot": 3.0,
        "train/win_rate_vs_bot": pytest.approx((0.0 + 0.5 + 1.0) / 3),
        "train/own_bank_mean_vs_bot": pytest.approx(49.0 / 3),
        "train/opponent_bank_mean_vs_bot": pytest.approx(57.0 / 3),
        "train/margin_mean_vs_bot": pytest.approx(-8.0 / 3),
    }
    assert fixed_opponent_metrics([], [], prefix="eval/") == {
        "eval/bank_games_vs_bot": 0.0
    }
    with pytest.raises(ValueError, match="-1, 0 or 1"):
        split_fixed_opponent_games([1.0], [2.0], [2.0])


def test_checkpoints_carry_no_opponent_state(tmp_path: Any) -> None:
    torch.manual_seed(19)
    env = _env(n_envs=2, bot="e776", bot_envs=2, episode_steps=3)
    trainer = _trainer(env, _tiny_model())
    trainer.train_iteration()
    path = tmp_path / "checkpoint.pt"
    trainer.write_checkpoint(path, env_steps=6)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert set(checkpoint) <= ppo.CHECKPOINT_KEYS | ppo.OPTIONAL_CHECKPOINT_KEYS
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    assert b"e776" not in buffer.getvalue()
