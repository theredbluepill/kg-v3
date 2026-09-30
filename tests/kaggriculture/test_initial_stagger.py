"""rl.initial_stagger (staggered game phases) and the long credit window.

Stagger: every env's first game is cut at a per-env step through the stateless
truncation path, so envs sit in different game phases afterwards. Credit: the
presets' 256-step segments with lambda 1, whose advantage is the Monte Carlo
return to the segment end plus the critic's bootstrap. No learning-quality
claim.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any

import pytest
import torch
from owl.kaggriculture.config import KaggricultureEnvConfig
from owl.kaggriculture.env import KaggricultureVectorizedEnv
from owl.kaggriculture.rewards import KaggricultureRewardConfig
from owl.kaggriculture.types import (
    KaggricultureActionConfig,
    KaggricultureGameConfig,
    KaggricultureObsConfig,
)
from owl.model.kaggriculture import (
    KaggricultureTransformer,
    KaggricultureTransformerConfig,
)
from owl.model.kaggriculture_teacher import TEACHER_TARGET_BYTES_PER_ROW
from owl.model.kaggriculture_workload import (
    check_workload_headroom,
    ppo_forward_workloads,
)
from owl.train import FullConfig, ppo
from owl.train.advantages import compute_gae
from owl.train.ppo import (
    STAGGER_PHASE_BUCKETS,
    InitialStagger,
    PPOConfig,
    PPOTrainer,
    initial_stagger_steps,
)
from pydantic import BaseModel, ValidationError

ROOT = Path(__file__).parents[2]
_CONFIGS = ROOT / "configs"
# The Kaggriculture game: 720 episode steps, 719 env transitions.
_EPISODE_STEPS = 720
_GAME_STEPS = _EPISODE_STEPS - 1
_GLOBAL_ENV_STEPS_PER_ITERATION = 16_384


# --- configuration ------------------------------------------------------------


def test_default_off_is_omitted_from_the_config_dump() -> None:
    # Existing configs keep their config.yaml and v3/config_sha256.
    assert "initial_stagger" not in PPOConfig().model_dump(mode="json")
    assert "initial_stagger" not in PPOConfig().model_dump(mode="json", round_trip=True)
    on = PPOConfig(initial_stagger=True)
    assert on.model_dump(mode="json")["initial_stagger"] is True
    assert PPOConfig.model_validate(on.model_dump(mode="json")) == on
    for name in ("kaggriculture_4rank.yaml", "kaggriculture_4rank_margin.yaml"):
        cfg = FullConfig.from_file(_CONFIGS / name)
        assert not cfg.rl.initial_stagger
        assert "initial_stagger" not in cfg.model_dump(mode="json")["rl"]


def test_stagger_cannot_combine_with_per_game_truncation() -> None:
    with pytest.raises(ValidationError, match="initial_stagger"):
        PPOConfig(initial_stagger=True, truncation_step=5, truncation_prob=0.5)
    with pytest.raises(ValidationError, match="initial_stagger"):
        PPOConfig(initial_stagger=True, truncation_step=5)


def test_stagger_is_kaggriculture_only_and_needs_a_game_step() -> None:
    orbit = FullConfig.from_file(_CONFIGS / "scaling_6m.yaml").model_dump(
        mode="json", round_trip=True
    )
    orbit["rl"]["initial_stagger"] = True
    with pytest.raises(ValidationError, match="Kaggriculture-only"):
        FullConfig.model_validate(orbit)
    farm = FullConfig.from_file(_CONFIGS / "kaggriculture.yaml").model_dump(
        mode="json", round_trip=True
    )
    farm["rl"]["initial_stagger"] = True
    farm["env"]["config"]["episode_steps"] = 1
    with pytest.raises(ValidationError, match="episodeSteps >= 2"):
        FullConfig.model_validate(farm)


# --- offsets --------------------------------------------------------------------


def test_offsets_are_deterministic_uniform_draws_per_global_env() -> None:
    first = initial_stagger_steps(
        seed=5, rank=0, n_envs=64, episode_steps=_EPISODE_STEPS
    )
    again = initial_stagger_steps(
        seed=5, rank=0, n_envs=64, episode_steps=_EPISODE_STEPS
    )
    assert first.game_steps == _GAME_STEPS
    assert torch.equal(first.first_game_steps, again.first_game_steps)
    steps = first.first_game_steps
    assert steps.dtype == torch.long
    assert int(steps.min()) >= 1
    assert int(steps.max()) <= _GAME_STEPS
    # Independent draws, so envs differ except for birthday collisions: 64
    # draws from 719 values share about 2.7 on average (61 distinct here). The
    # ranks and seeds draw different vectors.
    assert len(set(steps.tolist())) >= 60
    ranks = [
        initial_stagger_steps(
            seed=5, rank=rank, n_envs=64, episode_steps=_EPISODE_STEPS
        ).first_game_steps
        for rank in range(4)
    ]
    for a in range(4):
        for b in range(a + 1, 4):
            assert not torch.equal(ranks[a], ranks[b])
    other_seed = initial_stagger_steps(
        seed=6, rank=0, n_envs=64, episode_steps=_EPISODE_STEPS
    )
    assert not torch.equal(other_seed.first_game_steps, steps)
    # The offset depends on the global env index only, not on the split.
    whole = initial_stagger_steps(
        seed=5, rank=0, n_envs=256, episode_steps=_EPISODE_STEPS
    ).first_game_steps
    assert torch.equal(torch.cat(ranks), whole)
    # 256 draws cover the game: every sixth of it holds some env's offset.
    buckets = (whole - 1) * STAGGER_PHASE_BUCKETS // _GAME_STEPS
    assert set(buckets.tolist()) == set(range(STAGGER_PHASE_BUCKETS))


def test_offsets_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="episodeSteps >= 2"):
        initial_stagger_steps(seed=0, rank=0, n_envs=2, episode_steps=1)
    with pytest.raises(ValueError, match="n_envs"):
        initial_stagger_steps(seed=0, rank=0, n_envs=0, episode_steps=720)
    with pytest.raises(ValueError, match=r"1\.\.5"):
        InitialStagger(first_game_steps=torch.tensor([0, 3]), game_steps=5)
    with pytest.raises(ValueError, match=r"1\.\.5"):
        InitialStagger(first_game_steps=torch.tensor([6]), game_steps=5)
    with pytest.raises(ValueError, match="int64"):
        InitialStagger(first_game_steps=torch.tensor([1.0]), game_steps=5)


def _game_ends_per_iteration(
    offsets: torch.Tensor, *, horizon: int, iterations: int, game_steps: int
) -> list[int]:
    """Natural game ends per rollout under the stagger schedule.

    Env ``i`` is cut after ``u_i`` transitions (or completes its first game
    there when ``u_i == game_steps``); its later games end every
    ``game_steps`` transitions after that, at transition indices
    ``u_i - 1 + k * game_steps`` (``k >= 1``, and ``k = 0`` when uncut).
    """
    ends = [0] * iterations
    for u in offsets.tolist():
        k = 0 if u == game_steps else 1
        while (index := u - 1 + k * game_steps) < horizon * iterations:
            ends[index // horizon] += 1
            k += 1
    return ends


@pytest.mark.parametrize(
    ("world_size", "n_envs", "horizon"),
    [(4, 16, 256), (2, 32, 256), (4, 64, 64)],
)
def test_every_rollout_after_the_first_cycle_carries_game_ends(
    world_size: int, n_envs: int, horizon: int
) -> None:
    assert world_size * n_envs * horizon == _GLOBAL_ENV_STEPS_PER_ITERATION
    offsets = torch.cat(
        [
            initial_stagger_steps(
                seed=seed, rank=rank, n_envs=n_envs, episode_steps=_EPISODE_STEPS
            ).first_game_steps
            for seed in (0,)
            for rank in range(world_size)
        ]
    )
    # The layout's global offsets (the presets' 64 at env.seed 0) cover every
    # sixth of the game.
    buckets = (offsets - 1) * STAGGER_PHASE_BUCKETS // _GAME_STEPS
    assert set(buckets.tolist()) == set(range(STAGGER_PHASE_BUCKETS))
    iterations = 40
    ends = _game_ends_per_iteration(
        offsets, horizon=horizon, iterations=iterations, game_steps=_GAME_STEPS
    )
    first_cycle = math.ceil(2 * _GAME_STEPS / horizon)
    assert all(count > 0 for count in ends[first_cycle:])
    mean = sum(ends[first_cycle:]) / len(ends[first_cycle:])
    # Global envs x horizon / 719 steps per game.
    assert mean == pytest.approx(world_size * n_envs * horizon / _GAME_STEPS, rel=0.1)
    # Unstaggered, every env ends its games together: most rollouts carry none.
    lockstep = _game_ends_per_iteration(
        torch.full((world_size * n_envs,), _GAME_STEPS),
        horizon=horizon,
        iterations=iterations,
        game_steps=_GAME_STEPS,
    )
    assert sum(count == 0 for count in lockstep) > iterations // 2


# --- the native trainer -------------------------------------------------------


def _native_env(
    *, n_envs: int, episode_steps: int, opponent_envs: int = 0
) -> KaggricultureVectorizedEnv:
    return KaggricultureVectorizedEnv(
        n_envs=n_envs,
        seed=401,
        seed_stride=1,
        config=KaggricultureGameConfig(episodeSteps=episode_steps),
        reward_config=KaggricultureRewardConfig(
            econ_shaping=0.0,
            econ_starvation_weight=4.0,
            econ_drought_weight=1.0,
            econ_cap=0.25,
            econ_ineffective_weight=0.0,
            econ_ineffective_cap=0.10,
            econ_bank_weight=0.25,
            econ_bank_scale=150_000.0,
            econ_bank_cap=0.25,
            econ_margin_weight=0.25,
            econ_margin_scale=100_000.0,
            econ_margin_cap=0.25,
        ),
        reward_mode="win_loss",
        native_threads=1,
        pin_memory=False,
        transfer_device=torch.device("cpu"),
        obs_spec=KaggricultureObsConfig(),
        action_spec=KaggricultureActionConfig(),
        opponent_bot="starter" if opponent_envs else None,
        opponent_envs=opponent_envs,
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
    *,
    horizon: int,
    stagger: InitialStagger | None,
    **config: Any,
) -> PPOTrainer:
    return PPOTrainer(
        config=PPOConfig.model_validate(
            {
                "horizon": horizon,
                "ppo_epochs": 1,
                "segments_per_minibatch": 1,
                "gradient_accumulation_steps": 1,
                "gamma": 1.0,
                "gae_lambda": 1.0,
                "dtype": "float32",
                "model_compile": "none",
                "compile_mode": None,
                "teacher_mode": None,
                "initial_stagger": stagger is not None,
                **config,
            }
        ),
        env=env,
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
        device=torch.device("cpu"),
        initial_stagger=stagger,
    )


def test_trainer_requires_offsets_exactly_when_configured() -> None:
    env = _native_env(n_envs=2, episode_steps=6)
    stagger = InitialStagger(first_game_steps=torch.tensor([1, 2]), game_steps=5)
    with pytest.raises(ValueError, match="InitialStagger"):
        PPOTrainer(
            config=PPOConfig(initial_stagger=True, teacher_mode=None),
            env=env,
            model=_tiny_model(),
            optimizer=torch.optim.Adam(_tiny_model().parameters()),
            device=torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="InitialStagger"):
        PPOTrainer(
            config=PPOConfig(teacher_mode=None),
            env=env,
            model=_tiny_model(),
            optimizer=torch.optim.Adam(_tiny_model().parameters()),
            device=torch.device("cpu"),
            initial_stagger=stagger,
        )
    with pytest.raises(ValueError, match="shape"):
        _trainer(
            env,
            _tiny_model(),
            horizon=2,
            stagger=InitialStagger(
                first_game_steps=torch.tensor([1, 2, 3]), game_steps=5
            ),
        )


def test_stagger_cuts_only_first_games_and_publishes_no_cut_game(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(401)
    # episodeSteps 6: five transitions per game. The horizon equals a game, so
    # after the first rollout every env ends exactly one game per rollout.
    game_steps, n_envs = 5, 4
    env = _native_env(n_envs=n_envs, episode_steps=game_steps + 1)
    offsets = torch.tensor([1, 3, 4, 5])
    stagger = InitialStagger(first_game_steps=offsets, game_steps=game_steps)
    model = _tiny_model()
    trainer = _trainer(env, model, horizon=game_steps, stagger=stagger)
    # Record every cut (env mask), each rollout step's inputs to the policy
    # forward and each env's published observation at that moment.
    cuts: list[torch.Tensor] = []
    truncate = env.truncate_envs

    def recording_truncate(mask: torch.Tensor) -> object:
        cuts.append(mask.clone())
        return truncate(mask)

    forward_inputs: list[tuple[Any, Any, Any]] = []
    model_forward = ppo._model_forward

    def recording_forward(model: Any, obs: Any, *, hidden_state: Any) -> Any:
        forward_inputs.append(
            (
                obs.model_copy(deep=True),
                env.observations.model_copy(deep=True),
                hidden_state,
            )
        )
        return model_forward(model, obs, hidden_state=hidden_state)

    monkeypatch.setattr(env, "truncate_envs", recording_truncate)
    monkeypatch.setattr(ppo, "_model_forward", recording_forward)
    results = []
    rollouts = []
    for _ in range(4):
        results.append(trainer.train_iteration())
        rollouts.append(
            (
                trainer.rollout.truncated.clone(),
                trainer.rollout.dones.clone(),
                trainer.rollout.bootstrap_values.clone(),
            )
        )
    for metrics in results:
        assert all(math.isfinite(value) for value in metrics.values())
        phases = [metrics[f"train/game_phase_frac_{k}"] for k in range(6)]
        assert sum(phases) == pytest.approx(1.0)
    # Only the first rollout cuts, and only first games with offset < 5; the
    # offset-5 env completes its first game naturally at transition 5.
    cut_envs = torch.stack(cuts).nonzero()[:, 1].tolist()
    assert sorted(cut_envs) == [0, 1, 2]
    assert [m["train/stagger_cuts"] for m in results] == [3.0, 0.0, 0.0, 0.0]
    first_truncated, first_dones, first_bootstrap = rollouts[0]
    for env_index, u in enumerate(offsets.tolist()):
        cut_steps = first_truncated[:, env_index, 0].nonzero().flatten().tolist()
        assert cut_steps == ([] if u == game_steps else [u - 1])
        for step in cut_steps:
            # The cut ends the trajectory and bootstraps from the critic.
            assert bool(first_dones[step, env_index].all())
            assert bool(first_truncated[step, env_index].all())
            assert bool(torch.isfinite(first_bootstrap[step, env_index]).all())
            assert bool(first_bootstrap[step, env_index].ne(0).any())
    for truncated, _dones, bootstrap in rollouts:
        assert torch.equal(
            bootstrap[~truncated], torch.zeros_like(bootstrap[~truncated])
        )
    for truncated, _dones, _bootstrap in rollouts[1:]:
        assert not bool(truncated.any())
    # Game ends: only the uncut env in the first rollout, then every env once
    # per rollout. Bank telemetry counts exactly those completed games.
    assert [m["train/game_ends"] for m in results] == [1.0, 4.0, 4.0, 4.0]
    assert [m["train/bank_games"] for m in results] == [1.0, 4.0, 4.0, 4.0]
    assert trainer.total_games_played == 13
    # After the first rollout each env acts once in every game step 0..4.
    expected = [0.0] * STAGGER_PHASE_BUCKETS
    for step in range(game_steps):
        expected[step * STAGGER_PHASE_BUCKETS // game_steps] += 1 / game_steps
    for metrics in results[1:]:
        assert [
            metrics[f"train/game_phase_frac_{k}"] for k in range(STAGGER_PHASE_BUCKETS)
        ] == pytest.approx(expected)
    # Stateless: the policy's input is the env's published observation and
    # nothing else; no hidden state crosses a step.
    assert len(forward_inputs) == 4 * game_steps
    for obs, published, hidden_state in forward_inputs:
        assert hidden_state is None
        assert _tensors(obs) == _tensors(published)


def _tensors(value: object, prefix: str = "") -> dict[str, bytes]:
    """Every tensor of a (nested) observation model, by field path."""
    if isinstance(value, torch.Tensor):
        return {prefix: value.detach().cpu().contiguous().numpy().tobytes()}
    if isinstance(value, BaseModel):
        names = list(type(value).model_fields)
    else:
        assert dataclasses.is_dataclass(value), type(value)
        names = [field.name for field in dataclasses.fields(value)]
    out: dict[str, bytes] = {}
    for name in names:
        out.update(_tensors(getattr(value, name), f"{prefix}.{name}"))
    return out


def test_stagger_works_with_a_fixed_opponent_mix() -> None:
    # One hosted-bot env and one self-play env; the cut is a reset, so the
    # learner's seat in the cut env switches with the new game.
    torch.manual_seed(421)
    env = _native_env(n_envs=2, episode_steps=6, opponent_envs=1)
    stagger = InitialStagger(first_game_steps=torch.tensor([2, 3]), game_steps=5)
    trainer = _trainer(env, _tiny_model(), horizon=5, stagger=stagger)
    assert trainer.rollout.learner is not None
    first = [trainer.train_iteration()]
    assert trainer.rollout.learner is not None
    rollout_one_learner = trainer.rollout.learner.clone()
    first.append(trainer.train_iteration())
    for metrics in first:
        assert all(math.isfinite(value) for value in metrics.values())
    assert [m["train/stagger_cuts"] for m in first] == [2.0, 0.0]
    # Second games end at transitions 6 and 7 (rollout 2): one vs the bot,
    # one self-play.
    assert [m["train/game_ends"] for m in first] == [0.0, 2.0]
    assert [m["train/bank_games_vs_bot"] for m in first] == [0.0, 1.0]
    assert [m["train/bank_games"] for m in first] == [0.0, 1.0]
    # In rollout 1 the hosted env's cut after transition 2 (step index 1)
    # starts a new game, whose learner seat is the other one.
    assert torch.equal(rollout_one_learner[0, 0], rollout_one_learner[1, 0])
    assert not torch.equal(rollout_one_learner[1, 0], rollout_one_learner[2, 0])


def test_default_off_trainer_has_no_stagger_state_or_metrics() -> None:
    torch.manual_seed(409)
    env = _native_env(n_envs=2, episode_steps=4)
    trainer = _trainer(env, _tiny_model(), horizon=2, stagger=None)
    metrics = trainer.train_iteration()
    assert not trainer._truncation_enabled
    assert not any(
        key.startswith(("train/game_phase_frac", "train/game_ends", "train/stagger"))
        for key in metrics
    )
    assert trainer._phase_counts.numel() == 0
    assert trainer.rollout.truncated.any().item() is False


def test_stagger_metrics_are_reduced_across_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The phase mix, game ends and cuts go through ``all_reduce_sum``."""
    torch.manual_seed(411)
    env = _native_env(n_envs=2, episode_steps=4)
    stagger = InitialStagger(first_game_steps=torch.tensor([1, 2]), game_steps=3)
    trainer = _trainer(env, _tiny_model(), horizon=3, stagger=stagger)
    local = {
        key: value
        for key, value in trainer.train_iteration().items()
        if key.startswith(("train/game_phase_frac", "train/game_ends", "train/stagger"))
    }
    assert local["train/stagger_cuts"] == 2.0
    reduced: list[torch.Tensor] = []

    def two_identical_ranks(tensor: torch.Tensor, _context: object) -> torch.Tensor:
        reduced.append(tensor.clone())
        return tensor * 2

    monkeypatch.setattr(ppo, "all_reduce_sum", two_identical_ranks)
    doubled = trainer._stagger_metrics()
    assert len(reduced) == 1
    assert reduced[0].numel() == STAGGER_PHASE_BUCKETS + 2
    assert doubled["train/stagger_cuts"] == 2 * local["train/stagger_cuts"]
    assert doubled["train/game_ends"] == 2 * local["train/game_ends"]
    for bucket in range(STAGGER_PHASE_BUCKETS):
        key = f"train/game_phase_frac_{bucket}"
        assert doubled[key] == pytest.approx(local[key])


# --- the credit window ----------------------------------------------------------


def test_gae_lambda_one_is_the_monte_carlo_return_plus_bootstrap() -> None:
    generator = torch.Generator().manual_seed(419)
    n_envs, horizon, seats = 3, 256, 2
    shape = (n_envs, horizon, seats)
    rewards = torch.randn(shape, generator=generator, dtype=torch.float64)
    values = torch.randn(shape, generator=generator, dtype=torch.float64)
    last_values = torch.randn((n_envs, seats), generator=generator, dtype=torch.float64)
    dones = torch.zeros(shape, dtype=torch.bool)
    truncated = torch.zeros(shape, dtype=torch.bool)
    bootstrap = torch.zeros(shape, dtype=torch.float64)
    # Env 0 runs uncut; env 1 ends a game at step 100; env 2 is cut (a stagger
    # truncation) at step 37 and ends a game at step 200.
    dones[1, 100] = True
    dones[2, 37] = True
    truncated[2, 37] = True
    bootstrap[2, 37] = torch.tensor([0.25, -0.75], dtype=torch.float64)
    dones[2, 200] = True
    advantages, returns = compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        last_values=last_values,
        gamma=1.0,
        gae_lambda=1.0,
        truncated=truncated,
        bootstrap_values=bootstrap,
    )
    expected = torch.empty(shape, dtype=torch.float64)
    for env in range(n_envs):
        for t in range(horizon):
            total = torch.zeros(seats, dtype=torch.float64)
            for k in range(t, horizon):
                total = total + rewards[env, k]
                if bool(dones[env, k].all()):
                    if bool(truncated[env, k].all()):
                        total = total + bootstrap[env, k]
                    break
            else:
                total = total + last_values[env]
            expected[env, t] = total
    torch.testing.assert_close(returns, expected, rtol=0.0, atol=1e-9)
    torch.testing.assert_close(advantages, expected - values, rtol=0.0, atol=1e-9)
    # The first turn's credit reaches the segment end: a reward at step 255
    # moves the step-0 advantage of an uncut env one for one.
    bumped = rewards.clone()
    bumped[0, horizon - 1] += 1.0
    bumped_advantages, _ = compute_gae(
        rewards=bumped,
        values=values,
        dones=dones,
        last_values=last_values,
        gamma=1.0,
        gae_lambda=1.0,
        truncated=truncated,
        bootstrap_values=bootstrap,
    )
    torch.testing.assert_close(
        bumped_advantages[0, 0] - advantages[0, 0],
        torch.ones(seats, dtype=torch.float64),
    )


_CREDIT = {
    "kaggriculture_4rank_bank_critic_credit.yaml": 4,
    "kaggriculture_2rank_bank_critic_credit.yaml": 2,
}


@pytest.mark.parametrize(("name", "world_size"), _CREDIT.items())
def test_credit_presets_keep_the_global_work_and_per_rank_rows(
    name: str, world_size: int
) -> None:
    cfg = FullConfig.from_file(_CONFIGS / name)
    base = FullConfig.from_file(_CONFIGS / f"kaggriculture_{world_size}rank.yaml")
    assert isinstance(cfg.env, KaggricultureEnvConfig)
    assert isinstance(base.env, KaggricultureEnvConfig)
    rl, env = cfg.rl, cfg.env
    assert rl.initial_stagger
    assert (rl.horizon, rl.gae_lambda, rl.gamma) == (256, 1.0, 1.0)
    assert env.config.episode_steps == _EPISODE_STEPS
    # Equal global env steps and optimizer steps per iteration.
    assert env.n_envs * rl.horizon * world_size == _GLOBAL_ENV_STEPS_PER_ITERATION
    assert (
        base.env.n_envs * base.rl.horizon * world_size
        == _GLOBAL_ENV_STEPS_PER_ITERATION
    )
    divisor = rl.segments_per_minibatch * rl.gradient_accumulation_steps
    assert env.n_envs % divisor == 0
    optimizer_steps = rl.ppo_epochs * env.n_envs // divisor
    base_divisor = base.rl.segments_per_minibatch * base.rl.gradient_accumulation_steps
    assert optimizer_steps == base.rl.ppo_epochs * base.env.n_envs // base_divisor == 16
    # Samples per optimizer step are unchanged: spm x horizon env steps per rank.
    assert (
        rl.segments_per_minibatch * rl.horizon
        == base.rl.segments_per_minibatch * base.rl.horizon
    )
    # Forward rows: the minibatch and teacher rows (and the teacher cache) are
    # unchanged; the rollout and evaluation forwards shrink with n_envs.
    ours = _workload_rows(cfg)
    theirs = _workload_rows(base)
    assert ours["minibatch"] == theirs["minibatch"]
    assert ours["teacher_chunk"] == theirs["teacher_chunk"]
    assert ours["rollout"][0] * 4 == theirs["rollout"][0]
    assert ours["evaluation"][0] * 4 == theirs["evaluation"][0]
    cached_rows = env.n_envs * rl.horizon * 2
    assert (
        cached_rows * TEACHER_TARGET_BYTES_PER_ROW
        == {
            4: 837_287_936,
            2: 1_674_575_872,
        }[world_size]
    )
    # Everything else is the owner reward on the margin preset's recipe.
    shaping = env.reward_shaping
    assert (
        shaping.econ_shaping,
        shaping.econ_bank_weight,
        shaping.econ_bank_scale,
        shaping.econ_bank_cap,
        shaping.econ_margin_weight,
        shaping.econ_margin_scale,
        shaping.econ_margin_cap,
    ) == (0.0, 0.25, 150_000.0, 0.25, 0.25, 100_000.0, 0.25)
    assert shaping.terminal_scale == 0.5
    assert env.opponent_mix is None
    margin = FullConfig.from_file(_CONFIGS / "kaggriculture_4rank_margin.yaml")
    assert isinstance(margin.env, KaggricultureEnvConfig)
    # Undo the reward, the stagger, the credit window and the per-rank split.
    assert (
        cfg.model_copy(
            update={
                "env": env.model_copy(
                    update={
                        "reward_shaping": margin.env.reward_shaping,
                        "n_envs": margin.env.n_envs,
                    }
                ),
                "rl": rl.model_copy(
                    update={
                        "horizon": 64,
                        "gae_lambda": 0.9,
                        "initial_stagger": False,
                        "segments_per_minibatch": margin.rl.segments_per_minibatch,
                    }
                ),
            }
        )
        == margin
    )


def _workload_rows(cfg: FullConfig) -> dict[str, tuple[int, int, int]]:
    reports = check_workload_headroom(
        cfg.model,
        ppo_forward_workloads(
            n_envs=cfg.env.n_envs,
            horizon=cfg.rl.horizon,
            segments_per_minibatch=cfg.rl.segments_per_minibatch,
            teacher_segments_per_minibatch=cfg.rl.teacher_segments_per_minibatch,
        ),
    )
    return {r.name: (r.rows, r.max_trunk_calls, r.head_calls) for r in reports}
