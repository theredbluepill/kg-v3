"""Exercise the original PPO loop against the real v3 native game boundary."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from owl.kaggriculture.env import KaggricultureVectorizedEnv
from owl.kaggriculture.types import KaggricultureActions, KaggricultureObsBatch
from owl.model import create_model
from owl.train import FullConfig, PPOConfig, PPOTrainer
from owl.train.ppo import _flatten_obs_time, _obs_index, _obs_to_device


@pytest.fixture(autouse=True)
def torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_config() -> FullConfig:
    return FullConfig.from_file(
        Path("configs/kaggriculture.yaml"),
        overrides={
            "model.embed_dim": 16,
            "model.depth": 1,
            "model.n_heads": 2,
            "model.mlp_ratio": 2.0,
            "rl.horizon": 2,
            "env.pin_memory": False,
        },
    )


def test_original_ppo_native_update_density_and_checkpoint(tmp_path: Path) -> None:
    torch.manual_seed(10)
    cfg = tiny_config()
    with KaggricultureVectorizedEnv(
        n_envs=2,
        obs_spec=cfg.env.obs_spec,
        action_spec=cfg.env.action_spec,
        pin_memory=False,
        configuration={"episodeSteps": 5},
    ) as env:
        model = create_model(
            cfg.model, obs_spec=cfg.env.obs_spec, action_spec=cfg.env.action_spec
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        trainer = PPOTrainer(
            config=cfg.rl,
            env=env,
            model=model,
            optimizer=optimizer,
            device=torch.device("cpu"),
        )
        trainer._collect_rollout()
        assert isinstance(trainer.rollout.obs, KaggricultureObsBatch)
        assert isinstance(trainer.rollout.actions, KaggricultureActions)
        assert trainer.rollout.logp.shape == (2, 2, 2)
        assert trainer.rollout.obs.context[..., 0].tolist() == [[0, 0], [1, 1]]
        segments = trainer.rollout.segment_major()
        assert isinstance(segments.obs, KaggricultureObsBatch)
        evaluated = model.evaluate_actions(segments.obs, segments.actions)
        torch.testing.assert_close(
            evaluated.log_probs.per_player_entity.sum(-1),
            segments.logp,
            rtol=2e-5,
            atol=2e-5,
        )
        flat = _flatten_obs_time(segments.obs)
        assert isinstance(flat, KaggricultureObsBatch)
        assert flat.features.shape == (4, 2, 8176)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        metrics = trainer.train_iteration()
        assert all(math.isfinite(v) for v in metrics.values())
        assert trainer.optimizer_steps == 1
        assert trainer.total_games_played == 2
        assert any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
        assert metrics["train/action_frames"] >= 8
        checkpoint = tmp_path / "checkpoint.pt"
        trainer.write_checkpoint(checkpoint, env_steps=8, wandb_run_id="local-check")
        saved = {k: v.clone() for k, v in model.state_dict().items()}
        with torch.no_grad():
            next(model.parameters()).add_(1)
        metadata = trainer.load_checkpoint(checkpoint)
        assert metadata.env_steps == 8
        assert metadata.wandb_run_id == "local-check"
        assert trainer.optimizer_steps == 1
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, saved[name], rtol=0, atol=0)


def test_cpu_observation_copy_and_time_limit_bootstrap() -> None:
    cfg = tiny_config()
    with KaggricultureVectorizedEnv(n_envs=2, pin_memory=False) as env:
        obs = env.reset()
        copied = _obs_to_device(obs, torch.device("cpu"))
        assert isinstance(copied, KaggricultureObsBatch)
        assert copied.features.data_ptr() != obs.features.data_ptr()
        model = create_model(
            cfg.model, obs_spec=cfg.env.obs_spec, action_spec=cfg.env.action_spec
        )
        trainer = PPOTrainer(
            config=PPOConfig(
                horizon=2,
                segments_per_minibatch=2,
                model_compile="none",
                truncation_step=1,
                truncation_prob=1.0,
            ),
            env=env,
            model=model,
            optimizer=torch.optim.Adam(model.parameters()),
            device=torch.device("cpu"),
        )
        trainer._collect_rollout()
        assert trainer.rollout.truncated.all()
        assert not trainer.rollout.rewards.any()
        assert torch.isfinite(trainer.rollout.bootstrap_values).all()
        assert env.observations.context[:, 0].tolist() == [0, 0]
        assert copied.context[:, 0].tolist() == [0, 0]
        indexed = _obs_index(copied, torch.tensor([1]))
        assert isinstance(indexed, KaggricultureObsBatch)
        assert indexed.context.shape == (1, 4)


@pytest.mark.parametrize(
    "override",
    [
        {"rl.ppo_clip_mode": "per_entity"},
        {"rl.teacher_mode": "last_best"},
        {"rl.eval_replay_games": 1},
        {"env.reward_mode": "margin"},
    ],
)
def test_invalid_game_training_combinations_fail_before_launch(override) -> None:
    with pytest.raises(ValueError, match=r"Kaggriculture|reward_mode"):
        FullConfig.from_file(Path("configs/kaggriculture.yaml"), overrides=override)


def test_reward_mode_matches_critic_range() -> None:
    for reward, value in (
        ("margin", "margin"),
        ("win_share", "win_loss"),
        ("win_only", "win_only"),
    ):
        config = FullConfig.from_file(
            Path("configs/kaggriculture.yaml"),
            overrides={"env.reward_mode": reward, "model.value_mode": value},
        )
        assert config.model.value_mode == value


def test_evaluation_selects_complete_seat_action_programs() -> None:
    from scripts.run_ppo import _select_actions

    first = KaggricultureActions(
        torch.zeros(2, 2, 252, 12, dtype=torch.int64), torch.full((2, 2), 2)
    )
    second = KaggricultureActions(
        torch.ones(2, 2, 252, 12, dtype=torch.int64), torch.full((2, 2), 3)
    )
    selected = _select_actions(
        first, second, torch.tensor([[True, False], [False, True]])
    )
    assert isinstance(selected, KaggricultureActions)
    assert selected.lengths.tolist() == [[2, 3], [3, 2]]
    assert selected.tokens[:, :, 0, 0].tolist() == [[0, 1], [1, 0]]


def test_checkpoint_selection_uses_bank_winner_when_shaping_reverses_returns() -> None:
    from scripts.run_ppo import _evaluation_outcome

    scores = _evaluation_outcome(
        tiny_config(), {"bank_0": 100.0, "bank_1": 90.0}, torch.tensor([-0.6, 0.6])
    )
    assert scores.tolist() == [100.0, 90.0]


def test_truncation_retains_current_step_economic_rewards() -> None:
    cfg = tiny_config()
    with KaggricultureVectorizedEnv(n_envs=2, pin_memory=False) as env:
        model = create_model(
            cfg.model, obs_spec=cfg.env.obs_spec, action_spec=cfg.env.action_spec
        )
        trainer = PPOTrainer(
            config=PPOConfig(
                horizon=1,
                segments_per_minibatch=2,
                model_compile="none",
                truncation_step=1,
                truncation_prob=1,
            ),
            env=env,
            model=model,
            optimizer=torch.optim.Adam(model.parameters()),
            device=torch.device("cpu"),
        )
        reward = torch.tensor([[0.1, -0.1], [-0.2, 0.2]])
        expected = reward.clone()
        truncated, bootstrap = trainer._apply_truncation(
            env.observations, reward, torch.zeros(2, 2, dtype=torch.bool)
        )
        assert truncated.all()
        torch.testing.assert_close(reward, expected, rtol=0, atol=0)
        assert torch.isfinite(bootstrap).all()


def test_perf_summary_excludes_warmup_and_weights_complete_work() -> None:
    from scripts.benchmark_kaggriculture import summarize_updates

    updates = [
        {
            "train/env_steps": steps,
            "train/player_step_total": 2 * steps,
            "time/iteration_seconds": elapsed,
            "time/rollout_seconds": elapsed / 2,
            "time/update_seconds": elapsed / 3,
        }
        for steps, elapsed in [(100, 50), (200, 2), (300, 3)]
    ]
    result = summarize_updates(updates, warmup=1, measured=2)
    assert result["game_transitions"] == 200
    assert result["learner_seat_turns"] == 400
    assert result["game_sps"] == 40
    assert result["complete_update_seconds"] == 5
    with pytest.raises(ValueError, match="expected 4 updates"):
        summarize_updates(updates, warmup=1, measured=3)
    updates[-1]["loss/total_loss"] = float("nan")
    with pytest.raises(ValueError, match="non-finite training metrics"):
        summarize_updates(updates, warmup=1, measured=2)
