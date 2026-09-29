"""Bounded native two-update functional check; no learning-quality claim."""

from __future__ import annotations

import math
from pathlib import Path

import torch
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
from owl.train.ppo import PPOConfig, PPOTrainer


def test_no_teacher_two_updates(tmp_path: Path) -> None:
    torch.manual_seed(307)
    device = torch.device("cpu")
    obs_spec, action_spec = KaggricultureObsConfig(), KaggricultureActionConfig()
    env = KaggricultureVectorizedEnv(
        n_envs=1,
        seed=307,
        seed_stride=1,
        config=KaggricultureGameConfig(episodeSteps=3),
        reward_config=KaggricultureRewardConfig(
            econ_shaping=0.2,
            econ_starvation_weight=4.0,
            econ_drought_weight=1.0,
            econ_cap=0.25,
            econ_ineffective_weight=0.0,
            econ_ineffective_cap=0.10,
        ),
        reward_mode="win_loss",
        native_threads=1,
        pin_memory=False,
        transfer_device=device,
        obs_spec=obs_spec,
        action_spec=action_spec,
    )
    model = KaggricultureTransformer(
        KaggricultureTransformerConfig(
            embed_dim=16, depth=1, n_heads=1, mlp_ratio=1, n_scratch_tokens=0
        ),
        obs_spec=obs_spec,
        action_spec=action_spec,
    )
    trainer = PPOTrainer(
        config=PPOConfig(
            horizon=2,
            ppo_epochs=1,
            segments_per_minibatch=1,
            gradient_accumulation_steps=1,
            dtype="float32",
            model_compile="none",
            compile_mode=None,
            teacher_mode=None,
        ),
        env=env,
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
        device=device,
    )
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    results = [trainer.train_iteration() for _ in range(2)]
    for metrics in results:
        assert all(math.isfinite(value) for value in metrics.values())
        assert "loss/total_loss" in metrics
        assert metrics["teacher/cache_bytes"] == 0
        assert metrics["train/2p_rate"] == 1.0
    terminal = results[1]
    # The engine completes at step episodeSteps - 1: four turns finish two games.
    assert terminal["train/total_games_played"] == 2.0
    assert terminal["train/terminal_bank_0"] >= 0
    assert terminal["train/terminal_bank_1"] >= 0
    assert math.isclose(
        terminal["train/terminal_margin_0"],
        terminal["train/terminal_bank_0"] - terminal["train/terminal_bank_1"],
    )
    assert trainer.optimizer_steps == 2
    assert trainer.player_step_total == 8
    assert trainer.total_games_played == 2
    assert 0 < trainer.total_active_entities <= 2 * 2 * 2 * 252
    assert trainer.target_kl_exceeded_total == 0
    assert trainer._hidden_state is None
    assert trainer.teacher_model is None
    assert any(
        not torch.equal(initial[name], value)
        for name, value in model.state_dict().items()
    )
    checkpoint = tmp_path / "native-two-updates.pt"
    trainer.write_checkpoint(checkpoint, env_steps=4)
    saved = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        next(model.parameters()).zero_()
    metadata = trainer.load_checkpoint(checkpoint)
    assert metadata.env_steps == 4
    assert metadata.player_step_total == 8
    assert metadata.total_games_played == 2
    assert trainer.optimizer_steps == 2
    assert all(
        torch.equal(value, saved[name]) for name, value in model.state_dict().items()
    )
