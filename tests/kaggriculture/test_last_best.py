"""Real native PPO/checkpoint/eval coverage for the retained incumbent loop."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from owl.kaggriculture.env import KaggricultureVectorizedEnv
from owl.model import create_model
from owl.train import FullConfig, PPOTrainer
from owl.train.distributed import DistributedContext

from scripts import run_ppo


@pytest.fixture(autouse=True)
def one_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _config() -> FullConfig:
    return FullConfig.from_file(
        Path("configs/kaggriculture.yaml"),
        overrides={
            "model.embed_dim": 16,
            "model.depth": 1,
            "model.n_heads": 2,
            "model.mlp_ratio": 2.0,
            "rl.horizon": 2,
            "rl.checkpoint_freq": 1000,
            "env.pin_memory": False,
        },
    )


class _Logger:
    def __init__(self) -> None:
        self.events: list[tuple[int, dict[str, float]]] = []

    def log(self, metrics: dict[str, float], *, step: int) -> None:
        self.events.append((step, metrics))


def _native_factory(monkeypatch: pytest.MonkeyPatch):
    created = []

    def create_short_env(**kwargs):
        # Only shorten the episode; use the actual native factory boundary,
        # observations, actions, termination, terminal banks and auto-reset.
        assert kwargs["two_player_weight"] == 1
        env = KaggricultureVectorizedEnv(
            n_envs=kwargs["n_envs"],
            obs_spec=kwargs["obs_spec"],
            action_spec=kwargs["action_spec"],
            reward_mode=kwargs["reward_mode"],
            reward_shaping=kwargs["reward_shaping"],
            pin_memory=False,
            seed=kwargs["seed"],
            threads=kwargs["native_threads"],
            configuration={"episodeSteps": 3},
        )
        created.append(env)
        return env

    monkeypatch.setattr(run_ppo, "create_env", create_short_env)
    return created


def _trainer(cfg: FullConfig, env: KaggricultureVectorizedEnv) -> PPOTrainer:
    model = create_model(
        cfg.model, obs_spec=cfg.env.obs_spec, action_spec=cfg.env.action_spec
    )
    return PPOTrainer(
        config=cfg.rl,
        env=env,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        device=torch.device("cpu"),
    )


def _state(model) -> dict[str, torch.Tensor]:
    return {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}


def _assert_state(actual, expected) -> None:
    assert actual.keys() == expected.keys()
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def _loop(trainer, cfg, run_dir, logger, **kwargs):
    return run_ppo._run_training_loop(
        trainer=trainer,
        logger=logger,
        run_dir=run_dir,
        cfg=cfg,
        env_steps_per_iteration=4,
        # Begin just before the real minimum checkpoint cadence. Only four
        # native game transitions are needed to cross its boundary.
        start_env_steps=kwargs.pop("start_env_steps", 996),
        max_env_steps=kwargs.pop("max_env_steps", 1000),
        max_runtime_seconds=None,
        dist_ctx=DistributedContext.single_process_cpu(),
        wandb_run_id="native-incumbent-test",
        **kwargs,
    )


@pytest.mark.parametrize(("selection_rate", "promoted"), [(0.699, False), (0.7, True)])
def test_native_checkpoint_loop_preserves_initial_best_and_promotion_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selection_rate: float,
    promoted: bool,
) -> None:
    torch.manual_seed(83)
    cfg = _config()
    environments = _native_factory(monkeypatch)
    actual_evaluate = run_ppo._evaluate_against_last_best
    observations = []

    def evaluate_with_controlled_selection(**kwargs):
        # Run genuine native evaluation, then control only the selection scalar
        # so both sides of the fixed threshold are reproducible in a tiny test.
        initial_best = torch.load(
            tmp_path / run_ppo.CHECKPOINT_LAST_BEST, weights_only=False
        )
        result = actual_evaluate(**kwargs)
        assert 0 <= result["eval/win_rate_against_last_best"] <= 1
        assert "eval/win_rate_against_last_best_2p" in result
        assert "eval/win_rate_against_last_best_4p" not in result
        observations.append(initial_best)
        return {**result, "eval/win_rate_against_last_best": selection_rate}

    monkeypatch.setattr(
        run_ppo, "_evaluate_against_last_best", evaluate_with_controlled_selection
    )
    try:
        with KaggricultureVectorizedEnv(
            n_envs=2, pin_memory=False, configuration={"episodeSteps": 3}
        ) as env:
            trainer = _trainer(cfg, env)
            initial = _state(trainer.model)
            logger = _Logger()
            assert _loop(trainer, cfg, tmp_path, logger) == 1000
            assert len(observations) == 1
            assert observations[0]["env_steps"] == 996
            _assert_state(observations[0]["model"], initial)
            numbered = torch.load(
                tmp_path / "checkpoint_00_000_001_000.pt", weights_only=False
            )
            incumbent = torch.load(
                tmp_path / run_ppo.CHECKPOINT_LAST_BEST, weights_only=False
            )
            assert numbered["env_steps"] == 1000
            assert numbered["optimizer_steps"] == 1
            assert numbered["wandb_run_id"] == "native-incumbent-test"
            assert any(
                not torch.equal(initial[k], numbered["model"][k]) for k in initial
            )
            assert incumbent["env_steps"] == (1000 if promoted else 996)
            _assert_state(
                incumbent["model"], numbered["model"] if promoted else initial
            )
            assert logger.events[-1][0] == 1000
            assert (
                logger.events[-1][1]["eval/win_rate_against_last_best"]
                == selection_rate
            )
            assert environments
            assert environments[0].terminal_metrics(0) is not None
    finally:
        for environment in environments:
            environment.close()


def test_native_resume_preserves_separate_current_and_last_best_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(92)
    cfg = _config()
    cfg.to_file(tmp_path / "config.yaml")
    environments = _native_factory(monkeypatch)
    actual_evaluate = run_ppo._evaluate_against_last_best

    def nonpromoting_evaluate(**kwargs):
        result = actual_evaluate(**kwargs)
        return {**result, "eval/win_rate_against_last_best": 0.0}

    monkeypatch.setattr(run_ppo, "_evaluate_against_last_best", nonpromoting_evaluate)
    try:
        with KaggricultureVectorizedEnv(
            n_envs=2, pin_memory=False, configuration={"episodeSteps": 3}
        ) as env:
            trainer = _trainer(cfg, env)
            assert _loop(trainer, cfg, tmp_path, _Logger()) == 1000
            current_state = _state(trainer.model)
            incumbent_bytes = (tmp_path / run_ppo.CHECKPOINT_LAST_BEST).read_bytes()

        launch = run_ppo._resolve_resume_launch(tmp_path)
        assert launch.checkpoint_path.name == "checkpoint_00_000_001_000.pt"
        assert launch.last_best_checkpoint_path.name == run_ppo.CHECKPOINT_LAST_BEST
        with KaggricultureVectorizedEnv(
            n_envs=2, pin_memory=False, configuration={"episodeSteps": 3}
        ) as resumed_env:
            resumed = _trainer(cfg, resumed_env)
            current_metadata = resumed.load_checkpoint(launch.checkpoint_path)
            _assert_state(resumed.model.state_dict(), current_state)
            assert resumed.optimizer_steps == 1
            last_best = run_ppo._create_eval_model_for_config(
                cfg, device=torch.device("cpu")
            )
            best_metadata = run_ppo._load_model_from_checkpoint(
                last_best,
                path=launch.last_best_checkpoint_path,
                device=torch.device("cpu"),
            )
            run_ppo._validate_last_best_run_id(
                best_metadata,
                resume_run_id=current_metadata.wandb_run_id,
                checkpoint_path=launch.last_best_checkpoint_path,
            )
            assert current_metadata.env_steps == 1000
            assert best_metadata.env_steps == 996
            assert any(
                not torch.equal(current_state[k], last_best.state_dict()[k])
                for k in current_state
            )
            with pytest.raises(ValueError, match="does not match"):
                run_ppo._validate_last_best_run_id(
                    best_metadata,
                    resume_run_id="different-run",
                    checkpoint_path=launch.last_best_checkpoint_path,
                )
            assert (
                _loop(
                    resumed,
                    cfg,
                    tmp_path,
                    _Logger(),
                    start_env_steps=1000,
                    max_env_steps=1004,
                    last_best_model=last_best,
                )
                == 1004
            )
            assert resumed.optimizer_steps == 2
            # Resume retains the 1,000-step cadence; no checkpoint at 1,004.
            assert not (tmp_path / "checkpoint_00_000_001_004.pt").exists()
            assert (
                run_ppo._resolve_resume_launch(tmp_path).checkpoint_path
                == launch.checkpoint_path
            )
            assert (
                tmp_path / run_ppo.CHECKPOINT_LAST_BEST
            ).read_bytes() == incumbent_bytes
    finally:
        for environment in environments:
            environment.close()
