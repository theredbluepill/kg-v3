"""Dry run of the skipped T19b launch tests with the configs dependency bypassed.

Working evidence, not a suite test. Copy to tests/kaggriculture/test_zz_t19b_dryrun.py,
run ``OMP_NUM_THREADS=2 uv run pytest tests/kaggriculture/test_zz_t19b_dryrun.py``,
then delete the copy. It builds the Kaggriculture FullConfig with ``model_construct``
(this branch's FullConfig rejects the Kaggriculture model) and replaces
``FullConfig.from_file``/``to_file``; everything else in the two tests is unchanged.
Results and mutations: t19b-dryrun.log.
"""
from pathlib import Path

import pytest
import scripts.run_ppo as run_ppo
from owl.kaggriculture import types as kt
from owl.model import kaggriculture as km
from owl.rl import EnvConfig
from owl.train import FullConfig, PPOConfig
from owl.train.optimizer import AdamWConfig

import tests.kaggriculture.test_teacher as t


def _cfg() -> FullConfig:
    env = EnvConfig.model_construct(
        n_envs=2, obs_spec=kt.KaggricultureObsConfig(),
        action_spec=kt.KaggricultureActionConfig(), two_player_weight=0.5,
        reward_mode="win_loss", pin_memory=False,
    )
    return FullConfig.model_construct(
        env=env, model=km.KaggricultureTransformerConfig(**t._TINY_MODEL),
        optimizer=AdamWConfig(optimizer="adamw", learning_rate=0.001),
        rl=PPOConfig(horizon=4, teacher_mode="last_best", model_compile="none", compile_mode=None, dtype="float32"),
        runtime=run_ppo.FullConfig.model_fields["runtime"].default_factory(),
    )


@pytest.fixture(autouse=True)
def _bypass(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg()
    monkeypatch.setattr(t, "_kaggriculture_run_config", lambda: cfg)
    monkeypatch.setattr(FullConfig, "to_file", lambda self, path: Path(path).write_text("x"))
    monkeypatch.setattr(run_ppo.FullConfig, "from_file", classmethod(lambda cls, *a, **k: cfg))


def test_resume(tmp_path, monkeypatch):
    t.test_run_ppo_resume_restores_the_teacher_from_checkpoint_last_best(tmp_path, monkeypatch)


def test_fresh(tmp_path, monkeypatch):
    t.test_run_ppo_fresh_launch_from_weights_activates_the_last_best_teacher(tmp_path, monkeypatch)
