"""Default-off byte identity of the critic-offset change (model.critic_offset).

Run from a tree root on the pre-change tree (3e89425) and on the change; equal
digests show the flag-off path is byte-identical:
- native: the none-mix native env digest (platform independent);
- trainer: a tiny native 2-update PPO run's metrics and final weights
  (CPU/BLAS dependent: compare on the same Mac at the same OMP_NUM_THREADS);
- model: a tiny model's deterministic forward (actions, log-probs, values,
  winner probabilities) and evaluate_actions on a fixed batch;
- configs: config_sha256 and the config.yaml bytes of every Kaggriculture preset.

Usage: OMP_NUM_THREADS=2 python ops/critic-offset-2026-09-30/baseline_digest.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, ".")
from tests.kaggriculture.test_native_env import (  # noqa: E402
    buffers,
    make_env,
    pass_actions,
)


def native_digest() -> str:
    import numpy as np

    h = hashlib.sha256()
    env = make_env(2, seed=11, stride=2, config='{"episodeSteps":4}')
    arrays = buffers(2)
    env.observe(**arrays)
    tokens, lengths = pass_actions(2)
    for name in sorted(arrays):
        h.update(arrays[name].tobytes())
    for step in range(9):
        info = env.step(tokens, lengths, **arrays)
        h.update(json.dumps(info, sort_keys=True).encode())
        for name in sorted(arrays):
            h.update(arrays[name].tobytes())
        if step == 4:
            env.truncate_envs(np.array([True, False]), **arrays)
            for name in sorted(arrays):
                h.update(arrays[name].tobytes())
    h.update(json.dumps(env.seed_state()).encode())
    return h.hexdigest()


def trainer_digest() -> str:
    from tests.kaggriculture.test_training_smoke import (
        _native_env,
        _tiny_model,
        _trainer,
    )

    torch.manual_seed(307)
    env = _native_env()
    model = _tiny_model()
    trainer = _trainer(env, model)
    h = hashlib.sha256()
    for _ in range(2):
        metrics = trainer.train_iteration()
        keys = sorted(k for k in metrics if not k.startswith(("time/", "perf/")))
        h.update(json.dumps({k: metrics[k] for k in keys}).encode())
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def model_digest() -> str:
    from tests.kaggriculture.conftest import make_obs
    from tests.kaggriculture.helpers import _tiny

    model = _tiny(seed=5).eval()
    obs = make_obs(envs=3, own_actors=[2, 3, 4], rival_actors=[1, 2, 3])
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.numpy().tobytes())
    with torch.no_grad():
        torch.manual_seed(11)
        out = model(obs)
        for tensor in (
            out.actions.tokens,
            out.actions.lengths,
            out.log_probs.event,
            out.values,
            out.winner_probabilities,
        ):
            h.update(tensor.numpy().tobytes())
        evaluation = model.evaluate_actions(obs, out.actions)
        for tensor in (
            evaluation.log_probs.event,
            evaluation.values,
            evaluation.winner_probabilities,
        ):
            h.update(tensor.numpy().tobytes())
        h.update(model.compute_value(obs).numpy().tobytes())
    return h.hexdigest()


def config_digests() -> dict[str, str]:
    from owl.train.config import FullConfig
    from owl.train.logging import config_sha256

    out: dict[str, str] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for path in sorted(Path("configs").glob("kaggriculture*.yaml")):
            cfg = FullConfig.from_file(path)
            dump = Path(tmp) / "config.yaml"
            cfg.to_file(dump)
            out[path.name] = (
                config_sha256(cfg)
                + " "
                + hashlib.sha256(dump.read_bytes()).hexdigest()
            )
    return out


if __name__ == "__main__":
    print(
        json.dumps(
            {
                "native": native_digest(),
                "trainer": trainer_digest(),
                "model": model_digest(),
                "configs": config_digests(),
            },
            indent=1,
            sort_keys=True,
        )
    )
