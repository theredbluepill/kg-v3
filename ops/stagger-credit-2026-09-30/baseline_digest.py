"""Digests of the none-mix native env and a tiny 2-update PPO run.

Run once on the pre-change tree (25412a7) and again after the opponent-mix
change; equal digests show the default path is byte-identical. The native digest
is platform independent; the trainer digest is CPU/BLAS dependent (same Mac).
"""

from __future__ import annotations

import hashlib
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from tests.kaggriculture.test_native_env import buffers, make_env, pass_actions  # noqa: E402


def native_digest() -> str:
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
    from tests.kaggriculture.test_training_smoke import _native_env, _tiny_model, _trainer

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


if __name__ == "__main__":
    print(json.dumps({"native": native_digest(), "trainer": trainer_digest()}))
