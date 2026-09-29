"""Pod-only real-binding DMA proof with a fence-removal control."""

from __future__ import annotations

import pytest
import torch

from tests.kaggriculture.test_env import make_env, native_actions, output_tensors


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_step_does_not_overwrite_pending_dma(monkeypatch):
    device = torch.device("cuda")
    env = make_env(pin_memory=True, transfer_device=device)

    def delayed_read():
        actions = native_actions(env)
        buffers = output_tensors(env)
        before = {name: value.clone() for name, value in buffers.items()}
        copies = {
            name: torch.empty_like(value, device=device)
            for name, value in buffers.items()
        }
        torch.cuda.synchronize(device)
        torch.cuda._sleep(200_000_000)
        for name, value in buffers.items():
            copies[name].copy_(value, non_blocking=True)
        env.step(actions)
        torch.cuda.synchronize(device)
        return before, {name: value.cpu() for name, value in copies.items()}

    before, after = delayed_read()
    for name in before:
        assert torch.equal(before[name], after[name]), name
    with monkeypatch.context() as mutation:
        mutation.setattr(env, "_fence", lambda: None)
        before, after = delayed_read()
        assert any(not torch.equal(before[name], after[name]) for name in before), (
            "fence removal must expose pending DMA overwrite; increase the delay "
            "if this pod completes it before native step"
        )
