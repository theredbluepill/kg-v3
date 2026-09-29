"""Isolate the process-global compile state that ``configure_model_compile`` claims.

The trainer claims ``torch._inductor.config.max_autotune_gemm_backends`` for one
game per process. The test session compiles both games' models, so every test
starts unclaimed with the backends it found and restores both afterwards.
"""

from __future__ import annotations

import pytest
import torch._inductor.config as inductor_config
from owl.train import utils as train_utils


@pytest.fixture(autouse=True)
def _isolate_compile_gemm_backend_claim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train_utils, "_GEMM_BACKEND_CLAIM", None)
    monkeypatch.setattr(
        inductor_config,
        "max_autotune_gemm_backends",
        inductor_config.max_autotune_gemm_backends,
    )
