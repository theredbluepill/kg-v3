"""Isolate the process-global compile state that every model compile claims.

``configure_model_compile`` and each ``compile_transformer_trunk`` claim
``torch._inductor.config.max_autotune_gemm_backends`` for one game per process
(``owl.model.compile_gemm``). The test session compiles both games' models, so
every test starts unclaimed with the backends it found and restores both
afterwards.
"""

from __future__ import annotations

import pytest
import torch._inductor.config as inductor_config
from owl.model import compile_gemm


@pytest.fixture(autouse=True)
def _isolate_compile_gemm_backend_claim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compile_gemm, "_GEMM_BACKEND_CLAIM", None)
    monkeypatch.setattr(
        inductor_config,
        "max_autotune_gemm_backends",
        inductor_config.max_autotune_gemm_backends,
    )
