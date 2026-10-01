"""Bounded benchmark admission and CPU arm reporting."""

from __future__ import annotations

import importlib.util
import math
import sys
from argparse import Namespace
from pathlib import Path

import pytest
from owl.train.config import FullConfig

_ROOT = Path(__file__).parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "bench_rollout_step", _ROOT / "scripts/bench_rollout_step.py"
)
assert _SPEC is not None
assert _SPEC.loader is not None
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--seat-rows", "1"],
        ["--seat-rows", "3"],
        ["--steps", "0"],
        ["--warmup", "-1"],
        ["--threads", "0"],
    ],
)
def test_cli_rejects_invalid_work_before_allocating(
    monkeypatch: pytest.MonkeyPatch, arguments: list[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["bench_rollout_step", *arguments])
    monkeypatch.setattr(
        bench,
        "benchmark_arm",
        lambda *_args, **_kwargs: pytest.fail("invalid workload reached benchmark"),
    )
    with pytest.raises(SystemExit) as error:
        bench.main()
    assert error.value.code == 2


@pytest.mark.parametrize("with_env", [False, True])
def test_cpu_arms_report_effective_switches_and_completed_work(
    tmp_path: Path, with_env: bool
) -> None:
    cfg = FullConfig.from_file(
        _ROOT / "configs/kaggriculture_4rank_bank_critic.yaml",
        {
            "model.embed_dim": 16,
            "model.depth": 1,
            "model.n_heads": 2,
            "model.n_scratch_tokens": 1,
            "model.force_flash_attn": False,
            "rl.model_compile": "none",
        },
    )
    config_path = tmp_path / "tiny.yaml"
    cfg.to_file(config_path)
    args = Namespace(
        config=config_path,
        device="cpu",
        seat_rows=4,
        steps=2,
        warmup=0,
        seed=73,
        with_env=with_env,
    )
    arms = []
    for variant in ("baseline", "packing", "d2h", "telemetry"):
        result = bench.benchmark_arm(args, variant)
        arms.append(result)
        assert result["variant"] == variant
        assert result["packing_effective"] is False
        assert result["d2h_effective"] is False
        assert result["telemetry_effective"] is (with_env and variant == "telemetry")
        assert result["measured_steps"] == 2
        assert len(result["step_seconds"]) == 2
        assert result["cold_first_step_seconds"] > 0
        assert all(math.isfinite(t) and t > 0 for t in result["step_seconds"])
        assert result["seat_rows_per_second"] == 4 / result["mean_step_seconds"]
        if with_env:
            assert result["game_steps_per_second"] == 2 / result["mean_step_seconds"]
        else:
            assert result["game_steps_per_second"] is None
        assert result["resolved_config"]["env"]["n_envs"] == 2
        assert result["resolved_config"]["env"]["seed"] == 73
        assert result["resolved_config"]["env"]["pin_memory"] is False
        assert result["resolved_config"]["rl"]["model_compile"] == "none"
        assert result["hashed_steps"] == 3
        assert result["cuda_rng_sha256"] is None
    parity = bench._parity_summary(arms)
    assert parity["reference_variant"] == "baseline"
    assert parity["comparison_count"] == 3
    assert all(result["all_exact"] for result in parity["comparisons"])


@pytest.mark.parametrize(
    "field",
    ["action_sha256", "final_observation_sha256", "cpu_rng_sha256", "cuda_rng_sha256"],
)
def test_parity_summary_detects_each_independent_divergence(field: str) -> None:
    reference = {
        "variant": "baseline",
        "action_sha256": "actions",
        "final_observation_sha256": "observation",
        "cpu_rng_sha256": "cpu-rng",
        "cuda_rng_sha256": "cuda-rng",
    }
    altered = reference | {"variant": "actor", field: "different"}
    # Baseline remains the comparison reference even if the user lists it last.
    result = bench._parity_summary([altered, reference])
    assert result["reference_variant"] == "baseline"
    assert result["comparisons"][0]["matches"][field] is False
    assert result["comparisons"][0]["all_exact"] is False
