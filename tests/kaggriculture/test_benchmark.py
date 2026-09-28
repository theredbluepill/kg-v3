from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts import benchmark_kaggriculture as benchmark


@pytest.mark.parametrize("flash_available", [False, True])
def test_runtime_identity_records_native_bytes_and_dispatch_scope(
    monkeypatch: pytest.MonkeyPatch, flash_available: bool
) -> None:
    monkeypatch.setattr(benchmark, "flash_attn_available", lambda: flash_available)
    monkeypatch.setattr(benchmark.shutil, "which", lambda _: None)
    identity = benchmark._runtime_identity(dtype="bfloat16")
    native_path = Path(identity["native_extension"]["path"])
    assert native_path.is_file()
    assert (
        identity["native_extension"]["sha256"]
        == hashlib.sha256(native_path.read_bytes()).hexdigest()
    )
    attention = identity["attention"]
    assert attention["flash_attn_package_available"] is flash_available
    assert attention["expected_trunk_api"] == (
        "flash_attn_varlen_func"
        if flash_available
        else "torch.nn.functional.scaled_dot_product_attention"
    )
    assert "timeline verification" in attention["kernel_qualification"]
    assert identity["nsys"] == {"path": None, "version": None}
    fp32 = benchmark._runtime_identity(dtype="float32")
    assert fp32["attention"]["expected_trunk_api"].endswith(
        "scaled_dot_product_attention"
    )


def test_runtime_identity_records_installed_profiler_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(benchmark.shutil, "which", lambda _: "/opt/nsys")

    def version(command: list[str], *, text: bool) -> str:
        assert command == ["/opt/nsys", "--version"]
        assert text
        return "NVIDIA Nsight Systems test-version\n"

    monkeypatch.setattr(benchmark.subprocess, "check_output", version)
    identity = benchmark._runtime_identity(dtype="bfloat16")
    assert identity["nsys"] == {
        "path": "/opt/nsys",
        "version": "NVIDIA Nsight Systems test-version",
    }


def test_optimization_work_reports_actual_steps_and_early_stopped_updates() -> None:
    updates = [
        {"optimizer/steps": steps, "policy/target_kl_exceeded": stopped}
        for steps, stopped in [(4.0, 0.0), (5.0, 1.0), (9.0, 0.0)]
    ]
    work = benchmark._optimization_work(updates, warmup=1)
    assert work["optimizer_steps_per_rank"] == 5
    assert work["target_kl_stopped_updates"] == 1
    assert (
        benchmark._optimization_work(updates, warmup=0)["optimizer_steps_per_rank"] == 9
    )


def test_observation_work_distinguishes_counters_and_seat_denominators() -> None:
    updates = [
        {
            "train/env_steps": game_steps,
            "train/player_step_total": seat_turns,
            "train/total_active_entities": actors,
            "train/action_frames": frames,
            "train/max_entities": max_actors,
        }
        for game_steps, seat_turns, actors, frames, max_actors in [
            (100.0, 200.0, 250.0, 800.0, 12.0),
            (200.0, 350.0, 650.0, 650.0, 4.0),
            (300.0, 500.0, 1250.0, 700.0, 8.0),
        ]
    ]
    work = benchmark._observation_work(updates, warmup=1)
    assert work["observed_seat_observations"] == 400
    assert work["observed_own_actor_occurrences"] == 1000
    assert work["mean_own_actors_per_seat_observation"] == 2.5
    assert work["max_own_actors_per_seat_observation"] == 8
    assert work["action_frames"] == 1350
    assert work["mean_action_frames_per_learner_seat_turn"] == 4.5
    assert "including any inactive seats" in work["observation_work_note"]
    assert benchmark._observation_work(updates, warmup=0)["action_frames"] == 2150
    updates[-1]["train/env_steps"] = 100
    with pytest.raises(ValueError, match="positive game and learner-seat counts"):
        benchmark._observation_work(updates, warmup=1)
