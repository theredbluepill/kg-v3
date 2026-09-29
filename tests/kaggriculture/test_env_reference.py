"""Exact complete-game TrainingBatch oracle, with no missing-fixture skip."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from owl import rs

from .test_native_env import buffers

ROOT = Path(__file__).resolve().parents[2]


def _recorder() -> Any:
    path = ROOT / "scripts/record_kaggriculture_env_reference.py"
    assert path.is_file(), "Task 1.4 reference recorder is missing"
    spec = importlib.util.spec_from_file_location("env_reference_recorder_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _math_reward(
    before: np.ndarray,
    after: np.ndarray,
    banks: np.ndarray,
    done: bool,
    cfg: dict[str, Any],
) -> np.ndarray:
    # Independent scalar mathematical formula: only the final result is f32.
    def penalty(c: np.ndarray) -> float:
        death = (
            min(
                cfg["econ_cap"],
                cfg["econ_shaping"]
                * (
                    cfg["econ_starvation_weight"] * int(c[0])
                    + cfg["econ_drought_weight"] * int(c[1])
                ),
            )
            if cfg["econ_shaping"]
            else 0.0
        )
        ineffective = min(
            cfg["econ_ineffective_cap"], cfg["econ_ineffective_weight"] * int(c[2])
        )
        return death + ineffective

    delta = [penalty(after[s]) - penalty(before[s]) for s in range(2)]
    scale = (
        1.0
        - (cfg["econ_cap"] if cfg["econ_shaping"] else 0.0)
        - (cfg["econ_ineffective_cap"] if cfg["econ_ineffective_weight"] else 0.0)
    )
    return np.asarray(
        [
            delta[1 - s]
            - delta[s]
            + (
                scale
                * (
                    (float(banks[s]) > float(banks[1 - s]))
                    - (float(banks[s]) < float(banks[1 - s]))
                )
                if done
                else 0.0
            )
            for s in range(2)
        ],
        dtype=np.float32,
    )


def _same(
    field: str,
    actual: np.ndarray,
    expected: np.ndarray,
    game: int,
    step: int,
    tokens: np.ndarray,
    lengths: np.ndarray,
    actors: np.ndarray,
    limits: np.ndarray,
) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    # Equality is bitwise, including signed zero and all float64 bank mantissas.
    got = (
        actual.view(np.uint32 if actual.dtype == np.float32 else np.uint64)
        if actual.dtype.kind == "f"
        else actual
    )
    want = expected.view(got.dtype)
    if np.array_equal(got, want):
        return
    first = tuple(int(i) for i in np.argwhere(got != want)[0])
    seat = first[0]
    action = rs.kaggriculture_decode(
        tokens[0, seat],
        int(lengths[0, seat]),
        int(actors[seat]),
        int(limits[seat]),
        241,
    )
    raise AssertionError(
        f"first divergence game={game} seed={17000 + game} "
        f"step={step} seat={seat} action={action} "
        f"field={field}{first} actual={actual[first]} "
        f"expected={expected[first]}"
    )


def test_native_matches_training_batch_16_complete_games() -> None:
    recorder = _recorder()
    manifest, fixture = recorder.load_fixture()
    assert manifest["games"] == 16
    assert manifest["steps_per_game"] == 719
    assert fixture["lengths"].shape == (16, 719, 2)
    compared = 0
    for game in range(16):
        reward = recorder.reward_config(game)
        env = rs.KaggricultureEnv(1, 17000 + game, 1, "{}", reward, 1, hire_limit=241)
        out = buffers(1)
        env.observe(**out)
        for step in range(719):
            assert int(fixture["transition_indices"][game, step]) == step
            tokens = np.zeros((1, 2, 252, 12), dtype=np.int64)
            lengths = fixture["lengths"][game, step].reshape(1, 2).copy()
            for seat in range(2):
                index = (game * 719 + step) * 2 + seat
                start, end = fixture["program_offsets"][index : index + 2]
                assert end - start == lengths[0, seat]
                tokens[0, seat, : lengths[0, seat]] = fixture["tokens"][start:end]
            actors = out["globals_int"][0, :, 14].copy()
            limits = out["order_limits"][0].copy()
            metrics = env.step(tokens, lengths, **out)
            for published, recorded in [
                ("rewards", "rewards"),
                ("dones", "dones"),
                ("transition_banks_before", "banks_before"),
                ("transition_banks_after", "banks_after"),
                ("transition_econ_before", "econ_before"),
                ("transition_econ_after", "econ_after"),
            ]:
                _same(
                    published,
                    out[published][0],
                    fixture[recorded][game, step],
                    game,
                    step,
                    tokens,
                    lengths,
                    actors,
                    limits,
                )
            # For this recipe terminal scales are .65/.75 and a per-step
            # economic term is bounded by .35/.25. A non-draw terminal result
            # has magnitude >=.30/.50, so rounding the economic term first
            # contributes <= half an output ULP; final rounding adds <= half.
            # Draws/nonterminal steps have only one rounding in either formula.
            mathematical = _math_reward(
                out["transition_econ_before"][0],
                out["transition_econ_after"][0],
                out["transition_banks_after"][0],
                step == 718,
                reward,
            )
            ulp = np.maximum(
                np.abs(
                    np.nextafter(
                        mathematical, np.float32(np.inf), dtype=np.float32
                    ).astype(np.float64)
                    - mathematical
                ),
                np.abs(
                    mathematical.astype(np.float64)
                    - np.nextafter(mathematical, np.float32(-np.inf), dtype=np.float32)
                ),
            )
            assert np.all(
                np.abs(out["rewards"][0].astype(np.float64) - mathematical) <= ulp
            ), (game, step, "mathematical reward")
            assert set(metrics) == {
                "total_games_played",
                "terminal_bank_0",
                "terminal_bank_1",
                "terminal_margin_0",
            }
            if step < 718:
                assert all(value == [] for value in metrics.values())
                assert env.seed_state() == (17001 + game, (17000 + game,))
                assert env.terminal_metrics(0) is None
            else:
                record = env.terminal_metrics(0)
                assert record is not None
                assert (
                    record["episode_steps"]
                    == int(fixture["terminal_steps"][game])
                    == 719
                )
                assert record["winner"] == int(fixture["terminal_winner"][game])
                expected_banks = fixture["terminal_banks"][game]
                actual_banks = np.asarray(
                    [record["bank_0"], record["bank_1"]], dtype=np.float64
                )
                _same(
                    "terminal_banks",
                    actual_banks,
                    expected_banks,
                    game,
                    step,
                    tokens,
                    lengths,
                    actors,
                    limits,
                )
                assert (
                    np.float64(record["margin_0"]).tobytes()
                    == np.float64(expected_banks[0] - expected_banks[1]).tobytes()
                )
                _same(
                    "terminal_econ",
                    np.stack([record["econ_0"], record["econ_1"]]),
                    fixture["terminal_econ"][game],
                    game,
                    step,
                    tokens,
                    lengths,
                    actors,
                    limits,
                )
                assert env.seed_state() == (
                    int(fixture["next_seed"][game]),
                    (int(fixture["autoreset_seed"][game]),),
                )
                assert metrics == {
                    "total_games_played": [1.0],
                    "terminal_bank_0": [record["bank_0"]],
                    "terminal_bank_1": [record["bank_1"]],
                    "terminal_margin_0": [record["margin_0"]],
                }
                assert not out["globals_int"][0, :, 0].any()
                assert out["still_playing"].all()
                assert json.loads(env.state_snapshot(0))["public"]["step"] == 0
            compared += 1
        del env  # One live logical game throughout all sixteen trajectories.
    assert compared == 16 * 719 == 11504
