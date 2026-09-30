"""Deterministic legal actions and bytewise native-step trajectory receipts.

The golden is recorded from an independently built pristine source export. The
encoder validates every generated action; sampling and hashing are outside the
microbenchmark's timed native ``step`` calls.
"""

from __future__ import annotations

import hashlib
import json
import random
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from owl import rs
from owl.kaggriculture.types import (
    _SCHEMA,
    ANIMALS,
    CROPS,
    ITEMS,
    MARKET_KINDS,
    PRODUCTS,
    UNIT_KINDS,
)

Array = NDArray[Any]
GOLDEN = Path(__file__).resolve().parents[2] / "ops/sps-2026-10-01/native-golden.json"
BASE_COMMIT = "b2276bc5b70073b58a27f9e5fbd52473dbd48569"
SEED = 941003
STRIDE = 7
PARITY_ENVS = 4
PARITY_STEPS = 1440
REWARD: rs.KaggricultureRewardDict = {
    "reward_mode": "win_loss",
    "econ_shaping": 0.0,
    "econ_starvation_weight": 4.0,
    "econ_drought_weight": 1.0,
    "econ_cap": 0.25,
    "econ_ineffective_weight": 0.0,
    "econ_ineffective_cap": 0.1,
    "econ_bank_weight": 0.25,
    "econ_bank_scale": 150000.0,
    "econ_bank_cap": 0.25,
    "econ_margin_weight": 0.25,
    "econ_margin_scale": 100000.0,
    "econ_margin_cap": 0.25,
}


def buffers(n_envs: int) -> dict[str, Array]:
    dtypes = {
        torch.int64: np.int64,
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.bool: np.bool_,
    }
    result = {
        name: np.zeros((n_envs, 2, *shape), dtype=dtypes[dtype])
        for name, (dtype, shape, _, _) in _SCHEMA.items()
    }
    result.update(
        can_act=np.zeros((n_envs, 2, 252), dtype=np.bool_),
        rewards=np.zeros((n_envs, 2), dtype=np.float32),
        dones=np.zeros((n_envs, 2), dtype=np.bool_),
        transition_banks_before=np.zeros((n_envs, 2), dtype=np.float64),
        transition_banks_after=np.zeros((n_envs, 2), dtype=np.float64),
        transition_econ_before=np.zeros((n_envs, 2, 32), dtype=np.int64),
        transition_econ_after=np.zeros((n_envs, 2, 32), dtype=np.int64),
    )
    assert len(result) == 35  # 29 observation arrays plus six transition arrays.
    return result


def make_env(
    n_envs: int, threads: int, *, seed: int = SEED, stride: int = STRIDE
) -> rs.KaggricultureEnv:
    return rs.KaggricultureEnv(
        n_envs, seed, stride, "{}", REWARD, threads, hire_limit=241
    )


def _unit(rng: random.Random) -> list[str | int]:
    kind = rng.choice(UNIT_KINDS[1:])
    command: list[str | int] = [kind]
    if kind in ("PICKUP", "PLACE"):
        command.append(rng.choice(ITEMS))
        if rng.randrange(2):
            command.append(rng.choice((1, 2, 31, 32, 33, 1023)))
    elif kind == "PLANT":
        command.append(rng.choice(CROPS[1:]))
    return command


def random_actions(
    arrays: dict[str, Array], rngs: list[random.Random]
) -> tuple[Array, Array]:
    """Canonical grammar-legal actions, including EMPTY, quantities and HIRE.

    Economic success is deliberately not required by the game's grammar. Each
    environment has its own PRNG so processing order cannot change its actions.
    """
    tokens = np.zeros((len(rngs), 2, 252, 12), dtype=np.int64)
    lengths = np.empty((len(rngs), 2), dtype=np.int64)
    for env, rng in enumerate(rngs):
        for seat in range(2):
            actors = int(arrays["actor_mask"][env, seat, :241].sum())
            order_limit = int(arrays["order_limits"][env, seat])
            units = [_unit(rng) for _ in range(actors)]
            market: list[list[str | int]] = []
            remaining_hires = 241 - actors
            for _ in range(rng.randrange(order_limit + 1)):
                kind = rng.choice(MARKET_KINDS[1:])
                if kind == "HIRE" and remaining_hires == 0:
                    kind = "EMPTY"
                command: list[str | int] = [] if kind == "EMPTY" else [kind]
                if kind == "HIRE":
                    remaining_hires -= 1
                elif kind in ("BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"):
                    items = (
                        CROPS[1:]
                        if kind == "BUY_SEED"
                        else ANIMALS[1:]
                        if kind == "BUY_ANIMAL"
                        else ("WHEAT", "FERTILIZER")
                        if kind == "BUY_PRODUCT"
                        else PRODUCTS
                    )
                    command.extend(
                        (rng.choice(items), rng.choice((0, 1, 2, 31, 32, 33, 1023)))
                    )
                market.append(command)
            action = {"farmer": units[0], "hands": units[1:], "market": market}
            lengths[env, seat] = rs.kaggriculture_encode(
                json.dumps(action, separators=(",", ":")),
                actors,
                order_limit,
                241,
                tokens[env, seat],
            )
    return tokens, lengths


def exact_bytes(value: Any) -> bytes:
    """Encode metrics/diagnostics with float bit patterns and structural tags."""
    if value is None:
        return b"n"
    if isinstance(value, np.ndarray):
        return b"a" + exact_bytes((value.dtype.str, value.shape, value.tobytes()))
    if isinstance(value, bool):
        return b"t" if value else b"f"
    if isinstance(value, int):
        return b"i" + str(value).encode() + b";"
    if isinstance(value, float):
        return b"d" + struct.pack("<d", value)
    if isinstance(value, str | bytes):
        raw = value.encode() if isinstance(value, str) else value
        return (
            (b"s" if isinstance(value, str) else b"b")
            + struct.pack("<Q", len(raw))
            + raw
        )
    if isinstance(value, dict):
        return (
            b"{"
            + b"".join(exact_bytes((key, value[key])) for key in sorted(value))
            + b"}"
        )
    if isinstance(value, list | tuple):
        return b"[" + b"".join(exact_bytes(item) for item in value) + b"]"
    raise TypeError(f"unsupported receipt value {type(value)}")


def diagnostics(env: rs.KaggricultureEnv, n_envs: int) -> bytes:
    return exact_bytes(
        (
            env.seed_state(),
            [env.state_snapshot(i) for i in range(n_envs)],
            [env.terminal_metrics(i) for i in range(n_envs)],
        )
    )


def trajectory(threads: int) -> dict[str, Any]:
    """Hash every publication and reset; the golden contains no model outputs."""
    n = PARITY_ENVS
    env = make_env(n, threads)
    arrays = buffers(n)
    env.observe(**arrays)
    rngs = [random.Random(8128 + i) for i in range(n)]
    hashes = {name: hashlib.sha256() for name in arrays}
    hashes.update(
        {
            name: hashlib.sha256()
            for name in ("actions", "diagnostics", "metrics", "events")
        }
    )
    terminal_steps: list[int] = []
    max_actors = 0

    def record(event: Any, metrics: Any = None) -> None:
        for name, value in arrays.items():
            hashes[name].update(value.tobytes())
        hashes["diagnostics"].update(diagnostics(env, n))
        hashes["metrics"].update(exact_bytes(metrics))
        hashes["events"].update(exact_bytes(event))

    record("initial_observe")
    for step in range(PARITY_STEPS):
        tokens, lengths = random_actions(arrays, rngs)
        hashes["actions"].update(tokens.tobytes())
        hashes["actions"].update(lengths.tobytes())
        metrics = env.step(tokens, lengths, **arrays)
        max_actors = max(max_actors, int(arrays["actor_mask"][..., :241].sum(-1).max()))
        if arrays["dones"].any():
            assert arrays["dones"].all()
            terminal_steps.append(step + 1)
        record(("step", step + 1), metrics)
    assert terminal_steps == [719, 1438]
    # Preserve the full-game denominator above; test truncations only afterwards.
    for mask in ([True, False, True, False], [False] * n, [False, True, False, True]):
        env.truncate_envs(np.asarray(mask, dtype=np.bool_), **arrays)
        record(("truncate", mask))
        tokens, lengths = random_actions(arrays, rngs)
        hashes["actions"].update(tokens.tobytes())
        hashes["actions"].update(lengths.tobytes())
        record("step_after_truncate", env.step(tokens, lengths, **arrays))
    env.reset(**arrays)
    record("explicit_reset")
    return {
        "n_envs": n,
        "steps_before_truncations": PARITY_STEPS,
        "terminal_steps": terminal_steps,
        "completed_games": n * len(terminal_steps),
        "max_actors": max_actors,
        "final_seed_state": [env.seed_state()[0], list(env.seed_state()[1])],
        "sha256": {name: value.hexdigest() for name, value in hashes.items()},
    }


def benchmark(repetitions: int = 5) -> dict[str, Any]:
    """Time only 720 native step calls, excluding sampler, allocation and hashes."""
    seconds = []
    final_hashes = []
    action_hashes = []
    max_actors = 0
    for _ in range(repetitions):
        env = make_env(20, 4)
        arrays = buffers(20)
        env.observe(**arrays)
        rngs = [random.Random(8128 + i) for i in range(20)]
        elapsed = 0
        action_hash = hashlib.sha256()
        for _ in range(720):
            tokens, lengths = random_actions(arrays, rngs)
            action_hash.update(tokens.tobytes())
            action_hash.update(lengths.tobytes())
            start = time.perf_counter_ns()
            env.step(tokens, lengths, **arrays)
            elapsed += time.perf_counter_ns() - start
            max_actors = max(
                max_actors, int(arrays["actor_mask"][..., :241].sum(-1).max())
            )
        seconds.append(elapsed / 1e9)
        action_hashes.append(action_hash.hexdigest())
        final_hashes.append(
            hashlib.sha256(exact_bytes(arrays) + diagnostics(env, 20)).hexdigest()
        )
    assert len(set(final_hashes)) == 1
    assert len(set(action_hashes)) == 1
    return {
        "n_envs": 20,
        "native_threads": 4,
        "steps": 720,
        "repetitions": repetitions,
        "max_actors": max_actors,
        "step_seconds": seconds,
        "median_step_seconds": float(np.median(seconds)),
        "median_env_steps_per_second": 14400 / float(np.median(seconds)),
        "final_sha256": final_hashes[0],
        "actions_sha256": action_hashes[0],
    }
