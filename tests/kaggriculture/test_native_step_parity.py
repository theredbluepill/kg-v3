"""Source-bound pre-optimization oracle plus ordering/transaction regressions."""

from __future__ import annotations

import json
import random

import numpy as np
import pytest

from .native_step_oracle import (
    BASE_COMMIT,
    GOLDEN,
    SEED,
    STRIDE,
    buffers,
    diagnostics,
    exact_bytes,
    make_env,
    random_actions,
    trajectory,
)


@pytest.mark.parametrize("threads", [1, 4, 8])
def test_two_complete_games_match_pristine_step_golden(threads: int) -> None:
    golden = json.loads(GOLDEN.read_text())
    assert golden["base_commit"] == BASE_COMMIT
    assert trajectory(threads) == golden["trajectory"]


@pytest.mark.parametrize("threads", [1, 4, 8])
def test_malformed_peer_preserves_every_output_and_rng(threads: int) -> None:
    env = make_env(4, threads)
    control = make_env(4, threads)
    out, expected = buffers(4), buffers(4)
    env.observe(**out)
    control.observe(**expected)
    rngs = [random.Random(8128 + i) for i in range(4)]
    for _ in range(8):
        tokens, lengths = random_actions(out, rngs)
        before = exact_bytes(out), diagnostics(env, 4)
        malformed = tokens.copy()
        malformed[-1, -1, 0, 1] = 2**62
        with pytest.raises(ValueError, match=r"env=3.*seat=1"):
            env.step(malformed, lengths, **out)
        assert (exact_bytes(out), diagnostics(env, 4)) == before
        assert exact_bytes(env.step(tokens, lengths, **out)) == exact_bytes(
            control.step(tokens, lengths, **expected)
        )
        assert exact_bytes(out) == exact_bytes(expected)
        assert diagnostics(env, 4) == diagnostics(control, 4)


def test_batch_rows_equal_reverse_order_independent_envs() -> None:
    # Disjoint single-env seed streams reproduce the batched global stream at
    # simultaneous autoresets. Run singles in reverse order on every step.
    n = 4
    batch = make_env(n, 8)
    out = buffers(n)
    batch.observe(**out)
    singles = [
        make_env(1, 1, seed=SEED + i * STRIDE, stride=n * STRIDE) for i in range(n)
    ]
    rows = [buffers(1) for _ in range(n)]
    for env, row in zip(singles, rows, strict=True):
        env.observe(**row)
    rngs = [random.Random(8128 + i) for i in range(n)]
    for _ in range(722):
        tokens, lengths = random_actions(out, rngs)
        metrics = batch.step(tokens, lengths, **out)
        singles_metrics = [{} for _ in range(n)]
        for i in reversed(range(n)):
            singles_metrics[i] = singles[i].step(
                tokens[i : i + 1], lengths[i : i + 1], **rows[i]
            )
            for name, value in out.items():
                assert value[i : i + 1].tobytes() == rows[i][name].tobytes(), name
            assert batch.state_snapshot(i) == singles[i].state_snapshot(0)
            assert exact_bytes(batch.terminal_metrics(i)) == exact_bytes(
                singles[i].terminal_metrics(0)
            )
            assert batch.seed_state()[1][i] == singles[i].seed_state()[1][0]
        for name, value in metrics.items():
            assert exact_bytes(value) == exact_bytes(
                [entry for per_env in singles_metrics for entry in per_env[name]]
            )
    assert np.asarray(out["globals_int"])[0, 0, 0] == 3
