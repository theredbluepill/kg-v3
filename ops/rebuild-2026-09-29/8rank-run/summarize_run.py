"""Summarize a Kaggriculture PPO run from its W&B history (plan 6.3b, DRAFT).

    uv run --no-sync python ops/rebuild-2026-09-29/8rank-run/summarize_run.py RUN_DIR \
        --out RECEIPT_DIR [--skip-first 1]

RUN_DIR is the timestamped directory run_ppo created (it holds attempts.jsonl).
Writes ``iterations.tsv`` (one row per training iteration) and
``summary.json`` to ``--out``:

- complete iterations, and whether every iteration advanced
  ``optimizer/steps`` by 16 and ``train/env_steps`` by 16,384;
- complete-work SPS over complete iterations after ``--skip-first``
  (compile warm-up): global env steps and seat rows (2 per env step, both
  self-play seats) per wall second, from ``train/env_steps`` and the rows'
  ``_timestamp``. Seat rows are not valid learner actions: ``run_ppo`` logs no
  valid-action count, so plan 6.3b's "learner turns (valid learner-seat
  actions)" figure is not available from this summary;
- mean phase seconds (rollout, teacher, update, iteration) and the mean of
  ``perf/steps_per_second``;
- ``teacher/cache_bytes``, the evaluation rows (win rate, promoted) and the
  game-interval banks (``train/own_bank_mean`` where ``train/bank_games > 0``);
- the count of nonfinite logged values.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from watchdog import is_nonfinite, wandb_run_path  # noqa: E402

COLUMNS = (
    "_step",
    "_timestamp",
    "train/env_steps",
    "optimizer/steps",
    "time/rollout_seconds",
    "time/teacher_seconds",
    "time/update_seconds",
    "time/iteration_seconds",
    "perf/steps_per_second",
    "teacher/cache_bytes",
    "teacher/kl",
    "train/explained_variance",
    "policy/approx_kl",
    "optimizer/grad_norm",
    "train/bank_games",
    "train/own_bank_mean",
)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def main() -> int:
    import wandb

    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-first", type=int, default=1)
    args = parser.parse_args()
    run_path = wandb_run_path(args.run_dir)
    rows = list(wandb.Api(timeout=60).run(run_path).scan_history(page_size=1000))
    nonfinite = sum(
        1
        for row in rows
        for key, value in row.items()
        if not key.startswith("_") and is_nonfinite(value)
    )
    train = [r for r in rows if r.get("train/env_steps") is not None]
    train.sort(key=lambda r: r["train/env_steps"])
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "iterations.tsv").open("w") as out:
        out.write("\t".join(COLUMNS) + "\n")
        for row in train:
            out.write("\t".join(str(row.get(c, "")) for c in COLUMNS) + "\n")
    env_deltas = [
        b["train/env_steps"] - a["train/env_steps"]
        for a, b in zip(train, train[1:], strict=False)
    ]
    opt_deltas = [
        b["optimizer/steps"] - a["optimizer/steps"]
        for a, b in zip(train, train[1:], strict=False)
        if a.get("optimizer/steps") is not None and b.get("optimizer/steps") is not None
    ]
    window = train[args.skip_first - 1 :] if args.skip_first > 0 else train
    sps: dict[str, Any] = {}
    if len(window) >= 2:
        steps = window[-1]["train/env_steps"] - window[0]["train/env_steps"]
        seconds = window[-1]["_timestamp"] - window[0]["_timestamp"]
        sps = {
            "iterations": len(window) - 1,
            "env_steps": steps,
            "wall_seconds": seconds,
            "global_game_sps": steps / seconds,
            "seat_rows_per_second": 2 * steps / seconds,
        }
    body = train[args.skip_first :]
    summary = {
        "run": run_path,
        "iterations": len(train),
        "env_step_deltas": sorted(set(env_deltas)),
        "optimizer_step_deltas": sorted(set(opt_deltas)),
        "complete_work_after_skip": sps,
        "mean_seconds": {
            phase: _mean(
                [
                    r[f"time/{phase}_seconds"]
                    for r in body
                    if r.get(f"time/{phase}_seconds") is not None
                ]
            )
            for phase in ("rollout", "teacher", "update", "iteration")
        },
        "mean_perf_steps_per_second": _mean(
            [
                r["perf/steps_per_second"]
                for r in body
                if r.get("perf/steps_per_second") is not None
            ]
        ),
        "teacher_cache_bytes": sorted(
            {
                r["teacher/cache_bytes"]
                for r in train
                if r.get("teacher/cache_bytes") is not None
            }
        ),
        "game_interval_banks": [
            (r["_step"], r["train/own_bank_mean"])
            for r in train
            if (r.get("train/bank_games") or 0) > 0
            and r.get("train/own_bank_mean") is not None
        ],
        "evaluations": [
            {
                k: r.get(k)
                for k in (
                    "_step",
                    "eval/win_rate_against_last_best",
                    "eval/promoted",
                    "eval/games",
                    "time/eval_seconds",
                )
            }
            for r in rows
            if r.get("eval/promoted") is not None
        ],
        "nonfinite_values": nonfinite,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
