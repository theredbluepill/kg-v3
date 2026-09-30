"""Summarise the first iterations of a main_probe(_auto) run log.

Usage: python3 first_iters.py LOG [N_RANKS=8] [N_ENVS=12] [HORIZON=720] [MAX_ITERS=10]

Prints the numa_bind record of every rank, the W&B run, and per rank-0
iteration: ranks reporting, train/bank_games (expected N_RANKS * N_ENVS when
HORIZON is the whole 720-step game: every env ends one game per iteration),
the optimizer-step delta (expected N_ENVS: one segment per rank per step with
rl.segments_per_minibatch=1), env steps/s, wall seconds and own bank. It ends
with OK/CHECK lines. Stdlib only; works on the Mac copy of the log.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict

TAG = "[nt-probe] "


def records(text: str) -> list:
    dec = json.JSONDecoder()
    out = []
    i = text.find(TAG)
    while i >= 0:
        try:
            obj, _ = dec.raw_decode(text, i + len(TAG))
            if isinstance(obj, dict):
                out.append(obj)
        except json.JSONDecodeError:
            pass
        i = text.find(TAG, i + 1)
    return out


def main(argv: list) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    path = argv[1]
    n_ranks = int(argv[2]) if len(argv) > 2 else 8
    n_envs = int(argv[3]) if len(argv) > 3 else 12
    horizon = int(argv[4]) if len(argv) > 4 else 720
    max_iters = int(argv[5]) if len(argv) > 5 else 10
    with open(path, errors="replace") as handle:
        recs = records(handle.read())
    problems = []
    binds = sorted((r for r in recs if r.get("kind") == "numa_bind"), key=lambda r: r["rank"])
    for r in binds:
        print(
            f"numa_bind rank={r['rank']} local_rank={r.get('local_rank')} gpu={r.get('gpu_index')} "
            f"bus={r.get('bus_id')} node={r.get('node')} cpus={r.get('cpus')} "
            f"ranks_on_node={r.get('ranks_on_node')} oversubscribed={r.get('oversubscribed')}"
        )
        if r.get("oversubscribed"):
            problems.append(f"rank {r['rank']} oversubscribed")
    if len(binds) != n_ranks:
        problems.append(f"{len(binds)} numa_bind records, expected {n_ranks} (KG_NT_NUMA=off gives 0)")
    for r in recs:
        if r.get("kind") == "wandb":
            print(f"wandb url={r.get('url')} id={r.get('id')} group={r.get('group')}")
    starts = {r["rank"] for r in recs if r.get("kind") == "start"}
    if len(starts) != n_ranks:
        problems.append(f"start records from {len(starts)} ranks, expected {n_ranks}")
    its: dict = defaultdict(dict)
    for r in recs:
        if r.get("kind") == "iteration":
            its[r["iteration"]][r["rank"]] = r
    expected_games = n_ranks * n_envs if horizon == 720 else None
    prev_steps = None
    shown = 0
    for n in sorted(its):
        if 0 not in its[n]:
            continue
        m = its[n][0]["metrics"]
        steps = m.get("optimizer/steps")
        delta = None if prev_steps is None or steps is None else steps - prev_steps
        prev_steps = steps
        games = m.get("train/bank_games")
        bank = m.get("train/own_bank_mean")
        sps = m.get("perf/steps_per_second")
        nonfinite = [
            f"{k}@rank{rk}"
            for rk, rec in its[n].items()
            for k, v in rec["metrics"].items()
            if k.startswith("loss/") and isinstance(v, (int, float)) and not math.isfinite(v)
        ]
        print(
            f"iter {n:3d} ranks={len(its[n])}/{n_ranks} bank_games={games} "
            f"opt_steps={steps} (+{delta}) sps={sps and round(sps)} "
            f"wall={its[n][0]['wall_seconds']:.1f}s own_bank_mean={bank and round(bank)}"
            + (f" NONFINITE={nonfinite}" if nonfinite else "")
        )
        if nonfinite:
            problems.append(f"iteration {n}: nonfinite {nonfinite}")
        if len(its[n]) != n_ranks and n != max(its):
            problems.append(f"iteration {n}: {len(its[n])} ranks reported")
        if expected_games is not None and games is not None and games != expected_games:
            problems.append(f"iteration {n}: bank_games {games} != {expected_games}")
        if delta is not None and delta != n_envs:
            problems.append(f"iteration {n}: optimizer step delta {delta} != {n_envs}")
        shown += 1
        if shown >= max_iters:
            break
    if shown == 0:
        problems.append("no rank-0 iteration record yet")
    print(f"env steps per iteration = {horizon} x {n_envs} x {n_ranks} = {horizon * n_envs * n_ranks:,}")
    for p in problems:
        print("CHECK", p)
    if not problems:
        print("OK first iterations match the expected shape")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
