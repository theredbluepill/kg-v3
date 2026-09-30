"""Aggregate per-game receipts into games-<policy>.jsonl and a markdown table."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

W = Path(__file__).resolve().parent
ANCHORS = ["cha22", "smaller_market_shock", "v43", "starter"]


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def p99(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(0.99 * (len(ordered) - 1) + 0.5))]


def load(policy: str) -> list[dict]:
    rows = []
    for path in sorted((W / "games" / policy / "receipts").glob("*.json")):
        r = json.loads(path.read_text())
        s = r["summary"]
        seat = r["agent_seats"][0]
        own, opp = s["final_banks"][seat], s["final_banks"][1 - seat]
        st = s["seats"][str(seat)]
        anchor = r["label"].split("-")[1]
        rows.append(
            {
                "policy": policy,
                "label": r["label"],
                "anchor": anchor,
                "seed": r["seed"],
                "seat": seat,
                "own_bank": own,
                "anchor_bank": opp,
                "margin": own - opp,
                "result": "win" if own > opp else "loss" if own < opp else "draw",
                "final_rewards": s["final_rewards"],
                "final_statuses": s["final_statuses"],
                "bad_status_count": s["bad_status_count"],
                "bad_statuses": s["bad_statuses"][:5],
                "calls": st["calls"],
                "expected_calls": r["expected_calls_per_seat"],
                "exceptions": st["exceptions"],
                "invalid_raw_actions": st["invalid_raw_actions"],
                "pass_fallbacks": st["default_pass_returns"],
                "turn0_s": st["turn0_duration_s"],
                "steady_s": st["steady_duration_s"],
                "min_remaining_overage_s": st["min_remaining_overage_s"],
                "qualified": r["qualified"],
                "wall_s": r["wall_s"],
                "replay_sha256": r["replay"]["sha256"],
                "manifest_sha256": r["manifest_sha256"],
                "runtime": r["runtime"],
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    w = sum(r["result"] == "win" for r in rows)
    l = sum(r["result"] == "loss" for r in rows)
    d = n - w - l
    lo, hi = wilson(w, n)
    mean = lambda k: sum(r[k] for r in rows) / n if n else math.nan  # noqa: E731
    seat = {}
    for s in (0, 1):
        sr = [r for r in rows if r["seat"] == s]
        seat[s] = (sum(r["result"] == "win" for r in sr), len(sr),
                   sum(r["margin"] for r in sr) / len(sr) if sr else math.nan)
    # p99 over this anchor's pooled steady turns is approximated by the max of
    # per-game p99 (conservative); the pooled max is reported too.
    return {
        "n": n, "w": w, "l": l, "d": d, "wr": w / n if n else math.nan, "lo": lo, "hi": hi,
        "own": mean("own_bank"), "opp": mean("anchor_bank"), "margin": mean("margin"),
        "seat": seat,
        "bad": sum(r["bad_status_count"] for r in rows),
        "exc": sum(r["exceptions"] for r in rows),
        "inv": sum(r["invalid_raw_actions"] for r in rows),
        "fb": sum(r["pass_fallbacks"] for r in rows),
        "short": sum(r["calls"] != r["expected_calls"] for r in rows),
        "p99_max": max((r["steady_s"]["p99"] for r in rows), default=math.nan),
        "p99_med": sorted(r["steady_s"]["p99"] for r in rows)[n // 2] if n else math.nan,
        "turn_max": max((r["steady_s"]["max"] for r in rows), default=math.nan),
        "turn0_max": max((r["turn0_s"] for r in rows), default=math.nan),
    }


def table(policy: str, summaries: dict[str, dict]) -> str:
    out = [
        f"**{policy}**\n",
        "| anchor | games | W-L-D | win rate [95% Wilson] | mean own bank | mean anchor bank | mean margin | seat0 W/n (margin) | seat1 W/n (margin) | errors/timeouts (bad statuses) | exceptions | invalid | PASS fallbacks | turn p99 s (max game / median game) | max turn s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for a, s in summaries.items():
        s0, s1 = s["seat"][0], s["seat"][1]
        out.append(
            f"| {a} | {s['n']} | {s['w']}-{s['l']}-{s['d']} | {s['wr']:.2f} [{s['lo']:.2f}, {s['hi']:.2f}] | "
            f"{s['own']:,.0f} | {s['opp']:,.0f} | {s['margin']:+,.0f} | {s0[0]}/{s0[1]} ({s0[2]:+,.0f}) | "
            f"{s1[0]}/{s1[1]} ({s1[2]:+,.0f}) | {s['bad']} | {s['exc']} | {s['inv']} | {s['fb']} | "
            f"{s['p99_max']:.3f} / {s['p99_med']:.3f} | {s['turn_max']:.3f} |"
        )
    return "\n".join(out)


def main() -> None:
    out = {}
    for policy in ("candidate", "bc"):
        rows = load(policy)
        if not rows:
            continue
        with (W / f"games-{policy}.jsonl").open("w") as handle:
            for r in rows:
                handle.write(json.dumps(r, sort_keys=True) + "\n")
        by = {a: summarize([r for r in rows if r["anchor"] == a]) for a in ANCHORS}
        by = {a: s for a, s in by.items() if s["n"]}
        by["ALL"] = summarize(rows)
        out[policy] = by
        print(table(policy, by))
        print()
    if "candidate" in out and "bc" in out:
        print("**candidate minus BC (same seeds and seats)**\n")
        print("| anchor | paired games | win-rate diff | mean own-bank diff | mean margin diff | paired margin diff (per game, mean) |")
        print("|---|---|---|---|---|---|")
        cand = {r["label"].split("-", 1)[1]: r for r in load("candidate")}
        bc = {r["label"].split("-", 1)[1]: r for r in load("bc")}
        for a in [*ANCHORS, "ALL"]:
            keys = [k for k in cand if k in bc and (a == "ALL" or cand[k]["anchor"] == a)]
            if not keys:
                continue
            c, b = out["candidate"].get(a), out["bc"].get(a)
            paired = sum(cand[k]["margin"] - bc[k]["margin"] for k in keys) / len(keys)
            print(f"| {a} | {len(keys)} | {c['wr'] - b['wr']:+.2f} | {c['own'] - b['own']:+,.0f} | "
                  f"{c['margin'] - b['margin']:+,.0f} | {paired:+,.0f} |")
    (W / "aggregate.json").write_text(json.dumps(out, indent=1, default=str) + "\n")


if __name__ == "__main__":
    sys.exit(main())
