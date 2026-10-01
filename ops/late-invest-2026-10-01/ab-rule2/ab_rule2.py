"""Three-arm paired A/B of rule 2 (late-investment filter) on c50, fixed-shop engine.

OFF = games-fixedshop/c50 (pkg-c50, no rules), A = games-fixedshop/c50-ft-on
(pkg-c50-ft, 4c99768a, rule 1 on), B = games-fixedshop/c50-r12-on (pkg-c50-r12,
kg/submit-08bc 31c99619, rules 1+2 on, every block logged by run_game_logged.py).

Per game: B receipt health; the first replay step where B's actions differ from
A's; the rule-2 blocks (tap) and their stdout cross-check; whether the first
block's pre-filter market equals A's market at that step (play before the first
block is identical, so the model's order must be the same); whether B equals A
exactly when nothing was blocked; banks, gap and unsold goods at the end.
Margin changes are summarized with SE over the 8 seeds (seat pairs are often
identical games, so the seed is the independent unit).

Run: venv-kaggle python, PYTHONPATH=<pkg-c50-r12>:<kenv-fixedshop>
usage: python ab_rule2.py OUT.jsonl OUT_TABLES.md
"""

from __future__ import annotations

import collections
import gzip
import json
import statistics
import sys
from pathlib import Path

from owl.kaggriculture.late_invest import GameClock, blocked_reason

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ab import ANCHORS, SEEDS, health, load, unsold  # noqa: E402

W = Path(__file__).resolve().parents[1]
OFF = W / "games-fixedshop" / "c50"
A = W / "games-fixedshop" / "c50-ft-on"
B = W / "games-fixedshop" / "c50-r12-on"


def first_diff(sa: list, sb: list) -> int | None:
    for t in range(max(len(sa), len(sb))):
        if t >= len(sa) or t >= len(sb):
            return t
        if any(sa[t][s]["action"] != sb[t][s]["action"] for s in (0, 1)):
            return t
    return None


def category(reason: str) -> str:
    return reason.split(":")[0]


def main() -> None:
    out_jsonl, out_md = map(Path, sys.argv[1:3])
    rows = []
    for anchor in ANCHORS:
        for seed in SEEDS:
            for me in (0, 1):
                label = f"c50-{anchor}-s{seed}-seat{me}"
                r_off, _ = load(OFF, label)
                r_a, p_a = load(A, label)
                r_b, p_b = load(B, label)
                log = json.loads((B / "blocks" / f"{label}.json").read_text())
                s_a, s_b = p_a["steps"], p_b["steps"]
                clock = GameClock.of(p_b["configuration"])
                blocks = log["blocks"]
                reasons = [x for b in blocks for x in b["reasons"]]
                stdout_blocks = [
                    ln for ln in log["stdout_lines"] if "late-invest blocked" in ln["line"]
                ]
                stdout_reason_count = sum(
                    ln["line"].count("can sell by step") for ln in stdout_blocks
                )
                blocked_orders = [
                    {"step": b["step"], "day": clock.day(b["step"]),
                     "hour": b["step"] % clock.turns_per_day, "order": o,
                     "reason": blocked_reason(o, b["step"], clock)}
                    for b in blocks
                    for o, after in zip(b["market_before"], b["market_after"], strict=True)
                    if o != after
                ]
                fd = first_diff(s_a, s_b)
                first_block = blocks[0]["step"] if blocks else None
                # replay index t holds the action taken on observation t - 1
                first_block_matches_a = (
                    None if first_block is None
                    else s_a[first_block + 1][me]["action"]["market"] == blocks[0]["market_before"]
                )
                b_off = r_off["summary"]["final_banks"]
                b_a, b_b = r_a["summary"]["final_banks"], r_b["summary"]["final_banks"]
                gap = {arm: v[me] - v[1 - me] for arm, v in (("off", b_off), ("a", b_a), ("b", b_b))}
                rows.append({
                    "label": label, "anchor": anchor, "seed": seed, "seat": me,
                    "health_b": health(r_b, me), "wall_s_b": r_b["wall_s"],
                    "manifest_b": r_b["manifest_sha256"],
                    "kaggriculture_py_b": r_b["runtime"]["kaggriculture_py_sha256"],
                    "tapped": log["tapped"],
                    "switch_line": next((ln["line"] for ln in log["stdout_lines"]
                                         if "load_s=" in ln["line"]), None),
                    "liquidation_lines": sum("final_turn_liquidation farmer" in ln["line"]
                                             for ln in log["stdout_lines"]),
                    "blocks": blocked_orders, "n_blocked": len(reasons),
                    "stdout_reason_count": stdout_reason_count,
                    "first_block_step": first_block,
                    "first_diff_index": fd,
                    "first_block_matches_a": first_block_matches_a,
                    "bank": {"off": b_off[me], "a": b_a[me], "b": b_b[me]},
                    "opp": {"off": b_off[1 - me], "a": b_a[1 - me], "b": b_b[1 - me]},
                    "gap": gap,
                    "d_b_a": gap["b"] - gap["a"], "d_b_off": gap["b"] - gap["off"],
                    "d_a_off": gap["a"] - gap["off"],
                    "unsold_a": unsold(s_a, me), "unsold_b": unsold(s_b, me),
                })
    out_jsonl.write_text("".join(json.dumps(r) + "\n" for r in rows))

    def se(xs: list[float]) -> float:
        return statistics.stdev(xs) / len(xs) ** 0.5

    def wl(gaps: list[float]) -> str:
        ties = sum(g == 0 for g in gaps)
        return f"{sum(g > 0 for g in gaps)}-{sum(g < 0 for g in gaps)}" + (f"-{ties}" if ties else "")

    def seed_stat(sel: list[dict], key: str) -> str:
        means = [statistics.fmean(r[key] for r in sel if r["seed"] == s) for s in SEEDS]
        return f"{statistics.fmean(means):+,.0f} ± {se(means):,.0f}"

    summary = [
        "| anchor | n | W-L OFF | W-L A (r1) | W-L B (r1+r2) | mean gap A | mean gap B | "
        "B − A mean ± SE | B − A min / max | B − OFF mean ± SE | games with blocks | "
        "blocked orders | unsold shed/carried/tile A | B |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for group in (*ANCHORS, "all"):
        sel = [r for r in rows if group in ("all", r["anchor"])]
        u = {arm: {k: statistics.fmean(r[f"unsold_{arm}"][k] for r in sel)
                   for k in ("shed", "carried", "tile")} for arm in ("a", "b")}
        summary.append(
            f"| {group} | {len(sel)} | {wl([r['gap']['off'] for r in sel])} | "
            f"{wl([r['gap']['a'] for r in sel])} | {wl([r['gap']['b'] for r in sel])} | "
            f"{statistics.fmean(r['gap']['a'] for r in sel):+,.0f} | "
            f"{statistics.fmean(r['gap']['b'] for r in sel):+,.0f} | "
            f"{seed_stat(sel, 'd_b_a')} | "
            f"{min(r['d_b_a'] for r in sel):+,.0f} / {max(r['d_b_a'] for r in sel):+,.0f} | "
            f"{seed_stat(sel, 'd_b_off')} | {sum(r['n_blocked'] > 0 for r in sel)} | "
            f"{sum(r['n_blocked'] for r in sel)} | "
            + " / ".join(f"{u['a'][k]:.1f}" for k in ("shed", "carried", "tile")) + " | "
            + " / ".join(f"{u['b'][k]:.1f}" for k in ("shed", "carried", "tile")) + " |"
        )
    cats: collections.Counter[tuple[str, str]] = collections.Counter()
    for r in rows:
        for b in r["blocks"]:
            cats[(r["anchor"], category(b["reason"]))] += 1
    cat_lines = ["| anchor | category | blocked orders |", "|---|---|---|"] + [
        f"| {a} | {c} | {n} |" for (a, c), n in sorted(cats.items())
    ]
    no_block = [r for r in rows if r["n_blocked"] == 0]
    with_block = [r for r in rows if r["n_blocked"] > 0]
    checks = {
        "games": len(rows),
        "b_qualified": sum(r["health_b"]["qualified"] for r in rows),
        "b_calls_719": sum(r["health_b"]["calls"] == 719 for r in rows),
        "b_exceptions": sum(r["health_b"]["exceptions"] for r in rows),
        "b_invalid": sum(r["health_b"]["invalid"] for r in rows),
        "b_default_pass": sum(r["health_b"]["default_pass"] for r in rows),
        "b_bad_statuses": sum(r["health_b"]["bad_statuses"] for r in rows),
        "b_single_manifest": len({r["manifest_b"] for r in rows}),
        "b_single_engine": len({r["kaggriculture_py_b"] for r in rows}),
        "tap_installed": sum(r["tapped"] for r in rows),
        "switch_lines_both_on": sum(
            (r["switch_line"] or "").endswith("final_turn_liquidation=1 block_late_investments=1")
            for r in rows
        ),
        "rule1_liquidation_line_once": sum(r["liquidation_lines"] == 1 for r in rows),
        "tap_count_equals_stdout_count": sum(r["n_blocked"] == r["stdout_reason_count"] for r in rows),
        "blocked_orders_total": sum(r["n_blocked"] for r in rows),
        "games_with_blocks": len(with_block),
        "no_block_games_identical_to_a": sum(r["first_diff_index"] is None for r in no_block),
        "no_block_games": len(no_block),
        "block_games_first_diff_is_first_block": sum(
            r["first_diff_index"] == r["first_block_step"] + 1 for r in with_block
        ),
        "block_games_first_block_market_equals_a": sum(bool(r["first_block_matches_a"]) for r in with_block),
        "every_blocked_order_rederives_a_reason": sum(
            all(b["reason"] is not None for b in r["blocks"]) for r in rows
        ),
        "d_b_a_negative": sum(r["d_b_a"] < 0 for r in rows),
        "d_b_a_zero": sum(r["d_b_a"] == 0 for r in rows),
        "d_b_a_positive": sum(r["d_b_a"] > 0 for r in rows),
        "win_to_loss_vs_a": sum(r["gap"]["a"] > 0 >= r["gap"]["b"] for r in rows),
        "loss_to_win_vs_a": sum(r["gap"]["a"] <= 0 < r["gap"]["b"] for r in rows),
    }
    per_block = ["| game | step (day h) | blocked order | reason |", "|---|---|---|---|"]
    for r in rows:
        for b in r["blocks"]:
            per_block.append(
                f"| {r['label']} | {b['step']} (d{b['day']} h{b['hour']}) | "
                f"`{json.dumps(b['order'])}` | {b['reason']} |"
            )
    per_game = [
        "| game | blocked | first diff idx | gap OFF | gap A | gap B | B − A | bank A → B | "
        "opp A → B | unsold A → B |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        ua, ub = r["unsold_a"], r["unsold_b"]
        per_game.append(
            f"| {r['label']} | {r['n_blocked']} | {r['first_diff_index']} | "
            f"{r['gap']['off']:+,.0f} | {r['gap']['a']:+,.0f} | {r['gap']['b']:+,.0f} | "
            f"{r['d_b_a']:+,.0f} | {r['bank']['a']:,.0f} → {r['bank']['b']:,.0f} | "
            f"{r['opp']['a']:,.0f} → {r['opp']['b']:,.0f} | "
            f"{ua['shed']}/{ua['carried']}/{ua['tile']} → {ub['shed']}/{ub['carried']}/{ub['tile']} |"
        )
    out_md.write_text(
        "## Checks\n\n```json\n" + json.dumps(checks, indent=1) + "\n```\n\n## Summary\n\n"
        + "\n".join(summary) + "\n\n## Blocked orders by category\n\n" + "\n".join(cat_lines)
        + "\n\n## Every blocked order\n\n" + "\n".join(per_block)
        + "\n\n## Per game\n\n" + "\n".join(per_game) + "\n"
    )
    print(json.dumps(checks, indent=1))
    print("\n".join(summary))
    print("\n".join(cat_lines))


if __name__ == "__main__":
    main()
