"""Paired A/B of the final-turn liquidation rule on c50 (fixed-shop engine).

OFF = games-fixedshop/c50 (pkg-c50, rule absent), ON = games-fixedshop/c50-ft-on
(pkg-c50-ft, kg/submit-08bc 4c99768a, KAGGRICULTURE_FINAL_TURN_LIQUIDATION=1).
Per game: receipt health (qualified, exceptions, invalid raw actions, default-PASS
returns, bad statuses), the replay steps whose actions differ between arms (must be
only our final-turn action), whether the ON final action equals the packaged
rule applied to the OFF observation and action, banks, gap and the unsold goods
at the end by location. Margin change is summarized with SE over the 8 seeds.

Run: venv-kaggle python, PYTHONPATH=<pkg-c50-ft>:<kenv-fixedshop>
usage: python ab.py OUT.jsonl OUT_TABLES.md
"""

from __future__ import annotations

import gzip
import json
import statistics
import sys
from pathlib import Path

from owl.kaggriculture.final_turn import liquidate_final_turn
from owl.kaggriculture.types import PRODUCTS

W = Path(__file__).resolve().parents[1]
OFF, ON = W / "games-fixedshop" / "c50", W / "games-fixedshop" / "c50-ft-on"
ANCHORS = ("smaller_market_shock", "cha22", "v56")
SEEDS = tuple(range(93001, 93009))
ANIMAL_PRODUCT = {"COW": "MILK", "SHEEP": "WOOL", "GOOSE": "EGG"}


def load(root: Path, label: str) -> tuple[dict, dict]:
    receipt = json.loads((root / "receipts" / f"{label}.json").read_text())
    replay = json.load(gzip.open(root / "replays" / f"replay-{label}.json.gz"))
    return receipt, replay


def health(receipt: dict, me: int) -> dict:
    seat = receipt["summary"]["seats"][str(me)]
    return {
        "qualified": receipt["qualified"],
        "calls": seat["calls"],
        "exceptions": seat["exceptions"],
        "invalid": seat["invalid_raw_actions"],
        "default_pass": seat["default_pass_returns"],
        "bad_statuses": receipt["summary"]["bad_status_count"],
    }


def unsold(steps: list, me: int) -> dict[str, int]:
    obs = steps[-1][me]["observation"]
    shared = steps[-1][0]["observation"]
    private = obs["private"]
    shed = sum(v for k, v in private["shed"].items() if k in PRODUCTS and v > 0)
    carried = sum(
        v for inv in private["inventories"] for k, v in inv.items() if k in PRODUCTS and v > 0
    )
    tile = 0
    for row in shared["farms"][me]["tiles"]:
        for t in row:
            if isinstance(t, dict) and t.get("yield_units"):
                item = t.get("crop") if t.get("kind") == "PLANT" else ANIMAL_PRODUCT.get(t.get("animal"))
                if item:
                    tile += t["yield_units"]
    return {"shed": shed, "carried": carried, "tile": tile}


def obs_at(steps: list, t: int, me: int) -> dict:
    o = dict(steps[t][me]["observation"])
    for k, v in steps[t][0]["observation"].items():
        o.setdefault(k, v)
    o.setdefault("step", t)
    return o


def main() -> None:
    out_jsonl, out_md = map(Path, sys.argv[1:3])
    rows = []
    for anchor in ANCHORS:
        for seed in SEEDS:
            for me in (0, 1):
                label = f"c50-{anchor}-s{seed}-seat{me}"
                r_off, p_off = load(OFF, label)
                r_on, p_on = load(ON, label)
                s_off, s_on = p_off["steps"], p_on["steps"]
                diff = [
                    (t, seat)
                    for t in range(max(len(s_off), len(s_on)))
                    for seat in (0, 1)
                    if t >= len(s_off) or t >= len(s_on) or s_off[t][seat]["action"] != s_on[t][seat]["action"]
                ]
                final = len(s_on) - 1  # replay index holding the action for obs final-1
                cfg = p_on["configuration"]
                obs718 = obs_at(s_off, final - 1, me)
                expected = liquidate_final_turn(
                    obs718, cfg, s_off[final][me]["action"],
                    order_limit=int(cfg["maxMarketOrdersPerTurn"]),
                )
                b_off, b_on = r_off["summary"]["final_banks"], r_on["summary"]["final_banks"]
                gap_off, gap_on = b_off[me] - b_off[1 - me], b_on[me] - b_on[1 - me]
                rows.append({
                    "label": label, "anchor": anchor, "seed": seed, "seat": me,
                    "health_on": health(r_on, me), "health_off": health(r_off, me),
                    "wall_s_on": r_on["wall_s"],
                    "diff_steps": diff,
                    "final_index": final,
                    "on_equals_rule": s_on[final][me]["action"] == expected,
                    "rule_changed_action": expected != s_off[final][me]["action"],
                    "obs_step_of_rule": obs718["step"],
                    "bank_off": b_off[me], "bank_on": b_on[me],
                    "opp_off": b_off[1 - me], "opp_on": b_on[1 - me],
                    "gap_off": gap_off, "gap_on": gap_on, "delta": gap_on - gap_off,
                    "unsold_off": unsold(s_off, me), "unsold_on": unsold(s_on, me),
                    "unsold_final_obs_off": unsold(s_off[: final], me),
                })
    out_jsonl.write_text("".join(json.dumps(r) + "\n" for r in rows))

    def se(xs: list[float]) -> float:
        return statistics.stdev(xs) / len(xs) ** 0.5

    def wld(gaps: list[float]) -> str:
        return f"{sum(g > 0 for g in gaps)}-{sum(g < 0 for g in gaps)}" + (
            f"-{sum(g == 0 for g in gaps)}" if any(g == 0 for g in gaps) else ""
        )

    lines = ["| anchor | n | W-L OFF | W-L ON | flips L->W | mean gap OFF | mean gap ON | "
             "margin change mean ± SE (8 seeds) | min / max change | "
             "unsold shed/carried/tile OFF | unsold shed/carried/tile ON |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for group in (*ANCHORS, "all"):
        sel = [r for r in rows if group in ("all", r["anchor"])]
        seed_means = [statistics.fmean(r["delta"] for r in sel if r["seed"] == s) for s in SEEDS]
        flips = sum(r["gap_off"] <= 0 < r["gap_on"] for r in sel)
        back = sum(r["gap_off"] > 0 >= r["gap_on"] for r in sel)
        u = {arm: {k: statistics.fmean(r[f"unsold_{arm}"][k] for r in sel)
                   for k in ("shed", "carried", "tile")} for arm in ("off", "on")}
        lines.append(
            f"| {group} | {len(sel)} | {wld([r['gap_off'] for r in sel])} | "
            f"{wld([r['gap_on'] for r in sel])} | {flips}" + (f" (W->L {back})" if back else "") + " | "
            f"{statistics.fmean(r['gap_off'] for r in sel):+,.0f} | "
            f"{statistics.fmean(r['gap_on'] for r in sel):+,.0f} | "
            f"{statistics.fmean(seed_means):+,.0f} ± {se(seed_means):,.0f} | "
            f"{min(r['delta'] for r in sel):+,.0f} / {max(r['delta'] for r in sel):+,.0f} | "
            + " / ".join(f"{u['off'][k]:.1f}" for k in ("shed", "carried", "tile")) + " | "
            + " / ".join(f"{u['on'][k]:.1f}" for k in ("shed", "carried", "tile")) + " |"
        )
    checks = {
        "games": len(rows),
        "on_qualified": sum(r["health_on"]["qualified"] for r in rows),
        "on_calls_719": sum(r["health_on"]["calls"] == 719 for r in rows),
        "on_exceptions": sum(r["health_on"]["exceptions"] for r in rows),
        "on_invalid": sum(r["health_on"]["invalid"] for r in rows),
        "on_default_pass": sum(r["health_on"]["default_pass"] for r in rows),
        "on_bad_statuses": sum(r["health_on"]["bad_statuses"] for r in rows),
        "only_our_final_action_differs": sum(
            all(t == r["final_index"] and seat == r["seat"] for t, seat in r["diff_steps"])
            for r in rows
        ),
        "rule_changed_action": sum(r["rule_changed_action"] for r in rows),
        "on_final_action_equals_rule_on_off_obs": sum(r["on_equals_rule"] for r in rows),
        "rule_obs_step_718": sum(r["obs_step_of_rule"] == 718 for r in rows),
        "opponent_bank_unchanged": sum(r["opp_off"] == r["opp_on"] for r in rows),
        "delta_negative": sum(r["delta"] < 0 for r in rows),
    }
    per_game = ["| game | gap OFF | gap ON | change | unsold shed/carried/tile OFF -> ON |", "|---|---|---|---|---|"]
    for r in rows:
        uo, un = r["unsold_off"], r["unsold_on"]
        per_game.append(
            f"| {r['label']} | {r['gap_off']:+,.0f} | {r['gap_on']:+,.0f} | {r['delta']:+,.0f} | "
            f"{uo['shed']}/{uo['carried']}/{uo['tile']} -> {un['shed']}/{un['carried']}/{un['tile']} |"
        )
    out_md.write_text(
        "## Checks\n\n```json\n" + json.dumps(checks, indent=1) + "\n```\n\n## Summary\n\n"
        + "\n".join(lines) + "\n\n## Per game\n\n" + "\n".join(per_game) + "\n"
    )
    print(json.dumps(checks, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
