"""60M vs c50 on the fixed-shop anchors, rule 1 off and on (rule 2 not applied).

Arms (48 games each, 8 seeds 93001-93008 x 2 seats x {smaller_market_shock, cha22, v56}):
  c50-off  games-fixedshop/c50        (pkg-c50, no rules)
  c50-ft   games-fixedshop/c50-ft-on  (pkg-c50-ft, 4c99768a, rule 1 on)
  60M-off  games-fixedshop/60M        (pkg-60M, 4c99768a, rule 1 switch 0)
  60M-ft   games-fixedshop/60M-ft-on  (pkg-60M, 4c99768a, rule 1 switch 1)

Reuses endgame/ab.py (load, health, unsold, obs_at) and the packaged
liquidate_final_turn. Comparisons pair 1:1 by (anchor, seed, seat):
  60M-off vs c50-off, 60M-ft vs c50-ft (different policies: no step-identity check);
  60M-ft vs 60M-off (rule-1 effect: replays must differ only in our step-718 action,
  which must equal the rule applied to the OFF obs 718 and model action).
SE is over the 8 per-seed means (each averages the 2 seats per anchor, 6 games for all).

Run: venv-kaggle python, PYTHONPATH=<pkg-60M>:<kenv-fixedshop>
usage: python eval60m.py OUT_DIR
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / "endgame"))
from ab import ANCHORS, SEEDS, health, load, obs_at, unsold  # noqa: E402
from owl.kaggriculture.final_turn import liquidate_final_turn  # noqa: E402

G = W / "games-fixedshop"
ARMS = {
    "c50-off": (G / "c50", "c50"),
    "c50-ft": (G / "c50-ft-on", "c50"),
    "60M-off": (G / "60M", "60M"),
    "60M-ft": (G / "60M-ft-on", "60M"),
}


def game_row(root: Path, prefix: str, anchor: str, seed: int, me: int) -> tuple[dict, list, dict]:
    label = f"{prefix}-{anchor}-s{seed}-seat{me}"
    receipt, replay = load(root, label)
    steps = replay["steps"]
    banks = receipt["summary"]["final_banks"]
    row = {
        "label": label, "anchor": anchor, "seed": seed, "seat": me,
        "health": health(receipt, me),
        "manifest_sha256": receipt["manifest_sha256"],
        "kaggriculture_py_sha256": receipt["runtime"]["kaggriculture_py_sha256"],
        "wall_s": receipt["wall_s"],
        "own": banks[me], "opp": banks[1 - me], "gap": banks[me] - banks[1 - me],
        "unsold": unsold(steps, me),
        "n_steps": len(steps),
    }
    return row, steps, replay["configuration"]


def se(xs: list[float]) -> float:
    return statistics.stdev(xs) / len(xs) ** 0.5


def wld(gaps: list[float]) -> str:
    s = f"{sum(g > 0 for g in gaps)}-{sum(g < 0 for g in gaps)}"
    return s + (f"-{sum(g == 0 for g in gaps)}" if any(g == 0 for g in gaps) else "")


def main() -> None:
    out = Path(sys.argv[1])
    manifest_sha = hashlib.sha256((W / "pkg-60M" / "manifest.json").read_bytes()).hexdigest()
    rows: dict[str, dict[tuple, dict]] = {a: {} for a in ARMS}
    rule_rows = []
    for anchor in ANCHORS:
        for seed in SEEDS:
            for me in (0, 1):
                key = (anchor, seed, me)
                for arm in ("c50-off", "c50-ft"):
                    rows[arm][key], _, _ = game_row(*ARMS[arm], anchor, seed, me)
                r_off, s_off, _ = game_row(*ARMS["60M-off"], anchor, seed, me)
                r_on, s_on, replay_cfg = game_row(*ARMS["60M-ft"], anchor, seed, me)
                rows["60M-off"][key], rows["60M-ft"][key] = r_off, r_on
                diff = [
                    (t, seat)
                    for t in range(max(len(s_off), len(s_on)))
                    for seat in (0, 1)
                    if t >= len(s_off) or t >= len(s_on) or s_off[t][seat]["action"] != s_on[t][seat]["action"]
                ]
                final = len(s_on) - 1
                obs718 = obs_at(s_off, final - 1, me)
                expected = liquidate_final_turn(
                    obs718, replay_cfg, s_off[final][me]["action"],
                    order_limit=int(replay_cfg["maxMarketOrdersPerTurn"]),
                )
                rule_rows.append({
                    "label": r_on["label"], "anchor": anchor, "seed": seed, "seat": me,
                    "diff_steps": diff, "final_index": final,
                    "only_our_final_action_differs": all(t == final and s == me for t, s in diff),
                    "on_equals_rule": s_on[final][me]["action"] == expected,
                    "rule_changed_action": expected != s_off[final][me]["action"],
                    "obs_step_of_rule": obs718["step"],
                    "gap_off": r_off["gap"], "gap_on": r_on["gap"], "delta": r_on["gap"] - r_off["gap"],
                    "opp_off": r_off["opp"], "opp_on": r_on["opp"],
                    "unsold_off": r_off["unsold"], "unsold_on": r_on["unsold"],
                })
                del s_off, s_on
                print(anchor, seed, me, "done", flush=True)

    keys = list(rows["c50-off"])
    health_checks = {}
    for arm in ARMS:
        rs = [rows[arm][k] for k in keys]
        health_checks[arm] = {
            "games": len(rs),
            "qualified": sum(r["health"]["qualified"] for r in rs),
            "calls_719": sum(r["health"]["calls"] == 719 for r in rs),
            "exceptions": sum(r["health"]["exceptions"] for r in rs),
            "invalid_raw_actions": sum(r["health"]["invalid"] for r in rs),
            "default_pass_fallbacks": sum(r["health"]["default_pass"] for r in rs),
            "bad_statuses": sum(r["health"]["bad_statuses"] for r in rs),
            "manifest_sha256": sorted({r["manifest_sha256"] for r in rs}),
            "kaggriculture_py_sha256": sorted({r["kaggriculture_py_sha256"] for r in rs}),
            "mean_wall_s": round(statistics.fmean(r["wall_s"] for r in rs), 1),
        }
    for arm in ("60M-off", "60M-ft"):
        health_checks[arm]["manifest_is_pkg_60M"] = health_checks[arm]["manifest_sha256"] == [manifest_sha]
    rule_checks = {
        "games": len(rule_rows),
        "only_our_final_action_differs": sum(r["only_our_final_action_differs"] for r in rule_rows),
        "on_final_action_equals_rule_on_off_obs": sum(r["on_equals_rule"] for r in rule_rows),
        "rule_changed_action": sum(r["rule_changed_action"] for r in rule_rows),
        "rule_obs_step_718": sum(r["obs_step_of_rule"] == 718 for r in rule_rows),
        "opponent_bank_unchanged": sum(r["opp_off"] == r["opp_on"] for r in rule_rows),
        "delta_negative": sum(r["delta"] < 0 for r in rule_rows),
        "flips_L_to_W": sum(r["gap_off"] <= 0 < r["gap_on"] for r in rule_rows),
    }

    md: list[str] = []
    md.append("## Health\n\n```json\n" + json.dumps(health_checks, indent=1) + "\n```\n")
    md.append("## Rule-1 checks (60M-ft vs 60M-off)\n\n```json\n" + json.dumps(rule_checks, indent=1) + "\n```\n")

    # Per-arm per-anchor table
    md.append("## Per arm and anchor\n")
    md.append("| arm | anchor | n | W-L | own bank | anchor bank | margin | unsold shed/carried/tile |")
    md.append("|---|---|---|---|---|---|---|---|")
    arm_summary = {}
    for arm in ARMS:
        for group in (*ANCHORS, "all"):
            sel = [rows[arm][k] for k in keys if group in ("all", k[0])]
            u = {x: statistics.fmean(r["unsold"][x] for r in sel) for x in ("shed", "carried", "tile")}
            s = {
                "n": len(sel), "wl": wld([r["gap"] for r in sel]),
                "own": statistics.fmean(r["own"] for r in sel),
                "opp": statistics.fmean(r["opp"] for r in sel),
                "gap": statistics.fmean(r["gap"] for r in sel),
                "unsold": u,
            }
            arm_summary[f"{arm}/{group}"] = s
            md.append(
                f"| {arm} | {group} | {s['n']} | {s['wl']} | {s['own']:,.0f} | {s['opp']:,.0f} | "
                f"{s['gap']:+,.0f} | {u['shed']:.1f} / {u['carried']:.1f} / {u['tile']:.1f} |"
            )
    md.append("")

    # Paired comparisons
    pairs = {
        "60M-off vs c50-off": ("60M-off", "c50-off"),
        "60M-ft vs c50-ft": ("60M-ft", "c50-ft"),
        "60M-ft vs 60M-off (rule 1)": ("60M-ft", "60M-off"),
    }
    paired = {}
    for name, (a, b) in pairs.items():
        md.append(f"## Paired: {name}\n")
        md.append(
            "| anchor | W-L " + a + " | W-L " + b + " | flips L->W / W->L | "
            "Δ margin mean ± SE (8 seeds) | better on k/8 seeds | Δ own bank ± SE | "
            "Δ anchor bank ± SE | min / max Δ margin (game) |"
        )
        md.append("|---|---|---|---|---|---|---|---|---|")
        for group in (*ANCHORS, "all"):
            ks = [k for k in keys if group in ("all", k[0])]

            def seed_means(field: str, ks: list = ks) -> list[float]:
                return [
                    statistics.fmean(rows[a][k][field] - rows[b][k][field] for k in ks if k[1] == s)
                    for s in SEEDS
                ]

            dm, do, dp = seed_means("gap"), seed_means("own"), seed_means("opp")
            per_game = [rows[a][k]["gap"] - rows[b][k]["gap"] for k in ks]
            flips = sum(rows[b][k]["gap"] <= 0 < rows[a][k]["gap"] for k in ks)
            back = sum(rows[b][k]["gap"] > 0 >= rows[a][k]["gap"] for k in ks)
            rec = {
                "wl_a": wld([rows[a][k]["gap"] for k in ks]), "wl_b": wld([rows[b][k]["gap"] for k in ks]),
                "flips": flips, "back": back,
                "d_margin": statistics.fmean(dm), "se_margin": se(dm),
                "better_seeds": sum(x > 0 for x in dm), "worse_seeds": sum(x < 0 for x in dm),
                "d_own": statistics.fmean(do), "se_own": se(do),
                "d_opp": statistics.fmean(dp), "se_opp": se(dp),
                "min_game": min(per_game), "max_game": max(per_game),
                "seed_means_margin": dm,
            }
            paired[f"{name}/{group}"] = rec
            md.append(
                f"| {group} | {rec['wl_a']} | {rec['wl_b']} | {flips} / {back} | "
                f"**{rec['d_margin']:+,.0f} ± {rec['se_margin']:,.0f}** | {rec['better_seeds']}/8 | "
                f"{rec['d_own']:+,.0f} ± {rec['se_own']:,.0f} | {rec['d_opp']:+,.0f} ± {rec['se_opp']:,.0f} | "
                f"{rec['min_game']:+,.0f} / {rec['max_game']:+,.0f} |"
            )
        md.append("")

    # Closest losses and all wins for the 60M arms
    for arm, ref in (("60M-off", "c50-off"), ("60M-ft", "c50-ft")):
        rs = sorted((rows[arm][k] for k in keys), key=lambda r: -r["gap"])
        wins = [r for r in rs if r["gap"] > 0]
        losses = [r for r in rs if r["gap"] <= 0][:8]
        md.append(f"## {arm}: wins and closest losses (with {ref} same game)\n")
        md.append(f"| game | {arm} margin | {ref} margin | own | anchor |")
        md.append("|---|---|---|---|---|")
        for r in wins + losses:
            k = (r["anchor"], r["seed"], r["seat"])
            md.append(
                f"| {r['anchor']} s{r['seed']} seat{r['seat']}{' (win)' if r['gap'] > 0 else ''} | "
                f"{r['gap']:+,.0f} | {rows[ref][k]['gap']:+,.0f} | {r['own']:,.0f} | {r['opp']:,.0f} |"
            )
        md.append("")

    # Per game
    md.append("## Per game margins\n")
    md.append("| game | c50-off | 60M-off | c50-ft | 60M-ft | 60M unsold shed/carried/tile off -> ft | c50 off unsold |")
    md.append("|---|---|---|---|---|---|---|")
    for k in keys:
        u0, u1, uc = rows["60M-off"][k]["unsold"], rows["60M-ft"][k]["unsold"], rows["c50-off"][k]["unsold"]
        md.append(
            f"| {k[0]} s{k[1]} seat{k[2]} | {rows['c50-off'][k]['gap']:+,.0f} | {rows['60M-off'][k]['gap']:+,.0f} | "
            f"{rows['c50-ft'][k]['gap']:+,.0f} | {rows['60M-ft'][k]['gap']:+,.0f} | "
            f"{u0['shed']}/{u0['carried']}/{u0['tile']} -> {u1['shed']}/{u1['carried']}/{u1['tile']} | "
            f"{uc['shed']}/{uc['carried']}/{uc['tile']} |"
        )
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval60m_tables.md").write_text("\n".join(md) + "\n")
    (out / "eval60m_games.jsonl").write_text("".join(
        json.dumps({"arm": arm, **rows[arm][k]}) + "\n" for arm in ARMS for k in keys))
    (out / "eval60m_rule1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rule_rows))
    (out / "eval60m_summary.json").write_text(json.dumps({
        "health": health_checks, "rule1_checks": rule_checks,
        "arm_summary": arm_summary, "paired": paired,
        "pkg_60M_manifest_sha256": manifest_sha,
    }, indent=1) + "\n")
    print("\n".join(md[:60]))


if __name__ == "__main__":
    main()
