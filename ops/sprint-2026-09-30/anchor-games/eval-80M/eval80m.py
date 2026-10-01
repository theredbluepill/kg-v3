"""80M vs 70M vs 60M on the fixed-shop anchors, rule 1 on (played) and off (derived); rule 2 not applied.

Copy of eval-70M/eval70m.py extended by one checkpoint. Arms (48 games each, 8 seeds
93001-93008 x 2 seats x {smaller_market_shock, cha22, v56}):
  60M-off  games-fixedshop/60M        (pkg-60M, 4c99768a, rule 1 switch 0, played)
  60M-ft   games-fixedshop/60M-ft-on  (pkg-60M, 4c99768a, rule 1 switch 1, played)
  70M-ft   games-fixedshop/70M-ft-on  (pkg-70M, 4c99768a, rule 1 switch 1, played)
  80M-ft   games-fixedshop/80M-ft-on  (pkg-80M, 4c99768a, rule 1 switch 1, played)
  70M-off  derived (eval-70M/derive70_derivation.jsonl, games-fixedshop/70M-off-derived)
  80M-off  derived (eval-80M/derive80_derivation.jsonl, games-fixedshop/80M-off-derived):
           80M-ft trajectory to obs 717, the opponent's recorded final action, our model's
           own pre-rule obs-718 action, replayed through the same engine (derive_off.py).

Pairing, SE, W-L and table layout follow eval70m.py (SE over the 8 per-seed means).
Also: the running per-anchor margin table over the earlier fixed-shop arms.

Run: venv-kaggle python, PYTHONPATH=<pkg-80M>:<kenv-fixedshop>
usage: python eval80m.py OUT_DIR
"""

from __future__ import annotations

import gzip
import hashlib
import json
import statistics
import sys
from pathlib import Path

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / "endgame"))
from ab import ANCHORS, SEEDS, health, load, unsold  # noqa: E402

G = W / "games-fixedshop"
E = Path(__file__).resolve().parent
E70 = W / "eval-70M"
PLAYED = {
    "60M-off": (G / "60M", "60M"),
    "60M-ft": (G / "60M-ft-on", "60M"),
    "70M-ft": (G / "70M-ft-on", "70M"),
    "80M-ft": (G / "80M-ft-on", "80M"),
}
DERIVED = {  # arm: (derivation jsonl, derived replay folder, ON arm it was derived from)
    "70M-off": (E70 / "derive70_derivation.jsonl", G / "70M-off-derived", "70M-ft"),
    "80M-off": (E / "derive80_derivation.jsonl", G / "80M-off-derived", "80M-ft"),
}
RUNNING = [  # (name, folder, prefix) -- all rule-off played arms, then rule-on
    ("BC", "bc", "bc"), ("fc6b", "candidate", "candidate"), ("f610", "best-f610", "best-f610"),
    ("60f2", "p3-60f2", "p3-60f2"), ("08bc", "p4", "p4"), ("c50", "c50", "c50"), ("60M", "60M", "60M"),
]
RUNNING_ON = [("c50 ft", "c50-ft-on", "c50"), ("60M ft", "60M-ft-on", "60M")]


def played_row(root: Path, prefix: str, anchor: str, seed: int, me: int) -> dict:
    label = f"{prefix}-{anchor}-s{seed}-seat{me}"
    receipt, replay = load(root, label)
    banks = receipt["summary"]["final_banks"]
    return {
        "label": label, "anchor": anchor, "seed": seed, "seat": me,
        "health": health(receipt, me),
        "manifest_sha256": receipt["manifest_sha256"],
        "kaggriculture_py_sha256": receipt["runtime"]["kaggriculture_py_sha256"],
        "wall_s": receipt["wall_s"],
        "own": banks[me], "opp": banks[1 - me], "gap": banks[me] - banks[1 - me],
        "unsold": unsold(replay["steps"], me), "n_steps": len(replay["steps"]),
    }


def se(xs: list[float]) -> float:
    return statistics.stdev(xs) / len(xs) ** 0.5


def wld(gaps: list[float]) -> str:
    s = f"{sum(g > 0 for g in gaps)}-{sum(g < 0 for g in gaps)}"
    return s + (f"-{sum(g == 0 for g in gaps)}" if any(g == 0 for g in gaps) else "")


def main() -> None:
    out = Path(sys.argv[1])
    manifest_sha = hashlib.sha256((W / "pkg-80M" / "manifest.json").read_bytes()).hexdigest()
    keys = [(a, s, p) for a in ANCHORS for s in SEEDS for p in (0, 1)]
    rows: dict[str, dict[tuple, dict]] = {a: {} for a in (*PLAYED, *DERIVED)}
    for arm, (root, prefix) in PLAYED.items():
        for k in keys:
            rows[arm][k] = played_row(root, prefix, *k)
    derivs: dict[str, dict[tuple, dict]] = {}
    for arm, (jsonl, folder, on_arm) in DERIVED.items():
        deriv = {(r["anchor"], r["seed"], r["seat"]): r for r in map(json.loads, jsonl.read_text().splitlines())}
        derivs[arm] = deriv
        for k in keys:
            d = deriv[k]
            me = k[2]
            steps = json.load(gzip.open(folder / "replays" / f"replay-{d['label']}.json.gz"))["steps"]
            b = d["derived_banks"]
            rows[arm][k] = {
                "label": d["label"] + " (derived)", "anchor": k[0], "seed": k[1], "seat": me,
                "own": b[me], "opp": b[1 - me], "gap": b[me] - b[1 - me],
                "unsold": unsold(steps, me), "n_steps": d["derived_n_steps"],
                "health": rows[on_arm][k]["health"],  # same model calls; only obs-718 output differs
            }
    health_checks = {}
    for arm in PLAYED:
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
    health_checks["80M-ft"]["manifest_is_pkg_80M"] = health_checks["80M-ft"]["manifest_sha256"] == [manifest_sha]
    derivation_checks = {}
    for arm, deriv in derivs.items():
        dv = list(deriv.values())
        derivation_checks[arm] = {
            "games": len(dv),
            "prerule_record_obs718": len(dv),
            "rule_action_equals_on_replay": sum(r["rule_action_equals_on_replay"] for r in dv),
            "rule_changed_action": sum(r["rule_changed_action"] for r in dv),
            "prefix_actions_equal_on_0_717": sum(r["derived_prefix_actions_equal_on"] for r in dv),
            "opp_final_action_equal_on": sum(r["derived_final_opp_action_equal_on"] for r in dv),
            "own_final_action_is_policy": sum(r["derived_final_own_action_is_policy"] for r in dv),
            "derived_bad_statuses": sum(r["derived_bad_statuses"] for r in dv),
            "cf_market_model_matches_engine": sum(r["cf"]["cf_matches_engine"] for r in dv),
            "opp_bank_unchanged_vs_on": sum(
                r["derived_banks"][1 - r["seat"]] == r["on_banks"][1 - r["seat"]] for r in dv),
        }
    v60 = [json.loads(x) for x in (E70 / "validate60_derivation.jsonl").read_text().splitlines()]
    derivation_checks["validation_on_60M_derived_equals_played_off"] = (
        f"{sum(r['derived_equals_off_played'] for r in v60)}/{len(v60)}")
    for arm, base in (("70M-off", E70), ("80M-off", E)):
        spot_path = base / "spot_check_off.json"
        if spot_path.is_file():
            derivation_checks[arm]["spot_check_played_off"] = json.loads(spot_path.read_text())["summary"]
        rc = base / ("replaycheck70.jsonl" if arm == "70M-off" else "replaycheck80.jsonl")
        if rc.is_file():
            rr = [json.loads(x) for x in rc.read_text().splitlines()]
            derivation_checks[arm]["replay_check_no_override"] = {
                "games": len(rr),
                "banks_equal": sum(r["replay_equals_on_banks"] for r in rr),
                "actions_equal": sum(r["replay_actions_equal"] for r in rr),
                "obs_equal_ex_overage": sum(r["replay_obs_equal"] for r in rr)}

    md: list[str] = []
    md.append("## Health (played arms)\n\n```json\n" + json.dumps(health_checks, indent=1) + "\n```\n")
    md.append("## Derivation checks (rule-1-off arms)\n\n```json\n" + json.dumps(derivation_checks, indent=1) + "\n```\n")
    md.append("## Per arm and anchor\n")
    md.append("| arm | anchor | n | W-L | own bank | anchor bank | margin | unsold shed/carried/tile |")
    md.append("|---|---|---|---|---|---|---|---|")
    arm_summary = {}
    for arm in ("60M-off", "70M-off", "80M-off", "60M-ft", "70M-ft", "80M-ft"):
        for group in (*ANCHORS, "all"):
            sel = [rows[arm][k] for k in keys if group in ("all", k[0])]
            u = {x: statistics.fmean(r["unsold"][x] for r in sel) for x in ("shed", "carried", "tile")}
            s = {"n": len(sel), "wl": wld([r["gap"] for r in sel]),
                 "own": statistics.fmean(r["own"] for r in sel), "opp": statistics.fmean(r["opp"] for r in sel),
                 "gap": statistics.fmean(r["gap"] for r in sel), "unsold": u}
            arm_summary[f"{arm}/{group}"] = s
            name = arm + (" (derived)" if arm in DERIVED else "")
            md.append(f"| {name} | {group} | {s['n']} | {s['wl']} | {s['own']:,.0f} | {s['opp']:,.0f} | "
                      f"{s['gap']:+,.0f} | {u['shed']:.1f} / {u['carried']:.1f} / {u['tile']:.1f} |")
    md.append("")
    pairs = {
        "80M-ft vs 70M-ft (rule 1 on, both played)": ("80M-ft", "70M-ft"),
        "80M-ft vs 60M-ft (rule 1 on, both played)": ("80M-ft", "60M-ft"),
        "80M-off (derived) vs 70M-off (derived)": ("80M-off", "70M-off"),
        "80M-off (derived) vs 60M-off (played)": ("80M-off", "60M-off"),
        "80M-ft vs 80M-off (rule-1 effect on 80M)": ("80M-ft", "80M-off"),
    }
    paired = {}
    for name, (a, b) in pairs.items():
        md.append(f"## Paired: {name}\n")
        md.append("| anchor | W-L " + a + " | W-L " + b + " | flips L->W / W->L | "
                  "Δ margin mean ± SE (8 seeds) | better on k/8 seeds | Δ own bank ± SE | "
                  "Δ anchor bank ± SE | min / max Δ margin (game) |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for group in (*ANCHORS, "all"):
            ks = [k for k in keys if group in ("all", k[0])]

            def seed_means(field: str, ks: list = ks) -> list[float]:
                return [statistics.fmean(rows[a][k][field] - rows[b][k][field] for k in ks if k[1] == s)
                        for s in SEEDS]

            dm, do, dp = seed_means("gap"), seed_means("own"), seed_means("opp")
            per_game = [rows[a][k]["gap"] - rows[b][k]["gap"] for k in ks]
            flips = sum(rows[b][k]["gap"] <= 0 < rows[a][k]["gap"] for k in ks)
            back = sum(rows[b][k]["gap"] > 0 >= rows[a][k]["gap"] for k in ks)
            rec = {"wl_a": wld([rows[a][k]["gap"] for k in ks]), "wl_b": wld([rows[b][k]["gap"] for k in ks]),
                   "flips": flips, "back": back,
                   "d_margin": statistics.fmean(dm), "se_margin": se(dm),
                   "better_seeds": sum(x > 0 for x in dm), "worse_seeds": sum(x < 0 for x in dm),
                   "d_own": statistics.fmean(do), "se_own": se(do),
                   "d_opp": statistics.fmean(dp), "se_opp": se(dp),
                   "min_game": min(per_game), "max_game": max(per_game), "seed_means_margin": dm}
            paired[f"{name}/{group}"] = rec
            md.append(f"| {group} | {rec['wl_a']} | {rec['wl_b']} | {flips} / {back} | "
                      f"**{rec['d_margin']:+,.0f} ± {rec['se_margin']:,.0f}** | {rec['better_seeds']}/8 | "
                      f"{rec['d_own']:+,.0f} ± {rec['se_own']:,.0f} | {rec['d_opp']:+,.0f} ± {rec['se_opp']:,.0f} | "
                      f"{rec['min_game']:+,.0f} / {rec['max_game']:+,.0f} |")
        md.append("")

    # Running table (margin mean and W-L per anchor)
    running: dict[str, dict] = {}
    cols = [(n, G / f, p) for n, f, p in RUNNING]
    for name, root, prefix in cols + [(n, G / f, p) for n, f, p in RUNNING_ON]:
        rs = {k: played_row(root, prefix, *k) for k in keys}
        running[name] = {
            g: {"wl": wld([rs[k]["gap"] for k in keys if g in ("all", k[0])]),
                "gap": statistics.fmean(rs[k]["gap"] for k in keys if g in ("all", k[0])),
                "qualified": sum(rs[k]["health"]["qualified"] for k in keys if g in ("all", k[0])),
                "kpy": sorted({rs[k]["kaggriculture_py_sha256"][:8] for k in keys})}
            for g in (*ANCHORS, "all")}
    for name, arm in (("70M (derived off)", "70M-off"), ("80M (derived off)", "80M-off"),
                      ("70M ft", "70M-ft"), ("80M ft", "80M-ft")):
        running[name] = {g: {"wl": wld([rows[arm][k]["gap"] for k in keys if g in ("all", k[0])]),
                             "gap": statistics.fmean(rows[arm][k]["gap"] for k in keys if g in ("all", k[0]))}
                         for g in (*ANCHORS, "all")}
    order = ([n for n, _, _ in RUNNING] + ["70M (derived off)", "80M (derived off)"]
             + [n for n, _, _ in RUNNING_ON] + ["70M ft", "80M ft"])
    md.append("## Running table: mean margin (W-L) per anchor, 16 games each, fixed-shop engine\n")
    md.append("Rule-1-off arms first (all played except 70M and 80M, derived), then rule-1-on arms.\n")
    md.append("| checkpoint | " + " | ".join(ANCHORS) + " | all 48 |")
    md.append("|---|" + "---|" * (len(ANCHORS) + 1))
    for n in order:
        md.append(f"| {n} | " + " | ".join(
            f"{running[n][g]['gap']:+,.0f} ({running[n][g]['wl']})" for g in (*ANCHORS, "all")) + " |")
    md.append("")
    md.append("Running-table health: " + json.dumps(
        {n: {"qualified": running[n]["all"]["qualified"], "kaggriculture_py": running[n]["all"]["kpy"]}
         for n in running if "qualified" in running[n]["all"]}) + "\n")

    md.append("## Per game margins\n")
    md.append("| game | 70M-off (derived) | 80M-off (derived) | 60M-ft | 70M-ft | 80M-ft | "
              "80M unsold shed/carried/tile off -> ft |")
    md.append("|---|---|---|---|---|---|---|")
    for k in keys:
        u0, u1 = rows["80M-off"][k]["unsold"], rows["80M-ft"][k]["unsold"]
        md.append(f"| {k[0]} s{k[1]} seat{k[2]} | {rows['70M-off'][k]['gap']:+,.0f} | "
                  f"{rows['80M-off'][k]['gap']:+,.0f} | {rows['60M-ft'][k]['gap']:+,.0f} | "
                  f"{rows['70M-ft'][k]['gap']:+,.0f} | {rows['80M-ft'][k]['gap']:+,.0f} | "
                  f"{u0['shed']}/{u0['carried']}/{u0['tile']} -> {u1['shed']}/{u1['carried']}/{u1['tile']} |")
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval80m_tables.md").write_text("\n".join(md) + "\n")
    (out / "eval80m_games.jsonl").write_text("".join(
        json.dumps({"arm": arm, **rows[arm][k]}) + "\n" for arm in rows for k in keys))
    (out / "eval80m_summary.json").write_text(json.dumps({
        "health": health_checks, "derivation_checks": derivation_checks,
        "arm_summary": arm_summary, "paired": paired, "running": running,
        "pkg_80M_manifest_sha256": manifest_sha,
    }, indent=1) + "\n")
    print("\n".join(md[:120]))


if __name__ == "__main__":
    main()
