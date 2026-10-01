"""Win/loss and bank margin per anchor from a folder of run_game.py receipts.

usage: python3 summarize.py GAMES_DIR [--compare OTHER_GAMES_DIR] [--json OUT]

Receipts only (no replays): our seat is receipt["agent_seats"][0], banks are
summary.final_banks. Games pair across folders by (anchor, seed, seat) whatever
the label prefix. SE is over the per-seed means (each averages the two seats),
since mirrored seats are not independent. Plain stdlib; any python3 works.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

LABEL = re.compile(r"^(?P<prefix>.+)-(?P<anchor>[a-z0-9_]+)-s(?P<seed>\d+)-seat(?P<seat>[01])$")


def load(folder: Path) -> dict[tuple[str, int, int], dict]:
    rows = {}
    for path in sorted((folder / "receipts").glob("*.json")):
        r = json.loads(path.read_text())
        m = LABEL.match(r["label"])
        if m is None:
            raise ValueError(f"unexpected label {r['label']!r} in {path}")
        me = r["agent_seats"][0]
        if me != int(m["seat"]):
            raise ValueError(f"{path}: agent seat {me} != label seat {m['seat']}")
        banks = r["summary"]["final_banks"]
        seat = r["summary"]["seats"][str(me)]
        rows[(m["anchor"], int(m["seed"]), me)] = {
            "label": r["label"],
            "qualified": r["qualified"],
            "own": banks[me],
            "opp": banks[1 - me],
            "gap": banks[me] - banks[1 - me],
            "default_pass": seat["default_pass_returns"],
            "wall_s": r["wall_s"],
            "manifest": r["manifest_sha256"],
            "engine": r["runtime"]["kaggriculture_py_sha256"],
            "platform": r["runtime"]["platform"],
            "torch": r["runtime"]["torch"],
        }
    return rows


def se_over_seeds(rows: list[dict], key: str) -> float:
    by_seed: dict[int, list[float]] = {}
    for r in rows:
        by_seed.setdefault(r["seed"], []).append(r[key])
    means = [statistics.fmean(v) for v in by_seed.values()]
    return statistics.stdev(means) / len(means) ** 0.5 if len(means) > 1 else float("nan")


def table(rows: dict[tuple[str, int, int], dict]) -> tuple[list[str], dict]:
    anchors = sorted({k[0] for k in rows})
    out = ["| anchor | games | qualified | W-L-D | own bank | anchor bank | margin (SE seeds) | default passes |",
           "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    summary = {}
    for name in anchors + ["ALL"]:
        sel = [dict(v, seed=k[1]) for k, v in rows.items() if name in ("ALL", k[0])]
        gaps = [r["gap"] for r in sel]
        s = {
            "games": len(sel),
            "qualified": sum(r["qualified"] for r in sel),
            "wins": sum(g > 0 for g in gaps),
            "losses": sum(g < 0 for g in gaps),
            "draws": sum(g == 0 for g in gaps),
            "own_mean": statistics.fmean(r["own"] for r in sel),
            "opp_mean": statistics.fmean(r["opp"] for r in sel),
            "gap_mean": statistics.fmean(gaps),
            "gap_se_seeds": se_over_seeds(sel, "gap"),
            "default_pass": sum(r["default_pass"] for r in sel),
        }
        summary[name] = s
        out.append(f"| {name} | {s['games']} | {s['qualified']} | {s['wins']}-{s['losses']}-{s['draws']} | "
                   f"{s['own_mean']:.1f} | {s['opp_mean']:.1f} | {s['gap_mean']:+.1f} ({s['gap_se_seeds']:.1f}) | "
                   f"{s['default_pass']} |")
    return out, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("games_dir", type=Path)
    ap.add_argument("--compare", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    rows = load(a.games_dir)
    lines, summary = table(rows)
    ids = {k: sorted({r[k] for r in rows.values()}) for k in ("manifest", "engine", "platform", "torch")}
    print(f"## {a.games_dir.name}: {len(rows)} receipts")
    print(json.dumps(ids, indent=1))
    print("\n".join(lines))
    report = {"games_dir": str(a.games_dir), "ids": ids, "summary": summary}
    if a.compare:
        other = load(a.compare)
        keys = sorted(set(rows) & set(other))
        d = [rows[k]["gap"] - other[k]["gap"] for k in keys]
        same = sum(rows[k]["own"] == other[k]["own"] and rows[k]["opp"] == other[k]["opp"] for k in keys)
        per_seed: dict[int, list[float]] = {}
        for k, x in zip(keys, d):
            per_seed.setdefault(k[1], []).append(x)
        means = [statistics.fmean(v) for v in per_seed.values()]
        se = statistics.stdev(means) / len(means) ** 0.5 if len(means) > 1 else float("nan")
        print(f"\n## paired vs {a.compare.name}: {len(keys)} games, identical final banks in {same}; "
              f"margin difference {statistics.fmean(d):+.1f} (SE seeds {se:.1f})")
        _, osum = table(other)
        for name in summary:
            if name in osum:
                print(f"- {name}: margin {summary[name]['gap_mean']:+.1f} vs {osum[name]['gap_mean']:+.1f}; "
                      f"W-L {summary[name]['wins']}-{summary[name]['losses']} vs {osum[name]['wins']}-{osum[name]['losses']}")
        report["compare"] = {"other": str(a.compare), "paired": len(keys), "identical_final_banks": same,
                             "margin_diff_mean": statistics.fmean(d), "margin_diff_se_seeds": se,
                             "other_summary": osum}
    if a.json:
        a.json.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
