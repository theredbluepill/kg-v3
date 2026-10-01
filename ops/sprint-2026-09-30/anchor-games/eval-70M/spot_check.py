"""Compare played 70M rule-1-OFF spot games with the derived OFF games.

Played: games-fixedshop/70M-off-spot (run_fixedshop_70M.sh 70M-off-spot 0, pkg-70M).
Derived: derive70_derivation.jsonl + games-fixedshop/70M-off-derived/replays.
Checks per game: receipt qualified/health, final banks equal, every step's actions
equal, every observation equal except remainingOverageTime.
usage: python spot_check.py OUT.json
"""
import gzip, json, sys
from pathlib import Path

W = Path(__file__).resolve().parents[1]
G = W / "games-fixedshop"
E = Path(__file__).resolve().parent
deriv = {r["label"]: r for r in map(json.loads, (E / "derive70_derivation.jsonl").read_text().splitlines())}
rows = []
for rp in sorted((G / "70M-off-spot" / "receipts").glob("*.json")):
    rec = json.loads(rp.read_text())
    label = rec["label"]; me = rec["agent_seats"][0]
    played = json.load(gzip.open(G / "70M-off-spot" / "replays" / f"replay-{label}.json.gz"))["steps"]
    derived = json.load(gzip.open(G / "70M-off-derived" / "replays" / f"replay-{label}.json.gz"))["steps"]
    strip = lambda o: {k: v for k, v in o.items() if k != "remainingOverageTime"}
    seat = rec["summary"]["seats"][str(me)]
    rows.append({
        "label": label, "qualified": rec["qualified"], "calls": seat["calls"],
        "exceptions": seat["exceptions"], "invalid": seat["invalid_raw_actions"],
        "default_pass": seat["default_pass_returns"], "bad_statuses": rec["summary"]["bad_status_count"],
        "played_banks": rec["summary"]["final_banks"], "derived_banks": deriv[label]["derived_banks"],
        "banks_equal": rec["summary"]["final_banks"] == deriv[label]["derived_banks"],
        "n_steps": [len(played), len(derived)],
        "all_actions_equal": len(played) == len(derived) and all(
            played[t][p]["action"] == derived[t][p]["action"] for t in range(len(played)) for p in (0, 1)),
        "all_obs_equal_ex_overage": len(played) == len(derived) and all(
            strip(played[t][p]["observation"]) == strip(derived[t][p]["observation"])
            for t in range(len(played)) for p in (0, 1)),
    })
summary = {"games": len(rows), "qualified": sum(r["qualified"] for r in rows),
           "banks_equal": sum(r["banks_equal"] for r in rows),
           "all_actions_equal": sum(r["all_actions_equal"] for r in rows),
           "all_obs_equal_ex_overage": sum(r["all_obs_equal_ex_overage"] for r in rows)}
Path(sys.argv[1]).write_text(json.dumps({"summary": summary, "games": rows}, indent=1) + "\n")
print(json.dumps(summary)); [print(json.dumps(r)) for r in rows]
