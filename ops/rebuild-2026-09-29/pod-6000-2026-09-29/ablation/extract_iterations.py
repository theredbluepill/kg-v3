"""Per-iteration table and completed-game banks from a run.log's [kg-probe] records.

Usage: extract_iterations.py RUN_LOG [MAX_ITER]  -> TSV on stdout (rank-0 records;
metrics are already rank-reduced by the trainer), then a '# games' block.
"""

import json
import sys

COLS = [
    ("entropy", "policy/entropy"),
    ("approx_kl", "policy/approx_kl"),
    ("clipfrac", "policy/clipfrac"),
    ("adv_std", "train/advantage_std"),
    ("expl_var", "train/explained_variance"),
    ("teacher_kl", "teacher/kl"),
    ("unit_kind_kl", "teacher/unit_kind_kl"),
    ("market_kind_kl", "teacher/market_kind_kl"),
    ("lr", "optimizer/learning_rate"),
    ("kl_coef", "teacher/kl_coef"),
    ("opt_steps", "optimizer/steps"),
]

path = sys.argv[1]
max_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
dec = json.JSONDecoder()
rows: dict[int, dict] = {}
for line in open(path, errors="replace"):
    pos = 0
    while (i := line.find("[kg-probe] ", pos)) >= 0:
        start = i + len("[kg-probe] ")
        try:
            rec, end = dec.raw_decode(line, start)
        except json.JSONDecodeError:
            pos = start
            continue
        pos = end
        if rec.get("kind") == "iteration" and rec.get("rank") == 0 and rec["iteration"] <= max_iter:
            rows[rec["iteration"]] = rec
print("iter\t" + "\t".join(c for c, _ in COLS) + "\tnonfinite")
games = []
for it in sorted(rows):
    m = rows[it]["metrics"]
    print(f"{it}\t" + "\t".join(f"{m[k]:.4g}" if k in m else "" for _, k in COLS) + f"\t{len(rows[it]['nonfinite'])}")
    if "train/terminal_bank_0" in m:
        games.append((it, m["train/terminal_bank_0"], m["train/terminal_bank_1"], m.get("train/total_games_played")))
print("# games: game\titer\tseat0_bank\tseat1_bank\ttotal_games_played")
for g, (it, b0, b1, tg) in enumerate(games, 1):
    print(f"# {g}\t{it}\t{b0:.0f}\t{b1:.0f}\t{tg}")
