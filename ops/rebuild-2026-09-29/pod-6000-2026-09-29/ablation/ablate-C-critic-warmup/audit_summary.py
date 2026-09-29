"""Summarise critic warm-up hook records from a run.log (ablation C receipts).

Usage: audit_summary.py RUN_LOG -> one line per grad_audit record (rank,
iteration, optimizer steps before the step, warm-up flag, loss-call counts,
per-group grad L2 norm / max |grad| / None grads / relative parameter change),
then the diagnostic_hook record(s) and rank-0 iteration loss terms.
"""

import json
import sys

dec = json.JSONDecoder()
records = []
for line in open(sys.argv[1], errors="replace"):
    pos = 0
    while (i := line.find("[kg-probe] ", pos)) >= 0:
        start = i + len("[kg-probe] ")
        try:
            rec, end = dec.raw_decode(line, start)
        except json.JSONDecodeError:
            pos = start
            continue
        pos = end
        records.append(rec)

print("rank\titer\tsteps_before\twarmup\tmasked\tpassthrough\t" + "\t".join(
    f"{g}_{s}" for g in ("actor_only", "critic_only", "shared")
    for s in ("grad_norm", "grad_max_abs", "none_grads", "param_rel_change")))
for rec in records:
    if rec["kind"] != "grad_audit":
        continue
    groups = rec["groups"]
    cells = []
    for g in ("actor_only", "critic_only", "shared"):
        v = groups[g]
        cells += [f"{v['grad_norm']:.4g}", f"{v['grad_max_abs']:.4g}",
                  str(v["none_grads"]), f"{v['param_rel_change']:.3e}"]
    print(f"{rec['rank']}\t{rec['iteration']}\t{rec['optimizer_steps_before']}\t"
          f"{int(rec['warmup'])}\t{rec['loss_calls']['masked']}\t"
          f"{rec['loss_calls']['passthrough']}\t" + "\t".join(cells))
for rec in records:
    if rec["kind"] == "diagnostic_hook":
        print("# hook", json.dumps(rec))
    if rec["kind"] == "grad_audit" and rec["rank"] == 0 and rec["iteration"] == 1:
        print("# params per group", {g: v["params"] for g, v in rec["groups"].items()})
        break
keys = ("loss/policy_loss", "loss/teacher_kl_loss", "loss/entropy_loss",
        "loss/value_loss", "policy/entropy", "policy/approx_kl", "teacher/kl",
        "train/explained_variance")
for rec in records:
    if rec["kind"] == "iteration" and rec["rank"] == 0:
        m = rec["metrics"]
        print(f"# iter {rec['iteration']} " + " ".join(f"{k}={m.get(k)}" for k in keys))
