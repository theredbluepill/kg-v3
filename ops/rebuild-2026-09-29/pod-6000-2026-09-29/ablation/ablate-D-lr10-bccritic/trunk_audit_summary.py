"""Summarise trunk_audit_launcher.py records from a run.log (ablation D receipts).

Usage: trunk_audit_summary.py RUN_LOG -> TSV blocks on stdout:
  # theta0: per rank, per group parameter count and ||theta0||;
  # trunk_audit: per rank and audited iteration, per group relative parameter change
    after that iteration's last optimizer step;
  # grad_audit: per rank and iteration, per group gradient L2 norm / max |grad| /
    None grads / relative parameter change before that iteration's first step.
"""

import json
import sys

GROUPS = ("actor_only", "critic_only", "shared")
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

print("# theta0\trank\t" + "\t".join(f"{g}_params\t{g}_theta0_norm" for g in GROUPS)
      + "\tcritic_head_norm")
for rec in records:
    if rec["kind"] == "trunk_audit_theta0":
        gs = rec["groups"]
        print(f"# theta0\t{rec['rank']}\t" + "\t".join(
            f"{gs[g]['params']}\t{gs[g]['theta0_norm']:.6g}" for g in GROUPS)
            + f"\t{gs['critic_only']['critic_head_norm']:.6g}")
print("trunk_audit\trank\titer\toptimizer_steps\t" + "\t".join(f"{g}_rel_change" for g in GROUPS))
for rec in records:
    if rec["kind"] == "trunk_audit":
        gs = rec["groups"]
        print(f"trunk_audit\t{rec['rank']}\t{rec['iteration']}\t{rec['optimizer_steps']}\t"
              + "\t".join(f"{gs[g]['param_rel_change']:.4e}" for g in GROUPS))
print("grad_audit\trank\titer\tsteps_before\t" + "\t".join(
    f"{g}_{s}" for g in GROUPS for s in ("grad_norm", "grad_max_abs", "none_grads", "rel_change")))
for rec in records:
    if rec["kind"] == "grad_audit":
        gs = rec["groups"]
        cells = []
        for g in GROUPS:
            v = gs[g]
            cells += [f"{v['grad_norm']:.4g}", f"{v['grad_max_abs']:.4g}",
                      str(v["none_grads"]), f"{v['param_rel_change']:.3e}"]
        print(f"grad_audit\t{rec['rank']}\t{rec['iteration']}\t{rec['optimizer_steps_before']}\t"
              + "\t".join(cells))
for rec in records:
    if rec["kind"] == "diagnostic_hook":
        print("# hook", json.dumps(rec))
