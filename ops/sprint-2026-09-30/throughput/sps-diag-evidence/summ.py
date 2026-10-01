import json, sys, statistics as st
for path in sys.argv[1:]:
    recs = []
    for line in open(path, errors="replace"):
        i = line.find('[nt-probe] {"kind": "iteration"')
        if i < 0: continue
        s = line[i+11:]
        recs.append(json.JSONDecoder().raw_decode(s)[0])
    print("==", path, "iters", len(recs))
    keys = ["time/rollout_seconds", "time/update_seconds", "perf/steps_per_second"]
    san = [k for k in (recs[0]["metrics"] if recs else {}) if any(x in k for x in ("approx_kl", "ratio", "clipfrac", "entropy", "value_loss", "policy_loss", "grad_norm", "teacher_kl"))]
    for r in recs:
        m = r["metrics"]
        print(r["iteration"], "wall=%.2f native=%.2f calls=%d cpu=%.1f" % (r["wall_seconds"], r["native_step_seconds"], r["native_step_calls"], r["proc_cpu_seconds"]), " ".join("%s=%.3f" % (k.split("/")[-1], m[k]) for k in keys if k in m))
    later = recs[1:]
    if later:
        for k in keys:
            v = [r["metrics"][k] for r in later if k in r["metrics"]]
            if v: print("  median(iter2+)", k, "%.3f" % st.median(v))
        print("  median(iter2+) native_step_seconds %.3f wall %.3f" % (st.median(r["native_step_seconds"] for r in later), st.median(r["wall_seconds"] for r in later)))
    if recs:
        print("  iter1 sanity:", {k: round(recs[0]["metrics"][k], 6) for k in sorted(san)})
        if len(recs) > 1: print("  iter2 sanity:", {k: round(recs[1]["metrics"][k], 6) for k in sorted(san)})
    if recs: print("  all time/ keys:", sorted(k for k in recs[0]["metrics"] if k.startswith(("time/", "perf/"))))
