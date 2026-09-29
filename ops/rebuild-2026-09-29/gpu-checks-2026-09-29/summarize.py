"""Summarize the GPU checks bundle (attempt 2) into summary.json.

Reads only the retained records under pod/attempt2/ (and attempt1's c2 for the
stop record). Usage: python summarize.py  (from this directory).
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
A2 = HERE / "pod" / "attempt2"
A1 = HERE / "pod" / "attempt1"


def recs(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def c1() -> dict:
    out: dict = {}
    for case in ("c1_aten", "c1_default"):
        for r in recs(A2 / f"{case}.jsonl"):
            if r.get("event") != "result":
                continue
            key = f"{case}/{r['density']}"
            out[key] = {
                "present_tokens": r["present_tokens"],
                "row_tokens": [r["row_tokens_min"], r["row_tokens_max"]],
                "ref_eff_vs_math_max_abs": r["ref_eff_vs_ref_math"]["max_abs"],
                "compiled_repeat_max_abs": r["compiled_repeat_max_abs"],
                "vs_fp32": {
                    p: {k: m[k] for k in ("max_abs", "mean_abs", "rms", "outside_tol",
                                          "outside_tol_frac", "ref_abs_max",
                                          "rows_with_outliers")}
                    | {"outliers_by_ref_abs": [[b["ref_abs"][0], b["elements"],
                                                b["outliers"]] for b in m["by_ref_abs"]],
                       "worst": {k: m["worst"][0][k] for k in
                                 ("row", "token", "group", "channel", "ref", "out",
                                  "abs_diff", "ulps")}}
                    for p, m in r["vs_fp32"].items()
                },
                "pairwise_outside_tol_frac": {k: v["outside_tol_frac"]
                                              for k, v in r["pairwise"].items()},
                "pairwise_max_abs": {k: v["max_abs"] for k, v in r["pairwise"].items()},
                "outlier_jaccard": {k: v["jaccard"] for k, v in r["outlier_overlap"].items()},
                "verdict": r["verdict"]["result"],
                "ratio_to_median": {m: r["verdict"][m]["ratio_to_median"]
                                    for m in ("mean_abs", "outside_tol_frac")},
            }
    return out


def c2() -> dict:
    out: dict = {}
    for attempt, path in (("attempt1", A1 / "c2_aten_bwd.jsonl"),
                          ("attempt2", A2 / "c2_aten_bwd.jsonl")):
        for r in recs(path):
            if r.get("event") != "result":
                continue
            p = r["params_compiled_vs_eager"]
            fp = r["floor_params_eager2_vs_eager"]
            kb = p["blocks.0.attn.k.bias"]
            rec = {
                "rows": r["rows"], "packed_tokens": r["packed_tokens"],
                "trunk_call_M": r["compiled"]["trunk_call_M"],
                "out": r["out_compiled_vs_eager"],
                "dx": {k: r["dx_compiled_vs_eager"][k] for k in
                       ("rel_max", "rel_fro", "token_rel_l2_max", "tokens_rel_gt_0.5",
                        "nonfinite", "masked_nonzero")},
                "floor_dx_rel_max": r["floor_dx_eager2_vs_eager"]["rel_max"],
                "param_rel_max_excl_kbias": max(v["rel_max"] for k, v in p.items()
                                                if not k.endswith("k.bias")),
                "floor_param_rel_max_excl_kbias": max(
                    v["rel_max"] for k, v in fp.items() if not k.endswith("k.bias")),
                "kbias_rel_max": kb["rel_max"],
                "memory_compiled": r["compiled"]["memory"],
                "memory_eager": r["eager"]["memory"],
            }
            if "ref_max_abs" in kb:
                qb = p["blocks.0.attn.q.bias"]["ref_max_abs"]
                rec["kbias_abs"] = {k: kb[k] for k in ("max_abs_diff", "ref_max_abs",
                                                      "out_max_abs")}
                rec["qbias_ref_max_abs"] = qb
                rec["kbias_worst_over_qbias"] = max(
                    kb["max_abs_diff"], kb["ref_max_abs"], kb["out_max_abs"]) / qb
            out[f"{attempt}/{r['label']}/{r['spec']}"] = rec
    ctl = [r for r in recs(A2 / "driver.jsonl") if r.get("stage") == "c2_ctl_default_bwd"
           and r.get("status") != "launch"]
    out["control"] = {"status": ctl[-1]["status"], "rc": ctl[-1]["rc"],
                      "finding": ctl[-1]["findings"][0][:200]}
    return out


def c3() -> dict:
    out: dict = {}
    for case in ("c3_mid_aten", "c3_dense_aten", "c3_mid_default", "c3_dense_default"):
        r = [x for x in recs(A2 / f"{case}.jsonl") if x.get("event") == "result"][0]
        out[case] = {
            "joint_logp_row_mean": r["replay_256"]["joint_logp_row"]["mean"],
            **{k: {"logratio_mean": r[k]["logratio_per_row"]["mean"],
                   "logratio_abs_mean": r[k]["logratio_per_row"]["abs_mean"],
                   "logratio_abs_max": r[k]["logratio_per_row"]["abs_max"],
                   "event_max_abs_diff": r[k]["event_max_abs_diff"],
                   "values_max_abs_diff": r[k]["values_max_abs_diff"]}
               for k in ("replay_256", "replay_1024")},
            "compute_value_vs_sample": r["compute_value_256"]["max_abs_diff_vs_sample"],
            **{k: {"kl_row_mean": r[k]["kl_per_row"]["mean"],
                   "kl_row_max": r[k]["kl_per_row"]["max"],
                   "kl_event_min": r[k]["kl_event_min"],
                   "value_kl_mean": r[k]["value_kl_ce_minus_teacher_entropy"]["mean"]}
               for k in ("teacher_self_combined", "teacher_self_cached",
                         "teacher_perturbed_combined", "teacher_perturbed_cached")},
            "cached_vs_combined_kl_max_abs": [
                r["teacher_self_cached_vs_combined_kl_max_abs"],
                r["teacher_perturbed_cached_vs_combined_kl_max_abs"]],
            "loss_backward_1024": r["loss_backward_1024"],
            "memory": r["memory"],
        }
    return out


def c4() -> dict:
    out: dict = {}
    for d in ("mid", "dense"):
        r = json.loads((A2 / f"c4_{d}_aten.derived.json").read_text())
        out[d] = r["splits"]
        tim = [x for x in recs(A2 / f"c4_{d}_aten.jsonl") if x.get("event") == "timing"]
        out[f"{d}_memory"] = {t["workload"]: [t["max_allocated_gib"], t["max_reserved_gib"]]
                              for t in tim}
        out[f"{d}_first_call_s"] = {t["workload"]: t["first_call_s_incl_compile"]
                                    for t in tim}
    return out


def main() -> None:
    drv = [r for r in recs(A2 / "driver.jsonl") if r.get("status") != "launch"]
    summary = {
        "source": "8fde43cd7408c9c4f9147eeb8f916bd08f8266ed",
        "driver": [{k: r.get(k) for k in ("utc", "stage", "gpu", "status", "rc")}
                   for r in drv],
        "c1": c1(), "c2": c2(), "c3": c3(), "c4": c4(),
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("wrote summary.json")


if __name__ == "__main__":
    main()
