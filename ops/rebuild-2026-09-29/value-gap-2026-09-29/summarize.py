"""Summarize the value-gap diagnostic and apply the pre-declared rules.

Usage: python3 summarize.py POD_DIR [FRERUN_DIR] > summary.json
POD_DIR holds attempt 3's stage JSONL files copied from the pod run dir;
FRERUN_DIR the Amendment 3 rerun of the compiled Isaiah stages. With it,
prediction 5 uses the rerun's compiled cells (attempt 3's 1,024-row compiled F
cells ran eagerly after Dynamo's recompile limit). The rules are
those of run-statements/value-gap-diagnostic.md, "Pre-declared discriminating
observations"; nothing here changes a threshold.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROWS = (256, 1024)


def result(d: Path, name: str) -> dict[str, Any] | None:
    p = d / f"{name}.jsonl"
    if not p.exists():
        return None
    res = [json.loads(x) for x in p.read_text().splitlines() if x]
    res = [r for r in res if r.get("event") == "result"]
    return res[0] if res else None


def g(r: dict[str, Any] | None, *keys: Any) -> Any:
    for k in keys:
        if r is None:
            return None
        r = r.get(k)
    return r


def first(*vals: Any) -> Any:
    """The first value that is not None (0.0 is a valid value)."""
    for v in vals:
        if v is not None:
            return v
    return None


def verdict(h1: bool, h2: bool) -> str:
    if h1 and not h2:
        return "H1"
    if h2 and not h1:
        return "H2"
    return "inconclusive"


def main() -> None:
    d = Path(sys.argv[1])
    names = ["kgA_bf16_comp_aten", "kgA_bf16_comp_default", "kgB_bf16_eager",
             "kgC_fp32_comp_aten", "kgC_fp32_eager", "kgC_fp32_comp_default",
             "isF_eager", "isF_comp_default", "isF_comp_aten"]
    R = {n: result(d, n) for n in names}
    frerun = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    if frerun is not None:
        for n in ("isF_comp_default", "isF_comp_aten"):
            R[f"{n}_frerun"] = result(frerun, n)
    table: dict[str, Any] = {}
    for n, r in R.items():
        if r is None:
            table[n] = None
            continue
        row: dict[str, Any] = {}
        for k in ROWS:
            dg = r.get(f"default_gain_{k}", {})
            hid = r.get(f"hidden_{k}", {})
            cell = {
                "V_max": g(dg, "pair", "values", "max_abs"),
                "V_mean": g(dg, "pair", "values", "mean_abs"),
                "critic_logit_diff_max": first(
                    g(dg, "pair", "critic_logit_diff", "max_abs"),
                    g(dg, "pair", "winner_logp_centered", "max_abs")),
                "L_max": first(g(dg, "pair", "logp_event", "max_abs"),
                               g(dg, "pair", "logp_per_player_entity", "max_abs")),
                "logratio_row_absmax": first(
                    g(dg, "pair", "logratio_row", "abs_max"),
                    g(dg, "pair", "logratio_per_player", "abs_max")),
                "logp_per_player_max": g(dg, "pair", "logp_per_player", "max_abs"),
                "eval_nograd_vs_sample_V_max": g(dg, "eval_nograd_vs_sample", "pair",
                                                 "values", "max_abs")
                if "pair" in (dg.get("eval_nograd_vs_sample") or {})
                else g(dg, "eval_nograd_vs_sample", "values", "max_abs"),
                "compute_value_grad_vs_sample_V_max": g(
                    dg, "compute_value_grad_vs_sample", "values", "max_abs"),
                "compute_value_nograd_vs_sample_V_max": g(
                    dg, "compute_value_nograd_vs_sample", "values", "max_abs"),
                "replay_grad_repeat_V_max": g(dg, "replay_grad_repeat", "values", "max_abs"),
                "sample_repeat_V_max": g(dg, "sample_nograd_repeat", "values", "max_abs"),
                "sample_repeat_actions_equal": g(dg, "sample_nograd_repeat",
                                                 "actions_equal"),
                "HS_bf16_head": g(hid, "head_swap_values", "max_abs"),
                "HS_fp32_head": g(hid, "head_swap_fp32_head_values", "max_abs"),
            }
            for grp in ("all", "critic", "plan", "own_actor"):
                for cmp in ("grad_vs_nograd", "nograd_vs_fp32", "grad_vs_fp32"):
                    m = g(hid, grp, cmp)
                    if m:
                        cell[f"h_{grp}_{cmp}"] = {k2: m[k2] for k2 in (
                            "max_abs", "mean_abs", "rel_mean", "exact_equal_frac")}
                m = g(hid, grp)
                if m and "max_abs" in m:  # Isaiah hidden: plain diff per group
                    cell[f"h_{grp}_grad_vs_nograd"] = {k2: m[k2] for k2 in (
                        "max_abs", "mean_abs", "rel_mean", "exact_equal_frac")}
            if hid.get("trunk_input"):
                cell["trunk_input_max"] = hid["trunk_input"]["max_abs"]
            g1 = r.get(f"actor_gain1_{k}")
            if g1:
                cell["gain1_L_max"] = g(g1, "pair", "logp_event", "max_abs")
                cell["gain1_V_max"] = g(g1, "pair", "values", "max_abs")
                cell["gain1_logratio_row_absmax"] = g(g1, "pair", "logratio_row", "abs_max")
            st = r.get(f"states_{k}")
            if st:
                cell["states"] = st
            row[str(k)] = cell
        row["dynamo_counters"] = {k: v for k, v in (r.get("dynamo_counters") or {}).items()
                                  if k in ("stats", "aot_autograd", "inductor")} \
            if isinstance(r.get("dynamo_counters"), dict) else r.get("dynamo_counters")
        row["compiled_trunk_calls"] = r.get("compiled_trunk_calls")
        row["use_flash_attn_flags"] = r.get("use_flash_attn_flags")
        row["output_layer_spectral_norms"] = r.get("output_layer_spectral_norms")
        row["memory"] = r.get("memory")
        table[n] = row

    def cell(n: str, k: int) -> dict[str, Any]:
        return (table.get(n) or {}).get(str(k)) or {}

    preds: dict[str, Any] = {}
    A = "kgA_bf16_comp_aten"
    # 0 reproduction
    rep = {n: [0.005 <= (cell(n, k).get("V_max") or 0) <= 0.05 for k in ROWS]
           for n in (A, "kgA_bf16_comp_default") if table.get(n)}
    preds["0_reproduction"] = rep
    # 1 eager BF16
    p1 = {}
    for k in ROWS:
        a, b = cell(A, k).get("V_max"), cell("kgB_bf16_eager", k).get("V_max")
        if a is None or b is None:
            continue
        p1[k] = {"V_A": a, "V_B": b,
                 "verdict": verdict(b <= 1e-3 and b <= 0.1 * a, b >= 0.25 * a)}
    preds["1_eager_bf16"] = p1
    # 2 fp32
    p2 = {}
    for k in ROWS:
        a = cell(A, k).get("V_max")
        c = cell("kgC_fp32_comp_aten", k).get("V_max")
        e = cell("kgC_fp32_eager", k).get("V_max")
        cd = cell("kgC_fp32_comp_default", k).get("V_max")
        if None in (a, c, e):
            continue
        p2[k] = {"V_A": a, "V_fp32_compiled_aten": c, "V_fp32_eager": e,
                 "V_fp32_compiled_default": cd,
                 "verdict": verdict(c <= 1e-3 and e <= 1e-4 and (cd is None or cd <= 1e-3),
                                    max(c, e, cd or 0) >= 0.25 * a)}
    preds["2_fp32"] = p2
    # 3 actor gain 1.0
    p3 = {}
    for n in (A, "kgA_bf16_comp_default"):
        for k in ROWS:
            l0, l1 = cell(n, k).get("L_max"), cell(n, k).get("gain1_L_max")
            le = cell("kgB_bf16_eager", k).get("gain1_L_max")
            if None in (l0, l1):
                continue
            ratio = l1 / l0 if l0 else float("inf")
            p3[f"{n}_{k}"] = {"L_gain0.01": l0, "L_gain1": l1, "ratio": ratio,
                              "eager_L_gain1": le,
                              "verdict": verdict(ratio >= 10 and l1 >= 5e-3
                                                 and (le is None or le <= 1e-3),
                                                 ratio < 3)}
    preds["3_actor_gain"] = p3
    # 4 hidden states
    p4 = {}
    for n in (A, "kgA_bf16_comp_default"):
        for k in ROWS:
            c = cell(n, k)
            try:
                gn = c["h_critic_grad_vs_nograd"]["mean_abs"]
                nf = c["h_critic_nograd_vs_fp32"]["mean_abs"]
                gf = c["h_critic_grad_vs_fp32"]["mean_abs"]
                rc = c["h_critic_grad_vs_nograd"]["rel_mean"]
                ra = c["h_all_grad_vs_nograd"]["rel_mean"]
                hs, v = c["HS_bf16_head"], c["V_max"]
            except (KeyError, TypeError):
                continue
            i = gn <= 2 * nf
            ii = 0.5 * nf <= gf <= 2 * nf
            iii = hs >= 0.5 * v
            iv = rc <= 3 * ra
            h2 = (not iii) or (not iv) or gf > 2 * nf
            p4[f"{n}_{k}"] = {"crit_grad_vs_nograd_mean": gn, "crit_nograd_vs_fp32_mean": nf,
                              "crit_grad_vs_fp32_mean": gf, "crit_rel_mean": rc,
                              "all_rel_mean": ra, "HS": hs, "V": v,
                              "i": i, "ii": ii, "iii": iii, "iv": iv,
                              "verdict": verdict(i and ii and iii and iv, h2)}
    preds["4_hidden"] = p4
    # 5 Isaiah control
    p5 = {}
    for k in ROWS:
        kg_hs = cell(A, k).get("HS_bf16_head")
        eager_hs = cell("isF_eager", k).get("HS_bf16_head")
        f_names = (("isF_comp_default_frerun", "isF_comp_aten_frerun")
                   if frerun is not None else ("isF_comp_default", "isF_comp_aten"))
        comp = {n: cell(n, k).get("HS_bf16_head") for n in f_names}
        if kg_hs is None or eager_hs is None or None in comp.values():
            continue
        h1 = any(v >= 0.25 * kg_hs for v in comp.values()) and eager_hs <= 1e-3
        h2 = all(v < 0.25 * kg_hs for v in comp.values())
        p5[k] = {"kg_HS_compiled_aten": kg_hs, "is_HS_eager": eager_hs,
                 **{f"is_HS_{n}": v for n, v in comp.items()},
                 "is_V_pair": {n: cell(n, k).get("V_max") for n in
                               ("isF_eager", *f_names)},
                 "verdict": verdict(h1, h2)}
    preds["5_isaiah"] = p5
    json.dump({"table": table, "predictions": preds}, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
