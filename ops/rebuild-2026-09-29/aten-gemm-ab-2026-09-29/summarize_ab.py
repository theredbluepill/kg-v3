"""Post-run summary of the ATEN-only GEMM A/B timing (written after launch).

Reads pod/bench_{default,aten}/results_{mid,dense}.json and prints, per
density and workload, median / p90 (CUDA events, ms), first-call seconds,
peak allocated/reserved GiB, and the conditional update wall
64*t_A + t_C + 16*t_B + t_D with the ATEN/default ratio. Writes
ab_summary.json next to this file. Pure JSON arithmetic.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
POD = HERE / "pod"


def load(backend: str, density: str) -> dict:
    return json.loads((POD / f"bench_{backend}" / f"results_{density}.json").read_text())


def main() -> None:
    summary: dict = {}
    for density in ("mid", "dense"):
        d = {b: load(b, density) for b in ("default", "aten")}
        rows = {}
        for w in "ABCD":
            r = {}
            for b in ("default", "aten"):
                x = d[b]["results"][w]
                r[b] = {"median_ms": x["median_ms"], "p90_ms": x["p90_ms"],
                        "min_ms": x["min_ms"], "max_ms": x["max_ms"],
                        "first_s": x["first_call_s_incl_compile"],
                        "alloc_gib": x["max_memory_allocated_gib"],
                        "reserved_gib": x["max_memory_reserved_gib"]}
            r["ratio_median"] = r["aten"]["median_ms"] / r["default"]["median_ms"]
            r["ratio_p90"] = r["aten"]["p90_ms"] / r["default"]["p90_ms"]
            rows[w] = r
        walls = {b: d[b]["derived"]["update_wall_s_median"] for b in ("default", "aten")}
        walls90 = {b: d[b]["derived"]["update_wall_s_p90"] for b in ("default", "aten")}
        summary[density] = {
            "workloads": rows,
            "update_wall_s_median": walls,
            "update_wall_ratio": walls["aten"] / walls["default"],
            "update_wall_s_sum_of_p90": walls90,
            "ceiling_sps_rank": {b: d[b]["derived"]["ceiling_sps_per_rank_median"]
                                 for b in walls},
            "share": {b: d[b]["derived"]["share"] for b in walls},
            "git_head": {b: d[b]["info"]["git_head"] for b in walls},
        }
    (HERE / "ab_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for density, s in summary.items():
        print(f"== {density}")
        for w, r in s["workloads"].items():
            print(f"  {w}: default med {r['default']['median_ms']:.3f} p90 "
                  f"{r['default']['p90_ms']:.3f} | aten med {r['aten']['median_ms']:.3f} "
                  f"p90 {r['aten']['p90_ms']:.3f} | ratio {r['ratio_median']:.4f} "
                  f"(p90 {r['ratio_p90']:.4f}) | first {r['default']['first_s']:.2f}/"
                  f"{r['aten']['first_s']:.2f}s | alloc {r['default']['alloc_gib']:.2f}/"
                  f"{r['aten']['alloc_gib']:.2f} res {r['default']['reserved_gib']:.2f}/"
                  f"{r['aten']['reserved_gib']:.2f}")
        w = s["update_wall_s_median"]
        c = s["ceiling_sps_rank"]
        print(f"  wall default {w['default']:.4f}s aten {w['aten']:.4f}s ratio "
              f"{s['update_wall_ratio']:.4f}; ceiling {c['default']:.0f} vs {c['aten']:.0f}")


if __name__ == "__main__":
    main()
