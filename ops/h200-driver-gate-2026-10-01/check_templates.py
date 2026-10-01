"""Supplementary generated-code census for the ATEN-only GEMM A/B.

For every per-case cache under RUN/inductor_cache/, count over all generated
.py files: wrappers (files with ``def call(``), extern GEMM calls
(``extern_kernels.<op containing mm>(``), Triton template kernel definitions
and launches (``triton_tem_``), and any text naming decompose-K, persistent /
TMA matmul templates or the contiguous-subgraph mm templates. Pure text
parsing; complements analyze_kernels.py revision 2 (unchanged). Writes
RUN/template_census.json and prints a table.

Usage: python check_templates.py RUN_DIR
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

EXTERN_MM = re.compile(r"extern_kernels\.(\w*mm\w*)\(")
TEM_DEF = re.compile(r"^def (triton_tem_\w+)\(", re.MULTILINE)
TEM_RUN = re.compile(r"(triton_tem_\w+)\.run\(")
MARKERS = {
    "decompose_k": re.compile(r"decompose_k", re.IGNORECASE),
    "persistent_tma": re.compile(r"persistent_tma|tma_descriptor|make_tensor_descriptor|experimental_descriptor"),
    "contiguous_subgraph": re.compile(r"contiguous_(mm|addmm)|mm_contiguous|addmm_contiguous"),
    "triton_mm_kernel_name": re.compile(r"\btriton_(mm|addmm|bmm)_\d+"),
}


def census(case_dir: Path) -> dict:
    out: dict = {"files": 0, "wrappers": 0, "extern_mm_calls": {},
                 "triton_tem_defs": 0, "triton_tem_launches": 0,
                 "marker_files": {k: [] for k in MARKERS}}
    for f in sorted(case_dir.rglob("*.py")):
        text = f.read_text(errors="replace")
        out["files"] += 1
        rel = str(f.relative_to(case_dir))
        if "def call(" in text:
            out["wrappers"] += 1
            for op in EXTERN_MM.findall(text):
                out["extern_mm_calls"][op] = out["extern_mm_calls"].get(op, 0) + 1
            out["triton_tem_launches"] += len(TEM_RUN.findall(text))
        out["triton_tem_defs"] += len(TEM_DEF.findall(text))
        for name, pat in MARKERS.items():
            if pat.search(text):
                out["marker_files"][name].append(rel)
    return out


def main() -> None:
    run = Path(sys.argv[1]).resolve()
    report = {d.name: census(d) for d in sorted((run / "inductor_cache").iterdir())
              if d.is_dir()}
    (run / "template_census.json").write_text(json.dumps(report, indent=1) + "\n")
    for case, c in report.items():
        markers = {k: len(v) for k, v in c["marker_files"].items()}
        print(f"{case}: files={c['files']} wrappers={c['wrappers']} "
              f"extern_mm={c['extern_mm_calls']} tem_defs={c['triton_tem_defs']} "
              f"tem_launches={c['triton_tem_launches']} marker_files={markers}")


if __name__ == "__main__":
    main()
