"""Summarize generated Inductor code per case: which GEMM backend each compiled
graph used (Triton template vs extern cuBLAS), and each template kernel's size-arg
dtypes and 32-bit-prone index lines. Reads only the per-case cache dirs.

Revision 2 (after Codex verify-gemm-limits-r1 finding 4). Revision 1 took the
first ``'signature'`` in the whole file for every template (usually a pointwise
kernel's ``xnumel: i32``), printed only the first variant of each template name,
and its extern regex missed ``extern_kernels.bias_addmm``. This revision:

- attributes to each ``def triton_*`` the ``triton_meta`` signature that sits in
  the same kernel source block (between the previous kernel's ``def`` and this
  one), and its index lines only from that kernel's body;
- keeps every distinct variant (name, signature, index lines) with the files it
  appears in;
- counts every ``extern_kernels.<op>(`` call whose op name contains ``mm``.

Pure text parsing; no torch/triton import.

Usage: ``python analyze_kernels.py [RUN_DIR]``. RUN_DIR holds ``inductor_cache/``; it
defaults to this file's directory, as laid out on the pod. The local rerun used
``python3 probe/analyze_kernels.py pod-run > pod-run/kernel_analysis.txt``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parent
SIG = re.compile(r"'signature': \{([^}]*)\}")
KERNEL_DEF = re.compile(r"^def (triton_\w+)\(", re.MULTILINE)
EXTERN_MM = re.compile(r"extern_kernels\.(\w*mm\w*)\(")
TEMPLATE_RUN = re.compile(r"(triton_tem_\w+)\.run\(([^\n]*)\)")
INDEX_TOKENS = ("xindex =", "INDEX_DTYPE", "rm = ", "rn = ", "idx_m = ", "idx_n = ",
                "tl.load(A", "tl.load(B", "tl.store(", "pid_m = ", "group_size",
                ".to(tl.int64)")


def kernel_blocks(text: str) -> list[tuple[str, str, str]]:
    """Return (name, signature, body) for every Triton kernel defined in ``text``.

    The signature is the last ``'signature': {...}`` between the previous kernel
    definition and this ``def``; the body runs to the next ``'''`` (end of the
    embedded source in a wrapper) or the next kernel ``def``.
    """
    defs = list(KERNEL_DEF.finditer(text))
    out = []
    for i, m in enumerate(defs):
        start_hdr = defs[i - 1].end() if i else 0
        sigs = list(SIG.finditer(text, start_hdr, m.start()))
        sig = sigs[-1].group(1) if sigs else None
        end = defs[i + 1].start() if i + 1 < len(defs) else len(text)
        close = text.find("'''", m.end(), end)
        body = text[m.start(): close if close != -1 else end]
        out.append((m.group(1), sig, body))
    return out


def template_meta(sig: str | None, body: str) -> dict:
    """Size-arg dtype, INDEX_DTYPE, and the A-load / output row strides of a template."""
    ks0 = re.search(r"'ks0': '(\w+)'", sig or "")
    idx = re.search(r"INDEX_DTYPE : tl.constexpr = tl\.(\w+)", body)
    strides = re.findall(r"xindex = idx_n \+ (\d+)\*idx_m", body)
    return {"ks0": ks0.group(1) if ks0 else "none (static M)",
            "index_dtype": idx.group(1) if idx else None,
            "a_row_stride": int(strides[0]) if strides else None,
            "out_row_stride": int(strides[-1]) if strides else None}


def main() -> None:
    report: dict = {}
    for case_dir in sorted((RUN / "inductor_cache").iterdir()):
        wrappers = []
        variants: dict[tuple, dict] = {}
        for f in sorted(case_dir.rglob("*.py")):
            rel = str(f.relative_to(RUN))
            text = f.read_text(errors="replace")
            blocks = kernel_blocks(text)
            is_wrapper = "def call(" in text
            local_sig: dict = {}
            local_meta: dict = {}
            for name, sig, body in blocks:
                local_sig[name] = sig
                local_meta[name] = template_meta(sig, body)
                if not name.startswith("triton_tem_"):
                    continue
                lines = tuple(ln.strip() for ln in body.splitlines()
                              if any(tok in ln for tok in INDEX_TOKENS))
                key = (name, sig, lines)
                v = variants.setdefault(key, {"name": name, "signature": sig,
                                              "index_lines": list(lines), "files": []})
                v["files"].append(rel + (" (wrapper)" if is_wrapper else " (kernel)"))
            if is_wrapper:
                launches = [{"template": n, "signature": local_sig.get(n), "args": a,
                             **local_meta.get(n, {})}
                            for n, a in TEMPLATE_RUN.findall(text)]
                wrappers.append({
                    "file": rel,
                    "extern_gemm": sorted(EXTERN_MM.findall(text)),
                    "template_launches": launches,
                })
        report[case_dir.name] = {"wrappers": wrappers, "template_variants": list(variants.values())}
    out = RUN / "kernel_analysis.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    for case, info in report.items():
        print(f"== {case}: {len(info['wrappers'])} wrappers, "
              f"{len(info['template_variants'])} distinct template variants")
        for w in info["wrappers"]:
            print(f"   wrapper {w['file']}")
            print(f"      extern GEMM calls: {w['extern_gemm'] or 'none'}")
            counts: dict[tuple, int] = {}
            for t in w["template_launches"]:
                key = (t["template"], t.get("ks0"), t.get("index_dtype"),
                       t.get("a_row_stride"), t.get("out_row_stride"))
                counts[key] = counts.get(key, 0) + 1
            for (name, ks0, idt, a_st, o_st), n in counts.items():
                print(f"      template x{n} {name}: ks0={ks0} INDEX_DTYPE={idt} "
                      f"A-load {a_st}*idx_m, store {o_st}*idx_m")
        for v in info["template_variants"]:
            print(f"   variant {v['name']} sig={{{v['signature']}}}  [{len(v['files'])} file(s)]")
            for ln in v["index_lines"]:
                if any(tok in ln for tok in ("xindex =", "INDEX_DTYPE", "tl.load(A", "tl.load(B")):
                    print(f"      {ln}")
    sys.stdout.flush()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        RUN = Path(sys.argv[1]).resolve()
    main()
