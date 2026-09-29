The A/B supports ATEN-only as a workaround for the observed GEMM-template failure on this stack. I found no blocking correctness or run-conduct issue. Reporting needs these edits:

1. **Keep “immune” narrowly scoped.** [results.md:261](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/results.md:261) should describe exclusion of the identified GEMM template paths, rather than imply universal compiled-kernel safety. FlexAttention generates its own Triton templates independently of this setting; static-shape correctness, `bmm` numerics, and real-trunk backward above the bound remain unmeasured.

2. **Label the slowdown explanation as inference.** The “because” attribution to lost fusion in [README.md:71](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/aten-gemm-ab-2026-09-29/README.md:71) and [results.md:292](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/results.md:292) exceeds the evidence. Generated code supports “consistent with lost fusion”; component totals without a timeline do not establish its contribution.

3. **Correct the numerical summaries.** [README.md:70](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/aten-gemm-ab-2026-09-29/README.md:70) understates the largest maximum: dense/default/D is **2.1362%** above its median, not ≤1.95%. Largest p90 deviation is **0.533924%**—use “about 0.53%” or “within 0.54%.” Dense/default derived wall is **8.561475 s**, rounding to **8.561 s**.

4. **Qualify memory and correct the orchestrator’s kernel count.** “Allocated memory unchanged” should be “overall peak approximately unchanged”: dense D falls **0.950041→0.862261 GiB**, while dense B stays approximately 40.27 GiB. See [README.md:72](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/aten-gemm-ab-2026-09-29/README.md:72). The trunk has **24 `bias_addmm` + 24 `mm` calls per wrapper**, totaling 48+48 across two wrappers; the repository table already reports this correctly.

The installed `.venv` source is Torch **2.9.0**, git **`0fabc3ba…`**, matching the pod’s recorded source revision. The requested exclusions are confirmed:

| Path | Installed-source evidence |
|---|---|
| Backend filter | [utils.py:1628](/Users/poonszesen/kg-v3/.venv/lib/python3.12/site-packages/torch/_inductor/utils.py:1628) parses the list; [utils.py:1664](/Users/poonszesen/kg-v3/.venv/lib/python3.12/site-packages/torch/_inductor/utils.py:1664) requires `TRITON`. |
| `mm`, persistent-TMA, decompose-K, contiguous choices | All sit inside the gate beginning at [mm.py:760](/Users/poonszesen/kg-v3/.venv/lib/python3.12/site-packages/torch/_inductor/kernel/mm.py:760). |
| `addmm`, persistent-TMA, contiguous choices | Gated at [mm.py:997](/Users/poonszesen/kg-v3/.venv/lib/python3.12/site-packages/torch/_inductor/kernel/mm.py:997). Decompose-K is added by `tuned_mm`. |
| `bmm` Triton template | Gated at [bmm.py:211](/Users/poonszesen/kg-v3/.venv/lib/python3.12/site-packages/torch/_inductor/kernel/bmm.py:211). |
| ATEN/cuBLAS path | ATEN remains admitted at [utils.py:2112](/Users/poonszesen/kg-v3/.venv/lib/python3.12/site-packages/torch/_inductor/utils.py:2112); [mm.py:581](/Users/poonszesen/kg-v3/.venv/lib/python3.12/site-packages/torch/_inductor/kernel/mm.py:581) documents `bias_addmm`’s cuBLASLt dispatch. |
| FlexAttention exception | Its forward and backward templates are appended independently at [flex_attention.py:348](/Users/poonszesen/kg-v3/.venv/lib/python3.12/site-packages/torch/_inductor/kernel/flex/flex_attention.py:348) and [line 776](/Users/poonszesen/kg-v3/.venv/lib/python3.12/site-packages/torch/_inductor/kernel/flex/flex_attention.py:776). |

These specific GEMM gates apply to static and dynamic shapes; that source conclusion does not establish static-shape numerical qualification.

Verification also confirmed:

- Commit `9904121` records the statement and scripts at **07:06:00Z**, before the **07:06:29Z** launch. Script bytes, receipt hashes and all **69 manifest entries** agree.
- Prelaunch receipts satisfy the idle rule. The driver implements first-failed-stage stopping, the 44-minute remaining-budget timeout, and the outer 45-minute timeout. Recorded completion is **503.9 s**, exit 0.
- The trunk guard is genuinely bypassed, with one full packed call per target. Comparisons cover every valid token/channel; synthetic forward and dX cover every element. Parameter gradients use the declared **tensorwise** `max|Δ|/max|ref|` metric. Both controls reproduce the failure.
- Revision-2 analyzer and census reports agree: ATEN caches contain zero template definitions/launch sites and retain other Triton fusions. **Generated caches remain pod-only**, so this review checked the analyzer and retained reports, not a fresh recount of those files.
- All **320 timing samples** reproduce the saved statistics. Conditional totals are **4.666633→4.969930 s (+6.4993%)** and **8.561475→8.943540 s (+4.4626%)**. The reported rounded SPS values are correct.

Adopting ATEN-only as an infrastructure workaround is supported. Production qualification and removing guards everywhere are not established by this probe. No files changed or GPU work launched.

**VERDICT: APPROVE WITH EDITS**