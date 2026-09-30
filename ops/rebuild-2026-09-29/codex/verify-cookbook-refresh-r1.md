Seven edits are needed:

1. **P2 — SPS comparison cannot establish overhead attribution.** [SPS Reference:52](/Users/poonszesen/kg-v3/cookbook/references/model-only-sps-ceiling-bounds-per-rank-throughput.md:52) claims comparison with complete-work SPS reveals engine/host/all-reduce cost. Surrogate losses and uniform densities prevent that attribution without matched trainer phase measurements. Remove the exclusive “only if” explanation for exceeding the ceiling.

2. **P2 — BF16 spacing claim exceeds evidence.** [Flash Reference:25](/Users/poonszesen/kg-v3/cookbook/references/pod-v3-environment-runs-flash-attn-2-8-3-forward-on-sm120.md:25), its description, and the index say agreement is within BF16 spacing. [Results:157](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/results.md:157) explicitly says the maximum-error element’s magnitude was not retained. Keep the measured maximum difference and tolerance results; remove the spacing claim.

3. **P2 — Qualify process-termination causality.** [Plan:129](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/plan.md:129) says the parent agent’s return killed the child. The cited transcripts establish incomplete reviews, without termination or parent-lifecycle evidence. Label the cause operator-reported/inferred, or cite an orchestration receipt.

4. **P2 — Record the execution-rule adaptation.** Commit `9bfa0a0` changes only the plan. Its reusable stdin, foreground and recovery rules lack the cookbook note/index/log record required by [AGENTS.md:130](/Users/poonszesen/kg-v3/AGENTS.md:130).

5. **P3 — Preserve sampled custody scope.** [Flash Reference:40](/Users/poonszesen/kg-v3/cookbook/references/pod-v3-environment-runs-flash-attn-2-8-3-forward-on-sm120.md:40) says all 19,796 files share inodes with the new venv through uv’s cache. Evidence establishes hard-linked files, but cross-venv identity was checked for two samples; cache causation remains inferred.

6. **P3 — Correct manifest wording.** [SPS Reference:57](/Users/poonszesen/kg-v3/cookbook/references/model-only-sps-ceiling-bounds-per-rank-throughput.md:57) should say **21/21 checksums verified; 20 non-README artifacts unchanged**. The README and its checksum changed.

7. **P3 — Specify actor heads.** [Flash Reference:31](/Users/poonszesen/kg-v3/cookbook/references/pod-v3-environment-runs-flash-attn-2-8-3-forward-on-sm120.md:31) says “the heads did not exist.” The critic head existed at `69397da`; only actor heads were absent.

Checks passed: cookbook lint on all five changed concept notes; required metadata and all 34 `repository:` sources; manifests **30/30** and **21/21**; SPS arithmetic; prepended log preservation. Task 0.3 and historical-path corrections match receipts. Decision edits distinguish implementation metrics from owner wording. No Lesson was introduced, and no specified in-flight lane file was touched. Review remained read-only.

**VERDICT: APPROVE WITH EDITS**