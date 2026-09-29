Verified commit `432abe0` read-only. All four r2 wording items are accurately resolved:

- **O5:** [results.md:108](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/results.md:108) and [Reference:44](/Users/poonszesen/kg-v3/cookbook/references/compiled-gemm-template-overflows-above-2-21-rows.md:44) correctly separate the all-buffer storage check for `INDEX_DTYPE` from the output-numel decision for size-argument dtype.
- **O7:** Both documents restrict attribution to retained, attributable kernels and explicitly preserve the unlocalized autotune failures for 768→256 and 512→256.
- **O11:** The [run statement:16](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/run-statements/gemm-limits-probe.md:16) correctly identifies Triton 3.5.0 as the lockfile pin; its installed version was not captured.
- **B2/O9:** The Reference uses `L_in = floor(2³¹/K)` and the strict `< 2³¹` design bound, consistent with the guard. Equality safety remains explicitly unmeasured.

The [log entry](/Users/poonszesen/kg-v3/cookbook/log.md:3) accurately summarizes these corrections. Source and retained-log spot checks support the wording. No new overclaim found in the changes; backward, IR-promotion and production-verification limits remain explicit.

No files changed or GPU runs launched.

VERDICT: APPROVE