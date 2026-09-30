The merge preserves both parents’ intended contents, but mutation checks exposed a false-success bug.

Findings:

- **P2 — [driver.py:450](/Users/poonszesen/kg-v3-m-gpu-receipts/ops/rebuild-2026-09-29/gpu-checks-2026-09-29/scripts/driver.py:450):** Malformed backend JSON or a process-launch exception kills a worker, yet Phase 2 runs and the driver returns **0**. Catch worker exceptions, propagate failure, clean up owned processes and prevent subsequent stages. Add both regression tests.
- **P3 — [C3 value guard:202](/Users/poonszesen/kg-v3-m-gpu-receipts/ops/rebuild-2026-09-29/gpu-checks-2026-09-29/scripts/driver.py:202):** Two teacher paths omit value validation. Injected nonfinite/out-of-range values pass. Check all four teacher paths.
- **P3 — [C1 classifier:151](/Users/poonszesen/kg-v3-m-gpu-receipts/ops/rebuild-2026-09-29/gpu-checks-2026-09-29/scripts/c1_fp32_ref.py:151):** A zero median suppresses the declared comparison, labelling a single erroneous path “similar.” Apply `v > 2 * med` even when `med == 0`.
- **P3 — [multi-GPU Decision:45](/Users/poonszesen/kg-v3-m-gpu-receipts/cookbook/decisions/start-multi-gpu-qualification-with-two-ranks.md:45):** Evidence still says “pending merge.” Update wording and source metadata to the local receipts.
- **P3 — [compiled-GEMM Reference:70](/Users/poonszesen/kg-v3-m-gpu-receipts/cookbook/references/compiled-gemm-template-overflows-above-2-21-rows.md:70):** Current limits omit the newly measured depth-1 backward results. Credit that scope while retaining depth-8 and production-integration gaps; similarly reconcile the teacher Reference’s blanket GPU-unmeasured wording.

All requested checks passed:

| Check | Result |
|---|---|
| Engine Cargo, locked/offline | 69 passed |
| Root Cargo | 254 passed, 4 ignored |
| Engine trim | OK |
| Requested pytest suites | 1,690 passed, 11 skipped |
| Mypy | 63 files clean |

No merge loss found: all 113 incoming paths match the GPU parent; integration tests are unchanged. Three older test names were already intentionally replaced in base. Receipt hashes and regenerated summaries match.

Scratch verification completed 142 expected-behavior assertions and six cleanup scenarios; **five additional fault mutations survived**, producing findings 1–3. Scratch sources matched byte-for-byte afterward. Tracked working tree remains clean. No GPU run was launched.

[Full report and evidence](/private/tmp/verify-merge-gpu-Y3LygR/review.md)

VERDICT: REJECT