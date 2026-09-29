Checked `ddf1fb2` against r1, including the [README](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/model-sps-ceiling-2026-09-29/README.md), [run statement addendum](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/run-statements/model-sps-ceiling.md:27), and [results.md](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/results.md:211).

1. **RESOLVED — Ceiling qualification.** Results are conditional estimates for the synthetic A/B/C/D schedule. Surrogate workloads, unavailable trainer compile integration, and missing numerical checks are explicitly disclosed.

2. **RESOLVED — Synchronization and attribution.** The documents acknowledge the timer’s explicit synchronization, describe A−D as incremental sampling-path cost, and withdraw heads/Muon attribution. `nsys` absence is correctly labeled operator-reported without a retained receipt.

3. **RESOLVED — Prelaunch and stopping claims.** The 563 MiB prelaunch reading, launch without the prescribed wait, and unenforced first-failure/aggregate limits are documented accurately. Logs confirm three successful exits over 227 seconds. These are corrected historical disclosures; the driver remains unchanged. The original pre-run text is byte-identical to `ed61770`.

4. **RESOLVED — Memory interpretation.** Allocated peak **40.275 GiB** is distinguished from reserved peak **84.994 GiB**. The allocation target is scoped to these components; carryover/fragmentation remains an untested hypothesis, and complete-trainer memory fit remains unqualified.

5. **RESOLVED — Variability.** Recalculation confirms sparse B’s **+0.818%** p90 difference and dense B’s **545.106 ms** tail sample. Both are disclosed. The weighted component-p90 sum is explicitly distinguished from a measured whole-update p90.

6. **RESOLVED — Method descriptions.** Counted trunk chunks versus calculated head chunks, the packed-token bound, and per-player joint clipping are corrected. The exact engine allowance **E ≤ M/9** and reported budgets reproduce.

**Raw evidence unchanged:** all 20 non-README artifacts—3 result JSONs, 4 logs, 10 receipts, and 3 scripts—are byte-identical across `cb4af49`, `ddf1fb2`, current HEAD, and the worktree. All **21/21 manifest entries** verify in each state; only the README’s manifest entry changed.

Read-only verification; no files edited or GPU workloads rerun. Approval covers the corrected component-timing receipts.

**VERDICT: APPROVE**