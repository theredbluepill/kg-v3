Independent verification of kg/merge-gpu-receipts, round 3

Reviewed HEAD `666deec789b2e75f56afb6dbb7b7acd40171c6b1` against base `ca370893eff25f98a1a00a0bbcde82750eff1c4c` using the three-dot diff. The initial tracked working tree and index were clean. This review changes no tracked file or index entry; its deliverables are untracked receipts in this directory. No GPU run was launched.

No new actionable findings. No required fix remains from r2.

Disposition of every finding in `/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/codex/verify-merge-gpu-receipts-r2.md` (the historical verdict is preserved):

| Finding | Status | Current location, implemented fix and independent evidence |
|---|---|---|
| r2 #1, P2: failed compute-process query passes the launcher idle gate | **RESOLVED** | `ops/rebuild-2026-09-29/gpu-checks-2026-09-29/scripts/launch.sh:19` and `:20` check both query exit statuses before using their output. Whitespace normalization follows capture. The seven-case Bash stub test accepts only successful empty-process/zero-utilization queries. Restoring the pre-r2 launcher fails the regression; independently removing either query's status check also fails it. |
| r2 #2, P3: eager-reference NaN passes C2 and hides a finite gradient error in its chunk | **RESOLVED** | `ops/rebuild-2026-09-29/gpu-checks-2026-09-29/scripts/c2_trunk_bwd.py:68`, `:93`, `:139` count nonfinite values in both operands; `:96` sanitizes the reference before gradient reductions. Tests span 513 rows across `ROW_CHUNK=512`, including the NaN plus 10% error reproduction. An additional 48 tensor faults cover NaN/+Inf/-Inf, either operand, either chunk, outputs/input gradients and ordinary/key-bias parameter gradients: every case is rejected. Restoring the pre-r2 comparator fails the regression. Removing each of the three new reference counts independently also fails it. |
| r1 #1, P2: worker exceptions allow Phase 2 and success | **RESOLVED** | Current stream boundary returns failure 5 and cleans up for malformed backend records and spawn failures; later stages do not launch. Removing the boundary on scratch reproduces exit 0 and Phase 2 launch. |
| r1 #2, P3: C3 omits values on two teacher paths | **RESOLVED** | All four teacher paths reject nonfinite/out-of-range values. Removing the two formerly omitted paths makes the regression fail; all four retained C3 records pass. |
| r1 #3, P3: C1 zero median suppresses the comparison | **RESOLVED** | A positive error beside two zero-error paths is classified path-specific; all-zero paths remain similar. Restoring the median-positive condition fails the regression. All six retained verdicts regenerate identically. |
| r1 #4, P3: multi-GPU Decision says evidence is pending merge | **RESOLVED** | The current Decision cites the merged local bundle and repository sources. Earlier log entries remain explicitly historical. |
| r1 #5, P3: compiled-GEMM/teacher notes omit GPU evidence | **RESOLVED** | Current notes and index credit depth-1 ATEN backward and C3/C4 measurements while preserving depth-8, production backend-claim wiring, teacher cross-chunk/minibatch equality, multi-rank and trainer-integration gaps. |

Mutation evidence for r2 is in `root/r2-probes.json` and its reproducible `root/r2_probes.py`. The seven source regressions there were all killed. Its 71 scratch files were restored byte-for-byte, checked again against captured SHA-256 values, then the duplicate scratch copy was removed. The hashes remain in the receipt.

Merge preservation and semantic resolution:

- Original merge `d793d16` has parents `ca37089` and `24380f3`, common ancestor `a78f609`, and exactly five incoming GPU-check commits. There are no overlapping changed paths between parents and no unexpected merge-tree paths. All incoming changes were preserved.
- The supplied resolution notes describe that original merge correctly: 113 changed paths, 16,325 inserted lines, zero deletions, all under `ops/rebuild-2026-09-29/`. More precisely, these are 112 new files and one append-only `results.md` change. They do not describe the complete current HEAD: r1/r2 repairs bring the three-dot diff to 198 files, including cookbook updates.
- Production source, configs, tests, engine trim manifest and parity docs remain identical to base. A Python 3.12 AST and Rust declaration scan finds 1,257 named live test declarations in base, original merge and HEAD (930 Python, 327 Rust); none was lost. The GPU parent has 775. Three older GPU-parent encoder test names were already replaced in integration base before this merge: deferred heads, the old width calculation and packed-overflow rejection. They were superseded by current-head/hidden-state, actual-width and packed-chunking coverage. No incoming test disappeared during this merge or either repair. Declaration counts are distinct from test execution counts.
- All 94 as-run files under `pod/attempt*/` remain identical to the GPU parent. Every one of the 116 bundle manifest entries and all 18 as-run script hashes verify, with no unlisted bundle file. Current scripts are explicitly distinguished from as-run versions.
- Summary regeneration on scratch is byte-identical. Independently reducing all 24 C4 timing medians and six rank-split update-wall/global-SPS calculations agrees with the retained results.
- `results.md` retains the integration prefix and a single GPU component section following the ATEN-only section in time order. The post-run repairs are disclosed without rewriting the recorded attempts or promoting component timings to complete-work throughput.
- Cookbook reconciliation preserves the evidence's measured scope and remaining gaps. Direct note shape/source validation and docs freshness checks pass. These checks do not prove runtime hook discovery, empirical GPU correctness on HEAD or production qualification.

Detailed merge/test inventories and custody checks are in `merge-docs/`.

Requested checks, all exit 0:

| Command | Result | Receipt |
|---|---|---|
| `cargo test --manifest-path engine_rs/Cargo.toml --locked --offline` | 69 passed (41 + 9 + 19), 0 failed | `checks/engine-cargo-test.log` |
| `cargo test` | 254 passed, 4 ignored, 0 failed | `checks/root-cargo-test.log` |
| `uv run python scripts/check_engine_trim.py` | engine trim manifest: OK | `checks/engine-trim.log` |
| `uv run pytest tests/kaggriculture tests/owl tests/scripts tests/tools -m 'not slow' -q` | 1,690 passed, 11 skipped, 0 failed in 48.87 s | `checks/pytest-full.log` |
| `uv run mypy python/owl scripts` | 63 source files clean | `checks/mypy.log` |

No pytest resource guard tripped and no sharding was needed. Existing skips retain native integration and hardware/backend limits. `checks/summary.json` records commands and log hashes.

Additional oracle/guard verification:

- 136 CPU assertions pass across retained baselines and mutated records/tensors: C1 reference finiteness, numerical comparison and classification; C2 outputs/input and parameter gradients, masking and key-bias exception; C3 sampling/replay, teacher paths, values, KL, loss and gradients; C4 loss/chunk/FlashAttention checks; source identity, CUDA/compile/FlashAttention guards; wrapper inputs/environment; template census markers; summary format and recomputation. Each new oracle/guard family has a discriminating mutation.
- The driver control-flow matrix passes 11/11 cases covering stop/deadline, backend start/end records, timeout, judge failure and the expected control. Malformed-record and spawn-error probes independently confirm failure propagation.
- Current `test_driver_guards.py` passes 8/8 groups: four driver scenarios, C3 values, C1 verdict, launcher idle and C2 nonfinite checks. `test_driver_cleanup.py` passes 6/6 scenarios, including Phase 1/Phase 2 timeouts, both deterministic spawn windows, ordinary completion and the old-driver positive control.
- Five additional source regressions are killed: old C1 median bypass, old C3 omissions, removed stream exception boundary and disabled signal deferral in each spawn window. Together with the seven root regressions, 12 source-regression cases demonstrate test sensitivity.
- The broader scratch audit restores all 117 bundle files byte-for-byte in both scratch copies and verifies that tracked source bytes still match. Exact recorded PIDs, including the spawn-boundary PID even when the stage has not written its own PID file, are checked after cleanup. All 13 IDs retained for the final cleanup/source-regression check are absent.

Limit: sandbox restrictions prevent `ps`/`pgrep` inventory, so cleanup verification uses exact recorded PIDs and process groups, with no process-name sweep claim. GNU timeout is emulated on macOS. Current repairs were CPU-tested here; no GPU execution or current production PPO qualification is claimed. The immutable GPU evidence belongs to `8fde43c`, synthetic inputs and fresh weights. Neither r2 defect is evidenced in the retained attempts.

VERDICT: APPROVE
