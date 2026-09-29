Independent verification of kg/merge-gpu-receipts, round 2

Reviewed HEAD `3f26e490582db12ae1295c591900e3b648d6a9a0` against base `ca370893eff25f98a1a00a0bbcde82750eff1c4c` using the three-dot diff. The initial working tree was clean. No tracked file or index entry was modified; this untracked evidence directory is the only deliverable. No GPU run was launched.

Findings, in severity order:

1. **P2 — `ops/rebuild-2026-09-29/gpu-checks-2026-09-29/scripts/launch.sh:15`: failed process queries can pass the idle gate.** `gpu_idle` ignores the exit status of the compute-process query. If that command fails with empty stdout while the utilization query returns 0, the function returns success and the launcher can begin work without establishing that the GPU has no existing compute processes. Zero utilization alone is insufficient: a resident learner can be between steps. Independently extracting the exact function and stubbing `nvidia-smi` produced exit 0 for this fault; busy-process, busy-utilization, both-query-failure and utilization-only-failure controls were rejected. **Fix:** check each query's exit status before treating its output as evidence of idleness; capture utilization before whitespace normalization so a pipeline cannot hide its query status. Add a compute-query-only failure regression. The one-line scratch `apps=$(...) || return 1` change rejects the demonstrated mutation. Preserve immutable as-run launchers and describe any post-run repair separately. Evidence: `additional_probes.py`, `additional-probes.json`, `proposed-fix-probes.json`.

2. **P3 — `ops/rebuild-2026-09-29/gpu-checks-2026-09-29/scripts/c2_trunk_bwd.py:87`: an eager-reference NaN can pass the backward oracle.** `cmp_dx` counts nonfinite values only in compiled gradients. Python `max(previous, NaN)` then discards NaN reductions for an affected chunk. With 513 valid rows, compiled gradients all 1 and one eager-reference NaN in the first 512-row chunk, it reports `nonfinite=0`, `rel_max=0` and zero bad tokens; `judge_c2` accepts the record despite NaN Frobenius/quantile metrics. A real 10% compiled-gradient error in that same chunk can also be hidden. **Fix:** reject nonfinite values in either operand before reductions and add a reference-NaN regression spanning `ROW_CHUNK`. Counting nonfinite values in both operands on a scratch copy makes the judge reject the reproduction. Every numeric field in the retained attempt1/attempt2 C2 result records is finite, so the recorded attempts do not show this defect firing. Evidence: `additional-probes.json`, `repro_c2_eager_nan.py`, `proposed-fix-probes.json`.

Prior-review disposition (the five findings in `/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/codex/verify-merge-gpu-receipts-r1.md`; the historical verdict file is preserved):

| Prior finding | Status | Evidence on current HEAD |
|---|---|---|
| 1, P2, worker exceptions permit Phase 2 and exit 0 | **RESOLVED** | `Driver.stream` catches stage exceptions, logs `driver_error`, sets failure 5 and runs bounded cleanup. Fresh child scenarios cover Phase 1 malformed backend JSON, Phase 1 launch failure with a TERM-resistant peer, and Phase 2 malformed JSON. All exit 5, suppress subsequent launches and leave no recorded stage PID alive. Removing the exception boundary in scratch reproduces exit 0 and a later launch. |
| 2, P3, two C3 teacher paths omit value checks | **RESOLVED** | All four teacher paths reject nonfinite, below-range and above-range student values. All four retained C3 records still pass. Removing the two added paths reproduces the false pass. |
| 3, P3, C1 zero median suppresses comparison | **RESOLVED** | Positive error beside two zero-error paths is classified path-specific; all-zero paths remain similar. All six retained verdicts regenerate unchanged. Reintroducing the median guard reproduces the false classification. |
| 4, P3, multi-GPU Decision says pending merge | **RESOLVED** | Current Decision cites local merged receipts with repository source metadata. Historical log wording remains appropriately historical. |
| 5, P3, compiled-GEMM and teacher limits omit GPU evidence | **RESOLVED** | Notes, source metadata, References index and log credit depth-1 ATEN backward and C3/C4 evidence while retaining depth-8, current production backend-claim wiring, cross-chunk/minibatch, multi-rank and trainer-integration limits. |

Merge preservation and semantic resolution:

- Merge `d793d16` has parents `ca37089` and `24380f3`, common ancestor `a78f609`, and five incoming GPU-check commits. All 113 paths changed by that merge match the GPU parent byte-for-byte. There were no divergent overlapping paths or unexpected merge-tree changes.
- The original resolution notes are correct for `d793d16`: 112 added paths plus one append-only `results.md` change, 16,325 insertions and zero deletions, all under `ops/rebuild-2026-09-29/`. The subsequent repair commit changes cookbook and current scripts; those notes are not a description of the complete current HEAD diff.
- `tests/`, `engine_rs/tests/`, `src/` and all integration production code remain identical to base. Named-test comparison finds no integration test lost. Three old GPU-parent encoder test names were already replaced in base by hidden-state/current-head, actual-GEMM-width and safe packed-chunking tests; this merge removes none. The added ops guard tests are separate from the normal pytest suites.
- Count reconciliation: the Python 3.12 AST/source scan has 1,257 named declarations in base/merge/HEAD and 775 in the GPU parent. The initial broad scan in `test-name-audit.json` counted another 167 historical copies under ops (1,424 total); the prior review's Python 3.9 scan skipped four declarations in a file using newer syntax (1,253/771). `test_name_audit.py` and both count-reconciliation JSON files record these differences. None changes the preservation result; declaration counts are not pytest execution counts.
- Every as-run `pod/attempt*/` file remains unchanged after the repair. All 114 manifest entries and 18 as-run script hashes verify. Scratch summary regeneration is byte-identical. All 24 C4 timing medians and six update-wall/global-SPS formulas independently recompute.
- `results.md` retains the complete integration prefix and has one GPU section after the ATEN-only section in chronological order. Engine trim manifest and parity docs were untouched. Declared source `8fde43c` resolves to tree `70d50fa37e66aa322e8db00c9359d3f9e3031499`; engine/build/lock inputs did not change from `e1458d2`, supporting the recorded native-extension reuse.
- Cookbook reconciliation is semantically sound within the stated measurement scope. The three changed concept notes pass direct shape/source checks and all repository source paths exist. Documentation freshness passes. These static checks do not establish GPU execution of current HEAD or runtime hook discovery.

Requested checks, all exit 0:

| Command | Result | Receipt |
|---|---|---|
| `cargo test --manifest-path engine_rs/Cargo.toml --locked --offline` | 69 passed: 41 + 9 + 19; 0 failed | `engine.log` |
| `cargo test` | 254 passed, 4 ignored, 0 failed | `rust.log` |
| `uv run python scripts/check_engine_trim.py` | engine trim manifest: OK | `trim.log` |
| `OMP_NUM_THREADS=2 uv run pytest tests/kaggriculture tests/owl tests/scripts tests/tools -m 'not slow' -q` | 1,690 passed, 11 skipped in 45.41 seconds; no sharding needed | `pytest.log` |
| `uv run mypy python/owl scripts` | 63 source files clean | `mypy.log` |

Scratch mutation/cleanup verification:

- 136 CPU oracle/guard assertions pass: retained baselines, C1–C4 record corruption, actual numerical tensor mutations, source/FlashAttention/compile guards, wrapper input/environment guards, template markers, summary format and recomputation, plus the repaired C1/C3 faults. Coverage includes each new oracle/guard family; CUDA computation itself was not rerun.
- Eleven driver control-flow probes pass, including stop/deadline, backend start/end, timeout, judge failure and expected control behavior. Independent malformed-record/launch-error probes now suppress Phase 2 and return 5.
- Current guard regression script passes all six checks (four driver scenarios and two numerical-guard groups). Three independent source regressions restore the old stage boundary, C3 omission and C1 zero-median defect and reproduce their expected failures.
- Six cleanup scenarios pass, including Phase 1, Phase 2, both deterministic spawn windows, normal completion and an as-run-driver positive control. Because `ps`/`pgrep` are unavailable in the sandbox, the test wrapper uses exact recorded PIDs, with no process-name sweep claim. GNU timeout is emulated. Every PID retained in the result logs was gone at final verification.
- Six launcher probes expose the compute-query-only false pass. The independent eager-reference tensor mutation exposes the C2 false pass. Targeted scratch repairs reject both. Every scratch bundle file matches its captured original bytes afterward, and tracked/index diffs remain empty (`restoration.json`).

Evidence belongs to the recorded source version, synthetic observations and fresh weights. It does not qualify current production PPO, multi-rank behavior or end-to-end throughput. No evidence indicates that the two newly found faults invalidated the retained runs, but the launcher’s fail-open preflight should be repaired before approval.

VERDICT: REJECT
