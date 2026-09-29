Independent verification of kg/merge-gpu-receipts

Reviewed HEAD d793d16f23ee529f9fc49120558c8e9731b843f1 against ca370893eff25f98a1a00a0bbcde82750eff1c4c...HEAD. No tracked changes were made.

Findings, in severity order:

1. P2 — ops/rebuild-2026-09-29/gpu-checks-2026-09-29/scripts/driver.py:450 (also 420 and 455–466): worker exceptions produce false success. On a scratch copy, corrupt backend JSON raises JSONDecodeError in the Phase 1 worker; an injected process-spawn OSError does the same. Neither sets failure or stop. main() joins the failed worker, launches the next Phase 2 stage, and returns 0. This violates the driver's first-unexpected-failure stopping contract. Catch and propagate exceptions at the worker boundary, log the failed stage, set a nonzero result, clean up owned stage groups with the bounded cleanup routine, and prevent later launches. Add regressions for both cases. Preserve immutable pod/attempt*/ scripts and logs. Evidence: oracle-probes/driver-mutations.json and driver-mutations.log. The retained completed attempts do not exhibit this failure.

2. P3 — ops/rebuild-2026-09-29/gpu-checks-2026-09-29/scripts/driver.py:202: C3 value validation omits teacher_self_cached.student_values and teacher_perturbed_combined.student_values. Mutating either retained result to nonfinite=1 and min/max outside [-1,1] leaves judge_c3 returning no errors. Check all four teacher result paths. Evidence: independent-oracle-repro.json and oracle-probes/mutations.json. Actual retained values remain valid; this is a guard coverage defect.

3. P3 — ops/rebuild-2026-09-29/gpu-checks-2026-09-29/scripts/c1_fp32_ref.py:151: the additional `if med` condition disables the declared >2×median classifier when median is zero. With positive compiled error and zero eager/padded error, verdict() returns similar. Compare v > 2*med even at zero; keep divide-by-zero handling confined to ratio reporting. Add the zero-median regression without rewriting as-run evidence. Evidence: independent-oracle-repro.json.

4. P3 — cookbook/decisions/start-multi-gpu-qualification-with-two-ranks.md:45 (metadata line 9): the living Decision still calls the GPU bundle pending merge and uses pending-merge-branch provenance. The cited evidence is now in this tree. Replace those with current repository provenance while preserving historical log entries.

5. P3 — cookbook/references/compiled-gemm-template-overflows-above-2-21-rows.md:70 (also 53 and 76): current limits still say real-trunk backward is unmeasured/unverified above the bound. Merged results.md:353–370 compares output, dX and parameter gradients for a depth-1 real block plus final norm at 4,194,305 and 4,198,400 packed tokens under ATEN. Credit that exact scope, retaining depth-8, production backend-claim wiring, broader stack/shape and guard-retirement gaps. Likewise scope the teacher Reference's blanket CPU-only/GPU-unmeasured wording at cookbook/references/kaggriculture-teacher-distills-per-slot-kl-and-per-seat-winner-ce.md:147 and :150 to its original checks and link the new C3/C4 evidence. Cross-chunk/minibatch equality, multi-rank and full trainer integration remain open. Reconcile affected note metadata/index/log under the cookbook contract.

Merge custody and test preservation:

- Parents: ca370893eff25f98a1a00a0bbcde82750eff1c4c and 24380f3eb0e9c7090d874f8f1a45c0d4e3477f40; common ancestor a78f609e36d40d5254ff909066fe837f2d2540be. Exactly five incoming commits, 5f2ee2d/a4c75e0/73b822a/6392160/24380f3.
- 113 changed paths, all under ops/rebuild-2026-09-29/: 112 added files and one append-only results.md modification, 16,325 insertions, zero deletions. No production source, config, cookbook, docs, lockfile, test or engine manifest changed from integration.
- Every incoming changed path is byte-identical to the GPU parent. results.md itself equals the GPU parent's file; all pre-existing integration lines remain. Its six sections occur once and in chronological order; GPU checks follow ATEN-only A/B.
- Named-test scan (Python AST and Rust test attributes, excluding ops): base/HEAD each 1,253 names, GPU parent 771. No base test name lost. The three older GPU-parent names absent in HEAD were already replaced in integration: hidden-state/deferred-heads -> hidden-state plus dones rejection after heads landed; gemm_kmax -> actual Linear width enumeration and preset-width test; packed-overflow rejection -> safe chunking boundary/order/equality/unfittable-row tests. Those replacements match the earlier implementation changes, and this merge changes none of them. Incoming cleanup tests are byte-identical to the GPU parent.
- 110 retained manifest hashes and 18 as-run script hashes match. All six numeric summary sections independently regenerate identically. C1 coordinates match aggregates/buckets/token groups. C4's 24 component measurements (20 samples each), six update formulas/SPS and chunk counts agree; maximum p90/median is 1.01385. No receipt numerical mismatch found.
- GPU evidence belongs to source 8fde43c, synthetic observations and fresh weights; it does not certify this integration's production compile wiring or complete PPO training. Source/attempt separation, 0.25*MSE deviation, default-control compile failure, absent Nsight and unmeasured runtime scope are disclosed. Rust/Cargo/lock inputs were unchanged between e1458d2 and 8fde43c, supporting native-extension reuse.

Requested checks (all exit 0, executed in this worktree):

| Command | Result | Receipt |
|---|---|---|
| cargo test --manifest-path engine_rs/Cargo.toml --locked --offline | 69 passed (41+9+19), 0 failed | engine.log |
| cargo test | 254 passed, 4 ignored, 0 failed | rust.log |
| uv run python scripts/check_engine_trim.py | engine trim manifest: OK | trim.log |
| uv run pytest tests/kaggriculture tests/owl tests/scripts tests/tools -m 'not slow' -q | 1,690 passed, 11 skipped in 47.35s; no sharding needed | pytest.log |
| uv run mypy python/owl scripts | no issues in 63 source files | mypy.log |

Python module paths were checked: model, PPO and native extension resolve inside this worktree. No CUDA/GPU run was launched during this verification. No tracked code edits occurred, so formatting/prepare mutations were unnecessary. Optional git diff --check reports whitespace in immutable raw logs; receipts were preserved byte-for-byte.

Mutation and cleanup checks:

- 127 general CPU/scratch assertions passed, including nine real-receipt baselines, C1–C4 criteria, output/dX/parameter comparisons, source/FlashAttention/compile guards, wrapper input/environment guards, template census, summary regeneration and summary-format guard.
- 4 launcher idle-gate probes passed using an extracted shell function and stubbed nvidia-smi: idle baseline, existing compute process, nonzero GPU utilization and query failure. No launcher or GPU was executed; original bytes are unchanged (launcher-idle-probes.json).
- 11 driver control-flow assertions passed: baseline, prior stop, deadline, missing/wrong-start/wrong-end backend records, nonzero exit, timeout, judge exception/error and expected-control behavior. Environment contamination stripping was also asserted. GPU and subprocess launch effects were safely stubbed for these probes.
- Two worker-error mutations survived incorrectly; three numerical/value-guard mutations survived. Findings 1–3 above identify the causes. Root independently reproduced the C1/C3 cases.
- Scratch cleanup fallback passed 6/6 scenarios: five current-driver cases including both deterministic spawn windows, plus the old-driver positive control. Sandbox denied ps/pgrep, so the fallback used exact recorded PIDs, with no process-token sweep claim. Revised cases left no live recorded PIDs; the old control's four surviving PIDs were killed within that invocation. GNU timeout was emulated, not executed.
- The original unmodified cleanup test aborted at sandbox-denied ps before old-control teardown. Its dummy processes had a 300-second bound. A final exact-PID check after 416 seconds returned ProcessLookupError for all four initial recorded PIDs, closing that teardown uncertainty (cleanup-final-pids.json). This sandbox failure is separate from the requested checks, all of which passed.
- All 111 copied bundle files match the source byte-for-byte after mutation work; aggregate SHA-256 2f18b97ac420895b8ee617f7dc36f59370fb616d38546c4ad9bad05a1c619e82. Final git status/diff checks confirm no tracked modification.

Supporting files: parent-audit.json, evidence-audit.json, independent-oracle-repro.json, requested-check logs and oracle-probes/ (mutation scripts, JSON results, logs and restoration.json).

VERDICT: REJECT
