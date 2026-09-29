# Change log

## 2026-09-30 — Test the resolved warm-start path and name the actor-drift observables

Claude verification r3 of the BC handoff (`ops/rebuild-2026-09-29/codex/claude-verify-bchandoff-r3.md`, APPROVE WITH EDITS; a Claude stand-in, not a Codex verdict) re-ran the r1 and r2 mutations with the recorded counts, which re-verifies the r2 fixes. It raised three P3s. First, `.resolve()` → `.absolute()` (N1) passed, because the test's relative path had no `..`. The `main` test now passes `sub/../checkpoint.pt`, and N1 gives 3 failures (`ops/rebuild-2026-09-29/bc-handoff/mutations-r3fix.txt`). Second, the fresh head's probe gradient norm (12.4) is above `max_grad_norm` 10.0, so early critic gradients could move the BC actor through the shared trunk. The [[references/bc-best-starts-ppo-with-a-fresh-critic-head|handoff Reference]] now requires the Phase 6.2 run statement to name `optimizer/grad_norm`, `policy/approx_kl`, `policy/clipfrac` and held-out BC NLL against 0.480, and adds actor drift to its reopen condition. Third, the records no longer call the r2 fixes unverified. The r3 fixes are not re-verified by a separate reviewer.

## 2026-09-30 — Make the warm-start custody test read the whole checkpoint

Claude verification r2 of the BC handoff (`ops/rebuild-2026-09-29/codex/claude-verify-bchandoff-r2.md`, stand-in for Codex) found both r1 P2s resolved and raised four P3s. The `main` test's checkpoint was empty and absolute, so a digest that read no bytes or only the first 1 MiB, or an unresolved path, still passed. The test now writes (1 << 20) + 17 random bytes, passes the checkpoint as a relative path and checks the resolved path and SHA-256; those three mutations each give 3 failures (`ops/rebuild-2026-09-29/bc-handoff/mutations-r2fix.txt`). The [[references/top-1-team-bc-checkpoint-is-selected-by-held-out-nll-only|BC checkpoint Reference]] now says eager/2-/8-rank with 4-rank by equal model sections. The [[references/bc-best-starts-ppo-with-a-fresh-critic-head|handoff Reference]] restates the unreproducible "14 targeted tests" as 13, with the 14th unidentified. No separate reviewer has re-verified these fixes.

## 2026-09-30 — Test run_ppo's warm-start mode wiring and record the warm-start checkpoint

Claude verification r1 of the BC handoff (`ops/rebuild-2026-09-29/codex/claude-verify-bchandoff-r1.md`, REQUEST CHANGES) raised two P2s. No test reached `main`'s wiring of `--load-model-weights-mode`: passing `model_only` in place of the launch mode (M8) or loading optimizer state for `model_fresh_critic_head` (M9) passed the whole suite. And no run record named the warm-start checkpoint. The fake-trainer `main` test now runs once per mode and catches M8 and M9. A fresh launch with `--load-model-weights` writes `warm_start.json` (path, SHA-256, mode) and `warm_start/*` summary keys, and mutations of those are caught too (`ops/rebuild-2026-09-29/bc-handoff/mutations-r1fix.txt`). P3s: `launch-train.sbatch` accepts the new mode; `ppo.CHECKPOINT_KEYS`/`OPTIONAL_CHECKPOINT_KEYS` are the one allowed-key set for `run_ppo` and `PPOTrainer` loaders; a checkpoint missing run metadata now fails with a named error. The [[references/bc-best-starts-ppo-with-a-fresh-critic-head|handoff Reference]] narrows its loader scope to `run_ppo`/`PPOTrainer` and minimal checkpoints to `teacher_init`, labels the 30-row gradient sample and says eager/2-/8-rank (4-rank by equal model sections). `just py-prepare` passes (see `py-prepare-r1fix.log`). No separate reviewer has re-verified these fixes.

## 2026-09-29 — Start PPO from the BC best with a fresh critic head

Phase 6.2 needs the A100 BC best (`fd854587…`) to start PPO. On CPU it loads through `PPOTrainer.load_model_weights` into the eager, 2- and 8-rank models (the 4-rank model by equal model sections) with equal and finite outputs on a real validation shard. All 523 BC games were the imitated seat's wins, and its critic saturates: |value| > 1 − 2e-6 on 700 of 720 seat values of a held-out game, already at turn 20. Saturation starves the MSE value gradient. Isaiah had no imitation start, and his warm starts load the whole model. So `run_ppo.py` gains `--load-model-weights-mode model_fresh_critic_head`, which loads everything but `critic_head.*`. `ppo.reject_unknown_checkpoint_keys` now rejects unknown top-level keys such as `opponent_id` in every `run_ppo`/`PPOTrainer` loader. `PPOTrainer` loads and `run_ppo`'s `teacher_init` loader used to ignore them; the latter was Codex verification r1's P2 (`ops/rebuild-2026-09-29/codex/verify-bc-handoff.md`). `just py-prepare` passes. Codex round 2 did not run (usage limit), so the fix is not independently re-verified. See [[references/bc-best-starts-ppo-with-a-fresh-critic-head|handoff Reference]]; evidence in `ops/rebuild-2026-09-29/bc-handoff/`.

## 2026-09-29 — Record the top-1 team BC checkpoint (selection only)

The owner asked to train BC on the 7-day top-1 team data within 1.5–2 h. `configs/bc/kaggriculture_1gpu_eager.yaml` and `configs/kaggriculture_1gpu_eager.yaml` adapt the 2-rank BC config to one A100: same global batch, eager because the pod's driver is outside the probed compile stack. The run (`f0b7a38`, run statement `ops/rebuild-2026-09-29/run-statements/bc-a100.md`) stopped by the L9 rule at step 5,200. Its best held-out NLL was 0.480 at step 3,200. The checkpoint stays on the pod volume. See [[references/top-1-team-bc-checkpoint-is-selected-by-held-out-nll-only|BC checkpoint Reference]]; receipts are in `ops/rebuild-2026-09-29/bc-a100-2026-09-29/train/`.

## 2026-09-29 — Credit the SPS run's GPU chunk counts in the compiled-GEMM Reference

Codex verification r1 of the evidence merge (`4974888`, APPROVE WITH EDITS, `ops/rebuild-2026-09-29/codex/verify-merge-evidence-r1.md`) found one P3: the [[references/compiled-gemm-template-overflows-above-2-21-rows|compiled-GEMM Reference]] still said the chunked teacher-sized packed path was "not measured on the GPU", although the [[references/model-only-sps-ceiling-bounds-per-rank-throughput|model-only SPS ceiling]] run counted 1/2/3 teacher-proxy trunk chunks there. The Reference, its description and the [[references/index|References index]] line now credit that timing-only measurement and keep the numerical-correctness and production teacher-integration gaps.

## 2026-09-29 — Merge the kg/rebuild-model ops evidence and cookbook refresh onto the integration

The Codex-approved `kg/rebuild-model` commits `4fd40c7`..`8093d51` change only `ops/` and `cookbook/`: Phase 6.0 flash-attn, model-only SPS ceiling and ATEN-only GEMM A/B receipts, the cookbook refresh and the [[workflows/run-codex-exec-with-closed-stdin-and-wait-for-its-verdict|codex exec Workflow]]. Two files conflicted. This log keeps both sides, integration first. The [[references/index|References index]] keeps every integration line, including the cuBLAS-only lane's [[references/compiled-gemm-template-overflows-above-2-21-rows|compiled-GEMM Reference]] text, adds the two new References at the top and takes the corrected Task 0.3 Rust status. `results.md` auto-merged to the branch's copy, so the integration's "`results.md` lines 259–303" citations now resolve in this tree. The [[references/pod-v3-environment-runs-flash-attn-2-8-3-forward-on-sm120|pod environment Reference]] no longer says the compiled-GEMM FlashAttention limit awaits revision, because the cuBLAS-only lane revised it.

## 2026-09-29 — Keep BC patience on the scheduled cadence across restarts

Codex verification r2 of Task 5.2 (`ops/rebuild-2026-09-29/codex/verify-5.2-trainer-r2.md`, APPROVE WITH EDITS) found one P3: the terminal evaluation of a budget or runtime stop between scheduled evaluations counted toward patience, so a run stopped at step 1 and extended stopped at step 2 where the uninterrupted run stopped at step 4, contradicting the exact-restart claim. `HeldOutSelection.observe` now takes `scheduled`; the off-cadence terminal evaluation still selects (it can keep a lower checkpoint) but leaves patience alone, and `bc_history.jsonl` marks it with `bc/scheduled_evaluation` 0. The restart claim in `bc.py`, `README.md` and the Reference now names rows, updates, evaluation steps and stopping step, with a best never higher in NLL. New tests also pin the rank in the permutation seed and the attempt-record guards, two of r2's surviving-mutant gaps. 11/11 targeted mutants killed; `just py-prepare`: 1,703 passed, 11 skipped. See [[references/kaggriculture-bc-trainer-warm-starts-ppo-from-the-held-out-best|BC trainer Reference]].

## 2026-09-29 — Fix the BC trainer's verification r1 findings

Codex verification r1 of Task 5.2 (`ops/rebuild-2026-09-29/codex/verify-5.2-trainer-r1.md`, REJECT) found two P2s. `HeldOutSelection` gated the best checkpoint by `min_delta`, so NLLs 2.0 → 1.97 with `min_delta` 0.05 kept 2.0; selection now follows every strict minimum and a separate `improvement_nll` / `evals_since_improvement` pair carries patience. A restart reused `launch.json`'s source and admitted a changed seed or batch; `launch.json` is replaced by `bc_attempts.jsonl`, one record per launch or restart with that checkout's source, source lineage and parent `bc_state.pt` SHA-256, copied into the best record and `bc_result.json`, and `bc_state.pt` now stores the trajectory settings, which a restart must match except `max_steps`. New tests also add a hand-computed objective oracle and prove the real update path lowers the loss, closing the reported loss-sign and `optimizer.step()` gaps. Seven tests red before the fix, 11/11 targeted mutants killed; `just py-prepare`: 1,698 passed, 11 skipped. See [[references/kaggriculture-bc-trainer-warm-starts-ppo-from-the-held-out-best|BC trainer Reference]].

## 2026-09-29 — Build the Task 5.2 BC trainer script

`scripts/train_bc.py`, `python/owl/train/bc.py` and `python/owl/kaggriculture/bc_data.py` add the BC warm start on the rebuilt model. It reuses Isaiah's `create_model`, `create_optimizer`, `autocast_context`, `configure_model_compile` and DDP adapter, and it reads the refreshed 5.1 brief's `kaggriculture-bc-shard-v1` format (`bcefd02`). Changes from the reference trainer: the critic is trained with a raw-bank winner CE (Isaiah's equal policy/value distillation weights) instead of frozen; training order is deterministic per rank, so a restart reproduces the uninterrupted run; the best checkpoint by held-out NLL has exactly `run_ppo.py`'s keys; a patience rule stops the run (L9). `configs/bc/kaggriculture_2rank.yaml` carries unmeasured starting values. 18 CPU tests and 11 of 12 mutants killed; `just py-prepare`: 1,691 passed, 11 skipped. No GPU, real-data or W&B execution; the run waits for the 5.1 shards and the pod. See [[references/kaggriculture-bc-trainer-warm-starts-ppo-from-the-held-out-best|BC trainer Reference]].

## 2026-09-29 — Reconcile the teacher Reference's open dependencies after the merge

Codex verification r1 of the Phase 4 merge (`a424d8c`, APPROVE WITH EDITS,
`ops/rebuild-2026-09-29/codex/verify-merge-teacher-r1.md`) found one P3: the
[[references/kaggriculture-teacher-distills-per-slot-kl-and-per-seat-winner-ce|teacher Reference]]
still said T18 awaited the Task 3.2 value-mode guards, T19b and Task 4.4 awaited
the configs merge, and described the old skip reason, while its appended
post-merge line said those had landed. The launch-pair, T19-split and "Phase 4
is not complete" prose now states the current seams in place: T18 and the
trainer-checkpoint test wait for the Task 3.1 trainer mapping, the launch/resume
pair for the run_ppo game seam and Task 1.4 native env, and 4.4 is undone with
its configs prerequisite satisfied. The superseded appended line is folded in.
Documentation only; the verification evidence is committed beside the verdict.
`just py-prepare`: 1,673 passed, 11 skipped, docs-fresh passes
(`ops/rebuild-2026-09-29/merge-teacher-r1-fixes/py-prepare.log`).

## 2026-09-29 — Merge Phase 4 teacher distillation onto the Task 1.3 integration

Phase 4 (Codex APPROVE at `8fde43c`) forked from Task 3.1 at `4cac1a1`, before
Tasks 3.2–3.4, the cuBLAS-only compile claim and Task 1.3 merged. Four files
conflicted. `scripts/run_ppo.py` keeps both imports (`terminal_seat_banks` and
`KaggricultureObsConfig`); `docs/rl-api-specs.md` places Phase 4's Teacher
targets bullet before the integration's Environment line and native sections;
this log and the references index keep both sides, integration first. The
auto-merged teacher paths all encode through `_run_trunk`, so the cuBLAS-only
check covers them. An unskipped probe (`ops/rebuild-2026-09-29/merge-teacher/skip-probe.log`)
shows the four skipped trainer and run_ppo tests still fail, on the trainer's
missing `KaggricultureActionMask` mapping and run_ppo's no-env stop. Their skip
reasons now name those seams instead of the merged configs branch, and the
[[references/kaggriculture-teacher-distills-per-slot-kl-and-per-seat-winner-ce|teacher Reference]]
records it. No Rust changed; `docs/rules-parity-coverage.md` records the merged
counts. Full `just prepare`: engine 69, root Rust 254 with four ignored, Python
1,673 passed with 11 skips; receipt `ops/rebuild-2026-09-29/merge-teacher/prepare.log`.

## 2026-09-29 — Capture every engine build input in observation-oracle custody

Codex verification r1 of the Task 1.3 merge (`e197528`, REJECT) found that the
oracle source snapshot omitted the live engine `Cargo.toml` and `py_random.rs`,
so edits to either during regeneration survived every recheck. `regenerate.py`
now captures and rechecks those and `econ_attrib.rs`, recorded in
`dirty_files`, and fails on any undeclared engine module. Seven new custody
tests failed first and pass after the repair (53 total). The committed corpus
predates the repair; its gap is stated in the
[[references/structured-observations-preserve-legal-state-and-order|observation Reference]]
and `docs/rules-parity-coverage.md`. Receipts:
`ops/rebuild-2026-09-29/merge-1.3/verify-r1-fix/`.

## 2026-09-29 — Test each recorder size budget and archive hash (Codex verify r2)

Codex's second Task 1.4 verification confirmed both r1 fixes, found no
production defect and approved with one P3 edit.

- **Recorder tests.** The size and hash tests made the declared size disagree
  with the file, so the size check fired first. The compressed-size cap and
  both archive hash checks could be removed and every test still passed. The
  new tests change one field at a time: a cap lowered below a valid fixture, a
  declared size one byte too large, a same-length flipped byte, and a wrong
  archive or expanded digest. Each asserts its exact error before `np.load`.
  Removing any of the six size/hash guards now fails a named test.

The [[references/native-game-semantics-use-v3-owned-buffers|native buffer Reference]]
records the evidence. Receipts: `ops/rebuild-2026-09-29/1.4/verify-r2-fixes/`
and `ops/rebuild-2026-09-29/codex/verify-env-r2/`.

## 2026-09-29 — Test each recorder inventory guard and close the stale constructor status (Codex verify r1)

Codex's independent Task 1.4 verification approved with two P3 edits and found
no production defect.

- **Recorder tests.** The old inventory tests failed on the array hash before
  reaching the semantic guards, so seven guards could be removed unnoticed. A
  separate test keeps the hash check. Seventeen coherent probes refresh the
  array metadata and assert the exact error. Removing any of the 14 guards now
  fails a named test.
- **API doc.** `docs/rl-api-specs.md` now says contract v4.2 incorporates the
  approved Q1 constructor refinement.

The [[references/native-game-semantics-use-v3-owned-buffers|native buffer Reference]]
records the evidence. Receipts: `ops/rebuild-2026-09-29/1.4/verify-r1-fixes/`
and `ops/rebuild-2026-09-29/codex/verify-env-independent/`.

## 2026-09-29 — Qualify the Task 1.4 native env against the reference oracle (Claude review)

Claude reviewed Codex's Task 1.4 implementation (`8d98ea8`, `9dc2d02`) against
the brief and found no production defect.

- **Oracle on the pod.** The review ran the checks the Mac budget blocked. The
  pinned reference TrainingBatch was recorded for 16 games and the native env
  matches it bit for bit over all 11,504 transitions.
- **Mutations.** An inverted-`dones` mutant fails at game 0, step 0. The release
  overflow proof passes, and fails when the engine override is disabled.
  Advancing seeds by 1 instead of the stride fails world sizes 2 and 8. A
  premature terminal-record clear fails the reset/truncate rollback test.
- **New test.** A native L6 test proves that every call rewrites all 35 outputs
  of the one buffer set in place, including padding. Dropping one transition
  copy makes it fail.
- **Timings.** Release timings are recorded as component numbers only.
- **Contract.** v4.2 now records the agreed Q1 constructor refinement.
- **Checks.** Full `just prepare` passes on the Mac: Python 2,042 passed with 7
  skips.

The [[references/native-game-semantics-use-v3-owned-buffers|native buffer Reference]]
now carries this evidence. The
[[references/evaluation-and-truncation-follow-the-kaggriculture-objective|evaluation Reference]]
now places any training-only seed band in the Task 1.5 factory, not in native
admission. Receipts: `ops/rebuild-2026-09-29/1.4/claude-review/`.

## 2026-09-29 — Add transactional native lifecycle and the codec/table ABI

Task 1.4 extends the existing root module over the merged grammar and structured
encoder: staged game/seed/terminal publication, Rust rewards, selected-row
truncation and four cold codec/table functions. Restored mutations catch early
live seed writes, whole-batch truncate publication and colliding rank offsets;
world sizes 2 and 8 consume 67 seeds per rank. Native Python suites pass 383
cases; root Rust passes 274 with five ignored, and the engine passes 69. The
[[references/native-game-semantics-use-v3-owned-buffers|native buffer Reference]]
now records this current implementation, inventory and limits while preserving
reference-branch history. The sole reference-recording attempt stopped at the
Mac memory limit during compilation, recording zero games and publishing no
fixture. The requested five-file Python check reports 497 passes and two
missing-fixture failures. Both preparation commands pass their static checks;
full `prepare` also passes Rust, build and trim, then both exceed the Mac RSS
budget during pytest. Broad Python completion, full trajectory and release
qualification remain PENDING (pod); the Python adapter, device bridge and CUDA
entry fence belong to Task 1.5. Native seeds use the approved checked i64
domain; training-band separation is a factory
policy, not an added native cap.

## 2026-09-29 — Merge Task 1.3 structured observations onto the trainer-lane integration

Task 1.3 (Codex APPROVE at `dc6b200`) forked from `f464c3d`, after the Task 1.2
merge, and later merged an older integration (`78ab94f`). Only this log and
`docs/rules-parity-coverage.md` conflicted: both sides' log entries are kept,
integration first, and the coverage page keeps the trainer-lane counts as
history beside Task 1.3's grammar-bridge retirement and the merged counts. The
retirement of `engine_rs/tests/grammar_kernel.rs` and the root engine dependency
are kept; `engine_rs/TRIM_MANIFEST.json` equals Task 1.3's because the
integration side left it unchanged since the fork, and the trim checker passes.
The [[references/structured-observations-preserve-legal-state-and-order|observation Reference]]
index line moves under "Current rebuild". Full `just prepare`: engine 69, root
Rust 254 with four ignored, Python 1,618 passed with seven skips; receipt
`ops/rebuild-2026-09-29/merge-1.3/prepare.log`.

## 2026-09-29 — Serialize the GEMM-backend claim and scope old probe limits

Codex verification r2 of the cuBLAS-only wiring
(`ops/rebuild-2026-09-29/codex/verify-aten-r2.md`, APPROVE WITH EDITS) found
two P3s. A concurrent Orbit claim could land while a Kaggriculture claim read
its stack and then be silently overwritten; `claim_gemm_backends` in
`python/owl/model/compile_gemm.py` now holds a lock around check and write, and
a new test in `tests/kaggriculture/test_compile_gemm_backends.py` reproduces
the race (it failed before the lock). The
[[references/compiled-gemm-template-overflows-above-2-21-rows|compiled-GEMM Reference]]
Limits now scope "triton not captured" and "real flash-attn untested" to the
earlier overflow probe and credit the A/B's triton 3.5.0 and flash-attn 2.8.3;
the backward-above-bound and GPU-wiring gaps stay. The
[[decisions/kaggriculture-compiles-gemms-with-cublas-only|Decision]] records
the lock and the r2 check.

## 2026-09-29 — Claim cuBLAS-only GEMMs at every public compile entry

Codex verification r1 of the cuBLAS-only wiring
(`ops/rebuild-2026-09-29/codex/verify-aten-r1.md`, REJECT) found that a direct
`compile_transformer_trunk` call bypassed the claim: a Kaggriculture compile
with `"ATEN"` preset skipped the stack check, and a direct Orbit compile let a
later Kaggriculture compile change Orbit's backends. The claim and stack check
moved to `python/owl/model/compile_gemm.py`; both models'
`compile_transformer_trunk` now claim their game, and `configure_model_compile`
claims for `mlp`. New tests cover both orders across every pair of entry points.
The [[decisions/kaggriculture-compiles-gemms-with-cublas-only|Decision]],
README and run_ppo docstring also now say an installed triton is checked even
without CUDA; only a missing triton is skipped there.

## 2026-09-29 — Require cuBLAS-only GEMMs for compiled Kaggriculture regions

The owner asked that the compiled-GEMM CUDA crash never recur in v3. The
Codex-approved ATEN-only A/B (branch `kg/rebuild-model` at `9bdd82d`,
`ops/rebuild-2026-09-29/results.md` lines 259–303) found cuBLAS-only GEMM
backends correct above the int32 bound at +4.5–6.5 % model-only update wall.
`configure_model_compile` now checks the probed torch/triton/driver stack and
sets `max_autotune_gemm_backends = "ATEN"` before any Kaggriculture compile.
The model refuses other values at compile time and on every compiled trunk
call. Orbit keeps its backends, a process never compiles both games, and
`run_ppo` checks the stack at startup and records the claim in W&B
summaries. The overflow guards stay. New
[[decisions/kaggriculture-compiles-gemms-with-cublas-only|Decision]]; the
[[references/compiled-gemm-template-overflows-above-2-21-rows|compiled-GEMM Reference]]
Consequences carry the measured A/B.

## 2026-09-29 — Close the startup-assertion gap in the grammar-heads Reference

Codex's merge verification (`ops/rebuild-2026-09-29/codex/verify-merge-trainer-lanes-r1.md`,
P3) found the [[references/kaggriculture-grammar-heads-sit-behind-isaiahs-actor-projection|grammar-heads Reference]]
still listing the Task 3.4 startup workload assertion as open after the merge.
It now links that completed check to the
[[references/kaggriculture-configs-follow-isaiahs-scaling-6m-recipe|configs Reference]]
and keeps trainer integration (Task 3.1) as the open gap.

## 2026-09-29 — Record the merged trainer-lane check counts

After merging Tasks 3.1, 3.2/3.3, 3.4 and the 1.4/1.5 briefs onto the Task 1.2
integration, full `just prepare` passed: engine 87, root Rust 164 with two
ignored, Python 1,458 passed with five skips (two FlashAttention/CUDA, one
quantized backend, the native grammar binding, and the native Kaggriculture
evaluation env). `docs/rules-parity-coverage.md` now states these counts beside
the Task 1.2 merge counts; receipt
`ops/rebuild-2026-09-29/merge-trainer-lanes/prepare.log`.

## 2026-09-29 — Merge Task 3.4 configs and the startup workload check onto Tasks 3.1–3.3

Task 3.4 (Codex APPROVE at `71cebdb`, which already contains Task 3.1 at
`4cac1a1`) conflicted with Tasks 3.2/3.3 in `FullConfig`, `run_ppo.py` and the
run_ppo tests. A validated Kaggriculture env now runs both guard sets: Task
3.4's `_validate_kaggriculture_rl`, then Tasks 3.2/3.3's
`_validate_kaggriculture_training` (`mse`, `per_player`); Orbit keeps Isaiah's
checks in `_validate_orbit_constraints`. `_create_eval_env` rejects
Kaggriculture, then narrows Orbit with `require_orbit_env`. The 3.2/3.3 tests
now load the shipped `configs/kaggriculture.yaml` instead of unvalidated copies,
and the skipped YAML guard test runs over every `configs/kaggriculture*.yaml`.
The [[references/evaluation-and-truncation-follow-the-kaggriculture-objective|evaluation and truncation Reference]]
records the merge and drops its config-registration gap; the
[[references/kaggriculture-configs-follow-isaiahs-scaling-6m-recipe|configs Reference]]
names the added guards. The run_ppo startup check uses the registered
`KaggricultureTransformerConfig` from the loaded config.

## 2026-09-29 — Merge Tasks 3.2/3.3 evaluation and truncation semantics onto Task 3.1

Tasks 3.2/3.3 (Codex APPROVE at `530b8cc`) forked at `e1458d2`, before Task 1.2
and Task 3.1 merged. `FullConfig` keeps both guards: Task 3.1's rejection of a
Kaggriculture model under Orbit's `EnvConfig`, and Tasks 3.2/3.3's
`_validate_kaggriculture_training` for a Kaggriculture observation spec.
`docs/rl-api-specs.md` keeps the 3.2/3.3 Environment line (raw-bank winners,
truncation bootstrap, evaluation seed band) followed by the Task 1.2 native
grammar section. The references index lists both new References.

## 2026-09-29 — Merge Task 3.1 model registration onto the native grammar

Task 3.1's model side (Codex APPROVE at `4cac1a1`) forked at `e1458d2`, before
the Task 1.2 merge. The references index keeps Task 3.1's
compiled-GEMM and encoder lines and the integration branch's grammar-heads line,
which names the Task 1.4 binding as the remaining native-table dependency. Both
sides' log entries are kept, integration first.

## 2026-09-29 — Fix Task 1.3 verification round 1: retire the grammar bridge and guard pinned cases

Codex's independent verification of `7b5eacd` found no encoder semantic defect but rejected on four findings. All four are fixed test-first:
- The in-process pinned-memory probe could kill pytest on macOS (SIGSEGV). Pinned cases now run only with CUDA, as in the starter, and assert pinning.
- The contract v4.1 grammar bridge is retired. Nine kernel tests moved to root `src/kaggriculture/grammar_kernel_tests.rs`; the engine file and its trim registration are removed; a decode mutation fails the root route.
- Coverage docs now point to current receipts.
- The missing-oracle error no longer asserts the obsolete quota diagnosis.

`just prepare` passes: 254 root Rust (four ignored), 69 engine, 1,437 Python (six skipped). The [[references/structured-observations-preserve-legal-state-and-order|observation Reference]] carries the evidence. Optimized timing remains open.

## 2026-09-29 — Qualify the structured observation oracle after Claude's R1 correction

Claude reviewed Codex's incomplete Task 1.3 and merged integration, which brought the Task 2.1 schema and the Task 1.2 grammar. Claude then adopted seeded policy `observation-corpus-v2`: HIRE entries during hours 0–7, stopping at 16 hands. This makes R1's unchanged quota of more than 16 actors reachable, with 6 qualifying states.

Results:
- The frozen 512-state oracle (782 KB compressed) matches bitwise at all 8,176 offsets and reproduces byte for byte.
- The real `check_contract()` accepts all 512 records.
- Three restored mutations fail their oracles.
- Two defects are fixed: the Mac pinned-memory probe, and the watchdog charging pre-existing caller RSS.
- `just prepare` passes.

The [[references/structured-observations-preserve-legal-state-and-order|observation Reference]] carries the evidence. Optimized timing and the grammar-bridge retirement remain open.

## 2026-09-29 — Implement structured native observations and record qualification blocks

The [[references/structured-observations-preserve-legal-state-and-order|observation
Reference]] now records the full native encoder, exact caller-buffer binding,
source-bound oracle tooling and reconstruction controls. All 16 reconstruction
mutations fail and restore; 43 custody and 55 native boundary checks pass. The
snapshot counter fails with two acquisitions and passes with one. Full corpus
qualification remains blocked by the unchanged R1 actor quota, actual-schema
collection by Task 2.1, and optimized timing by the 1-GB Mac build limit. Final
command outcomes and the pod handoff remain explicit in the linked receipt:
233 root Rust passes/one corpus failure/four ignored, 59 engine passes and
1,056 broad fast Python passes/three platform skips. Preparation gates retain
the corpus and missing-schema failures; lint, typing and trim checks pass.

## 2026-09-29 — Complete native observation fields and diagnostic validation

The [[references/structured-observations-preserve-legal-state-and-order|observation
Reference]] adds ordered shops, exact markets, live-row masks and complete-row
checks. The 54-test B–F suite and Clippy pass; 72 corruptions are rejected.
Late second-environment failure preserves both published rows, serial/two-worker
results agree and reused dense buffers clear correctly. Corpus, Python binding,
shared-schema admission and optimized timing remain unqualified.

## 2026-09-29 — Preserve all actors and own private insertion order

The [[references/structured-observations-preserve-legal-state-and-order|observation
Reference]] adds exact actor/storage counts and ranks, public player facts and
both-seat privacy controls. Twelve new semantic failures become passes; the
combined B–E module passes 46 tests. Reversing actor/shed ranks fails two tests,
then restoration and Clippy pass. Market/mask completion and corpus/schema
qualification remain pending.

## 2026-09-29 — Admit strict engine-shaped tiles before publication

The [[references/structured-observations-preserve-legal-state-and-order|observation
Reference]] adds all seven tile tensors and strict key/type/sentinel admission.
Nine semantic failures become ten passing tile tests; the independent scan covers
2,880 official states and 576,000 tiles. Transposing cell coordinates fails the
asymmetric fixture; restoration and Clippy pass. Actors, markets, corpus and
shared-schema qualification remain pending.

## 2026-09-29 — Bind observation config and preserve exact hire costs

The [[references/structured-observations-preserve-legal-state-and-order|observation
Reference]] adds Task C's checked config, single-snapshot wrapper and exact
Fibonacci cost prefix. All 24 B/C tests and Clippy pass; a rival-count mutation
fails two hire tests and restoration passes. The pinned integer type lacks the
brief's proposed Display API, so exact serialization is used only at config
admission. Complete encoding, corpus and shared-schema checks remain pending.

## 2026-09-29 — Add checked structured observation storage

The [[references/structured-observations-preserve-legal-state-and-order|observation
Reference]] records Task B's 29 named buffers, checked sizes, safe serial/Rayon
views and atomic equal-capacity publication. Eleven boundary tests and Clippy
pass; disabling a length check produces the expected failure and restoring it
passes. State encoding and Python admission remain unfinished.

## 2026-09-29 — Integrate root numeric features and expose an observation-corpus limit

The [[references/structured-observations-preserve-legal-state-and-order|in-progress
observation Reference]] records Task A: the decimal regression passes before
feature unification, fails afterward, then passes with the test-only Number
repair. Root tests: 157 passed, two ignored; trim checker OK. The prescribed R1
corpus cannot supply four non-synthetic >16-actor states: the official selection
has none and the seeded policy cannot hire that many before daily reset. Quotas
remain unchanged. Task I still awaits the shared-schema merge; no completion or
performance claim is made. Independent Task A review also prompted relabelling
the restart Decision's no-root-dependency paragraph as the historical Task 1.1
checkpoint.

## 2026-09-29 — Point the grammar-table dependency at the Task 1.4 binding

After the Task 1.2 merge (Codex APPROVE WITH EDITS, `ops/rebuild-2026-09-29/codex/verify-merge-1.2-r1.md`), the Rust grammar tables exist and match the Python stand-in (964 bits, no mismatches). The [[references/kaggriculture-grammar-heads-sit-behind-isaiahs-actor-projection|grammar-heads Reference]], its index line and `docs/model-architecture.md` now name the Task 1.4 Python binding as the remaining dependency, including for brief item 11 (recorded reference programs), which also needs model integration.

## 2026-09-29 — Merge the native grammar with live parity and the heads

Task 1.2 (Codex APPROVE at `7877c46`) forked before Task 1.1b and Task 2.3
merged. Both it and Task 1.1b extended the trim checker's fixed authored set;
the merged set is exactly `replay_parity.rs`, `grammar_kernel.rs` and the
generated-trace `MANIFEST.json`, and the inventory regressions cover all three.
`engine_rs/TRIM_MANIFEST.json` was regenerated by a three-way updater
(`ops/rebuild-2026-09-29/merge-1.2/update_trim_manifest.py`): reference
inventory unchanged, authored hashes recomputed. The
[[references/native-game-semantics-use-v3-owned-buffers|native semantics Reference]]
and coverage document state merged counts: engine 87/87, root 164 with two
ignored, Python 1,337 passed with four skips. Native `grammar_tables()` matches
the heads' Python stand-in on all 964 bits; the binding stays with Task 1.4.
On a scratch copy, reverting `EDITABLE`, dropping the trace-hash check,
dropping the generated manifest from the authored set, or removing the
coupled-HIRE capacity guard each fails its regressions. Receipts:
`ops/rebuild-2026-09-29/merge-1.2/`.

## 2026-09-29 — Claude review of the Task 1.2 grammar with per-oracle mutation controls

The [[references/native-game-semantics-use-v3-owned-buffers|native semantics Reference]]
now records Claude's review: no production defect; 20 of 20 restored source
mutations fail their named grammar, kernel, checker or recorder test; the
reference oracle reproduces byte-for-byte; and the brief's nonexistent `reference`
extra is corrected. Replay-rejection categories remain synthetic-only.

## 2026-09-29 — Rebuild the native grammar under the reviewed Task 1.2 contract

The [[references/native-game-semantics-use-v3-owned-buffers|native semantics Reference]]
now records the compiled typed grammar, strict native codec and direct tables,
with the complete changed-path inventory and five agreed v4.1 clarifications.
The independent 320-program oracle reproduces exactly; nine shared grammar tests
and nine kernel tests cover malformed transport, coupled HIRE, both-seat dense
execution, the 240-to-241 boundary and 64 replay-state comparisons. Captured
stubs and restored negative controls distinguish real checks from self-agreement.
Full preparation passes: 164 root Rust tests (two ignored), 77 engine tests,
1,045 Python tests (three platform skips), formatting/lint/typing/docs and the
trim checker. The receipt records remaining L4/binding/GPU limits. Historical
adapter and performance evidence remains reference-scoped.

## 2026-09-29 — Record completed config/model integration in the recipe Decision

Codex approved Task 3.4 r3 with one P3 (`ops/rebuild-2026-09-29/codex/verify-3.4-r3.md`): the [[decisions/recipe-choices-align-to-isaiah-without-owner-escalation|recipe Decision]] still said the ranked configs could not load and called the alignment targets unapplied. Its heading, a new current-state paragraph, Limits, description and index line now record that both targets are encoded in the configs and model, the YAMLs load through `FullConfig`, and the verification's CPU checks passed (pytest 1,373 passed / 4 skipped, mypy 59 files clean, cargo 155 passed / 2 ignored, startup mutation 4 → 4 failed → 4 passed). Native env/trainer integration, teacher execution and GPU qualification stay pending. The owner's rule, quote and historical evidence are unchanged; the status text is implementation state, not owner adoption.

## 2026-09-29 — Load the Kaggriculture configs through the real FullConfig (Task 3.4 r2 fixes)

Codex rejected Task 3.4 again (`ops/rebuild-2026-09-29/codex/verify-3.4-r2.md`): the three configs failed `FullConfig.from_file` with five schema errors each, the startup tests bypassed that loader, and the ranked configs called `native_threads` measured. The fix:
- merges the Task 3.1 model registration (`kg/rebuild-trainer-model` at `4cac1a1`);
- adds `KaggricultureEnvConfig` and `KaggricultureRewardConfig` (`python/owl/kaggriculture/config.py`), selected in `FullConfig` by the observation tag, with both game pairings, `gamma = 1` and Isaiah's Orbit cross-checks unchanged;
- adds `require_orbit_env`, which stops `run_ppo` with an explicit error after the workload check and before the run directory, and narrows the Orbit-only sites in `run_ppo` and `benchmark_checkpoints`;
- rewrites the startup tests to run `main` from the shipped YAML through the real loader (fresh, override rejection and resume), removes all six integration skips, and relabels `native_threads` provisional.

`py-prepare`: 1,373 passed, 4 skipped; mypy 60 files; docs-fresh passed. The [[references/kaggriculture-configs-follow-isaiahs-scaling-6m-recipe|configs Reference]], the [[references/kaggriculture-model-joins-isaiahs-factory-compile-and-masked-critic|Task 3.1 model-side Reference]] and the [[references/compiled-gemm-template-overflows-above-2-21-rows|compiled-GEMM Reference]] now say `FullConfig` loads the model with its env.

## 2026-09-29 — Wire the Kaggriculture workload check into run_ppo startup

Codex rejected Task 3.4 (`ops/rebuild-2026-09-29/codex/verify-3.4-r1.md`). The fix:
- `scripts/run_ppo.py` now calls `_check_model_workload` once the per-rank shapes are final and before the run directory, env or model exist. Kaggriculture workloads the model cannot service fail there, and the main process prints the headroom. Orbit models are skipped.
- Head calls are exact. Trunk calls are now labelled an upper bound and trunk headroom a lower bound, because they assume full padding.
- The schema-coverage claim is narrowed to the observation, action, model, optimizer and PPO sections.
- `tests/scripts/test_run_ppo.py` drives `main` through a patched loader, because the configs load through `FullConfig` only after Task 3.1.

`py-prepare`: 1,331 passed and 10 skipped; ruff, format, mypy (59 files) and docs-fresh passed, with `DOCS_CURRENT=1` after README was reviewed and still current. See the [[references/kaggriculture-configs-follow-isaiahs-scaling-6m-recipe|configs Reference]].

## 2026-09-29 — Add Kaggriculture configs that follow Isaiah's scaling_6m recipe

Rebuild Task 3.4 adds `configs/kaggriculture_2rank.yaml` (128 envs/rank, spm 8) and `configs/kaggriculture_4rank.yaml` (64/4), deliberately aligned to Isaiah per the [[decisions/recipe-choices-align-to-isaiah-without-owner-escalation|recipe Decision]] (its Limits now link them). They use his multi-GPU rule, so the global workload equals `scaling_6m`; the optimizer, PPO, teacher and compile settings are his. A local CPU config and a tiny CPU model preset are added for Task 3.5. The startup workload check (`python/owl/model/kaggriculture_workload.py`) sizes rollout, minibatch, teacher-chunk and evaluation rows against the model's trunk and head chunking and logs the headroom; at 2 ranks the 16,384-row teacher chunk runs in at most 3 trunk calls (full padding). Tests validate the observation, action, model, optimizer and PPO sections against their schemas; env keys and reward-shaping values get exact key and value assertions only. FullConfig loading and model construction are skipped until Task 3.1. `py-prepare`: 1,324 passed, 10 skipped. No GPU or training run. See the [[references/kaggriculture-configs-follow-isaiahs-scaling-6m-recipe|configs Reference]].

## 2026-09-29 — Implement the T19b launch/resume tests and scope Orbit-only teacher rules (Phase 4 verification r1)

Codex's r1 verification of Phases 4.1–4.3 (APPROVE WITH EDITS) found that T19b was a placeholder: a docstring and an unconditional `AssertionError`. T19b is now three skipped tests: run_ppo resume restores the teacher from `checkpoint_last_best.pt`, a fresh launch from weights activates it, and trainer checkpoints hold no teacher cache. The resume and fresh-launch pair passes a dry run with config validation bypassed, and four `run_ppo.py` mutations each fail one of them. The skip now names the Task 3.1 run_ppo game seam as well as `kg/rebuild-configs`. `docs/model-architecture.md` scopes the discrete_targets-only cached KL and the launch-mode rule to Orbit (P3). The [[references/kaggriculture-teacher-distills-per-slot-kl-and-per-seat-winner-ce|Phase 4 teacher Reference]] records the tests, the dry run and `py-prepare` (1,372 passed, 8 skipped); Phase 4 remains incomplete.

## 2026-09-29 — Wire Kaggriculture teacher distillation through Isaiah's cached-teacher path (Phase 4.3)

`KaggricultureTransformer` now supports both cached distillation paths. The cached path admits targets before any kernel, including the grammar signature. It is bit-for-bit the combined path on CPU FP32. The model owns the value CE (`BaseModelAPI.teacher_value_cross_entropy`): Isaiah's joint-distribution sum stays the default and replaces the removed free function, while Kaggriculture averages each live seat's CE. PPO's two teacher wrappers dispatch statelessly (review P1-1) and log `teacher/cache_bytes`. `run_ppo._teacher_obs_spec_for_student` dispatches by game. The [[references/kaggriculture-teacher-distills-per-slot-kl-and-per-seat-winner-ce|Phase 4 teacher Reference]] records T12–T19a, three more killed mutations, `py-prepare` with 1,372 passed and Isaiah's suites at 1,048 passed. It also states that Phase 4 is not complete: T18 and T19b are skipped on the Task 3.1/3.2 trainer seams and `kg/rebuild-configs`, and 4.4 waits on that merge. The [[references/ppo-trainer-seams-map-any-schema-and-alarm-on-replay-drift|stream C Reference]]'s two open limits are revised in place: the trainer is protocol-typed, and Orbit alone keeps the asymmetric `concat`.

## 2026-09-29 — Cache Kaggriculture teacher targets under the TeacherTargets protocol (Phase 4.2)

`KaggricultureTeacherTargets` caches the teacher's per-slot masked logits and winner probabilities in the rollout lead layout. It uses 102,208 B per seat row, a constant derived from the contract widths. Its `concat` validates symmetrically, and it carries the teacher's grammar signature: a table SHA-256 taken at construction plus `hire_limit`. `BaseModelAPI`, `ppo.py` and the DDP adapter now type the cached-teacher API by the protocol, and Isaiah's model narrows it with `TypeError`. The protocol gains `nbytes`. The [[references/kaggriculture-teacher-distills-per-slot-kl-and-per-seat-winner-ce|Phase 4 teacher Reference]] records T7–T11, four more killed mutations and `py-prepare` with 1,360 passed.

## 2026-09-29 — Expose replay-conditioned per-slot logits and KL in the Kaggriculture grammar core (Phase 4.1)

The Phase 4 brief is now v2 after Codex's REVISE review (`ops/rebuild-2026-09-29/codex/brief-4-review.md`). The changes are stateless teacher dispatch, a grammar signature on the cached path, one KL dtype rule, and phase completion gated on the trainer tests. Task 4.1 follows it. `policy_core` returns each slot's masked logits and the liveness-weighted per-slot KL in the log-prob layout. Isaiah's `categorical_kl_from_logits` promotes instead of demoting FP64. The shared test helpers moved to `tests/kaggriculture/helpers.py`. The new [[references/kaggriculture-teacher-distills-per-slot-kl-and-per-seat-winner-ce|Phase 4 teacher Reference]] lists the checks: T1–T6, a brute-force oracle, four killed mutations and `py-prepare` with 1,343 passed. It also lists three test-level deviations: an FP32 rounding bound, `assert_close` for head chunking, and a small HIRE oracle case.

## 2026-09-29 — Reconcile the compiled-GEMM Reference with Task 3.1's registration

Codex verified `e1458d2...aadba6d` (APPROVE WITH EDITS, no functional defect; `ops/rebuild-2026-09-29/codex/verify-3.1-rest-r2.md`). Its one finding: the [[references/compiled-gemm-template-overflows-above-2-21-rows|compiled-GEMM Reference]] still said Kaggriculture was absent from `ModelConfig` and the factory and that `configure_model_compile` rejected its trunk target. The Reference now links the [[references/kaggriculture-model-joins-isaiahs-factory-compile-and-masked-critic|Task 3.1 model-side Reference]] for registration and the guarded trunk dispatch, and keeps the limits: CPU recording stand-ins only, `FullConfig` still rejects the model, and integrated workloads and real Inductor/CUDA compilation are unverified. Its description and index line match.

## 2026-09-29 — Apply Codex's r1 wording edits to the Task 3.1 model-side Reference

Codex verified `e1458d2...b1da613` (APPROVE WITH EDITS, no functional defect; `ops/rebuild-2026-09-29/codex/verify-3.1-rest-r1.md`). The [[references/kaggriculture-model-joins-isaiahs-factory-compile-and-masked-critic|Task 3.1 model-side Reference]] now:
- says the factory keeps Isaiah's exhaustive `match`/`assert_never` but matches config classes where Isaiah matches `model_arch` strings;
- classifies the 4 `py-prepare` skips as 3 hardware/backend and 1 unavailable native grammar binding, instead of "4 hardware skips";
- cites the Codex report and its committed evidence.

## 2026-09-29 — Register the Kaggriculture model and align its compile and critic with Isaiah

Task 3.1 model side, on `kg/rebuild-trainer-model` (base `e1458d2`). The new [[references/kaggriculture-model-joins-isaiahs-factory-compile-and-masked-critic|Task 3.1 model-side Reference]] records three changes:
- **Registration:** `KaggricultureTransformerConfig` joins `ModelConfig`, and `create_model` checks that the specs belong to the model's game. `FullConfig` rejects the model until a Kaggriculture env config exists.
- **Compile:** `configure_model_compile` dispatches the trunk target through a nominal `TrunkCompileAPI` instead of rejecting every model except `StatelessTransformerV1`. Tests pin the compiled region to the blocks, behind `_run_trunk`'s guard.
- **Critic:** the winner softmax uses Isaiah's masked form. The encoder and heads References now link here instead of listing these items as open.

Checks: `just py-prepare` passes (1,324 passed, 4 skipped), and Isaiah's suites give 1,048 passed. Red runs are in `ops/rebuild-2026-09-29/trainer-model/`. Everything ran on CPU; real compile and CUDA are unverified.

## 2026-09-29 — Report promotion only after it completes; narrow the evaluation-seed claims

Codex verified Tasks 3.2/3.3 (APPROVE WITH EDITS; `ops/rebuild-2026-09-29/codex/verify-3.2-3.3-independent/review.md`). The [[references/evaluation-and-truncation-follow-the-kaggriculture-objective|evaluation and truncation Reference]] now records both repairs.
- `scripts/run_ppo.py` logs the evaluation record, including `eval/promoted`, once, after the incumbent refresh, teacher update, promoted checkpoint and barrier (the reference's order). A new test injects a refresh failure and a promoted-checkpoint write failure; each failed before the change and now leaves no `eval/promoted`.
- `_evaluation_seed` is documented as distinct per step for a fixed base seed and per base seed for a fixed step, not injective over pairs and not a separator of consecutive seed ranges. Training/evaluation seed separation holds only below a `2**62` training-seed bound that the native seam (Tasks 1.4/1.5) must enforce. The docstring, `README.md`, `docs/rl-api-specs.md` and a characterization test agree.

`just py-prepare` passed (1,344 passed, 6 skipped). The native env and a real Kaggriculture evaluation remain pending.

## 2026-09-29 — Decide Kaggriculture evaluations by raw banks and keep the truncation reward

Rebuild Tasks 3.2 and 3.3 land on the trainer seam; the new [[references/evaluation-and-truncation-follow-the-kaggriculture-objective|evaluation and truncation Reference]] records the inventory.
- Kaggriculture evaluation games are decided by raw final banks, with candidate bank metrics logged.
- A Kaggriculture time-limit cut keeps the economic reward earned on that transition; Orbit still zeroes it.
- `FullConfig` rejects Kaggriculture settings other than `win_loss`, gamma 1, MSE value loss and joint `per_player` clipping.
- `_evaluation_seed` gives each evaluation a reproducible seed in `[2**62, 2**62 + 2**61)`.
- Every evaluation logs `eval/games`, `eval/promoted` and `eval/promotion_threshold`.

The checks were CPU TDD with a fake env, a mutation check and `just py-prepare` (1,341 passed, 6 skipped). The native env, config registration and the Kaggriculture evaluation env remain skipped placeholders for Tasks 1.4/1.5.

## 2026-09-29 — Reconcile overflow notes after merging the heads and GEMM evidence

Codex verified the staged merge of Task 2.3 and the GEMM-limit evidence (APPROVE WITH EDITS; `ops/rebuild-2026-09-29/codex/verify-merge-heads-r1.md`).
- The overflow comment in `python/owl/model/kaggriculture.py` and `docs/model-architecture.md` now cites the measured input-side limit, with `rows × max(in, out) < 2^31` as the training design bound.
- The [[references/compiled-gemm-template-overflows-above-2-21-rows|compiled-GEMM Reference]] now records that the Task 2.3 heads exist and chunk.
- The Reference scopes the "rejects L" measurement to `1ddc71d`, because the packed path now chunks instead (CPU-tested). Its index line matches.

## 2026-09-29 — Merge live parity with the hardened rules-kernel checker

Task 1.1b (Codex APPROVE at `16e56b6`) and Task 1.1's verification hardening
edited `scripts/check_engine_trim.py` independently. The merge keeps both
contracts: `verify_task` separates generated traces, runs the generic manifest
checks (including the `EDITABLE` set), then validates the generated manifest
(engine pin, budget, at least six traces), then applies the fixed Task 1.1
checks (Rayon lockfile derivation, pinned appendix hash). Two new `check()`-level tests reject an edited or unlisted generated
trace. The base's owner 1.32.7 project pin made one Task 1.1b test premise
stale; it now asserts that the project environment passes the engine guard and
that a mismatched pin exits before writing. The
[[references/live-differential-parity-checks-the-rust-kernel|live parity Reference]]
and coverage document say so. Engine 69/69; `just prepare` passes with Python
1,209 passed and 3 skipped. On a scratch copy, reverting `EDITABLE` or removing
the trace-hash check fails its regressions. Receipts are in
`ops/rebuild-2026-09-29/merge-1.1b/`.

## 2026-09-29 — Correct live-parity test totals after Codex re-verification

Codex re-verified Task 1.1b at `25ec814` (APPROVE WITH EDITS, no blocking findings; every round-one finding resolved). The [[references/live-differential-parity-checks-the-rust-kernel|live parity Reference]] and `docs/rules-parity-coverage.md` now state the current totals, 69 engine tests including 19 replay-parity tests, rather than 66 and 16. The compact verification evidence is in `ops/rebuild-2026-09-29/1.1b/verify-r2/`.

## 2026-09-29 — Close Codex verification of the live parity check

Codex approved Task 1.1b with edits; commit `6217868` fixes them test-first. The
[[references/live-differential-parity-checks-the-rust-kernel|live parity Reference]]
now records that the sweep classifies D1/D2 only from the observed mismatch (a
corrupted state on a D1 line stays unclassified), submits the `null` probe
exactly, checks full-state rollback on rejected steps and keeps seven minimal
one-step repros. The rerun 40-game sweep matches the first (306/343 agree, 37
confirmed D1/D2, 0 new); receipts are in `ops/rebuild-2026-09-29/1.1b/verify-r1/`.
Kernel bytes are unchanged.

## 2026-09-29 — Check the Rust kernel live against Kaggle's engine

Owner: “can you add a parity check after your rust engine, with kaggle envcironments?”
The new [[references/live-differential-parity-checks-the-rust-kernel|live parity Reference]]
records Task 1.1b. A hash-guarded generator runs kaggle-environments 1.32.7's
own engine in an isolated uv environment, leaving the project lock unchanged.
It records official-format traces plus `rejected` records. The Rust replay
checks official, committed and swept traces with one first-divergence comparator.
Eight committed games (3,960 transitions) and a 40-game local sweep (21,824)
agree. Of 303 probes, 37 diverge in two malformed-input classes: D1, Unicode digit
strings, and D2, unhashable verbs/items. The pinned kernel is not edited. Seven
repros are exact expected failures; failing traces are kept in
`ops/rebuild-2026-09-29/1.1b/`. The trim checker now pins the generated
manifest. No training or GPU work ran.

## 2026-09-29 — Report action-mask shapes in the replay-drift alarm and refresh its evidence

Codex's stream C re-verification (approve with edits) found stale evidence and one diagnostic gap, now corrected in the [[references/ppo-trainer-seams-map-any-schema-and-alarm-on-replay-drift|trainer seams Reference]]:
- The references index still cited 924 passes, and the note called that Phase 4 prep `py-prepare` run "final". The index now carries the current count, and the note labels the older run.
- The alarm's `RuntimeError` promised every observation tensor shape but omitted nested action-mask tensors. `_observation_tensor_shapes` now lists `action_mask.can_act` and `action_mask.max_launch` (7 red, then green, over all three Orbit mask types and a second schema). README says so.

Python suite: 941 passed, 3 skipped. No GPU, training or Rust check ran.

## 2026-09-29 — Narrow the trainer-seam claims after Codex's stream C review

Codex's stream C verification (approve with edits) found three overstated claims, now corrected in the [[references/ppo-trainer-seams-map-any-schema-and-alarm-on-replay-drift|trainer seams Reference]]:
- `TeacherTargets.concat` docs promised rejection of any optional-target mismatch; they now state the inherited first-chunk asymmetry, and tests pin both chunk orders. Behavior is unchanged; Phase 4 decides.
- The log-ratio alarm's units follow `ppo_clip_mode` (joint action under `per_player`, entity mean under `per_entity`), documented in `PPOConfig` and README with multi-entity and two-fake-rank tests. The 0.05 default awaits Phase 6 GPU noise measurement.
- The observation oracle is an extracted successful-path copy, not a frozen one; the invalid-limit tests assert exact Pydantic errors (7 red with the field removed).

Python suite: 934 passed, 3 skipped. No GPU, training or Rust check ran.

## 2026-09-29 — Make the trainer seams schema-generic and alarm on replay drift

Three game-neutral refactors of Isaiah's trainer, each test-first on Orbit types, are recorded in the new [[references/ppo-trainer-seams-map-any-schema-and-alarm-on-replay-drift|trainer seams Reference]]:
- Task 3.1: one schema-generic `_map_observation` replaces the Orbit field lists, checked against a frozen copy of Isaiah's helpers.
- Task 3.6: `rl.first_minibatch_logratio_limit` (0.05 nats, `None` disables) aborts before the first optimizer step when replay disagrees with the rollout. This is the alarm required by the [[references/compiled-gemm-template-overflows-above-2-21-rows|compiler overflow Reference]].
- Phase 4 prep: a `TeacherTargets` protocol replaces the free index/concat functions.

The full Python suite and `py-prepare` pass (924 passed, 3 skipped). No GPU, training or Rust check ran. The threshold's GPU noise margin, multi-rank behavior and Kaggriculture batches remain unverified.

## 2026-09-29 — Port replay selection and bound the data/evaluation rebuild

Stream D ports the engine-independent selector with typed manifests, unchanged
seeds/counts/splits and early duplicate rejection. The
[[references/rebuild-data-preparation-preserves-replay-identity|new Reference]]
records the 252-ID offline inventory match, synthetic tests and source-bound
check receipt. The three task briefs locate volume 4llk4uaf20, specify v4 native
preparation and seed replay requirements, and recommend four compact opponents
without importing them. Current pod reachability and raw-payload admission remain
unverified. The [[references/bc-bootstrap-uses-native-replay-features-and-current-heads|historical BC Reference]]
corrects its omitted-unit claim: inserted Python None is rejected by the codec,
not encoded as NONE. No engine, bot, exporter or preparation code was imported;
no training, download or network operation ran.
## 2026-09-29 — Close Codex's Task 2.3 verification edit on the heads branch

Codex verified `0e989a1..5ec3af1` (APPROVE WITH EDITS; `ops/rebuild-2026-09-29/codex/verify-2.3-r1.md`, evidence in `ops/rebuild-2026-09-29/codex/verify-2.3-heads/`). No functional defect. One wording correction: `forward` is not host-sync-free, because packed trunk dispatch sizes its packing on the host and oversized packed batches transfer per-row token counts. The `forward` docstring, `docs/model-architecture.md` and the [[references/kaggriculture-grammar-heads-sit-behind-isaiahs-actor-projection|heads Reference]] now say sampling adds no policy-validation host synchronization. The Reference also records Codex's checks and its source-inspection-only limit on market-noise independence. No code behavior changed.

## 2026-09-29 — Add the Kaggriculture grammar action heads (Task 2.3)

The new [[references/kaggriculture-grammar-heads-sit-behind-isaiahs-actor-projection|heads Reference]] records Task 2.3 as the approved brief v3 specifies. It covers Isaiah's `3D → D` actor input projection; a `KaggricultureGrammarActor` with nine `OutputProjectionMLP` slot heads and prefix embeddings; typed `GrammarTables`, with synthetic expected tables and a named native hook for Task 1.2; exact coupled-Gumbel HIRE sampling; and same-path replay with support, length and canonical flags checked in one host transfer. It also covers the §9 overflow guards (trunk width by enumeration, packed chunking, head-extent chunking). The preset has 6,252,223 parameters, inside the 6–10M budget at depth 8. `tests/kaggriculture`: 256 passed, 1 skipped (waiting on Task 1.2). `just py-prepare`: 979 passed, 4 skipped. Four deliberate mutations failed their tests and were restored byte-for-byte. The [[references/kaggriculture-encoder-reuses-isaiah-stateless-layers|encoder Reference]] now says the packed path chunks instead of raising. On `kg/rebuild-heads`, pending Codex verification.

## 2026-09-29 — Narrow the orchestration-receipt claim in the codex exec workflow

Codex's second cookbook-refresh review (P3) noted that an exit status proves only the child's outcome. The [[workflows/run-codex-exec-with-closed-stdin-and-wait-for-its-verdict|codex exec workflow]] now says that tying a parent's return to a child's exit needs a timestamped process-lifecycle log.

## 2026-09-29 — Apply Codex's cookbook-refresh review and record the codex exec workflow

Codex reviewed the refresh commits `bcd9627`, `5247daa`, `92e88e0` and `9bfa0a0` (APPROVE WITH EDITS; local `ops/rebuild-2026-09-29/codex/verify-cookbook-refresh-r1.md`). All seven edits are applied:
- The [[references/model-only-sps-ceiling-bounds-per-rank-throughput|model-only SPS ceiling Reference]] no longer claims that comparing complete-work SPS with the ceiling attributes cost to the engine, host work or all-reduce. That needs matched trainer phase measurements. It also drops the "only if" explanation for a trainer above the ceiling. Its r2 manifest line now reads 21/21 checksums verified, with 20 non-README artifacts unchanged and the README's checksum changed.
- The [[references/pod-v3-environment-runs-flash-attn-2-8-3-forward-on-sm120|pod environment Reference]], its description and its index line drop the "within BF16 spacing" claim, because the magnitude of the largest-error element was not retained. The measured maxima and tolerance results stay. The hard-link hazard is now scoped to two sampled cross-venv inode checks, with uv-cache causation inferred (the follow-up receipt is added as a source). "The heads did not exist" now reads "the actor heads did not exist (the critic head did)".
- `ops/rebuild-2026-09-29/plan.md` now labels the parent-return cause of the two incomplete Codex reviews as operator-reported and inferred.
- The new [[workflows/run-codex-exec-with-closed-stdin-and-wait-for-its-verdict|codex exec Workflow]] records the adaptation from `9bfa0a0`: stdin, foreground, verdict and resume rules, with the transcripts I checked and their limits.

## 2026-09-29 — Correct the Task 0.3 Rust status and the root index's References scope

The [[references/failed-training-reports-status-before-distributed-cleanup|failure-reporting Reference]] and its index line still said Claude's cross-review was pending and Rust parity was blocked (148 passed, 7 failed). Both contradicted the Task 0.3 receipt (`ops/rebuild-2026-09-29/0.3-results.md`, "Claude review") and the earlier log entry: the review found no blocker, and with the git-ignored Orbit fixtures copied in, `cargo test` gave 155 passed, 2 ignored. The [[index|root index]] no longer says every Reference describes `kg/reference-2026-09-29`; the References index states each note's scope. Moving current-tree notes out of that index's "Reference branch" section is left until the in-flight lanes land.

## 2026-09-29 — Record the model-only SPS ceiling and fix non-resolving ops paths

The new [[references/model-only-sps-ceiling-bounds-per-rank-throughput|model-only SPS ceiling Reference]] promotes the component probe (`cb4af49`, `ddf1fb2`; Codex `verify-sps-ceiling` r1 APPROVE WITH EDITS, r2 APPROVE; the ATEN A/B's default arm reproduced its update walls). It is a scoped upper bound with an engine budget of 53–116 µs per env step, not a throughput result, and it is not rankable.
- The [[decisions/throughput-means-correct-complete-work|throughput Decision]] cited `ops/gpu-sps-2026-09-29/results.md` as if it were in the current tree; it exists only on `kg/reference-2026-09-29`. It now says so and names the rebuild's component Reference.
- The [[decisions/start-multi-gpu-qualification-with-two-ranks|multi-GPU Decision]] scopes its host-history paths the same way. It now names the memory metric as the plan's `max_memory_allocated` (not the owner's words), and records that reserved memory and torch's 94.97 GiB denominator differ from that target.

## 2026-09-29 — Record the pod's flash-attn 2.8.3 environment (Phase 6.0)

The new [[references/pod-v3-environment-runs-flash-attn-2-8-3-forward-on-sm120|pod environment Reference]] promotes the Phase 6.0 receipts (`4fd40c7`, `78df33c`; Codex `verify-flash-attn` r1 APPROVE WITH EDITS, r2 APPROVE). It records the separate venv and its versions, the forward-only flash evidence, and three hazards: the torch 2.9 wheel is a mutable release asset that `uv.lock` does not pin, the old and new venvs share hard-linked files, and two custody claims are operator-reported. Trunk numerics stay unqualified. The compiled-GEMM Reference's "no flash-attn" limit is now stale; its revision is left to the ATEN lane. Evidence: `ops/rebuild-2026-09-29/results.md`, "Phase 6.0".

## 2026-09-29 — Tighten the compiled-GEMM Reference after Codex's second review

Codex re-verified the GEMM-limit evidence (APPROVE WITH EDITS, no blocking findings; `ops/rebuild-2026-09-29/codex/verify-gemm-limits-r2.md`). The [[references/compiled-gemm-template-overflows-above-2-21-rows|compiled-GEMM Reference]] and `results.md` now:
- separate the emitted `INDEX_DTYPE` (an all-buffer storage check) from the template size-arg dtype (output numel);
- limit the "every failing graph" claim to kernels attributable from retained code, since two failures occurred during autotuning;
- write L_in as floor(2³¹/K);
- state the design bound as the strict < 2³¹ the guard implements.

The run statement no longer implies Triton 3.5.0 was the installed version.

## 2026-09-29 — Measure the compiled-GEMM limit at the rebuild's shapes (Codex-verified)

A bounded probe ran on GPU 0 of the running pod (GEMM-limits run statement in `ops/rebuild-2026-09-29/run-statements/`). Codex verified it (`ops/rebuild-2026-09-29/codex/verify-gemm-limits-r1.md`, APPROVE WITH EDITS), and all its findings are applied. This revises the [[references/compiled-gemm-template-overflows-above-2-21-rows|compiled-GEMM Reference]]:
- **Input-side overflow.** At these shapes the overflow is on the input side: the A-load wraps once M·K > 2³¹ while the template size argument stays int32. Input-wide GEMMs failed at L_in+1, and output-wide GEMMs stayed correct to M·N = 2³². The universal `M·max(K, N)` rule is withdrawn. M × max(in, out) remains the design bound, justified by backward reading forward outputs as inputs.
- **Silent in the trunk.** With the guard off, the real trunk corrupts silently at L+1. The guard is measured correct at L−1 and rejects L with zero trunk calls. Unguarded-at-L and real-trunk backward are inferred, not measured.
- **Production compliance is unproven.** Kaggriculture is not yet wired through `ModelConfig`, the factory or `configure_model_compile`.
- **No flash-attn on the pod.** The pod venv has no `flash-attn`, and its config had `force_flash_attn: false`. The plan's Phase 6 now installs and verifies it before any qualification.
- **Analyzer and custody fixes.** `analyze_kernels.py` was fixed and its summaries regenerated. The git-archive hash was reconciled: `--prefix` accounts for the difference.

Full evidence is in `ops/rebuild-2026-09-29/results.md`.

## 2026-09-29 — Close Codex's lane B re-verification edits on the model branch

Codex re-verified `4fdb526` (APPROVE WITH EDITS; `ops/rebuild-2026-09-29/codex/verify2-lane-B-r1b.md`). Three edits: `check_contract` now rejects −1 on the `tiles_int` count channels (yield, unwatered, unfed) while keeping the day/deadline sentinels; the contract-valid fixture uses the engine's `farmHandCostMult × fib(hires_today)` next-hire cost and rejects a zero configured order limit; the critic tests assert hand-computed, nonuniform winner probabilities and changed-seat responsiveness, and fail when critic logits are forced to zero (3 failures, source restored). `just py-prepare`: 884 passed, 3 skipped. The [[references/kaggriculture-encoder-reuses-isaiah-stateless-layers|encoder reference]] is unchanged: its claims were already scoped to these checks.

## 2026-09-29 — Close Codex's Task 2.1 verification findings on the model branch

Codex's [verification](../ops/rebuild-2026-09-29/codex/verify-stream-b-2.1.md) approved Task 2.1 with edits. `check_contract` now requires leading dims exactly `[E, 2]` and per-field bounds (non-negative counts, ranks, globals and order limits; −1 tile sentinels and signed `market_int` still pass). The synthetic fixture now writes each seat's legal view of one game. New tests cover packed dispatch, distinct-mask chunk slices, marker-checked offsets, the mypy generics probe, and initialization, API and Muon membership, each seen failing against the old code or a deliberate one-line mutation (the compile-key and SiLU checks were not mutation-tested). The [[references/kaggriculture-encoder-reuses-isaiah-stateless-layers|encoder Reference]] records the checks: `tests/kaggriculture` 155 passed, `just py-prepare` 878 passed, 3 skipped.

## 2026-09-29 — Add Isaiah's critic to the Kaggriculture model (Task 2.2)

The [[references/kaggriculture-encoder-reuses-isaiah-stateless-layers|encoder Reference]] now covers the critic: an `OutputProjectionMLP` shared over the self and opponent critic-value tokens, a winner softmax, and value `2p(self) − 1` from each seat's own view, with Isaiah's output gain and Muon exclusion. `just py-prepare` gives 767 passed. This is on `kg/rebuild-model`, pending Codex verification.

## 2026-09-29 — Build the Kaggriculture encoder on Isaiah's layers (Task 2.1)

The new [[references/kaggriculture-encoder-reuses-isaiah-stateless-layers|encoder Reference]] covers the contract types, the PEP 696 generic model base, and an encoder built entirely from Isaiah's classes, with one-hot stems, per-role tokens, a typed trunk config, his initialization, and the compiled-GEMM chunking guard. `just py-prepare` gives 762 passed. The work sits on branch `kg/rebuild-model`, pending Codex verification before merge.
## 2026-09-29 — Correct the kaggle-environments pin record after Codex verification

Codex verified `fb65e1f` (APPROVE WITH EDITS): lock reproduced byte-for-byte, `just prepare` green, engine hash matches. The [[decisions/restart-the-port-from-isaiahs-clean-base|restart Decision]] now states 55 other changed dependencies (not 50), notes that `click` is shared with W&B, and names the 2 ignored Rust tests as action-angle audits rather than parity tests. Evidence: `ops/deps-independent-verification-fb65e1f/verification.md`.

## 2026-09-29 — Pin kaggle-environments 1.32.7

Owner: “Isaiah's lockfile pins kaggle-environments 1.29.0, which lacks Kaggriculture … we will have to update it.” The [[decisions/restart-the-port-from-isaiahs-clean-base|restart Decision]] records the pin: 1.32.7 via uv, the git source removed, its engine hash matching the Rust kernel, and Isaiah's torch/triton/flash-attn pins unchanged. `just prepare` passes: Rust 155/2 ignored, Python 722/3 skipped. A contract Markdown lint fix is included.

## 2026-09-29 — Record model size, torch pin and RTX PRO 6000 resource fit

Owner decisions:
- “6-10M ok as long as topologies aligned with isaiah”, recorded in the [[decisions/the-policy-is-stateless-and-observation-only|stateless Decision]]: Isaiah's ladder width and ratios, with depth reaching the budget.
- “let's stick with 2.9”, and fit to RTX 6000 rather than Isaiah's B200s, recorded in the [[decisions/start-multi-gpu-qualification-with-two-ranks|multi-GPU Decision]]: global config unchanged, per-rank shapes and spm/accumulation fitted by memory smoke, ≤ 85% peak.

The Kaggriculture contract reaches v4 (Codex accept-with-edits applied), with a companion section in `docs/rl-api-specs.md`.
## 2026-09-29 — Harden the rules-kernel checker after verification

Verification round 1 found that manifest declarations alone could authorize
LICENSE edits or drop the Task 1.1 provenance appendix. The
[[decisions/restart-the-port-from-isaiahs-clean-base|restart Decision]] now records
the fixed editable set, derived lockfile and pinned appendix hash. Eight
`check()`-level regressions were added test-first; three failed before the fix.
Engine tests (59), the checker and `just prepare` pass. The receipt now matches
the committed 59-test state, and four ignored replay receipts are tracked.

## 2026-09-29 — Rebuild and verify the trimmed rules kernel

The [[decisions/restart-the-port-from-isaiahs-clean-base|restart Decision]] records
Task 1.1's reviewed implementation: 12 retained/113 excluded reference paths,
exact seven-line trim and `c4b9bac5…` hash, Python checker/pytest, and separate
engine fmt/Clippy/test invocations. All 59 engine tests pass; four native-reset
replays cover 2,876 transitions and 2,880 snapshots, checking key order in
every public and private object (extended in Claude's review, test-first). Broken comparator and
corrupted-snapshot tests fail before repair. Root Rust remains 155 passed/two
ignored; final prepare passes 769 Python tests with three platform skips.
Inherited formatting and six Clippy style findings require two exact
formatter exclusions and three engine-only lint allowances, re-denied in authored
replay code. The `ops/rebuild-2026-09-29/1.1/results.md` receipt records all
commands, file inventory, independent review, limitations and preserved dirty
history. No training or network operation ran; Claude reviewed and committed it.

## 2026-09-29 — Specify the trimmed rules kernel before implementation

The [[decisions/restart-the-port-from-isaiahs-clean-base|restart Decision]] links
the Task 1.1 brief for Claude's review. The audit accounts for all 125 reference
engine paths, pins 12 retained files, identifies seven source-line removals,
and requires new direct replay tests because the retained 41 unit and nine RNG
tests never read the four episode traces. Standalone packaging is proposed to
avoid L4 until a real root consumer needs the engine. All reference crates are
cached; offline root checks pass 155 Rust tests with two ignored and 722 Python
tests with three skips. This change adds only the brief, baseline receipt and
cookbook record; no kernel or test implementation is installed.

## 2026-09-29 — Rebuild failure status and traceback ordering on the clean base

Task 0.3 implements the reference plan's logging fix under the
[[decisions/restart-the-port-from-isaiahs-clean-base|restart Decision]]. The new
[[references/failed-training-reports-status-before-distributed-cleanup|failure-reporting Reference]]
records keyword-only logger exit codes, failed W&B finalization and rank-tagged
stderr before process-group destruction. TDD goes from 9 failures/2 passes to
11 passes; the Python suite and final py-prepare each pass 722 with 3 skips.
The 7 Rust failures were missing generated Orbit fixtures in the fresh worktree; with them copied in, Claude's rerun gives 155 passed, 2 ignored.
The distributed test mocks clean-base CUDA/NCCL rather than importing the old
CPU-DDP hook. No training, evaluation or network operation ran.

## 2026-09-29 — Trace the CUDA illegal memory access to a compiler GEMM overflow

A blocking rerun of the reference PPO-from-BC setup on pod `w7ia3zvxqsvs3g` reproduced the fault after 3 iterations. It points both ranks at an Inductor max-autotune Triton GEMM template in the trunk MLP, whose 32-bit offsets wrap once rows × inner dim exceeds 2^31. A controlled probe matched eager at 2,088,960 rows and faulted at 2,105,344. The new [[references/compiled-gemm-template-overflows-above-2-21-rows|compiler overflow Reference]] records the mechanism, and why the reference's first-update log-ratio (−3.77) and BC→PPO deterioration came from silent corruption. It adds the rebuild requirements: a trunk-size guard and a first-minibatch log-ratio alarm. The BC, PPO and restart notes no longer call the fault unresolved.

## 2026-09-29 — Plan the rebuild with explicit reference-branch dispositions

Owner: “Please make plans adjusted to be work with clean base, and make sure we are utilizing the reference branch properly, without blindly copying.” `ops/rebuild-2026-09-29/plan.md` classifies every reference component as vendored-and-pinned (rules kernel only), port-after-review, rebuild-with-oracle, or reference-only. It maps 15 reference lessons to tasks, carries Isaiah's principles including the same layer topology, and splits the work between Claude and Codex with contract-first, cross-reviewed tasks. The [[decisions/restart-the-port-from-isaiahs-clean-base|restart Decision]] records the rule. No code has changed yet.

## 2026-09-29 — Restart the port from Isaiah's clean base

Owner: “restore to starting points 32b3ec900ad406eedd965f53a1a0f4490d31c589 and work again … I want you & codex started from a clean state. Carry the cookbooks with you with .claude/.codex setup.”
- Local `main` and `kg/isaiah-gap-closure` are reset to `32b3ec9`; `origin` is unchanged.
- The prior port is kept as branch `kg/reference-2026-09-29` and tag `kg-reference-2026-09-29` (`65f0eac`).
- The cookbook, `.claude/`, `.codex/`, `AGENTS.md`, the Base, `ops/pre-commit`, `ops/cookbook-setup-checks.md` and `.gitignore` carry over.
- 88 sources whose files no longer exist now point at the reference branch.

The new [[decisions/restart-the-port-from-isaiahs-clean-base|restart Decision]] records the owner directives: same layer topology with only game I/O differing (also in the [[decisions/the-policy-is-stateless-and-observation-only|stateless Decision]]), a required BC rerun, 2/4-rank RTX PRO 6000 verification (also in the [[decisions/start-multi-gpu-qualification-with-two-ranks|multi-GPU Decision]]), and working with Codex. References now describe the reference branch until rebuilt.

## 2026-09-29 — Remove contradictions and trim operational history

Before gap-closure work starts, the owner asks to fix cookbook contradictions and trim as needed. Owner directives recorded: “The plan must respect Isaiah principals while on different game (Kaggriculture)” in the [[decisions/v3-reuses-isaiah-infrastructure-with-kaggriculture-semantics|scope Decision]], and “Make sure we are refering to his stateless approach” in the [[decisions/the-policy-is-stateless-and-observation-only|stateless Decision]] (StatelessTransformerV1 is the reference, never the recurrent model). Contradictions fixed:
- The [[decisions/preserve-the-ppo-recipe-across-bc-bootstrap|recipe Decision]]'s “do not change PPO batches” now applies to the BC period only.
- The [[references/shared-ppo-adapts-game-batches-without-a-second-loop|PPO Reference]] now states the 20M cadence instead of 100k.
- The [[references/bc-bootstrap-uses-native-replay-features-and-current-heads|BC Reference]] records the PPO-from-BC failure instead of “pending”.
- Stale “being implemented” wording is corrected.
- The multi-GPU Decision is renamed [[decisions/start-multi-gpu-qualification-with-two-ranks|two ranks]].

The PPO Reference is trimmed from 29 KB to its current contract, with measurements linked to receipts; pod history moves to `ops/cookbook-cleanup-2026-09-29/host-history.md`. Pre-edit bytes and SHA-256 are in `ops/cookbook-cleanup-2026-09-29/before/`. No note was deleted.

## 2026-09-29 — Resolve recipe choices toward Isaiah, not the owner

Owner: “these are NOT my decisions please patch the cookbook or whatever, and align to Isaiah later on in the list.” The new [[decisions/recipe-choices-align-to-isaiah-without-owner-escalation|alignment Decision]] makes optimizer-step cadence and critic/value-distillation semantics implementer choices resolved toward Isaiah. It withdraws the records that treated 4,096 envs/rank with a fixed minibatch as an owner-retained constraint of the aligned recipe. The [[decisions/preserve-the-ppo-recipe-across-bc-bootstrap|recipe Decision]], [[decisions/v3-reuses-isaiah-infrastructure-with-kaggriculture-semantics|scope Decision]] and [[references/shared-ppo-adapts-game-batches-without-a-second-loop|PPO Reference]] now link to it. No config value changed and no run started; both alignments are queued after the crash and failure-status items.

## 2026-09-29 — Separate retained core from incomplete integration

The owner asks whether the Isaiah base was broken. The [[references/shared-ppo-adapts-game-batches-without-a-second-loop|PPO Reference]] records byte/function-body comparisons and259 passing software regressions. Core reuse remains, but missing teachers, workload/parameter semantics and the unresolved KG CUDA failure prevent whole-pipeline qualification. W&B finish/cleanup behavior is inherited; no GPU root-cause attribution or wholesale replacement is justified by these checks.

## 2026-09-29 — Correct parameter equality versus effective workload

The owner supplies an external review; the [[references/shared-ppo-adapts-game-batches-without-a-second-loop|PPO Reference]] confirms the missed2048-versus16 optimizer updates/iteration and fourfold scheduler data-exposure difference. A new resolved-config/workload audit also records normalization windows, Muon grouping and FlashAttention requirements. Compile settings had already been corrected; BC inherited a previously PPO-trained critic. Batch alternatives remain unadopted and no new run starts during this fact-check.

## 2026-09-29 — Inventory rejected alignment gaps and partial parameter repair

Owner rejects leaving teacher adapters unsupported and requests all gaps. The [[references/shared-ppo-adapts-game-batches-without-a-second-loop|PPO Reference]] links a14-item inventory and records two partially aligned GPU YAMLs, exact optimizer/scheduler checks and94 passing existing tests without a GPU qualification claim. Teacher action/cache/value integration, replay export, runtime correctness and BC quality remain open. Final old-run traceback shows CUDA illegal memory access followed by NCCL timeout, correcting the earlier OOM suspicion. The [[decisions/preserve-the-ppo-recipe-across-bc-bootstrap|recipe Decision]] preserves historical configurations while allowing the new alignment direction.

## 2026-09-29 — Correct the upstream hyperparameter alignment claim

A fetched, pinned upstream comparison in the [[references/shared-ppo-adapts-game-batches-without-a-second-loop|PPO Reference]] distinguishes reused infrastructure from changed recipes. Isaiah baseline Adam and scaling6m enable LR warmup/cosine; the actual v3 run disables scheduling and changes other coefficients. Upstream LR warmup is not critic-only warmup. No configuration was silently changed or run restarted during the audit.

## 2026-09-29 — Launch separately requested PPO from BC best

Owner explicitly requests half-hour PPO from the best BC checkpoint. The [[decisions/preserve-the-ppo-recipe-across-bc-bootstrap|recipe Decision]] records unchanged configuration, fresh optimizer, separate identity and preserved original incumbent artifacts. The [[references/bc-bootstrap-uses-native-replay-features-and-current-heads|BC Reference]] distinguishes the interrupted BC run from a completed75-minute result. Best weights are snapshotted before stopping BC; quality limitations remain and automatic handoff stays disabled.

## 2026-09-29 — Separate BC quality from PPO continuation

Owner: “接回ppo是另外一件事，有做好bc嗎？” The [[decisions/preserve-the-ppo-recipe-across-bc-bootstrap|configuration Decision]] removes the automatic continuation interpretation. The [[references/bc-bootstrap-uses-native-replay-features-and-current-heads|BC Reference]] records a verified remote handoff guard, preserved BC run and best checkpoint, and clear overfitting after the best sampled holdout loss. Pipeline smoke success does not establish BC quality; full holdout/action/economic checks remain missing.

## 2026-09-29 — Diagnose market depletion and qualify the BC data path

The [[references/bc-bootstrap-uses-native-replay-features-and-current-heads|BC Reference]] records zero-bank versus market-disabled/PASS controls, the explicit-state Rust encoding boundary, exact admitted replay labels and a separate current-model BC objective.814 Python/155 Rust checks and eager/compiled two-rank smoke pass; the compiled holdout NLL moves3.925→3.853 without changing critic-head weights.252 sampled public episodes from the owner’s first volume admit158,772 paired turns and explicitly reject22,416. The75-minute run and economic handoff are still pending outcomes, not promotion claims; original PPO configuration and checkpoint custody remain preserved.

## 2026-09-29 — Preserve the PPO recipe before BC bootstrap

Owner requests zero-bank diagnosis, an hour-plus BC using v2 experience, and remembering the training setup. The [[decisions/preserve-the-ppo-recipe-across-bc-bootstrap|configuration Decision]] saves the complete verified YAML and separates BC data/optimizer/selection from the existing PPO run. The current v3 model remains; the chosen75-minute duration is an implementation interpretation.

## 2026-09-29 — Verify uninterrupted half-hour execution

The [[references/shared-ppo-adapts-game-batches-without-a-second-loop|PPO Reference]] closes the repeated-execution gap with30m38s, nine full2,048-optimizer-step/rank updates and normal exit without new KL stops, timeout or OOM. The last six updates measure3,380.64 game SPS across two PRO6000 GPUs; wall throughput including evaluation/startup is2,567.24. All nine128-game evaluations have zero banks,50% win rate and no promotion, so execution success is not evidence of stronger play. The pod is retained as requested; source-bound telemetry and final/best custody are in the continuation receipt.

## 2026-09-29 — Close the repaired evaluation run with scoped results

The [[references/shared-ppo-adapts-game-batches-without-a-second-loop|PPO Reference]] records813 passing software tests/three hardware skips and a normal-exit two-rank repair run: two128-game evaluations at50%/49.61%, no promotion, unchanged incumbent and2,097,152 cumulative game transitions. One full2,048-step PPO update measures2,290.76 game SPS across two PRO6000 GPUs; the earlier3,679.43-SPS update only executes418 optimizer steps due KL stopping. The [[references/reward-reuse-preserves-objective-and-critic-semantics|reward Reference]] records observed.25 shaping reward without a learning claim. The original half-hour failed/interrupted; the repair is a separate bounded attempt. Checkpoints and W&B telemetry remain source-bound, and the replacement pod stays RUNNING at the owner’s request.

## 2026-09-29 — Separate evaluation width after a measured collective timeout

The resumed4,096-game root-only evaluation exceeds rank1’s600-second NCCL broadcast timeout. The [[references/shared-ppo-adapts-game-batches-without-a-second-loop|PPO Reference]] records optional `rl.eval_n_games`, with128 selected in both GPU recipes while training stays4,096 env/rank. Unset preserves starter behavior; raw-bank70% promotion remains intact. Smaller selection samples change variance. The failed attempt, source change and bounded repair verification are kept separately in the run receipt.

## 2026-09-29 — Correct a premature evaluation interruption

The [[references/shared-ppo-adapts-game-batches-without-a-second-loop|PPO Reference]] records the agent’s premature stall inference during4,096-game root-only evaluation. A full524,288-step PPO update and checkpoint succeeded; the interruption traceback shows native env.step and no OOM. Sparse progress output did not prove a hang. Resume uses the same current/optimizer/incumbent pair and W&B ID with approximately18 minutes remaining; no production code or promotion criterion is changed.

## 2026-09-29 — Retain the training pod and replace the slow setup host

Owner: “不用關pod”, then “換主機，訓練後保留新 pod”. The [[decisions/start-multi-gpu-qualification-with-two-ranks|compute Decision]] now retains the training host after bounded runs. US allocation attempts failed; the selected same-price EUR-IS-1 host downloads the2MiB probe in0.349s versus21.102s on the discarded EUR-IS-2 host. The slow owned host is confirmed EXITED and the selected host stays available after training. No training result is inferred from installation or allocation.

## 2026-09-29 — Enable econ0.2 before the half-hour run

The owner asks whether econ shaping0.2 is enabled. It was implemented but off in the preceding verification run. The [[references/reward-reuse-preserves-objective-and-critic-semantics|reward Reference]] and both GPU recipes now explicitly select coefficient0.2, starvation/drought weights4/1, cap0.25 and terminal scale0.75, with gamma1 and ineffective penalties off. Raw-bank promotion is unchanged; learning effects remain to be measured.

## 2026-09-29 — Default GPU collection to4,096 environments per rank

Owner: “4096吧，default.” and “讓他跑半小時看看”. Both GPU configs now use4,096 environments/rank while retaining128-seat PPO minibatches/rank and accumulation2. The [[references/shared-ppo-adapts-game-batches-without-a-second-loop|PPO Reference]] and README distinguish the new default from the optional32-environment comparison and prior-model capacity evidence. A bounded two-PRO6000 half-hour W&B run uses the canonical trainer and original promotion logic; its outcome belongs in the run receipt. The later owner instruction “不用關pod” keeps the allocated pod running after training stops.

## 2026-09-29 — Verify promotion boundaries and expose bank perspectives

The [[references/shared-ppo-adapts-game-batches-without-a-second-loop|PPO Reference]] records bank snapshots/terminal aliases, candidate-relative evaluation banks and promotion/game-count telemetry without adding per-turn transport work. The latest8,353,727-parameter model passes a two-PRO6000 canonical DDP contract probe: reject69.9%, promote70%, synchronized incumbent/candidate weights, separate best checkpoint and exact model/optimizer resume. Injected scores are isolated test evidence. Full Python preparation passes811 with three hardware/backend skips. The genuine compiled W&B run `spoon/kg-v3/vw4ohx4a` finishes13 updates/13,312 game transitions including same-ID resume. API read-back verifies bank metrics and three50% evaluations without promotion; incumbent remainsstep0. Direct run navigation loads the dashboard after the initial home/project date-parser error. Final/current-best artifacts are hash-verified locally and the owned GPU pod is confirmed EXITED.

## 2026-09-29 — Restore the independent player token

Owner: “修復差異吧 player token”. The [[references/explicit-game-tokens-and-grammar-replace-orbit-heads|model Reference]] records the restored player query and entity/player/plan projection, shared private-seat parameters, batched execution, 8,353,727 parameters and explicit old-checkpoint incompatibility. The focused suite passes33 cases; canonical py-prepare passes811 with three hardware/backend skips after repairing a stale assertion for concurrently added evaluation-log fields. Model docs and the [[references/full-turn-intentions-coordinate-batched-action-heads|optional coordination candidate]] distinguish current structure from prior benchmarks; U/A remains unimplemented. Original dirty bytes, checks and remaining GPU/strength limits are preserved in the repair receipt.

## 2026-09-29 — Audit scratch and plan alignment precisely

The [[references/explicit-game-tokens-and-grammar-replace-orbit-heads|model Reference]] now distinguishes retained scratch/plan roles, initialization and shared-attention paths from changed parameter tying and actor inputs. Pinned Isaiah uses entity+player+plan inputs; Kaggriculture uses separately encoded seat views and entity+plan, omitting a separate player readout. Source inspection supports the comparison, not lossless equivalence or a learning benefit. The audit records exact paths, masks and limits; no model code or training changed.

## 2026-09-29 — Measure environment capacity while preserving PPO minibatches

The owner asks to scale environments only and then test the per-GPU limit with minimal workflow. The [[references/shared-ppo-adapts-game-batches-without-a-second-loop|PPO record]] preserves128-seat minibatches/rank and512-seat global optimizer batches: combined two-PRO6000 throughput rises from3,627 at8 environments/GPU to7,167 at4,096;8,192 fails with CUDA OOM during warmup. The [[references/explicit-game-tokens-and-grammar-replace-orbit-heads|model]] and [[references/native-game-semantics-use-v3-owned-buffers|native]] records close their bounded fused-GPU execution gaps, retaining density, duration and learning limits. The [[decisions/start-multi-gpu-qualification-with-two-ranks|compute record]] confirms the owned pod stopped after58 artifact hashes matched. Full software checks remain805 Python passes,155 root Rust passes and120 engine passes. No larger PPO batch, new profiler or additional subagent workflow is introduced.

## 2026-09-29 — Separate coordinated behavior from an extra coordination module

The owner questions whether Isaiah needs a special coordinator and notes that our scratch/plan tokens remain. A pinned upstream and current-source review confirms both shared-context paths. The [[references/full-turn-intentions-coordinate-batched-action-heads|coordination Reference]] and design withdraw the assistant's prior adoption recommendation: U/A remains an optional hypothesis, not a required final-system component. Separate sampling does not prevent observation-based complementary actions. The comparison preserves source anchors, limits, reopening conditions and exact pre-correction dirty history; this records an assistant correction, not owner adoption or new training evidence.

## 2026-09-29 — Prepare the complete efficient coordination design

The [[references/full-turn-intentions-coordinate-batched-action-heads|coordination proposal]] specifies one observation trunk, sampled whole-turn intentions, one action-token communication block and final batched heads. Engine review preserves worker/market chains and unknown rival settlement; independent probability and architecture reviews resolve joint U/A density, STOP handling and per-market-position states. Exact finite-space gradient/probability checks and a parameter envelope are retained. This is a complete design response to the owner, not adoption, implementation, a learner run or a speed/strength result.

## 2026-09-29 — Keep the original PPO minibatch while widening collection

The owner explicitly requests environment-only scaling. The [[decisions/v3-reuses-isaiah-infrastructure-with-kaggriculture-semantics|scope Decision]], [[references/shared-ppo-adapts-game-batches-without-a-second-loop|PPO record]], README and GPU receipt now make the fixed-PPO 32-environment configuration the sole active scaling experiment. The four-segment proposal is abandoned without execution; it was a throughput heuristic, not a requirement. Original effective optimizer batch is preserved, while collection batching/freshness still changes. No speed/quality benefit or default promotion is inferred.

## 2026-09-29 — Recheck the complete corrected Python source

The [[references/shared-ppo-adapts-game-batches-without-a-second-loop|shared PPO record]] now distinguishes the current 805-pass/three-hardware-skip preparation from the initial 772-case receipt. Formatting, Ruff, Python 3.11 syntax, strict typing over 58 source files and documentation freshness pass. The preserved GPU-episode log pins the executed checks; combined runtime and quality claims remain separately scoped.

## 2026-09-29 — Qualify the fused native lifecycle locally and bind GPU telemetry to work

The [[references/native-game-semantics-use-v3-owned-buffers|native record]] identifies the typed fused lifecycle, preserved transactional staging, checked array boundaries, exact final CPU test scope and paired fixed-action measurements. These do not substitute for the pending combined GPU run. The [[references/shared-ppo-adapts-game-batches-without-a-second-loop|PPO record]] adds the persistent owned telemetry monitor, warmup-excluding receipt window and device-snapshot limits; results are not inferred for earlier unmonitored runs. Exact implementation/evidence paths remain in the GPU receipt.

## 2026-09-29 — Require complete final delivery

Owner: “這裏沒有第一版，第二版，只有最終版”. The [[decisions/v3-reuses-isaiah-infrastructure-with-kaggriculture-semantics|scope Decision]] withdraws the assistant's seed-only staged-delivery proposal and requires one complete final adaptation. Internal checks and revisions serve that delivery; required coordination and integration cannot be deferred. The correction adopts no proposed allocator, performance threshold or unverified completion claim.

## 2026-09-29 — Measure batched heads without changing the native binary

The [[references/explicit-game-tokens-and-grammar-replace-orbit-heads|model record]] now links a finite two-GPU heads-only result: 1,774.752 game SPS versus 310.819 for the rejected decoder at equal nominal five-update work, with independent weights/action distributions and no quality claim. Full-model CUDA diagnostics preserve raw BF16 errors and synthetic-capacity limits. The GPU receipt adds completed raw-trace custody and actual named-range launch attribution. [[references/shared-ppo-adapts-game-batches-without-a-second-loop|PPO guidance]] records the explicit 32-environment/four-segment experiment and its fourfold optimizer-batch change; README removes stale old-model/no-GPU claims. Fused-native final checks and combined GPU measurement remain pending; no standing board or empirical Lesson is created.

## 2026-09-29 — Attribute the rejected decoder and preserve executed-work scope

The [[references/shared-ppo-adapts-game-batches-without-a-second-loop|PPO record]] adds native-binary/attention/profiler identity, executed optimizer/KL work and actor/frame denominators. The GPU receipt independently reads post-warmup Nsight CSVs: decoder host ranges dominate the outer iteration-range sum, while aggregate launch counts remain unattributed to individual modules. Nested host ranges are not GPU utilization. The [[references/native-game-semantics-use-v3-owned-buffers|native record]] records compact-mask/HIRE checks and pending fused lifecycle; the [[references/explicit-game-tokens-and-grammar-replace-orbit-heads|model record]] distinguishes preliminary new-model CPU checks from unmeasured replacement GPU performance. The [[decisions/start-multi-gpu-qualification-with-two-ranks|compute Decision]] records the owned running pod; the [[workflows/profile-cuda-bottlenecks-with-nsight-systems|profiling Workflow]] retains capture limits. No old result qualifies the replacement or its playing strength.

## 2026-09-29 — Reject the Python-driven decoder and correct the entire pipeline

The owner rejects Python-driven autoregressive heads and mere PyO3 interface alignment; the [[decisions/v3-reuses-isaiah-infrastructure-with-kaggriculture-semantics|scope Decision]] now requires efficient Isaiah reuse across the complete pipeline. The [[references/explicit-game-tokens-and-grammar-replace-orbit-heads|model Reference]] withdraws the old decoder endorsement, links frozen source/docs/contract and returns to draft while batched heads/native masks are implemented. The [[decisions/the-policy-is-stateless-and-observation-only|stateless Decision]] preserves the original interpretation as history and applies the stricter current boundary. `ops/gpu-sps-2026-09-29/results.md` retains two independently measured old-model baselines, exact work denominators and pending profile/correction scope. Measured throughput is unacceptable to the owner; no replacement result, performance floor, Lesson or ranking is invented.

## 2026-09-28 — Verify the repaired pinned Rust toolchain

The [[references/native-game-semantics-use-v3-owned-buffers|native contract]] closes the partial-nightly-installation gap: Rustup repair succeeds, and the original pinned nightly passes the same 155-case `rs-prepare` suite with two existing ignored plus formatting, Clippy and documentation checks. Stable verification remains retained. The final pod inventory confirms only the two preexisting EXITED pods; GPU execution remains unqualified. Repair/check logs are included in the evidence hashes; no code semantics changed.

## 2026-09-28 — Qualify the normal locked environment

The [[references/shared-ppo-adapts-game-batches-without-a-second-loop|PPO contract]] now records completed `uv sync --locked`, 232 project-environment tests, typing over all 58 source files and an actual native PPO CLI update/checkpoint. The earlier incomplete-sync gap is closed; the 772-case suite keeps its separate-environment provenance. Pytest root imports and a shadowed grammar scalar are corrected without changing game semantics. CUDA measurement now validates finite metrics and records effective override/runtime metadata. GPU, container, strength and live hook-trust gaps remain.

## 2026-09-28 — Close source-scoped port records against retained checks

The final `ops/v3-port-checks.md` receipt reconciles the [[references/native-game-semantics-use-v3-owned-buffers|native]], [[references/explicit-game-tokens-and-grammar-replace-orbit-heads|model]], [[references/shared-ppo-adapts-game-batches-without-a-second-loop|PPO]] and [[references/reward-reuse-preserves-objective-and-critic-semantics|reward]] contracts to verified-scoped status: 772 Python cases, 155 starter Rust cases and 120 vendored engine cases pass, with declared hardware skips/ignored cases. The records identify optional reference dependencies, fixture recipes, container path-dependency changes, two/four-rank starting configs and canonical CUDA-only measurement. Default sync was still finishing downloads; executed validation used the documented separate environment. GPU performance, container execution, submission packaging, competitive strength and live hook trust remain unverified. Model timing wording is corrected to means, as the retained JSON specifies. No standing board or empirical Lesson is created.

## 2026-09-28 — Bind two-rank measurement to the canonical trainer

The [[references/shared-ppo-adapts-game-batches-without-a-second-loop|shared PPO record]] adds the initial two-rank config, bounded CUDA-only measurement harness, full-model CPU DDP diagnostic and optional reference dependency extra. Normal training retains incumbent evaluation; only measured invocations disable periodic file/evaluation overhead and state that exclusion. CUDA execution remains unavailable; CPU rejection and readiness do not establish GPU SPS.

## 2026-09-28 — Reduce repeated decoder work with bounded evidence

The [[references/explicit-game-tokens-and-grammar-replace-orbit-heads|model contract]] records masked Gumbel categorical sampling, factored projection, cached grammar/ranges and inactive-branch parameter participation, with19CPU regressions and the measured scalar-extraction reductions. It preserves the exact starter/new-code provenance, sampling RNG difference and remaining STOP/validation synchronization; local timing is not CUDA SPS or policy-strength evidence.

## 2026-09-28 — Preserve incumbent selection and prefer two-GPU qualification

The later owner directive changes the [[decisions/start-multi-gpu-qualification-with-two-ranks|compute preference]] to two RTX 5090/RTX PRO 6000 GPUs first; two allocation attempts failed and no pod was created. The [[references/shared-ppo-adapts-game-batches-without-a-second-loop|PPO record]] names unchanged incumbent/resume function bodies, three real-native regression tests, the initial checkpoint cadence, engine cache keys and opt-in CPU DDP diagnostic boundary. The [[references/native-game-semantics-use-v3-owned-buffers|native record]] adds reusable-buffer optimization, bounded paired CPU adapter evidence and the retained starter fixture/parser repair. Final shared receipt reconciliation remains pending; no CUDA SPS, learning or general scaling claim follows.

## 2026-09-28 — Reconcile source-confirmed integration repairs and bounded checks

The owner reiterates starter architecture reuse and efficient training. The [[references/shared-ppo-adapts-game-batches-without-a-second-loop|PPO contract]] records independent review repairs: bank-based evaluation, retained shaping on truncation and disjoint rank seed streams. [[references/native-game-semantics-use-v3-owned-buffers|Native checks]] report19Python/120Rust; the [[references/explicit-game-tokens-and-grammar-replace-orbit-heads|model record]] reports15CPU tests and8,294,450 parameters, retaining exact provenance and GPU limits. Reward tests and current formulas are reconciled in the [[references/reward-reuse-preserves-objective-and-critic-semantics|reward record]]. Port References remain provisional until the final shared receipt; no learning or throughput gain is inferred.

## 2026-09-28 — Reuse reward semantics, not the v2 model

The owner clarifies “dont take model implementation v2” while permitting economic shaping, own-minus-opponent bank and W/L/D rewards. The [[decisions/v3-reuses-isaiah-infrastructure-with-kaggriculture-semantics|scope Decision]], agent contract and model Reference explicitly retain the starter neural implementation/new game heads. The [[references/reward-reuse-preserves-objective-and-critic-semantics|reward Reference]] preserves exact source formulas/defaults, distinguishes raw margin from dense bank deltas, and identifies critic-range/bootstrap requirements. V3 reward checks and coefficient selection remain separate.

## 2026-09-28 — Record four-rank GPU direction and audit capacity

The owner permits RTX 5090 or RTX PRO 6000 at four/eight ranks and recommends four first. The [[decisions/start-multi-gpu-qualification-with-two-ranks|compute Decision]] records the directive without a scaling claim. Read-only RunPod catalog/capacity reads found no four-GPU stock; no pod was changed. Local W&B credential-file mode was checked without reading or recording its key; live telemetry remains separate. Exact calls and base-price/available-price distinctions are in `ops/cookbook-setup-checks.md`.

## 2026-09-28 — Record each port adaptation as an inspectable contract

Owner: “every adaption you made to this repo, please add a cookbook record.” The agent contract now requires an identified record for every material adaptation. [[references/starter-history-and-knowledge-remain-retrievable|Starter/history/lifecycle]], [[references/native-game-semantics-use-v3-owned-buffers|Rust I/O and game semantics]], and [[references/shared-ppo-adapts-game-batches-without-a-second-loop|shared PPO]] records name code, reasons and qualification gaps. Source inspection is distinguished from executed checks; incomplete port work stays draft. The [[references/explicit-game-tokens-and-grammar-replace-orbit-heads|model/head contract]] explicitly distinguishes current-action prefix state from prohibited between-turn memory; final verification is reconciled before close-out.

## 2026-09-28 — Start v3 with selective discipline and the project lifecycle

The owner directs reuse of the Isaiah starter and useful v2 decisions, then says “just clone the repo in and start adapting”. The [[decisions/v3-reuses-isaiah-infrastructure-with-kaggriculture-semantics|scope Decision]] records inherited repository/change boundaries as implementation choices, not interview answers. Five Decisions retain one trainer, evidence-backed debugging, stateless observation-only behavior, complete-work throughput, and general evaluation/custody. The later owner clarification explicitly retains v3 Rust I/O and PPO; v2 contributes game semantics through the v3 adapter. The [[workflows/profile-cuda-bottlenecks-with-nsight-systems|profiling Workflow]] applies the conditional NVIDIA contract. Each note is indexed; the Base remains derived. No old ranking or empirical result transfers and no empty board is created. Claude/Codex lifecycle scripts and Git pre-commit are installed; actual checks and unavailable runtime trust are recorded in `ops/cookbook-setup-checks.md`.
