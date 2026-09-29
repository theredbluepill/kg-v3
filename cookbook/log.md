# Change log

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
