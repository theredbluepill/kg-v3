# Rebuild phase status

**As of 2026-09-29 22:15 HKT (14:15Z).** The integration branch `kg/isaiah-gap-closure` is at `666deec`. This file is on `kg/merge-custody`, which sits on `666deec` and adds these: the value-gap merge `2f535ea`, the phase-map merge `2ebf538`, the architecture-image record `1df17dd`, the evidence-custody commits `c8aaaac`, `470b527` and `7911b16`, and the custody-review fixes that follow them (including the value-gap r1 edits). Rows marked **merged (custody)** land with that branch.

This is a working artifact, not a durable cookbook claim. Update it at each landing: a merge, a Codex verdict, or a state change. When a row changes, re-check it against git and the cited report. Don't copy a row into a cookbook note without re-checking it. The cookbook stays organised by concept (`cookbook/references/`, `cookbook/decisions/`). This file is the one place that maps those notes and receipts onto the plan's phases (`ops/rebuild-2026-09-29/plan.md`).

## Path conventions

- **Bare paths** are tracked in this tree, relative to the repository root. `rebuild/` is short for `ops/rebuild-2026-09-29/`.
- **`codex/…`** is `rebuild/codex/…`. The custody sweep committed the compact reports for merged work. The inventory and the manifest are described in `rebuild/evidence-custody.md` and `rebuild/evidence-custody.json`.
- **`codex/…` (local)** means the report is not tracked on any branch. It exists only in the main worktree, `/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/codex/`. This covers in-progress work, which lands with its own merge, and reports written after the 13:48Z inventory.
- **`<branch>:<path>`** is a path tracked on that branch but not in this tree.
- **"Merged"** means the commit is an ancestor of `666deec`, or of this branch for the custody rows. Each was checked with `git merge-base --is-ancestor` at 22:10 HKT. Branch heads were read at 22:12 HKT; in-progress branches may have moved since.

State values: **merged**, **merged (custody)**, **approved, not merged** (Codex APPROVE, not on integration), **in review**, **in progress**, **blocked**, **not started**.

## Summary

The done count only includes plan checkboxes whose work is merged. Rows without a plan checkbox (1.1b and the pod evidence runs) are listed in their phase's table but are not counted.

| Phase | Done / total | State | Next landing |
|---|---|---|---|
| 0 Contract and diagnostics | 3 / 3 | merged | None. Contract v4 has no Codex re-review (see 0.1). |
| 1 Engine and native env | 3 / 5 (plus 1.1b, merged) | 1.4 approved, not merged; 1.5 in progress | 1.4 and 1.5 together (`kg/merge-env-adapter` not created yet). 1.5 stage 2's Codex run is active. |
| 2 Model | 5 / 5 | merged | None. CUDA/BF16 is unqualified and waits on Phase 6. |
| 3 Trainer | 4 / 6 | blocked on 1.4/1.5 | 3.1's Kaggriculture rollout/mask mapping, then 3.5 |
| 4 Teacher | 3 / 4 | 4.1–4.3 merged | 4.4 configs (implemented on `kg/rebuild-4-4`, in verification) |
| 5 BC | 0 / 2 | in progress | 5.2 trainer approved and 5.1 preparer approved at `356d19f`, both unmerged; two later preparer commits unreviewed |
| 6 GPU verification | 1 / 6 | 6.0 merged; evidence merged | 6.1 after 1.4/1.5, 3.5 and 5.2 |
| 7 Evaluation and packaging | 0 / 5 | 7.1 approved; 7.3 in review; 7.4 brief awaiting re-review | 7.1 merge; 7.3 verification (Codex run active) |
| 8 Docs and closeout | 0 / 3 | 8.1a partial (custody) | Needs the earlier phases |

Live processes at 22:11 HKT (`ps`):

- A Codex `codex exec` in `/Users/poonszesen/kg-v3-t73` (Task 7.3), started 22:07.
- The Task 1.5 stage-2 run, whose shell (PID 52486) started 21:51. Its transcript `codex/task-1.5-s2-transcript.log` (local) was still growing at 22:11.

## Phase 0 — Contract and diagnostics

| Task | State | Branch / commit | Codex verdict | Receipts | Cookbook | Gaps |
|---|---|---|---|---|---|---|
| 0.1 Contract | merged | Committed directly on integration as `08b0092`; v4.1 clarification `cdd2617`. `docs/kaggriculture-contract.md:3` reads "Status: v4.1, accepted". | `codex/task-0.1-review.md` (no VERDICT line, "Draft v1 needs revision"); `codex/task-0.1-rereview.md` ACCEPT WITH EDITS. No report reviews v4. | `docs/kaggriculture-contract.md` "Review and changes" | No dedicated note. `08b0092` edited `cookbook/decisions/start-multi-gpu-qualification-with-two-ranks.md` and `cookbook/decisions/the-policy-is-stateless-and-observation-only.md`. | No Codex re-review of Claude's v4 edits. |
| 0.2 CUDA crash repro | merged | Committed directly on integration as `ba9ab8d` | No 0.2 report. The follow-up GEMM-limit reviews are under Phase 6. | `rebuild/results.md` "Task 0.2"; `rebuild/run-statements/cuda-repro-blocking.md` | `cookbook/references/compiled-gemm-template-overflows-above-2-21-rows.md` | The plan's checkbox now describes the blocking run plus `int32_probe.py`. Staying on torch 2.9. Production compliance is unproven. |
| 0.3 Failed-run reporting | merged | `kg/rebuild-codex` `47f4d12`, merged by `ab0de01` | Codex implemented it (`codex/task-0.3-report.md`, no VERDICT). Claude's review is in `rebuild/0.3-results.md`. | `rebuild/0.3-results.md`, `rebuild/0.3-*.log`, `rebuild/briefs/0.3.md` | `cookbook/references/failed-training-reports-status-before-distributed-cleanup.md` | Offline unit tests only; no live W&B or distributed failure run. |

## Phase 1 — Engine and native environment

| Task | State | Branch / commit | Codex verdict | Receipts | Cookbook | Gaps |
|---|---|---|---|---|---|---|
| 1.1 Rules kernel | merged | `kg/rebuild-codex` `5378763`, merged by `90ed86c` | `codex/verify-1.1-r1.md` APPROVE WITH EDITS; `codex/verify-1.1-r2.md` APPROVE at `5378763` | `rebuild/1.1/results.md`, `rebuild/briefs/1.1-rules-kernel.md` | `cookbook/decisions/restart-the-port-from-isaiahs-clean-base.md` ("Task 1.1") | Trace `rng_schedule`/`shop_schedule` headers are not compared. Covers four pinned worlds. |
| 1.1b Live parity (owner request, no plan checkbox) | merged | `kg/rebuild-parity` `16e56b6`, merged by `8209193`; fix-ups `7529e01`, `99732ea` | `codex/verify-1.1b-r3.md` APPROVE at `16e56b6`; merge `codex/verify-merge-1.1b-r3.md` APPROVE | `rebuild/1.1b/results.md`, `rebuild/merge-1.1b/` | `cookbook/references/live-differential-parity-checks-the-rust-kernel.md` | D1 (Unicode digits) and D2 (unhashable items) are kept as expected failures. The pod-scale sweep is still open. `verify-merge-1.1b-r2-transcript.log` is excluded from custody (encrypted blobs). |
| 1.2 Grammar kernel | merged | `kg/rebuild-grammar` `7877c46`, merged by `84dd76a`; fix-ups `195bf7a`, `f464c3d` | `codex/verify-1.2-r1.md` APPROVE; merge `codex/verify-merge-1.2-r3.md` APPROVE at `f464c3d` | `rebuild/1.2/results.md`, `rebuild/merge-1.2/results.md` | `cookbook/references/native-game-semantics-use-v3-owned-buffers.md` | The PyO3 grammar-table binding belongs to 1.4/1.5. No GPU validation. |
| 1.3 Observation encoder | merged | `kg/rebuild-observe` `dc6b200`, merged by `e197528`; custody fix `b8747b6` | `codex/verify-1.3-r3.md` APPROVE at `dc6b200`; merge `codex/verify-merge-1-3-r2.md` APPROVE (`b8747b6`) | `rebuild/1.3/`, `rebuild/verify-merge-1.3/`, `rebuild/verify-e197528-independent/`, `rebuild/verify-b8747b6-independent/` (all committed by the custody sweep) | `cookbook/references/structured-observations-preserve-legal-state-and-order.md` (status `oracle-qualified-timing-pending`) | Optimized timing is handed to the pod. |
| 1.4 Native env | approved, not merged | `kg/rebuild-env` `b6b722f` (base `e197528`). `b6b722f` is merged into `kg/rebuild-adapter` (`558ac3c`). `kg/rebuild-bc-now` (`66a4aeb`) and `kg/rebuild-7-3` (`37c1e51`) merged the earlier `1e63597`, so they lack `b6b722f`, the fixture-pair publication rollback fix. | `codex/verify-1.4-r1.md`, `-r2.md`, `-r3.md` APPROVE WITH EDITS (local); `codex/confirm-1.4-p3.md` APPROVE at `b6b722f` (local) | `kg/rebuild-env:ops/rebuild-2026-09-29/1.4/`; `rebuild/briefs/1.4.md` | `kg/rebuild-env` edits `native-game-semantics-use-v3-owned-buffers.md` and `evaluation-and-truncation-follow-the-kaggriculture-objective.md` | Lands together with 1.5. The ABI stub must end byte-identical across 1.4 and 1.5. |
| 1.5 Python adapter | in progress | `kg/rebuild-adapter` `558ac3c`: stage 1 `43354c8`, Claude's review `6bc5290`, then 1.4 merged in. The worktree `/Users/poonszesen/kg-v3-adapter` has 19 uncommitted stage-2 paths. | `codex/task-1.5-s1-report.md` STAGE1-COMPLETE (local; the implementer's status, not a review). Stage 2 has no report yet. | `kg/rebuild-adapter:ops/rebuild-2026-09-29/stage1-adapter/`; `/Users/poonszesen/kg-v3-adapter/ops/rebuild-2026-09-29/stage2-adapter/` (uncommitted); `rebuild/briefs/1.5.md` | `kg/rebuild-adapter` also edits `reward-reuse-preserves-objective-and-critic-semantics.md` | Stage 2 is running. No Codex verify yet. |

## Phase 2 — Model

| Task | State | Branch / commit | Codex verdict | Receipts | Cookbook | Gaps |
|---|---|---|---|---|---|---|
| 2.1 Encoder | merged | Lane B `94f778d`, merged by `31e1e6c` | `codex/verify2-lane-B-r2.md` APPROVE at `94f778d`; `codex/verify-merge-heads-r2.md` APPROVE at `e1458d2` | `rebuild/briefs/2.1-encoder.md` | `cookbook/references/kaggriculture-encoder-reuses-isaiah-stateless-layers.md`, including the architecture image pinned to `8093d51` (custody) | CPU only. Semantic zero-fill and privacy are asserted only on a synthetic fixture. The image predates the 2.3 merge. |
| 2.2 Critic | merged | Lane B `94f778d`, merged by `31e1e6c`; masked softmax `4cac1a1`, merged by `a19a5ff` | `codex/verify2-lane-B-r2.md` APPROVE | `rebuild/briefs/2.2-critic.md` | Encoder note Limits; `cookbook/references/kaggriculture-model-joins-isaiahs-factory-compile-and-masked-critic.md` | The GPU value gap is attributed in Phase 6 (value-gap row). |
| 2.3 Grammar heads | merged | `kg/rebuild-heads` `88f95f6`, merged by `1210b54`; reconciled in `e1458d2` | `codex/verify-2.3-r2.md` APPROVE; `codex/verify-merge-heads-r2.md` APPROVE at `e1458d2` | `rebuild/briefs/2.3-action-heads.md`, `rebuild/codex/verify-2.3-heads/` | `cookbook/references/kaggriculture-grammar-heads-sit-behind-isaiahs-actor-projection.md` | Synthetic grammar tables until the 1.4/1.5 binding lands. |
| 2.4 Model size | merged | Owner answer recorded in `08b0092` | None (owner decision) | `cookbook/decisions/the-policy-is-stateless-and-observation-only.md` "Model size" | Same Decision | — |
| 2.5 Model docs | merged | `docs/kaggriculture-model.md` (`f161fc3`; count in `a57111f`/`4f2bcd7`) | Covered by the lane-B and 2.3 verifies | `docs/kaggriculture-model.md` | — | — |

## Phase 3 — Trainer

| Task | State | Branch / commit | Codex verdict | Receipts | Cookbook | Gaps |
|---|---|---|---|---|---|---|
| 3.1 Schema-generic seams | merged | Stream C `62899de`, merged by `07568e0` | `codex/verify2-lane-C-r2.md` APPROVE | `rebuild/3.1-*.log`, `rebuild/stream-c-*.log` | `cookbook/references/ppo-trainer-seams-map-any-schema-and-alarm-on-replay-drift.md` | Proven on Orbit types only. |
| 3.1 Model side | merged | `kg/rebuild-trainer-model` `4cac1a1`, merged by `a19a5ff` | `codex/verify-3.1-rest-r3.md` APPROVE; `codex/verify-merge-trainer-lanes-r2.md` APPROVE at `f4ecd90` | `rebuild/trainer-model/`, `rebuild/merge-trainer-lanes/` | `cookbook/references/kaggriculture-model-joins-isaiahs-factory-compile-and-masked-critic.md` | No real `torch.compile`/CUDA run through `run_ppo`. |
| 3.1 Kaggriculture rollout and mask mapping | blocked | No commit found on any branch (commit-message search since 18:00) | None | `rebuild/merge-teacher/skip-probe.log` (TypeError: no `KaggricultureActionMask` mapping) | Gap stated in the model-joins, configs and teacher notes | Waits on 1.5. The plan's 3.1 checkbox stays open. |
| 3.2 Game semantics | merged | `kg/rebuild-semantics` `530b8cc`, merged by `d1f8d59` | `codex/verify-3.2-3.3-r2.md` APPROVE | `rebuild/3.2-3.3-*.log` | `cookbook/references/evaluation-and-truncation-follow-the-kaggriculture-objective.md` | Kaggriculture `_apply_truncation` is untested until the mask mapping lands. |
| 3.3 Evaluation | merged | Same as 3.2 | Same as 3.2 | Same as 3.2 | Same as 3.2 | `_create_eval_env` rejects Kaggriculture on integration. |
| 3.4 Configs | merged | `kg/rebuild-configs` `71cebdb`, merged by `e7ce331`; 8-rank config merged by `a3a5cb8` | `codex/verify-3.4-r3.md` APPROVE WITH EDITS; `codex/confirm-3.4.md` APPROVE | `rebuild/configs-r2/` | `cookbook/references/kaggriculture-configs-follow-isaiahs-scaling-6m-recipe.md`; `cookbook/decisions/kaggriculture-compiles-gemms-with-cublas-only.md` | Launch stops after the workload check. The spm/accum split needs the 6.1 smoke. |
| 3.5 Local functional check | blocked | None | None | None | None | Needs 1.4/1.5 and the 3.1 Kaggriculture seam. |
| 3.6 Replay-drift alarm | merged | Stream C `62899de`, merged by `07568e0` | `codex/verify2-lane-C-r2.md` APPROVE | `rebuild/3.6-*.log` | `cookbook/references/ppo-trainer-seams-map-any-schema-and-alarm-on-replay-drift.md` | The 0.05-nat threshold is unqualified on GPU/BF16. |

## Phase 4 — Teacher

| Task | State | Branch / commit | Codex verdict | Receipts | Cookbook | Gaps |
|---|---|---|---|---|---|---|
| 4.1 Per-slot KL | merged | `e584a0d` (phase tip `8fde43c`), merged by `a424d8c`; follow-up `2390c8e` | Brief: `codex/brief-4-review.md` REVISE (v1; no v2 re-review found). Code: `codex/verify-4-teacher-r2.md` APPROVE (4.1–4.3). Merge: `codex/verify-merge-teacher-r1.md` APPROVE WITH EDITS, `codex/verify-merge-teacher-r2.md` APPROVE. | `rebuild/briefs/4-teacher.md`, `rebuild/trainer-model/4.1-*.log` | `cookbook/references/kaggriculture-teacher-distills-per-slot-kl-and-per-seat-winner-ce.md` | GPU component evidence exists (GPU bundle C3/C4); cross-chunk/minibatch equality, multi-rank and trainer integration are open. |
| 4.2 Targets and cache | merged | `a90b820`; as 4.1 | As 4.1 | `rebuild/trainer-model/4.2-*.log` | As 4.1 | Cache bytes are computed, not measured in a training run. |
| 4.3 Model methods and wiring | merged | `b94b8dc` plus r1 fixes `8fde43c`; as 4.1 | `codex/verify-4-teacher-r1.md` APPROVE WITH EDITS; `codex/verify-4-teacher-r2.md` APPROVE | `rebuild/trainer-model/4.3-*.log`, `rebuild/merge-teacher/skip-probe.log` | As 4.1 | T18 and the launch/resume tests stay skipped until the 3.1 seam and the native env are wired. |
| 4.4 Teacher configs | implemented, in verification | `kg/rebuild-4-4` | Listed as outstanding in `codex/verify-4-teacher-r2.md`; verify loop `codex/verify-4.4-rN.md` | None | None | Teacher fields pinned, cache bytes at startup, teacher checkpoint required at a fresh Kaggriculture launch; also sends Kaggriculture PPO runs to W&B `kg-v3`. |

## Phase 5 — BC

| Task | State | Branch / commit | Codex verdict | Receipts | Cookbook | Gaps |
|---|---|---|---|---|---|---|
| 5.1 Selection (Stream D) | merged | `kg/rebuild-codex-data` `9d54a8b`, merged by `1177a5b` | `codex/verify2-lane-D-r1.md` APPROVE | `rebuild/briefs/5.1-bc-data.md`, `rebuild/checks/` | `cookbook/references/rebuild-data-preparation-preserves-replay-identity.md` | — |
| 5.1 Brief (owner directions) | approved, not merged | `kg/rebuild-bc-brief` `74428e8` | `codex/brief-5.1-rereview.md` APPROVE WITH EDITS; `codex/brief-5.1-owner-edits.md` REVISE, `-r2`/`-r3` APPROVE WITH EDITS, `-r4` APPROVE at `74428e8` (all local) | `kg/rebuild-bc-brief:ops/rebuild-2026-09-29/briefs/5.1-bc-data.md` | `kg/rebuild-bc-brief` edits `rebuild-data-preparation-preserves-replay-identity.md` | — |
| 5.1 Shard preparer | in progress | `kg/rebuild-bc-now` `89ca39c`: preparer `49255ac`, resume binding `356d19f`, then `42a8b39` (`--team`, `--include-losses`) and `89ca39c` (team-name leak, draws). It merges 1.4 (`66a4aeb`) and 5.2 (`8d29ae3`). | `codex/verify-5.1-prepare.md` REQUEST CHANGES; `codex/verify-5.1-prepare-r2.md` APPROVE at `356d19f` (local). No review of `42a8b39` or `89ca39c` was found. | On `kg/rebuild-bc-now` | None beyond the 5.2 note it carries | The two later commits are unreviewed. It depends on the unmerged 1.4. |
| 5.2 BC trainer | approved, not merged | `kg/rebuild-bc-trainer` `b626f24` (base `2390c8e`) | `codex/verify-5.2-trainer-r1.md` REJECT; `-r2` APPROVE WITH EDITS; `-r3` APPROVE (transcript reviews `b626f24`) (all local); independent `/Users/poonszesen/kg-v3-bc/ops/rebuild-2026-09-29/codex/verify-5.2-r3-independent/report.md` APPROVE (uncommitted) | On `kg/rebuild-bc-trainer` | `kg/rebuild-bc-trainer:cookbook/references/kaggriculture-bc-trainer-warm-starts-ppo-from-the-held-out-best.md` | No BC training run is recorded here. It needs 1.5 for native data. |

## Phase 6 — GPU verification and pod evidence

| Task | State | Branch / commit | Codex verdict | Receipts | Cookbook | Gaps |
|---|---|---|---|---|---|---|
| 6.0 flash-attn on the pod | merged | `kg/rebuild-model` `4fd40c7`, `78df33c`, `bcd9627`, merged by `4974888` | `codex/verify-flash-attn-r2.md` APPROVE (`78df33c`); merge `codex/verify-merge-evidence-r1.md` APPROVE WITH EDITS, `-r2.md` APPROVE | `rebuild/flash-attn-setup-2026-09-29/`; `rebuild/results.md` "Phase 6.0" | `cookbook/references/pod-v3-environment-runs-flash-attn-2-8-3-forward-on-sm120.md` | Forward path only. Trunk numerics are not qualified. Two transcripts are excluded from custody (unredacted pod host). |
| Evidence: GEMM limits | merged | `kg/rebuild-model` up to `432abe0`, merged by `053840f` | `codex/verify-gemm-limits-r3.md` APPROVE (`432abe0`) | `rebuild/gemm-limits-2026-09-29/` | `cookbook/references/compiled-gemm-template-overflows-above-2-21-rows.md` | Triton version and torch git identity were not captured at probe time. |
| Evidence: model-only SPS ceiling | merged | `ed61770` … `5247daa`, merged by `4974888`; follow-up `0b8cf98` | `codex/verify-sps-ceiling-r2.md` APPROVE (`ddf1fb2`) | `rebuild/model-sps-ceiling-2026-09-29/` | `cookbook/references/model-only-sps-ceiling-bounds-per-rank-throughput.md` | Component estimates only. End-to-end SPS is unmeasured. |
| Evidence: ATEN-only GEMM A/B | merged | `9904121`, `8fcfada`, `9bdd82d`, merged by `4974888` | `codex/verify-aten-ab-r2.md` APPROVE (`9bdd82d`) | `rebuild/aten-gemm-ab-2026-09-29/` | `cookbook/decisions/kaggriculture-compiles-gemms-with-cublas-only.md` | Component timing only. |
| Evidence: GPU checks bundle | merged | `kg/rebuild-gpu-checks` `24380f3`, merged by `d793d16`; merge fixes `3f26e49`, `666deec` | `codex/verify-gpu-bundle-r1..r3` (r3 APPROVE at `24380f3`; committed by custody); merge `codex/verify-merge-gpu-receipts-r1.md`, `-r2.md` REJECT; `-r3.md` APPROVE at `666deec` (local) | `rebuild/gpu-checks-2026-09-29/`; `rebuild/results.md` "GPU checks bundle" | No dedicated note; the merge fixes credit its evidence in the compiled-GEMM and teacher References and the multi-GPU Decision | Channel 229 is unattributed. The r3 merge-verify evidence directory is uncommitted in `/Users/poonszesen/kg-v3-m-gpu-receipts`. |
| Evidence: value-gap diagnostic | merged (custody) | `kg/rebuild-value-gap` `bd31cb6`, merged by `2f535ea` | `codex/verify-value-gap-r1.md` APPROVE WITH EDITS on `bd31cb6` (committed by custody) | `rebuild/value-gap-2026-09-29/`; `rebuild/run-statements/value-gap-diagnostic.md`; `rebuild/results.md` "Value gap diagnostic" | None (promotion left to the owner's workflow) | r1 edits applied on `kg/merge-custody` after `verify-merge-custody-r1` confirmed them (three P2: actor-gain attribution, "bit-identical" scope, rerun observation custody; two P3: large-file manifest paths, two numerical summaries). Gain-only causality and rerun input identity stay unverified. H1 is supported for fresh weights only. |
| 6.1 Memory smoke, 2 ranks | not started | None | None | None | Target in `cookbook/decisions/start-multi-gpu-qualification-with-two-ranks.md` | Blocked on 1.4/1.5, 3.5 and 5.2. |
| 6.2 Complete-work run, 2 ranks | not started | None | None | None | None | Depends on 6.1 and 5.2. |
| 6.3 Four ranks | not started | `configs/kaggriculture_4rank.yaml` exists; no run | None | None | None | Depends on 6.2. |
| 6.3b 8-rank qualification | not started (config and plan merged) | `kg/rebuild-8rank` `7ad45fc`, merged by `a3a5cb8`; follow-up `ca37089` | `codex/verify-8rank-r2.md` APPROVE; merge `codex/verify-merge-8rank-r1.md` APPROVE WITH EDITS, `-r2.md` APPROVE | `rebuild/8rank/`, `rebuild/codex/verify-merge-8rank-r2/` | `cookbook/references/kaggriculture-configs-follow-isaiahs-scaling-6m-recipe.md`; `cookbook/decisions/start-multi-gpu-qualification-with-two-ranks.md` | No 8-GPU run. Creating the pod needs a stated live price and owner approval. |
| 6.4 Orbit end-to-end | not started | None | None | None | None | No run statement. |

## Phase 7 — Evaluation and packaging

| Task | State | Branch / commit | Codex verdict | Receipts | Cookbook | Gaps |
|---|---|---|---|---|---|---|
| 7.1 Opponents | approved, not merged | `kg/rebuild-7-1` `908c73f` (base `b8747b6`): Codex run 2 `7ae9bbf`, Claude review `f15a413`, verify r1 fixes `908c73f` | `codex/verify-7.1-r1.md` REJECT; `codex/verify-7.1-r2.md` APPROVE at `908c73f` (both local) | `kg/rebuild-7-1:ops/rebuild-2026-09-29/7.1/` | `kg/rebuild-7-1:cookbook/references/snapshot-view-isolates-byte-exact-evaluation-opponents.md` | Not rebased on the current integration. |
| 7.2 Panel script | not started | None | None | None | Contract: `cookbook/decisions/evaluation-preserves-generality-and-evidence.md` | Depends on 7.1 and 6.2. |
| 7.3 Replay export | in review | `kg/rebuild-7-3` `f23cd4f` (base `0b8cf98`, merges 1.4 at `37c1e51`) | `codex/verify-7.3-r1.md` REJECT; `codex/verify-7.3-r2.md` REJECT on `822c951` (local); `f23cd4f` addresses r2; a Codex run started 22:07 | `kg/rebuild-7-3:ops/rebuild-2026-09-29/7.3/` | `kg/rebuild-7-3:cookbook/references/native-replay-export-preserves-kaggle-episodes.md` | No approving verdict yet. Depends on the unmerged 1.4. |
| 7.4 Packaging | in review (brief) | `kg/rebuild-7-4-brief` `ff6195e`; review applied in `2ac7ddb` | `codex/brief-7.4-review.md` REVISE on `eccdd12` (local); no re-review found | `kg/rebuild-7-4-brief:ops/rebuild-2026-09-29/briefs/7.4-packaging.md` | `kg/rebuild-7-4-brief:cookbook/references/kaggle-packaging-reuses-the-starter-submission-path.md` | No implementation. The owner decides any submission. |
| 7.5 Parity docs | not started | Kaggriculture sections already in `docs/rules-parity-coverage.md` | None | None | `cookbook/references/live-differential-parity-checks-the-rust-kernel.md` | Needs a closing pass after 7.1, 7.3 and 7.4. |

## Phase 8 — Docs and closeout

| Task | State | Branch / commit | Codex verdict | Receipts | Cookbook | Gaps |
|---|---|---|---|---|---|---|
| 8.1a Phase-based record (owner direction) | in progress | This tracker and the phase-grouped `cookbook/references/index.md` (`kg/rebuild-phase-map`, merged by `2ebf538`, custody) | `codex/verify-phase-map-r2.md` APPROVE at `adcb4e7` | This file | `cookbook/references/index.md` | Finalize and re-verify at closeout. |
| 8.1 References and rl-api-specs | not started | None | None | None | See "Notes not yet on integration" | The rl-api-specs differences are unchecked. |
| 8.2 Closeout | not started | None | None | None | None | Needs every earlier phase. |

## Cross-cutting

| Item | State | Branch / commit | Codex verdict | Receipts | Cookbook | Gaps |
|---|---|---|---|---|---|---|
| kaggle-environments 1.32.7 pin | merged | `kg/rebuild-deps` `c452f66`, merged by `69397da` | `codex/verify-deps-r3.md` APPROVE | `ops/deps-independent-verification-fb65e1f/` | `cookbook/decisions/restart-the-port-from-isaiahs-clean-base.md` | — |
| cuBLAS-only compiled GEMMs | merged | `b2ef2bd`, `3b735c0`, `b51b0c0` | `codex/verify-aten-r3.md` APPROVE (`b51b0c0`) | `rebuild/aten-wiring/` | `cookbook/decisions/kaggriculture-compiles-gemms-with-cublas-only.md` | CPU-tested enforcement. |
| Cookbook refresh and codex exec workflow | merged | `kg/rebuild-model` `bcd9627` … `8093d51`, merged by `4974888` | `codex/verify-cookbook-refresh-r3.md` APPROVE (`8093d51`) | `cookbook/log.md` | `cookbook/workflows/run-codex-exec-with-closed-stdin-and-wait-for-its-verdict.md` | — |
| Architecture image | merged (custody) | Main-checkout edits on `8093d51`, applied by `1df17dd` | None | `ops/v3-architecture-image-2026-09-29/` | Encoder Reference "Architecture illustration" | Pinned to `8093d51`; regenerate before reuse as current status. |
| Evidence custody | merged (custody) | `c8aaaac`, `470b527`, `7911b16` | None | `rebuild/evidence-custody.md`, `rebuild/evidence-custody.json` | Log entry only | MANIFEST and DEFER files remain local. Three transcripts are excluded (secret triage). Anything written after 13:48Z is unclassified. |

## Notes not yet on integration

These notes exist only on unmerged branches, found with `git diff --diff-filter=A HEAD...<branch> -- cookbook/` at 22:12 HKT. They are not linked from `cookbook/references/index.md`. Link each one when its branch merges.

| Note | Branches |
|---|---|
| `cookbook/references/kaggriculture-bc-trainer-warm-starts-ppo-from-the-held-out-best.md` | `kg/rebuild-bc-trainer`, `kg/rebuild-bc-now` |
| `cookbook/references/snapshot-view-isolates-byte-exact-evaluation-opponents.md` | `kg/rebuild-7-1` |
| `cookbook/references/native-replay-export-preserves-kaggle-episodes.md` | `kg/rebuild-7-3` |
| `cookbook/references/kaggle-packaging-reuses-the-starter-submission-path.md` | `kg/rebuild-7-4-brief` |

Several unmerged branches also edit integration notes. `kg/rebuild-env`, `kg/rebuild-adapter`, `kg/rebuild-bc-now` and `kg/rebuild-7-3` edit `native-game-semantics-use-v3-owned-buffers.md` and `evaluation-and-truncation-follow-the-kaggriculture-objective.md`. `kg/rebuild-adapter` also edits `reward-reuse-preserves-objective-and-critic-semantics.md`. `kg/rebuild-bc-brief` and `kg/rebuild-7-3` edit `rebuild-data-preparation-preserves-replay-identity.md`. Their merges must reconcile those notes and keep the index's phase grouping.
