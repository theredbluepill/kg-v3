# Rebuild phase status

**As of 2026-09-29 19:00 HKT (11:00Z).** Integration branch `kg/isaiah-gap-closure` at `b8747b6`.

This is a working artifact, not a durable cookbook claim. Update it at each landing: a merge, a Codex verdict, or a state change. When a row changes, re-check it against git and the cited report. Don't copy a row into a cookbook note without re-checking it. The cookbook stays organised by concept (`cookbook/references/`, `cookbook/decisions/`). This file is the one place that maps those notes and receipts onto the plan's phases (`ops/rebuild-2026-09-29/plan.md`).

## Path conventions

- **Bare paths** are tracked on integration at `b8747b6`, relative to the repository root. `rebuild/` is short for `ops/rebuild-2026-09-29/`.
- **`codex/…`** is `/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/codex/…`. That is the main worktree (`kg/rebuild-model`) copy. Six cited reports are also tracked on integration at `b8747b6`, byte-identical to that copy: `task-0.1-review.md`, `task-0.1-rereview.md`, `task-0.3-report.md` and `verify-3.4-r1.md`/`-r2.md`/`-r3.md`. `kg/merge-teacher` additionally tracks `brief-4-review.md` and `verify-merge-teacher-r1.md`. Every other cited `codex/…` report is untracked on every branch.
- **`<branch>:<path>`** is a path tracked on that branch, not on integration.
- **Absolute paths** are uncommitted files in the named worktree.
- **"Merged"** means the commit is an ancestor of `b8747b6`. Each was checked with `git merge-base --is-ancestor <sha> b8747b6` at 19:00 HKT.

State values: **merged**, **approved, not merged** (Codex APPROVE, not on integration), **in review**, **in progress**, **blocked**, **not started**.

## Summary

The done count only includes plan tasks whose work is merged into integration. Tasks without a plan checkbox (1.1b, 6.3b and the pod evidence runs) are listed in their phase's table but are not counted.

| Phase | Done / total | State | Next landing |
|---|---|---|---|
| 0 Contract and diagnostics | 3 / 3 | merged | None. Contract v4 has no Codex re-review (see 0.1). |
| 1 Engine and native env | 3 / 5 (plus 1.1b, merged) | in progress | 1.4 native env (Codex running), then 1.5 stage 2 |
| 2 Model | 5 / 5 | merged | None. CUDA/BF16 is unqualified and waits on Phase 6. |
| 3 Trainer | 4 / 6 | blocked on 1.4/1.5 | 3.1's Kaggriculture rollout/mask mapping, then 3.5 |
| 4 Teacher | 0 / 4 | approved, not merged (4.1–4.3) | Teacher merge `kg/merge-teacher` `2390c8e` (merge verify r2 running), then 4.4 |
| 5 BC | 0 / 2 | blocked | 5.1 `prepare.py` waits on the 1.4/1.5 explicit-state constructor |
| 6 GPU verification | 0 / 5 | 6.0 approved, not merged; evidence runs partly merged | Merge the `kg/rebuild-model` evidence; then 6.1 after 1.4/1.5/3.5/5.2 |
| 7 Evaluation and packaging | 0 / 5 | 7.1 blocked, 7.4 brief in review | 7.4 brief revision; 7.1 needs an engine-API decision |
| 8 Docs and closeout | 0 / 2 | not started | Needs the earlier phases |

Live processes at 19:00 HKT (`ps`): Codex `codex exec resume` for Task 1.4 (PID 55836, started 18:36) and the Codex teacher merge verification r2 in `/Users/poonszesen/kg-v3-m-teacher` (PID 89983, started 18:57). The Task 7.1 Codex run (PID 67210) has exited, and its report is `codex/task-7.1-impl-report.md`.

## Phase 0 — Contract and diagnostics

| Task | State | Branch / commit | Codex verdict | Receipts | Cookbook | Gaps |
|---|---|---|---|---|---|---|
| 0.1 Contract | merged | Committed directly on integration as `08b0092`; v4.1 clarification `cdd2617`. `docs/kaggriculture-contract.md:3` reads "Status: v4.1, accepted". | `codex/task-0.1-review.md` (no VERDICT line, "Draft v1 needs revision"); `codex/task-0.1-rereview.md` ACCEPT WITH EDITS. No report reviews v4. | `docs/kaggriculture-contract.md` "Review and changes" (lines 271–296) | No dedicated note. `08b0092` edited `cookbook/decisions/start-multi-gpu-qualification-with-two-ranks.md` and `cookbook/decisions/the-policy-is-stateless-and-observation-only.md`. | No Codex re-review of Claude's v4 edits. |
| 0.2 CUDA crash repro | merged | Committed directly on integration as `ba9ab8d` | No 0.2 report. The follow-up GEMM-limit reviews are listed under Phase 6. | `rebuild/results.md` "Task 0.2" (lines 5–32); `rebuild/run-statements/cuda-repro-blocking.md` | `cookbook/references/compiled-gemm-template-overflows-above-2-21-rows.md` | The plan names `cuda_repro.py`, which is absent; the receipt cites `int32_probe.py`. Staying on torch 2.9 (`results.md:28-32`). Production compliance is unproven. |
| 0.3 Failed-run reporting | merged | `kg/rebuild-codex` `47f4d12`, merged by `ab0de01` | Codex implemented it (`codex/task-0.3-report.md`, no VERDICT). Claude's review is in `rebuild/0.3-results.md:107-111`. | `rebuild/0.3-results.md`, `rebuild/0.3-*.log`, `rebuild/0.3-sha256.txt`, `rebuild/briefs/0.3.md` | `cookbook/references/failed-training-reports-status-before-distributed-cleanup.md` | Offline unit tests only; no live W&B or distributed failure run. |

## Phase 1 — Engine and native environment

| Task | State | Branch / commit | Codex verdict | Receipts | Cookbook | Gaps |
|---|---|---|---|---|---|---|
| 1.1 Rules kernel | merged | `kg/rebuild-codex` `5378763`, merged by `90ed86c` | `codex/verify-1.1-r1.md` APPROVE WITH EDITS; `codex/verify-1.1-r2.md` APPROVE at `5378763` | `rebuild/1.1/results.md`, `rebuild/briefs/1.1-rules-kernel.md`, `rebuild/briefs/1.1-review-claude.md` | `cookbook/decisions/restart-the-port-from-isaiahs-clean-base.md` (section "Task 1.1") | Trace `rng_schedule`/`shop_schedule` headers are not compared (`1.1/results.md:201`). Covers four pinned worlds. L4 was repaired later, in 1.3. |
| 1.1b Live parity (owner request, no plan checkbox) | merged | `kg/rebuild-parity` `16e56b6`, merged by `8209193`; fix-ups `7529e01`, `99732ea` | `codex/verify-1.1b-r1.md`, `codex/verify-1.1b-r2.md` APPROVE WITH EDITS; `codex/verify-1.1b-r3.md` APPROVE at `16e56b6`. Merge verify: `codex/verify-merge-1.1b-r3.md` APPROVE. | `rebuild/1.1b/results.md`, `rebuild/merge-1.1b/` | `cookbook/references/live-differential-parity-checks-the-rust-kernel.md` | Divergences D1 (Unicode digits) and D2 (unhashable items) are kept as expected failures. The pod-scale sweep is still open. |
| 1.2 Grammar kernel | merged | `kg/rebuild-grammar` `7877c46`, merged by `84dd76a`; fix-ups `195bf7a`, `f464c3d` | `codex/verify-1.2-r1.md` APPROVE. Merge verify: `codex/verify-merge-1.2-r3.md` APPROVE at `f464c3d`. | `rebuild/1.2/results.md`, `rebuild/merge-1.2/results.md`, `rebuild/briefs/1.2.md` | `cookbook/references/native-game-semantics-use-v3-owned-buffers.md` | The PyO3 grammar-table binding and the skipped native-table equality test belong to 1.4. No GPU validation. |
| 1.3 Observation encoder | merged | `kg/rebuild-observe` `dc6b200`, merged by `e197528`; custody fix `b8747b6` | `codex/verify-1.3-r1.md` REJECT; `codex/verify-1.3-r3.md` APPROVE at `dc6b200`. Merge: `codex/verify-merge-1-3-r1.md` REJECT; `codex/verify-merge-1-3-r2.md` APPROVE (`b8747b6`). | `rebuild/1.3/results.md` (its header "incomplete" is stale), `rebuild/merge-1.3/`, `rebuild/briefs/1.3.md` | `cookbook/references/structured-observations-preserve-legal-state-and-order.md` (status `oracle-qualified-timing-pending`) | Optimized timing hit the Mac memory stop and is handed to the pod. |
| 1.4 Native env | in progress | `kg/rebuild-env` `8d98ea8` ("partial"), worktree `/Users/poonszesen/kg-v3-env` with uncommitted edits. Brief merged by `60d5cdd`. | Brief: `codex/confirm-briefs-1.4-1.5-r2.md` APPROVE at `bb0f7e4`. Implementation: no verdict yet (Codex PID 55836 running). | `/Users/poonszesen/kg-v3-env/ops/rebuild-2026-09-29/1.4/progress.md` (uncommitted); `rebuild/briefs/1.4.md` | None yet | Release overflow proof is "PENDING (pod)". The ≥16-game oracle is unconfirmed. The `rs.pyi` stub must end byte-identical across 1.4 and 1.5. |
| 1.5 Python adapter | in progress | `kg/rebuild-adapter` `43354c8` (stage 1), worktree `/Users/poonszesen/kg-v3-adapter` with Claude's uncommitted review | `codex/task-1.5-s1-report.md` STAGE1-COMPLETE (the implementer's status, not a review). No Codex verify. | `kg/rebuild-adapter:ops/rebuild-2026-09-29/stage1-adapter/results.md`; `rebuild/briefs/1.5.md` | `kg/rebuild-adapter` edits `native-game-semantics-use-v3-owned-buffers.md` and `reward-reuse-preserves-objective-and-critic-semantics.md` | Stage 2 is blocked on the 1.4 binding. Native reward, codec replay and DMA-fence tests are skipped. |

## Phase 2 — Model

| Task | State | Branch / commit | Codex verdict | Receipts | Cookbook | Gaps |
|---|---|---|---|---|---|---|
| 2.1 Encoder | merged | Lane B `94f778d`, merged by `31e1e6c` | `codex/verify2-lane-B-r2.md` APPROVE at `94f778d`; `codex/verify-merge-heads-r2.md` APPROVE at `e1458d2` | `rebuild/briefs/2.1-encoder.md` | `cookbook/references/kaggriculture-encoder-reuses-isaiah-stateless-layers.md` | CPU only. The semantic zero-fill and privacy rules are asserted only on a synthetic fixture. |
| 2.2 Critic | merged | Lane B `94f778d`, merged by `31e1e6c`; masked softmax `4cac1a1`, merged by `a19a5ff` | `codex/verify2-lane-B-r2.md` APPROVE | `rebuild/briefs/2.2-critic.md`, `rebuild/trainer-model/critic-red.log` | Encoder note Limits; `cookbook/references/kaggriculture-model-joins-isaiahs-factory-compile-and-masked-critic.md` | No GPU run. Reopen the no-raise critic choice if a real row can have `still_playing=False`. |
| 2.3 Grammar heads | merged | `kg/rebuild-heads` `88f95f6`, merged by `1210b54`; reconciled in `e1458d2` | `codex/verify-2.3-r2.md` APPROVE; `codex/verify-merge-heads-r2.md` APPROVE at `e1458d2` | `rebuild/briefs/2.3-action-heads.md`, `rebuild/codex/verify-2.3-heads/` | `cookbook/references/kaggriculture-grammar-heads-sit-behind-isaiahs-actor-projection.md` | Synthetic grammar tables until the 1.4 binding. CUDA/BF16 is pending. |
| 2.4 Model size | merged | Owner answer recorded in `08b0092` | None (owner decision) | `cookbook/decisions/the-policy-is-stateless-and-observation-only.md` "Model size" (quote plus `user-directive:2026-09-29:model-size-6-10m`) | Same Decision | The plan's "Open question for the owner" (`plan.md:310`) is still listed, though answered. |
| 2.5 Model docs | merged | `docs/kaggriculture-model.md` added in `f161fc3`; parameter count in `a57111f`/`4f2bcd7` | No separate 2.5 review; covered by the lane-B and 2.3 verifies above | `docs/kaggriculture-model.md:30` (6,252,223 parameters) | — | — |

## Phase 3 — Trainer

| Task | State | Branch / commit | Codex verdict | Receipts | Cookbook | Gaps |
|---|---|---|---|---|---|---|
| 3.1 Schema-generic seams | merged | Stream C `62899de`, merged by `07568e0` | `codex/verify2-lane-C-r2.md` APPROVE | `rebuild/3.1-*.log`, `rebuild/stream-c-*.log` | `cookbook/references/ppo-trainer-seams-map-any-schema-and-alarm-on-replay-drift.md` | Proven on Orbit types only. |
| 3.1 Model side | merged | `kg/rebuild-trainer-model` `4cac1a1`, merged by `a19a5ff` | `codex/verify-3.1-rest-r3.md` APPROVE; `codex/verify-merge-trainer-lanes-r2.md` APPROVE at `f4ecd90` | `rebuild/trainer-model/`, `rebuild/codex/verify-3.1-rest/`, `rebuild/merge-trainer-lanes/` | `cookbook/references/kaggriculture-model-joins-isaiahs-factory-compile-and-masked-critic.md` | No real `torch.compile`/CUDA run. |
| 3.1 Kaggriculture rollout and mask mapping | blocked | None | None | `kg/merge-teacher:ops/rebuild-2026-09-29/merge-teacher/skip-probe.log` (TypeError: no `KaggricultureActionMask` mapping) | The gap is stated in the model-joins and configs notes | Waits on the 1.5 types and the 1.4 env. `run_ppo` keeps its explicit Kaggriculture stop. The plan's 3.1 checkbox stays open for this. |
| 3.2 Game semantics | merged | `kg/rebuild-semantics` `530b8cc`, merged by `d1f8d59` | `codex/verify-3.2-3.3-r2.md` APPROVE | `rebuild/3.2-3.3-*.log`, `rebuild/codex/verify-3.2-3.3-independent/` | `cookbook/references/evaluation-and-truncation-follow-the-kaggriculture-objective.md` | Kaggriculture `_apply_truncation` is untested (needs the mask mapping). |
| 3.3 Evaluation | merged | Same as 3.2 | `codex/verify-3.2-3.3-r2.md` APPROVE | Same as 3.2 | Same as 3.2 | `_create_eval_env` rejects Kaggriculture. The native seed test is skipped. |
| 3.4 Configs | merged | `kg/rebuild-configs` `71cebdb`, merged by `e7ce331` | `codex/verify-3.4-r1.md`, `codex/verify-3.4-r2.md` REJECT; `codex/verify-3.4-r3.md` APPROVE WITH EDITS; `codex/confirm-3.4.md` APPROVE | `rebuild/configs-r2/`, `rebuild/codex/verify-3.4*/` | `cookbook/references/kaggriculture-configs-follow-isaiahs-scaling-6m-recipe.md`; `cookbook/decisions/recipe-choices-align-to-isaiah-without-owner-escalation.md`; `cookbook/decisions/kaggriculture-compiles-gemms-with-cublas-only.md` | Launch stops after the workload check. The spm 8 / accum 1 split needs the 6.1 smoke. |
| 3.5 Local functional check | blocked | None (only the CPU configs from `e7ce331`) | None | None | None | Needs 1.4 and the 3.1 Kaggriculture seam. |
| 3.6 Replay-drift alarm | merged | Stream C `62899de`, merged by `07568e0` | `codex/verify2-lane-C-r2.md` APPROVE | `rebuild/3.6-*.log` | `cookbook/references/ppo-trainer-seams-map-any-schema-and-alarm-on-replay-drift.md` | The 0.05-nat threshold is unmeasured on GPU/BF16. Codex's probe gave 0.0723 nats from coherent noise. |

## Phase 4 — Teacher

| Task | State | Branch / commit | Codex verdict | Receipts | Cookbook | Gaps |
|---|---|---|---|---|---|---|
| 4.1 Per-slot KL | approved, not merged | `kg/rebuild-trainer-model` `e584a0d` (phase tip `8fde43c`). The merge is staged on `kg/merge-teacher` as `a424d8c`, with follow-up `2390c8e`. | Brief: `codex/brief-4-review.md` REVISE (v1); no v2 re-review found. Code: `codex/verify-4-teacher-r2.md` APPROVE (covers 4.1–4.3). Merge: `codex/verify-merge-teacher-r1.md` APPROVE WITH EDITS (one P3); r2 is running (PID 89983). | `kg/merge-teacher:ops/rebuild-2026-09-29/briefs/4-teacher.md`, `kg/merge-teacher:ops/rebuild-2026-09-29/trainer-model/4.1-*.log` | `kg/merge-teacher:cookbook/references/kaggriculture-teacher-distills-per-slot-kl-and-per-seat-winner-ce.md` (not on integration) | CPU only, with synthetic grammar tables. |
| 4.2 Targets and cache | approved, not merged | `a90b820` (tip `8fde43c`); staged as above | As 4.1 | `kg/merge-teacher:ops/rebuild-2026-09-29/trainer-model/4.2-*.log` | As 4.1 | Cache bytes (102,208 B per seat row) are computed, not measured on GPU. |
| 4.3 Model methods and wiring | approved, not merged | `b94b8dc` plus r1 fixes `8fde43c`; staged as above | `codex/verify-4-teacher-r1.md` APPROVE WITH EDITS; `codex/verify-4-teacher-r2.md` APPROVE | `kg/merge-teacher:ops/rebuild-2026-09-29/trainer-model/4.3-*.log`, `kg/merge-teacher:ops/rebuild-2026-09-29/merge-teacher/skip-probe.log` | As 4.1 | T18 and T19b are skipped or bypassed until the 3.1 Kaggriculture seam exists. Phase 4 is not complete until they run unbypassed. |
| 4.4 Teacher configs | not started | None | `codex/verify-4-teacher-r2.md` lists it as outstanding | None | None | Unblocked once the teacher merge lands. |

## Phase 5 — BC

| Task | State | Branch / commit | Codex verdict | Receipts | Cookbook | Gaps |
|---|---|---|---|---|---|---|
| 5.1 Data preparation | blocked (selection half merged) | `kg/rebuild-codex-data` `9d54a8b`, merged by `1177a5b` (selection only) | `codex/verify2-lane-D-r1.md` APPROVE | `rebuild/briefs/5.1-bc-data.md`, `rebuild/checks/` | `cookbook/references/rebuild-data-preparation-preserves-replay-identity.md` | `prepare.py` is not built; it needs the 1.4/1.5 explicit-state constructor. The 158,772 / 22,416 counts are not reproduced. |
| 5.2 BC training | not started | None | None | None | Historical only: `cookbook/references/bc-bootstrap-uses-native-replay-features-and-current-heads.md` | Needs 5.1, 1.5 and 3.5. |

## Phase 6 — GPU verification and pod evidence

| Task | State | Branch / commit | Codex verdict | Receipts | Cookbook | Gaps |
|---|---|---|---|---|---|---|
| 6.0 flash-attn on the pod | approved, not merged | `kg/rebuild-model` `4fd40c7`, `78df33c`, `bcd9627` | `codex/verify-flash-attn-r2.md` APPROVE (`78df33c`) | `kg/rebuild-model:ops/rebuild-2026-09-29/flash-attn-setup-2026-09-29/`; `kg/rebuild-model:ops/rebuild-2026-09-29/results.md` "Phase 6.0" | `kg/rebuild-model:cookbook/references/pod-v3-environment-runs-flash-attn-2-8-3-forward-on-sm120.md` | Forward path only. Trunk numerics are not qualified. |
| Evidence: GEMM limits | merged | `kg/rebuild-model` up to `432abe0`, merged by `053840f` | `codex/verify-gemm-limits-r3.md` APPROVE (`432abe0`) | `rebuild/gemm-limits-2026-09-29/`, `rebuild/run-statements/gemm-limits-probe.md`, `rebuild/results.md` (line 34) | `cookbook/references/compiled-gemm-template-overflows-above-2-21-rows.md` | Triton version and torch git identity were not captured at probe time. |
| Evidence: model-only SPS ceiling | approved, not merged | `kg/rebuild-model` `ed61770`, `cb4af49`, `ddf1fb2`, `5247daa` | `codex/verify-sps-ceiling-r2.md` APPROVE (`ddf1fb2`) | `kg/rebuild-model:ops/rebuild-2026-09-29/model-sps-ceiling-2026-09-29/` | `kg/rebuild-model:cookbook/references/model-only-sps-ceiling-bounds-per-rank-throughput.md` | Component estimates only. End-to-end SPS is unmeasured. |
| Evidence: ATEN-only GEMM A/B | approved, not merged | `kg/rebuild-model` `9904121`, `8fcfada`, `9bdd82d` | `codex/verify-aten-ab-r2.md` APPROVE (`9bdd82d`) | `kg/rebuild-model:ops/rebuild-2026-09-29/aten-gemm-ab-2026-09-29/` | `cookbook/decisions/kaggriculture-compiles-gemms-with-cublas-only.md` (on integration; its A/B source is on `kg/rebuild-model`) | Component timing only. The real-trunk backward above the bound is unmeasured. |
| Evidence: GPU checks bundle | approved, not merged | `kg/rebuild-gpu-checks` `24380f3` | `codex/verify-gpu-bundle-r3.md` APPROVE (`24380f3`) | `kg/rebuild-gpu-checks:ops/rebuild-2026-09-29/gpu-checks-2026-09-29/` | None on any branch | Needs a cookbook record. The channel-229 outlier is unexplained. The F3 cleanup fix has not run on the pod. |
| Evidence: value-gap diagnostic | in progress | `kg/rebuild-value-gap` `c9b6cde` (attempt 2 recorded, Amendment 2 before relaunch) | None yet | `kg/rebuild-value-gap:ops/rebuild-2026-09-29/run-statements/value-gap-diagnostic.md` | None | No result yet. Attempt 1 hit an fp32 Inductor compile error; attempt 2 was an fp32 eager OOM at 1,024 rows (per `c9b6cde`). |
| 6.1 Memory smoke, 2 ranks | not started | None | None | None | Target in `cookbook/decisions/start-multi-gpu-qualification-with-two-ranks.md` | Blocked on 1.4/1.5, 3.5, 5.2 and the evidence and teacher merges. |
| 6.2 Complete-work run, 2 ranks | not started | None | None | None | None | Depends on 6.1 and 5.2. |
| 6.3 Four ranks | not started | `configs/kaggriculture_4rank.yaml` exists; no run | None | None | None | Depends on 6.2. |
| 6.3b 8-rank qualification (no plan checkbox on integration) | approved, not merged | `kg/rebuild-8rank` `7ad45fc` | `codex/verify-8rank-r2.md` APPROVE | `kg/rebuild-8rank:ops/rebuild-2026-09-29/8rank/` | `kg/rebuild-8rank` edits three integration notes (unmerged) | No 8-GPU run. It needs a stated live price and owner approval. |
| 6.4 Orbit end-to-end | not started | None | None | None | None | No run statement. |

## Phase 7 — Evaluation and packaging

| Task | State | Branch / commit | Codex verdict | Receipts | Cookbook | Gaps |
|---|---|---|---|---|---|---|
| 7.1 Opponents | blocked | `kg/rebuild-7-1` at `b8747b6`, no commits. Uncommitted work is in `/Users/poonszesen/kg-v3-t71`. | `codex/task-7.1-impl-report.md`: "Stopped at your explicit API-boundary condition. Task 7.1 remains unimplemented." | `/Users/poonszesen/kg-v3-t71/ops/rebuild-2026-09-29/7.1/results.md` (uncommitted); `rebuild/briefs/7.1-opponents.md` | Uncommitted draft `frozen-engine-api-blocks-standalone-opponent-import.md` in `/Users/poonszesen/kg-v3-t71` | Byte-exact controllers need private engine items (`fib`, `Game.config`, missing `Game` accessors). No opponent is qualified. |
| 7.2 Panel script | not started | None | None | None | Contract: `cookbook/decisions/evaluation-preserves-generality-and-evidence.md` | Depends on 7.1 and 6.2. |
| 7.3 Replay export | not started (brief only) | Brief merged in `1177a5b` | Covered only by `codex/verify2-lane-D-r1.md` | `rebuild/briefs/7.3-replay-export.md` | Referenced from `rebuild-data-preparation-preserves-replay-identity.md` | Needs the 1.4 env. |
| 7.4 Packaging | in review | `kg/rebuild-7-4-brief` `2ac7ddb` (applies the review) | `codex/brief-7.4-review.md` REVISE (one P1, six P2) on `eccdd12`; no re-review yet | `kg/rebuild-7-4-brief:ops/rebuild-2026-09-29/briefs/7.4-packaging.md` | `kg/rebuild-7-4-brief:cookbook/references/kaggle-packaging-reuses-the-starter-submission-path.md` | No implementation. No submission is authorized; the owner decides. |
| 7.5 Parity docs | not started | Kaggriculture sections already in `docs/rules-parity-coverage.md` from 1.1/1.1b/1.3 | None | None | `cookbook/references/live-differential-parity-checks-the-rust-kernel.md` | Needs a closing pass after 7.1, 7.3 and 7.4. |

## Phase 8 — Docs and closeout

| Task | State | Branch / commit | Codex verdict | Receipts | Cookbook | Gaps |
|---|---|---|---|---|---|---|
| 8.1 References and rl-api-specs | not started | This tracker and the phase-grouped `cookbook/references/index.md` are a partial step (`kg/rebuild-phase-map`) | None | None | See "Notes not yet on integration" | The rl-api-specs differences are unchecked. |
| 8.2 Closeout | not started | None | None | None | None | Needs every earlier phase. |

## Cross-cutting

| Item | State | Branch / commit | Codex verdict | Receipts | Cookbook | Gaps |
|---|---|---|---|---|---|---|
| kaggle-environments 1.32.7 pin | merged | `kg/rebuild-deps` `c452f66`, merged by `69397da` | `codex/verify-deps-r3.md` APPROVE | `ops/deps-independent-verification-fb65e1f/` | `cookbook/decisions/restart-the-port-from-isaiahs-clean-base.md` | None open in r3. |
| cuBLAS-only compiled GEMMs | merged | `b2ef2bd`, `3b735c0`, `b51b0c0` on the integration first-parent line | `codex/verify-aten-r3.md` APPROVE (`b51b0c0`) | `rebuild/aten-wiring/` | `cookbook/decisions/kaggriculture-compiles-gemms-with-cublas-only.md` | CPU-tested enforcement. The cited A/B evidence is not on integration. |
| Stream C (3.1, 3.6, Phase 4 prep) | merged | `62899de`, merged by `07568e0` | `codex/verify2-lane-C-r2.md` APPROVE | `rebuild/stream-c-*.log`, `rebuild/4-prep-*.log` | `cookbook/references/ppo-trainer-seams-map-any-schema-and-alarm-on-replay-drift.md` | — |
| Stream D (selection, 5.1/7.1/7.3 briefs) | merged | `9d54a8b`, merged by `1177a5b` | `codex/verify2-lane-D-r1.md` APPROVE | `rebuild/checks/` | `cookbook/references/rebuild-data-preparation-preserves-replay-identity.md` | — |
| Cookbook refresh and codex exec workflow | approved, not merged | `kg/rebuild-model` `bcd9627` … `8093d51` | `codex/verify-cookbook-refresh-r3.md` APPROVE (`8093d51`) | `kg/rebuild-model:cookbook/log.md` | `kg/rebuild-model:cookbook/workflows/run-codex-exec-with-closed-stdin-and-wait-for-its-verdict.md` | The evidence merge has not started. The main worktree has uncommitted cookbook edits (architecture image). |
| Codex report custody | open | — | — | `codex/` (main worktree) | — | Integration tracks six cited reports (Task 0.1 review/rereview, Task 0.3 report, Task 3.4 r1–r3); `kg/merge-teacher` adds two. The other cited reports are untracked. |

## Notes not yet on integration

These notes are not linked from `cookbook/references/index.md` on this branch. Link each one when its branch merges. Found with `git cat-file -e <branch>:<path>` at 19:00 HKT.

| Note | Branches |
|---|---|
| `cookbook/references/pod-v3-environment-runs-flash-attn-2-8-3-forward-on-sm120.md` | `kg/rebuild-model`, `kg/rebuild-gpu-checks`, `kg/rebuild-value-gap` |
| `cookbook/references/model-only-sps-ceiling-bounds-per-rank-throughput.md` | `kg/rebuild-model`, `kg/rebuild-gpu-checks`, `kg/rebuild-value-gap` |
| `cookbook/workflows/run-codex-exec-with-closed-stdin-and-wait-for-its-verdict.md` | `kg/rebuild-model`, `kg/rebuild-gpu-checks`, `kg/rebuild-value-gap` |
| `cookbook/references/kaggriculture-teacher-distills-per-slot-kl-and-per-seat-winner-ce.md` | `kg/merge-teacher`, `kg/rebuild-trainer-model` |
| `cookbook/references/kaggle-packaging-reuses-the-starter-submission-path.md` | `kg/rebuild-7-4-brief` |
| `cookbook/references/frozen-engine-api-blocks-standalone-opponent-import.md` | Uncommitted in `/Users/poonszesen/kg-v3-t71` |

`kg/rebuild-model` (`8093d51`) predates the Phase 1 merges. Its copies of `native-game-semantics-use-v3-owned-buffers.md`, `failed-training-reports-status-before-distributed-cleanup.md`, `compiled-gemm-template-overflows-above-2-21-rows.md`, `cookbook/decisions/restart-the-port-from-isaiahs-clean-base.md` and `cookbook/references/index.md` differ from integration (`git diff --stat kg/isaiah-gap-closure kg/rebuild-model -- cookbook`). The evidence merge must reconcile them, and it must also keep this branch's phase grouping of the index.
