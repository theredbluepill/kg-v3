Reviewer: independent Claude subagent (substitute for Codex during its usage limit; owner-approved). Not a Codex verdict.

# Verify-merge: learner-perspective bank telemetry onto the integration

- **Staging branch.** `kg/merge-bank-metrics-c` in `/Users/poonszesen/kg-v3-m-bank-metrics`. Merge commit `cb82973` has parents `4895b4e` and `1759764`. The landing record comes after it.
- **BASE.** The integration tip `4895b4e` (`kg/isaiah-gap-closure`, the Phase 5 BC landing). It contains the landed 3.1/3.5 (`821b446`), the W&B wiring (`1fb5bcc`), 7.5 (`458505c`) and BC (`18c534b`, `6c82f69`, `5399139`).
- **Merged.** `kg/rebuild-bank-metrics` `1759764`, merged with `git merge --no-ff`. The merge base is `2413c9e`, the cut point on `kg/rebuild-3-1`. The branch's review, `codex/claude-verify-bank-metrics-r1.md`, returned APPROVE WITH EDITS on `7425db7`, and its edits are applied at `1759764`.
- **Scope.** This covers the merge's own resolutions and how the branch fits the integration's single W&B path. `claude-verify-bank-metrics-r1` reviewed the branch's content, and earlier verify-merge reports reviewed the integration's content.
- **Who did what.** The landing subagent resolved the merge and then verified it here, so this pass is not independent of the resolution. It re-read the merged diff against both parents and did not rely on the resolution notes.

## Conflicts and resolutions

Three files conflicted. Both sides were kept in each.

| File | Conflict | Resolution |
|---|---|---|
| `cookbook/log.md` | The branch's two entries (01:06 and 01:15 HKT) against the integration's entries up to the BC landing (03:20). | The branch entries go at their chronological place: below the 01:47 Task 3.5 entry and above the 01:06 "Close claude-verify-wandb-r3" entry. The log stays newest-first, and no integration entry changed (`git diff HEAD^1 -- cookbook/log.md` is +8 lines, no deletions). |
| `cookbook/references/index.md` | The integration regrouped the index by phase after `2413c9e`. The branch added one line to the old flat list. | The integration's layout is kept. The branch's single line (`learner-bank-telemetry-…`) goes under Phase 3, after the W&B telemetry Reference. The other lines on the branch side (model-only SPS, flash-attn, teacher) are integration notes that appeared only as conflict context. They stay in their Phase 4/6 places. The index diff against BASE is +1 line. |
| `tests/scripts/test_run_ppo.py` | Both sides appended blocks at the end of the file: the W&B gate and attempt-receipt tests on the integration, and the two bank telemetry tests on the branch. | Both blocks kept, integration first. |

`python/owl/train/ppo.py`, `scripts/run_ppo.py`, `README.md`, `docs/rl-api-specs.md`, `tests/kaggriculture/test_teacher.py`, `tests/kaggriculture/test_training_smoke.py` and `tests/owl/train/test_ppo.py` auto-merged. The code diff against BASE matches the branch's diff against `2413c9e` exactly (+29 in `ppo.py`, +10 in `run_ppo.py`, and the new `telemetry.py`).

## Reconciliation with the single W&B path

- The integration's W&B landing (`1fb5bcc`) left one logger path: `create_metric_logger` returns a `WandbLogger` for project `kg-v3` behind the credential gate and receipt, or a `DebugLogger`. `WandbLogger.log` calls `wandb.log(metrics, step=step)`, and nothing filters keys (`python/owl/train/logging.py:675-676`).
- The branch adds no logger, flag or project. Its training keys join the `train_iteration` metrics dict for Kaggriculture batches, and its eval keys join `_evaluate_against_last_best`'s dict for `KaggricultureObsConfig`. Both dicts reach the same `logger.log`. No second W&B path was introduced, and none needed removing.
- The branch's end-to-end test, `test_kaggriculture_bank_telemetry_reaches_the_logger`, drives the integration's `_run_training_loop` into `_FakeLogger`. It passes on the merged tree, so the post-W&B loop signature is compatible.
- The BC warm start (`fresh_state_keys`, `CHECKPOINT_KEYS`) touches loading, not `train_iteration`'s metrics. No bank key reaches a checkpoint, and `reject_unknown_checkpoint_keys` is unchanged.
- Fake Kaggriculture steps: after the merge, `step` fakes appear only in `test_teacher.py`, `test_ppo.py` and `test_run_ppo.py`. The suite passes, so none of the integration's fakes returns `{}` into a Kaggriculture `train_iteration`. That would now fail fast by design.
- The note's limit, "`run_ppo` has no fixed-opponent panel", still holds at BASE. `run_ppo.py` has no panel code, and Task 7.1's `opponents_rs` seat hook is still skipped.

## Nothing lost from either parent

I compared top-level `def test_*` names across `tests/scripts/test_run_ppo.py` (BASE 142, branch 121, merged 145), `tests/owl/train/test_ppo.py` (66 / 68 / 68), `tests/kaggriculture/test_training_smoke.py` (3 / 3 / 3), `tests/kaggriculture/test_teacher.py` (33 / 33 / 33) and `tests/kaggriculture/test_telemetry.py` (0 / 8 / 8). Every BASE name survives, and the merged files have no duplicate names. One branch name is absent: `test_run_training_session_sets_trainable_parameter_summary`. The integration renamed it to `test_run_training_session_sets_launch_summaries` in BC commit `2d00284`, which the branch never edited, so the rename is the integration's change and not a merge loss.

## Checks run on this merge

| Check | Result |
|---|---|
| `uvx --from rust-just just prepare` on the merged tree (before commit `cb82973`, which has identical content) | exit 0. Rust: 274 passed, 5 ignored, plus 41, 9 and 22. Python: 2,683 passed, 18 skipped (CUDA, flash-attn, pod-bound, the `opponents_rs` hook and the env-gated real BC best). Docs freshness: "No doc updates required". Log: `ops/rebuild-2026-09-29/merge-bank-metrics-c/prepare.log`. |
| Conflict-marker scan of the three resolved files | none |
| Five mutations on the merged tree, each reverted (`ops/rebuild-2026-09-29/merge-bank-metrics-c/mutations.log`; baseline 309 passed across the five touched test modules, `-m "not slow"`) | 5/5 killed |

| # | Mutation | Killing tests |
|---|---|---|
| MM1 | drop the training hook | native smoke, truncation smoke, run_ppo end-to-end |
| MM2 | evaluation reads the seat-ordered `bank_0`/`bank_1` | run_ppo end-to-end and the pinned mixed-seat test |
| MM3 | evaluation telemetry emitted for Orbit | 4 tests, including `test_orbit_evaluation_logs_no_bank_telemetry` |
| MM4 | training telemetry emitted for Orbit | 20 Orbit trainer tests |
| MM5 | skip the cross-rank gather | rank-pooling test |

## Findings

No correctness defect. Two P3 observations follow; neither blocks the landing.

1. **P3 (carried).** The per-update `all_gather_object` for bank lists is a second pickled collective beside the existing env-metric key gather. Its cost on a multi-rank pod run is unmeasured, and this merge does not change that (branch review finding 2).
2. **P3 (naming, carried).** The W&B project now also holds `eval/margin_0` (seat 0), `eval/candidate_bank_margin` and `eval/margin_*` (candidate minus last-best). The README and `docs/rl-api-specs.md` distinguish them, but no W&B panel note exists yet.

## Limits

This ran on Mac CPU only. I did not exercise a live W&B upload, a real multi-rank gather, CUDA, or any pod path, and I did not touch the pod worktree. Because the landing subagent performed both the resolution and this check, the check is not independent of the resolution.

VERDICT: APPROVE
