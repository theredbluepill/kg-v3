Reviewer: independent Claude subagent (substitute for Codex during its usage limit; owner-approved). Not a Codex verdict.

# Verify-merge: Task 3.1 remainder and Task 3.5 onto the integration

VERDICT: APPROVE (no blocking findings; three P3 notes below, none merge-induced)

## Scope

- Staging branch `kg/merge-3-1-3-5-c` in `/Users/poonszesen/kg-v3-m-3-1-3-5`.
- BASE (integration tip at lock time): `f02ed02` (the Task 4.4 landing, containing the landed 1.4/1.5 `5b43062` and 7.1 `994818b`).
- Merged: `kg/rebuild-3-1` `266c5e7`, based on the staged 1.4/1.5 merge `49a4835`. Merge commit `821b446`. The landing record commit follows it.
- This pass read the whole merge diff against each parent. It did not re-review the Task 3.1 remainder or Task 3.5 themselves. Those were reviewed by the Claude substitute reviews `claude-verify-3.1-{runppo,storage,tests}-r{1,2}.md` and `claude-verify-3.5.md`. Codex's `verify-3.1-rest2-r1` stopped at its usage limit and gave no verdict.
- The same agent that resolved the merge wrote this report. No subagents were spawned, as the landing instructions required. Treat it as a self-check that is independent of the branch reviews, not as a second reviewer.

## Nothing lost from either parent

- **Test functions.** Compared statically: every `def test_*` under `tests/` in `f02ed02`, in `kg/rebuild-3-1` and in the merge.
  - Integration: 1,093. `kg/rebuild-3-1`: 1,086. Merge: 1,134.
  - Every `kg/rebuild-3-1` test is present, and no name is duplicated.
  - Two integration tests are absent: `test_require_orbit_env_narrows_orbit_and_fails_fast_for_kaggriculture` and `test_kaggriculture_policy_evaluation_names_remaining_mapping_blocker`. Both exist at the merge base `8699ca9` and at `49a4835`, and Task 3.1's `15ea55f` deleted them on purpose: it removed `require_orbit_env` and replaced the evaluation stop with `test_kaggriculture_policy_evaluation_runs_native_games`. The merge carries over that deletion; nothing was lost.
- **Skips.** Integration `just prepare` had 21 skips (`merge-4-4-c/prepare.log`), and the merge has 17. The four missing skips are the Task 3.1 teacher trainer/run_ppo seam skips, which Task 3.1 removed on purpose. Every remaining skip is hardware-related or already on the integration: CUDA, pinned memory, flash-attn, x86 quantization, nine native-opponent-seat checks, one pod-bound regeneration and one sibling-repo reread.
- **Code.** Against `kg/rebuild-3-1`, the merge adds only integration-side content to `run_ppo.py` and `logging.py`: Task 4.4's `_require_kaggriculture_teacher_source`, the offline notice, `WANDB_MODES` and `wandb_init_identity`. Against `f02ed02`, it adds Task 3.1's seam and keeps every Task 4.4 function. `just prepare` includes mypy over `python/` and `scripts/`, which would catch a dangling reference.
- **Rust.** Neither side changed Rust relative to the other, and `engine_rs/TRIM_MANIFEST.json` is unchanged, so it was not regenerated. `check_engine_trim.py` passes inside `rs-format`.

## Resolutions checked

1. **W&B logger.** Task 4.4 and Task 3.1 each implemented a W&B logger independently, and their semantics conflicted. The merge keeps one logger:
   - From Task 4.4: `wandb_init_identity`, `WANDB_MODES` and the printed offline line.
   - From Task 3.1: `wandb_mode` naming, the run name `ppo-<run dir>`, Orbit's online `wandb.init` arguments unchanged, and offline mode rejected for a resume in both `WandbLogger` and `_validate_args`.
   - The printed line no longer says W&B "ignores resume offline", because that case is now rejected. Rejecting it follows the repository's fail-fast rule.
   - One Task 4.4 test combined an offline run with `resume_run_id`. It now resumes only when online and asserts that an offline run carries no id.
   - Two duplicate-coverage tests are kept (`test_validate_args_rejects_wandb_mode_without_wandb_logging` and `test_parse_args_reads_the_wandb_mode`). The first now matches Task 3.1's error text.
2. **Teacher source vs Task 3.1 launch tests.** Task 4.4 requires `--load-model-weights` or `rl.teacher_init` at a fresh Kaggriculture `last_best` launch. That broke 16 Task 3.1/3.5 and Task 4.4 launch tests, and each fix keeps the test's assertion target:
   - Task 4.4's `_teacher_source_argv` had written an empty file. Task 3.1's startup `env_steps` read now parses that file, so the helper writes a metadata-only checkpoint at step 0.
   - The seed, replay, seed-budget and rollout-factory tests use that helper.
   - The W&B-forwarding test and the Task 3.5 functional check pass a real tiny `rl.teacher_init` checkpoint. The functional check's teacher assertions were rewritten for a launch-time teacher. Promoted, last_best becomes the final weights. Held, last_best and the active teacher stay equal to the teacher checkpoint. In both branches the teacher distills from update 1 (`teacher/cache_bytes > 0`).
   - Task 4.4's two `_NOT_WIRED` tests expected the old stop. They now expect the launch to reach `_create_run_dir`.
3. **Configs.** Task 3.1's headers and `eval_replay_games: 0` are kept, together with Task 4.4's teacher comments. The 3.1 and 4.4 config tests both pass.
4. **Docs.**
   - README: combines Task 3.1's launch text with Task 4.4's cache-bytes text, and the launch example now passes a teacher checkpoint.
   - `model-architecture.md`: Task 3.1's paragraph ending, with Task 4.4's cache-bytes sentence inserted.
   - `rl-api-specs.md`: Task 3.1's seed and trainer sections, which describe the current code.
   - `rules-parity-coverage.md`: keeps the landed 1.4/1.5 and 7.1 text and adds a sentence for this merge's receipt.
   - The branch's copy of `merge-env-adapter/prepare.log` is the staged merge's receipt. The integration's landed receipt was kept at that path.
5. **Cookbook.**
   - The References index keeps the integration's phase grouping, with Task 3.1's updated entries for native semantics, evaluation, configs and seams. The teacher and W&B entries were rewritten.
   - `cookbook/log.md` is newest-first with no duplicate headings. The staged 1.4/1.5 entry from `kg/rebuild-3-1` was dropped as a duplicate of the landed one, and a merge entry was prepended.
   - The W&B, seams, configs and teacher References were revised: descriptions, inventory and limits. The teacher Reference's "Phase 4 is not complete" heading had become false once T18/T19b ran, and was corrected.
   - Pre-commit cookbook hooks passed on `821b446`.
6. **Plan and phase tracker.** The plan ticks 3.1, 3.5 and the Phase 4 box; the plan's own stated condition for Phase 4 (T18 and T19b run) is met. `phase-status.md` has a landing note and updated Phase 3, 4 and 6 summary rows and 3.1, 3.2, 3.3, 3.4, 3.5, 4.3, 4.4 and 6.1 rows. Each names the reviewers as Claude substitutes, not Codex.

## Checks run (this tree, Mac CPU)

- Full `CARGO_BUILD_JOBS=2 OMP_NUM_THREADS=2 uvx --from rust-just just prepare` on the merge resolution (`ops/rebuild-2026-09-29/merge-3-1-3-5/prepare.log`): exit 0.
  - Engine 69 (41 + 9 + 19). Root 274 passed, 5 ignored. Opponents 22.
  - Python 2,494 passed, 17 skipped.
  - Ruff, format, mypy and docs-fresh pass.
  - The recipe peaked at 3.31 GB max RSS, cargo included, above the 1 GB guideline for single checks. The targeted pytest runs below stayed under 0.41 GB.
- A final `just prepare` on the record commit's tree is `merge-3-1-3-5/prepare-record.log`.
- `tests/scripts/test_run_ppo.py` alone: 148 passed, 0.41 GB max RSS.
- Six merge-seam mutations (`merge-3-1-3-5/mutations.py`, `mutations.log`). Each was restored byte for byte, with the SHA-256 checked, and each was killed:
  - offline resume allowed: 2 failed;
  - Kaggriculture name without the `ppo-` prefix: 2 failed;
  - Orbit online gets `mode`: 2 failed;
  - teacher-source check dropped: 2 failed;
  - startup `env_steps` read dropped: 7 failed, including both T19b tests;
  - `rl.teacher_init` not loaded: 3 failed, including both functional-check cases.
- No training, GPU run or live W&B call.

## Findings

None is blocking. None was introduced by this merge; each is left as is, with the reason.

- **P3: two historical notes read as current.** Both statements are identical on both parents and left unchanged, and the current state is in the configs and trainer seams References.
  - `cookbook/decisions/recipe-choices-align-to-isaiah-without-owner-escalation.md` still says "`run_ppo` still stops explicitly after the workload check". The sentence sits in a paragraph dated to `kg/rebuild-configs` `99e5124`. It is not edited here because Decision notes quote the owner.
  - `cookbook/references/reward-reuse-preserves-objective-and-critic-semantics.md` still says trainer use "waits for Task 3.1", inside its dated Stage 2 section.
- **P3: the recorded 3.5 CLI command would now be rejected.** `ops/rebuild-2026-09-29/3.5/cli-command.txt` has no teacher source, so the merged tree would reject it at startup. It remains valid evidence for `266c5e7`. It was not rerun, and the seams Reference says so.
- **P3: cross-lane risk for the BC hand-off.** A `--load-model-weights` launch needs a checkpoint with the full PPO metadata schema (`env_steps`, `optimizer_steps`, `wandb_run_id` and the rest). This is Isaiah's existing requirement in `PPOTrainer.load_model_weights`. Task 3.1's startup read now enforces it before allocation. The BC best checkpoint must carry this schema, or be passed as `rl.teacher_init`, which reads only `model` plus an adjacent `config.yaml`. Neither route was checked against a real BC checkpoint here.

## Residual risks

- CPU only. Pinned buffers, CUDA transfer, compiled BF16 replay, multi-rank collection, live W&B sync and GPU teacher memory are unverified until Phase 6.
- `kg/rebuild-7-3` also edits `run_ppo.py` and `ppo.py`, so its landing must reconcile with this seam and with the teacher-source guard.
