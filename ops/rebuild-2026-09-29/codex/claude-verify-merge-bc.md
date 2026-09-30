Reviewer: independent Claude subagent (substitute for Codex during its usage limit; owner-approved). Not a Codex verdict.

# Verify-merge: Phase 5 BC onto the integration (`kg/merge-bc-c`)

Date: 2026-09-30. Mac, CPU only. No training, no GPU, no pod contact. This pass was written by the landing agent itself, so it is a self-check, not an independent second opinion.

## Target

- BASE: integration `kg/isaiah-gap-closure` at `d89ddec` (the Task 7.5 landing). The landing lock `/tmp/kg-v3-int-landing.lock` was held for the whole landing.
- Staging branch `kg/merge-bc-c` in `/Users/poonszesen/kg-v3-m-bc`, three `--no-ff` merges plus a landing record:
  - `18c534b`: `kg/rebuild-bc-handoff` `41c95f7`. It contains `kg/rebuild-bc-now` up to `218a05b`: the 5.1 preparer (Codex `verify-5.1-prepare-r2`, `verify-5.1-team-filter-r2` APPROVE), the 5.2 trainer (Codex `verify-5.2-trainer-r3` APPROVE), the A100 receipts, and the BC-best warm start (Claude stand-in `claude-verify-bchandoff-r4` APPROVE).
  - `6c82f69`: `kg/rebuild-bc-now` `954f640`, the one bc-now commit not in bc-handoff (edits from the Codex receipt review).
  - `5399139`: `kg/rebuild-bc-brief` `74428e8` (5.1 brief, Codex `brief-5.1-owner-edits-r4` APPROVE).
  - The landing record commit follows. It contains the merge-induced fixes, the cookbook, plan, tracker and custody edits, and this report.

## Questions and findings

### Q1. Is the integration's newer Task 1.4 kept?

The BC branches fork from `2390c8e` and carry `1e63597`, an older Task 1.4 than the integration's `b6b722f` and later. After the handoff merge, all 482 paths in `git diff --name-only 1e63597 b6b722f` equal BASE, except `cookbook/log.md` and `cookbook/references/index.md`. Those two are shared cookbook files that were merged by hand, not 1.4 code. The bc-now and brief merges touch no 1.4 file. `cookbook/references/native-game-semantics-use-v3-owned-buffers.md` and `evaluation-and-truncation-follow-the-kaggriculture-objective.md` also equal BASE. **PASS.**

### Q2. Are the conflicts resolved semantically, keeping both sides?

- `scripts/run_ppo.py`:
  - imports: `json` from the handoff and `os` from the W&B landing;
  - `owl.train.ppo`: `_obs_to_device` from 3.1, plus `CHECKPOINT_KEYS`, `OPTIONAL_CHECKPOINT_KEYS` and `reject_unknown_checkpoint_keys` from the handoff;
  - constants: `WARM_START_RECORD`, `_JOB_TYPE` and `_SCRIPT_DIR`. The handoff's `_TRAINER` is dropped, because nothing on the merged tree uses it: 3.1 removed `require_orbit_env`;
  - functions: `_check_launch_telemetry` and `_warm_start_record` are both kept.
- `tests/scripts/test_run_ppo.py`: imports `math` and `os`.
- `docs/rl-api-specs.md`: keeps the integration's Environment paragraph. It supersedes the handoff's older text, which only restated the pre-3.1 seed rule.
- `cookbook/references/index.md`:
  - keeps the phase grouping;
  - adds the three BC lines under "Phase 5 and 7";
  - takes the brief's refreshed data-preparation line.
- `cookbook/log.md`: a paragraph-set check showed that the merged log holds every paragraph from both parents, for each of the three merges. The one brief-side paragraph with no blank line before the next heading is kept in the integration's corrected form.
- `cookbook/references/top-1-team-bc-checkpoint-is-selected-by-held-out-nll-only.md`: takes bc-now's synced W&B line and the handoff's newer load line.

**PASS.**

### Q3. Does the warm-start mode reconcile with 3.1's `run_ppo` and 4.4's teacher-source check?

- `_require_kaggriculture_teacher_source` returns as soon as `launch.load_model_weights_path` is set, so `--load-model-weights` satisfies it in every mode, `model_fresh_critic_head` included.
- A fresh `last_best` launch copies the teacher from the student after `load_model_weights(..., fresh_state_keys=...)`, so the teacher also has the fresh critic head. The critic head does not enter the actor KL, and the pod pre-landing staging branch `kg/pod-ppo-prelanding` resolved it the same way (`932ab3c`, `394fe02`; read, not modified).
- 3.1's `_require_unchanged_start_env_steps` applies to the warm-start path, and the BC best records `env_steps` 0.

A 3.1 seam test failed on the first `just prepare` (`merge-bc-c/prepare-r1-failed.log`: 1 failed, 2,666 passed). `test_main_kaggriculture_fails_when_the_checkpoint_changes_during_startup[load_model_weights]` uses a test double whose `load_model_weights` lacked the new `fresh_state_keys` keyword. The double now takes it and asserts `frozenset()` for the default `model_only` launch. This is a merge-induced fix to 3.1's test, not to production code. **PASS after the fix.**

### Q4. Do the BC configs load under the integration's schema and guards?

`configs/kaggriculture_1gpu_eager.yaml` was written against the older Task 1.5 reward schema and before 3.1's replay guard. Without `econ_ineffective_cap`, `FullConfig` rejects it. With `eval_replay_games: 8`, `run_ppo` rejects it at startup ("requires Task 7.3").

The landing adds `econ_ineffective_cap: 0.1` and sets `eval_replay_games: 0`, as in the 2-rank config. A header comment records that the A100 run read the file at `f0b7a38`. Neither field enters BC training: a BC resume compares against the run directory's own saved copy.

A new test, `test_one_gpu_ppo_config_is_the_two_rank_config_on_one_rank`, checks that this config equals the 2-rank config apart from the per-rank shape and eager compile. It fails when either revert is applied (`merge-bc-c/mutations.log`: M1 `eval_replay_games` 0→8, 1 failed; M2 drop the cap, 1 failed). Both reverts were restored byte-exact. **PASS.**

### Q5. Are test names preserved three-way?

Every `def test_*` on BASE, bc-handoff, bc-now and bc-brief exists on the merged tree, with one exception. BASE's `test_run_training_session_sets_trainable_parameter_summary` is the handoff's rename to `test_run_training_session_sets_launch_summaries`. The merged version keeps the W&B landing's `identity=_identity()` argument and adds the `warm_start/*` summary keys. The number of `def test_*` definitions is 1,185 on BASE and 1,237 after the merge. **PASS.**

### Q6. Does the real BC best still load on the merged tree?

`real-bc-best-tests.log` ran the checkpoint from a local copy (`/private/tmp/kg-v3-bc-best/checkpoint_bc_best.pt`, SHA-256 `fd8545872aca59c70e273e9655055e1104cd719588364f463ae87b0d488e6f51`) with the top-1 shards. The 16 handoff tests selected with `-k "bc_best or ranked or prohibited or fresh_critic"` pass on the merged tree, including `test_real_bc_best_loads_and_forwards_on_a_real_shard` and all three load modes into the eager, 2-, 4- and 8-rank configs. This tree builds the actor from the native grammar tables, which closes the handoff Reference's "load was not rerun on that tree" gap. **PASS.**

### Q7. Does full preparation pass?

`CARGO_BUILD_JOBS=2 OMP_NUM_THREADS=2 uvx --from rust-just just prepare` on the merge plus the landing edits exits 0 (`ops/rebuild-2026-09-29/merge-bc-c/prepare.log`):

- ruff and format (one file reformatted);
- mypy, no issues in 76 source files;
- docs-lint;
- root Rust 274 passed, 5 ignored, plus the engine and opponent crates;
- Python 2,668 passed, 18 skipped;
- docs-fresh: "No doc updates required".

The skip list includes the real-checkpoint test, which is environment-gated (Q6 ran it). **PASS.**

### Q8. Are the records current?

- **Plan.** 5.1 and 5.2 are checked. Each names its deviation from the plan text: the one preparer run used the owner's top-1 team data, not the 252-episode slice, and the run used 1× A100 eager, not two RTX PRO 6000 ranks. The entry also records the checkpoint SHA-256 and W&B `kvl4rfda`.
- **Tracker.** `phase-status.md` updates the Phase 5 rows, the 6.1/6.2 gaps and the W&B gap, and names the reviewers.
- **Cookbook.** Four References are revised.
  - Handoff Reference: r4's P3-1 and P3-2 are applied, the stale "run_ppo stops before the environment" limit is corrected, and the load rerun is recorded.
  - BC trainer Reference: now names the real run and the pending W&B-gate adoption.
  - Data-preparation Reference: records the preparer's landing and the unrun reference-slice comparison.
  - Credential-gate Reference: retitled, because `train_bc` is now on integration and has not adopted the gate, so "All v3 launchers on integration fail fast" was false. Link titles in two notes and in the index are updated to match.
- **Log.** `cookbook/log.md` has the landing entry.
- **Custody.** The BC review files move under custody: 28 former DEFER entries (6 of them already tracked, identical) and 18 post-inventory files. Two transcripts over 512 KiB stay local. Every source was re-hashed and scanned; see `evidence-custody.md` "Phase 5 BC landing".

**PASS.**

## Residual risks (non-blocking)

- P3: `scripts/train_bc.py` is now on integration with its own `BCWandbLogger`. It does not have the credential gate, the `attempts.jsonl` receipt or the loud offline outage that `run_ppo` has. The W&B Reference and the tracker record this as the next workflow step.
- P3: `configs/kaggriculture_1gpu_eager.yaml` now differs from the file the A100 run read at `f0b7a38`. The run directory's saved `config.yaml` stays the provenance of that run, and the header comment says so.
- Informational: no PPO update has run from the BC best, and `run_ppo.main()` has not loaded the real checkpoint; only the load function and a fake-trainer `main` have. The warm-start SHA-256 is the main rank's file (r4 P3-3).
- Informational: this verification is the landing agent's own, not an independent reviewer's, and not a Codex verdict.

VERDICT: APPROVE
