Reviewer: independent Claude subagent (substitute for Codex during its usage limit; owner-approved). Not a Codex verdict.

# Verify-merge: W&B wiring for all v3 experiments onto the integration

VERDICT: APPROVE

- **Staging branch.** `kg/merge-wandb-c` in `/Users/poonszesen/kg-v3-m-wandb`.
- **BASE.** The integration tip `5ed1c1a` (`kg/isaiah-gap-closure`, the Tasks 3.1/3.5 landing).
- **Merged.** `kg/rebuild-wandb` `8d5838e` (claude-verify-wandb-r3 APPROVE WITH EDITS, edits applied), with `git merge --no-ff`. The merge base is `994818b`.
- **Scope.** The merge's own resolutions only. The two parents were reviewed separately: `claude-verify-wandb-r1..r3`, `claude-verify-3.1-*`, `claude-verify-3.5`, `claude-verify-merge-3-1-3-5` and `verify-4.4-r1`.
- **Who did what.** The same subagent resolved and verified the merge, as the workflow directed (no further subagents). This pass re-read the merged diff against each parent rather than trusting the resolution notes.

## Conflicts and resolutions

Seven files conflicted: `README.md`, `cookbook/log.md`, `cookbook/references/index.md`, `python/owl/train/logging.py`, `scripts/run_ppo.py`, `tests/owl/train/test_logging.py` and `tests/scripts/test_run_ppo.py`. `scripts/run_ppo.py` had also auto-merged a duplicate `--wandb-mode` argparse option (one from each side), which would have raised `argparse.ArgumentError` at parse time; the merge removes Task 3.1's copy.

The two sides wired the same logger differently. The merge keeps the W&B branch's gated path (credential gate, `TelemetryMode`, `RunIdentity`, `attempts.jsonl`, observed-mode check, `create_metric_logger`) and folds Task 3.1's semantics into it.

| Point | Integration (Task 3.1/4.4) | `kg/rebuild-wandb` | Merged | Why |
|---|---|---|---|---|
| Flag | `--wandb-mode` with `WANDB_MODES` literal tuple | `--wandb-mode` with `WandbMode` enum | one flag, `WandbMode`; help text keeps "resume launches require online" | no duplicate flags |
| Project | Kaggriculture `kg-v3`, Orbit `orbit-wars` with Isaiah's arguments | `kg-v3` for every game, `game` tag | `kg-v3` for every game | one path: Orbit could not keep Isaiah's bare arguments and also get the gate, receipt, observed-mode check and v3 summary; the owner rule is W&B telemetry in `kg-v3`; the `orbit` tag separates Orbit runs |
| Run name | `ppo-<run dir>` (Kaggriculture) | `<run dir>` | `<job type>-<run dir>` (`ppo-…`, `bc-…`) | keeps Task 3.1's name; distinguishes launchers sharing a run-directory name |
| Group | constant `ppo` | experiment id | experiment id | stable v3 experiment identity (CLAUDE.md custody rule); the job type stays in `job_type` and the tags |
| Offline resume | rejected in `_validate_args` and `WandbLogger` | allowed; wandb starts a same-ID offline segment, sync behaviour unverified | rejected in both places | fail fast; offline only when explicit; an unverified segment merge could lose telemetry history |
| Debug + offline | `_validate_args`: "requires --log-mode wandb" | `telemetry_mode`: "applies only to --log-mode wandb" | one check in `telemetry_mode`; its message keeps both phrasings | one source of truth; both sides' tests keep matching |
| Offline notice | stdout line naming `<run_dir>/wandb` | startup banner and "OUTAGE recorded" line on stderr | banner, receipt line, and Task 3.1's run-dir line (now on stderr, after the receipt) | outage visible and repairable |

## Nothing lost from either parent

- **Test names.** Top-level `def test_*` names were compared across `tests/owl/train/test_logging.py`, `tests/scripts/test_run_ppo.py` and `tests/scripts/test_export_wandb_netrc_entry.py` (46, 141 and 3 merged).
  - Every name from `5ed1c1a` survives.
  - Every name from `8d5838e` survives except `test_kaggriculture_policy_evaluation_names_remaining_mapping_blocker`. The integration deleted that test in Task 3.1's `15ea55f`, when evaluation mapping landed, and the W&B branch never edited it, so its absence is the integration's change.
- **Tests adapted to the unified API, same names.**
  - Task 3.1's seven logger tests now build a `RunIdentity`. The fake `wandb` run gains `project`, `entity`, `url`, `offline` and `disabled`.
  - `test_orbit_wandb_init_kwargs_are_unchanged` keeps its name. It now asserts that Orbit takes the shared v3 arguments (a comment records the change). This is the one deliberate change of meaning, from the project decision above.
  - `test_validate_args_rejects_wandb_mode_without_wandb_logging` duplicated `test_validate_args_rejects_offline_wandb_with_debug_logging`. It now drives the same rejection through `_parse_args`.
  - The W&B branch's `test_resume_forwards_the_saved_run_id_with_resume_must` is online only; offline resume is covered by `test_offline_wandb_rejects_resume_before_initialization`.
  - `test_main_resume_with_a_receipt_plans_attempt_one` and `test_main_resume_without_receipts_fails_before_the_env` resume online with a test key. Attempt 0 in the receipt stays offline, which is the documented sync-then-resume case.
  - `test_main_offline_mode_announces_the_outage_without_credentials` now reaches `_create_run_dir`, because Kaggriculture is wired. It passes a teacher source, since Task 4.4 requires one.
- **One-sided files.** For every file only one side changed, the merged bytes equal that side's, except the three W&B cookbook notes, which were revised on purpose.

## Merge-induced breakage found and fixed

1. **Duplicate `--wandb-mode` option (auto-merged).** Removed, as above.
2. **Task 3.1 resume tests had no `attempts.jsonl`.** A resume now requires one: `test_main_kaggriculture_resume_starts_a_disjoint_seed_stream`, `…fails_when_the_checkpoint_changes_during_startup[resume]` and `test_resume_startup_checks_the_runtime_adapted_workload`. Each now writes a one-attempt online receipt.
3. **T19b in `tests/kaggriculture/test_teacher.py`.** `test_run_ppo_resume_restores_the_teacher_from_checkpoint_last_best` failed in `just prepare` for lack of a receipt (`merge-wandb-c/prepare-2-teacher-resume-fail.log`).
   - Its `_run_ppo_main` also depended on the host's `~/.netrc` for the online gate, so on a machine without a W&B credential both of its launch tests would fail.
   - Fix: the helper sets a test key and unsets `WANDB_MODE`, and the resume test writes a receipt.
   - The launch-related tests (`tests/scripts`, `tests/owl/train`, `test_teacher.py`, `test_training_smoke.py`, `test_configs.py`) pass with an empty `HOME` and no `WANDB_*`/`NETRC`: 897 passed, 1 skipped (`merge-wandb-c/hermetic-home-shard.log`).
4. **E501 on the moved offline line.** Wrapped (`merge-wandb-c/prepare-1-lint-fail.log`).

## Checks on the merged tree

- **Full preparation.** `CARGO_BUILD_JOBS=2 OMP_NUM_THREADS=2 uvx --from rust-just just prepare` exited 0 (`ops/rebuild-2026-09-29/merge-wandb-c/prepare.log`). It covers ruff, format, the 3.11 syntax check, docs-lint, mypy (72 files, no issues), Rust (root 274 passed and 5 ignored, plus the engine and opponent crates), Python 2,587 passed and 17 skipped, and docs-fresh.
- **Mutations.** Seven mutations of the merge's own resolutions were run (`ops/rebuild-2026-09-29/merge-wandb-c/mutations.py`, `mutations.log`). All were killed by `test_logging.py` plus `test_run_ppo.py` and restored byte for byte (SHA-256 checked):
  - drop the offline-resume check in `_validate_args` (1 failure) or in `WandbLogger` (2);
  - name back to `<run dir>` (6);
  - drop the offline run-dir line (1);
  - Orbit back to `orbit-wars` (4);
  - debug accepting offline (4);
  - group back to `ppo` (7).
- **TRIM_MANIFEST.** `engine_rs/` differs from neither parent, so `engine_rs/TRIM_MANIFEST.json` needs no regeneration. `scripts/check_engine_trim.py` passed inside `just prepare`.

## Docs and cookbook consistency

- **README.** It has one W&B launch section: gate, flags, receipt, run shape and online-only resume. The later training-logging paragraph now points to it instead of claiming `orbit-wars` for Orbit.
- **Cookbook notes.**
  - `v3-launchers-fail-fast-without-wandb-credentials.md` gains a "Landing merge" section and corrected limits: offline resume rejected, and Kaggriculture now runs.
  - `ppo-runs-publish-kaggriculture-telemetry-to-the-v3-wandb-project.md` now describes the current state, with Task 3.1's helper kept as history.
  - The trainer-seams note gains a pointer.
  - The pod Workflow's offline-resume advice is replaced.
  - Index lines and descriptions agree with the note bodies.
- **Log.** `cookbook/log.md` keeps both sides' entries once each, newest first by commit time, under a new landing entry.

## Findings

No P1 or P2 remains. P3 (recorded, not blocking):

- **P3-1.** Moving Orbit from `orbit-wars` to `kg-v3` is a merge-time decision between two reviewed parents. It follows the owner rule ("W&B telemetry in kg-v3") and the W&B branch's reviewed design, not an explicit owner ruling on Orbit. Reopen it if the owner wants Isaiah's Orbit runs kept in `orbit-wars`.
- **P3-2.** No live W&B call, pod run or `wandb sync` of an offline run followed by an online resume has run. All W&B checks use test doubles.
