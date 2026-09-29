Reviewer: independent Claude subagent (substitute for Codex during its usage limit; owner-approved). Not a Codex verdict.

# Verify: BC trainer on the shared v3 W&B path (`kg/rebuild-bc-wandb` `619bf50`)

**Scope.** This review covers `scripts/train_bc.py`, `python/owl/train/bc.py`, `python/owl/train/logging.py` (`json_sha256`, `announce_recorded_outage`), `scripts/run_ppo.py` (the outage lines moved into the helper), `tests/kaggriculture/test_bc.py`, the README BC section and the cookbook edits. The base is the integration tip `ff9ebcb`. The review ran as a separate pass in a scratch worktree (detached at `619bf50`, under the session scratchpad), not in the implementation worktree.

**Requirement checked** (owner: "make sure all v3 experiments wired to w&b"):

- project `kg-v3` with v3 identifiers;
- online by default;
- fail fast without credentials, naming the fix;
- offline only by explicit flag, with a loud warning, recorded in `bc_result.json` and the receipts;
- reuse of the helper, not a fork.

## Findings against the requirement

| Requirement | Evidence | Status |
| --- | --- | --- |
| Project `kg-v3`, v3 identifiers | `create_metric_logger(game="kaggriculture")` with `plan_attempt(job_type="bc")`. Test `test_script_logs_online_to_the_v3_project_by_default` checks the init project, job type `bc`, group = `--experiment-id`, tags `kaggriculture-v3`/`bc`/`kaggriculture`, `config["v3"]`, the `v3/*` summary and `attempts.jsonl` (project, run ID, config hash equal to `bc_config_sha256`). | Met |
| Online default | `--wandb-mode` is the shared `WandbMode`, default `ONLINE`. The same test launches with no mode flag and asserts `mode == "online"`. | Met |
| Fail fast, naming the fix | `_check_launch_telemetry` is the first call inside the session, before `load_bc_configs`, `load_bc_dataset` and `resolve_source_commit`. The parametrized test patches those three to raise and checks `MissingWandbCredentialsError` naming `WANDB_API_KEY`, the pod workflow and `--wandb-mode offline`, and that it rejects a contradicting `WANDB_MODE`. No run directory is created and `wandb.init` is never called. | Met |
| Offline only by explicit flag, loud, recorded | Offline and debug each run without credentials. Stderr carries the banner and the "recorded" line (and, offline, the `wandb sync` hint). `telemetry_mode` is in `attempts.jsonl`, `bc_attempts.jsonl`, `checkpoint_bc_best.json`, `bc_result.json` and the final stdout JSON. | Met |
| Helper reused, not forked | `BCWandbLogger`/`create_bc_logger` are deleted. The gate, `plan_attempt`, `create_metric_logger`, `record_attempt` and `resolve_source_commit` are called directly. The shared outage announcement moved into `owl.train.logging` and `run_ppo` calls it. `config_sha256` now delegates to `json_sha256`. | Met |
| Restart rules match `run_ppo` | Restarts reject debug, offline and `--experiment-id`; a debug-origin `bc_state.pt` (no W&B run ID) is rejected before any receipt; a restart continues the saved run (`id`, `resume="must"`) and experiment. | Met |

## Mutations (at least 3 required; 15 run, scratch worktree)

The mutations are listed in `ops/rebuild-2026-09-29/bc-wandb/review-mutations.log`, and the runner is `review-mutations.py` beside it. Each mutation was applied, the suite `tests/kaggriculture/test_bc.py` was run, and the file was restored with `git checkout`. The tree was clean afterwards.

| # | Mutation | Result |
| --- | --- | --- |
| M1 | Rank 0 skips the credential gate | killed (fail-fast test) |
| M2 | BC configs load before the gate | killed (fail-fast test) |
| M3 | `telemetry_mode` dropped from the BC receipt/provenance | killed |
| M4 | Offline restart accepted | killed |
| M5 | Debug-origin restart accepted | killed |
| M6 | Wrong `game` tag | killed |
| M7 | BC receipt vs shared receipt agreement unchecked | killed |
| M8 | `wandb_run_id` dropped from `bc_result.json` | killed (rerun separately: the batch runner's pattern missed the formatted line) |
| M9 | Recorded-outage line not printed | killed |
| M10 | `--source-commit` overrides git again (the old `_git_head` behaviour) | killed |
| M11 | Shared `attempts.jsonl` not written | killed |
| M12 | `--experiment-id` not forwarded | killed |
| M13 | Settings hash includes the `ppo_config` path | killed |
| M14 | Restart `--experiment-id` accepted | killed |
| M15 | Parse-time debug-plus-offline check removed | survived: equivalent. `check_telemetry` rejects the same pair, with the same message, before any run directory exists (the same redundancy as `run_ppo`). |

## Live-library probe

The reviewer ran one `train_bc.main()` with the real wandb 0.26.1, not the fake, in `--wandb-mode offline`, without credentials, on the tiny synthetic shards (CPU, 2 steps). The temporary test file was deleted afterwards. It exited 0. wandb wrote `offline-run-*-ettkydp5`. `attempts.jsonl` recorded `job_type` `bc`, `telemetry_mode` `wandb-offline`, project `kg-v3`, experiment `probe-exp` and run ID `ettkydp5`, and `bc_result.json` carried the same mode and run ID. The banner and the sync hint printed (`ops/rebuild-2026-09-29/bc-wandb/real-wandb-offline-probe.log`). So the real SDK accepts BC's init arguments and config, and the observed-mode check passes offline. No online call was made.

## Findings

- **P3-1 (applied).** `_resume_wandb_run_id` returned a value that `main` discarded, re-reading `resume.wandb_run_id`. `main` now passes the checked value. Behaviour is unchanged, and `test_bc.py` passes (63 passed, 1 skipped).
- **P3-2 (accepted, recorded).** The two receipts are appended back to back, `attempts.jsonl` first and then `bc_attempts.jsonl`, after W&B starts. A crash between the two appends would make every later restart fail loudly with "receipts are inconsistent", and the directory would need a manual repair. The window is two file appends, and the failure is explicit, not silent.
- **P3-3 (accepted, recorded).** Restarting a run whose earlier attempt was offline and never synced fails inside the online `wandb.init`, after data load and model build. That is the contract `run_ppo` and the pod workflow already document (sync first). No early check exists, because the launcher cannot know the sync state.
- **P3-4 (accepted, recorded).** The W&B name stays `bc-bc-<timestamp>`, because the helper names runs `<job type>-<run dir>` and BC run directories keep their `bc-` prefix. The credential Reference records this. Renaming the directories would make the default experiment ID a bare timestamp, like PPO's.
- **Observation (pre-existing, unverified live).** On a restart the W&B config passed to `wandb.init` includes the new attempt's provenance, so the resumed run's config changes. The old BC logger and `run_ppo` behave the same way; this change adds nothing, and no live resume has run.

No P1 or P2 finding. Nothing blocks.

## Checks run by the reviewer

- Baseline `tests/kaggriculture/test_bc.py` at `619bf50` in the scratch worktree: 63 passed, 1 skipped (the real-checkpoint test, which needs `KG_V3_BC_BEST`).
- 15 mutations as above, plus the real-wandb offline probe.
- The implementer's `just py-prepare` receipt (`ops/rebuild-2026-09-29/bc-wandb/py-prepare.log`: 2,699 passed, 18 skipped, ruff, format, mypy, docs-fresh) was read, not rerun by the reviewer.

**Limits.** No live online W&B call, GPU, multi-rank or real-shard BC run. The fake-wandb tests use wandb's real `Settings` validator and errors. Nothing checked the tests' concurrency with torchrun ranks beyond the non-main credential skip.

VERDICT: APPROVE WITH EDITS (P3-1 applied; P3-2 to P3-4 recorded; no blocking issue)
