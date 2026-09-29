---
type: "Reference"
title: "PPO runs publish Kaggriculture telemetry to the v3 W&B project"
description: "run_ppo's W&B logger sends Kaggriculture PPO runs to project kg-v3 as ppo-<run dir> (job type and group ppo, tags kaggriculture-v3 and ppo), where it had sent every run to Isaiah's orbit-wars; Orbit runs keep orbit-wars and their online init arguments. --wandb-mode offline keeps a syncable run in the run directory and prints the outage; it is rejected for resume launches and with --log-mode debug. Tasks 4.4 and 3.1 built this in parallel; the 3.1/3.5 merge keeps one logger. Tests with a fake wandb module, killed mutations and Task 3.5's one unsynced offline CLI run are the only checks; no live W&B call was made. The BC trainer (on its own branch) already uses kg-v3; its A100 run statement plans offline mode for lack of a pod key, so that run's telemetry would reach W&B only through wandb sync (no launch receipt or offline artifact inspected)."
tags: ["kaggriculture-v3", "training", "adaptation", "diagnostics"]
status: "verified-scoped"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-09-29"}
sources:
  - resource: "user-directive:2026-09-29:wire-all-v3-experiments-to-wandb"
  - resource: "repository:python/owl/train/logging.py"
  - resource: "repository:scripts/run_ppo.py"
  - resource: "repository:tests/owl/train/test_logging.py"
  - resource: "repository:tests/scripts/test_run_ppo.py"
  - resource: "repository:README.md"
  - resource: "repository:ops/rebuild-2026-09-29/teacher-configs-4.4/mutations.log"
  - resource: "repository:ops/rebuild-2026-09-29/teacher-configs-4.4/py-prepare.log"
  - resource: "repository:ops/rebuild-2026-09-29/merge-3-1-3-5/prepare.log"
  - resource: "repository:ops/rebuild-2026-09-29/3.5/cli-wandb-history.json"
  - resource: "reference-branch:kg/reference-2026-09-29/python/owl/train/logging.py"
  - resource: "bc-now-branch:f0b7a38:python/owl/train/bc.py"
  - resource: "bc-now-branch:f0b7a38:scripts/train_bc.py"
  - resource: "bc-now-branch:f0b7a38:ops/rebuild-2026-09-29/run-statements/bc-a100.md"
---

# PPO runs publish Kaggriculture telemetry to the v3 W&B project

The owner asked why BC training had no W&B report and to "make sure all v3 experiments wired to w&b". The [[../decisions/evaluation-preserves-generality-and-evidence|evaluation and telemetry Decision]] requires W&B telemetry under v3 identifiers, with outages kept visible. A search found the historical [[shared-ppo-adapts-game-batches-without-a-second-loop|shared PPO Reference]], whose reference-branch `logging.py` chose `kg-v3` for Kaggriculture. The rebuilt tree had lost that choice: Isaiah's `WandbLogger` sent every run to `orbit-wars`. The [[failed-training-reports-status-before-distributed-cleanup|failure-status Reference]] covers the logger's exit codes and is unchanged.

## Adaptation inventory

- `python/owl/train/logging.py`: `wandb_init_identity(cfg)` returns `project="kg-v3"`, `job_type="ppo"`, `group="ppo"` and tags `kaggriculture-v3`, `ppo` when `cfg.env` is a `KaggricultureEnvConfig` (the config validator pairs it with the Kaggriculture model). Otherwise it returns `project="orbit-wars"` for Isaiah's retained Orbit runs. `WandbLogger` and `create_logger` take `wandb_mode` (`WandbMode`, `"online"` or `"offline"`). A Kaggriculture run is named `ppo-<run dir>` and always passes its mode to `wandb.init`; an Orbit run keeps Isaiah's online init arguments and adds `mode` only offline. `WandbLogger` rejects an offline resume before `wandb.init`, so an offline run never takes a saved run's id. The BC trainer's `BCWandbLogger` uses the same project and tag scheme.
- `scripts/run_ppo.py`: `--wandb-mode {online,offline}` (default online, as in `scripts/train_bc.py`) reaches `create_logger` through `_run_training_session`. `_validate_args` rejects offline mode for a resume launch and with `--log-mode debug`. An offline run prints a line saying that telemetry stays under `<run_dir>/wandb` until `wandb sync`.
- **Merge reconciliation (2026-09-30).** Task 4.4 (landed first) and the Task 3.1 remainder each wired this logger. Task 4.4 allowed an offline resume and printed that W&B then ignores `resume` and starts a local run with the saved id (the installed SDK's warning path in `wandb/sdk/wandb_init.py`, read, not exercised). Task 3.1 rejected it, since that offline run could not continue the saved run's telemetry. The merge keeps Task 3.1's rejection (fail fast, per the repository's error-handling rule), its run name and Orbit-unchanged arguments, and Task 4.4's helper, mode tuple and printed outage line.
- `README.md` documents the project, labels and offline mode.

## Why the BC run has no W&B report

This was a read-only inspection of `kg/rebuild-bc-now` at `f0b7a38`; that branch is run separately and was not changed. BC is wired: `python/owl/train/bc.py` opens `BCWandbLogger` under project `kg-v3` (`job_type="bc"`) with an online/offline mode. The A100 run statement (`ops/rebuild-2026-09-29/run-statements/bc-a100.md`) plans `--wandb-mode offline` because the pod had no W&B key. That statement records the intended launch configuration; no launch receipt or offline-run artifact was inspected here. If the run was launched as stated, its telemetry sits in its run directory and reaches W&B only through `wandb sync` with the owner's key. Providing a key on the pod does not upload existing offline telemetry on its own: `wandb sync` is still required, and only later online launches publish directly. Supplying the credential is the owner's call.

## Checks (this version, CPU only)

- `tests/owl/train/test_logging.py`: the Kaggriculture and Orbit identities, and a fake `wandb` module that records `wandb.init`'s project, name, tags, mode (online and offline), resume id (online only), directory and config; Orbit's online arguments unchanged; an offline resume rejected before `wandb.init` for both games.
- `tests/scripts/test_run_ppo.py`: the parser reads `--wandb-mode` (online by default); `--wandb-mode offline` with `--log-mode debug` or a resume is rejected; `_run_training_session` forwards the offline mode and prints the notice; `main` forwards `--wandb-mode offline` from a real tiny launch.
- The merge's full `just prepare` (`ops/rebuild-2026-09-29/merge-3-1-3-5/prepare.log`) passes with both parents' logging and launch tests. Task 3.5's one CLI run (`--wandb-mode offline`) wrote an unsynced offline run in `kg-v3` (`3.5/cli-wandb-history.json`).
- Mutations (`ops/rebuild-2026-09-29/teacher-configs-4.4/mutations.log`), each restored byte-for-byte: forcing `orbit-wars` for Kaggriculture failed 3 tests, and dropping `mode=` from `wandb.init` failed 2. `just py-prepare`: 1,712 passed, 11 skipped (`py-prepare.log`).

## Limits

- No live W&B call was made; authentication, upload and `wandb sync` of an offline run are unverified. Offline resume is rejected, not supported. A keyless online launch fails at `wandb.init`, which is the intended fail-fast.
- Only `run_ppo.py` and (on its branch) the BC trainer log to W&B. Evaluation and benchmark scripts outside the trainer have no W&B logger; wiring them is open.
- Orbit runs keep Isaiah's `orbit-wars` project, as the reference branch did.
