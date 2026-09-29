---
type: "Reference"
title: "PPO runs publish Kaggriculture telemetry to the v3 W&B project"
description: "run_ppo's W&B logger sends Kaggriculture PPO runs to project kg-v3 (job type and group ppo, tags kaggriculture-v3 and ppo), where it had sent every run to Isaiah's orbit-wars; Orbit runs keep orbit-wars. --wandb-mode offline keeps a syncable run in the run directory and prints the outage. Tests with a fake wandb module and two killed mutations are the only checks; no live W&B call was made. The BC trainer (on its own branch) already uses kg-v3, and its A100 run has no W&B report because it ran offline for lack of a pod key."
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
  - resource: "reference-branch:kg/reference-2026-09-29/python/owl/train/logging.py"
  - resource: "bc-now-branch:f0b7a38:python/owl/train/bc.py"
  - resource: "bc-now-branch:f0b7a38:scripts/train_bc.py"
  - resource: "bc-now-branch:f0b7a38:ops/rebuild-2026-09-29/run-statements/bc-a100.md"
---

# PPO runs publish Kaggriculture telemetry to the v3 W&B project

The owner asked why BC training had no W&B report and to "make sure all v3 experiments wired to w&b". The [[../decisions/evaluation-preserves-generality-and-evidence|evaluation and telemetry Decision]] requires W&B telemetry under v3 identifiers, with outages kept visible. A search found the historical [[shared-ppo-adapts-game-batches-without-a-second-loop|shared PPO Reference]], whose reference-branch `logging.py` chose `kg-v3` for Kaggriculture. The rebuilt tree had lost that choice: Isaiah's `WandbLogger` sent every run to `orbit-wars`. The [[failed-training-reports-status-before-distributed-cleanup|failure-status Reference]] covers the logger's exit codes and is unchanged.

## Adaptation inventory

- `python/owl/train/logging.py`: `wandb_init_identity(cfg)` returns `project="kg-v3"`, `job_type="ppo"`, `group="ppo"` and tags `kaggriculture-v3`, `ppo` when `cfg.model` is a `KaggricultureTransformerConfig`. Otherwise it returns `project="orbit-wars"` for Isaiah's retained Orbit runs. `WandbLogger` and `create_logger` take a `mode` (`WandbMode`, `"online"` or `"offline"`) and pass it to `wandb.init`. The BC trainer's `BCWandbLogger` uses the same project and tag scheme.
- `scripts/run_ppo.py`: `--wandb-mode {online,offline}` (default online, as in `scripts/train_bc.py`) reaches `create_logger` through `_run_training_session`. With `--log-mode debug` a non-default mode is rejected. An offline run prints a line saying that telemetry stays under `<run_dir>/wandb` until `wandb sync`, and that W&B ignores `resume` offline and starts a local run with the saved id. That is the installed SDK's warning path in `wandb/sdk/wandb_init.py`; it was read, not exercised.
- `README.md` documents the project, labels and offline mode.

## Why the BC run has no W&B report

This was a read-only inspection of `kg/rebuild-bc-now` at `f0b7a38`; that branch is run separately and was not changed. BC is wired: `python/owl/train/bc.py` opens `BCWandbLogger` under project `kg-v3` (`job_type="bc"`) with an online/offline mode. The A100 run statement (`ops/rebuild-2026-09-29/run-statements/bc-a100.md`) chose `--wandb-mode offline` because the pod had no W&B key. The run's telemetry therefore sits in its run directory and reaches W&B only after `wandb sync` from a machine with the owner's key, or after a key is provided on the pod. Supplying the credential is the owner's call.

## Checks (this version, CPU only)

- `tests/owl/train/test_logging.py`: the Kaggriculture and Orbit identities, and a fake `wandb` module that records `wandb.init`'s project, tags, mode (online and offline), resume id, directory and config.
- `tests/scripts/test_run_ppo.py`: the parser reads `--wandb-mode` (online by default); `--wandb-mode offline` with `--log-mode debug` is rejected; `_run_training_session` forwards the offline mode and prints the notice.
- Mutations (`ops/rebuild-2026-09-29/teacher-configs-4.4/mutations.log`), each restored byte-for-byte: forcing `orbit-wars` for Kaggriculture failed 3 tests, and dropping `mode=` from `wandb.init` failed 2. `just py-prepare`: 1,712 passed, 11 skipped (`py-prepare.log`).

## Limits

- No live W&B call was made; authentication, upload, `wandb sync` of an offline run and offline resume are unverified. A keyless online launch fails at `wandb.init`, which is the intended fail-fast.
- Only `run_ppo.py` and (on its branch) the BC trainer log to W&B. Evaluation and benchmark scripts outside the trainer have no W&B logger; wiring them is open.
- Orbit runs keep Isaiah's `orbit-wars` project, as the reference branch did.
