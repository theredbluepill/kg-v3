---
type: "Reference"
title: "v3 launchers fail fast without W&B credentials"
description: "run_ppo now logs to W&B project kg-v3 by default, grouped by a v3 experiment id. Rank 0 fails fast before config load when online mode has no WANDB_API_KEY and no api.wandb.ai netrc password. Offline or debug telemetry takes an explicit flag, prints a loud outage banner, and is written with source commit and config hash into the run's attempts.jsonl receipt. The shared owl.train.logging path extends Isaiah's loggers. CPU tests with W&B doubles only; the BC trainer has not adopted it yet."
tags: ["kaggriculture-v3", "adaptation", "telemetry"]
status: "verified-scoped"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-09-29"}
sources: [{"resource": "user-directive:2026-09-29:make-sure-all-v3-experiments-wired-to-wandb"}, {"resource": "repository:python/owl/train/logging.py"}, {"resource": "repository:scripts/run_ppo.py"}, {"resource": "repository:tests/owl/train/test_logging.py"}, {"resource": "repository:tests/scripts/test_run_ppo.py"}, {"resource": "repository:README.md"}, {"resource": "repository:ops/rebuild-2026-09-29/wandb-audit.md"}, {"resource": "repository:ops/rebuild-2026-09-29/wandb-2026-09-29/prepare.log"}]
---

# v3 launchers fail fast without W&B credentials

**Adaptation.** The owner's request: "make sure all v3 experiments wired to w&b." Before this change, Isaiah's `run_ppo` logged to project `orbit-wars` with no v3 identity. It found a missing key only inside `wandb.init`, after config, env and model setup. Its `--log-mode debug` outage was silent. The BC trainer's `--wandb-mode offline` was just as silent, and its one real run was never reported as offline. The audit and per-entry-point table are in `ops/rebuild-2026-09-29/wandb-audit.md`.

## What changed (branch `kg/rebuild-wandb`)

- **`python/owl/train/logging.py`** is extended in place. Isaiah's `LogMode`, `MetricLogger`, `DebugLogger`, `WandbLogger`, `create_logger` and `close(exit_code=...)` are kept.
  - **Project.** `WANDB_PROJECT = "kg-v3"` for every run launched from this repository, Orbit or Kaggriculture.
  - **Mode and gate.** `WandbMode` (online/offline) and `LogMode` resolve to a `TelemetryMode` (`wandb-online`, `wandb-offline`, `disabled`).
    - `check_telemetry` is the startup gate. Online requires `wandb_credential_source`, which mirrors wandb 0.26.1's lookup: a non-empty `WANDB_API_KEY`, else a password in the `NETRC` (default `~/.netrc`) entry for the `WANDB_BASE_URL` host (default `api.wandb.ai`). Otherwise it raises `MissingWandbCredentialsError`, whose message names the pod workflow, `WANDB_API_KEY`, `--wandb-mode offline` and `--log-mode debug`.
    - A malformed netrc raises without quoting the file.
    - A disagreeing `WANDB_MODE` is rejected.
    - Offline and disabled print a `W&B TELEMETRY OUTAGE` banner to stderr.
  - **Identity and receipt.** `RunIdentity`, `plan_attempt`, `record_attempt` and `read_attempts` keep one strict-schema JSON record per launch or resume in `<run_dir>/attempts.jsonl`. The record holds the attempt, experiment id (explicit or the run directory name; resumes keep it), job type, source commit (`git HEAD`, `-dirty` for tracked edits, or `--source-commit`) with all earlier ones, config SHA-256 (canonical JSON), `telemetry_mode`, W&B project, entity, run ID and URL, start env steps, and start time.
  - **W&B run shape.** The W&B run is grouped by experiment id, typed by job, and tagged `kaggriculture-v3`, job and game. It stores `v3.experiment_id` and `v3.job_type` in its config, which stays constant on resume, and the `v3/*` attempt fields in its summary.
  - **Reuse.** `create_metric_logger` takes a launcher's own config and game, so BC and future probes can reuse the same path.
- **`scripts/run_ppo.py`** stays the one trainer.
  - It adds `--wandb-mode`, `--experiment-id` (fresh launches only) and `--source-commit`.
  - The gate is the first step inside the distributed session, on rank 0, before config load. Non-main ranks only resolve the mode.
  - Rank 0 plans the attempt once the run directory exists, before env or model setup. That rejects a resume without receipts early. It records the attempt immediately after the logger opens, inside the failure-closing logger session, and prints the receipt path again on an outage.
  - `_validate_args` rejects offline mode combined with debug logging, and an `--experiment-id` on resume.
- **`README.md`** documents the gate, the flags and the receipt.
- **Pod procedure.** [[../workflows/install-the-wandb-credential-before-any-pod-launch|Install the W&B credential before any pod launch]].

## Verification (this version, CPU)

- **Unit tests with W&B test doubles.**
  - `tests/owl/train/test_logging.py` covers the credential sources and their failures, including a malformed file whose error omits the file text. It also covers every telemetry mode, banner and conflict, fresh and resumed attempts with experiment-id rules and strict receipts, a receipt that contradicts its logger, config hashing, and the `kg-v3` init arguments for online, offline and a BC-style config.
  - `tests/scripts/test_run_ppo.py` drives `main`:
    - online without credentials fails with `MissingWandbCredentialsError` before the config-override log, and creates no run directory;
    - offline without credentials passes the gate, prints the banner and stops at the existing not-wired error;
    - the session writes one receipt per telemetry mode, with the outage line only for outages;
    - a main rank without an identity is rejected.
- **Full preparation.** `CARGO_BUILD_JOBS=2 OMP_NUM_THREADS=2 uvx --from rust-just just prepare` exited 0 (`ops/rebuild-2026-09-29/wandb-2026-09-29/prepare.log`): ruff, format, mypy (64 files), docs-lint, Rust 254 passed and 4 ignored, engine 69 passed, Python 1,718 passed and 11 skipped, docs-fresh.
- **Codex.** Independent verification in `ops/rebuild-2026-09-29/codex/verify-wandb-r*.md`.

## Limits and gaps

- **No live run.** No live W&B run was created, and no pod or GPU check ran. The gate proves that a key is present, not that it is valid; the pod workflow's `wandb.Api()` check covers that.
- **Kaggriculture can't run yet.** `run_ppo` still stops at its explicit not-wired error for Kaggriculture, so the Kaggriculture receipt path is exercised only by tests up to that stop and by the session tests.
- **BC lags.** `scripts/train_bc.py` (BC branches) has not adopted the gate or the receipt. It still accepts offline silently and fails late without a key. The follow-up is in the audit.
- **Other entry points.** The Orbit benchmark scripts and the frozen, checksummed ops probes are unchanged and log locally. A future Kaggriculture evaluation harness must use the contract.
- **Old runs.** A resume needs `attempts.jsonl`, so run directories created before this change cannot resume under the new code. No v3 PPO run exists yet.
