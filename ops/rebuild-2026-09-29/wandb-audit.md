# W&B wiring audit — 2026-09-29

**Trigger.** The owner asked why BC training had no W&B report, and said "make sure all v3 experiments wired to w&b." The A100 BC run (`kg/rebuild-bc-now`, run statement `run-statements/bc-a100.md` at `f0b7a38`) was launched with `--wandb-mode offline` because the pod had no W&B key. The run statement says so, but nothing in the run's receipts or its W&B run records it, and nobody raised it with the owner.

**Scope.** This audit covers every experiment entry point on integration `kg/isaiah-gap-closure` at `faed717`. It also covers `scripts/train_bc.py` and `python/owl/train/bc.py` on `kg/rebuild-bc-trainer` at `b626f24`, read-only with `git show`; the `kg/rebuild-bc-now` diff does not touch W&B. The fix lives on `kg/rebuild-wandb` and changes integration code only.

## Entry points on integration (before the fix, `faed717`)

| Entry point | W&B project | Entity | Name / tags / group / config | v3 identifiers (experiment, attempt, source commit, config hash) | Default mode | Without credentials |
| --- | --- | --- | --- | --- | --- | --- |
| `scripts/run_ppo.py` (Isaiah's `--log-mode`, `python/owl/train/logging.py`) | `orbit-wars`, hard-coded, **wrong** | W&B default for the key (`WANDB_ENTITY` if set); not recorded | name = run dir; no tags, group or job type; config = `FullConfig` dump | none | `--log-mode wandb`, online | Not checked. `wandb.init` runs only after config load, run-dir creation, env, model and optimizer build, then prompts or fails late. `--log-mode debug` is silent: no warning and no record. |
| `scripts/benchmark_checkpoints.py` | no W&B | — | — | none; prints win rates | stdout | n/a. Orbit only: `require_orbit_env` refuses Kaggriculture, so this is not yet a v3 experiment. |
| `scripts/benchmark_envs.py` | no W&B | — | — | none | stdout | n/a. Orbit/Kaggle env step-throughput tool. |
| `scripts/slurm/launch-train.sbatch` | wraps `run_ppo` | — | — | — | `ORBIT_WARS_LOG_MODE=wandb` | Inside the allocated job, after sourcing `~/.config/orbit-wars/wandb.env` and before the container and trainer start, it exits when `WANDB_API_KEY` is empty. Submission itself is not gated. This is Isaiah's Slurm path; pods do not use it. |
| `scripts/slurm/launch_scaling_experiments.sh`, `launch_scaling_5090.sh`, `launch_truncation_experiments.sh` | submit `launch-train.sbatch` jobs | — | — | — | the sbatch's `wandb` default | The same in-job check per submitted job. |
| `scripts/slurm/launch-interactive.sh` | interactive `srun` shell; forwards `WANDB_API_KEY` into the container | — | — | — | whatever is run inside | It sources the env file; the trainer's own gate applies to anything launched inside. |
| `scripts/slurm/smoke_scaling.sh` | wraps `run_ppo --log-mode debug` | — | — | — | explicit debug | n/a |
| GPU checks bundle (`ops/.../gpu-checks-2026-09-29/scripts/{driver,c1..c4}.py`), value-gap probes (`ops/.../value-gap-2026-09-29/scripts/{driver,kg_gap,is_gap}.py`), SPS ceiling (`ops/.../model-sps-ceiling-2026-09-29/bench_model_sps.py`), GEMM limits (`ops/.../gemm-limits-2026-09-29/probe/driver.py`), ATEN A/B (`ops/.../aten-gemm-ab-2026-09-29/scripts/driver.py`), flash smoke (`ops/.../flash-attn-setup-2026-09-29/smoke_flash.py`) | no W&B | — | — | Local JSON/JSONL receipts, run statements and `MANIFEST.sha256` custody | local files | n/a. These are completed, frozen probes; their scripts are checksummed evidence and were not edited. |
| Data and fixture tools (`generate_reference_fixtures.py`, `kaggriculture_parity/*`, `kaggriculture_bc/select_replays.py`, observation oracle, `download_replays.py`) | no W&B | — | — | — | — | Not experiments. |

## Fix on `kg/rebuild-wandb` (integration code only)

- `python/owl/train/logging.py` is extended in place; Isaiah's `LogMode`, `MetricLogger`, `DebugLogger`, `WandbLogger`, `create_logger` and close semantics are kept.
  - The project is `kg-v3` for every run from this repository.
  - `--wandb-mode {online,offline}` (`WandbMode`) combines with `--log-mode` into `TelemetryMode`: `wandb-online`, `wandb-offline` or `disabled`.
  - `check_telemetry` is the startup gate. Online without `WANDB_API_KEY` and without a password in the `NETRC`/`~/.netrc` entry for the `WANDB_BASE_URL` host (default `api.wandb.ai`) raises `MissingWandbCredentialsError`. The error names the pod workflow, `WANDB_API_KEY`, `--wandb-mode offline` and `--log-mode debug`. A malformed netrc raises without quoting the file.
  - Offline or debug prints a `W&B TELEMETRY OUTAGE` banner to stderr.
  - A `WANDB_MODE` value that disagrees with the flag is rejected.
- **Run identity.** `plan_attempt` / `record_attempt` append one record per launch or resume to the run directory's `attempts.jsonl`: attempt, experiment id, job type, source commit and all earlier ones, config SHA-256, `telemetry_mode`, and the W&B project, entity, run ID and URL.
  - The W&B run gets group = experiment id, job type, tags `kaggriculture-v3`/job/game, `v3.*` config and `v3/*` summary fields.
  - `create_metric_logger` lets a non-PPO launcher reuse the same logger with its own config.
- `scripts/run_ppo.py` adds `--wandb-mode`, `--experiment-id` (fresh runs only) and `--source-commit`. It runs the gate as the first step inside the distributed session, on rank 0 and before the config loads. It plans the attempt once the run dir exists, records it right after the logger opens, and prints the receipt path again when there is an outage.
- **After Codex `verify-wandb-r1` (REQUEST CHANGES).**
  - The credential gate rejects a set-but-blank or padded `WANDB_API_KEY` even when a netrc exists, and any set `WANDB_MODE`, including an empty one, that differs from the flag.
  - `WANDB_BASE_URL` must be an http(s) URL with a host and no embedded credentials. It is validated before the key source is chosen, and never quoted.
  - `read_attempts` validates every field's type and value, the attempt order, a constant experiment id and job type, and the source-commit history. A resume also keeps the recorded job type.
  - The pod copy uses `scripts/export_wandb_netrc_entry.py` (netrc-module parse, `machine api.wandb.ai` only, no `default`, no unsafe tokens, refuses a terminal) instead of an `awk` line filter. The filter copied a second credential from a packed line and missed split tokens.
  - The install side rejects an empty stream.
  - Tests were added for each case, for receipt order and for a real `FullConfig.from_file` sentinel before the gate.
  - README and the workflow now say how to resume an offline run (sync it first, or continue offline).
- The benchmark scripts and the frozen ops probes are unchanged. Any future Kaggriculture evaluation harness or pod probe follows the contract in `cookbook/workflows/install-the-wandb-credential-before-any-pod-launch.md`, or its run statement declares the outage.

## BC trainer findings (`kg/rebuild-bc-trainer` `b626f24`, read-only; not edited)

| Aspect | Finding |
| --- | --- |
| Project | `kg-v3` (`BC_WANDB_PROJECT`), correct. |
| Entity | Not handled: the key's default entity. The synced run is `spoon/kg-v3/kvl4rfda`. |
| Name / group / tags | name `bc-{run_dir.name}`. Run dirs are already `bc-<timestamp>`, so the name reads `bc-bc-20260929-142216`. group = the constant `"bc"`, not an experiment id. tags `kaggriculture-v3`, `bc`; job type `bc`. |
| v3 identifiers | Good provenance in the W&B config and `bc_attempts.jsonl`: attempt, source commit and all earlier ones, dataset root and manifest SHA-256, parent-state SHA-256, world size. No experiment id and no config hash (the full BC and PPO configs are dumped instead). |
| Default mode | `--log-mode wandb`, `--wandb-mode online`. |
| Offline | `--wandb-mode offline` is accepted silently. No warning is printed. `bc_attempts.jsonl` has no telemetry field. The W&B run has no mode field: a read-only `wandb.Api()` check of `spoon/kg-v3/kvl4rfda` (state `finished`, job type `bc`, group `bc`, tags `bc`/`kaggriculture-v3`, source commit `f0b7a38`) found config keys `bc`/`method`/`ppo`/`provenance` and 26 summary fields, none recording offline mode or the later sync. |
| Without credentials, online | No startup check. `wandb.init` runs only after the compile-stack check, dataset load, run-dir creation, attempt record, model build, compile and optimizer creation, so a missing key fails late (or prompts). The exact wandb 0.26 no-tty behavior was not exercised here. The attempt record is written before the failure, so a failed attempt leaves a record that holds no telemetry information. |

**Does `train_bc` need the same fail-fast? Yes.** It is a v3 experiment that trains on a GPU pod, and it is the one that produced this incident. The follow-up for the BC lane (not done here, BC code is off-limits):

1. Call `owl.train.logging.check_telemetry(args.log_mode, WandbMode(args.wandb_mode), environ=os.environ, home=Path.home())` on rank 0 before `load_bc_configs`/`load_bc_dataset`.
2. Write the returned `telemetry_mode` into each `bc_attempts.jsonl` record and the W&B summary. Either extend `_start_attempt` or adopt `plan_attempt`/`record_attempt` alongside its dataset fields.
3. Replace `BCWandbLogger` with `create_metric_logger(run_dir, config=_logger_config(...), game="kaggriculture", identity=...)`, with job type `bc` and an experiment id as the group. This also removes the `bc-bc-` name.
4. Add tests for each guard, as `tests/owl/train/test_logging.py` does.

**Merge compatibility.** The BC branch imports `DebugLogger`, `LogMode` and `MetricLogger` from `owl.train.logging`, and all three still exist. `MetricLogger` gains `wandb_run_facts()`, which raises `NotImplementedError` only when called, and only `record_attempt` calls it. BC's `BCWandbLogger` and `_NoopLogger` therefore keep working unchanged until the BC lane adopts the contract. `create_logger`'s new keyword-only `identity` does not affect BC, which calls its own `create_bc_logger`.

## Limits

- Everything here runs on CPU with W&B test doubles. No live W&B run was created by this change. The only network call was the read-only `wandb.Api()` check above, which used this Mac's `~/.netrc` and printed no credential.
- The online-mode credential check mirrors wandb 0.26.1's lookup (`wandb/sdk/lib/wbauth/wbnetrc.py`: `NETRC`, then `~/.netrc`, host from the base URL). It proves a key is present, not that it is valid. That is why the pod workflow requires `wandb.Api()` before any launch.
- Offline resume (`resume="must"` with `mode="offline"`) is unchanged Isaiah behavior and was not exercised.
