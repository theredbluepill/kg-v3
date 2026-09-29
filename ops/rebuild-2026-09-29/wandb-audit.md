# W&B wiring audit — 2026-09-29

**Trigger.** The owner asked why BC training had no W&B report, and said "make sure all v3 experiments wired to w&b." The A100 BC run (`kg/rebuild-bc-now`, run statement `run-statements/bc-a100.md` at `f0b7a38`) was launched with `--wandb-mode offline` because the pod had no W&B key. The run statement says so, but nothing in the run's receipts or its W&B run records it, and nobody raised it with the owner.

**Scope.** This audit covers every experiment entry point on integration `kg/isaiah-gap-closure` at `faed717`. It also covers `scripts/train_bc.py` and `python/owl/train/bc.py` on `kg/rebuild-bc-trainer` at `b626f24`, read-only with `git show`; the `kg/rebuild-bc-now` diff does not touch W&B. The fix lives on `kg/rebuild-wandb` and changes integration code only. After this branch merged the integration tip `994818b`, the new scripts there (`check_opponent_import.py`, `kaggriculture_env_reference_policy.py`, `record_kaggriculture_env_reference.py`) were checked: they are import-check and fixture-recording tools with no W&B use, not experiments, so `run_ppo` remains the only v3 launcher on integration.

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
- **Verification status.** Round 2 (`verify-wandb-r2-prompt.md`) stopped at a Codex usage limit (reset shown as 2026-10-06), and the prescribed `codex exec resume` hit the same limit. Before stopping, it had killed and restored all 13 of its mutations on the gate: project, key sources, netrc, host, missing key, redaction, banner, online quiet and `WANDB_MODE` (`ops/rebuild-2026-09-29/codex/verify-wandb-r2-attempt1-partial-mutations.json`). It returned no verdict on the r1 findings. Claude then ran 20 mutations of its own on the remaining and new guards (not an independent check; `ops/rebuild-2026-09-29/wandb-2026-09-29/claude-mutations.json`). They cover the resume id, receipt existence, job type, id regex, attempt order, field types, bool attempt, constant identity, source history, logger mismatch, gate-before-config, outage line, W&B group, debug consistency, URL userinfo and scheme, the netrc export's default fallback and unsafe tokens, and the export script's terminal and missing-entry handling. All 20 were killed and restored.
- **After `claude-verify-wandb-r1` (Claude subagent standing in for Codex, REQUEST CHANGES).** In either W&B mode the gate now rejects a set-but-empty or malformed `WANDB_BASE_URL` and a blank or padded `WANDB_API_KEY`, which wandb 0.26.1 rejects inside `wandb.init(mode="offline")`. `--source-commit` is accepted only where git metadata is missing and rejected when it disagrees with git. `started_at` must be a timezone-aware ISO time. `WandbLogger` rejects a run that W&B started in a mode other than the requested one. `main`-level tests cover fresh, resumed and receipt-less launches. The branch merged the integration tip `994818b`. The scope wording now says "all v3 launchers on integration".
- **After `claude-verify-wandb-r2` (Claude subagent standing in for Codex, REQUEST CHANGES).** In either W&B mode the gate also runs the installed wandb's own `wandb.Settings(base_url=...)` and `wandb.Settings(api_key=...)` validators, which `wandb.init` applies in every mode, and reports only wandb's error type because its messages quote the value (`5186d70`). `https://wandb.ai`, `http://api.wandb.ai`, `https://app.wandb.ai`, `https://ho st.example` and a host-less `https://` now fail before config load; tests cover both modes. Claude's own recheck killed N7 and the removal of either validator (`ops/rebuild-2026-09-29/wandb-2026-09-29/claude-verify-r2-fix-mutations.log`). `claude-verify-wandb-r3` (`ops/rebuild-2026-09-29/codex/claude-verify-wandb-r3.md`) then killed all 13 of its mutations on `5186d70` and returned APPROVE WITH EDITS for two P3 record findings, both applied: a set `WANDB_MODE` that differs from the flag is rejected by policy, not because `wandb.init` would reject it (wandb 0.26.1 silently lets the flag win and raises only on an empty value), and the Reference now cites `prepare-claude-r2-fix.log`.
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

**Does `train_bc` need the same fail-fast? Yes.** It is a v3 experiment that trains on a GPU pod, and it is the one that produced this incident. `kg/rebuild-wandb` wires all v3 launchers on integration, which today means `run_ppo` alone; `train_bc` is not on integration. Its adoption is the step that runs right after the BC landing in this same workflow (BC code is not edited on this branch):

1. Call `owl.train.logging.check_telemetry(args.log_mode, WandbMode(args.wandb_mode), environ=os.environ, home=Path.home())` on rank 0 before `load_bc_configs`/`load_bc_dataset`.
2. Write the returned `telemetry_mode` into each `bc_attempts.jsonl` record and the W&B summary. Either extend `_start_attempt` or adopt `plan_attempt`/`record_attempt` alongside its dataset fields.
3. Replace `BCWandbLogger` with `create_metric_logger(run_dir, config=_logger_config(...), game="kaggriculture", identity=...)`, with job type `bc` and an experiment id as the group. This also removes the `bc-bc-` name.
4. Add tests for each guard, as `tests/owl/train/test_logging.py` does.

**Merge compatibility.** The BC branch imports `DebugLogger`, `LogMode` and `MetricLogger` from `owl.train.logging`, and all three still exist. `MetricLogger` gains `wandb_run_facts()`, which raises `NotImplementedError` only when called, and only `record_attempt` calls it. BC's `BCWandbLogger` and `_NoopLogger` therefore keep working unchanged until the BC lane adopts the contract. `create_logger`'s new keyword-only `identity` does not affect BC, which calls its own `create_bc_logger`.

## Limits

- Everything here runs on CPU with W&B test doubles. No live W&B run was created by this change. The only network call was the read-only `wandb.Api()` check above, which used this Mac's `~/.netrc` and printed no credential.
- The online-mode credential check mirrors wandb 0.26.1's lookup (`wandb/sdk/lib/wbauth/wbnetrc.py`: `NETRC`, then `~/.netrc`, host from the base URL). It proves a key is present, not that it is valid. That is why the pod workflow requires `wandb.Api()` before any launch.
- Offline resume (`resume="must"` with `mode="offline"`) is unchanged Isaiah behavior. wandb 0.26.1 ignores `resume` offline (`wandb/sdk/wandb_init.py`, probed by `claude-verify-wandb-r2`): it warns and starts a new offline segment with the same run ID. Syncing several same-ID segments was not exercised.
