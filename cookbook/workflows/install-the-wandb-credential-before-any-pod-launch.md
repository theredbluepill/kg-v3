---
type: "Workflow"
title: "Install the W&B credential before any pod launch"
description: "Pod setup step for every v3 launch: copy only the operator's api.wandb.ai netrc entry to the pod through stdin, chmod 600, never print or commit it, and check wandb.Api() before any launch. Launchers then default to online W&B in kg-v3 and fail fast without a key. Offline or disabled telemetry takes an explicit flag, a loud banner, an attempts.jsonl record and a report to the owner. The pipeline was simulated locally on synthetic files; it has not yet run on a pod."
tags: ["kaggriculture-v3", "workflows", "telemetry", "pods"]
status: "verified-scoped"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-09-29"}
sources: [{"resource": "user-directive:2026-09-29:make-sure-all-v3-experiments-wired-to-wandb"}, {"resource": "repository:ops/rebuild-2026-09-29/wandb-audit.md"}, {"resource": "repository:python/owl/train/logging.py"}, {"resource": "repository:scripts/run_ppo.py"}, {"resource": "repository:tests/owl/train/test_logging.py"}, {"resource": "repository:tests/scripts/test_run_ppo.py"}, {"resource": "repository:README.md"}, {"resource": "repository:ops/rebuild-2026-09-29/plan.md"}]
---

# Install the W&B credential before any pod launch

**Why.** The owner asked "why BC training has nop W&B report? make sure all v3 experiments wired to w&b." The A100 BC run went out with `--wandb-mode offline` because the pod had no key. Only its run statement noted it; its receipts and its W&B run did not (`ops/rebuild-2026-09-29/wandb-audit.md`). The step below is part of pod setup, alongside the pod rules in `ops/rebuild-2026-09-29/plan.md` (live price and state, idle check, run statement). Do it on every new or rebuilt pod, before its first launch.

## 1. Copy only the api.wandb.ai entry, through stdin

Run this on the operator's Mac. `<pod>` is the SSH target from the live pod read; never write it into tracked files.

```bash
awk '$1=="machine"||$1=="default"{keep=($1=="machine"&&$2=="api.wandb.ai")} keep' ~/.netrc \
  | ssh <pod> 'set -eu; umask 077
      if [ -e "$HOME/.netrc" ]; then
        echo "pod ~/.netrc exists; merge the api.wandb.ai entry by hand" >&2; exit 1
      fi
      cat > "$HOME/.netrc"; chmod 600 "$HOME/.netrc"'
```

- The `awk` filter keeps the `machine api.wandb.ai` entry and its continuation lines, and drops every other machine and any `default` entry. To check the extract without printing it, pipe it to `grep -c '^machine api.wandb.ai'` instead of `ssh`; the count must be 1.
- The key travels only on stdin. It must never be a command-line argument (visible in `ps`), an environment variable written to a file, part of a run statement or transcript, or committed. Do not `cat`, `echo` or `grep -v` the file on either host.
- If the pod already has a `~/.netrc`, the step refuses to overwrite it. Merge only the api.wandb.ai entry by hand, then run `chmod 600`.
- Use the netrc, not `WANDB_API_KEY` in shell profiles or `docker run -e`, so the key never enters logs that capture the environment.

## 2. Check before any launch

```bash
ssh <pod> 'stat -c %a ~/.netrc; env | grep -c "^WANDB_MODE=" || true
  cd /workspace/kg-v3-rebuild && .venv/bin/python -c \
  "import wandb; print(\"wandb.Api ok, entity:\", wandb.Api(timeout=30).default_entity)"'
```

Expect `600`, `0` (no `WANDB_MODE` override), and the entity name. `default_entity` queries the server, so a revoked or mistyped key fails here, not an hour into a run. The launcher's own gate proves only that a key is present. Record the check's exit status and the entity in the run statement, never the key.

## 3. Launch online (the default)

`scripts/run_ppo.py` defaults to `--log-mode wandb --wandb-mode online` in project `kg-v3`. Rank 0 checks credentials before loading the config. Without `WANDB_API_KEY` and an `api.wandb.ai` netrc password, it raises `MissingWandbCredentialsError`, whose message names this workflow. Pass `--experiment-id` for a stable W&B group (the default is the run directory name). After launch:

- Read the run's `attempts.jsonl`; it should show `telemetry_mode: "wandb-online"` and a `wandb_url`.
- Confirm with `wandb.Api().run("<entity>/kg-v3/<run id>").state == "running"` that data arrives live.

## 4. An outage takes an explicit flag and a report

Run without live telemetry only when the owner accepts it or the credential cannot be installed. Pass `--wandb-mode offline` (metrics are kept for `wandb sync`) or `--log-mode debug` (stdout only). The launcher then:

- prints a `W&B TELEMETRY OUTAGE` banner at startup;
- records `telemetry_mode` in `attempts.jsonl` and in the W&B summary field `v3/telemetry_mode`;
- prints the receipt path.

The operator must also:

- state the outage in the run statement;
- tell the owner in the same message that reports the launch;
- once synced (`wandb sync <run_dir>/wandb/offline-run-*`), record the sync and the W&B URL in the run's evidence.

A sync does not change the recorded mode.

## Launcher contract

Every v3 launcher that runs on a pod uses the same shared path as `run_ppo`, from `owl.train.logging`:

1. `check_telemetry` at startup, on rank 0, before loading data or models;
2. `plan_attempt` once the run directory exists;
3. `create_metric_logger` (with its own `job_type`, `game` and config) or `create_logger` for PPO;
4. `record_attempt`.

`scripts/train_bc.py` on the BC branches predates this contract: it has no startup gate and records no telemetry mode. The needed follow-up is in the audit. A probe that deliberately skips W&B must say so in its run statement and keep its local receipts. The completed, checksummed ops probes were not changed.

## Verification

- Checks of this version on `kg/rebuild-wandb` (CPU, W&B test doubles):
  - `tests/owl/train/test_logging.py` covers the credential lookup (env key, blank key, netrc host and password, `NETRC` and `WANDB_BASE_URL`, and a malformed file whose error does not quote it). It also covers the online, offline and debug gates, the loud banner, the `WANDB_MODE` conflict, attempt planning and receipts, config hashing, and the `kg-v3` init arguments.
  - `tests/scripts/test_run_ppo.py` covers the fail-fast before config load, the offline banner without credentials, the per-mode receipt and the argument rules.
  - Counts are in the Reference [[../references/v3-launchers-fail-fast-without-wandb-credentials|v3 launchers fail fast without W&B credentials]].
- The step 1 pipeline was simulated on this Mac with a synthetic four-entry netrc, with `sh -c` and a temporary `HOME` standing in for `ssh <pod>`:
  - the extract kept exactly the api.wandb.ai entry (count 1; no other secret present);
  - the installed file had mode 600;
  - a second install refused to overwrite;
  - `require_wandb_credentials` accepted the installed file.
- On this Mac, step 2's Python check printed `wandb.Api ok, entity: spoon`. A read-only `wandb.Api()` call from this Mac's netrc found the synced BC run `spoon/kg-v3/kvl4rfda`. It is `finished`, with no field recording its offline launch.

## Limits

- Steps 1 and 2 have not yet run on a pod. The first pod that uses them should confirm `stat -c` (GNU) and the venv path, and record the result here.
- Neither the presence check nor `default_entity` proves that metrics arrive during a run; step 3's live read does. Multi-node launches would need the key on the rank-0 node only; none is planned.
