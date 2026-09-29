# Independent verification: W&B wiring for v3 launchers (verify-wandb r1)

You are an **independent verifier**. You did not write this change. Verify it; do not extend it.

- Worktree: `/Users/poonszesen/kg-v3-wandb`, branch `kg/rebuild-wandb`, based on integration `kg/isaiah-gap-closure` at `faed717`. Review the commit(s) `faed717..HEAD`.
- **Leave no tracked modification.** When you finish, `git status --short` in the worktree must show nothing tracked as modified; untracked scratch goes under `/tmp` or `/private/tmp`, not in the repo. Do not commit. Do not push.
- **Mutations.** For each new guard listed below, apply at least one mutation on a scratch copy, or edit in place and then restore byte-for-byte with `git checkout -- <file>` / `git diff --exit-code`. Run the relevant tests, and report which tests failed (killed) or passed (survived). Restore every mutation afterwards and prove the restoration.
- Checks: `OMP_NUM_THREADS=2 uv run pytest tests/owl/train/test_logging.py tests/scripts/test_run_ppo.py -m "not slow" -q`. Also `OMP_NUM_THREADS=2 uvx --from rust-just just py-prepare` (the change is Python and docs only). Report the pass/skip/fail counts. No training, no GPU. Tiny CPU checks only: <1 GB, <2 min each, `OMP_NUM_THREADS=2`.
- **Never read, print or copy any real credential.** Do not open `~/.netrc`, and do not print `WANDB_API_KEY`. Do not make network calls to W&B; tests must stay hermetic. Use synthetic netrc files in temp dirs.

## Owner request and context

The owner said: "why BC training has nop W&B report? make sure all v3 experiments wired to w&b." The A100 BC run was launched with `--wandb-mode offline` because the pod had no W&B key, and nothing surfaced that. CLAUDE.md/AGENTS.md require W&B telemetry with v3 identifiers, and require telemetry outages to stay visible. `scripts/run_ppo.py` must remain the one canonical trainer, extended on its shared path, with Isaiah's logging semantics kept (extend, don't fork). The policy is stateless and observation-only: nothing here may feed opponent identity or history into the model. BC code (`kg/rebuild-bc-*` branches) must not be edited; its findings go in the audit note only.

## What to verify

1. **Audit** (`ops/rebuild-2026-09-29/wandb-audit.md`). Check the entry-point table against the code at `faed717` and at `HEAD`: `scripts/run_ppo.py`, `scripts/benchmark_*.py`, `scripts/slurm/*`, and the ops probe scripts under `ops/rebuild-2026-09-29/{gpu-checks,value-gap,model-sps-ceiling,gemm-limits,aten-gemm-ab,flash-attn-setup}-2026-09-29/`. Is any v3 experiment launcher missing? Check the BC findings read-only with `git show kg/rebuild-bc-trainer:scripts/train_bc.py` and `git show kg/rebuild-bc-trainer:python/owl/train/bc.py`: project, name, group, default mode, silent offline mode, and the late failure without credentials (ordering in `main`). Also check the merge-compatibility claim (BC imports and `MetricLogger` subclasses).
2. **Guards in `python/owl/train/logging.py` and `scripts/run_ppo.py`.** Mutate each at least once:
   - G1 project is `kg-v3` for every run;
   - G2 the online credential check (`wandb_credential_source` / `require_wandb_credentials`): env key, blank key, netrc host and password, `NETRC` and `WANDB_BASE_URL`;
   - G3 malformed netrc never quotes file content;
   - G4 `check_telemetry` prints the outage banner for offline and debug, and not for online;
   - G5 `WANDB_MODE` disagreement is rejected;
   - G6 debug + offline is rejected (`telemetry_mode`, `_validate_args`);
   - G7 resume + `--experiment-id` is rejected;
   - G8 `plan_attempt`: a fresh run with an existing receipt, a resume without receipts, and a changed experiment id are rejected; the experiment-id regex;
   - G9 `read_attempts` strict schema and attempt order;
   - G10 `record_attempt` rejects a logger/telemetry mismatch, and writes `telemetry_mode`;
   - G11 `run_ppo.main` runs the telemetry gate before config load and run-dir creation;
   - G12 `_run_training_session` records the attempt and prints the outage line;
   - G13 W&B init arguments (group, job_type, tags, v3 config, v3 summary, mode);
   - G14 `create_logger` log-mode/telemetry consistency.
3. **Semantics kept.** Isaiah's `LogMode`, `MetricLogger.close(exit_code)`, `_logger_session`, resume `wandb_run_id` handling, and the worker-rank no-logger path are unchanged in behavior. The credential check runs only on rank 0. Non-main ranks do not read git or credentials. Check that no secret can reach stdout, stderr, receipts or exception messages.
4. **Docs and cookbook.**
   - `README.md` matches the code.
   - `cookbook/workflows/install-the-wandb-credential-before-any-pod-launch.md`: does the `awk` extract keep only the `api.wandb.ai` entry for single-line and multi-line netrc entries, with `default` entries dropped? Check it on synthetic files only. Does the ssh install step refuse to overwrite and set 600? Is the key ever on argv or printed?
   - `cookbook/references/v3-launchers-fail-fast-without-wandb-credentials.md`, `cookbook/references/index.md`, `cookbook/workflows/index.md` and the prepended `cookbook/log.md` entry: claims are scoped to actual checks, the frontmatter is complete (first tag `kaggriculture-v3`), and every `repository:` source exists.
   - Is `ops/rebuild-2026-09-29/wandb-2026-09-29/prepare.log` a green `just prepare` for this change?
5. Anything else that is wrong: bugs, edge cases (for example a netrc `default` entry, a `WANDB_BASE_URL` without a scheme, an empty `WANDB_MODE`, resume from an offline run), Python 3.11 syntax (`scripts/check_python_311_syntax.py`), or mypy.

## Report

Write your full report to the `-o` file. Include: the commands you ran with their pass/skip/fail counts; a mutation table (guard, mutation, killing tests or SURVIVED, restored yes/no); the audit check; and findings with severity (P1 blocking, P2 should-fix, P3 edit), `file:line` and a concrete fix. End with exactly one line:

`VERDICT: APPROVE` or `VERDICT: APPROVE WITH EDITS` or `VERDICT: REQUEST CHANGES`
