Reviewer: independent Claude subagent (substitute for Codex during its usage limit; owner-approved). Not a Codex verdict.

# claude-verify-wandb-r1: W&B wiring for v3 launchers

## Scope

- Branch `kg/rebuild-wandb`, tip `06a50d8d10c6a62bb6559976b639eaf8c35a78b8`; diff `faed717...HEAD` (4 commits, 22 files). Code is unchanged since `b90286a`; `6a23239` and `06a50d8` touch only ops and cookbook files.
- Spec: the owner's "make sure all v3 experiments wired to w&b"; CLAUDE.md telemetry rules (kg-v3, v3 identifiers, outages visible, custody); the Codex r1 report `verify-wandb-r1.md`, with every finding claimed fixed in `b90286a`.
- Out of scope: `scripts/train_bc.py` and `python/owl/train/bc.py`, which are not on this branch or on integration. Per the task, the docs must say "all v3 launchers on integration" and name the train_bc follow-up that runs right after the BC landing.
- Scratch worktree `/tmp/cv-wandb-r1` (detached at HEAD). Fixtures copied from `/Users/poonszesen/kg-v3/tests/fixtures`. `owl.rs` built there with `CARGO_BUILD_JOBS=2 uv run maturin develop --release`. The worktree is now removed. `/Users/poonszesen/kg-v3-wandb` has 0 tracked modifications and HEAD is still `06a50d8`. The other scratch worktree `cv-wandb-r3` belongs to another session and was not touched.

## Checks (scratch, CPU, OMP_NUM_THREADS=2)

| Check | Result |
| --- | --- |
| Targeted: `pytest tests/owl/train/test_logging.py tests/scripts/test_run_ppo.py tests/scripts/test_export_wandb_netrc_entry.py -m "not slow"` | 156 passed, 1 skipped (native Kaggriculture env), 0.66 s |
| `ruff check --select I`, `ruff format --check`, `check_python_311_syntax.py`, `ruff check` | pass; 121 files already formatted |
| `mypy python/ scripts/` | no issues, 65 source files |
| `check_doc_freshness.py`; `pymarkdownlnt` over root `*.md` and the code/docs dirs | both exit 0 |
| Full `pytest tests -m "not slow"`, run as 4 shards | owl 832 passed / 3 skipped; scripts 226/1; kaggriculture 524/7; tools 159/0. **Total 1,741 passed, 11 skipped**, which matches the committed `prepare.log` for `b90286a` |
| Peak RSS per shard | 531 MB, 519 MB, **2,025 MB** (tests/kaggriculture), 411 MB. The kaggriculture shard exceeds the 1 GB guideline. That shard does not touch the W&B code, and it was not rerun. |
| Rust `just rs-prepare` | not rerun; the code diff has no Rust changes. The committed `prepare.log` reports Rust 254 passed / 4 ignored and the engine 69 passed on `b90286a`. |
| Cookbook `repository:` sources in both new notes | all exist |
| Trial merge with the integration tip `kg/isaiah-gap-closure` `994818b` (`git merge-tree`) | code merges cleanly. Content conflicts appear only in `cookbook/log.md` and `cookbook/references/index.md`. The merged tree was not tested. |
| wandb 0.26.1 behaviour probes (offline `wandb.init`, synthetic env, `HOME` in scratchpad, no network) | `WANDB_BASE_URL=""` raises `ValidationError`, `WANDB_BASE_URL=api.wandb.ai` raises `ValidationError`, and `WANDB_API_KEY="  "` raises `UsageError`, all inside `wandb.init(mode="offline")`. The gate accepts all three (see F1). |

## Codex r1 findings: re-check

1. The awk netrc copy was replaced by `scripts/export_wandb_netrc_entry.py`, which uses the `netrc` module and `hosts.get(host)`. Packed, split, conventional, default-only and unsafe-token layouts are tested. **Fixed.**
2. An empty `WANDB_MODE`, a whitespace key and a URL bypassing the gate are **fixed for online mode**. The same class remains open for `WANDB_BASE_URL=""` and for the whole offline path (F1).
3. Receipts now have a strict schema: types, bool-as-int, the telemetry enum, the sha256 pattern, constant identity and source history. **Fixed.** `started_at` validation has no test (F4).
4. URL userinfo is rejected without being echoed. **Fixed and tested.**
5. The attempt-order test and the `FullConfig.from_file` sentinel now exist. **Fixed.** My mutation M15 is killed.
6. Offline-to-online resume is documented in the README and in the workflow. **Fixed.**
7. The Slurm audit's timing and inventory are corrected. **Fixed.**

## Mutations (scratch only; each restored, and the file bytes matched HEAD afterwards)

Every mutation ran against the three targeted test files.

| # | Mutation | Result |
| --- | --- | --- |
| M1 | `record_attempt` mismatch check made one-directional (a W&B logger is allowed under `telemetry=disabled`) | **SURVIVED** (156 passed). An independent probe test killed it. |
| M2 | `main`: `plan_attempt(resume=False)` always, so every resume would raise `FileExistsError` | **SURVIVED** |
| M3 | `main`: `experiment_id=None`, silently dropping `--experiment-id` | **SURVIVED** |
| M4 | `main`: constant `config_sha256` | **SURVIVED** |
| M5 | `main`: `--source-commit` ignored | **SURVIVED** |
| M6 | `git_source_commit` drops the `-dirty` suffix | **SURVIVED**. An independent probe test (tmp git repo) killed it. |
| M7 | receipt type check: `started_at` no longer validated | **SURVIVED** |
| M8 | receipt type check: `source_commit` no longer validated | survived, but equivalent: the `attempt_source_commits` text and history checks reject the same records |
| M9 | `WANDB_MODE` conflict checked only in online mode | killed |
| M10 | padded `WANDB_API_KEY` accepted | killed |
| M11 | export script's terminal refusal removed | killed |
| M12 | export falls back to the netrc `default` entry | killed |
| M13 | `v3/*` W&B summary dropped | killed |
| M14 | URL userinfo allowed | killed |
| M15 | telemetry gate moved after the config read | killed |
| M16 | constant experiment id and job type across receipts not enforced | killed |
| M17 | resume ignores the recorded job type | killed |
| M18 | outage-recorded line printed for online too | killed |
| M19 | export drops the `login or "user"` placeholder | **SURVIVED**. An independent probe test killed it. |
| M20 | `--log-mode debug --wandb-mode offline` allowed | killed |

Totals: 20 mutations. 12 were killed and 8 survived; one survivor (M8) is equivalent, which leaves 7 real survivors. The independent probe tests (3 passed on HEAD) killed M1, M6 and M19 and were then deleted.

## Findings

### P2

**F1. The fail-fast gate still lets predictable wandb settings errors through; they fail inside `wandb.init`, after env and model setup.**
- Locations: `python/owl/train/logging.py:183` (key and URL validated only when `mode is WANDB_ONLINE`) and `:222` (`environ.get("WANDB_BASE_URL") or DEFAULT_WANDB_BASE_URL` treats a present-but-empty URL as the default).
- Failure scenario, reproduced against the installed wandb 0.26.1:
  - `WANDB_BASE_URL=""` with a valid key: `check_telemetry(..., ONLINE)` returns `wandb-online`, but wandb's `Settings` raises "Input should be a valid URL, input is empty".
  - With `--wandb-mode offline`, the gate validates neither the URL nor the key. `WANDB_BASE_URL=api.wandb.ai` and `WANDB_API_KEY="  "` both pass the gate, then `wandb.init(mode="offline")` raises `ValidationError` or `UsageError` after the run directory, env, model and optimizer exist.
- This is the residual of r1 finding 2, and it contradicts the README's claim that the gate runs before config load.
- **Fix:** in `check_telemetry`, for both W&B modes:
  - reject a present-but-empty `WANDB_BASE_URL` and validate it with `wandb_host`;
  - reject a present `WANDB_API_KEY` that is blank or padded;
  - add offline-mode and empty-URL tests.

**F2. `main`'s receipt wiring is untested, so a regression that breaks every resume or corrupts the receipt passes the whole suite.**
- Location: `scripts/run_ppo.py:220-231` (and `:182`).
- The only `main`-level W&B tests use the Kaggriculture config, which stops at the not-wired error before `plan_attempt`. No test drives an Orbit fresh launch or a resume through `plan_attempt`.
- M2 (`resume=False`, so every resume raises `FileExistsError`), M3 (`--experiment-id` silently dropped, so the W&B group is wrong), M4 (a constant config hash) and M5 (`--source-commit` ignored) all survive.
- The cookbook Reference's statement that planning before env or model setup "rejects a resume without receipts early" holds only by code reading; no test checks it.
- **Fix:** add `main`-level tests with a stubbed Orbit startup:
  - a fresh launch with `--experiment-id` and `--source-commit`, asserting the receipt and `config_sha256(cfg)`;
  - a resume of a run directory that has a receipt, which must reach attempt 1;
  - a resume without `attempts.jsonl`, which must fail before `VectorizedEnv` is constructed.

**F3. The launcher-scope wording overclaims and omits the required follow-up.**
- Locations:
  - `cookbook/log.md:3`, "Wire every v3 launcher to W&B";
  - the Reference title at `cookbook/references/v3-launchers-fail-fast-without-wandb-credentials.md:3`, "v3 launchers fail fast";
  - the Workflow description at `cookbook/workflows/install-the-wandb-credential-before-any-pod-launch.md:4`, "Launchers then default to online W&B … and fail fast".
- The body text does admit that BC lags (`:59`, `:81`, audit `:56`). But none of these files says "all v3 launchers on integration". None says the train_bc adoption is a recorded step that runs right after the BC landing in this workflow; the audit frames it as "the follow-up for the BC lane (not done here)".
- The scope rule for this review requires both.
- **Fix:**
  - Retitle the log entry to "Wire all v3 launchers on integration to W&B …".
  - Scope the Reference title, the Workflow description and the index lines the same way. If the Reference file name changes, keep its incoming links.
  - In the audit, the Reference and the Workflow, state that the train_bc gate, explicit-offline and receipt adoption runs immediately after the BC landing in this same workflow.

### P3

**F4. Test gaps on smaller guards.** Each surviving mutant was killed by a small independent test.
- `logging.py:441`: the logger/telemetry mismatch check is tested in one direction only (M1).
- `logging.py:279`: the `-dirty` suffix has no test (M6), although README and cookbook claim it as source custody.
- `logging.py:421`: `started_at` validation has no test (M7).
- `logging.py:143`: the `login or "user"` placeholder has no test (M19).
- **Fix:** add these four tests.

**F5. `--source-commit` overrides git even when git metadata exists** (`scripts/run_ppo.py:182`).
- A mistyped or stale value becomes the receipt's source identity, and nothing flags it.
- **Fix:** accept `--source-commit` only when `git rev-parse` fails, or record both values and reject a mismatch.

**F6. The receipt's `telemetry_mode` is the requested mode, not the observed one** (`logging.py:432-460`).
- `record_attempt` never reads the W&B run's actual mode, such as `run.offline` or `run.settings.mode`. The W&B SDK can downgrade a run, for example to a noop on a login-prompt timeout. It is unlikely once the gate has passed, but the receipt would then still say `wandb-online`.
- **Fix:** assert that the observed mode matches in `WandbLogger`, or record it.

**F7. The branch is behind the integration tip.**
- Integration moved past `faed717`: the Tasks 1.4/1.5 native env and a new Kaggriculture not-wired guard in `run_ppo`.
- A trial merge auto-merges `run_ppo.py` and its tests, and conflicts only in `cookbook/log.md` and `cookbook/references/index.md`. The merged tree is untested.
- **Fix:** merge the tip (regular merge commit), resolve the two index and log conflicts, and rerun `just prepare`.

## Owner-rule checks

- `scripts/run_ppo.py` stays the one trainer. No v2 model code is added.
- No observation, model input, critic, reward or opponent-identity path changed; the stateless policy is unaffected.
- The W&B project is `kg-v3` and every mode is tested.
- Outages are visible in three places: the stderr banner, the `attempts.jsonl` `telemetry_mode` field and the W&B summary `v3/telemetry_mode`.
- Credentials never enter receipts or error text; the tests assert it.
- The pod credential step exists, and it is tested only by a local simulation.

VERDICT: REQUEST CHANGES
