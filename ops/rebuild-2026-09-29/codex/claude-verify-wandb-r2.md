Reviewer: independent Claude subagent (substitute for Codex during its usage limit; owner-approved). Not a Codex verdict.

# claude-verify-wandb-r2: W&B wiring for all v3 launchers on integration

## Scope

- **Branch.** `kg/rebuild-wandb` at tip `6d4ef48`, reviewed in `/Users/poonszesen/kg-v3-wandb`.
  - Requested diff: `faed717...HEAD`. That range includes the integration merge `46d05ba`, so most of its 2,354 files come from integration.
  - The W&B-specific surface was reviewed as `994818b..HEAD`: 26 files. The code files are `python/owl/train/logging.py`, `scripts/run_ppo.py` and `scripts/export_wandb_netrc_entry.py`, plus their three test files, the README, two cookbook notes, the log, two indexes, the audit and ops receipts.
- **Spec.**
  - The owner's instruction: "make sure all v3 experiments wired to w&b".
  - The CLAUDE.md telemetry rules: kg-v3, v3 identifiers, outages visible, custody.
  - Codex `verify-wandb-r1.md`.
  - The three findings carried over from `claude-verify-wandb-r1.md`: F1, F2 and F3.
- **Out of scope.** `scripts/train_bc.py` and `python/owl/train/bc.py`, which are not on the branch or on integration. The docs were checked only for scoping them correctly.
- **Scratch copy.**
  - Detached worktree `/tmp/cv-wandb-r2` at `6d4ef48`. Fixtures were copied from `/Users/poonszesen/kg-v3/tests/fixtures`, and `owl.rs` was built with `CARGO_BUILD_JOBS=2 uv run maturin develop --release` (exit 0).
  - The worktree was removed with `git worktree remove --force`, so `/tmp/cv-wandb-r2` no longer exists.
  - `/Users/poonszesen/kg-v3-wandb` has 0 tracked modifications, and its HEAD is still `6d4ef48`.

## Checks (scratch, Mac CPU, OMP_NUM_THREADS=2)

| Check | Result |
| --- | --- |
| Targeted suite: `pytest tests/owl/train/test_logging.py tests/scripts/test_run_ppo.py tests/scripts/test_export_wandb_netrc_entry.py -m "not slow"` | **190 passed**, 1.09 s |
| `ruff check --select I`, `ruff format --check`, `check_python_311_syntax.py`, `ruff check` (python/ scripts/ tests/) | all pass; 141 files already formatted |
| `mypy python/ scripts/` | no issues in 72 source files |
| `check_doc_freshness.py`; `pymarkdownlnt scan *.md` and `--recurse python/ scripts/ tests/ src/ docs/` | all exit 0 |
| Full Python `-m "not slow"`, sharded | tests/owl 857 passed / 12 skipped (536 MB RSS); tests/scripts 252 / 1 (520 MB); tests/tools 283 / 1 (477 MB); tests/kaggriculture run per file, 1,088 / 7 (peak 1,037 MB in `test_model_heads.py`; the rest are ≤ 969 MB). **Total: 2,480 passed, 21 skipped.** This matches the committed `prepare-claude-r1-fix.log` for `0e608e2`. |
| Rust `rs-prepare` | Not rerun. `994818b..HEAD` has no Rust changes; the Rust diffs in `faed717...HEAD` come from the reviewed integration merge. |
| Cookbook lint (`.claude/hooks/cookbook-lint.mjs`) on both new notes | pass. Every `repository:` source exists. |
| Probes against the installed wandb 0.26.1: each case in its own process (wandb caches its settings per process), offline mode, scratch `HOME`, no network | See F1 below. Four `WANDB_BASE_URL` values pass the gate, then `wandb.init(mode="offline")` raises on them. `WANDB_DISABLED=true` does not change the mode. `resume="must"` in offline mode is ignored: wandb warns and starts a new offline run with the same ID. |

## Prior findings: re-check

- **F1 (P2, the gate passes settings that wandb rejects later): PARTIAL.**
  - Resolved:
    - The three cases the r1 reviewer reproduced are now rejected in both W&B modes: an empty `WANDB_BASE_URL`, `WANDB_BASE_URL=api.wandb.ai`, and a blank or padded `WANDB_API_KEY`.
    - These cases are tested by `test_gate_rejects_settings_wandb_init_would_reject_in_both_modes`, which is parametrised over online and offline.
    - The mutants below are killed: N5 (URL and key checked only in online mode), N6 (a padded key accepted) and N8 (an empty URL treated as the default).
  - Still open: the gate only mirrors part of wandb's URL validation, so four realistic values still pass it (see new finding R1).
- **F2 (P2, `main`'s receipt wiring was untested): RESOLVED.**
  - The new `main`-level tests with a stubbed Orbit startup:
    - `test_main_fresh_launch_plans_attempt_zero_with_the_flags`;
    - `test_main_rejects_a_source_commit_that_disagrees_with_git`;
    - `test_main_resume_with_a_receipt_plans_attempt_one`;
    - `test_main_resume_without_receipts_fails_before_the_env`.
  - All four r1 survivors are now killed: N1 (`resume=False`), N2 (`--experiment-id` dropped), N3 (a constant config hash) and N4 (`--source-commit` ignored).
  - The claim that a resume without receipts is rejected before `VectorizedEnv` is now tested (`envs_built == []`).
- **F3 (P2, the scope wording overclaimed): RESOLVED.** Each of the following now says "all v3 launchers on integration" (`run_ppo` only) and names the `train_bc` gate, explicit offline mode and receipt adoption as the step that runs right after the BC landing in this workflow:
  - the log title;
  - the Reference title, description and Scope paragraph;
  - the Workflow description and its Launcher contract section;
  - both index lines;
  - the audit, including its "Does `train_bc` need the same fail-fast? Yes" section with four steps.

  The Reference kept its file name, so no incoming link broke. A grep of the added text found no remaining "every v3 launcher" overclaim.

## Mutations (scratch only; each is one exact string replacement, restored and byte-checked)

Each mutant ran against the targeted suite above with `-x`.

| # | Mutation | Result |
| --- | --- | --- |
| N1 | `main`: `plan_attempt(resume=False)` | killed |
| N2 | `main`: `experiment_id=None` | killed |
| N3 | `main`: constant `config_sha256` | killed |
| N4 | `main`: `resolve_source_commit(override=None)` | killed |
| N5 | gate: URL and key checks only when online | killed |
| N6 | padded `WANDB_API_KEY` accepted (`if not key.strip()`) | killed |
| N7 | `wandb_host`: `or not parts.hostname` removed (an http(s) URL without a host is accepted) | **SURVIVED** |
| N8 | an empty `WANDB_BASE_URL` falls back to the default | killed |
| N9 | `_observed_wandb_mode` always returns online | killed |
| N10 | receipt attempt order not checked | killed |
| N11 | `--experiment-id` allowed on resume | killed |
| N12 | `_validate_args` drops the offline-with-debug check | killed |
| N13 | receipt always records `wandb-online` | killed |
| N14 | resume drops the earlier source commits | killed |
| N15 | a `--source-commit` that disagrees with git is accepted | killed |
| N16 | `-dirty` suffix dropped | killed |
| N17 | a mismatched-mode run is not finished before raising | killed |
| N18 | outage banner printed only for offline, not for debug | killed |
| N19 | a `WANDB_MODE` conflict is ignored | killed |
| N20 | a fresh run may append to existing receipts | killed |
| N21 | the export falls back to netrc `default` (`authenticators`) | killed |
| N22 | telemetry gate moved after the source-commit read, the CLI overrides and the config read | killed |

**Totals: 22 mutations, 21 killed, 1 survived (N7).** N7 is not equivalent. Without the hostname check, `WANDB_BASE_URL=https://` yields the netloc `""` and passes the gate, and wandb then rejects it inside `wandb.init`. No test covers a URL without a host.

## Findings

### P2

**R1. The gate still passes `WANDB_BASE_URL` values that wandb 0.26.1 rejects inside `wandb.init`, and the docs claim it rejects what `wandb.init` would reject.**
- **Where.**
  - Code: `python/owl/train/logging.py:236-257` (`wandb_host`) and `:201` (`check_telemetry`).
  - Claims: `README.md:326-330`; the Reference description (`cookbook/references/v3-launchers-fail-fast-without-wandb-credentials.md:4`, "in either W&B mode on the settings wandb.init would reject later"); the References index line.
- **Scenario.** This was reproduced one process per case, with `check_telemetry(LogMode.WANDB, WandbMode.OFFLINE, ...)` followed by `wandb.init(mode="offline")`. The gate returns `wandb-offline` for each of these values, and then `wandb.init` raises `ValidationError`:
  - `https://wandb.ai` ("not a valid server address");
  - `http://api.wandb.ai` ("http is not secure");
  - `https://app.wandb.ai`;
  - `https://ho st.example` (pydantic `url_parsing`).
- **Consequence.**
  - The same `Settings` validation runs online. With `WANDB_API_KEY` set, the online gate also passes these values.
  - The launch fails only after the run directory, env, model and optimizer exist. The fail-fast contract exists to prevent exactly that. The first two values are plausible operator typos.
  - This is the residual of r1 F1: the three cases r1 reproduced are fixed, but the gate still mirrors wandb's URL rules only in part.
- **Fix.**
  - In `check_telemetry`, for both W&B modes, validate with wandb's own rules after the existing userinfo rejection.
  - For example, `wandb.Settings(base_url=environ["WANDB_BASE_URL"])` when the variable is set, and `wandb.Settings(api_key=...)` for a set key. Catch the error and re-raise it without quoting the value. I checked that `wandb.Settings` rejects all four URLs, `"  "` and `"x "`, and accepts `https://api.wandb.ai` and `https://w.example:8443/`.
  - Alternatively, mirror the two `wandb.ai` regexes and a strict URL parse.
  - Add tests for `https://wandb.ai`, `http://api.wandb.ai` and a host-less `https://`; the last one also kills N7.
  - Or narrow the claims to the exact list the gate checks.

### P3

**R2. A host-less http(s) URL is untested** (`python/owl/train/logging.py:255`). Mutant N7 survives. Fix: add `"https://"` to the cases in `test_base_url_is_validated_first_and_never_quoted`, or fold it into the R1 tests.

**R3. The offline-resume recipe is stated as fact but was not exercised, and wandb does not honour `resume="must"` offline.**
- **Where.**
  - `README.md:351-355` says "A resume reopens the saved W&B run ID with `resume="must"` in the chosen mode" and "To keep the outage deliberately, resume with `--wandb-mode offline`".
  - The pod Workflow step 4, `cookbook/workflows/install-the-wandb-credential-before-any-pod-launch.md:70`, says the same.
- **What happens.** In wandb 0.26.1 (`wandb/sdk/wandb_init.py:1002-1006`, and probed), offline mode ignores `resume` and warns "`resume` will be ignored ... Starting a new run with run id ...". The run directory then holds several `offline-run-*` folders with the same ID.
- **What is unverified.** What the Workflow's `wandb sync <run_dir>/wandb/offline-run-*` does with several folders for one ID.
- **Conflict.** The audit's own Limits section says offline resume "was not exercised".
- **Fix.**
  - State in the README and the Workflow that an offline resume starts a same-ID offline segment, and that wandb warns that `resume` is ignored.
  - Mark the multi-segment sync as unverified, or verify it once on a scratch project.

**R4. Test memory.** `tests/kaggriculture/test_model_heads.py` alone peaks at 1,037 MB RSS, above the 1 GB guideline. The branch did not introduce this, and it does not affect the W&B verdict. Noted for whoever sets the prepare watchdog.

## Owner-rule checks

- `scripts/run_ppo.py` stays the one trainer. No v2 model code is added.
- No observation, model input, critic, reward, normalisation or opponent-identity path changed, so the stateless policy is unaffected.
- The project is `kg-v3`. The group, job type, tags, `v3.*` config and `v3/*` summary are tested.
- Outages are visible in three places: the stderr banner (tested for offline and debug), `telemetry_mode` in `attempts.jsonl`, and `v3/telemetry_mode` in the summary. A run the SDK starts in another mode is rejected.
- The credential never enters receipts or error text; the tests assert this with `_SECRET`.
- The pod credential step exists. The export script is tested, and the install pipeline is only simulated locally, as the Workflow's Limits section states.
- The `train_bc` scope is recorded correctly as a follow-up that runs right after the BC landing.
- No push, no commit to the reviewed branch, and no changes to the pod workflow's worktrees or branches.

VERDICT: REQUEST CHANGES
