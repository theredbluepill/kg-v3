Reviewer: independent Claude subagent (substitute for Codex during its usage limit; owner-approved). Not a Codex verdict.

# claude-verify-wandb-r3: W&B wiring for all v3 launchers on integration

## Scope

- **Branch.** `kg/rebuild-wandb` at tip `5186d70` (the claude-verify-wandb-r2 fix), in `/Users/poonszesen/kg-v3-wandb`.
  - The requested diff is `faed717...HEAD`. It contains 2,357 files because it includes the integration merge `46d05ba`.
  - I reviewed the W&B-specific surface as `994818b..HEAD`, which is 29 files. The code files are `python/owl/train/logging.py`, `scripts/run_ppo.py` and `scripts/export_wandb_netrc_entry.py`, plus their three test files. The rest are the README, the Reference and Workflow notes, both indexes, `cookbook/log.md`, the audit and the ops receipts. I read the r2-fix commit `5186d70` line by line.
- **Spec.**
  - The owner: "make sure all v3 experiments wired to w&b".
  - The CLAUDE.md telemetry rules: `kg-v3`, v3 identifiers, outages kept visible, custody.
  - Codex `verify-wandb-r1.md`.
  - The prior finding to re-check: claude-verify-wandb-r2 R1, which carries forward r1 F1. I also re-checked r2's R2 to R4.
- **Out of scope.** `scripts/train_bc.py` and `python/owl/train/bc.py`, which are on neither the branch nor integration. I checked only that the docs scope them correctly.
- **Scratch copy.**
  - I created a detached worktree `/tmp/cv-wandb-r3` at `5186d70` and copied fixtures in from `/Users/poonszesen/kg-v3/tests/fixtures`. `uv sync` and `maturin develop --release` (with `CARGO_BUILD_JOBS=2`) exited 0.
  - I removed it with `git worktree remove --force`.
  - Afterwards `/Users/poonszesen/kg-v3-wandb` has 0 tracked modifications and HEAD is still `5186d70`.
  - `git worktree list` still shows an older scratch worktree that I did not create, at `.../scratchpad/cv-wandb-r3` on `06a50d8`. It looks left over from an earlier attempt. I left it alone.

## Checks (scratch copy, Mac CPU, OMP_NUM_THREADS=2)

| Check | Result |
| --- | --- |
| Targeted suite: `pytest tests/owl/train/test_logging.py tests/scripts/test_run_ppo.py tests/scripts/test_export_wandb_netrc_entry.py -m "not slow"` | **204 passed**, 1.47 s, 344 MB RSS |
| `ruff check`, `ruff format --check`, `check_python_311_syntax.py` | all pass; 141 files already formatted; 3.11 syntax check exit 0 |
| `mypy python/ scripts/` | no issues in 72 source files |
| `check_doc_freshness.py`; `pymarkdownlnt scan *.md` and `--recurse python/ scripts/ tests/ src/ docs/` | all exit 0 |
| Full Python `-m "not slow"`, sharded | tests/owl 871 passed / 12 skipped (551 MB); tests/scripts 252 / 1 (540 MB); tests/tools 283 / 1 (439 MB); tests/kaggriculture run one file at a time, 1,088 / 7 (largest file checked is `test_teacher.py` at 974 MB). **Total: 2,494 passed, 21 skipped.** This matches `prepare-claude-r2-fix.log`. |
| Rust | Not rerun. `994818b..HEAD` touches no Rust. |
| Cookbook lint (`.claude/hooks/cookbook-lint.mjs`) on the Reference and the Workflow | both exit 0. Every `repository:` source path exists. |
| Probes against the installed wandb 0.26.1, one process per case: `check_telemetry(WANDB, OFFLINE)`, then the real `wandb.init(mode="offline")` if the gate passed. Scratch `HOME`, no network. | Results below. |

**Gate probes.**

- **Now rejected by the gate, before any config.** Each is rejected by its error type only, and the value is never quoted:
  - `https://wandb.ai`, `http://api.wandb.ai` and `https://app.wandb.ai` (`value_error`);
  - `https://ho st.example` (`url_parsing`);
  - `https://` ("http(s) URL with a host").
- **Passed by the gate, and `wandb.init` also succeeds offline:**
  - `https://api.wandb.ai/`, `https://API.WANDB.AI`, `https://w.example:8443/`, `http://self-hosted.internal`, `https://h.example/path?q=1` and `https://h.example#frag`;
  - `WANDB_API_KEY=x` and `WANDB_API_KEY="abc def"`;
  - `WANDB_MODE=offline`.

  So I found no remaining case where the gate passes a base URL or key and offline `wandb.init` then rejects it.
- **A different `WANDB_MODE` is rejected by the gate, but wandb itself would accept it.** The gate rejects `WANDB_MODE=online` and `WANDB_MODE=disabled` when the flag is offline. Without the gate, `wandb.init(mode="offline")` with either value starts an **offline** run and raises nothing, because init arguments override environment settings (`wandb_init.py` `make_run_settings`). Only an empty `WANDB_MODE` makes `wandb.init` raise `ValidationError`. This matters for finding P3-1.
- **Other settings.** `WANDB_DISABLED=true` still gives offline. `wandb.Settings(base_url=...)` does not read the environment (I checked with `WANDB_SILENT=maybe`), so the validator the gate calls depends only on the value passed in.

## Prior findings: re-check

- **claude-verify-wandb-r2 R1 / r1 F1 (P2: the gate passed settings that wandb rejects later): RESOLVED.**
  - `wandb_host` (`python/owl/train/logging.py:242-268`) and `_check_environment_key` (`:102-121`) now call `_wandb_settings_rejection` (`:271-287`), which runs the installed `wandb.Settings(base_url=...)` or `wandb.Settings(api_key=...)` validator. `check_telemetry` (`:207-208`) calls both in either W&B mode.
  - All four URLs the reviewer reproduced, plus `https://`, now fail at the gate. The probes above confirm this in separate processes against the real wandb.
  - These cases are tested in both modes in `test_gate_rejects_settings_wandb_init_would_reject_in_both_modes`. A test that stubs wandb's key validator checks that the key never appears in the error and that the error has no `__cause__`. `test_gate_accepts_server_addresses_wandb_accepts` checks that valid addresses are not rejected.
  - The README, the Reference and the index line now name the three checked variables and say "Other W&B settings are not pre-checked." One rationale in that wording is still wrong for `WANDB_MODE` (P3-1 below), but that does not reopen R1.
- **r2 R2 (P3: host-less `https://` untested): RESOLVED.** `https://` is in both the never-quoted test and the both-modes test. Mutant V10 (below) is killed.
- **r2 R3 (P3: offline-resume claim): RESOLVED.** The README (`:356-363`), Workflow step 4, the Reference Limits and the audit Limits all say wandb 0.26.1 ignores `resume` offline and starts a same-ID segment. They also say syncing several segments is unverified, and the Workflow asks the first such sync to record the result.
- **r2 R4 (P3: test memory): RESOLVED as a record only.** It is noted in `prepare-claude-r2-fix.log`. It was not introduced by this branch.

## Mutations (scratch copy only; each is one exact replacement, run against the targeted suite with `-x`, restored and SHA-256-checked)

| # | Mutation | Result |
| --- | --- | --- |
| V1 | `wandb_host`: wandb URL validator skipped | killed (1 failed) |
| V2 | `_check_environment_key`: wandb key validator skipped | killed |
| V3 | `_wandb_settings_rejection` returns `str(error)`, which quotes the value | killed |
| V4 | `UsageError` swallowed as accepted (`return None`) | killed |
| V5 | `UsageError` not caught (only `KeyError`) | killed |
| V6 | `check_telemetry`: key check dropped from both-modes block (so it runs online only) | killed |
| V7 | `check_telemetry`: URL check dropped from both-modes block (so it runs online only) | killed |
| V8 | validator fed `http://` rewritten to `https://`, which hides `http://api.wandb.ai` | killed |
| V9 | `WANDB_MODE` conflict ignored | killed |
| V10 | `wandb_host`: `or not parts.hostname` removed (r2's N7) | killed |
| V11 | `wandb_host`: userinfo rejection removed | killed |
| V12 | `WandbLogger`: observed-mode mismatch check removed | killed |
| V13 | `run_ppo.main`: telemetry gate moved after the source-commit read, CLI-override log and config load | killed |

**Totals: 13 mutations, 13 killed, 0 survived.** Every new r2-fix guard (V1 to V8) has at least one killing test.

## Findings

No P1 or P2 findings.

### P3

**P3-1. The docs say `wandb.init` would reject a differing `WANDB_MODE`, but wandb 0.26.1 overrides it silently.**
- **Where.**
  - `README.md:326-330`: "three environment settings that `wandb.init` would otherwise reject later: a set `WANDB_MODE`, even an empty one, that differs from `--wandb-mode`".
  - The Reference description (`cookbook/references/v3-launchers-fail-fast-without-wandb-credentials.md:4`): "a WANDB_MODE, WANDB_BASE_URL or WANDB_API_KEY that wandb.init would reject later".
  - The matching `cookbook/references/index.md` line.
- **Evidence.** In separate processes, `wandb.init(mode="offline")` with `WANDB_MODE=online` or `WANDB_MODE=disabled` starts an offline run and raises nothing. Only an empty `WANDB_MODE` raises `ValidationError`.
- **Impact.** The gate is stricter than wandb, and rejecting the conflict is good policy, so behaviour is safe. Only the stated rationale is wrong.
- **Fix.** Reword along these lines: "rejects a set `WANDB_MODE` that differs from the flag (wandb would silently let the flag win, or reject an empty value), and a `WANDB_BASE_URL` or `WANDB_API_KEY` that `wandb.init` would reject".

**P3-2. The Reference's "Verification (this version)" does not cite the prepare run for the current code.**
- **Where.** `cookbook/references/v3-launchers-fail-fast-without-wandb-credentials.md`, the "Full preparation" bullet.
- **What.**
  - The bullet cites `just prepare` on `0e608e2` and `b90286a`. It does not mention `prepare-claude-r2-fix.log`, although that log is in the note's sources. That log is `py-prepare` plus `docs-lint`, with 2,494 passed and 21 skipped, on "`6d4ef48` plus the uncommitted fix", which is the content of `5186d70`.
  - The "Claude verification r2" bullet says the recheck killed N7 and "the removal of wandb's URL validator". It leaves out V2, the key-validator removal, which the log shows was killed too.
  - The audit's "Fix" section has no bullet for the r2 fix (the `wandb.Settings` validator). Only its Limits line changed.
- **Impact.** The counts are correct; I reproduced 2,494 / 21 on `5186d70`. The record is incomplete, not wrong.
- **Fix.** Add a line citing `prepare-claude-r2-fix.log` (Python and docs only; no Rust changed since `994818b`) and this r3 run. Mention the key validator in the r2 bullet. Add one "After claude-verify-wandb-r2" bullet to the audit.

## Owner-rule checks

- **One trainer.** `scripts/run_ppo.py` stays the one trainer. No v2 model code is added.
- **Stateless policy unaffected.** No observation, model input, critic, reward, normalisation or opponent-identity path changed.
- **Project and identifiers.** The project is `kg-v3`. The group, job type, tags, `v3.*` config and `v3/*` summary are covered by tests.
- **Outages are visible:**
  - a stderr banner for offline and for debug;
  - `telemetry_mode` in `attempts.jsonl` and in `v3/telemetry_mode`;
  - the receipt path printed again;
  - a run that the SDK starts in a different mode is rejected (V12 killed).
- **Credential safety.** The credential never reaches receipts or error text. The tests check this with `_SECRET`, and V3 is killed.
- **Pod credential step.** It exists (`cookbook/workflows/install-the-wandb-credential-before-any-pod-launch.md` plus the tested export script). It is simulated locally only, and the Workflow's Limits say so.
- **Scope wording.** Everywhere the docs say "all v3 launchers on integration (`run_ppo` only)", and they name `train_bc`'s adoption of the gate, explicit offline and receipt as the step that runs right after the BC landing in this workflow. `scripts/benchmark_checkpoints.py` refuses Kaggriculture (`require_orbit_env`), so the audit is right that it is not a v3 experiment today.
- **Branch hygiene.** No push. No commit to the reviewed branch. I did not touch the pod workflow's worktree or branches, or `~/kaggriculture-v2`.

VERDICT: APPROVE WITH EDITS
