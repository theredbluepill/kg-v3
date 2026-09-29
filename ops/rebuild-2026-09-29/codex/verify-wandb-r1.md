Reviewed `faed717..05cbd88de6e017b33755d28d39fd77cfa1eb71f2` on `kg/rebuild-wandb`.

The [full report](/private/tmp/verify-wandb-r1/report.md) includes every mutation’s exact failing tests, counts, commands, and restoration hashes. No tracked files changed, commits, pushes, training experiments, GPU runs, real credential reads, or W&B network calls occurred.

**Findings**

1. **P2 — Credential extraction can copy unrelated credentials.**  
   `cookbook/workflows/install-the-wandb-credential-before-any-pod-launch.md:20`  
   Netrc is token-based; the `awk` filter assumes entry boundaries align with lines. Valid synthetic inputs containing a W&B entry followed on the same line by another machine or `default` copy both credentials. A valid split `machine`/hostname layout extracts nothing. The suggested grep count does not detect the extra credentials.  
   **Fix:** parse netrc, select exactly `hosts["api.wandb.ai"]`, safely serialize only that entry to SSH stdin, and reject empty extraction before installation. Test packed entries, defaults, and split tokens.

2. **P2 — Predictable configuration errors bypass the early gate.**  
   `python/owl/train/logging.py:87`, `:135`, `:174`  
   Compared against installed W&B 0.26.1 using synthetic inputs:
   - `WANDB_MODE=""` passes the gate but fails SDK settings validation.
   - A whitespace-only environment key plus valid netrc passes through netrc fallback, but the SDK rejects the environment key.
   - A nonempty environment key bypasses base-URL validation; a URL without a scheme therefore fails only later.

   These failures occur during logger initialization, after environment/model setup.  
   **Fix:** reject present-but-empty mode and whitespace keys, and validate the base URL before returning an environment credential source. Keep errors sanitized.

3. **P2 — Receipt validation is not a strict schema.**  
   `python/owl/train/logging.py:315`, `:277`  
   Receipts with the correct keys accept `attempt=false`, a list-valued experiment ID, an unknown telemetry mode, and a null config hash. Records can also change experiment identity between attempts. `plan_attempt` stringifies malformed persisted IDs instead of rejecting them. Five independent negative tests fail on unchanged HEAD.  
   **Fix:** validate field types and values, exclude booleans as attempt integers, validate persisted IDs and telemetry modes, and enforce consistent experiment/job identity and source history.

4. **P2 — URL errors can expose embedded credentials.**  
   `python/owl/train/logging.py:115`, `:175`  
   `_wandb_host` returns `.netloc`, including userinfo. A synthetic URL containing a password causes the missing-credential exception to include that password. The malformed-URL error at line 177 also quotes the raw URL. Distributed exception handling can print these messages to stderr.  
   **Fix:** reject URL userinfo without echoing it, avoid raw URL interpolation, and expose only validated host/port information. Test exception text and rendered tracebacks with synthetic markers.

5. **P3 — Two invariants lack effective repository tests.**  
   `tests/owl/train/test_logging.py:250`, `tests/scripts/test_run_ppo.py:3534`  
   Removing attempt-order validation survives all logging tests. Reading `FullConfig` before the telemetry gate survives all run_ppo tests because the existing test watches `_log_cli_overrides`, not the actual config read.  
   **Fix:** add a malformed-order receipt test and a direct `FullConfig.from_file` sentinel. Independent scratch tests killed both mutants.

6. **P3 — Document offline-to-online resume requirements.**  
   `README.md:390`, `python/owl/train/logging.py:439`, workflow `:62`  
   Generic resume defaults online and forwards the existing ID with `resume="must"`. A never-synced offline run has no remote run, so this fails. Installed SDK source and a local API double confirm this without network.  
   **Fix:** document and test sync-before-online-resume and deliberate offline continuation. Preserve strict `must` behavior for established remote runs.

7. **P3 — Slurm audit timing and inventory need correction.**  
   `ops/rebuild-2026-09-29/wandb-audit.md:14`  
   The credential guard runs inside the allocated sbatch job before container/trainer startup, not before submission. The experiment/scaling submission wrappers and interactive-shell wrapper are not explicitly inventoried.  
   **Fix:** list/group those wrappers and state the actual gate timing.

**Commands and results**

Tests used `OMP_NUM_THREADS=2`, synthetic credentials, and network/netrc-access guards.

| Check | Result |
|---|---|
| `uv run pytest tests/owl/train/test_logging.py tests/scripts/test_run_ppo.py -m "not slow" -q` | **133 passed, 1 skipped, 0 failed** |
| `uvx --from rust-just just py-prepare` | **1,718 passed, 11 skipped, 0 failed; exit 0** |
| Formatting/lint | Passed; formatter left 119 files unchanged |
| Python 3.11 syntax scanner | Passed |
| Mypy | Passed, 64 source files |
| Docs freshness | Passed |
| Unmutated scratch baseline | **133 passed, 1 skipped** |
| Existing-suite mutation campaign | **34 mutations: 32 killed, 2 survived** |
| Independent receipt/rank/resume probes | **4 passed, 5 failed**; failures demonstrate finding 3 |
| Supplemental invariant probes | Baseline **2 passed**; each surviving mutant then produced **1 failed, 1 passed**, with 8 deselected |
| Synthetic workflow checks | **3 passed, 3 failed** |
| AST comparison | Eleven selected definitions unchanged from baseline |
| Final tracked/staged diffs | Both clean; `git status --short` empty |

The focused check took 2.80 seconds overall and peaked at 315,408,384 bytes. Mutation/probe subprocesses stayed below 1 GB and two minutes.

**Resource-limit deviation:** full `py-prepare` completed in 54.21 seconds but peaked at **2,028,290,048 bytes**, exceeding the requested <1 GB ceiling. It was not repeated. An initial monitoring wrapper could not execute sandbox-restricted `ps`; its focused pytest completed successfully, and the command was repeated with child-resource measurement. An initial AST inspection used system Python 3.9; it was rerun successfully with the project interpreter. These were verifier harness issues, not product failures.

**Mutation table**

All mutations used scratch copies. Test names below omit the common `test_` prefix; exact parameterized identifiers and individual counts are in the full report.

| Guard | Mutations | Killing tests or SURVIVED | Restored |
|---|---|---|---|
| G1 | Change project to `orbit-wars` | `wandb_logger_opens_the_v3_project_with_v3_identity`; `other_launchers_reuse_the_logger_with_their_own_config` | Yes |
| G2 | Ignore env key; accept blank key; accept empty password; ignore `NETRC`; ignore custom host; remove missing-key rejection | Credential-source tests and `online_gate_requires_credentials_and_stays_quiet` killed all six | Yes |
| G3 | Include raw netrc parser error | `malformed_netrc_error_never_quotes_the_file` | Yes |
| G4 | Remove outage banner; print outage online | `outage_modes_skip_credentials_and_warn_loudly`; `main_offline_mode_announces_the_outage_without_credentials`; online quiet test | Yes |
| G5 | Remove `WANDB_MODE` conflict check | `wandb_mode_environment_cannot_override_the_flag` | Yes |
| G6 | Allow debug/offline; remove `_validate_args` telemetry validation | `telemetry_modes_and_the_offline_debug_conflict`; `validate_args_rejects_offline_wandb_mode_with_debug_logging` | Yes |
| G7 | Allow experiment ID on resume | `validate_args_rejects_resume_experiment_id` | Yes |
| G8 | Allow existing fresh receipt; return empty missing receipts; allow changed ID; remove regex | Fresh/resumed identity, malformed receipt, explicit-ID, and CLI malformed-ID tests killed all four | Yes |
| G9 | Remove schema check | `resume_needs_well_formed_attempt_records` | Yes |
| G9 | Remove attempt-order check | **SURVIVED: 21 passed**; independent `attempt_order_is_rejected` killed it | Yes |
| G10 | Remove logger mismatch check; always record disabled | Logger/telemetry mismatch test and per-mode session receipt tests | Yes |
| G11 | Bypass telemetry gate | Main missing-credential and offline-banner tests | Yes |
| G11 | Read config before gate | **SURVIVED: 112 passed, 1 skipped**; independent `missing_credentials_precedes_actual_config_read` killed it | Yes |
| G12 | Skip attempt write; suppress recorded-outage line | `run_training_session_records_the_attempt_and_its_telemetry_mode` | Yes |
| G13 | Wrong group/job type; empty tags; omit v3 config/summary; force offline | V3 logger identity tests and reusable-launcher config test killed all six | Yes |
| G14 | Allow debug/online identity; allow W&B/disabled identity | `create_logger_rejects_a_mode_that_contradicts_the_identity` killed both | Yes |

**Audit and preserved behavior**

- At `faed717`, PPO uses project `orbit-wars`, has no telemetry gate/receipt, and initializes logging after setup. HEAD extends the shared logging module and canonical trainer.
- Both benchmark scripts, Slurm scripts, and all six specified ops-probe families are unchanged. Benchmarks remain Orbit-only; frozen probes retain local evidence. No additional v3 training launcher was found in the requested scope.
- Read-only BC inspection at branch commit `b626f24` confirms project `kg-v3`, name `bc-bc`, group `bc`, online default, silent launcher-level offline selection, and no telemetry-mode receipt. Dataset, model, compile, and optimizer setup precede logger creation.
- BC imports remain available. `MetricLogger` is not abstract; the added method does not break existing subclasses until they adopt `record_attempt`. BC source was not edited.
- `LogMode`, logger close behavior, `_logger_session`, checkpoint W&B ID handling, and selected logging methods are unchanged. Online/offline resume still forwards the original ID and `resume="must"`.
- Credentials and git are accessed only on rank 0. An independent worker-main probe confirms neither is accessed; existing worker-session coverage confirms the no-logger path.
- The implemented gate precedes config load and run-directory creation. Launch resolution still precedes it and can inspect resume checkpoints.
- No observation, temporal state, model input, or opponent-identity path changed.
- A netrc `default` password is accepted by both the gate and SDK. Malformed-netrc error suppression works. Ordinary API keys/passwords are not written into receipts or logger config; the URL-error leak prevents an unconditional secrecy guarantee.

**Docs and evidence**

README matches normal flag/project/receipt behavior, subject to the resume qualification above. Both new cookbook notes have complete required frontmatter, first tag `kaggriculture-v3`, and all **14 repository sources exist**. Indexes and the prepended log link them. BC adoption, missing Kaggriculture environment, and CPU-only verification limits are explicit.

The committed `wandb-2026-09-29/prepare.log` records a green full preparation: Rust **254 passed / 4 ignored**, engine **69 passed**, Python **1,718 passed / 11 skipped**, mypy on **64 files**, and final `EXIT=0`. It lacks an explicit tested commit hash; current Python results were independently reproduced. Historical live W&B observations were not repeated.

The synthetic installation check confirms mode **600**, refusal to overwrite an existing file, and unchanged bytes after refusal. No SSH was executed.

**Restoration**

Every mutation was restored. Scratch files, worktree files, and HEAD bytes match:

- `logging.py`: `88c9cf53996b80411dd3cf2f1478cb39b0b76cca1de69c755f8f78b84fd9ddaa`
- `run_ppo.py`: `56f7dc634b547e4cfdedce32c6c9c90b59d8584ea21f1e4d42e0f84cfe4f816c`

Both tracked and staged diffs are empty. Final `git status --short` prints nothing.

VERDICT: REQUEST CHANGES