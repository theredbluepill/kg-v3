# Independent verification — Task 3.4, round 3

Reviewed `e1458d2a717d9d731a367cbb78b98616ee6649f4...HEAD` (three-dot) on
`kg/rebuild-configs`, HEAD `99e51240bf871cd1ab8c1cb6c4b193c1328a915b`.
The requested base is the merge base. Scope: plan Task 3.4, accepted contract
v4, `configs/scaling_6m.yaml`, Isaiah's pinned `32b3ec9` code, and every finding
in `/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/codex/verify-3.4-r2.md`
and its full report. Two independent read-only sub-audits corroborated recipe,
workload, integration and documentation inspection. No tracked modifications.

## Finding

1. **P3 — The recipe Decision still describes completed config integration as pending.**
   `cookbook/decisions/recipe-choices-align-to-isaiah-without-owner-escalation.md:39`
   says the ranked configs are “schema-tested only until Task 3.1 lets them load.”
   This is false at this HEAD: all three YAMLs load through `FullConfig`, the
   factory constructs their models, and the canonical startup tests use the real
   loader. The heading at line 28, “Alignment targets — queued, not yet applied,”
   likewise fails to distinguish implemented config/critic work from remaining
   integration and qualification.

   **Fix:** reconcile these current-state passages with the configs Reference:
   typed config/model integration and CPU startup checking are implemented;
   native environment/trainer integration, teacher execution and GPU qualification
   remain pending. Preserve the owner's adopted rule and historical evidence.
   No implementation repair is requested by this finding.

## Closure of previous findings

- **R2 P1: resolved.** `KaggricultureEnvConfig` and `KaggricultureRewardConfig`
  provide the real env schema, the shared model union/factory registers the
  Kaggriculture model, and FullConfig validates both game pairings and reward
  restrictions. Every shipped YAML loads; all six previously skipped config
  integration cases are active. The unchanged Orbit config glob loads them too.
  `tests/scripts/test_run_ppo.py` leaves `FullConfig.from_file` unpatched and
  drives `main` through real YAML loading, CLI overrides, and saved-config resume
  adaptation. Invalid workload rejection, valid headroom lines and allocation
  sentinels are asserted. Removing the call fails all four relevant cases.
- **R2 P3: resolved.** Both ranked configs call `native_threads: 2` provisional
  and assign its measurement/verification to Task 6.1.
- **R1 P2 remains resolved.** Trunk calls are full-padding upper bounds and
  trunk headroom is a lower bound; head calls are exact. The packed 300-token
  counterexample remains tested (two chunks versus the padded upper bound three).
- **R1 P3 remains resolved.** Historical log entries scope old schema coverage
  accurately. Current coverage includes the env/reward schemas through the real
  loader. The new cookbook finding above is separate from these prior defects.

## Contract, recipe and code inspection

- The ranked presets keep Isaiah's optimizer, scheduler, PPO coefficients,
  last-best teacher coefficients/chunk setting, compile settings and 20M
  checkpoint cadence. Per-rank shapes are 128 envs/spm 8/accumulation 1 for
  two ranks and 64/4/1 for four ranks. Both preserve 256 global environments,
  16 optimizer steps per iteration, 16 global segments per optimizer step and
  16,384 game transitions per iteration. Tests combine literal totals, the real
  `_minibatch_indices` and complete parsed optimizer/PPO equality.
- The local upstream recipe files remain unchanged from Isaiah `32b3ec9`.
  Reusing the existing FlashAttention-forcing `configs/model/kaggriculture.yaml`
  instead of the plan's proposed `kaggriculture_gpu.yaml` is documented.
  The local preset uses a tiny model, FP32 and no compilation.
- Observation schema 3, two independent seat rows, hire limit 241, win/loss
  reward, gamma 1 and economic shaping 0.2 match contract v4 and the adopted
  recipe. The typed reward cap checks and terminal_scale match the contract;
  winner_ce is rejected. This verifies config semantics, not reward execution.
- Per-rank rollout/minibatch/teacher/evaluation seat rows are
  256/1,024/16,384/256 (two ranks) and 128/512/8,192/128 (four ranks).
  At 709 padded tokens and trunk width 512 the trunk capacity is 5,915 rows;
  the head capacity is 11,096 rows. Teacher workloads imply at most 3/2 trunk
  calls and exactly 2/1 head calls for two/four ranks respectively.
- Startup checking occurs after runtime shape adaptation and before the run
  directory, environment or model allocation. Only the main process logs the
  headroom. The teacher is omitted when disabled; Orbit checking is bypassed.
- No shim wrappers or second trainer were found. `require_orbit_env` is an
  explicit unsupported-game guard and typed narrowing, with no fallback.
  Orbit validation retains Isaiah's constraints. The factory, nominal trunk
  compile dispatch and masked critic merge are consistent with their scoped
  records and passing tests. The workload path stays in canonical `run_ppo.py`.
- Other current implementation claims in the reviewed config/model/GEMM
  References agree with the code and named evidence. The cookbook's historical
  mypy count of 60 is for `python/ scripts`; the requested narrower command
  checks 59 files, so that count difference is not a defect.

## Executed checks

- `uv run pytest tests/kaggriculture tests/owl tests/scripts tests/tools -m 'not slow' -q`
  — **1,373 passed, 4 skipped, 24.74 s**, exit 0 (`pytest.log`). Skips are one
  unavailable native grammar binding, two FlashAttention/CUDA cases, and one
  unavailable quantized backend. No config-integration cases are skipped.
- Retained `tests/owl tests/scripts tests/tools` are green within that run:
  **1,058 passed, 3 skipped**. A separate collection lists 1,061 cases
  (`orbit-collection.log`); this is a subset count, not a second execution.
- `uv run mypy python/owl scripts`
  — **success, no issues in 59 source files**, exit 0 (`mypy.log`).
- `cargo test --locked --offline`
  — **155 passed, 0 failed, 2 ignored**, exit 0 (`cargo-test.log`), including
  Orbit generation and replay parity. Existing sibling-worktree fixtures were
  temporarily linked and SHA-256 recorded; the link was removed afterward
  (`fixture-custody.json`). Fixtures were not regenerated and parity was not skipped.
- Additional real-loader/model/JSON roundtrip probe passes for all three YAMLs
  (`loader-probe.log`), with model construction on the meta device.
- Source/config/docs `git diff --check` passes when excluding `ops/`.
  The full three-dot whitespace check exits 2 for preserved raw verification
  logs in `ops/` (trailing whitespace/blank EOF); see `diff-check-all.log` and
  `diff-check.json`. These archival formatting diagnostics are not counted as
  a code defect, and raw evidence was not rewritten.

## Non-vacuity and custody

Mutations ran on isolated byte-identical copies of the runner, relevant tests
and configs using this checkout's owl package. Production tracked files were
never mutated. Each experiment ran baseline, mutation, and restoration:

1. Remove only `main`'s `_check_model_workload(...)` call:
   **4 passed → 4 failed → 4 passed** (79 deselected each time).
2. Change two-rank `segments_per_minibatch: 8` to `4`:
   **2 passed → 1 failed / 1 passed → 2 passed** (37 deselected each time).
   The two-rank cadence assertion fails while the untouched four-rank control passes.

Both scratch sources were restored byte-for-byte and checked by SHA-256 before
removing the temporary directory. `mutations.json` and the six mutation logs
record exit codes and evidence. Before/after hashes confirm **all 833 tracked
files unchanged**, HEAD unchanged, and empty `git diff HEAD --` (`before.json`,
`after.json`). Only this untracked verification evidence directory was added.

## Limits

This approves the Task 3.4 configuration/startup scope with the documentation
edit above. `run_ppo` still deliberately stops after workload checking because
native env and trainer integration are pending; no claim of a trainable
Kaggriculture end-to-end path is made. BC is a future caller of the workload
checker. Teacher/replay execution, dense-state memory, real Inductor/FlashAttention,
GPU throughput and competitive strength remain unqualified. No training or GPU
run was performed.

VERDICT: APPROVE WITH EDITS
