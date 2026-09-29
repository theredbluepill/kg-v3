# Independent verification — Task 3.4, round 2

Reviewed `e1458d2a717d9d731a367cbb78b98616ee6649f4...HEAD` on
`kg/rebuild-configs`, HEAD `928e119753181624efb2ea24622a8f45522499e9`.
The merge base is the requested base. Scope: plan Task 3.4, contract v4,
`configs/scaling_6m.yaml`, Isaiah's pinned `32b3ec9` code, and every finding in
`/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/codex/verify-3.4-r1.md`
and its linked full report. Two independent read-only subreviews corroborated
config/contract and startup/cookbook inspection. No tracked edits remain.

## Findings

1. **P1 — The prior startup integration finding is only partially resolved.**
   `scripts/run_ppo.py:149` now invokes the real checker in the correct place,
   after runtime shape adaptation and before run-directory/env/model allocation.
   But `FullConfig.from_file` at `scripts/run_ppo.py:137` still rejects all three
   shipped Kaggriculture configs before that call. A direct probe reproduced
   five errors for each: unsupported observation, action and model union tags,
   plus forbidden `reward_shaping` and `native_threads`.

   `tests/scripts/test_run_ppo.py:2742` bypasses validation with
   `FullConfig.model_construct` and substitutes an Orbit environment;
   `tests/scripts/test_run_ppo.py:2787` replaces the loader. Those tests genuinely
   exercise the new call, but do not prove config-load integration.
   `tests/kaggriculture/test_configs.py:303` and the training-config glob still
   skip all six real config integration cases. R1's full report explicitly
   required finishing the game/config dependency; the plan requires the
   assertion at config load. The new documentation discloses the dependency,
   which is accurate, but does not satisfy it.

   **Fix:** complete the prerequisite typed Kaggriculture config integration
   (and factory dependency for the skipped construction test), then exercise
   canonical startup from the real YAML loader. Assert invalid workload
   rejection and valid headroom output before allocation without replacing
   `FullConfig.from_file`; remove the deferred integration skips. Keep the
   cookbook's P1 closure claim qualified until that check passes. BC remains an
   explicit future caller, not a verified workload in this branch.

2. **P3 — Ranked-config comments imply an unperformed measurement.**
   `configs/kaggriculture_2rank.yaml:29` and
   `configs/kaggriculture_4rank.yaml:29` say native threads were “Chosen from the
   measured native step time in Task 6.1.” The cookbook instead correctly says
   `native_threads: 2` is provisional until Task 6.1 measures it, and there is
   no such measurement in this change.

   **Fix:** label the value provisional and say Task 6.1 will measure/verify it.

## Prior findings

- **R1 P1: partially resolved, still blocking.** Production caller, logging and
  non-vacuous invocation/order tests are added; real loading remains impossible.
- **R1 P2: resolved.** Code, tests, architecture docs and the new cookbook
  Reference consistently describe full-padding trunk calls as an upper bound
  and trunk headroom as a lower bound. The 16,384-row/300-token counterexample
  now tests two packed chunks versus the reported upper bound of three.
- **R1 P3: resolved.** The test introduction and cookbook log now name the five
  schema-validated sections and distinguish env/reward key/value assertions.

## Confirmed scope and accuracy

- `configs/scaling_6m.yaml`, `configs/scaling_6m_halfbatch.yaml` and
  `configs/winner_ce_6m_4x5090.yaml` are unchanged from pinned Isaiah `32b3ec9`.
- Both ranked optimizer/PPO configurations equal `scaling_6m`, apart from the
  required per-rank minibatch division: two ranks use 128 envs/spm 8, four use
  64/4, accumulation 1. The unchanged global work is 256 environments,
  16 optimizer steps/iteration, 16 segments/optimizer step and 16,384 game
  transitions/iteration. Tests use Isaiah's real `_minibatch_indices` plus
  literal expected totals and complete parsed-config comparisons.
- Schema 3, independent two-seat rows, hire limit 241, `win_loss` and gamma 1
  match contract v4. Env/reward execution is not qualified by these YAML tests.
- The GPU preset forces FlashAttention; the tiny CPU preset disables it.
  Reusing `configs/model/kaggriculture.yaml` instead of creating the plan's
  `kaggriculture_gpu.yaml` is documented and introduces no behavioral defect.
- The checker uses the model's own chunk functions: 709 padded tokens,
  trunk width 512, 5,915 trunk rows/call and 11,096 head rows/call. Two-rank
  rollout/minibatch/teacher/evaluation seat rows are 256/1,024/16,384/256;
  four-rank rows are 128/512/8,192/128. Workload bounds are consistent with the
  actual chunking implementation. Teacher omission and Orbit bypass are tested.
- No shim wrappers or second trainer were introduced. New behavior extends
  canonical `scripts/run_ppo.py`.
- Cookbook inventory, narrowed coverage and conservative bounds are accurate;
  its disclosed loading/BC/GPU/teacher/replay dependencies remain limitations.
  A claim that all R1 findings are closed is not supported. The historical
  `py-prepare` mypy count of 59 concerns `python/ scripts`; the requested
  `python/owl scripts` command below checks 58 files.

## Executed checks

- `uv run pytest tests/kaggriculture tests/owl tests/scripts tests/tools -m 'not slow' -q`
  — **1,331 passed, 10 skipped, 24.75 s**, exit 0 (`pytest.log`). Orbit Python
  tests remain green. Six skips are the config-integration cases; four concern
  unavailable native grammar, FlashAttention (two), and quantized backend.
- `uv run mypy python/owl scripts`
  — **success, 58 source files**, exit 0 (`mypy.log`).
- `cargo test --locked --offline`
  — initial attempt: **148 passed, 7 failed, 2 ignored**, solely because this
  worktree lacks its ignored generation/replay fixtures (`cargo-test.log`).
  Repeated with the existing sibling worktree fixtures temporarily linked:
  **155 passed, 0 failed, 2 ignored**, including Orbit generation and replay
  parity (`cargo-test-with-fixtures.log`). No parity skips were enabled.
  `fixture-custody.json` records source paths and SHA-256 values; the temporary
  link was removed. This checks the retained fixtures, not their regeneration.
- `git diff --check e1458d2a717d9d731a367cbb78b98616ee6649f4...HEAD` — pass.
- Real-loader probe — five validation errors for each shipped YAML, captured in
  `loader-probe.log`.

## Mutation and custody

On isolated copies of the unchanged runner and its unchanged test file, using
this checkout's real configs and owl package, selected all four new main-path
startup cases. Removed only the `_check_model_workload(...)` call from `main`:

- Baseline: **4 passed, 79 deselected**.
- Mutation: **4 failed, 79 deselected**, each corresponding to the removed
  rejection/logging check.
- Restored: **4 passed, 79 deselected**.

Original/restored SHA-256:
`bc6e40d2c64c4cc45fe38e13793de4676f9ed2f76549619405d78e2a74e82582`.
The copy was restored byte-for-byte and the temporary directory removed.
`mutation.json` and `startup-{baseline,mutated,restored}.log` retain evidence.
The tracked source was never mutated. Before/after hashes of all **766 tracked
files** are identical, HEAD is unchanged, and `git diff HEAD --` is empty
(`before.json`, `after.json`). Only untracked verification evidence was added.

No training, GPU, performance, or playing-strength qualification ran.

VERDICT: REJECT
