# Independent verification — rebuild Task 3.4

Reviewed `e1458d2a717d9d731a367cbb78b98616ee6649f4...HEAD` on
`kg/rebuild-configs`, HEAD `18a28afa9342b4e5b993a4eec02d91e81a8e6461`.
The merge base equals the requested base. Scope: plan Task 3.4, contract v4,
Isaiah's pinned `32b3ec9` recipe/code, changed tests and cookbook claims.
No tracked modification remains. Evidence in this directory is untracked.

## Findings

1. **P1 — Required startup assertion is not integrated.**
   `python/owl/model/kaggriculture_workload.py:120` defines the checker, but
   repository searches find no production import/call of it or its logging
   helper. `scripts/run_ppo.py:130` still loads through `FullConfig` without
   the check. All three new YAML files fail that loader with five schema
   errors each, before a Kaggriculture workload can be checked. The load/model
   test at `tests/kaggriculture/test_configs.py:290` and the training-config
   glob skips hide that integration gap from the green suite. The Reference
   openly defers it at line 49, but Task 3.4 explicitly requires the assertion
   at config load and headroom in the run log. No BC startup is covered either.
   **Fix:** finish the required game/config dependency and invoke check/log on
   the canonical startup path before model allocation/compilation. Add an
   unskipped integration test proving an unserviceable workload fails at
   startup and valid configs emit headroom; account for BC when its caller
   exists. The helper alone does not complete Task 3.4.

2. **P2 — Trunk-call and headroom bounds are described incorrectly.**
   `python/owl/model/kaggriculture_workload.py:126` says reported calls are
   the calls the model will issue. The packed path instead uses actual token
   counts (`python/owl/model/kaggriculture.py:484`): an arithmetic probe with
   16,384 rows of 300 packed tokens gives **2** trunk chunks, while the new
   report says **3**. `cookbook/references/kaggriculture-configs-follow-isaiahs-scaling-6m-recipe.md:50`
   calls headroom an upper bound; full padded sizing gives a conservative
   **lower bound on headroom** and an **upper bound on trunk calls**.
   **Fix:** label the padded worst-case assumptions in the report/log and
   reconcile module, architecture document, cookbook table and log wording.
   The row counts and full-padded chunk arithmetic themselves are correct.

3. **P3 — Cookbook overstates schema coverage.**
   `cookbook/log.md:5` says tests validate each config section against its
   schema; `tests/kaggriculture/test_configs.py:5` makes the same broad claim.
   `_sections` validates observation, action, model, optimizer and PPO schemas,
   but env/reward sections only receive key/value assertions. The Reference
   correctly acknowledges this at line 48.
   **Fix:** narrow the log/test introduction to the schemas actually checked,
   and describe the env/reward assertions separately until integration exists.

## Confirmed behavior

- Worktree `configs/scaling_6m.yaml` is byte-identical to Isaiah `32b3ec9`:
  SHA-256 `8352461d4d9c93b8deacc25ef570a6002d2ca920606d948643b3921a9e2cc377`.
- Every parsed optimizer and PPO field of both ranked configs equals
  `scaling_6m`, except the required per-rank `segments_per_minibatch` division.
  Environments/minibatch segments are 128/8 at two ranks and 64/4 at four;
  accumulation is 1. Teacher chunk size remains 128, as intended. Both retain
  the full optimizer/scheduler, coefficients, compile defaults, BF16 and 20M
  checkpoint cadence.
- Global workload is exactly **256 environments, 16 optimizer steps per
  iteration, 16 segments per optimizer step, 16,384 transitions per iteration**.
  Tests use Isaiah's real `_minibatch_indices`, divisibility validation,
  literal expected totals and complete parsed-config comparisons.
- Contract v4's schema version 3, two-seat rows, hire limit 241, `win_loss`
  reward and gamma 1 are respected by these config values. Reward execution
  and full env validation remain outside this branch's implemented path.
- Existing GPU preset forces FlashAttention; the separately named CPU preset
  disables it. Reusing the existing GPU preset path instead of renaming it is
  documented. No shim wrappers or second trainer were introduced.
- The helper imports the model's sizing functions. Changing the model's
  element limit in an isolated process changes its reported capacities to the
  same model-derived values. Separate trunk- and head-limited configurations
  raise informative `ValueError`s without allocating a model. The patched
  process constant was restored. These are helper checks, not startup checks.
- At the full 709-token bound, capacities are 5,915 trunk rows and 11,096 head
  rows. Two-rank rollout/minibatch/teacher/evaluation rows are
  256/1,024/16,384/256; four-rank rows are 128/512/8,192/128.

## Executed checks and mutation

- Requested command:
  `uv run pytest tests/kaggriculture tests/owl tests/scripts tests/tools -m 'not slow' -q`
  — **1,324 passed, 10 skipped, 26.85 s**, exit 0 (`pytest.log`). Isaiah's
  included Orbit tests remain green. Six skips are the new Kaggriculture config
  integration cases; the other four cover native grammar availability,
  FlashAttention (two) and the local quantized backend.
- Requested command: `uv run mypy python/owl scripts`
  — **success, 58 source files**, exit 0 (`mypy.log`).
- Mutation: changed only 2-rank `n_envs: 128` to `256`; targeted workload
  tests produced **1 failed, 1 passed, 20 deselected**, exit 1. The failure is
  the intended global workload assertion. Restored the exact original bytes;
  before/after SHA-256 is
  `99ff77a0b32cb5324cd7f91ac8f1cea01b9c77b8f9d4d1ab7e9d12304539dca9`
  (`mutation.json`, `mutation.log`).
- After restoration: `uv run pytest tests/kaggriculture/test_configs.py -q`
  — **19 passed, 3 skipped**, exit 0 (`configs-restored.log`).
- Startup schema failures, both loud guard failures, model-constant sensitivity
  and the packed-count counterexample are in `guard-and-startup.json`.
- `git diff --exit-code` and `git diff --cached --exit-code` both pass;
  identity/source checksum is in `identity.json`.

No GPU, training, Rust, or `just py-prepare` run was performed for this review.
The historical cookbook test totals were reproduced; this does not reproduce
every historical preparation check or establish GPU qualification.

VERDICT: REJECT
