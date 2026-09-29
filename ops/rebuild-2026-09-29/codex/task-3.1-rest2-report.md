Implemented Task 3.1 in `kg/rebuild-3-1`. No commits, branches, stashes, resets, cookbook edits, YAML edits, Rust changes, downloads, or GPU runs.

Full evidence: [report](ops/rebuild-2026-09-29/3.1-rest2/results.md), [every verification command and printed result](ops/rebuild-2026-09-29/3.1-rest2/commands.md), [red receipts](ops/rebuild-2026-09-29/3.1-rest2/red.log).

Changed files:

| File | Change |
|---|---|
| `python/owl/train/ppo.py` | Shared schema storage/mapping, contiguous native actions, correct Kaggriculture winner layout. |
| `scripts/run_ppo.py` | Rollout factory, native evaluation, teacher lifecycle, startup guards, W&B flag. |
| `python/owl/train/logging.py` | V3 W&B identifiers and offline mode. |
| `python/owl/train/config.py` | Removed `require_orbit_env`. |
| `python/owl/train/__init__.py` | Removed its export. |
| `scripts/benchmark_checkpoints.py` | Local typed guards retain Orbit-only scope. |
| `tests/owl/train/test_ppo_observation_mapping.py` | Kaggriculture mapping and base-49a4835 Orbit equality checks. |
| `tests/kaggriculture/test_training_smoke.py` | Native two-update smoke and checkpoint round trip. |
| `tests/kaggriculture/test_teacher.py` | Unskipped all four requested tests. |
| `tests/scripts/test_run_ppo.py` | Native evaluation, startup, transport, telemetry and factory tests. |
| `tests/owl/train/test_logging.py` | Fake W&B initialization and metric tests. |
| `tests/kaggriculture/test_configs.py` | Removed obsolete blocker test. |
| `tests/scripts/test_benchmark_checkpoints.py` | Tested retained benchmark guards. |
| `README.md` | Launch and telemetry documentation. |
| `docs/rl-api-specs.md` | Storage, mapping, transfer and seed contracts. |
| `docs/kaggriculture-contract.md` | Shared trainer/evaluation integration. |
| `ops/rebuild-2026-09-29/3.1-rest2/` | Logs, resource measurements, coverage manifest and source hashes. |

Storage and mapping retain schema-field iteration with typed mask/action dispatch. Kaggriculture actions remain int64 with their native shapes and become contiguous CPU tensors before submission. Orbit storage and mapping match the base helpers exactly across all three action families.

Game branches are confined to construction, schema handling, terminal metrics and startup validation; the collector, PPO update and evaluation loops remain shared. Both native factories receive `transfer_device`.

Startup reserves training seeds below `2**62`, including resets, truncation and update overshoot. Excessive step limits fail before allocation. Kaggriculture replay requests fail before allocation with an explicit Task 7.3 error.

W&B uses `kg-v3`, job type/group `ppo`, requested tags and `ppo-{run_dir.name}`. Offline mode retains local telemetry; resume plus offline fails explicitly. Orbit’s default initialization arguments remain unchanged.

Verification:

| Check | Printed result |
|---|---|
| Isaiah suite | **1276 passed, 3 skipped in 41.28s** |
| Kaggriculture bounded coverage | **1092 passed, 3 skipped**, all **1095** collected IDs covered exactly once |
| Final mapping suite | **191 passed in 0.21s** |
| Final runner suite | **123 passed in 0.67s** |
| W&B tests | **10 passed in 0.04s** |
| Benchmark/config tests | **88 passed in 1.47s** |
| Native smoke | **1 passed in 0.42s** |
| Formatting, lint, syntax, mypy | Passed; mypy checked **70 source files** |
| Docs freshness, Markdown lint, diff check | Passed without `DOCS_CURRENT` |

Total verified coverage: **2368 passed, 6 skipped**, matching preparation’s **2374 collected cases**. The [coverage manifest](ops/rebuild-2026-09-29/3.1-rest2/kg-coverage-manifest.json) records every bounded command, result and node ID.

The smoke’s complete subprocess took **4.603 seconds**, with **342,327,296 bytes** sampled watchdog-plus-pytest RSS. It performed two updates using one native environment.

`just py-prepare` was run because no Rust changed. Its static phases passed, but monolithic pytest hit the RSS watchdog at **944,816,128 bytes**. The standalone Kaggriculture suite also hit the watchdog. Complete bounded replacements passed; **the monolithic commands are not reported as passing**.

Changed test intent is limited to obsolete stops: evaluation now executes native games, startup tests reach allocation, and the removed shared blocker test became benchmark-specific guard coverage. Teacher-test intent is unchanged.

Remaining qualification gaps: Task 7.3 replay export, live W&B synchronization, CUDA DMA/compiled execution and real multi-rank training. Model documentation was left unchanged under the model-surface restriction; its legacy startup-status sentence still needs reconciliation. Cookbook records remain with Claude.

VERDICT: IMPLEMENTED