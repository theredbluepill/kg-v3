# Task 3.1 remaining trainer seam

Implemented in worktree `kg/rebuild-3-1`, unchanged HEAD
`49a48350cdd72d6b546320441d677fbcbae2d893`. No commit, branch, stash, reset,
cookbook edit, config YAML edit, Rust edit, model implementation edit, download,
GPU run or long training run. Claude owns review, commit and cookbook records.

The target was native Kaggriculture transport through the existing collector,
rollout storage, PPO update, teacher lifecycle and evaluation. Completion was
bounded native execution, exact Orbit tensor equality, all requested test cases,
static checks and documentation freshness. This is execution evidence, not
learning-quality or throughput qualification.

## Changed-file inventory

- `python/owl/train/ppo.py`: typed game unions in the existing storage and mapping helpers; native observation-shape metadata, exact int64 action storage, contiguous CPU transport, actor-mask metrics and preserved per-seat winner layout.
- `scripts/run_ppo.py`: game-dispatched rollout factory, shared observation mapper, native policy evaluation and raw-bank metrics, teacher load/refresh, startup seed budget/replay rejection and W&B mode CLI.
- `python/owl/train/logging.py`: v3 W&B project/identifiers and explicit offline support, preserving Orbit's default init kwargs.
- `python/owl/train/config.py`: removed obsolete `require_orbit_env` blocker.
- `python/owl/train/__init__.py`: removed that obsolete export.
- `scripts/benchmark_checkpoints.py`: local typed guards retain its Orbit-only scope after removing the shared blocker.
- `tests/owl/train/test_ppo_observation_mapping.py`: Kaggriculture schema/transport tests and exact base-49a4835 Orbit storage/action oracle.
- `tests/kaggriculture/test_training_smoke.py`: native two-update CPU smoke and checkpoint round trip.
- `tests/kaggriculture/test_teacher.py`: four requested tests unskipped; env patch point moved to `create_env`.
- `tests/scripts/test_run_ppo.py`: native policy evaluation, launcher/factory/seed/replay/mode/metric tests, seat-wise action selection and preserved nonblocking CUDA transfer.
- `tests/owl/train/test_logging.py`: fake W&B init, identity, mode, resume and metric tests; no network.
- `tests/kaggriculture/test_configs.py`: removed the obsolete trainer-blocker helper test.
- `tests/scripts/test_benchmark_checkpoints.py`: retained Orbit-only boundary tested at both benchmark sites.
- `README.md`: launch, W&B mode, replay restriction and seed-budget documentation.
- `docs/rl-api-specs.md`: shared transport/storage/fence and startup contracts.
- `docs/kaggriculture-contract.md`: canonical trainer/evaluation integration.
- `ops/rebuild-2026-09-29/3.1-rest2/`: red/green receipts, resource monitors, exact case inventory, commands and source fingerprints.

## Design choices

Observation mapping retains `type(obs).model_fields`, rebuilding the same
schema. Masks/actions dispatch by explicit types. Kaggriculture rollout storage
uses its native allocator for one-row field metadata and allocates final tensors
on the rollout device; actions remain int64 `[T,E,2,252,12]` and `[T,E,2]`, with
bool masks `[T,E,2,252]`. Native input transport materializes contiguous CPU
tensors without silently changing dtype. Orbit retains its allocation
expressions, output tensor bytes, losses and default logger init arguments.
The existing factory's `BaseModelAPI[Any, Any, Any]` boundary typing is reused;
observations, actions and masks retain concrete typed unions. No model API changed.

`run_ppo` branches only where game I/O requires it: construction, action schema,
terminal scalar extraction and startup admission. It has one evaluation loop
and one PPO loop. The script-local observation mapper is removed. CUDA
evaluation still passes `non_blocking=True`, as Orbit previously did. Both
native rollout and evaluation factories receive `transfer_device` for the
adapter entry fence. Rollout uses `cfg.env.seed`, distributed rank/world;
evaluation uses `_evaluation_seed`, rank 0/world 1. Winners use terminal raw
banks, never shaped returns or autoreset observations.

Startup requires Kaggriculture base seed in `[0,2**61)` for evaluation input.
For global width G and update steps B, the step ceiling is
`min((2**62 - seed - 2*G)//2, 2**61 - 1) - (B - 1)`.
This reserves construction/reset plus both a reset and a truncation per
transition and one final update overshoot. Too-large explicit limits reject
before allocation; an unspecified limit receives the conservative ceiling.
This also keeps evaluation counters in range. Orbit limits remain unchanged.

Kaggriculture replay requests greater than zero fail before any run directory,
env or model, naming Task 7.3. No Task 7.3 implementation was copied.

Kaggriculture W&B uses project `kg-v3`, job type/group `ppo`, tags
`[kaggriculture-v3, ppo]`, and name `ppo-{run_dir.name}`. Online is default;
explicit offline mode stores in the run directory for later `wandb sync`.
There is no silent online-failure fallback. Resume plus offline is rejected
before reading the checkpoint. Training/evaluation use the existing log calls.

## Checks and resources

All commands used `CARGO_BUILD_JOBS=2 OMP_NUM_THREADS=2`.
[commands.md](commands.md) records the verification commands, printed summaries
and their receipts. [red.log](red.log) preserves the six test-first failure
receipts. [source-sha256.json](source-sha256.json) binds the final 16 changed
source/test/doc files to this evidence.

- Isaiah's requested suite: **1276 passed, 3 skipped in 41.28s**.
- Kaggriculture: **1092 passed, 3 skipped** across 15 bounded batches. Every
  one of the **1095** collected node IDs occurs exactly once, with no missing,
  extra or duplicate IDs; see [kg-coverage-manifest.json](kg-coverage-manifest.json)
  and [kg-batches-summary.md](kg-batches-summary.md).
- Combined coverage: **2368 passed, 6 skipped**, matching the **2374** cases
  collected by preparation. Existing skips are CUDA/FlashAttention, pinned
  memory and unavailable quantized backend; none is a trainer-seam skip.
- Orbit mapping final: **191 passed in 0.21s**. The base-49a4835 oracle covers
  three action families, three seeds and simple/cross-attention observations,
  comparing initial storage, seeded copies, index/segment/flatten/CPU mappings
  for exact values, dtypes and shapes. It requires that commit in local history.
- Runner final: **123 passed in 0.67s**. W&B tests: **10 passed in 0.04s**.
- All four teacher tests run; the original focused/full receipts and the final
  bounded **50 + 1 + 1** teacher coverage are retained.
- Native smoke: **1 passed in 0.42s**; complete child wall time
  **4.6032227909890935 seconds**; sampled watchdog+pytest peak RSS
  **342327296 bytes**; child ru_maxrss **325681152 bytes**. The uv launcher is
  excluded, and this smoke creates no descendants. `/usr/bin/time -l` could
  not report RSS because the sandbox denied sysctl; its failed measurement
  attempt remains visible beside the working known-PID monitor.
- `just py-prepare` was run because no Rust changed. Formatting, lint, Python
  3.11 syntax and mypy (**70 source files**) pass. Its monolithic pytest phase
  stopped at the RSS watchdog (**944816128 bytes**, exit 86), before the
  unchanged typing probe finished. The exact standalone Kaggriculture command
  likewise stopped (**944144384 bytes**), and a teacher batch stopped at
  **952156160 bytes**. These are incomplete commands, not passes.
- Complete bounded replacements retain test bodies and parameters. The pure
  mypy test alone uses `--noconftest` to avoid importing unrelated Torch fixtures;
  all other batches retain normal fixtures. Largest successful pytest RSS is
  **814465024 bytes**. Resource scope excludes descendant processes.
- `just docs-fresh` passes without `DOCS_CURRENT`; Markdown lint and
  `git diff --check` pass. No Rust check was run.

## Changed test intent and remaining gaps

Only tests of removed blockers changed intent: the evaluation-stop test now
runs a real tiny policy/native evaluation; `_NOT_WIRED` startup cases now verify
that checked configs reach allocation; the removed shared-helper test is
replaced by the benchmark's actual Orbit-only guards. Teacher tests retain their
original intent. New-test expectation fixes use the existing metric names and
engine terminal-step semantics rather than changing production behavior.

No open implementation question remains in this task. Unqualified: live W&B
upload/sync, pinned CUDA DMA, compiled/BF16 production execution, real multi-rank
training and learning strength. Task 7.3 replay export remains separate.
Monolithic preparation is not claimed to pass under this Mac's resource cap;
the complete test-case coverage above is process-isolated evidence.
`docs/model-architecture.md` was reviewed but left unchanged because the model
surface did not change, as requested; its legacy Task 3.1 startup-status sentence
still needs reconciliation by the documentation owner. Cookbook records remain
for Claude, per the task instruction.
