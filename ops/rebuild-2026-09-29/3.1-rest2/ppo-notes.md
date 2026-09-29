# Shared PPO schema and native smoke checks

Target: transport native Kaggriculture observations/actions through Isaiah's canonical rollout and PPO update, retaining base-49a4835 Orbit tensor results. Stop at schema equality checks and two CPU no-teacher updates. No learning-quality, GPU, DMA or multi-rank claim.

Implementation: schema-field iteration stays canonical; mask/action helpers dispatch by explicit types. Native Kaggriculture allocation supplies one-row field metadata, expanded directly to the rollout device; Orbit retains its allocation expressions. Game-specific common dimensions are two seats and 252 frames. CPU action transport materializes contiguous native int64 inputs without silent dtype coercion. The trainer uses Kaggriculture actor_mask for the max-entities metric and preserves its per-seat categorical winner layout; Orbit's existing winner view remains unchanged. Model boundary typing matches the existing factory's BaseModelAPI[Any, Any, Any] while observations, actions and masks use typed unions. No model implementation or model API changed.

## Commands and receipts

All commands used `CARGO_BUILD_JOBS=2 OMP_NUM_THREADS=2`. Commands that run uv used `--offline`.

- `uv run --offline pytest tests/owl/train/test_ppo_observation_mapping.py tests/kaggriculture/test_training_smoke.py -q`: red before production changes, **3 failed, 189 passed in 0.35s** (`ppo-red.log`). Failures: native storage, action CPU transfer, native trainer initialization. Base Orbit equality cases passed.
- Same command after first edits: **1 failed, 191 passed in 0.35s** (`ppo-green-first.log`); the smoke reached update and exposed an unconditional Orbit-shaped winner-log-probability view. The repair narrows that view to Orbit.
- Smoke targeted first follow-up: **1 failed in 0.14s** (`smoke-first.log`), an incorrect new-test metric name (`loss/total`; actual existing key `loss/total_loss`). The next combined follow-up (**1 failed, 191 passed in 0.98s**) corrected the new-test expectation from one to two games: `episodeSteps=3` finishes at step 2. It made no production change. That intermediate combined log was subsequently superseded by the green receipt.
- Same combined command final: **192 passed in 0.65s** (`ppo-green.log`).
- `uv run --offline mypy python/owl/train/ppo.py`: **Success: no issues found in 1 source file** (`ppo-mypy.log`).
- `uvx --offline ruff check ... --select I --fix`, `uvx --offline ruff format ...`, `uvx --offline ruff check ...` on the three owned files: imports/formatted; final **All checks passed!** Two compound test assertions found on the first lint pass were split.
- `/usr/bin/time -l uv run --offline pytest tests/kaggriculture/test_training_smoke.py::test_no_teacher_two_updates -q`: **1 passed in 0.28s**, **3.52 real**, but time exits 1 because sandbox rejects `sysctl kern.clockrate`. No RSS from that command (`smoke-measured.log`).
- `uv run --offline python ops/rebuild-2026-09-29/3.1-rest2/measure_smoke.py`: **1 passed in 0.42s** (`smoke-watchdog.log`), full child process wall **4.6032227909890935 seconds**, sampled watchdog+pytest peak RSS **342327296 bytes**, child `ru_maxrss` **325681152 bytes**, exit 0; limits **115 seconds / 960 MiB** (`smoke-resources.json`). The sandbox also denies process enumeration, so the harness samples known parent/child PIDs; this smoke creates no descendants. The uv launcher is excluded. Initial enumeration attempt failed before a measured receipt; the direct PID version is the retained script.

The native smoke uses one env, exact tiny CPU FP32 model, horizon 2, one PPO epoch/minibatch/accumulation step, two updates, native_threads 1, no pinning/compile/teacher. It checks finite metrics/losses, raw banks and margin, two completed native games, eight player turns, two optimizer steps, active-entity bounds, no replay-alarm trip, statelessness and model/checkpoint/counter round trip.

The base-49a4835 equality fixture reads the base module using `git show`; no git mutation. It requires that base commit in local history and tests all three action families, three seeds and both simple/cross-attention observations. Existing mapping tests also retain the older successful-path Isaiah oracle and the independent synthetic schema tests.
