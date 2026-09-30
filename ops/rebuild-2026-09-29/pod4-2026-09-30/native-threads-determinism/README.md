# native_threads determinism check (Mac, CPU)

**Question.** Does `env.native_threads` change any native Kaggriculture output?
If it does not, the 4-rank native_threads sweep is only a throughput choice.

**Result.** At the tested seed and settings, the outputs were byte-identical
for native_threads 1, 2, 4 and 8. Every step matched on all 35 caller-owned
buffers (29 observation buffers plus rewards, dones and the four transition
bank/econ buffers), the step metrics dict, `terminal_metrics(i)`,
`state_snapshot(i)` and `seed_state()`. No first difference was found. The
final buffer SHA-256 was `4e12825b…95dd1` for all four thread counts. The
serial seed+1 negative control was detected. Its `seed_state()` differed from
construction, and its buffers first differed at step 23 (`tile_kind`), so the
comparator can see real divergence. See `result.json`.

## Inputs and code path

- Source: `/Users/poonszesen/kg-v3-int` at `7e87f5420b3696141ea41f6453d4fafb2748eea0`,
  which is the parent of this branch's pod receipt commit. The worktree was used
  read-only with its prebuilt `python/owl/rs.abi3.so` (sha256
  `927a92f11f5c45b778f5f6d99991c8fab8f1cb4f7ac33f24ac8ebac8bed096cf`, built
  after the last `src/` commit, with no newer `.rs` files).
- Host: Apple M5 (10 CPUs), Python 3.12.13, torch 2.9.0, CPU only. The run took
  about 5 s wall time and 351 MB max RSS.
- Construction goes through `owl.game.create_env` with a `KaggricultureEnvConfig`
  (the Task 1.5 adapter factory): n_envs 8, base_seed 20260930, rank 1 of
  world 4, the 4-rank recipe's reward coefficients, pin_memory off, and
  `episodeSteps` 60 so that games finish and auto-reset in the window.
- Actions come from a fixed-seed tiny `KaggricultureTransformer` (embed 16,
  depth 1) that samples grammar programs from the reference env's observation.
  The identical tokens and lengths are fed to every env. Over the run, 2993
  seat-rows sampled a program longer than PASS+STOP.
- The run made 200 steps, with one `reset()` before stepping and one
  `truncate_envs` (envs 0, 3, 6) after step 97. It saw 42 done flags and 21
  terminal records.

Command, run from `/Users/poonszesen/kg-v3-int`:

```
.venv/bin/python <this dir>/check_native_threads_determinism.py \
  --steps 200 --threads 1 2 4 8 --out <this dir>/result.json
```

## Limits

- This is one seed, one host (macOS arm64) and n_envs 8. The pod's Linux build
  and its n_envs 64 per rank were not run here. Rayon's per-env rows are written
  to disjoint slices by design (`src/kaggriculture/env.rs`, `buffers.rs`), and
  that design is consistent with this result. The result does not prove the
  pod binary is identical.
- The PPO learner was not run. This covers only the env I/O boundary.
- Side observation: with seed+1, the construction observations and state
  snapshots were identical, and only `seed_state()` differed until step 23.
  Initial layouts appear not to depend on the seed. This check did not
  investigate that further.
