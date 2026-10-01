I traced one rollout step from code in /Users/poonszesen/kg-v3-h200gate at b2276bc5 and cross-checked it against the live run's own probe log. The rollout is CPU-bound. About 60% of it is the native env step, and much of that runs serially on the main thread. Most of the rest is Python dispatching about 600 eager GPU ops per forward, plus 6 host-device syncs per step. The GPU is almost idle. No existing config knob overlaps env stepping with the model forward or compiles the action heads. Nothing was modified. The only thing I ran was one CPU forward of the model on the Mac to count ops (see step 2).

## Live-run numbers (pod log, read-only)
Source: [nt-probe] "iteration" records in `/root/runs/earn720-r30e01w30-8xh200-from-60M-20261001.log`, iterations 35-52.
- `time/rollout_seconds` is 26.98-27.8 s. It is the maximum over ranks (`ppo.py:1086-1089`).
- Native `KaggricultureVectorizedEnv.step` time per rank per iteration (720 calls) is 11.97-18.81 s, mean about 16 s. That is about 22 ms per step, or about 1.1 ms per env-step (range 0.83-1.31).
- Everything else in the rollout (forward, copies, syncs) is at most 27.5 minus the rank's native time: about 9-15 s, or up to about 12.5 ms per step.
- Ranks differ by up to about 30% in native time within one iteration. The slowest rank sets the pace.
- Old pod, same reward recipe, 12 envs per rank (`/Users/poonszesen/kg-v3-runs/pod-abl4mvr5w1mmn4-archive/runs/earn720-r30e01w30-from-c50-4rank-20260930.log`, iterations 348-355):
  - native 6.25-7.35 s per iteration, 0.72-0.85 ms per env-step;
  - rollout 11.6-11.9 s, so "other" is at most about 6.2 ms per step.
- Old pod at 6 and 16 envs: "other" is at most about 6.3-6.7 ms per step. It is roughly fixed per step and does not grow with the env count.
- So the H200 host is about 1.4x slower per env in native code and up to about 2x slower in Python/launch work. Busy cores sampled at 3.0-3.2 GHz under the schedutil governor. The start checkpoints differ (60M vs c50), so game content may also differ.
- Per-rank thread CPU (ps, lifetime averages): main thread about 63%, rayon workers 22-29% each.

## Per-step trace, in order (self-play; `_learner_host` is None)
Markers: [SYNC] is a host-device synchronisation. "Per-env loop" means a loop over envs.

**0. Loop.** `python/owl/train/ppo.py:1269`, `for step in range(horizon)` inside `torch.no_grad`. horizon = 720, which is one full game, so all 20 envs reset together at the game's last step.

**1. Model forward.** `ppo.py:1271-1275`: under bf16 autocast, `_model_forward` (`ppo.py:3293`) calls `DistributedModelAdapter.forward` (`python/owl/train/distributed.py:303`). DDP is built with `broadcast_buffers=False` (`distributed.py:288-294`), so it adds no collective. It then calls `KaggricultureTransformer.forward` (`python/owl/model/kaggriculture.py:637`). The batch is 20 envs x 2 seats = 40 seat rows, 709 padded tokens per row.
- **Stems, eager** (`kaggriculture.py:449-540`): one_hot and cat, 6 Linears, expanded tokens, token-mask cat.
- **Trunk dispatch** (`_run_trunk`, `kaggriculture.py:542`): flash-attn 2.8.3 is installed and the input is bf16, so the packed path is always taken (`kaggriculture.py:576-600`). This happens whatever `force_flash_attn` says; the flag only turns a missing flash-attn into an error.
- `build_packed_sequence` (`python/owl/model/stateless_transformer_v1.py:3175`) does three syncs per forward:
  - `nonzero` at `:3182` [SYNC];
  - `if not seqlens.gt(0).all()` at `:3184` [SYNC];
  - `seqlens.max().item()` at `:3186` [SYNC].
- 40 x 709 x 512 is below 2^31, so there is one chunk and no `tolist`.
- **Compiled region:** only the 8 blocks plus the final norm (`compile_transformer_trunk`, `kaggriculture.py:612-629`, `torch.compile(dynamic=True)`, mode `max-autotune-no-cudagraphs`, GEMMs cuBLAS only). Unpacking uses zeros plus index_put.
- **Heads, eager.** `_policy` (`kaggriculture.py:1017`) runs one chunk, because `head_rows_per_chunk` is about 11k rows (`:241`). The core is `self._compiled_actor_core or self.actor.policy_core` (`:1088`). `_compiled_actor_core` is initialised to None at `:320` and nothing ever sets it, so the heads are always eager.
- **Within-turn decoding** (`python/owl/model/kaggriculture_actor.py:263-502`):
  - Frames are batched: all 241 unit frames (padded, whatever the real actor count) and all 11 market positions are decided in parallel.
  - The fixed slot stages run one after another: 5 unit stages (`:311-354`), then market kind with the HIRE-capacity correction (`:357-409`), then 3 market stages (`:410-440`). That is 9 sequential head calls per forward.
  - Each stage does: mask gather, OutputProjectionMLP (Linear-GELU-Linear), Gumbel noise, argmax, masked log_softmax and entropy, and a prefix-embedding add.
  - There is no `.item()`, `.cpu()` or sync anywhere in the heads.
  - Token, log-prob and entropy assembly uses stack, pad and scatter (`:442-483`).
- **Values:** critic head plus a masked log_softmax (`kaggriculture.py:1170-1238`).
- **Measured op counts:** a CPU forward of the real model config (40 rows) under `torch.profiler` counted these top-level aten dispatches: stems 84, eager trunk 177 (replaced by Inductor kernels in production), policy/heads 491, outputs/values 33. So about 610 eager ops per step outside the trunk. Script: `/private/tmp/claude-501/-Users-poonszesen-kg-v3/3f7ce81c-f27d-4f24-bb54-c9e0de2d998a/scratchpad/opcount/count.py`.

**2. Actions device to host.** `_step_env` (`ppo.py:2847`) calls `_actions_to_cpu` (`ppo.py:2817-2832`).
- `tokens` [20,2,252,12] int64 (0.97 MB) goes `.to("cpu")` into fresh pageable memory [SYNC].
- `lengths` [20,2] does the same [SYNC].
- Then `.contiguous()`.

**3. Env step (Python).** `python/owl/kaggriculture/env.py:441-525`.
- `_fence()` at `env.py:394-396` calls `torch.cuda.current_stream().synchronize()` [SYNC]. It protects the pinned buffers the previous step copied from.
- Input checks, then one PyO3 call with 37 numpy views (`env.py:454-492`).
- **Per-step telemetry on the main thread** (`env.py:499-524`): `bank_rewards` and `margin_rewards` run CPU torch float64 ops plus `isfinite().all()` validation (`python/owl/kaggriculture/rewards.py:178-233`). Both terms are on in this run, so this is about 30 tiny CPU ops per step.

**4. Native step (Rust).** Binding at `src/kaggriculture/bindings.rs:722-885`.
- `prepare_step` runs under `native_work`, which releases the GIL (`bindings.rs:403-409, 853`). Inside `src/kaggriculture/env.rs:512` onward:
  - **Serial on the main thread, per-env loops:**
    - raw token admission over all 120,960 tokens (`env.rs:~531-568`);
    - terminal prediction (`:570`);
    - `grammar::plan` plus `grammar::decode` for each env and seat into `serde_json` Values (`:579-596`); the JSON uses `preserve_order` and `arbitrary_precision` (Cargo.toml);
    - a clone of every env's Game (`:598`);
    - allocating and zero-filling a fresh `ObsStaging::new(n)`, about 5.6 MB per step (`:599`, `buffers.rs:361-371, 1022`).
  - **Parallel:** `pool.install(into_par_iter)` over envs, with `native_threads` = 4 rayon threads (`env.rs:601-752`). Per env it does:
    - `step_with_market_metrics`, which clones the Game again internally (`engine_rs/src/lib.rs:1446`);
    - rewards;
    - on auto-reset, a new game from its seed;
    - `game.prepare()` (a snapshot that clones state, `observe.rs`);
    - `write_env` into staging (`env.rs:707-708`).
  - **Serial again:** result collection and the transition cache (`env.rs:754-788`).
- Python then builds the metrics dict while holding the GIL (`bindings.rs:856-881`).
- `commit` runs under `py.detach` (`bindings.rs:883`, `env.rs:791-827`): a serial `copy_from_slice` of all 29 fields (5.6 MB) into the pinned numpy buffers (`buffers.rs:1098+`), then slot swaps. Dropping the old games and staging buffers also happens on the main thread.
- The earlier sweep's `result.md` already read this code as mostly serial. The per-thread CPU split above agrees roughly, but the serial share has never been profiled.

**5. Rewards and dones host to device.** `ppo.py:1295-1302`, `non_blocking` from pinned memory. No sync.

**6. Rollout buffer write.** `write_step` (`ppo.py:669-716`) does about 45 device-to-device copies:
- 29 observation fields through a Python loop over fields, not envs (`_copy_observation_`, `ppo.py:2679-2718`);
- tokens and lengths, logp, entity_logp, values, rewards, dones;
- zero fills for truncated and bootstrap values;
- value_offsets.
- The on-device rollout observations total 4.06 GB per rank (282,246 B per env-step x 20 x 720).

**7. Hidden state.** `reset_hidden_state` at `ppo.py:1334` does nothing (the model is stateless).

**8. Next observation host to device.** `_copy_obs_to_device_` (`ppo.py:1339`, `2801-2814`): 29 `non_blocking` copies from pinned memory, 5.6 MB per step.

**9. Metrics.** `_extend_env_metrics` (`ppo.py:1291`, `2860`) extends Python lists, one per metric key.

**After the loop.** `_model_compute_value` for the last values (`ppo.py:1344-1349`) adds 3 more packing syncs once per rollout.

**Totals per step:**
- 6 host-device syncs: 3 in packing, 2 blocking device-to-host copies, 1 fence.
- No per-env Python loop.
- Serial per-env loops in Rust on the main thread: token admission, decode, Game clone, staging fill, publish.
- The host is serialised: forward, then device-to-host, then native step, then copies. Nothing overlaps.

## Existing knobs that affect rollout speed (effective `config.yaml` of the live run)
- **`env.n_envs = 20`** (preset 64). This is the main existing lever: the fixed per-step overhead is spread over more envs. Nothing is split or pipelined.
- **`env.native_threads = 4`.** Only sizes the rayon pool (`env.rs:287`).
- **`env.pin_memory = true`.** Makes host-to-device copies `non_blocking` and turns on the fence (`ppo.py:797-800`).
- **`env.opponent_mix`**: none (self-play). With a mix, `forward_learner_rows` skips scripted rows.
- **`rl.model_compile = trunk`.** Options are none, mlp or trunk. There is no option to compile the heads or stems.
- **`rl.model_compile_mode = max-autotune-no-cudagraphs`.** Options: default, reduce-overhead and max-autotune (both of the latter turn on CUDA graphs), max-autotune-no-cudagraphs.
  - The trunk compiles with `dynamic=True`, and the packed token count changes every step. A CUDA-graph mode is therefore untested and risky, and the trunk is not the bottleneck anyway.
- **`rl.compile_mode = default`.** Compiles only GAE and the PPO loss (`ppo.py:778-779, 2038`), which is update-side.
- **`rl.dtype = bfloat16`.** Autocast.
- **`model.force_flash_attn = true`.** Does not select the path (see step 1).
- **`rl.horizon = 720`, `rl.segments_per_minibatch = 1`, `gradient_accumulation_steps = 1`, `ppo_epochs = 1`.** These drive update time: about 3.05-4.14 s per iteration, 20 optimizer steps. n_envs must be divisible by `segments_per_minibatch * gradient_accumulation_steps` (`ppo.py:3727`).
- **`rl.teacher_segments_per_minibatch = 128`.** The teacher precompute (`ppo.py:1487-1504`) runs all 20 envs, 28,800 rows, in one chunk and takes 0.81 s. I suspect this sets the memory ceiling on n_envs (see Prior evidence).
- **`rl.teacher_kl_coef` / `rl.teacher_value_coef` = 0.005, `teacher_mode = last_best`.** Setting both to 0 skips the teacher pass, but that changes the recipe.
- **`rl.truncation_*` and `rl.initial_stagger`: off.** Turning them on would add per-step syncs (`ppo.py:1380, 1427`).
- **`rl.checkpoint_freq = 10,000,000`.** That is about every 87 iterations. The last-best evaluation (20 games) runs on rank 0 only while the other 7 ranks wait at `broadcast_object` (`scripts/run_ppo.py:652-679`). `rl.eval_replay_games = 0`.
- **Environment and launch:** `OMP_NUM_THREADS=1`, `KG_NT_NUMA=cpu`. The wrapper binds each rank to its GPU's 24-CPU node, 2 ranks per node (`main_probe_auto.py`, receipts `topology.json`). One rank per GPU is hard-wired (`device = cuda:local_rank`).
- **There is no knob for:** async or overlapped env/model execution, env groups or double buffering, CUDA graphs for the heads, or pinned device-to-host action buffers. A grep found none.

## Prior evidence
**native_threads sweep, 4-rank recipe J, 64 envs per rank, old pod.** `/Users/poonszesen/kg-v3-h200gate/ops/rebuild-2026-09-29/pod4-2026-09-30/native-threads-sweep/result.md` (plus `run-statement.md` and `pod-receipts/summary.json`):

| N | Game SPS (two-sample mean unless noted) |
|---|---|
| 2 | 3,093 |
| 4 | 3,256 |
| 8 | 3,104 |
| 16 | 2,960 (one sample) |
| 4, CPU-bound | 3,450 (+6%, one sample) |
| 8, CPU-bound | 3,385 (one sample) |

- Native step time stayed flat at about 32-46 ms per step for every N.
- Cores used per rank: 1.15 at N=2 up to about 2.5-2.9 at N=16.
- Rollout was about 80% native.
- Its conclusion: "The real lever for rollout time is that serial path, not `native_threads`". The serial share is unprofiled, and no nsys trace was taken.
- Memory binding failed with EPERM on `set_mempolicy`.

**Model-only SPS ceiling.** `/Users/poonszesen/kg-v3-h200gate/cookbook/references/model-only-sps-ceiling-bounds-per-rank-throughput.md` and `ops/rebuild-2026-09-29/model-sps-ceiling-2026-09-29/`.
- Component bound per rank, synthetic schedule: 2,085 / 1,757 / 957 steps/s (sparse / mid / dense).
- Engine budget for at most 10% loss: 53-116 µs per env-step. The live native cost is about 1,100 µs per env-step, roughly 10-20x over that budget.

**cuBLAS-only (ATEN) GEMM A/B.** `/Users/poonszesen/kg-v3-h200gate/ops/rebuild-2026-09-29/results.md:259-303` and `cookbook/decisions/kaggriculture-compiles-gemms-with-cublas-only.md`: +4.5-6.5% model-only update wall. Required for correctness, so not a speed knob.

**n_envs scaling on the old pod.** Horizon 720, earn recipes (`ops/rebuild-2026-09-29/pod4-2026-09-30/earn720/launch.md`, `earn720-12env/launch.md`):

| envs per rank | SPS | GPU memory |
|---|---|---|
| 6 | about 1,900 | 41-49 GB |
| 12 | about 2,410-2,460 | 68.5 GB |
| 16 | 2,615 | 90.96 of 96 GB (stopped for headroom) |

- The memory growth is about 5.5 GB per env, far more than the 0.2 GB per env of rollout storage. I suspect the teacher's one-chunk precompute, but that is unattributed.
- The live H200 run sits at 115 of 141 GB with 20 envs.

## Candidate levers (inference from this map; none measured)
**Config only, next launch** (not applicable to the live run):
1. More envs per rank. This spreads the ~6-12 ms per-step fixed cost and averages out rank imbalance.
2. Lower `rl.teacher_segments_per_minibatch` (for example 4-5) to cap the teacher's peak memory so n_envs can rise. The chunking should leave results the same apart from possible bf16 rounding differences; unmeasured.
3. native_threads: little to gain.

**Code changes:**
- (a) Move the serial Rust work into the rayon pool: decode, Game clone, staging allocation and publish. Also drop the double clone and reuse the staging buffers (`env.rs:570-599, 754-827`; `engine_rs lib.rs:1446`).
- (b) Pipeline two env groups per rank, so one group's native step overlaps the other group's forward. The GIL is already released during native work.
- (c) Compile the actor `policy_core`. It is sync-free with static shapes, the unused `_compiled_actor_core` hook already exists (`kaggriculture.py:320/1088`), and it could even be CUDA-graphed. It would fold about 490 ops into a few kernels.
- (d) Remove the 3 packing syncs in the rollout forward.
- (e) Copy actions to host into a pinned buffer.
- (f) Move the per-step reward telemetry out of `env.step`.

Each needs a complete-work SPS A/B and sampling/replay parity. Any GPU timeline attribution needs nsys on a non-live run.