# Result: plan 6.1 memory smoke on 2 ranks (pre-landing, pending Codex review)

**PASS** against the run statement, with one scoped limit (dense states were
measured in evaluation only). Pre-landing diagnostic on the unverified merge
`kg/pod-ppo-prelanding` `f916864` (0 porcelain lines on the pod); not
Codex-verified. Receipts: `pod-receipts/`. W&B:
`https://wandb.ai/spoon/kg-v3/runs/blb43l9j` (state `finished`, online).

Run: `configs/kaggriculture_2rank.yaml`, BC best (`fd854587…6f51`) with
`model_fresh_critic_head`, last-best teacher copied from the loaded model,
`-o rl.eval_replay_games=0 rl.checkpoint_freq=16384`, `--max-env-steps 16384`.
Exit 0, wall 150 s (launch 16:28:31Z, end 16:31:01Z), about $0.17.

## Peak CUDA memory per phase and rank (MiB; target: allocated <= 83,204 MiB = 85 % of 97,887)

| Phase (iteration 1) | Seconds r0 / r1 | Peak allocated r0 / r1 | Peak reserved | Share of target (allocated) |
|---|---|---|---|---|
| Rollout (includes first compile) | 16.4 / 16.5 | 3,273 / 3,273 | 3,874 | 3.9 % |
| Teacher precompute (16,384 rows) | 4.1 / 5.2 | 38,130 / 38,128 | 64,092 | 45.8 % |
| Update (16 minibatches, spm 8) | 26.1 / 26.2 | 24,886 / 24,868 | 64,180 | 29.9 % |
| Evaluation (rank 0, 128 x 719 steps, dense late game) | 74.0 | 3,365 | 64,180 | 4.0 % |

The highest reserved figure is 64,180 MiB (65.6 % of the device), and the
nvidia-smi peak is 65,776 MiB on each GPU. Per-phase peaks reset at each
phase start. The gap between the teacher phase and the update (about 20 s in
iteration 1) is not instrumented; it includes GAE and the first-call compile
of the loss/GAE functions.

## Other measurements

- **Teacher:** `teacher/cache_bytes` 1,674,575,872 B per rank. The headroom
  line printed the same figure: 3 trunk and 2 head calls for 16,384 rows. In
  iteration 1, `teacher/kl` was 0.00257 and `teacher/value_cross_entropy`
  0.664. The teacher is the loaded BC model with a fresh critic head.
- **Native step time at `native_threads` 2 (128 envs per rank):** training
  rollout (game steps 0–63) 52.7 ms per step on rank 0 and 45.7 ms on rank 1.
  Dense evaluation (steps 0–719) 73.6 ms per step. These times include the
  entry fence and the buffer publish.
- **FlashAttention varlen:** the config sets `force_flash_attn: true`, which
  raises on any trunk attention call that cannot take the varlen kernel. No
  such error occurred across rollout, teacher, update and evaluation. This is
  an absence-of-error confirmation; no kernel trace was taken.
- **Compile:** `Compiled GEMM backends: compile_gemm_backends=ATEN` (torch
  2.9.0+cu128, triton 3.5.0, driver 595.91.07), `compiled_model_modules` 1
  (trunk, `max-autotune-no-cudagraphs`). At exit, inductor
  `max_autotune_gemm_backends` was `ATEN`.
- **Update:** 16 optimizer steps. All metrics finite.
  `policy/logratio_mean` -0.0033 nats, under the 0.05 alarm, which did not
  fire. Entropy 4.40, grad norm 14.8, explained variance -0.67 (fresh critic
  head).
- **Forced evaluation:** 128 games, `eval/episode_steps` 719,
  `eval/win_rate_against_last_best` 0.461 (not promoted, threshold 0.7),
  candidate bank 69,807 vs last-best 72,377, 1,243 eval env steps/s.
- **Checkpoints (stay on the pod):** `checkpoint_00_000_016_384.pt`
  `d9d5a6d7…af1d7`, `checkpoint_last_best.pt` `a7a7d262…4b84305`,
  `checkpoint_final.pt` `ea204b49…2ef`.

## Decision: spm/accum split (I1)

Keep **spm 8 / accumulation 1**. The largest allocated peak, 38.1 GB in
teacher precompute, is 46 % of the target, and the update peaks at 24.9 GB.

**Limit:** the training rollout covers game steps 0–63 only, so the update
and teacher phases here ran on sparse early states. Dense late-game update
and teacher peaks come from 6.2, which keeps the same per-phase
instrumentation. If 6.2 exceeds the target or runs out of memory, the split
moves to spm 4 / accum 2.

Native steps take 46–74 ms per step for 128 envs at 2 threads. That is a
large share of rollout time, so `native_threads` is a tuning candidate on
this 256-vCPU pod. It was not measured at other values here.
