# Run statement: plan 6.1 memory smoke on 2 ranks (pre-landing)

Written before launch. **Pre-landing diagnostic** on `kg/pod-ppo-prelanding`
(Tasks 3.1 remainder, 4.4 and the BC handoff merged; not Codex-verified; Codex
is at its usage limit until Oct 6). Results: pending Codex review.

- **Question:** with the production 2-rank recipe, the BC-best warm start and
  the last-best teacher on, does one full PPO iteration plus a forced
  evaluation fit the plan's memory target (peak
  `torch.cuda.max_memory_allocated` <= 85 % of 97,887 MiB = 83,204 MiB per
  rank), and do the FlashAttention varlen path and the cuBLAS-only (ATEN)
  compiled trunk actually run? Which spm/accum split follows (I1)?
- **Inputs and code path:** pod `aki4vy8kpfldpa`, `/root/kg-v3` at the
  `kg/pod-ppo-prelanding` commit adding this statement; Rust unchanged from
  the built extension. `configs/kaggriculture_2rank.yaml` as merged (128
  envs/rank, spm 8, accum 1, horizon 64, depth-8 width-256 model,
  `force_flash_attn: true`, `model_compile: trunk` /
  `max-autotune-no-cudagraphs`, bf16, `native_threads: 2`, pinned buffers,
  `teacher_mode: last_best`, KL/value 0.005, teacher chunk 128). Weights:
  `--load-model-weights /root/bc-best/checkpoint_bc_best.pt`
  (SHA-256 `fd854587…6f51`, BC step 3200, held-out NLL 0.480)
  `--load-model-weights-mode model_fresh_critic_head`; with no
  `teacher_init`, the initial last-best teacher is copied from the loaded
  model (handoff note). Overrides: `rl.eval_replay_games=0` (required until
  Task 7.3) and `rl.checkpoint_freq=16384`, which forces one checkpoint and
  one last-best evaluation (128 full 720-step games on rank 0, BC-policy late
  game = dense positions) after iteration 1. `--max-env-steps 16384` stops
  after that one iteration. Launcher `early-smoke/launch_kg_run_ppo.py`
  (observes only; per-phase CUDA peaks with the peak counter reset at each
  phase start). Command: `timeout --kill-after=60 1200 .venv/bin/torchrun
  --nproc-per-node 2 <launcher> scripts/run_ppo.py
  configs/kaggriculture_2rank.yaml <run> --load-model-weights ... --load-model-weights-mode
  model_fresh_critic_head --log-mode wandb --wandb-mode online
  --max-env-steps 16384 -o rl.eval_replay_games=0 rl.checkpoint_freq=16384`,
  `OMP_NUM_THREADS=1 WANDB_MODE=online WANDB_ENTITY=spoon`.
- **Measurements:** per-phase peak allocated/reserved per rank (rollout,
  which includes first-call compile/autotune; teacher precompute; update;
  evaluation), nvidia-smi 1 s samples; `teacher/cache_bytes` and the
  headroom line (expected 1,674,575,872 B per rank); native step time at
  `native_threads` 2 (training and evaluation steps); the compile-claim line
  (ATEN GEMM backends) and `compiled_model_modules`; FlashAttention: the
  config forces it, so any trunk attention call that cannot use the varlen
  kernel raises (`force_flash_attn=True requires ...`) and a clean run is the
  confirmation; W&B URL/state.
- **Expected discriminating observation:** exit 0 with 1 iteration (16
  optimizer steps), one evaluation row (`eval/games` 128), finite metrics, no
  replay-drift alarm, every phase peak allocated <= 83,204 MiB. A phase above
  the target (or an OOM) means spm 4 / accum 2 for 6.2 (I1; global batch
  unchanged); below it keeps spm 8 / accum 1.
- **Limits known in advance:** the training rollout covers only game steps
  0–63 from reset, so the update and teacher phases are measured on sparse
  early states here; the evaluation phase is the dense-state measurement.
  6.2 keeps the same per-phase instrumentation and reaches late-game rollouts.
- **Stopping condition and budget:** stop at exit, on OOM, nonfinite
  values, the alarm or a native exception. External watchdog 20 min
  (`timeout 1200`), about $1.40.
