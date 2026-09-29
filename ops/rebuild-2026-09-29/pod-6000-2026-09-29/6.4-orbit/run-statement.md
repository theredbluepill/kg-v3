# Run statement: Plan 6.4 Orbit end-to-end (scaling_6m, 1 GPU, 2 iterations, forced evaluation)

Written before launch. Status of results: pending Codex review.

- **Question (regression smoke, not training):** at integration `994818b`, does
  the retained Orbit Wars path of the canonical trainer `scripts/run_ppo.py`
  still run end to end on one RTX PRO 6000 (sm_120)? That path covers rollout,
  teacher, PPO update, checkpoint, last-best evaluation and W&B logging, using
  `configs/scaling_6m.yaml` as shipped (256 envs, horizon 64, segments per
  minibatch 16, trunk compile `max-autotune-no-cudagraphs`, bf16,
  `teacher_mode: last_best`). Checking this confirms that the Kaggriculture
  merges left Orbit working ("Orbit unchanged", plan 1.5 E and 6.4).
- **Inputs and code path:**
  - Pod `aki4vy8kpfldpa`, `/root/kg-v3` at `994818b` (0 porcelain lines),
    `.venv` from the env receipt (`9ae0e05`): torch 2.9.0+cu128, triton 3.5.0,
    flash-attn 2.8.3, owl.rs release build, wandb 0.26.1. Only GPU 0 is used
    (`CUDA_VISIBLE_DEVICES=0`).
  - The sha256 hashes are: `configs/scaling_6m.yaml` `8352461d4d9c93b8deacc25ef570a6002d2ca920606d948643b3921a9e2cc377`,
    `configs/model/stateless_transformer_6m.yaml` `92cc32ab…c105fc`,
    `scripts/run_ppo.py` `c99f32ab…c574`, `python/owl/train/ppo.py` `0af32d17…f75f`,
    `python/owl/train/logging.py` `2247321f…6c19`.
  - Iteration count: run_ppo has no iteration flag. The flag
    `--max-env-steps 32768` equals 2 × (horizon 64 × 256 envs × world 1), so
    the run stops after exactly 2 complete iterations.
  - Forced evaluation: run_ppo has no evaluation flag either. Evaluation runs
    whenever a periodic checkpoint triggers, so `-o rl.checkpoint_freq=32768`
    gives one checkpoint and one last-best evaluation (256 games, 8 replays)
    after iteration 2 and nowhere else. That keeps both iterations' timing free
    of evaluation work. This override is the only config change.
  - W&B: `--log-mode wandb`, online (`WANDB_MODE=online`, `WANDB_ENTITY=spoon`,
    `WANDB_RUN_GROUP=kg-v3-6.4-orbit`,
    `WANDB_TAGS=kg-v3,task-6.4,orbit-regression,pod-aki4vy8kpfldpa,src-994818b`).
    The run name is the run dir name,
    `kg-v3-6.4-orbit-scaling6m-2it-20260930`. **Deviation:** `logging.py` at
    `994818b` hard-codes `project="orbit-wars"`, and in wandb 0.26.1 arguments
    passed to `init` override `WANDB_PROJECT`. So the run goes through
    `launch_run_ppo_v3.py` (sha256
    `6410d262112aa34e03cabf2bc2c6b14c74a04874323208eebb76bceecddf45c9`). This
    launcher is a runpy shim that changes only that one argument to `kg-v3`,
    raises if the argument is not `orbit-wars`, refuses an offline or disabled
    run, and prints torch's peak CUDA memory at exit. The trainer's code is
    unmodified. The hard-coded project is reported as an integration gap.
  - Command (from `/root/kg-v3`): `timeout --kill-after=60 1800 .venv/bin/python <shim>
    scripts/run_ppo.py configs/scaling_6m.yaml /root/runs/kg-v3-6.4-orbit-scaling6m-2it-20260930
    --log-mode wandb --max-env-steps 32768 -o rl.checkpoint_freq=32768`. It is
    launched with nohup. A background `nvidia-smi` sampler (1 s) records GPU
    memory and utilisation.
- **Records:** per-iteration `perf/steps_per_second`, `perf/rollout_sps`,
  `perf/update_sps`, `perf/teacher_sps` and the `time/*` phase seconds. The
  report gives SPS over complete iterations (both iterations, and
  32,768 / (sum of iteration seconds)); iteration 1 includes compile and
  autotune. It also records `train/player_step_total` (valid learner turns),
  compile status (`compiled_model_modules` and the compile-claim log line),
  eval metrics (`eval/win_rate_against_last_best`, `eval/games`,
  `eval/promoted`, `time/eval_seconds`), peak memory (torch allocated and
  reserved; nvidia-smi used), wall time, cost, W&B URL, warnings, and the
  checkpoint SHA-256 (bulk files stay on the pod).
- **Expected discriminating observation:** exit 0 after 2 iterations, with 2
  training log rows, 1 eval row with `eval/games` = 256, and the files
  `checkpoint_00_000_032_768.pt` and the last-best checkpoint. A traceback
  localizes an Orbit regression to its phase (rollout, teacher, update,
  checkpoint, eval or logging). This smoke makes no claim about learning,
  and no win-rate claim: last-best is the initial weights.
- **Stopping condition:** normal exit, the first failure (recorded as the
  result), or the 30-minute wall cap enforced by `timeout` on the pod as the
  external watchdog, whichever comes first. W&B is not allowed to fall back
  to offline: if init fails, the run fails and is reported.
- **Budget:** 30 min or less of wall time on the running pod ($4.18/h, so about
  $2.10 or less). No new resource.
- **Safety:** runs after the DMA-fence task, once both GPUs are idle again with
  no compute apps. No installs, driver changes or security changes, and
  `/root/kg-v3` is not edited. The pod is left running and idle.
