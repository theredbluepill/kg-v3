# Plan 6.4 Orbit end-to-end (scaling_6m, 1 GPU, 2 iterations, forced evaluation): receipt

Status: **pending Codex review** (Codex is at its usage limit until Oct 6). Not
independently verified. The run statement is `run-statement.md`, committed in
`648ff82` before launch.

## Identity

- Pod `aki4vy8kpfldpa`, RTX PRO 6000 Blackwell GPU 0 only (`CUDA_VISIBLE_DEVICES=0`),
  driver 595.91.07. `/root/kg-v3` at `994818b87041426c6fc442a85fb06932937d58a7`,
  0 porcelain lines (`pod/git_state.txt`).
- Software: torch 2.9.0+cu128, triton 3.5.0, flash-attn 2.8.3, wandb 0.26.1, and
  the owl.rs release build `5ad74acd…de69` (the env receipt, `9ae0e05`).
- Hashes (`pod/hashes.sha256`): `configs/scaling_6m.yaml`
  `8352461d4d9c93b8deacc25ef570a6002d2ca920606d948643b3921a9e2cc377`, and the resolved run
  config `config.yaml` (copied to `pod/config.yaml`)
  `bfa377742b14311744eb821506c6f9942d3e054d10a151b95efaae2e223d0ade`. The launcher shim
  `launch_run_ppo_v3.py` is `6410d262…45c9`, the same bytes locally and on the pod.
- Command: `pod/orbit64.sh`, run with `--max-env-steps 32768 -o rl.checkpoint_freq=32768
  --log-mode wandb`, under `timeout --kill-after=60 1800` as the watchdog.
- **W&B (online):** https://wandb.ai/spoon/kg-v3/runs/fmoj4eu9, entity `spoon`,
  project `kg-v3`, group `kg-v3-6.4-orbit`, tags `kg-v3, task-6.4,
  orbit-regression, pod-aki4vy8kpfldpa, src-994818b`. The API state is
  `finished` and 5 files were synced (`pod/wandb_history.json` was pulled from the
  API after the run). The run *name* is `20260929-160631`, because run_ppo
  creates a timestamped subdirectory under `output_dir` and names the run
  after it. The v3 identity is carried by the project, group and tags.

## Outcome

**Exit 0 after 2 complete iterations and 1 forced evaluation.** The run went
from 16:06:29Z to 16:08:01Z, **92 s wall** including startup, compile,
evaluation, checkpoints and the W&B finish (`pod/times.txt`). The expected
observation in the statement held: 2 training rows, 1 eval row with
`eval/games` = 256, and the checkpoints plus 8 eval replays were written.

| Iteration | env steps | iteration s | rollout s | teacher s | update s | SPS (`perf/steps_per_second`) | learner turns (Δ`train/player_step_total`) |
|---|---|---|---|---|---|---|---|
| 1 (includes compile and autotune) | 16,384 | 68.362 | 16.153 | 0 | 33.717 | 239.7 | 41,719 |
| 2 | 16,384 | 2.424 | 0.785 | 0 | 1.633 | **6,758.8** | 41,198 |

- **SPS over complete iterations:** 32,768 / (68.362 + 2.424 s) = **462.9 env steps/s**
  for both iterations. Iteration 2 alone reached 6,758.8 env steps/s,
  about 17.0k learner turns/s, with rollout 20,859 and update 10,034 steps/s.
  Iteration 2 is **one** post-warmup iteration, not a steady-state rate. The
  tqdm total including evaluation was 392.3 env steps/s.
- **Teacher:** `teacher_seconds` = 0, `teacher/cache_bytes` = 0 and `teacher/kl` = 0
  in both iterations. With `teacher_mode: last_best` the teacher only
  activates after a promotion, and none happened. So **the teacher phase was
  not exercised** by this smoke.
- **Compile:** `compiled_model_modules` = 1 (trunk,
  `max-autotune-no-cudagraphs`). The log line reads `Compiled GEMM
  backends: compile_gemm_game=orbit, compile_gemm_backends=ATEN,TRITON,CPP`, so
  Orbit keeps Isaiah's backends and the cuBLAS-only restriction applies only
  to Kaggriculture. There were 14 max-autotune GEMM benchmark blocks in
  iteration 1. The flash path comes from `force_flash_attn: true` in the
  resolved config, but no kernel-level check was made in this run.
- **Forced evaluation (after iteration 2):** `eval/win_rate_against_last_best`
  was **0.4609 over 256 games** (2p 0.4084, 4p 0.6154; the per-player-count
  game denominators are not logged). Evaluation took 12.68 s (`perf/eval_sps`
  10,053), with mean game length 279.4 and a full-length rate of 0.0117. It
  gave `eval/promoted` = 0 against a threshold of 0.7. Last-best is the
  initial weights, so this is a path check, not a strength claim.
- **Peak memory (GPU 0):** torch `max_memory_allocated` was 18,592,669,184 B
  (17.32 GiB, 18.1 % of 97,887 MiB), `max_memory_reserved` was 21,311,258,624 B
  (19.85 GiB), and the `nvidia-smi` peak was 21,163 MiB (21.6 %) at 99 % peak util
  over 92 one-second samples (`pod/nvsmi_samples.csv`). GPU 1 stayed at 0 MiB. The
  phase of the peak is not attributed.
- **Bulk artifacts stay on the pod**, under
  `/root/runs/kg-v3-6.4-orbit-scaling6m-2it-20260930/20260929-160631/`
  (`pod/checkpoints.sha256`):
  - `checkpoint_00_000_032_768.pt` `883b4bff53adbbaac31cf4c7bc5830322ec39f572fa34d3543852935c53dd863`
  - `checkpoint_last_best.pt` `6c89d1bcfec0ca7ae9d4d4f18345215dd4583eb9551de42d45d35b232d5693bd`
  - `checkpoint_final.pt` `2e7d20ef9b79ed75e770a8aab4db8224eb584886c9c7b1d296ae9b50233c011c`
  - 8 eval replays (`eval_replays/checkpoint_00_000_032_768/eval_game_*.jsonl`, 4.5 to 11.6 MB each)
- **Cost:** the run held the GPU for about 92 s, about $0.11 at $4.18/h.

## Warnings and observations

- No Python warning or error lines appeared in `pod/run.log`. The output
  consists of the W&B banner, the override echo, the compile-claim line,
  autotune tables and tqdm.
- `eval/max_entities_exceeded_per_game` was **46.94**, and training
  `train/max_entities` was 312 against `max_entities: 320`. At the evaluation's
  longer games, Orbit observations overflowed the 320-entity cap on average
  about 47 times per game. That behaviour comes from the shipped config, but
  it truncates information in long games. It is recorded here, not diagnosed.
- **Integration gap:** `python/owl/train/logging.py` at `994818b` hard-codes
  `project="orbit-wars"`, and in wandb 0.26.1 `init` arguments override
  `WANDB_PROJECT` (read in `wandb/sdk/wandb_init.py::make_run_settings`). The
  run reached `kg-v3` only through the launcher shim. Any run of
  `scripts/run_ppo.py` without that shim logs to `spoon/orbit-wars`. A code
  fix belongs in a reviewed change, not in this evidence branch.

## Limits

- This is a 2-iteration smoke on one GPU. It makes no learning claim and no
  steady-state SPS claim, and there was no DDP. It did not use Nsight (no
  `nsys` on this pod).
- The teacher path, resume and the Kaggriculture game were not exercised.
- The environment and the W&B URL are receipts, not an independent check.
