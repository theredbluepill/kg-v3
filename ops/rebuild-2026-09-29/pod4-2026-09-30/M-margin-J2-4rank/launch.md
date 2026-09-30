# Launch: M-margin-J2-4rank, J/2 warm start under reward term M (pod abl4mvr5w1mmn4, 2026-09-30)

**The run is live, with no step or time cap.** Only the owner stops it, or the watchdog on a nonfinite loss. This record covers the launch through iteration 46, including the first three game-end intervals. Run statement: `run-statement.md`, commit `ea3af51`, committed before the launch.

This is a **pre-landing launch** (owner: "can we accelerrate?"). The reward branch's docs, cookbook record and full `just prepare` follow after the launch, before landing.

## Identity

- **Code:** `kg/rebuild-reward-margin` at `87beaf0ffe719300c9ffcc2527fae68b22e0bcbe`, delivered to the pod as a git bundle (basis `111ae7d`) and checked out in `/root/kg-v3-M`. It had 0 porcelain lines at launch (`receipts/git_state.txt`). `maturin develop --release` rebuilt incrementally in 29.8 s, and `owl.rs.assert_release_build()` passed. The `.so` sha256 is `8ff9bf0b…a098`.
- **Config:** `configs/kaggriculture_4rank_margin.yaml` (sha256 `4d8e7c04…5977`), with no `-o` overrides. Loading it with `FullConfig.from_file` on the pod (`receipts/terminal_scale.txt`) gives:
  - `reward_mode` `win_loss`;
  - `econ_shaping` 0.0, `econ_bank_weight` 0.0, `econ_bank_cap` 0.0;
  - `econ_margin_weight` 0.5, `econ_margin_scale` 50000.0, `econ_margin_cap` 0.5;
  - **`terminal_scale` 0.5**;
  - muon_lr 1e-4, adamw_lr 5e-6, `checkpoint_freq` 10,000,000, `native_threads` 4.
- **Command:** `run_margin.sh` (this directory; sha256 `72a18512…0761`, byte-identical to `/root/M-margin/run_margin.sh`) was started as a session leader with `setsid nohup … & disown` at **2026-09-30T03:33:13Z**. It runs:
  `torchrun --nproc-per-node 4 /root/M-margin/main_probe.py scripts/run_ppo.py configs/kaggriculture_4rank_margin.yaml /root/runs/M-margin-J2-4rank-20260930 --load-model-weights /root/runs/control-J2-4rank-20260930/20260930-010131/checkpoint_final.pt --load-model-weights-mode model_and_optimizer --log-mode wandb --wandb-mode online --experiment-id M-margin-J2-4rank-20260930`
  - No `--max-env-steps`, no `--max-runtime-hours`.
  - Environment: `OMP_NUM_THREADS=1`, `KG_NT_NUMA=cpu`, `WANDB_ENTITY=spoon`.
  - GPUs had no compute apps before launch (`receipts/idle_prelaunch.csv`).
- **Warm start:** `warm_start.json` records sha256 `80e5667ecbf36966a3edc94d2e831ee5372cd7e791b5d235cf30cbfc32e161dd` and mode `model_and_optimizer`. `attempts.jsonl` has `start_env_steps` 2,457,600, `source_commit` `87beaf0f…`, `telemetry_mode` `wandb-online`. As with hz4bpjnq, the LR warm-up restarts (1.6e-6, 3.2e-6, 4.8e-6 at iterations 1-3) and the `last_best` teacher starts at J/2 final.
- **Wrapper and watchdog:** `main_probe.py` sha256 `26ba5b0c…` and `watchdog.py` sha256 `7a8c0e98…`, byte-identical to hz4bpjnq's. The watchdog still prints `reward_bank_mean`/`return_common_mean` (0.0 here); that is observe-only.

## Config diff against J/2

`diff /root/M-margin/j2_config.yaml <run dir>/config.yaml`. `j2_config.yaml` is byte-identical to J/2's run-dir `config.yaml` (`20260930-010131`). Every difference:

| Key (`env.reward_shaping`) | J/2 | M |
|---|---|---|
| `econ_shaping` | 0.2 | 0.0 |
| `econ_bank_weight` | absent (code `7e87f54` predates term A) | 0.0 |
| `econ_bank_scale` | absent | 100000.0 |
| `econ_bank_cap` | absent | 0.0 |
| `econ_margin_weight` | absent | 0.5 |
| `econ_margin_scale` | absent | 50000.0 |
| `econ_margin_cap` | absent | 0.5 |

Nothing else differs: the LRs, schedule, `checkpoint_freq`, `native_threads`, batch, PPO coefficients, teacher and compile settings are identical. The three `econ_bank_*` keys are inert at weight 0. **No defect.** Run-dir config sha256: M `5b5d4a7c…ff0d`, J/2 `6bcf71d1…1382`.

## Checks (all ranks, through iteration 46)

- **W&B:** **https://wandb.ai/spoon/kg-v3/runs/r350xr3w**, online. Run name `ppo-20260930-033319`, group and experiment id `M-margin-J2-4rank-20260930`.
- **16 optimizer steps per iteration:** rank-0 `optimizer/steps` read 16, 32 and 48 at iterations 1-3, and `optimizer/minibatches_per_update` was 16.
- **All ranks:** ranks 0-3 each wrote iteration records 1-46, with identical all-reduced losses at every iteration.
- **No alarm:** 0 `Traceback`, 0 `log-ratio`, and every `loss/*` finite on every rank.
- **Margin telemetry is nonzero:** rank-0 `train/reward_margin_abs_mean` read 5.6e-4, 4.0e-4 and 8.7e-4 at iterations 1-3, and about 2e-3 mid-game.
- **SPS:** walls at iterations 1-3 were 23.30, 3.74 and 4.07 s. Over iterations 3-46, **3,648 env SPS** from wall time, and the mean `perf/steps_per_second` was 3,720. hz4bpjnq read 3,395 over its iterations 3-31. No profiler was run, and the difference is not attributed.
- **Memory:** the peak was 38,957 / 38,957 / 38,961 / 38,975 MiB on GPUs 0-3, from 118 two-second samples, the same as hz4bpjnq. Host RAM was 166 of 1,511 GiB in use.

## First game ends (rank 0, 256 self-play games per interval)

| Iteration | `return_max` | `own_bank_mean` | `margin_abs_mean` |
|---:|---:|---:|---:|
| 12 | 0.533 | 63,899 | 16,346 |
| 23 | 0.510 | 64,034 | 13,495 |
| 34 | 0.512 | 62,501 | 12,917 |

`return_max` just above 0.5 at game ends fits a 0.5 terminal win plus the margin increments inside the segment. Before any game ends it reads 0.025-0.10, which is margin-potential steps only. hz4bpjnq read 62,645 and 63,203 at iterations 12 and 23. These are three intervals under a re-warming LR, not evidence for the question.

## Processes (pod) and how to stop

- `run_margin.sh` is pid **41072**, the session and process-group leader.
- torchrun is pid 41106, and ranks 0-3 are pids 41111-41114, each in its own session.
- The watchdog is pid **41753**, run as `watchdog.py 41072 /root/runs/M-margin-J2-4rank-20260930.log`. It stops on a nonfinite `loss/*` only.
- **To stop the run** (owner's decision only):
  1. `kill -TERM -41072; pkill -TERM -f M-margin/main_probe.py`.
  2. Then `kill 41753`.
  3. On the Mac, `pkill -f M-margin-J2-4rank/copyoff.sh`.
- A SIGTERM leaves no `checkpoint_final.pt` (seen when hz4bpjnq stopped). Only the 10M-cadence checkpoints are certain.

## Custody

- **Pod files:**
  - log `/root/runs/M-margin-J2-4rank-20260930.log`;
  - watchdog log `/root/runs/M-margin-watchdog.log`;
  - run dir `/root/runs/M-margin-J2-4rank-20260930/20260930-033319/`;
  - receipts `/root/receipts/M-margin-J2-4rank-20260930/`: git state, hashes, idle-prelaunch, 2 s and 60 s nvidia-smi samples, times and terminal_scale.
- **Mac copy-off:** `copyoff.sh` (this directory; sha256 `bb3909bc…9957`) runs under nohup as pid **23094**, every 10 min, to `/Users/poonszesen/kg-v3-runs/M-margin-J2-4rank-20260930/`, with `SHA256SUMS`. Its first pass was at 03:35:44Z. A duplicate loop started at 03:35:39Z (pid 23072) was killed after its first pass.
- **First checkpoint and evaluation:** at 10M env steps, about iteration 461, roughly 35 min after launch.

## Spend

- The pod costs $8.36/h (4x RTX PRO 6000), and the run keeps billing until the owner stops it.

## Unresolved

- One seed.
- Launched before the reward branch's docs, cookbook and full `just prepare`.
- The scale 50,000 is agent-proposed.
- Bank telemetry is self-play only.
