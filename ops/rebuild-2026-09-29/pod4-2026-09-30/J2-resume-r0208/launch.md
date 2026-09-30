# Launch: J/2 continued under the 0.2/0.8 reward, 4 ranks (pod abl4mvr5w1mmn4, 2026-09-30)

**The run is live, with no step or time cap.** Only the owner stops it, or the watchdog on a nonfinite loss. This record covers the launch through iteration 3, with bank intervals up to iteration 23. Run statement: `run-statement.md`, commit `12cb271`, written and committed before the launch.

## Launch path

- **Same-run resume was refused, so this is a warm start.** On `7e87f54`, `run_ppo._validate_args` raises "resume launches cannot use config overrides", and a resume also requires `checkpoint_last_best.pt`, which the J/2 run dir (`20260930-010131`) lacks. The trainer was not forked.
- The run is a fresh launch with `--load-model-weights /root/runs/control-J2-4rank-20260930/20260930-010131/checkpoint_final.pt --load-model-weights-mode model_and_optimizer`. `warm_start.json` records sha256 `80e5667ecbf36966a3edc94d2e831ee5372cd7e791b5d235cf30cbfc32e161dd`. The mode was accepted.
- **Continued from J/2:** the weights, the optimizer moments, and `env_steps`. The `attempts.jsonl` `start_env_steps` is **2,457,600**, which is J/2's 150 iterations. The progress bar reads 2,457,600 + 16,384 × iteration, and `player_step_total` starts at 4,915,200 + 32,768 per iteration.
- **Fresh:**
  - The LR scheduler: warm-up restarts, reading 1.6e-06, 3.2e-06 and 4.8e-06 at iterations 1-3, which is 1e-4 × step/1000.
  - The optimizer step counter, which restarts at 16.
  - The W&B run and the attempt record.
  - The `last_best` teacher, which starts as the J/2 final weights.
- **The iteration counter restarts at 1.** Iteration n here is J/2 iteration 150 + n by env steps.

## Identity

- **Checkout:** `/root/kg-v3` at `7e87f5420b3696141ea41f6453d4fafb2748eea0`, with 0 porcelain lines at launch (`receipts/git_state.txt`). `/root/kg-v3-A` was not used.
- **Command:** `run_resume.sh` (sha256 `36224291…0c52`, byte-identical to this directory's copy) was started as a session leader with `setsid nohup … & disown` at **2026-09-30T02:01:33Z**. It runs:
  `torchrun --nproc-per-node 4 /root/J2-resume/main_probe.py scripts/run_ppo.py configs/kaggriculture_4rank.yaml /root/runs/J2-resume-r0208-20260930 --load-model-weights <J/2 checkpoint_final.pt> --load-model-weights-mode model_and_optimizer --log-mode wandb --wandb-mode online --experiment-id J2-resume-r0208-20260930 -o optimizer.muon_lr=0.0001 optimizer.adamw_lr=0.000005 rl.checkpoint_freq=10000000 env.native_threads=4 env.reward_shaping.econ_cap=0.2`
  - It passes no `--max-env-steps` and no `--max-runtime-hours`.
  - Environment: `OMP_NUM_THREADS=1`, `KG_NT_NUMA=cpu` and `WANDB_ENTITY=spoon`.
  - The GPUs were idle before launch, with no compute apps.
- **Effective config** (run dir `20260930-020138/config.yaml`, config sha256 `1ac37627…97e9`):
  - `econ_shaping` 0.2, `econ_starvation_weight` 4.0, `econ_drought_weight` 1.0, `econ_ineffective_weight` 0, and **`econ_cap` 0.2**;
  - `reward_mode` `win_loss`;
  - muon_lr 1e-4, adamw_lr 5e-6, and `checkpoint_freq` 10,000,000;
  - `native_threads` 4 and `n_runtime_gpus` 4.
- **Terminal scale:** loading that config with `FullConfig` gives **`terminal_scale` 0.8** (`receipts/terminal_scale.txt`). The startup log does not print it. The trainer's `train/return_max` reads 0.2 at iterations 1-4, before any game ends, then 0.8 once games complete (iteration 23). That matches a +0.8 terminal win with no death penalty.
- **Wrapper and watchdog:** both are A2's, byte-identical. `main_probe.py` is sha256 `26ba5b0c…` and pins ranks 0-1 to NUMA node 0 and ranks 2-3 to node 1, 128 CPUs each. `watchdog.py` is sha256 `7a8c0e98…`. The watchdog's docstring still names run A. It reads `reward_bank_mean`/`return_common_mean`, which do not exist on `7e87f54`, so it prints `None` for them; that is observe-only.

## Processes (pod) and how to stop

- `run_resume.sh` is pid **30021**, the session and process-group leader.
- torchrun is pid 30055, and ranks 0-3 are pids 30060-30063, each in its own session.
- The watchdog is pid **30351**, run as `watchdog.py 30021 /root/runs/J2-resume-r0208-20260930.log`. It stops on a nonfinite `loss/*` only, and never on the bank.
- **To stop the run** (owner's decision only):
  1. `kill -TERM -30021; pkill -TERM -f J2-resume/main_probe.py`. Ranks are in their own sessions, so signal them too.
  2. Then `kill 30351` to stop the watchdog. If the run goes first, the watchdog exits by itself once the run is gone.
  3. On the Mac, `pkill -f J2-resume-r0208-20260930/copyoff.sh` stops the copy-off, after its last pass.
- Whether a SIGTERM leaves a `checkpoint_final.pt` was not checked. Only the 10M-cadence checkpoints are certain.

## Checks at iteration 3 (all ranks)

- **W&B:** **https://wandb.ai/spoon/kg-v3/runs/hz4bpjnq**, online. The wrapper refuses offline or disabled runs. The run name is `ppo-20260930-020138`, and the group and experiment id are `J2-resume-r0208-20260930`.
- **16 optimizer steps per iteration:** rank-0 `optimizer/steps` read 16, 32 and 48 at iterations 1-3, and `optimizer/minibatches_per_update` was 16.
- **All ranks:** ranks 0-3 each wrote iteration records 1-31, with identical all-reduced losses. Walls at iterations 1-3 were 21.92, 3.75 and 4.11 s.
- **No alarm:**
  - The log has 0 `Traceback` and 0 `log-ratio` matches.
  - Every `loss/*` was finite on every rank through iteration 31.
  - The watchdog heartbeat is clean.
- **SPS:** over iterations 3-31, **3,395 env SPS** from wall time (16,384 env steps per iteration), and the mean `perf/steps_per_second` was 3,469. J/2 read 3,653 over its iterations 3-57, and A read 3,724. Same hardware, precision and workload; no profiler was run. The difference is unexplained.
- **Memory:** the peak was 38,957 / 38,957 / 38,961 / 38,977 MiB on GPUs 0-3, from 372 two-second samples, the same as J/2. Host RAM was 162 of 1,511 GiB in use.
- **Early bank intervals (rank 0, 256 self-play games each):** 62,645 at iteration 12 and 63,203 at iteration 23. J/2 ended on a plateau of about 47-53k. Two intervals under a re-warming LR are not evidence for the question.

## Custody

- **Pod files:**
  - log `/root/runs/J2-resume-r0208-20260930.log`;
  - watchdog log `/root/runs/J2-resume-watchdog.log`;
  - run dir `/root/runs/J2-resume-r0208-20260930/20260930-020138/`;
  - receipts `/root/receipts/J2-resume-r0208-20260930/`, holding git state, hashes (config, model config, run_ppo, ppo.py, env.py, rewards.py, reward.rs, the .so, both lockfiles, the source checkpoint, the wrapper, the watchdog and the launcher), idle-prelaunch, 2 s and 60 s nvidia-smi samples, times and terminal_scale.
- **Mac copy-off:** `copyoff.sh` (this directory's file) runs under nohup as pid **92562**, every 10 min, to `/Users/poonszesen/kg-v3-runs/J2-resume-r0208-20260930/`, with `SHA256SUMS`. Its first pass was at 02:04:09Z; there are no checkpoints yet.
- **First checkpoint and evaluation:** at 10M env steps, about iteration 461. At 3,400 SPS that is roughly 37 min after launch.

## Spend

- The pod costs $8.36/h (4x RTX PRO 6000), and the run keeps billing until the owner stops it.

## Unresolved

- One seed.
- The reward change (terminal 0.75 → 0.8), the LR re-warm-up and the teacher starting at J/2 final are confounded.
- Bank telemetry is self-play only.
- The SPS gap to J/2 (about 7%) is not attributed.
