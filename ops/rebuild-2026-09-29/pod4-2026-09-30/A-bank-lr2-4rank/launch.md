# Launch: A + LR/2, own-bank reward, 4 ranks (pod abl4mvr5w1mmn4, 2026-09-30) — PRE-LANDING

**Pre-landing.** The code is `kg/rebuild-reward-bank` at `ab98e73`, which is not merged; its independent review is running in parallel. **The run is live, with no step or time cap.** Only the owner stops it, or the watchdog on a nonfinite loss. This record covers the launch through iteration 3, and the first bank intervals up to iteration 57.

Run statement: `run-statement.md`, commit `3baa5cc`, written and committed before the launch.

## Identity

- **Checkout.** `/root/kg-v3-A` was cloned from a git bundle of `kg/rebuild-reward-bank` (bundle sha256 `f453f5bd…d2942`, `/root/xfer/rewardA-ab98e73.bundle`).
  - `git rev-parse HEAD` is `ab98e736a7213b52a612ee040e5a5358c647099c`, with 0 porcelain lines at launch.
  - `/root/kg-v3` (`7e87f54`) was not touched.
- **Build.**
  - `UV_LINK_MODE=copy uv sync --frozen --group dev --extra flash-attn` took 37 s.
  - `maturin develop --release` took 34 s and was mandatory, because the Rust reward changed. It produced `python/owl/rs.abi3.so` (sha256 `3fa3bd12…`).
  - `owl.rs.assert_release_build()` passed.
  - No tests were run on the pod, and the Orbit fixtures were not copied.
- **Config check.** `FullConfig.from_file(configs/kaggriculture_4rank_bc_finetune_bank.yaml, {env.native_threads: 4})` loaded:
  - `econ_bank_weight` 1.0, `econ_bank_scale` 100000.0 and `econ_bank_cap` 0.25;
  - `muon_lr` 0.0001, `adamw_lr` 5e-06 and `checkpoint_freq` 10,000,000;
  - `n_envs` 64 per rank, horizon 64, `segments_per_minibatch` 4 and grad accumulation 1.
  - The preset needed no overrides beyond `env.native_threads=4`. With the comment lines dropped, its diff against `configs/kaggriculture_4rank.yaml` is only the two LRs and the two bank values. J/2 passed those LRs and the 10M cadence as overrides, so the effective config differs from J/2 only in the bank term. That term moves terminal_scale from 0.75 to 0.5.
- **Launch.** `run_A.sh` (sha256 `2344e526…`) was started as a session leader with `setsid nohup` at **2026-09-30T01:30:57Z**. It runs:
  `torchrun --nproc-per-node 4 /root/A-bank/main_probe.py scripts/run_ppo.py configs/kaggriculture_4rank_bc_finetune_bank.yaml /root/runs/A-bank-lr2-4rank-20260930 --load-model-weights /root/bc-best/checkpoint_bc_best.pt --load-model-weights-mode model_only --log-mode wandb --wandb-mode online --experiment-id A-bank-lr2-4rank-20260930 -o env.native_threads=4`
  - It passes no `--max-env-steps` and no `--max-runtime-hours`. `run_ppo` does not require either; without them it stops only at its rollout seed budget.
  - Environment: `OMP_NUM_THREADS=1`, `KG_NT_NUMA=cpu` and `WANDB_ENTITY=spoon`.
  - The GPUs were idle before launch, with no compute apps (`receipts/idle_prelaunch.csv`).
- **CPU binding.** The wrapper `main_probe.py` is main-J-4rank's, byte-identical (sha256 `26ba5b0c…`). It pins by LOCAL_RANK: ranks 0 and 1 to NUMA node 0, ranks 2 and 3 to node 1, with 128 CPUs each. `env_construct` reports `native_threads` 4 and seeds 0-3 with stride 4 on every rank.
- **Warm start.** `warm_start.json` shows the BC best `fd8545872aca59c70e273e9655055e1104cd719588364f463ae87b0d488e6f51`, loaded `model_only`.
- **Hashes.** `/root/receipts/A-bank-lr2-4rank-20260930/hashes.sha256` covers the preset, model config, run_ppo, ppo.py, env.py, rewards.py, reward.rs, the .so, both lockfiles, BC best, the wrapper, the watchdog and the launcher.

## Processes (pod)

- `run_A.sh` pid 26119 is the process-group and session leader; the watchdog signals process group **26119**.
- torchrun is pid 26152, and ranks 0-3 are pids 26157, 26158, 26159 and 26160, each in its own session.
- The watchdog is pid **26170**, started once. It is `watchdog.py` (sha256 `7a8c0e98…`, commit `3baa5cc`).
  - It stops on a **nonfinite `loss/*` only**, on any rank. It has no bank floor and no replay-text match.
  - It prints rank 0's bank, `reward_bank_mean` and `return_common_mean` at each game interval.
  - Before launch, it was tested on the pod against synthetic logs. A NaN `loss/value` on rank 2 produced a STOP. A low bank (4,000) produced no stop.
- The launch's two `&` subshells (26117 and 26165) held the SSH pipe open. They were killed after launch. That left run_A.sh and the watchdog reparented to init; both are alive, and the ranks were unaffected.

## Checks at iteration 3 (all ranks)

- **W&B:** **https://wandb.ai/spoon/kg-v3/runs/04cy2m6s**, online. The wrapper refuses offline or disabled runs. The run name is `ppo-20260930-013102`, and the group and experiment id are `A-bank-lr2-4rank-20260930`.
- **16 optimizer steps per iteration:** rank-0 `optimizer/steps` read 16, 32 and 48 at iterations 1, 2 and 3, and `optimizer/minibatches_per_update` was 16.
- **All ranks through iteration 3:** each of ranks 0-3 wrote iteration records 1-3, with walls of 22.84, 3.63 and 3.96 s, identical across ranks. The trainer metrics are all-reduced, so each iteration's values are the same on every rank.
- **LR warm-up:** 1.6e-06, 3.2e-06 and 4.8e-06 at iterations 1-3, which is 1e-4 × step/1000.
- **No alarm:**
  - The log has 0 `first-minibatch PPO log-ratio` matches, 0 `Traceback` and 0 `NaN`.
  - Every `loss/*` was finite (`entropy_loss`, `policy_loss`, `teacher_kl_loss`, `teacher_value_loss`, `total_loss` and `value_loss`).
  - The watchdog heartbeat is clean.
- **Memory:** the peak was **38,957 / 38,957 / 38,961 / 38,977 MiB** on GPUs 0-3, from 460 two-second samples in the first 15 min. The GPUs total 97,887 MiB, and J/2's peak was 38,977. Host RAM was 162 of 1,511 GiB in use.
- **SPS:**
  - Iteration 1 took 22.84 s (warm Inductor cache), and iterations 2-3 averaged 3.8 s.
  - Over iterations 3-45, throughput was **3,724 game SPS** from wall time (16,384 env steps per iteration), and the mean `perf/steps_per_second` was 3,792.
  - J/2 reached 3,653 over its iterations 3-57. Same hardware, precision and workload; no profiler was run.
- **Bank term active:** `train/reward_bank_mean` and `train/return_common_mean` are present and nonzero from iteration 1. Rank 0 read:

| Iteration | reward_bank_mean | return_common_mean |
|---|---|---|
| 1 | -4.58e-04 | -0.0293 |
| 2 | 4.66e-05 | 0.0030 |
| 3 | 1.38e-04 | 0.0088 |
| 6 | 1.03e-03 | 0.0657 |
| 7 | 1.13e-03 | 0.0723 |
| 11 | 6.09e-06 | 0.0004 |
| 12 | -4.58e-04 | -0.0293 |

- **How the term behaves:** the envs play synchronized games of about 11 iterations. Each game's first segment is negative, because banks fall early as the policy spends. The term is positive through mid-game, then near zero once banks pass the 25k saturation.
- **Averages over iterations 1-44** (four game cycles): `reward_bank_mean` 3.10e-04 and `return_common_mean` 0.0199, both positive.
- **Common-mode return:** `return_common_mean` equals 64 × `reward_bank_mean` at every row, which is the horizon times the per-step mean. The common-mode return is therefore all bank term: the other shaping and the win/loss term cancel between seats.
- **Early bank intervals (rank 0, 256 self-play games each):**

| Iteration | own_bank_mean (A) | own_bank_mean (J/2) | teacher/kl (A) |
|---|---|---|---|
| 12 | 74,497 | 74,250 | 0.020 |
| 23 | 74,725 | 74,629 | 0.104 |
| 34 | 73,811 | 77,285 | 0.307 |
| 45 | 71,621 | 81,644 | 0.733 |
| 57 | 68,830 | 77,419 | (watchdog line, 01:36:00Z) |

- **Reading these intervals:** this is too early to judge the prediction, which is about iterations 57-90. A has not repeated J/2's rise to 81.6k at iteration 45, and it is already 8.6k below J/2 at iteration 57. That is early evidence against the prediction; the loss condition is judged at iterations 68-90.
- **Mechanism observation (unresolved attribution):** the term charges early-game spending, the investment phase, and repays it mid-game. The value it can add above a 25k bank is zero.

## Custody

- **Pod files:**
  - log `/root/runs/A-bank-lr2-4rank-20260930.log`;
  - watchdog log `/root/runs/A-bank-watchdog.log`;
  - run dir `/root/runs/A-bank-lr2-4rank-20260930/20260930-013102/` (`config.yaml`, `attempts.jsonl`, `warm_start.json`, `wandb/`);
  - receipts `/root/receipts/A-bank-lr2-4rank-20260930/`, holding git state, hashes, idle-prelaunch, 2 s and 60 s nvidia-smi samples and times.
- **Copy-off:** a Mac copy-off loop (`copyoff.sh`, sha256 `dbf7bd19…`, a copy of this directory's file) runs under nohup as pid 74346, every 10 min.
  - Its first pass was at 01:33:50Z.
  - It writes to `/Users/poonszesen/kg-v3-runs/A-bank-lr2-4rank-20260930/`: the log, the watchdog log, receipts, finished `checkpoint_*.pt` files checked against the pod's sha256, and `SHA256SUMS`.
  - The pod has no network volume.
- **First checkpoint:** at 10M env steps, about iteration 611, roughly 45 min after launch at this SPS.

## Spend

- The pod costs **$8.36/h** (4x RTX PRO 6000). Setup and launch ran 01:24-01:35Z.
- **The run keeps billing at $8.36/h** until the owner stops it. At about 3,700 SPS, the rollout seed budget is far beyond any practical wall time.

## Unresolved

- One seed.
- The bank term and the terminal-scale change (0.75 to 0.5) are confounded.
- Bank telemetry is self-play only.
- The review of `ab98e73` is still open, so a finding there could invalidate this run.
