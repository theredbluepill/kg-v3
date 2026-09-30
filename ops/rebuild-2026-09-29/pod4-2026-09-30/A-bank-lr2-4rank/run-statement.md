# Run statement: A + LR/2, own-bank reward, 4 ranks (pod abl4mvr5w1mmn4, 2026-09-30) — PRE-LANDING

**Pre-landing.** The code is branch `kg/rebuild-reward-bank` at `ab98e73`, which is not merged. Its independent review is running in parallel with this run. Nothing here is a landed result.

- **Owner (verbatim):** "go with A then now." Earlier directions that apply here: "A is every 10M steps will be fine" (evaluation and checkpoint every 10M env steps); "no need a watch dog to watch the earn" (a nonfinite-loss stop is fine); "you should not set any time cap".
- **Question:** does owner term A keep the BC economy past the point where the J/2 control declined? Term A is absolute own-bank shaping (per step and seat, `min(.25, max(0, own bank)/100000)` after minus before, own seat only), with J's LRs halved.
- **Control:** J/2, W&B `spoon/kg-v3/nw3klj2s` (`../control-J2-4rank/result.md`, commit `6994c8b`). Its rank-0 `train/own_bank_mean` peaked at 81,644 at iteration 45, fell to 46,723 by iteration 90, then plateaued at about 47-53k through iteration 147.
- **What differs from J/2:**
  - The effective config differs only in the bank term: `econ_bank_weight` 0 → 1.0 and `econ_bank_cap` 0 → 0.25, with `econ_bank_scale` 100,000. This moves the terminal win/loss scale from 0.75 to 0.5.
  - LRs (muon 1e-4, adamw 5e-6), checkpoint_freq 10M, native_threads 4, workload (16 optimizer steps and 16,384 env steps per iteration), teacher, BC warm start (`fd854587…6f51`, `model_only`), wrapper and CPU binding are the same.
  - The code moves from `7e87f54` to `ab98e73`. The change is the Rust/Python bank reward, `train/return_common_mean` and `reward_bank_mean` telemetry, and the config comment and cadence edits of the merged 8-rank prep.
  - This run has no step or time cap, where J/2 ran 150 iterations.
- **Mechanism:** under the relative-only reward, mirror self-play pays nothing for a shared slide into poverty. Term A pays each seat for its own bank up to 25k. Above 25k it pays nothing, so it guards against collapse and does not reward growth beyond 25k.
- **Prediction:**
  - Rank-0 `train/own_bank_mean` stays at or above the BC level (~74k at iterations 12-23) past iteration 90.
  - `train/return_common_mean` > 0.
  - `train/reward_bank_mean` is logged and nonzero from iteration 1, which shows the term is active.
- **Loss condition:** the same dip as J/2 appears, a fall from its peak to about 47-53k by iterations 80-90. That would mean term A, which saturates at 25k, does not act on the observed decline, since that decline stays far above 25k.
- **Discriminating observation:** the rank-0 bank intervals at iterations 57, 68, 79 and 90 against J/2's 77,419, 66,577, 51,439 and 46,723.
- **Stop:**
  - None is set by bound or time: no `--max-env-steps` and no `--max-runtime-hours`, so `run_ppo` stops only at its rollout seed budget.
  - The only automatic stop is `watchdog.py`, on a nonfinite `loss/*` on any rank. It does not stop on the bank, and it drops main-J-4rank's replay-drift text match. The trainer's own log-ratio RuntimeError is unchanged.
  - The owner decides when to stop. Evaluation and checkpoints run every 10M env steps, about every 610 iterations.
- **Inputs/code path:** from `/root/kg-v3-A` (clean `ab98e73`, Rust extension built with `maturin develop --release`, `assert_release_build()` passing), via `run_A.sh`:
  `torchrun --nproc-per-node 4 /root/A-bank/main_probe.py scripts/run_ppo.py configs/kaggriculture_4rank_bc_finetune_bank.yaml /root/runs/A-bank-lr2-4rank-20260930 --load-model-weights /root/bc-best/checkpoint_bc_best.pt --load-model-weights-mode model_only --log-mode wandb --wandb-mode online --experiment-id A-bank-lr2-4rank-20260930 -o env.native_threads=4`
  Environment: `KG_NT_NUMA=cpu` (affinity by LOCAL_RANK), `OMP_NUM_THREADS=1` and `WANDB_ENTITY=spoon`. The wrapper `main_probe.py` is main-J-4rank's, unchanged (sha256 `26ba5b0c…`). The preset carries the LRs and cadence, so there are no LR or cadence overrides.
- **Custody:**
  - Log: `/root/runs/A-bank-lr2-4rank-20260930.log`.
  - Receipts: `/root/receipts/A-bank-lr2-4rank-20260930/`.
  - Mac copy-off every 10 min (`copyoff.sh`) to `/Users/poonszesen/kg-v3-runs/A-bank-lr2-4rank-20260930/` with `SHA256SUMS`, because the pod has no network volume.
- **Limits:**
  - One seed. The reward change and the terminal-scale change are confounded.
  - Rank-0 bank telemetry comes from self-play (both seats are the learner), not from games against a fixed opponent. Generality needs a later head-to-head against BC best and other opponents.
