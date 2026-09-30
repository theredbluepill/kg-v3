# Launch: main run J, 4 ranks (pod abl4mvr5w1mmn4, 2026-09-30)

**Outcome: the watchdog stopped the run after 6.6 minutes, at iteration 79 of about 15,000 planned.** Rank 0's `train/own_bank_mean` fell below the 20,000 floor on two bank intervals in a row: 17,936 at iteration 68 and 5,684 at iteration 79. The replay alarm never fired and all losses were finite. No checkpoint was written because the first is due at 10M steps. The run was not relaunched. What comes next is the owner's decision, and the pod is **idle but still billing** ($8.36/h).

The run statement is `run-statement.md` (commit `d0acd94`), written before the launch. Nothing here has been independently reviewed.

## Command and identity

- Pod checkout `7e87f54` with 0 porcelain lines. Hashes are in `receipts/hashes.sha256` and the BC best is `fd854587…6f51`. `run_ppo` recorded the same warm-start sha256 in `warm_start.json`, with mode `model_only`.
- Launched at 2026-09-30T00:39:59Z with `setsid nohup /root/main-J/run_main.sh > /root/runs/main-J-4rank-20260930.log 2>&1 < /dev/null &`. `run_main.sh` then ran:
  `torchrun --nproc-per-node 4 /root/main-J/main_probe.py scripts/run_ppo.py configs/kaggriculture_4rank.yaml /root/runs/main-J-4rank-20260930 --load-model-weights /root/bc-best/checkpoint_bc_best.pt --load-model-weights-mode model_only --log-mode wandb --wandb-mode online --experiment-id main-J-4rank-20260930 --max-runtime-hours 20 -o optimizer.muon_lr=0.0002 optimizer.adamw_lr=0.00001 rl.checkpoint_freq=10000000 env.native_threads=4`
  with `OMP_NUM_THREADS=1 WANDB_ENTITY=spoon KG_NT_NUMA=cpu`.
- CPU binding worked on every rank: ranks 0/1 used node 0 and ranks 2/3 used node 1, each with 128 CPUs. No memory policy was set.
- W&B run: **https://wandb.ai/spoon/kg-v3/runs/gq94cyyp**, online, group `main-J-4rank-20260930`, name `ppo-20260930-004004`. The group comes from `--experiment-id`, and no flag sets the run name. The API shows history through step 1,376,256, which includes all 7 bank points. The run state still read `running` right after the kill, because the SIGTERM ended wandb before it could finish, so W&B will mark it crashed.
- PIDs (all gone now):
  - `run_main.sh` process group 19215
  - torchrun 19248
  - ranks 19253-19256, each in its own session
  - watchdog 20168; this was its third start (see below)
  - Mac copy-off loop 57306, which I stopped by hand after the run ended

## Early checks (iteration 3 complete on all ranks)

| Check | Result |
|---|---|
| Per-rank env seeds | `seed`/`seed_stride` = 0/4, 1/4, 2/4, 3/4 for ranks 0-3. Rank r uses seeds r + 4k, so the ranks' seeds do not overlap. |
| Optimizer steps | 16 per iteration on every rank: `optimizer/steps` 16, 32, 48 at iterations 1-3, and 16 minibatches per update. This held through iteration 88. |
| Replay alarm | Quiet: the alarm text never appears in the log. The update-mean `policy/logratio_mean` was -0.0020 at iteration 3 and drifted to about -0.02 by iterations 60-82. That value is the whole-update mean, not the alarm's first-minibatch value. Limit 0.05. |
| Nonfinite loss | None on any rank. |
| Peak GPU memory | 39,516 MiB on GPU0 and 38,957-38,977 MiB on GPU1-3, of 97,887 MiB (2 s samples). |
| Iteration 1 | 21.8 s, with the Inductor/Triton cache warm from the sweep |
| Iterations 3-10 | 4.71 s per iteration, 3,480 game SPS, 6,960 seat SPS. Rollout 2.77 s, teacher 0.46 s, update 1.48 s. Native step 33.5/34.1/33.8/33.0 ms (r0-r3), 1.31 cores per rank. |
| Iterations 3-79 | 4.23 s per iteration, 3,870 game SPS. Native step 26.4/26.1/26.9/25.8 ms. It runs faster than iterations 3-10 because the native step is short, about 15 ms, just after each game resets and grows to about 40 ms late in the game. |

**Time per 10M-step promotion check:** 10M / about 3,870 game SPS ≈ **43 min of training between checks**. The check itself is 64 games of 720 steps on rank 0 (current vs last-best) while ranks 1-3 wait at the barrier. The run never got that far, so this is not measured. From 720 steps × (about 25 ms native step + two model forwards), the estimate is roughly 0.5-1 min per check.

## Bank economy (rank 0, both seats, 256 finished self-play games per interval)

| Iteration | Env steps | LR (muon) | own_bank_mean | margin_abs_mean |
|---|---|---|---|---|
| 12 | 196,608 | 3.8e-5 | 72,624 | 16,580 |
| 23 | 376,832 | 7.4e-5 | 73,095 | 17,140 |
| 34 | 557,056 | 1.09e-4 | 74,292 | 16,040 |
| 45 | 737,280 | 1.44e-4 | 68,371 | 15,230 |
| 57 | 933,888 | 1.82e-4 | 52,603 | 15,620 |
| 68 | 1,114,112 | 2.0e-4 (peak since 63) | 17,936 | 9,939 |
| 79 | 1,294,336 | 2.0e-4 | 5,684 | 5,116 |

The bank held for the first three games, up to LR 1.1e-4. It started falling at iteration 45, before the LR peak, and collapsed right after the peak. Draw rate was 0 throughout. `loss/teacher_kl_loss` rose steadily from 7e-6 to about 0.013, so the student moved away from the BC teacher as the LR rose. **Answer to the run question:** in this run, recipe J did not hold the BC economy past the LR warm-up peak. That rests on one seed and one run.

**Unresolved attribution:** these results do not separate an LR effect (the decline tracks the warm-up) from what the objective pushes toward. With self-play `win_loss` plus economic shaping, both seats can lose bank together while each game still has a winner, and `margin_abs_mean` falling alongside the bank fits that. They also cannot rule out a teacher coefficient (0.005) too weak to anchor the policy. The discriminating next checks are:
- rerun with a lower or flat LR, and see whether the bank still falls at the same step count;
- read the reward components per seat at iterations 45-79 against the bank;
- play the iteration-68/79 policy against the BC best. That needs a checkpoint, and none was saved.

## Watchdog notes

`watchdog.py` (committed version, sha256 `65050a89…`) is the one that stopped the run. It was started three times during launch, and `/root/runs/watchdog.log` records each swap:
1. The first version only signalled the process group. torchrun puts each rank in its own session, so the second version also signals every descendant. I tested it on a synthetic process tree before the swap.
2. The second version parsed records line by line. The ranks share the log file, and a record's newline can land after another rank's text, so it missed most records. The third version decodes every `[nt-probe]` tag wherever it appears. On the live log it counted 71/71/71/71 records per rank with none skipped, and I tested it on synthetic nan, bank and alarm logs before the swap.

The stop sequence: the NaN, bank and alarm triggers were checked, the bank trigger fired at 00:46:34Z, SIGTERM went to 148 pids, and everything had exited by 00:46:36Z. All 4 GPUs were back to 0 MiB afterwards. Because `run_main.sh` was in the killed group, I wrote its post-run receipts (`times.txt` note, `run_dir_listing.txt`, `idle_after.csv`) by hand.

## Custody

- Pod:
  - run dir `/root/runs/main-J-4rank-20260930/20260930-004004/`, holding config, attempts, warm_start and wandb, with no `.pt`
  - log `/root/runs/main-J-4rank-20260930.log`
  - `/root/runs/watchdog.log`
  - receipts `/root/receipts/main-J-4rank-20260930/`
- Mac: `/Users/poonszesen/kg-v3-runs/main-J-4rank-20260930/`, holding the run log, watchdog log, receipts, the run dir without `.wandb`, `SHA256SUMS`, `copyoff.log` and `copyoff.pid`.
- Run log sha256 `3fff7429…1588` (final). Compact copies are in `pod-receipts/`: `iterations.txt` is the per-iteration rank-0 table from `parse_iterations.py`, alongside the watchdog log and the pod receipts.

## How to stop everything (if it were running)

- Training: run `/root/kg-v3/.venv/bin/python -c "import os,signal; os.killpg(19215, signal.SIGTERM)"` on the pod, then `pkill -f main_probe.py` for the rank sessions. Alternatively, `kill <watchdog pid>` and send SIGTERM to torchrun, which forwards it to the ranks.
- Watchdog: run `pkill -f /root/main-J/watchdog.py` on the pod.
- Copy-off: run `kill $(cat /Users/poonszesen/kg-v3-runs/main-J-4rank-20260930/copyoff.pid)` on the Mac.
- Pod billing: stop or terminate pod `abl4mvr5w1mmn4`. That is the owner's decision; this run left it running and idle.

Spend: the run took 00:39:59-00:46:36Z (6.6 min, about $0.92 at $8.36/h). Setup and inspection on the idle pod took about 00:32-00:50Z (about $2.50 in total including the run).
