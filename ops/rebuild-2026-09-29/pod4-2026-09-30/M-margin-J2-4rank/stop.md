# Stop: M-margin-J2-4rank (W&B r350xr3w), 2026-09-30

**Stopped at 2026-09-30T05:17:07Z by SIGTERM, at the owner's request, to free the only 4-GPU pod for the cha22 fixed-opponent run.** The watchdog did not stop it, and no loss had gone nonfinite.

## Reason (owner, verbatim)

"Sure you can just launch the same reward on CHA22 run now?"

Before this message, the main agent reported that the run lost both of its evaluations against its starting point and recommended keeping teacher KL on for cha22. There is one pod, so the cha22 launch requires this stop. The cha22 run is recorded separately as `vs-cha22-4rank-20260930`.

## Final state

- **Last iteration:** 1,469, the last rank-0 iteration record. Ranks 1-3 wrote through 1,468 or 1,469. The progress bar read 26,525,696 env steps, which is 2,457,600 + 16,384 × 1,469.
- **Wall time:** 1 h 43 min 42 s from launch (03:33:13Z).
- **Checkpoints:** `checkpoint_00_010_010_624.pt` (04:06Z), `checkpoint_00_020_004_864.pt` (04:49Z) and `checkpoint_last_best.pt`. `last_best` was written at 04:06Z and is the start point that neither evaluation beat. SIGTERM left **no `checkpoint_final.pt`**, and the torchrun agent exited with `SignalException: Process 41106 got signal: 15`.

## Evaluations (candidate vs `last_best`, not promoted)

| Env steps | Win rate | Candidate own bank | `last_best` bank | Promoted |
|---|---:|---:|---:|---|
| 10M | 14.1% | 45.5k | 65.8k | no |
| 20M | 10.9% | 32.6k | 64.2k | no |

- The 10M row was checked in this session against the local W&B file (`run-r350xr3w.wandb`): `eval/win_rate_against_last_best` 0.140625, `eval/candidate_bank` 45,533, and `eval/margin_mean` −20,316.
- The 20M row is the main agent's W&B reading. This session did not re-extract it from the local file.
- **Reading:** under reward term M (margin 0.5/50,000/0.5, terminal 0.5, econ shaping 0), the policy lost to its J/2 warm start at both evaluations. The self-play own bank also fell. The watchdog's last reading was 38,100 at iteration 1,461, against about 64k at iterations 12-23.
- **Not attributed:** this is one seed with self-play telemetry. The LR re-warm-up and the J/2-final teacher are confounded with the reward change.

## Stop procedure (by PID; no pattern kill)

1. `kill -TERM -41072` (the process group of `run_margin.sh`), then `kill -TERM 41111 41112 41113 41114` (ranks 0-3). After 20 s none of 41072, 41106 or 41111-41114 remained.
2. The watchdog (pid 41753) exited on its own at 05:17:32Z with "run gone; last rank-0 iteration=1469".
3. `nvidia-smi --query-compute-apps` then listed no compute apps, so the GPUs were free.
4. On the Mac, `pkill -P 23094; kill 23094` stopped the copy-off loop. One final pass, the same script body run once, was then made by hand at 05:17:42-05:17:57Z. The run dir's non-checkpoint files (`config.yaml`, `attempts.jsonl`, `warm_start.json`, `wandb/`) were also copied, to `rundir-meta/`.

## Custody (Mac: `/Users/poonszesen/kg-v3-runs/M-margin-J2-4rank-20260930/`)

- `checkpoints/20260930-033319/checkpoint_00_010_010_624.pt`, sha256 `270caf20f65088310a74628ffb15bafe24f816d2062dbf766e5a7cbbd4b0c0f7`.
- `checkpoints/20260930-033319/checkpoint_00_020_004_864.pt`, sha256 `f834f80e82b6fc2a263da07e32d9b38aea5237bcc9f61af8bd4a8c552cfdab3a`.
- `checkpoints/20260930-033319/checkpoint_last_best.pt`, sha256 `bbd07fe6f13666e3df7c8e8a36f8eca43d7bc7c54e8cdec80fb0e80a3c5875c3`.
- `M-margin-J2-4rank-20260930.log`, sha256 `d1ac0cd503045dfcc8bc486c32b050666880e3d862d7644143277c908d78d84a`. This is the final log, and it matches the pod.
- `M-margin-watchdog.log`, sha256 `0bd15dadd1582053dc378e718fc6397d7c21dc7717f0eab154fb40a0950d2b5b`. It matches the pod.
- `SHA256SUMS`, `copyoff.log`, `receipts/` and `rundir-meta/` are also in that folder.
- The pod keeps the run dir `/root/runs/M-margin-J2-4rank-20260930/20260930-033319/` and the receipts `/root/receipts/M-margin-J2-4rank-20260930/`.
- W&B run `https://wandb.ai/spoon/kg-v3/runs/r350xr3w` is left as it ended. Its final state on W&B was not checked.
