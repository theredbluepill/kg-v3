# Stop: J/2 continued under the 0.2/0.8 reward (hz4bpjnq), 2026-09-30

**Stopped at 2026-09-30T03:20:16Z by SIGTERM, at the owner's request, to relaunch J/2 under a new reward.** The run was not stopped by the watchdog, and no loss had gone nonfinite.

## Reason (owner, verbatim)

1. "can we relaunch (before anchor) run, fix the reward, 0.5 Cash Diff (add this in) + 0.5 (Terminal loss 1/-1/0)?"
2. "implement the new rewrad first before we revisit the cha22 anchor setup."
3. "can we accelerrate?"

The main agent reads "(before anchor) run" as J/2 (control-J2, W&B `nw3klj2s`, run dir `20260930-010131`), which this run continued. The relaunch is a fresh warm start from J/2's `checkpoint_final.pt`, not from this run. It is recorded separately as `M-margin-J2-4rank-20260930`.

## Final state

- **Last iteration:** 985. That is the last rank-0 iteration record, and the progress bar read 18,595,840 env steps at the stop, which is 2,457,600 + 16,384 × 985.
- **Wall time:** 1 h 18 min from launch (02:01:33Z).
- **Checkpoints:** only the 10M-cadence checkpoint `checkpoint_00_010_010_624.pt` (about iteration 461, written 02:38Z) and `checkpoint_last_best.pt`. SIGTERM left **no `checkpoint_final.pt`**, and the torchrun agent exited with `SignalException ... got signal: 15`. The next checkpoint, at 20M env steps (about iteration 1,071), was not reached.

## Bank trend

These are rank-0 `train/own_bank_mean` intervals, 256 self-play games each.

| Iterations | Intervals | Mean own bank |
|---|---:|---:|
| 1-100 | 8 | 63,537 |
| 101-300 | 18 | 60,799 |
| 301-500 | 16 | 51,963 |
| 501-700 | 16 | 45,814 |
| 701-900 | 13 | 40,095 |
| 901-985 | 3 | 35,628 |

- The peak was 68,279 at iteration 113.
- The last intervals were 38,867 at iteration 922, 37,598 at 955 and 30,420 at 978. The watchdog also saw 35,658 at 967.
- `margin_abs_mean` stayed around 9-12k throughout, and `draw_rate` was about 0.
- **Reading:** the bank slid steadily from about 63k to about 36k over the run. That matches the reward never paying for earning. The econ penalty reads only the starvation and drought counters, and it saturates at the 0.2 cap after the first event. Banks enter the reward only as sign(bank_self − bank_opp) at game end.
- **Not attributed:** this is one seed, and the evidence is self-play telemetry only. The LR re-warm-up, the terminal 0.75 → 0.8 change and the J/2-final teacher remain confounded.

## Stop procedure (by PID; no pattern kill)

1. `kill -TERM -30021` (the process group of `run_resume.sh`), then `kill -TERM 30055 30060 30061 30062 30063` (torchrun and ranks 0-3). All six were gone within 4 s.
2. `kill 30351` stopped the watchdog. It was confirmed gone.
3. `nvidia-smi --query-compute-apps` then listed no compute apps, so the GPUs were free.
4. On the Mac, `kill 92562` stopped the copy-off loop. One final pass, the same script body run once, was then made by hand at 03:20:35-03:20:51Z.

## Custody (Mac: `/Users/poonszesen/kg-v3-runs/J2-resume-r0208-20260930/`)

- `checkpoints/20260930-020138/checkpoint_00_010_010_624.pt`, sha256 `a163f50a7458aa171bc99f7c6c3ef1aaf9864b960f6da42c956e45f62737d053`.
- `checkpoints/20260930-020138/checkpoint_last_best.pt`, sha256 `038d2573ca16978ece062a2cadc20cd86228e79b6bb2a2767643215d174a8a6d`.
- `J2-resume-r0208-20260930.log`, sha256 `db486e688afbe82ab4de53a2ff975f85a2757195a8030b16568684155c4d4bf7`. This is the final log, and it matches the pod.
- `J2-resume-watchdog.log`, sha256 `1bef17d90f2cd32fef5f330c52fb7da8c8c6dc26b7e54f0e6485e22392bbfb20`. It matches the pod.
- `SHA256SUMS` and `copyoff.log` are also in that folder.
- The pod keeps the run dir `/root/runs/J2-resume-r0208-20260930/20260930-020138/` and the receipts `/root/receipts/J2-resume-r0208-20260930/`. `times.txt` has no exit line, because the launcher's process group was signalled before its epilogue ran.
- W&B run `https://wandb.ai/spoon/kg-v3/runs/hz4bpjnq` is left as it ended. Its final state on W&B was not checked.
