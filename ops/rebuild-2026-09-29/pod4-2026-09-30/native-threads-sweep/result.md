# Result: 4-rank native_threads sweep (recipe J, pod abl4mvr5w1mmn4, 2026-09-30 00:07-00:31Z)

Throughput diagnostic only. No checkpoint from these runs is used, selected or ranked. The run statement and its two addenda are in `run-statement.md`, written before each launch. Nothing here has been independently reviewed.

## Recommendation

**`env.native_threads: 4`.** With the unbound rows averaged over two samples, N=4 had the highest game SPS (3,256). N=8 (3,104) was 4.7% lower, N=2 (3,093) 5.0% lower and N=16 (2,960, one sample) 9.1% lower, so none of them ties under the 3% rule. With CPU binding, N=4 (3,450) and N=8 (3,385) are within 3% of each other, so the tie also goes to N=4.

The effect of N is small. Per-rank native step time stayed at about 32-46 ms for every N, and the process used only 1.15 cores at N=2, 1.3 at N=4, 1.6-1.8 at N=8 and 2.5-2.9 at N=16. The spread in SPS between rows follows the slowest rank's step time, because `time/rollout_seconds` is the maximum over ranks. The rollout is about 80% native step time (slowest rank's step ms x 64 / rollout s = 0.79-0.83) in every row.

**Separate option: per-rank CPU binding to the GPU's NUMA node** gave the two best single samples: N=4 bound 3,450 (+6.0% over the unbound N=4 mean) and N=8 bound 3,385 (+9.1% over unbound N=8). Per-rank step times were also the most even (32.1-36.3 ms). Each of these has one sample only. Binding is not part of `run_ppo`. In these runs it came from `nt_probe.py`, which calls `os.sched_setaffinity` before torch loads: GPU0/1 get CPUs 0-63,128-191 and GPU2/3 get 64-127,192-255. A main run would need the same thing from a launch wrapper (for example `taskset` per rank). Memory binding is not available on this pod (see Failures).

## Table (iterations 3-10, rank 0 wall clock; 8 x 16,384 = 131,072 global env steps per row)

Game SPS is 131,072 divided by the wall time from the end of iteration 2 to the end of iteration 10. That time includes logging between iterations. Learner-seat SPS is the rise in `train/player_step_total` over the same time. It is exactly 2x game SPS here, because both seats are the learner and no game ends in the window. The phase columns are the trainer's own means of `time/rollout_seconds`, `time/teacher_seconds` and `time/update_seconds` (each the maximum over ranks). Native step ms is the mean wall time of each `KaggricultureVectorizedEnv.step` call (fence, native step and buffer publish), for ranks 0/1/2/3. Cores are the process CPU seconds per wall second, per rank. Machine CPU is the busy share of all 256 vCPUs in `/proc/stat`.

| Row | N | Bind | Mean iter s | Game SPS | Seat SPS | Rollout s | Teacher s | Update s | Native step ms r0/r1/r2/r3 | Cores r0/r1/r2/r3 | Machine CPU % | W&B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nt2 | 2 | - | 5.34 | 3,069 | 6,137 | 3.40 | 0.46 | 1.48 | 37.5 / 33.5 / 37.8 / 44.2 | 1.15 / 1.14 / 1.15 / 1.16 | 2.0 | `n3l8unt3` |
| nt2-r2 | 2 | - | 5.25 | 3,117 | 6,234 | 3.32 | 0.46 | 1.47 | 36.8 / 37.5 / 42.8 / 34.3 | 1.15 / 1.15 / 1.16 / 1.15 | 2.0 | `8bp284rk` |
| nt4 | 4 | - | 5.00 | 3,275 | 6,551 | 3.08 | 0.46 | 1.46 | 37.3 / 33.7 / 36.2 / 38.7 | 1.31 / 1.30 / 1.31 / 1.33 | 2.2 | `qzlvqbbr` |
| nt4-r2 | 4 | - | 5.06 | 3,236 | 6,473 | 3.12 | 0.46 | 1.48 | 38.1 / 39.4 / 34.6 / 39.1 | 1.32 / 1.33 / 1.30 / 1.32 | 2.2 | `5wqdfg0u` |
| nt8 | 8 | - | 5.22 | 3,138 | 6,276 | 3.28 | 0.46 | 1.48 | 38.6 / 34.8 / 32.9 / 41.6 | 1.72 / 1.60 / 1.54 / 1.77 | 2.8 | `8vxsttxy` |
| nt8-r2 | 8 | - | 5.34 | 3,070 | 6,139 | 3.41 | 0.46 | 1.47 | 34.3 / 39.7 / 43.8 / 39.3 | 1.56 / 1.71 / 1.78 / 1.70 | 2.9 | `a0h9fsdl` |
| nt16 | 16 | - | 5.53 | 2,960 | 5,920 | 3.60 | 0.46 | 1.47 | 39.8 / 45.2 / 46.2 / 42.6 | 2.46 / 2.85 / 2.86 / 2.67 | 4.4 | `5azu2fjs` |
| nt8-numa | 8 | CPU+mem | failed at startup (EPERM) | | | | | | | | | none |
| nt8-cpubind | 8 | CPU | 4.84 | 3,385 | 6,771 | 2.90 | 0.46 | 1.47 | 35.5 / 36.3 / 34.3 / 34.2 | 1.69 / 1.68 / 1.65 / 1.66 | 2.8 | `v45oe5ol` |
| nt4-cpubind | 4 | CPU | 4.75 | 3,450 | 6,899 | 2.81 | 0.46 | 1.47 | 34.9 / 32.1 / 33.3 / 33.4 | 1.32 / 1.30 / 1.30 / 1.30 | 2.4 | `jejoor29` |

Two-sample means of unbound game SPS: N=2 3,093, N=4 3,256, N=8 3,104. The repeat moved N=2 by +1.6%, N=4 by -1.2% and N=8 by -2.2%. Run-to-run spread is therefore about 1-2%, which is less than the gaps the rule decided on.

W&B runs are in `spoon/kg-v3`, group `kg-v3-native-threads-sweep`, all online. The first row's iteration 1 took 60.9 s because of the cold compile. Later rows took 21.6-24.9 s using the shared Inductor/Triton cache.

## Failures and warnings

- `nt8-numa` stopped within 1 s on all 4 ranks with `PermissionError: [Errno 1] set_mempolicy failed`. The container's syscall filter denies `set_mempolicy`, so `numactl --membind` would fail the same way (and `numactl` is not installed). The fix would weaken container security, which is out of scope. Memory binding is untested. Placement falls back to the kernel's default first-touch policy.
- The other 9 rows exited 0 after 10 iterations, with W&B online. The logs contain no warning, error or traceback lines besides the probe's own records. Peak GPU memory in the 2 s `nvidia-smi` samples was 38,977-38,979 MiB per GPU of 97,887 MiB, the same in every row.

## Identity and custody

- Pod checkout `7e87f5420b3696141ea41f6453d4fafb2748eea0`, 0 porcelain lines before every row (`pod-receipts/*/git_state.txt`). Config and source hashes are in each row's `hashes.sha256`. `nt_probe.py` had sha256 `3a229f83…` for nt2, nt4, nt8, nt16 and nt8-numa (the version in commit `2c20e34`). The other rows used `9bc34b2e…`, the committed version with the `KG_NT_NUMA=cpu` mode added; the observation code is unchanged.
- Command: `run_sweep.sh LABEL N BIND` (committed), recipe J exactly as in the run statement. `OMP_NUM_THREADS=1`. The Inductor and Triton caches were shared under `/root/sweep-cache/`.
- `pod-receipts/` holds every row's full `run.log`, timing, idle/after `nvidia-smi`, 2 s GPU samples, and `summary.json`, the output of `summarize.py`. The run directories (`/root/runs/sweep-*`, with W&B files and final checkpoints) stay on the pod and are not needed.
- Pod time for the sweep was 00:07:50-00:30:31Z (22.7 min, about $3.16 at $8.36/h). All 4 GPUs showed 0 MiB after the last row. The pod was left running and idle, and no main run was launched.

## Limits

- Two samples at most per row. The binding rows have one sample each. The +6% binding gain at N=4 is bigger than the 1-2% repeat spread, but it has not been replicated.
- The window covers steps 128-640 of the first 720-step game. Resets, game ends, the teacher refresh evaluation (every 10M steps) and checkpoint writes are not sampled. The main run's average SPS will be lower than these figures by those costs.
- No `nsys` timeline was taken. The claim that the native step is mostly serial comes from the flat step ms and the low core counts, together with a reading of `src/kaggriculture/env.rs` `prepare_step`. There, token admission, grammar decode, the clone of every env's game state and the `ObsStaging::new` allocation run outside the rayon pool, and only the per-env `step_with_market_metrics` block runs in `pool.install`. That split has not been profiled, and the serial share is unmeasured. The real lever for rollout time is that serial path, not `native_threads`.
