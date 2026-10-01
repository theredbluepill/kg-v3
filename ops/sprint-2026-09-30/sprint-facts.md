# Final-sprint facts (2026-09-30 15:30Z – 23:59Z), for cookbook recording

## Corrections (2026-10-01, after a review against the evidence)

The original text below is kept as written. Where it conflicts with this list, this list wins.

1. **Custody.** "its last run's training log/receipts not archived" is wrong. The Mac copy-off loop brought the r4zqqs49 log (14,256,784 B; 1,008 iteration records; last rank-0 iteration 705; ends with the SIGTERM) and its 15 receipts (including `stop.txt`) to `/Users/poonszesen/kg-v3-runs/earn720-r30e01w30-8xh200-from-130M-sps-20261001/`. The receipts of all four 8x H200 runs are now in `h200-run-receipts/`; the logs are hashed in `MANIFEST-skipped.tsv`.
2. **Promotion timing.** The 90M and 100M relaunches came before their panels ran. The from-90M launch was at 19:12:25Z, and the 90M panel ran from 19:12:57Z to about 19:18:57Z (`anchor-games/cpu-pod/eval-90M.log`). The from-100M launch was at 19:46:23Z, and the 100M panel started at 19:46:35Z (`eval-100M.log`). Only the 130M relaunch (21:03:58Z) followed its panel, which finished at about 21:02Z (`eval-130M.log`). Launch times: `h200-run-receipts/<run>/receipts/times.txt`.
3. **Who promoted.** Only 90M is labelled "owner manual promotion" here. 100M and 130M are labelled "(manual)" with no actor. No verbatim owner quote is recorded for any of the three.
4. **Sampling flips.** "0.6% token flips" counts rows, not tokens: 15 of 2,400 rows (0.625%) had any token mismatch (`throughput/parity-gpu/results.json`, `sampling_identical_noise`).
5. **Parity files.** The compiled-vs-eager numbers (max .100, implied ratio, values, gradients 1–9%, start-of-update ratio, clipfrac, stored-vs-replay max .67, sampling) come from `throughput/parity-gpu/results.json` (seed 4242). The fp64 floor, the eager-vs-fp64 gradient error and the stored-vs-replay max .37 come from `results2.json` (seed 4243). "Files: scratchpad/parity-gpu/" now means `throughput/parity-gpu/`.
6. **Diagnostic arm order.** First-update approx_kl / clipfrac "(A/B/C/D)" are off .00186/.0087, all Python switches .00181/.0082, native + all switches .00184/.0083 and native only .00181/.0080. Native + compile_actor_heads only was .00183/.0089. The native-only approx_kl is 0.0018148, so it rounds to .00181, not .00182 (`throughput/sps-diag-evidence/sps-diag-*-20261001.log`, iteration-1 `[nt-probe]` record).

All times UTC. "Panel" = fixed-shop anchor panel: 8 seeds (93001–93008) x 2 seats x 3 anchors (smaller_market_shock, cha22, v56) = 48 games, rule 1 on unless said, paired by (anchor, seed, seat); SE over the 8 seed means. Selection panel, not held-out.

## Reward / recipe (owner decisions, verbatim where known)
- Recipe of all sprint runs: configs/kaggriculture_4rank_margin.yaml + -o rl.horizon=720 rl.segments_per_minibatch=1 rl.gae_lambda=1.0, reward econ_bank 0.3/150000/cap 0.3, econ_margin 0.3/100000/cap 0.3, econ_shaping 0.01 cap 0.1, win/loss 0.30 (owner: "give back 0.075 ... win/lose to be 0.3"). Muon 1e-4 / AdamW 5e-6. checkpoint_freq 10M; promotion at >= 0.70 vs last_best over n_envs games; teacher_mode last_best (KL 0.005).
- LR: owner "let anchors speak will be good" — keep 1e-4 unless anchors stall + KL/clip shrink; never changed.
- Owner declined lambda scheduling ("i meant let it be") and PFSP league training ("maybe it's ok let's just focus in pure self-play, no need league"); PFSP left as future work. An implementation was started (branch kg/pfsp, worktree /Users/poonszesen/kg-v3-pfsp) and stopped with no commits.
- Owner: "do not spend submission slot unless we agreed tgt."

## Runs and checkpoints (W&B spoon/kg-v3)
| run | pod | start ckpt | W&B | notes |
|---|---|---|---|---|
| earn720-r30e01w30-from-c50-4rank-20260930 | 4x RTX PRO 6000 (abl4mvr5w1mmn4) | c50 (0cc80065) | 48gyi9m5 | ~2,210 env steps/s; 60M ckpt 20b1f795 (not promoted: 0.50 vs c50, +1.6k) |
| earn720-r30e01w30-8xh200-from-60M-20261001 | 8x H200 (7gbzus3pufmik6, driver 570.211.01, 96 vCPU, cgroup quota 81.6 CPU) | 60M | xuft2e2i | 70M promoted 0.80 (+3.0k), 80M promoted 0.95 (+4.1k), 90M 0.60 (+2.7k) not promoted |
| ...-from-90M-20261001 | 8xH200 | 90M (owner manual promotion) | 3w2ag52m | 100M 0.65 (+2.5k) not promoted |
| ...-from-100M-sps-20261001 | 8xH200, code 17b3068d + rl.compile_actor_heads=true | 100M (manual) | dxhey4da | 110M 0.50 (+1.3k), 120M 0.60 (+2.8k), 130M 0.65 (+4.1k) none promoted |
| ...-from-130M-sps-20261001 | same code | 130M (manual) | r4zqqs49 | 140M 0.70 promoted, 150M 0.65, 160M 0.60, 170M 0.80 promoted (+3.3k), 180M 0.90 promoted (+2.1k), 190M 0.55, 200M 0.65 (−0.4k), 210M 0.65 (+0.9k); stopped 23:51Z |
Checkpoint shas: 60M 20b1f795, 70M 4c8b9483, 80M ded916bd, 90M 6b196ef3, 100M 8e520566, 110M 4c85eb6a, 120M b4b7d784, 130M 86df0ee4, 200M a96d9c62, 210M 790b64d8. Local copies under /Users/poonszesen/kg-v3-runs/<run>/{checkpoints,ckpt-XXM}/ with SHA256SUMS.
Relaunch cost: each manual promotion/relaunch resets the optimizer (model_only load) and reruns the ~1,000-step LR warm-up (~50 iterations).

## Anchor panel results (rule 1 on; margins per game)
| ckpt | W-L | mean margin | notes |
|---|---|---|---|
| 08bc (Kaggle, rule off) | 0-48 | −11.9k | |
| c50 | 6-42 | −7.2k | |
| 60M | 8-40 | −3.3k | +3.9k vs c50 (off arm), self-play said tie |
| 70M | 16-32 | −1.55k | +1.75k ± 1.0k vs 60M |
| 80M | 30-18 | −0.55k | +1.0k ± 0.9k vs 70M |
| 90M | 38-10 | +1.94k | +2.5k ± 0.8k vs 80M; gain mostly from lowering anchor bank (anchor 94–96k vs 99–100k) |
| 100M | 40-8 | +2.22k | +279 ± 477 vs 90M |
| 110M | 40-8 | +2.08k | −140 ± 514 vs 100M |
| 120M | 30-18 | +0.49k | −1.73k ± 1.13k vs 100M, 3/8 seeds: REGRESSION despite self-play 0.60/+2.8k vs 100M |
| 130M | 46-2 | +2.85k | +628 ± 462 vs 100M; cha22 16/16, v56 16/16 |
| 170M | 48-0 | +6.90k | +4.96k ± 1.13k vs 90M |
| 180M | 48-0 | +7.32k | +417 ± 869 vs 170M; +5.38k ± 0.87k vs 90M (8/8 seeds) |
| 190M | 42-6 | +5.89k | |
| 200M | 48-0 | +7.89k | +568 ± 723 vs 180M |
| 210M | 48-0 | +7.51k | −383 ± 537 vs 200M |
Self-play vs anchors disagreed twice: 60M (self-play tie, anchors clearly better) and 120M (self-play better, anchors clearly worse). 190M (self-play 0.55) also dipped on the panel.
Receipts: /Users/poonszesen/kg-v3-int/ops/earn-money-2026-09-30/anchor-games/games-fixedshop{,-linux}/<label>-ft-on/receipts/, eval-60M/, eval-70M/, eval-80M/ tables; cpu-pod/eval-*.log.

## Kaggle submissions (owner-approved each time)
| ref | ckpt | archive sha256 | time |
|---|---|---|---|
| 56711278 | 08bc (earlier) | 8283e676… | 09-30 15:14 — score 1004.1 at ~19:30 |
| 56716929 | 90M | 5e460d20… | 09-30 19:26 |
| 56720629 | 170M | 428a63d7… | 09-30 22:47 |
| 56722061 | 210M | 45efe071… | 09-30 23:50:02 (final slot) |
Receipts committed on branch kg/package-60m (worktree /Users/poonszesen/kg-v3-pkg60): 31443af3 (90M), a22e6c4d (170M), 728c4f4b (210M). Leaderboard at 19:36Z: 10,225 teams; silver (top 5%) ≈ 2,087, bronze (top 10%) ≈ 1,854, gold ≈ top ~30 (~2,750+). Team rank 2,987 at 23:03Z. Deadline 2026-09-30 23:59:00 UTC (kaggle CLI).

## Throughput
- 4x RTX PRO 6000 (12 envs/rank): ~2,210 env steps/s. 8x H200 (20 envs/rank, 115 GB of 141): ~3,650.
- Clock keepers: host uses intel_cpufreq passive + schedutil (800 MHz–4.0 GHz); rayon env workers sampled at median 800 MHz vs main thread 3.2 GHz; burst test 2.7–2.9x slower at ~37% duty; with a SCHED_IDLE spinner 1.02x. Fix: one SCHED_IDLE busy loop per CPU (`/root/sps-diag/clock_keeper.sh`: `setsid taskset -c $c chrt -i 0 bash -c 'exec -a kg-clock-keeper-spin bash -c "while :; do :; done"'`). Result: rollout 27 s -> 18 s, iteration 31.8 -> 21.9 s, 3,620 -> ~5,250 env steps/s (+45%), no code change. Caveat: container cgroup quota 81.6 CPUs; spinners count against it (throttling ~15%, main-thread run-queue wait 11.6%). Negative result: 14 keepers/node (56 total) was WORSE (3,800–4,450 env steps/s) — workers land on unspun slow cores; restored 96. Keepers must be stopped before a restart/compile and restarted after the first iteration (a shell bug once skipped the restart: ~3 min at 3,700 env steps/s). SCHED_IDLE spinners also keep cores out of deep C-states.
- Diagnosis (py-spy blocked by ptrace_scope/CAP_SYS_PTRACE; used run's own [nt-probe] timers + /proc sampling): env.step ~68% of rollout; main thread serial Rust work (ObsStaging publish memcpy, grammar decode/serde_json, Game clones, 5.6 MB staging alloc per step); ~600 eager ops per forward in actor heads; 6 host-device syncs per step; nothing overlaps. Old pod per-env native 0.72–0.85 ms vs H200 host 0.83–1.31 ms.
- Codex (codex exec, sandbox could not commit to external git metadata; delivered bundles, imported by hand): native parallel step fa0cffc2 (branch kg/sps-rollout; bitwise parity vs pristine b2276bc5 golden at 1/4/8 threads over 5,760 transitions / 8 games; 413 Rust + 3,016 Python tests) and Python switches d564cd2 (branch kg/sps-python: rl.compile_actor_heads, rl.rollout_packing, rl.pinned_action_d2h, env.skip_reward_telemetry_validation; all default off; switch-off byte-identical). Combined merge 17b3068d on kg/sps-combined (worktree /Users/poonszesen/kg-v3-sps-all).
- 1x H200 diag (single rank, keepers on, 10 CPUs): baseline 840 env steps/s; python switches all on 929 (+10.5%); native + all switches 1,268 (+51%); native only 1,034 (+23%); native + compile_actor_heads only 1,263. Packing/pinned D2H/telemetry add nothing measurable -> left off. First-update approx_kl .00186/.00181/.00184/.00182, clipfrac .0087/.0082/.0083/.0080 (A/B/C/D).
- Independent review (4 Claude reviewers + Codex, workflow): GO WITH CONDITIONS; no confirmed blocker. Minor: compiled heads fullgraph dynamic=False with an 8-variant recompile limit that crashes (production uses 5); flag written to config.yaml (old code rejects it on resume); rank-0 first-eval compile under NCCL timeout; test gaps (replay_parity.rs:372 compares new API to itself; golden lacks fixed-opponent path). Codex: VERDICT GO.
- Live 8xH200 with 17b3068d + rl.compile_actor_heads=true: ~8,290–8,310 env steps/s (rollout 10.5 s, update 2.7 s, teacher 0.7 s, iteration 13.9 s; +49% over keepers alone); 5 compiled shapes per rank ("Online softmax is disabled" warning x5 per rank, harmless); KL/clipfrac/teacher-KL/EV curves match the old-code run iteration by iteration.
- GPU parity test (1x H200 driver 570.211.01, 110M weights, bf16, 20 envs x 720 steps): compiled vs eager heads per-player logp median .0057 / p99.9 .070 / max .100 (eager vs fp64 floor: .0057/.053/.068); mean signed bias +2e-5; implied ratio 0.91–1.105 mean 1.0001; values bitwise identical; grads rel err 1–9% cos >= .996 (eager vs fp64 6–9.5%); sampling with identical noise: 0.6% token flips. Rollout and update both use `_policy_chunk` -> `_compiled_actor_core` (kaggriculture.py:1170). FINDING (pre-existing, independent of speed-ups): trunk bf16 numerics depend on batch shape (40-row rollout vs 1,440-row update): stored-vs-replay logp median .021, p99.9 .23, max .37–.67; start-of-update ratio 0.51–1.36 (mean 0.9999), clipfrac 0.5–1.1% before any optimizer step. Files: scratchpad/parity-gpu/.
- H200 driver gate: ATEN-only GEMM correctness probe re-run on driver 570.211.01 (all pass, template census 0 triton_tem_, default-backend control reproduced the fault); 570.211.01 added on branch kg/h200-driver-gate commit b2276bc5 with cookbook decision update.

## Evaluation harness
- CPU pod (32 vCPU AMD EPYC 9965, cpu3c $0.96/hr): Python 3.11 / torch 2.6.0 CPU; fixed-shop kaggle-environments copy; Linux builds of cha22_agent (bd7ffaa0…) and v56_agent (34458b2a…) from v2 commit 23f75800 engine_rs (read-only export; replaying 23,008 recorded anchor actions reproduced all). Plays the real Linux Kaggle package. 70M and 60M reproduced the Mac results 48/48 with identical final banks and actions. ~6 min games + ~1 min package per 48-game panel. One command: /Users/poonszesen/kg-v3-int/ops/earn-money-2026-09-30/anchor-games/cpu-pod/eval_ckpt.sh CKPT CONFIG LABEL [--rule1-off] [--package DIR] (setup_pod.sh once).
- Mac fast path: rule-on arm played, rule-off derived exactly from the pre-rule final action (validated 48/48 on 60M; 3/3 played spot checks equal).

## Rules
- Rule 1 (final-turn liquidation; kg/submit-08bc 4c99768a): c50 A/B +874 ± 157 per game, all 48 better, 0 flips; 60M +870 ± 242; on 08bc ladder games would flip 2 of 7 losses (11-7 -> 13-5). Packager option --final-turn-liquidation (kg/package-60m de26cd68) bakes it into main.py.
- Rule 2 (late-investment filter; a53ad2bf, merged 31c99619): −1 ± 21 per game on c50 (only wheat seeds blocked); owner: "let's not apply rule2."

## Custody / pods
- 4x RTX pod archived to /Users/poonszesen/kg-v3-runs/pod-abl4mvr5w1mmn4-archive (8.5 GB, 36 .pt sha-verified) then terminated.
- Diagnostic H200 pods (huum96u1xqtzu3, ocxfe0zbrwoj7b) terminated after archiving results (scratchpad/sps-diag-pod-archive, parity-gpu).
- 8xH200 pod terminated at owner request right after the final submission; its last run's training log/receipts not archived (metrics in W&B r4zqqs49); all checkpoints 140M–210M + last_best were already on the Mac.
- CPU eval pod terminated after the 210M panel. RunPod list empty at 23:59Z.
