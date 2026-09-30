# Result: control J/2 without bank reward, 4 ranks (pod abl4mvr5w1mmn4, 2026-09-30)

**Outcome: the run completed all 150 iterations (2,457,600 env steps, exit 0, 714 s wall). The watchdog never fired.** Halving J's LR slowed the teacher-KL rise before the peak (1.51 vs 2.32 at iteration 63) and kept the bank economy far above main-J-4rank at every matched step: 51,439 vs 5,684 at iteration 79. It did not keep the BC economy intact, though. `own_bank_mean` peaked at 81,644 (iteration 45) and fell 43% to 46,723 by iteration 90. It then plateaued at about 47,000-53,000 through iteration 147, and teacher KL kept rising to 3.53. This is one seed and one run, with nothing independently reviewed.

Run statement: `run-statement.md` (commit `82ad59b`), written before the launch.

## Identity

- Pod checkout `7e87f54`, 0 porcelain lines before and after. Hashes are in `pod-receipts/hashes.sha256`. The warm start is BC best `fd854587…6f51`, and `warm_start.json` confirms `model_only` with the same sha256.
- The launch script is `run_control.sh` (sha256 `c40ac0c1…62bc`). It is `main-J-4rank/run_main.sh` with only the name, the LR (`muon_lr=0.0001`, `adamw_lr=0.000005`) and the stop changed: `--max-env-steps 2457600` replaces `--max-runtime-hours 20`. The wrapper `/root/main-J/main_probe.py` (`26ba5b0c…`) and the watchdog `/root/main-J/watchdog.py` (`65050a89…`) are the main-J-4rank files, unchanged.
- Launched at 2026-09-30T01:01:26Z and ended at 01:13:20Z. The `run_control.sh` process group was 20537 and the watchdog pid was 20585, started once.
- W&B: **https://wandb.ai/spoon/kg-v3/runs/nw3klj2s**, online, experiment id `control-J2-4rank-20260930`, and `config_sha256` `87222bac…`. The run exited normally.
- Checkpoint: `checkpoint_final.pt` (51,920,207 B, env steps 2,457,600). Its sha256 is `80e5667ecbf36966a3edc94d2e831ee5372cd7e791b5d235cf30cbfc32e161dd`, and the pod and Mac copies match. Both are listed below.

## Bank economy (rank 0, both seats, 256 finished self-play games per interval)

| Iteration | own_bank_mean J/2 | margin_abs_mean J/2 | own_bank_mean J | margin_abs_mean J |
|---|---|---|---|---|
| 12 | 74,250 | 17,380 | 72,624 | 16,585 |
| 23 | 74,629 | 18,104 | 73,095 | 17,141 |
| 34 | 77,285 | 16,046 | 74,292 | 16,036 |
| 45 | 81,644 | 14,997 | 68,371 | 15,230 |
| 57 | 77,419 | 14,591 | 52,603 | 15,621 |
| 68 | 66,577 | 17,740 | 17,936 | 9,939 |
| 79 | 51,439 | 15,938 | 5,684 (J stopped) | 5,116 |
| 90 | 46,723 | 15,036 | - | - |
| 102 | 50,177 | 16,985 | - | - |
| 113 | 51,731 | 17,585 | - | - |
| 124 | 49,038 | 16,785 | - | - |
| 135 | 48,039 | 17,863 | - | - |
| 147 | 53,196 | 17,812 | - | - |

Unlike J, `margin_abs_mean` did not shrink. It stayed at 14,600-18,100 throughout, so games stayed decisive while the absolute bank level fell.

## Metrics at fixed iterations (rank 0; each cell is J/2 / J)

| Iteration | teacher/kl | teacher/unit_kind_kl | policy/approx_kl | train/advantage_std | train/explained_variance | LR (muon) |
|---|---|---|---|---|---|---|
| 12 | 0.0168 / 0.0431 | 0.00864 / 0.0222 | 0.00248 / 0.00382 | 0.603 / 0.562 | 0.344 / 0.474 | 1.92e-05 / 3.84e-05 |
| 34 | 0.309 / 0.644 | 0.196 / 0.478 | 0.00682 / 0.0113 | 0.467 / 0.388 | 0.577 / 0.626 | 5.44e-05 / 1.09e-04 |
| 45 | 0.776 / 0.967 | 0.552 / 0.728 | 0.00988 / 0.0157 | 0.368 / 0.277 | 0.685 / 0.645 | 7.20e-05 / 1.44e-04 |
| 57 | 1.06 / 1.57 | 0.642 / 1.02 | 0.00834 / 0.0157 | 0.352 / 0.251 | 0.459 / 0.367 | 9.12e-05 / 1.82e-04 |
| 63 | 1.51 / 2.32 | 0.945 / 1.70 | 0.0134 / 0.0221 | 0.109 / 0.0674 | 0.967 / 0.922 | 1.00e-04 / 2.00e-04 |
| 79 | 2.04 / 1.68 | 1.49 / 1.27 | 0.0115 / 0.0163 | 0.241 / 0.213 | 0.410 / 0.154 | 1.00e-04 / 2.00e-04 |
| 100 | 3.07 / - | 2.22 / - | 0.0131 / - | 0.0584 / - | 0.907 / - | 1.00e-04 / - |
| 125 | 2.50 / - | 1.50 / - | 0.00928 / - | 0.0713 / - | 0.508 / - | 1.00e-04 / - |
| 150 | 3.53 / - | 2.57 / - | 0.0122 / - | 0.0507 / - | 0.728 / - | 1.00e-04 / - |

The full per-iteration rank-0 table is `pod-receipts/iterations.txt` (from `main-J-4rank/parse_iterations.py`), and `comparison-vs-main-J.md` is the generated side-by-side. The replay alarm never fired. The largest `policy/logratio_mean` magnitude was 0.0164, against a limit of 0.05, and all losses were finite. `loss/teacher_kl_loss` reached 0.0177 at iteration 150, with coefficient 0.005.

## Reading

- **Prediction check:** the prediction held up to the peak, then failed. Teacher KL rose more slowly before the peak, and the bank held longer, reaching its maximum at iteration 45 where J had already started to fall. It still declined after the peak, so "may still decline" was right.
- **LR is not the whole story.** Teacher KL in J/2 went past J's maximum: 2.04 at iteration 79 and 3.53 at 150, against J's 2.32. Yet the bank stayed above 46,000. By iteration 150, J/2 had also accumulated more LR × steps than J had by iteration 79. So neither distance from the teacher nor cumulative update size alone predicts the collapse J showed. The per-step size (LR 2e-4 vs 1e-4) is the remaining difference. J/2's decline from 81,644 to about 50,000 is still unexplained. It fits what the self-play objective pushes toward, with no bank term, and it plateaued rather than collapsed within 150 iterations.
- **Unresolved attribution:** one seed each, so there is no variance estimate. The plateau could be a slower version of J's collapse that falls outside the 150-iteration window. The per-seat reward components are unread. The final checkpoint exists, so a J/2-final vs BC-best head-to-head is now possible. That is the discriminating check for whether the 50k-bank policy is a real economic regression or both seats' banks falling together.

## Throughput (game SPS, 16,384 env steps per iteration)

- J/2 iterations 3-150: **3,598 game SPS** from wall time. Rollout averaged 2.61 s, teacher 0.45 s and update about 1.45 s, and iteration 1 took 22.1 s (warm Inductor cache).
- Same window, iterations 3-57, where both economies were healthy: J/2 3,653 vs J 3,724 (−1.9%). The per-iteration time follows the game phase: native step about 15 ms after a reset and about 44 ms late in a game. J's whole-run 3,936 figure is inflated by its collapsed late economy, with shorter native steps, so it is not like-for-like with J/2's 150 iterations. Same hardware, precision and code, and no profiler was run.
- Peak GPU memory was 38,977 MiB of 97,887 (2 s samples). The GPUs were at 0 MiB and idle after exit.

## Custody

- Pod:
  - run dir `/root/runs/control-J2-4rank-20260930/20260930-010131/`, holding `checkpoint_final.pt`, config, attempts, warm_start and wandb
  - log `/root/runs/control-J2-4rank-20260930.log` (sha256 `ed3047a8…e976`)
  - `/root/runs/control-J2-watchdog.log`
  - receipts `/root/receipts/control-J2-4rank-20260930/`
- Mac: `/Users/poonszesen/kg-v3-runs/control-J2-4rank-20260930/`, holding `checkpoints/checkpoint_final.pt`, the run log, the watchdog log, receipts, the run dir without `.wandb`/`.pt`, and `SHA256SUMS`.
- Compact copies are in `pod-receipts/`.

## Spend

The run took 01:01:26-01:13:20Z (11.9 min), about **$1.66** at $8.36/h. Including setup and copy-off on the pod (about 00:58-01:20Z), that is about $3.07. The pod stays **idle but billing**. Stopping it is the owner's call.
