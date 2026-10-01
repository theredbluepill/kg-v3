---
type: "Episode"
title: "The final sprint took 720-turn self-play from c50 to a 48-0 anchor panel"
description: "Closed result episode, 2026-09-30 15:30Z to 23:59Z UTC. One chained mirror self-play lineage (720-turn horizon, lambda 1, reward own bank 0.30 + cash difference 0.30 + per-event econ shaping 0.01 cap 0.1 + win/loss 0.30, Muon 1e-4) ran from the owner-promoted c50 (pcy5knet 50M) to 210M over five W&B runs (48gyi9m5, xuft2e2i, 3w2ag52m, dxhey4da, r4zqqs49). On the 48-game fixed-shop anchor panel (8 seeds x 2 seats x smaller_market_shock, cha22, v56; rule 1 on) W-L went c50 6-42 (-7.2k mean margin) -> 90M 38-10 (+1.94k) -> 170M 48-0 (+6.90k) -> 210M 48-0 (+7.51k). Self-play promotion and the panel disagreed at 60M and 120M. Throughput rose from ~2,210 (4x RTX PRO 6000) to ~8,300 env steps/s (8x H200, clock keepers, native parallel step, compiled heads). The owner submitted 90M, 170M and 210M (refs 56716929, 56720629, 56722061). Selection panel, one training seed; ladder ranks of the three submissions are not recorded. The four 8x H200 training logs are on the Mac (hashed), and their receipts are in ops/sprint-2026-09-30/h200-run-receipts/."
tags: ["kaggriculture-v3", "episode", "training", "evaluation", "submission", "throughput"]
status: "closed"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-10-01"}
sources:
  - resource: "user-directive:2026-09-30:give-back-0.075-to-win-lose-to-make-it-0.3"
  - resource: "user-directive:2026-09-30:let-anchors-speak-will-be-good"
  - resource: "user-directive:2026-10-01:record-pending-knowledge-into-the-cookbook-after-eval"
  - resource: "user-directive:2026-10-01:its-silver-zone"
  - resource: "user-directive:2026-10-01:one-day-earlier-and-2x-model-params-could-have-gone-further"
  - resource: "user-directive:2026-09-30:relaunch-from-90m-100m-130m"
  - resource: "repository:ops/sprint-2026-09-30/README.md"
  - resource: "repository:ops/sprint-2026-09-30/sprint-facts.md"
  - resource: "repository:ops/sprint-2026-09-30/MANIFEST-skipped.tsv"
  - resource: "repository:ops/sprint-2026-09-30/clock_keeper.sh"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/cpu-pod/README.md"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/cpu-pod/eval_ckpt.sh"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/cpu-pod/eval-90M.log"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/cpu-pod/eval-120M.log"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/cpu-pod/eval-130M.log"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/cpu-pod/eval-170M.log"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/cpu-pod/eval-180M.log"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/cpu-pod/eval-200M.log"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/cpu-pod/eval-210M.log"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/games-fixedshop-linux/120M-ft-on/summary.md"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/games-fixedshop-linux/170M-ft-on/summary.md"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/games-fixedshop-linux/210M-ft-on/summary.md"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/eval-60M/eval60m_tables.md"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/eval-70M/eval70m_tables.md"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/eval-80M/eval80m_tables.md"
  - resource: "repository:ops/sprint-2026-09-30/throughput/parity-gpu/results.json"
  - resource: "repository:ops/sprint-2026-09-30/throughput/parity-gpu/results2.json"
  - resource: "repository:ops/sprint-2026-09-30/h200-run-receipts/earn720-r30e01w30-8xh200-from-60M-20261001/receipts/launch.txt"
  - resource: "repository:ops/sprint-2026-09-30/h200-run-receipts/earn720-r30e01w30-8xh200-from-90M-20261001/receipts/times.txt"
  - resource: "repository:ops/sprint-2026-09-30/h200-run-receipts/earn720-r30e01w30-8xh200-from-100M-sps-20261001/receipts/times.txt"
  - resource: "repository:ops/sprint-2026-09-30/h200-run-receipts/earn720-r30e01w30-8xh200-from-130M-sps-20261001/receipts/times.txt"
  - resource: "repository:ops/sprint-2026-09-30/h200-run-receipts/earn720-r30e01w30-8xh200-from-130M-sps-20261001/receipts/stop.txt"
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/earnC/launch.md"
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/earnD/launch.md"
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/earnE/launch.md"
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/earnF/launch.md"
  - resource: "repository:ops/rebuild-2026-09-29/sprint-8gpu/launch.sh"
  - resource: "repository:ops/rebuild-2026-09-29/sprint-8gpu/mac_side.md"
  - resource: "repository:ops/submit-90m-2026-10-01/receipt.md"
  - resource: "repository:ops/submit-170m-2026-10-01/receipt.md"
  - resource: "repository:ops/submit-210m-2026-10-01/receipt.md"
  - resource: "repository:configs/kaggriculture_4rank_margin.yaml"
  - resource: "repository:cookbook/references/clock-keepers-native-parallel-step-and-compiled-heads-lifted-h200-rollout-throughput.md"
  - resource: "wandb-run:spoon/kg-v3/48gyi9m5"
  - resource: "wandb-run:spoon/kg-v3/xuft2e2i"
  - resource: "wandb-run:spoon/kg-v3/3w2ag52m"
  - resource: "wandb-run:spoon/kg-v3/dxhey4da"
  - resource: "wandb-run:spoon/kg-v3/r4zqqs49"
  - resource: "kaggle-submission:kaggriculture/56716929"
  - resource: "kaggle-submission:kaggriculture/56720629"
  - resource: "kaggle-submission:kaggriculture/56722061"
---

# The final sprint took 720-turn self-play from c50 to a 48-0 anchor panel

## Outcome

One chained self-play lineage ran from c50 to 210M in the last 8.5 hours before the deadline (2026-09-30 23:59:00 UTC). On the fixed-shop anchor panel it went from 6-42 to 48-0, and the mean margin went from -7.2k to +7.5k per game. The owner submitted three checkpoints from it. This is a closed episode: one training seed, a selection panel, and no recorded ladder rank for the three new submissions.

All times are UTC. The consolidated numbers are in `ops/sprint-2026-09-30/sprint-facts.md`; the evidence index is `ops/sprint-2026-09-30/README.md`.

## Recipe

- **Owner reward (verbatim).** "actually give back 0.075 from [own bank 0.375 / cash difference 0.375] to win/lose loss to make win/lose to be 0.3, and relaunch the current run again. Let's do that." (`earnF/launch.md`). The reward is econ_bank 0.3/150000/cap 0.3, econ_margin 0.3/100000/cap 0.3, econ_shaping 0.01 (starvation 4, drought 1) with econ_cap 0.1, and terminal win/loss 0.30. It followed the 0.375/0.375/0.25, 0.375/0.375/0.10/0.15 and 0.375/0.375/0.01/0.15 relaunches (`earnC`, `earnD`, `earnE`), each stopped within 30 iterations.
- **Training.** `configs/kaggriculture_4rank_margin.yaml` with `rl.horizon=720 rl.segments_per_minibatch=1 rl.gae_lambda=1.0` (`sprint-8gpu/launch.sh`). Muon 1e-4 / AdamW 5e-6. Checkpoint every 10M env steps; promotion at ≥ 0.70 against last_best; teacher_mode last_best (KL 0.005).
- **LR.** The owner said "let anchors speak will be good". The rule was to keep 1e-4 unless the anchors stalled and KL and clip fraction shrank. That never happened, so the LR was never changed.
- **Lineage before the sprint.** pcy5knet promoted 60f2 at 30M and 08bc at 40M. Its 50M checkpoint (`0cc80065`, "c50") was not promoted (8/12), and the owner promoted it by hand (`earnC/launch.md`).
- **Batch change.** The 8x H200 runs used 20 envs per rank (`h200-run-receipts/<run>/receipts/launch.txt`), against 12 on 4 GPUs. The owner accepted that 8 GPUs change the per-update batch ("is ok for 1:1", `mac_side.md`). That acceptance was given for the kit's recipe of 12 envs per rank. `mac_side.md` says any change to `--n-envs` must be told to the owner, and no record of the owner agreeing to 20 envs per rank was found. Each manual relaunch or relaunch loaded the model only (`model_only`), so it reset the optimizer and reran the ~1,000-step LR warm-up (~50 iterations).

## Runs (W&B spoon/kg-v3)

| Run | Hardware | Start | W&B | Self-play evaluations (vs last_best) |
|---|---|---|---|---|
| earn720-r30e01w30-from-c50-4rank-20260930 | 4x RTX PRO 6000 | c50 | 48gyi9m5 | 60M 0.50 (+1.6k), not promoted |
| …-8xh200-from-60M-20261001 | 8x H200, driver 570.211.01 | 60M | xuft2e2i | 70M 0.80 promoted; 80M 0.95 promoted; 90M 0.60 |
| …-from-90M-20261001 | 8x H200 | 90M (manual; labelled owner promotion) | 3w2ag52m | 100M 0.65 |
| …-from-100M-sps-20261001 | 8x H200, code 17b3068d + compiled heads | 100M (manual) | dxhey4da | 110M 0.50, 120M 0.60, 130M 0.65 |
| …-from-130M-sps-20261001 | same | 130M (manual) | r4zqqs49 | 140M 0.70 promoted, 150M 0.65, 160M 0.60, 170M 0.80 promoted, 180M 0.90 promoted, 190M 0.55, 200M 0.65, 210M 0.65; stopped 23:51 |

Trainer promotions: 70M, 80M, 140M, 170M and 180M. The 90M, 100M and 130M starts were manual relaunches. All three were owner-directed (verbatim orders in the [[../decisions/hold-the-learning-rate-and-let-the-anchor-panel-speak|LR Decision]]; e.g. 90M: "it's ok let's pull 90m to local, and we promote it, start a run with 90M, then eval with 90M."). The 90M and 100M relaunches (19:12:25 and 19:46:23) came before their panels ran (from 19:12:57 and 19:46:35). Only the 130M relaunch (21:03:58) followed its panel, which finished at about 21:02 ([[../decisions/hold-the-learning-rate-and-let-the-anchor-panel-speak|LR Decision]]).

Checkpoint sha256 prefixes:

| Checkpoint | sha256 prefix |
|---|---|
| 60M | `20b1f795` |
| 70M | `4c8b9483` |
| 80M | `ded916bd` |
| 90M | `6b196ef3` |
| 100M | `8e520566` |
| 110M | `4c85eb6a` |
| 120M | `b4b7d784` |
| 130M | `86df0ee4` |
| 200M | `a96d9c62` |
| 210M | `790b64d8` |

## Anchor panel trajectory

Each row is 48 games: seeds 93001–93008 x both seats x {smaller_market_shock, cha22, v56}. The engine is the fixed-shop copy of the Kaggle engine (`kaggriculture.py` sha256 `f73d27ce…`, `cpu-pod/README.md`), and rule 1 (final-turn liquidation) is on unless noted. Games are paired by (anchor, seed, seat), and the SE is over the 8 seed means. Rows from 90M on were played on the Linux CPU pod with the real Linux package (`cpu-pod/eval-*.log`, `games-fixedshop-linux/`); 60M–80M also have Mac fast-path tables (`eval-60M/`, `eval-70M/`, `eval-80M/`).

| Checkpoint | W-L | Mean margin | Paired delta |
|---|---|---|---|
| 08bc (rule off) | 0-48 | −11.9k | |
| c50 | 6-42 | −7.2k | |
| 60M | 8-40 | −3.3k | +3.9k vs c50 (rule-off arm) |
| 70M | 16-32 | −1.55k | +1.75k ± 1.0k vs 60M |
| 80M | 30-18 | −0.55k | +1.0k ± 0.9k vs 70M |
| 90M | 38-10 | +1.94k | +2.5k ± 0.8k vs 80M |
| 100M | 40-8 | +2.22k | +279 ± 477 vs 90M |
| 110M | 40-8 | +2.08k | −140 ± 514 vs 100M |
| 120M | 30-18 | +0.49k | −1.73k ± 1.13k vs 100M |
| 130M | 46-2 | +2.85k | +628 ± 462 vs 100M |
| 170M | 48-0 | +6.90k | +4.96k ± 1.13k vs 90M |
| 180M | 48-0 | +7.32k | +417 ± 869 vs 170M; +5.38k ± 0.87k vs 90M |
| 190M | 42-6 | +5.89k | |
| 200M | 48-0 | +7.89k | +568 ± 723 vs 180M |
| 210M | 48-0 | +7.51k | −383 ± 537 vs 200M |

Notes on the table:

- The 90M gain came mostly from lowering the anchor's bank: anchors ended at 94–96k, against 99–100k before.
- At 130M the policy won all 16 games against cha22 and all 16 against v56.
- 140M–160M were not played on the panel.

**Self-play and the panel disagreed twice.**

- At 60M, self-play called it a tie with c50 (0.50), but the panel was clearly better.
- At 120M, self-play favoured it over 100M (0.60, +2.8k), but the panel regressed (30-18, −1.73k ± 1.13k).
- 190M (self-play 0.55) also dipped on the panel.

No paneled trainer-promoted checkpoint (70M, 80M, 170M, 180M) had a lower mean margin than the checkpoint before it on the panel. The 80M (+1.0k ± 0.9k) and 180M (+417 ± 869) deltas are within about one SE.

## Throughput history

| Stage | Env steps/s |
|---|---|
| 4x RTX PRO 6000, 12 envs/rank | ~2,210 |
| 8x H200, 20 envs/rank | ~3,650 |
| + one SCHED_IDLE clock keeper per CPU (`clock_keeper.sh`) | ~5,250 (iteration 31.8 s → 21.9 s) |
| + native parallel step and `rl.compile_actor_heads=true` (17b3068d) | ~8,290–8,310 (iteration 13.9 s) |

With the compiled heads, the KL, clip fraction, teacher KL and explained variance curves matched the old-code run iteration by iteration. The mechanisms and limits are in the [[../references/clock-keepers-native-parallel-step-and-compiled-heads-lifted-h200-rollout-throughput|H200 throughput Reference]], and the evidence is in `ops/sprint-2026-09-30/throughput/` and `ops/sps-2026-10-01/`.

## Submissions (owner-approved each time)

The owner's rule (verbatim) was "do not spend submission slot unless we agreed tgt."

| Ref | Checkpoint | Archive sha256 | Time |
|---|---|---|---|
| 56711278 | 08bc (before the sprint) | `8283e676…` | 09-30 15:14; score 1004.1 at ~19:30 |
| 56716929 | 90M | `5e460d20…` | 19:26 |
| 56720629 | 170M | `428a63d7…` | 22:47 |
| 56722061 | 210M | `45efe071…` | 23:50:02 (the final slot) |

Each package has rule 1 baked on, rule 2 off, and the clean native module. The receipts are in `ops/submit-{90m,170m,210m}-2026-10-01/receipt.md`.

210M was submitted before its panel, as the owner asked. The panel ran afterwards on the same archive: 48-0, +7.51k.

**Leaderboard context.** At 19:36 there were 10,225 teams. The recorded cutoffs were silver (top 5%) ≈ 2,087, bronze (top 10%) ≈ 1,854 and gold ≈ top ~30 (~2,750+). The team's rank was 2,987 at 23:03.

## Custody

**Archived:**

- Checkpoints 60M–210M and last_best are on the Mac under `/Users/poonszesen/kg-v3-runs/<run>/` with `SHA256SUMS`.
- The 4x RTX pod archive is 8.5 GB, with 36 `.pt` files sha-verified.
- The compact text evidence is in `ops/sprint-2026-09-30/`. Skipped bulk files are hashed in `MANIFEST-skipped.tsv`.
- The four 8x H200 runs' training logs were copied off to the Mac before the pod was terminated. They are under `/Users/poonszesen/kg-v3-runs/<run>/` and hashed in `MANIFEST-skipped.tsv`. Their receipts, watchdog logs and copy-off logs are in `ops/sprint-2026-09-30/h200-run-receipts/`.
- The last run (r4zqqs49) is included. Its log is 14,256,784 bytes, with 1,008 iteration records up to rank-0 iteration 705, and it ends with the SIGTERM. `stop.txt` records `2026-09-30T23:51:10Z STOP ... reason='owner: shut down H200 after final submission'`. The final copy-off pass ran from 23:51:28 to 23:52:34.

**Terminated:** the 8x H200 pod was terminated at the owner's request right after the final submission. All pods were terminated by 23:59.

An earlier version of this episode said the r4zqqs49 log and receipts were lost. That was wrong. They were copied to the Mac; the receipts were added to `ops/` on 2026-10-01, and the log is hashed there.

## Gaps and limits

- **One training seed, chained starts.** Each relaunch reset the optimizer, so gains mix more training with the starts.
- **Selection panel, not held-out qualification.** It has 3 anchors and 8 seeds. 08bc was played rule off and the rest rule on; 60M's delta is from the rule-off arm.
- **The fixed-shop engine is not the board's earlier harness.** Margins here do not compare directly with the earlier f610 −21.1k / −28.8k table in [[../references/a-long-credit-window-turned-bc-start-self-play-from-sliding-to-improving|the credit-window Reference]].
- **Ladder results.** Kaggle scores and ranks for 90M, 170M and 210M are not recorded here. After the deadline the owner reported (verbatim, 2026-10-01): "it's silve zone", i.e. the team was within the silver cutoff at that time. That is owner-reported, not checked here, and provisional: per a Kaggle staff post the owner shared, submissions keep playing episodes for two weeks after the deadline and a single Bradley-Terry tournament then sets the final leaderboard.
- **Owner hypothesis for next time (untested).** Owner (verbatim, 2026-10-01): "if we've done this 1 day earlier and 2X~ the model param it could have gone further." Supporting reasoning, not evidence: the run was still promoting at 180M, and the GPUs were 80–90% idle during rollout (the per-turn cost was CPU stepping and Python dispatch), so a ~2x model would mainly lengthen the ~2.7 s update. Neither a larger model nor a longer run was tested.
- **Attribution.** Reward, window, batch size and more steps changed together along the lineage. The panel gain is not attributed to any one of them.

Related: [[../decisions/the-kaggriculture-v3-board|the v3 board]], the owner's sprint Decisions on [[../decisions/hold-the-learning-rate-and-let-the-anchor-panel-speak|the LR hold]], [[../decisions/keep-the-sprint-in-pure-self-play-without-pfsp-or-lambda-scheduling|pure self-play]] and [[../decisions/spend-kaggle-submission-slots-only-on-owner-agreed-checkpoints|submission slots]], [[../references/clock-keepers-native-parallel-step-and-compiled-heads-lifted-h200-rollout-throughput|H200 throughput]], [[../references/final-turn-liquidation-sells-the-shed-on-the-last-resolved-turn|rule 1]], [[../references/late-investment-filter-drops-only-purchases-that-cannot-sell-in-time|rule 2]], [[../references/kaggle-packaging-reuses-the-starter-submission-path|Kaggle packaging]].
