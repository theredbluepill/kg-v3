---
type: "Reference"
title: "A long credit window turned BC-start self-play from sliding to improving"
description: "Finding, one training seed per arm, first recorded 2026-09-30, revised 2026-10-01 with the final sprint. With gae_lambda 1.0 and a 256-step or full 720-step window, plus an own-bank + cash-difference reward, BC-start mirror self-play PPO at Muon 1e-4 improved instead of sliding: h3lpxy6q (256) beat BC 16/16 at 10M and promoted fc6b123c; cmwjclbe (720) promoted f61006d9 at 20M; on the earlier Kaggle harness the margin moved BC -67.2k -> fc6b -41.2k -> f610 -21.1k vs smaller_market_shock. The same 720-turn lambda-1 window then kept improving for 160M more env steps: pcy5knet promoted 60f2 (30M) and 08bc (40M), and the sprint lineage from c50 (reward 0.30 bank / 0.30 cash difference / 0.01 econ shaping / 0.30 win-loss) had five trainer promotions to 180M and took the 48-game fixed-shop anchor panel from 6-42 (-7.2k) to 48-0 (+7.5k at 210M) with no slide. Every earlier 64-step lambda-0.9 run slid. Attribution is unresolved: reward, batch structure, batch size and training length changed with the window."
tags: ["kaggriculture-v3", "training", "credit-assignment", "evaluation", "finding"]
status: "provisional"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-10-01"}
sources:
  - resource: "user-directive:2026-09-30:switch-back-to-self-play-and-earn-money-for-real"
  - resource: "user-directive:2026-09-30:ok-go-ahead-bank-margin-sign-reward"
  - resource: "user-directive:2026-09-30:ok-go-with-720-and-relaunch-the-run"
  - resource: "repository:ops/earn-money-2026-09-30/plan.md"
  - resource: "repository:ops/earn-money-2026-09-30/evidence-runs.md"
  - resource: "repository:ops/earn-money-2026-09-30/x1/summary.md"
  - resource: "repository:ops/earn-money-2026-09-30/anchor-games/results.md"
  - resource: "repository:ops/earn-money-2026-09-30/anchor-games/games-bc.jsonl"
  - resource: "repository:ops/earn-money-2026-09-30/anchor-games/games-candidate.jsonl"
  - resource: "repository:ops/earn-money-2026-09-30/anchor-games/games-best-f610.jsonl"
  - resource: "repository:ops/earn-money-2026-09-30/anchor-games/aggregate_f610.py"
  - resource: "repository:ops/earn-money-2026-09-30/anchor-games/tables-best-f610.md"
  - resource: "repository:ops/earn-money-2026-09-30/anchor-games/manifests/best-f610.json"
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/earn-bank-credit-4rank/launch.md"
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/earn512/launch.md"
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/earn720/launch.md"
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/earn720-12env/launch.md"
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/earnlr/launch.md"
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/earnB/launch.md"
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/control-J2-4rank/result.md"
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/A-bank-lr2-4rank/launch.md"
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/J2-resume-r0208/stop.md"
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/M-margin-J2-4rank/stop.md"
  - resource: "repository:configs/kaggriculture_4rank_margin.yaml"
  - resource: "wandb-run:spoon/kg-v3/h3lpxy6q"
  - resource: "wandb-run:spoon/kg-v3/cmwjclbe"
  - resource: "wandb-run:spoon/kg-v3/pcy5knet"
  - resource: "wandb-run:spoon/kg-v3/nw3klj2s"
  - resource: "wandb-run:spoon/kg-v3/04cy2m6s"
  - resource: "wandb-run:spoon/kg-v3/bqtke7iq"
  - resource: "wandb-run:spoon/kg-v3/hz4bpjnq"
  - resource: "wandb-run:spoon/kg-v3/r350xr3w"
  - resource: "repository:cookbook/episodes/the-final-sprint-took-720-turn-self-play-from-c50-to-a-48-0-anchor-panel.md"
  - resource: "repository:ops/sprint-2026-09-30/sprint-facts.md"
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/earnC/launch.md"
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/earnF/launch.md"
  - resource: "repository:ops/rebuild-2026-09-29/sprint-8gpu/launch.sh"
  - resource: "wandb-run:spoon/kg-v3/48gyi9m5"
  - resource: "wandb-run:spoon/kg-v3/xuft2e2i"
  - resource: "wandb-run:spoon/kg-v3/r4zqqs49"
---

# A long credit window turned BC-start self-play from sliding to improving

## Claim

With `gae_lambda` 1.0 and a credit window of 256 steps or a full 720-step game, together with the owner's own-bank plus cash-difference reward, mirror self-play PPO started from BC improved instead of sliding. The 720-turn window then kept improving through 210M env steps without a slide (see Continuation through 210M). It is one training seed per arm. Two things changed together with the window, so the window alone is not attributed (see Unresolved attribution).

This is a finding recorded as a Reference, not an owner decision. The owner decided the direction and the reward. The reading of the runs is the agent's.

## Owner direction (verbatim)

- "let's switch back to self play no matter what, and think about how do we get the agent to earn moneny for real?"
- On the agent-proposed reward (.25 own bank /150k + .25 cash difference /100k + .5 terminal sign): "OK go ahead."
- On the window: "ok go with 720 and relaunch the run"

## Mechanism the runs tested

Plan `ops/earn-money-2026-09-30/plan.md`, section 1.1:
- **M3 (a), credit inversion.** λ 0.9 over a 64-step horizon credits about 10 turns, while plant, animal and land investments pay back over 100-300 turns. X1d found that strawberry investment breaks first in the short-window slide (`x1/summary.md`).
- **M2, lockstep phases.** Every 64-step rollout trains one game phase. A 720-step segment is exactly one whole game, so every update spans every phase and the return is pure Monte Carlo.
- **M4, no reward for a shared rise.** Term M (margin only) pays nothing when both seats earn more. The .25 own-bank term does.

The [[../decisions/stagger-game-phases-and-lengthen-the-credit-window|stagger Decision]] built the 256-step, λ 1 presets with the stagger and the per-seat critic offset. None of the runs below used those presets: they ran at `0f70773` with `configs/kaggriculture_4rank_margin.yaml` plus command-line overrides, with no stagger and no critic offset.

## Evidence

**Before: every short-window run slid** (horizon 64, λ 0.9, BC or J/2 start, Muon 1e-4). Values are rank-0 `train/own_bank_mean` at game-end windows (W&B); an iteration is 16,384 env steps (`_step`). Runs A and A2 are identified by their W&B groups `A-bank-lr2-4rank-20260930` and `A2-bank-w025-lr2-4rank-20260930` (`evidence-runs.md`):

| Run (W&B) | Reward | Own bank |
| --- | --- | --- |
| J/2 (`nw3klj2s`) | recipe J shaping | peak 81.6k at iteration 45, 46.7k by 90, plateau 47-53k |
| A (`04cy2m6s`) | + own bank term, w_b 1.0 (saturates at a 25k bank) | 74.5k -> 60.9k over 6 windows (iterations 12-68) |
| A2 (`bqtke7iq`) | + own bank term, w_b .25 (saturates at a 100k bank) | 71.8k, peak 79.1k at iteration 57, 50.9k at iteration 248; 22 windows |
| J/2 resume (`hz4bpjnq`) | J shaping .2, terminal .8 | 62.6k -> 30.4k over 87 windows |
| M (`r350xr3w`) | term M .5 margin + .5 sign | about 64k -> 38.1k; lost both evaluations to its start (14.1%, 10.9%) |

**After: the long-window runs** (same reward, Muon 1e-4 / AdamW 5e-6, λ 1.0):

| Run (W&B) | Start | Window, envs/rank | Outcome |
| --- | --- | --- | --- |
| `h3lpxy6q` | BC (`fd854587…`) | 256, 16 | Own bank first window 73.9k. Means of `train/own_bank_mean` over six equal `_step` ranges of its 357 logged windows (W&B `run.history` with that key): 78.1k, 81.2k, 87.4k, 94.7k, 89.8k, 89.2k; equal-count sixths differ by 0.3k or less. 10M evaluation vs BC: 16/16 won, own 117.6k vs 90.4k, margin +27.1k; promoted `fc6b123c…`. Stopped at iteration 1,007 (about 16.5M steps) for the 512 window. |
| `ssoc84zg` | fc6b | 512, 8 | Stopped at iteration 41 for the 720 window; own bank about 95k. |
| `pw6qjsz3` | fc6b | 720, 6 | Stopped at iteration 90 to raise envs per rank; own bank about 96-101k. |
| `cmwjclbe` | fc6b | 720, 12 | Promoted at its 20M checkpoint: `f61006d9…`, weights bit-identical to `checkpoint_00_020_033_024`. The evaluation metrics were never logged because the run was stopped about 40 s after the checkpoint; promotion requires ≥ 70% of 12 games. Self-play own bank drifted from about 97k to 93k over its windows. |
| `pcy5knet` | f610 | 720, 12 | Promoted 60f2 at 30M (11/12) and 08bc at 40M (12/12); 50M (`0cc80065`, c50) 8/12, not promoted, then promoted by the owner (`earnC/launch.md`). Stopped at iteration 889. |

**Independent check: local Kaggle-harness games.** Packaged agents, greedy, CPU, strict mode. Seeds 93001-93008, each in both seats: 16 games per policy per anchor. All games qualified, with 0 errors, invalid actions or PASS fallbacks (`anchor-games/results.md`, `tables-best-f610.md`).

| Policy | vs smaller_market_shock: own / margin | vs cha22: own / margin | Wins vs cha22, smaller_market_shock, v43 |
| --- | --- | --- | --- |
| BC `fd854587` | 77.1k / -67.2k | 86.9k / -60.6k | 0 / 48 |
| fc6b (256 window) | 92.3k / -41.2k | 88.7k / -36.0k | 0 / 48 |
| f610 (720 window) | 113.1k / -21.1k | 96.9k / -28.8k | 0 / 32 (v43 not played) |

Paired on the same games, f610's margin beat BC's in 32/32 games and fc6b's in 23/32 (+13.7k mean). fc6b beat BC in 58/64 games across all four anchors.

**Reference levels, not targets.** X1a: the BC teacher (leaderboard #1) banked about 102.8k per ladder game. X1b: cha22 against itself banked about 95.4k per seat. Both are from different opponents and worlds, so they do not compare directly with the anchor games.

## Continuation through 210M (final sprint, 2026-09-30)

The [[../episodes/the-final-sprint-took-720-turn-self-play-from-c50-to-a-48-0-anchor-panel|final-sprint episode]] continued the same window from c50 for 160M more env steps. It kept the 720-step horizon, λ 1.0, one segment per minibatch and Muon 1e-4.

- **The reward changed.** It was own bank 0.30, cash difference 0.30, per-event econ shaping 0.01 (cap 0.1) and win/loss 0.30, set by the owner in `earnF/launch.md`.
- **The batch grew.** From 60M the runs used 20 envs per rank on 8x H200.

| Measure | Result |
| --- | --- |
| Trainer promotions | 70M (0.80), 80M (0.95), 140M (0.70), 170M (0.80), 180M (0.90) |
| Manual relaunch starts | 90M (labelled an owner promotion), 100M and 130M (actor not recorded), after self-play scores of 0.60–0.65; no owner quote |
| Fixed-shop panel W-L (48 games) | c50 6-42 → 80M 30-18 → 90M 38-10 → 130M 46-2 → 170M, 180M, 200M, 210M 48-0 |
| Fixed-shop panel mean margin | −7.2k → +7.5k at 210M |

No sustained slide appeared. The panel dipped at 120M (30-18, −1.73k ± 1.13k vs 100M) and at 190M (42-6), and both were followed by recoveries.

The fixed-shop panel is a different engine copy and anchor set (v56 for v43) from the table above, so its margins do not continue that table's numbers.

## Limits

- **One training seed per arm.** The sprint is one more chained lineage, not a second seed. There is no variance estimate across training seeds.
- **Self-play bank is a weak signal after the switch.** Under the 720 window, self-play own bank stayed flat or eased (cmwjclbe 97k -> 93k). The improvement shows in the promotion and the anchor games, not in the self-play curve. h3lpxy6q peaked in its middle third and eased afterwards.
- **Anchor games are a local strength check, not qualification.** 16 games per anchor cannot separate a win rate between 0 and about 0.19. Up to f610 every policy lost every anchor game on the earlier harness. On the sprint's fixed-shop selection panel, 170M and later won 48 of 48; that panel is not held-out qualification. The cha22 binary is not freshly pinned (`results.md`, Limits). The runs were not evaluated at Kaggle latency.
- **Starts are chained.** The 720 runs began from fc6b with a fresh optimizer and a re-warmed LR, so the 720 gain over fc6b is also more training.

## Unresolved attribution

- **Reward, batch size and length changed along the lineage.** The sprint changed the reward to 0.30/0.30/0.01/0.30 and moved to 20 envs per rank on 8 GPUs. It also kept training, so its gains are not attributed to the window either.
- **Reward changed with the window.** No short-window run used the .25 bank + .25 cash difference + .5 sign reward from BC. So window versus reward is not separated. The own-bank term under the short window still slid: A2 (`bqtke7iq`, w_b .25) peaked at 79.1k and ended at 50.9k. That weakly counts against reward alone. Run A (`04cy2m6s`) is a weak test here, because its term stopped acting above a 25k bank.
- **Batch structure changed.** At 256 each optimizer step used 1 segment of 256 steps per rank (4 envs globally) instead of 4 segments of 64 (16 envs). At 720 each step used one whole game per rank. Correlation within a minibatch is higher.
- **Information, decision, execution or architecture.** Which class the window repaired is not measured. The plan's X5 paired investment forks (M3 a against c) were not run.

## Reopening conditions

- A 64-step λ 0.9 run with this reward from BC that does not slide would reopen the attribution to the window.
- A second seed of the 256 or 720 run that slides would demote this finding.
- A long 720 run whose paneled promotions fall below their predecessors would weaken the "keeps improving" part. This has not happened through 180M; the dips at 120M and 190M were unpromoted checkpoints.
- A promoted checkpoint whose paired anchor margin on seeds 93001-93008 falls below the previous promotion's would show that promotion against the last best no longer tracks strength.

## Promotion basis

Independent check: the Kaggle-harness games are separate from the training telemetry and the in-trainer evaluation. Existing concepts searched: the stagger Decision (the mechanism, untested there) and the [[../decisions/replace-the-reward-with-half-cash-difference-and-half-terminal-sign|term M Decision]] (the reward that slid). Consequence: the [[../decisions/the-kaggriculture-v3-board|v3 board]] ranks the 720 continuation first. The sprint episode is a second chained episode in the same lineage, not an independent seed, so this stays a provisional Reference rather than a Lesson.
