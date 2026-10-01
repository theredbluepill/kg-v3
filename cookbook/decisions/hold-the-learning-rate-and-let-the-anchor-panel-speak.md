---
type: "Decision"
title: "Hold the learning rate and let the anchor panel speak"
description: "Owner decision for the final sprint (2026-09-30): \"let anchors speak will be good\", given on the learning rate. Keep Muon 1e-4 (AdamW 5e-6); per the sprint fact sheet, change it only if the anchor panel stalls and KL and clip fraction shrink. It never changed. The trainer's self-play promotion (>= 0.70 vs last_best every 10M steps) stayed on. Three runs were relaunched by hand from checkpoints self-play had not promoted: 90M (the fact sheet labels it an owner manual promotion), 100M and 130M (labelled manual, actor not recorded). No owner quote exists for any of the three. 90M and 100M were relaunched before their panels ran; only 130M was relaunched after its panel. Each relaunch loaded model weights only, which reset the optimizer and reran about 50 warm-up iterations. Self-play and the panel disagreed at 60M and 120M. Agent-proposed, not owner-adopted: check each promotion on the panel before choosing the next start. Selection panel, not held-out qualification."
tags: ["kaggriculture-v3", "decisions", "training", "evaluation"]
status: "adopted"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-10-01"}
decider: "Owner, 2026-09-30 (Claude Code session), on the learning rate: \"let anchors speak will be good\". Only this LR hold is owner-adopted. The manual relaunches are recorded as facts (90M labelled an owner promotion in the sprint fact sheet; 100M and 130M with no recorded actor), with no owner quote. The general rule of checking promotions on the panel is agent-proposed."
sources:
  - resource: "user-directive:2026-09-30:let-anchors-speak-will-be-good"
  - resource: "repository:ops/sprint-2026-09-30/sprint-facts.md"
  - resource: "repository:ops/sprint-2026-09-30/README.md"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/cpu-pod/eval_ckpt.sh"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/cpu-pod/eval-90M.log"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/cpu-pod/eval-100M.log"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/cpu-pod/eval-120M.log"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/cpu-pod/eval-130M.log"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/games-fixedshop-linux/90M-ft-on/summary.md"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/games-fixedshop-linux/100M-ft-on/summary.md"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/games-fixedshop-linux/120M-ft-on/summary.md"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/games-fixedshop-linux/130M-ft-on/summary.md"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/eval-60M/eval60m_tables.md"
  - resource: "repository:ops/sprint-2026-09-30/h200-run-receipts/earn720-r30e01w30-8xh200-from-60M-20261001/receipts/times.txt"
  - resource: "repository:ops/sprint-2026-09-30/h200-run-receipts/earn720-r30e01w30-8xh200-from-90M-20261001/receipts/times.txt"
  - resource: "repository:ops/sprint-2026-09-30/h200-run-receipts/earn720-r30e01w30-8xh200-from-90M-20261001/watchdog.log"
  - resource: "repository:ops/sprint-2026-09-30/h200-run-receipts/earn720-r30e01w30-8xh200-from-100M-sps-20261001/receipts/times.txt"
  - resource: "repository:ops/sprint-2026-09-30/h200-run-receipts/earn720-r30e01w30-8xh200-from-100M-sps-20261001/watchdog.log"
  - resource: "repository:ops/sprint-2026-09-30/h200-run-receipts/earn720-r30e01w30-8xh200-from-130M-sps-20261001/receipts/times.txt"
  - resource: "repository:ops/sprint-2026-09-30/h200-run-receipts/earn720-r30e01w30-8xh200-from-130M-sps-20261001/watchdog.log"
  - resource: "repository:ops/rebuild-2026-09-29/sprint-8gpu/launch.sh"
  - resource: "repository:ops/rebuild-2026-09-29/sprint-8gpu/mac_side.md"
  - resource: "repository:configs/kaggriculture_4rank_margin.yaml"
  - resource: "repository:scripts/run_ppo.py"
  - resource: "wandb-run:spoon/kg-v3/48gyi9m5"
  - resource: "wandb-run:spoon/kg-v3/xuft2e2i"
  - resource: "wandb-run:spoon/kg-v3/3w2ag52m"
  - resource: "wandb-run:spoon/kg-v3/dxhey4da"
  - resource: "wandb-run:spoon/kg-v3/r4zqqs49"
---

# Hold the learning rate and let the anchor panel speak

## Decision

On the learning rate during the final sprint, the owner said, verbatim (Claude Code session, 2026-09-30):

> let anchors speak will be good

- **Learning rate (owner-adopted).** The sprint fact sheet records the operating rule: keep Muon at 1e-4 (AdamW 5e-6) unless the anchor panel stalls and approx-KL and clip fraction shrink. It was never changed in any of the five sprint runs.
- **Trainer promotion (unchanged).** The trainer's own rule ran as before. Every 10M global env steps it compares the checkpoint with last_best, and promotes at a win rate of at least 0.70 over `n_envs` games (`teacher_mode: last_best`, KL 0.005).

This Decision was first titled "Promote and relaunch on anchor-panel evidence", which presented a general promotion rule as owner-adopted. It was retitled on 2026-10-01. The owner's quote was about the learning rate, and the rest is recorded below as facts or as the agent's proposal.

## Manual relaunches (recorded facts, no owner quote)

Three runs were started by hand from checkpoints that self-play had not promoted. Each one meant stopping the running job and relaunching from that checkpoint.

| Start | Fact-sheet label | Previous run stopped | Relaunch | Panel of the start |
| --- | --- | --- | --- | --- |
| 90M `6b196ef3` | "owner manual promotion" | 19:11:43Z | 19:12:25Z | 19:12:57Z to about 19:18:57Z, after the relaunch |
| 100M `8e520566` | "(manual)", no actor | 19:46:02Z | 19:46:23Z | started 19:46:35Z, after the relaunch |
| 130M `86df0ee4` | "(manual)", no actor | 21:03:45Z | 21:03:58Z | finished about 21:02Z, before the relaunch |

So the panel could not have chosen the 90M or 100M relaunch, and the trigger for those two is not recorded. Only the 130M relaunch followed its own panel. All times are UTC. The stop and launch times are from `h200-run-receipts/<run>/receipts/times.txt` and `watchdog.log`, and the panel times are from `anchor-games/cpu-pod/eval-{90M,100M,130M}.log`.

## Agent-proposed rule (not owner-adopted)

**Proposal.** Check every promotion on the fixed-shop anchor panel before choosing the next start, and do not rely on self-play alone. This is the agent's reading of the sprint evidence. The owner never stated it as a rule.

**Evidence for it.**
- At 60M, self-play called it a tie with c50, but the panel's rule-off arm put it +3.9k ahead of c50.
- At 120M, self-play favoured it (0.60, +2.8k vs last_best), but the panel had it 1.73k ± 1.13k worse than 100M (30-18).
- At 190M, self-play gave only 0.55, and the panel also dipped, to 42-6.

## Outcome

Panel: 8 seeds × 2 seats × {smaller_market_shock, cha22, v56} = 48 games, rule 1 on, paired SE over the 8 seed means. All runs used the same recipe and pinned launcher (`ops/rebuild-2026-09-29/sprint-8gpu/launch.sh`).

| Start | Run (W&B) | Self-play at the start's own check | Panel of the start |
| --- | --- | --- | --- |
| c50 `0cc80065` (first sprint start) | `48gyi9m5` (4× RTX PRO 6000) | — | 6-42, −7.2k |
| 60M `20b1f795` (latest checkpoint at the move to 8× H200; not promoted) | `xuft2e2i` | 0.50 vs c50, +1.6k | 8-40, −3.3k |
| 90M `6b196ef3` | `3w2ag52m` | 0.60 vs 80M, +2.7k | 38-10, +1.94k; +2.5k ± 0.8k vs 80M |
| 100M `8e520566` | `dxhey4da` (code `17b3068d`, compiled heads) | 0.65 vs last_best, +2.5k | 40-8, +2.22k; +279 ± 477 vs 90M |
| 130M `86df0ee4` | `r4zqqs49` | 0.65 vs last_best, +4.1k | 46-2, +2.85k; +628 ± 462 vs 100M |

- **Trainer promotions.** The trainer itself promoted 70M (0.80), 80M (0.95), 140M (0.70), 170M (0.80) and 180M (0.90).
- **Later panels.** 170M and 180M went 48-0, at +6.90k and +7.32k. 200M and 210M also went 48-0, at +7.89k and +7.51k.
- **Cost of a relaunch.** A relaunch loads model weights only. That resets the optimizer and reruns the learning-rate warm-up of about 1,000 steps (about 50 iterations).

## Limits

- **Selection, not qualification.** The panel used three anchors and eight seeds. The same panel both chose the checkpoints and measured them, so it is selection evidence, not held-out qualification ([[evaluation-preserves-generality-and-evidence|evaluation Decision]]).
- **No counterfactual.** No run continued without a manual relaunch, so the effect of relaunching by hand on the final strength is not measured.
- **Who decided.** The fact sheet labels only 90M as an owner promotion. For 100M and 130M it records no actor, and no relaunch has an owner quote.
- **Logs.** The four 8× H200 training logs are on the Mac, hashed in `ops/sprint-2026-09-30/MANIFEST-skipped.tsv`. Their receipts, watchdog logs and copy-off logs are in `ops/sprint-2026-09-30/h200-run-receipts/`, and their metrics are in W&B.

Results: [[../episodes/the-final-sprint-took-720-turn-self-play-from-c50-to-a-48-0-anchor-panel|sprint episode]].
