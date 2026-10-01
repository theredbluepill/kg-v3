---
type: "Decision"
title: "Promote and relaunch on anchor-panel evidence"
description: "Owner decision for the final sprint (2026-09-30/10-01): \"let anchors speak will be good\". Keep Muon 1e-4. Per the sprint fact sheet, change it only if the anchors stall and KL and clip fraction shrink; it never changed. The trainer's self-play promotion (>= 0.70 vs last_best every 10M steps) stayed on. The owner also hand-promoted 90M, 100M and 130M after their fixed-shop anchor panels, though self-play had not promoted them (0.60, 0.65, 0.65). Each was relaunched from model weights only, which reset the optimizer and reran the roughly 50-iteration LR warm-up. Self-play and the panel disagreed at 60M and 120M. The panel is a selection panel, not held-out qualification."
tags: ["kaggriculture-v3", "decisions", "training", "evaluation"]
status: "adopted"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-10-01"}
decider: "Owner, 2026-09-30/10-01 (Claude Code session): \"let anchors speak will be good\". The manual promotions of 90M, 100M and 130M are owner actions recorded in the sprint fact sheet. Turning them into a general rule is the agent's reading."
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

# Promote and relaunch on anchor-panel evidence

## Decision

On the learning rate during the final sprint, the owner said, verbatim (Claude Code session, 2026-09-30/10-01):

> let anchors speak will be good

- **Learning rate.** The sprint fact sheet records the operating rule: keep Muon at 1e-4 (AdamW 5e-6) unless the anchor panel stalls and approx-KL and clip fraction shrink. It was never changed, in any of the five sprint runs.
- **Trainer promotion.** The trainer's own rule ran unchanged: every 10M global env steps it compares the checkpoint with last_best, and promotes at a win rate of at least 0.70 over `n_envs` games (`teacher_mode: last_best`, KL 0.005).
- **Manual promotion.** On top of that, the owner hand-promoted 90M, 100M and 130M after their fixed-shop anchor panels, although self-play had not promoted them. Each manual promotion meant stopping the run and relaunching from that checkpoint.
- **Agent's reading.** The general form, "the anchor panel, not self-play alone, picks the next start", is the agent's reading of these actions. The owner gave no general statement of it.

## Outcome

Panel: 8 seeds × 2 seats × {smaller_market_shock, cha22, v56} = 48 games, rule 1 on, paired SE over the 8 seed means. All runs used the same recipe and pinned launcher (`ops/rebuild-2026-09-29/sprint-8gpu/launch.sh`).

| Start | Why | Run (W&B) | Self-play at the start's own check | Panel of the start |
| --- | --- | --- | --- | --- |
| c50 `0cc80065` | first sprint start | `48gyi9m5` (4× RTX PRO 6000) | — | 6-42, −7.2k |
| 60M `20b1f795` | latest checkpoint at the move to 8× H200; not promoted | `xuft2e2i` | 0.50 vs c50, +1.6k | 8-40, −3.3k |
| 90M `6b196ef3` | owner manual promotion | `3w2ag52m` | 0.60 vs 80M, +2.7k | 38-10, +1.94k; +2.5k ± 0.8k vs 80M |
| 100M `8e520566` | manual | `dxhey4da` (code `17b3068d`, compiled heads) | 0.65 vs last_best, +2.5k | 40-8, +2.22k; +279 ± 477 vs 90M |
| 130M `86df0ee4` | manual | `r4zqqs49` | 0.65 vs last_best, +4.1k | 46-2, +2.85k; +628 ± 462 vs 100M |

- **Trainer promotions.** The trainer itself promoted 70M (0.80), 80M (0.95), 140M (0.70), 170M (0.80) and 180M (0.90).
- **Later panels.** 170M and 180M went 48-0, at +6.90k and +7.32k. 200M and 210M also went 48-0, at +7.89k and +7.51k.
- **Where the two signals disagreed.**
  - 60M: self-play called it a tie with c50, while the panel's rule-off arm put it +3.9k ahead of c50.
  - 120M: self-play favored it (0.60, +2.8k vs last_best), while the panel had it 1.73k ± 1.13k worse (30-18).
  - 190M: self-play gave only 0.55, and the panel also dipped, to 42-6.
- **Cost of a relaunch.** A relaunch loads model weights only. That resets the optimizer and reruns the learning-rate warm-up of about 1,000 steps (about 50 iterations).

## Limits

- **Selection, not qualification.** The panel used three anchors and eight seeds. The same panel both chose the checkpoints and measured them, so it is selection evidence, not held-out qualification ([[evaluation-preserves-generality-and-evidence|evaluation Decision]]).
- **No counterfactual.** No run continued without a manual promotion, so the effect of promoting by hand on the final strength is not measured.
- **Source of the record.** The fact sheet records which checkpoints were hand-promoted, but no per-checkpoint owner quote.
- **Missing log.** The training log of the last run (`r4zqqs49`) was not archived before its pod was terminated. Its metrics are only in W&B.
