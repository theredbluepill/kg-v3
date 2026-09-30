---
type: "Decision"
title: "Halve the Kaggriculture checkpoint interval to 10M steps"
description: "Owner decision (2026-09-30): every Kaggriculture GPU training config (2-, 4- and 8-rank, the 1-GPU eager config and the two BC fine-tune presets) sets rl.checkpoint_freq to 10,000,000 global env steps, half of Isaiah's scaling_6m 20M. The one interval drives both the periodic checkpoint and the last-best evaluation with promotion at win rate >= 0.7, so both now come about every 610 iterations of 16,384 env steps (the first at iteration 611). It is a deliberate, owner-decided deviation from Isaiah; the CPU test config and the Orbit configs are unchanged. CPU config tests only; no run has used the new cadence."
tags: ["kaggriculture-v3", "training", "decisions", "adaptation"]
status: "adopted"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-09-30"}
decider: "Owner, 2026-09-30: \"cut the interval into half. thanks. also stop any running J/K. We will prepare the run for 8-rank\""
sources: [{"resource": "user-directive:2026-09-30:cut-the-checkpoint-interval-in-half"}, {"resource": "repository:configs/kaggriculture_2rank.yaml"}, {"resource": "repository:configs/kaggriculture_4rank.yaml"}, {"resource": "repository:configs/kaggriculture_8rank.yaml"}, {"resource": "repository:configs/kaggriculture_1gpu_eager.yaml"}, {"resource": "repository:configs/kaggriculture_2rank_bc_finetune.yaml"}, {"resource": "repository:configs/kaggriculture_8rank_bc_finetune.yaml"}, {"resource": "repository:configs/kaggriculture.yaml"}, {"resource": "repository:configs/scaling_6m.yaml"}, {"resource": "repository:scripts/run_ppo.py"}, {"resource": "repository:tests/kaggriculture/test_configs.py"}, {"resource": "repository:README.md"}, {"resource": "repository:ops/rebuild-2026-09-29/8rank-run/prepare.log"}]
---

# Halve the Kaggriculture checkpoint interval to 10M steps

## Decision

The owner asked how often a promotion check happens. The answer was: once every `checkpoint_freq` = 20,000,000 global env steps, about 1,221 iterations of 16,384 env steps. The owner replied, verbatim: "cut the interval into half. thanks. also stop any running J/K. We will prepare the run for 8-rank".

So every Kaggriculture GPU training config sets `rl.checkpoint_freq: 10_000_000`. The CPU test config `configs/kaggriculture.yaml` keeps its 1,000-step test cadence, and the Orbit configs keep Isaiah's values.

## One interval, two effects

In `scripts/run_ppo.py`'s training loop one threshold, `_next_periodic_checkpoint_step(checkpoint_freq=…)`, triggers everything that happens at the interval:

1. the periodic checkpoint `checkpoint_<env steps>.pt`;
2. the evaluation against the last-best model;
3. promotion when the win rate is at least `LAST_BEST_WIN_RATE_THRESHOLD` (0.7). Promotion rewrites `checkpoint_last_best.pt` and, with `teacher_mode: last_best`, refreshes the teacher in place.

Halving the interval therefore doubles the rate of checkpoints, evaluations and possible promotions (teacher refreshes) together. At 16,384 env steps per iteration the first interval closes after iteration 611 (10,010,624 steps), and each later one after 610 or 611 iterations. The trainer checks the threshold after whole iterations.

## Relation to Isaiah

This is a **deliberate deviation from Isaiah's `scaling_6m` (20M)**, decided by the owner. It is not an implementer choice under the [[recipe-choices-align-to-isaiah-without-owner-escalation|recipe-choices Decision]]. That Decision's rule still covers every other recipe value. `tests/kaggriculture/test_configs.py` now excludes `checkpoint_freq` from the whole-section equality with `scaling_6m` and pins the owner's value in a separate named test.

## Adaptation inventory

- `configs/kaggriculture_2rank.yaml`, `configs/kaggriculture_4rank.yaml`, `configs/kaggriculture_8rank.yaml`, `configs/kaggriculture_1gpu_eager.yaml`: `checkpoint_freq` 10M, a comment quoting the owner and citing this note, and header and teacher comments that no longer say the cadence is unchanged or 20M.
- `configs/kaggriculture_2rank_bc_finetune.yaml` and `configs/kaggriculture_8rank_bc_finetune.yaml` copy their ranked configs, so they carry the same 10M (see the [[../references/bc-fine-tune-presets-divide-both-learning-rates-by-ten|fine-tune preset Reference]]).
- `tests/kaggriculture/test_configs.py`: `test_checkpoint_interval_is_the_owners_half_of_scaling_6m` covers every GPU config (the four above and both presets). It pins `scaling_6m` at 20M, ours at 10M, 16,384 env steps per iteration at each config's world size, and 611 / 610 iterations. `test_ranked_config_optimizer_and_ppo_equal_scaling_6m` exempts the field.
- `README.md` (the Kaggriculture config paragraph), the [[../references/kaggriculture-configs-follow-isaiahs-scaling-6m-recipe|configs Reference]] and the [[preserve-the-ppo-recipe-across-bc-bootstrap|BC-recipe Decision]] now give 10M. The [[../references/shared-ppo-adapts-game-batches-without-a-second-loop|shared PPO Reference]] keeps 20M, because it describes the reference branch, where 20M was the value.

## Checks and limits

- `uvx --from rust-just just prepare` on this change: see `ops/rebuild-2026-09-29/8rank-run/prepare.log`.
- No run has used the new cadence. Evaluation cost per interval is unchanged, but it now comes twice as often. Its share of wall time is unmeasured at 8 ranks.
- Reopen if the owner changes the interval, or if a measurement shows the evaluation cost or promotion rate at 10M harms throughput or learning.
