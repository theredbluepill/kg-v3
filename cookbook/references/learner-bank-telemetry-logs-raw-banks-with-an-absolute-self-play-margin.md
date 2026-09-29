---
type: "Reference"
title: "Learner bank telemetry logs raw banks with an absolute self-play margin"
description: "Kaggriculture PPO logs learner-perspective raw final banks to W&B, keeping the seat-ordered keys: per update own-bank mean/p10/p50/p90 over both learner seats, absolute margin, winner/loser banks, draw rate and a game count gathered over ranks; per last-best evaluation own/opponent bank and signed margin distributions. Telemetry only, absent for Orbit; CPU tests on the native trainer and a fake W&B logger, nine killed mutations. Live W&B, multi-rank gather and a fixed-opponent panel are unverified."
tags: ["kaggriculture-v3", "adaptation", "evaluation", "diagnostics"]
status: "verified-scoped"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-09-30"}
sources:
  - resource: "user-directive:2026-09-30:carry-margin-own-bank-into-wandb"
  - resource: "repository:python/owl/kaggriculture/telemetry.py"
  - resource: "repository:python/owl/train/ppo.py"
  - resource: "repository:scripts/run_ppo.py"
  - resource: "repository:tests/kaggriculture/test_telemetry.py"
  - resource: "repository:tests/kaggriculture/test_training_smoke.py"
  - resource: "repository:tests/kaggriculture/test_teacher.py"
  - resource: "repository:tests/owl/train/test_ppo.py"
  - resource: "repository:tests/scripts/test_run_ppo.py"
  - resource: "repository:docs/rl-api-specs.md"
  - resource: "repository:README.md"
---

# Learner bank telemetry logs raw banks with an absolute self-play margin

The owner asked (2026-09-30): "make sure we carry the margin/own_bank into w&b
diagnostics during ppo." Before this change PPO logged only seat-ordered raw
banks: `train/terminal_bank_0`, `train/terminal_bank_1` and
`train/terminal_margin_0` from the native completed-game records, plus
`eval/candidate_bank`, `eval/last_best_bank` and `eval/candidate_bank_margin`.
In self-play both seats are the learner, so the seat-0 margin averages to about
zero and says nothing about the learner. This Reference records the added
learner-perspective keys. It serves the
[[../decisions/evaluation-preserves-generality-and-evidence|evaluation Decision]]
(win rate plus bank margin, with denominators) and extends the
[[evaluation-and-truncation-follow-the-kaggriculture-objective|raw-bank evaluation Reference]].
Searches for own bank, margin, telemetry, W&B and bank percentile found only
those seat-ordered keys and no learner-side telemetry.

## Contract and adaptation inventory

- `python/owl/kaggriculture/telemetry.py` (new) holds the math.
  `self_play_bank_metrics` pools both seats' banks. `opponent_bank_metrics`
  keeps a signed margin against a distinct opponent, and its key prefix carries
  any panel label. Percentiles use `torch.quantile` linear interpolation. The
  functions reject unequal lengths and non-finite banks. They always return
  `bank_games`, and every other key only when a game completed, so no key is
  NaN.
- `python/owl/train/ppo.py`: for a Kaggriculture batch, `train_iteration`
  gathers every rank's per-game `terminal_bank_0`/`terminal_bank_1` lists
  (`all_gather_object`) and logs `train/bank_games`,
  `train/own_bank_{mean,p10,p50,p90}`, `train/margin_abs_{mean,p50}`,
  `train/winner_bank_mean`, `train/loser_bank_mean` (decisive games only,
  omitted when all draw) and `train/draw_rate`. A step that lacks the native
  bank lists fails with an explicit error.
- `scripts/run_ppo.py`: `_evaluate_against_last_best` adds `eval/bank_games`
  and the mean and p10/p50/p90 of `eval/own_bank_*`, `eval/opponent_bank_*`
  and `eval/margin_*` (candidate minus last-best, signed). The means equal the
  unchanged candidate metrics.
- The training margin is absolute on purpose. A self-play game contributes
  `+m` from one learner seat and `-m` from the other, so a signed learner margin
  is identically zero.
- The values are raw bank (money), not normalized. They are telemetry only:
  computed after collection and returned to the logger, never read by model
  inputs, rewards, losses, normalization or promotion, which keeps the
  [[../decisions/the-policy-is-stateless-and-observation-only|stateless policy]]
  intact. Orbit gets no new key.
- Tests: `test_telemetry.py` covers the math on synthetic records (asymmetric
  banks, a draw, only draws, empty intervals, one game, signed evaluation
  margin, malformed records). The native smoke matches the trainer's keys to
  the recomputed records, and a truncation-only update logs only
  `train/bank_games = 0`. `test_ppo.py` pins the rank gather, the missing-list
  error and the absence of bank keys for Orbit. `test_run_ppo.py` runs a real
  native trainer and a real last-best evaluation through `_run_training_loop`
  into the fake W&B logger, and checks that Orbit evaluation adds no bank key.
  The fake Kaggriculture env in `test_teacher.py` now returns the native step's
  four metric keys.
- `docs/rl-api-specs.md` (Kaggriculture trainer section) and `README.md`
  document the keys and why the margin is absolute.

## Verification and limits

Branch `kg/rebuild-bank-metrics` from `kg/rebuild-3-1` at `2413c9e`, on a Mac
CPU. `just py-prepare` passes (format, lint, mypy, 2,399 Python passed with 6
hardware skips, docs freshness). Nine mutations
were each killed by the new tests: swapped evaluation own/opponent, no training
hook, Orbit hook, seat-0-only own bank, draws counted as decisive, signed
training margin, no rank gather, NaN on an empty interval, and evaluation keys
for Orbit. No independent review ran in this change.

Live W&B upload, a real multi-rank gather (tested with a stubbed collective),
CUDA and the per-update cost of one small `all_gather_object` at production
scale are unmeasured. `run_ppo` has no fixed-opponent panel on this base. Task
7.1's native opponents must reuse `opponent_bank_metrics` with the opponent's
name only in the key prefix before they log `own_bank`/`margin`. A branch whose
fake Kaggriculture env returns `{}` from `step` now fails fast and needs the
four native keys. Reopen if a panel lands, if live telemetry disagrees with
these definitions, or if the gather shows up in an update profile.
