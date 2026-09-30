---
type: "Decision"
title: "Train PPO against a fixed opponent with a learner mask"
description: "Owner decision (2026-09-30): the anchor setup trains PPO against a fixed scripted bot (cha22) instead of mirror self-play, after the new reward. Track A adds the bot-agnostic mechanism on kg/rebuild-opponent-mix. env.opponent_mix = {bot, fraction} hosts an opponents_rs controller natively in the first fraction x n_envs envs of each rank, with the learned seat alternating by env index and episode. The rollout forward runs on learner rows only. A learner mask removes the bot's seat from every loss term, advantage normalization and denominator. Per-update *_vs_bot telemetry and a fixed-bot evaluation in both seats at each checkpoint_freq are added; promotion stays vs last_best. Absent, the pipeline is byte-identical to the pre-mix tree (golden digest). A default Cargo feature keeps the controllers out of the Kaggle build. Merged with the Track B import (kg/rebuild-cha22-opponent ab09708), cha22 is a registry key, and the anchor presets configs/kaggriculture_{4,2}rank_vs_cha22.yaml host it in every env under term M with J/2's halved LRs, warm-started from the BC best. Nothing has been trained. Review r1 (a Claude substitute for Codex) approved 0f70773; its follow-ups pin the fixed-bot evaluation's seat attribution and the learner-row value routing (both surviving mutations now caught), document degenerate telemetry under fraction 1.0 and the unmeasured evaluation time before the rank-0 broadcast, and record the trainer digest's thread count. Mechanism details and the preset's warm start are implementer choices. CPU checks only."
tags: ["kaggriculture-v3", "training", "opponents", "decisions", "adaptation"]
status: "adopted"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-09-30"}
decider: "Owner, 2026-09-30: \"OK, for fixed bot, we can use cha22 (check ~/kaggriculture-v2).\""
sources: [{"resource": "user-directive:2026-09-30:fixed-bot-use-cha22"}, {"resource": "user-directive:2026-09-30:can-we-accelerate-this-setup"}, {"resource": "user-directive:2026-09-30:implement-the-new-reward-first"}, {"resource": "user-directive:2026-09-30:is-anchor-thing-ready"}, {"resource": "repository:opponents_rs/src/hosted.rs"}, {"resource": "repository:opponents_rs/src/lib.rs"}, {"resource": "repository:opponents_rs/src/registry.rs"}, {"resource": "repository:opponents_rs/tests/hosted.rs"}, {"resource": "repository:opponents_rs/OPPONENT_MANIFEST.json"}, {"resource": "repository:opponents_rs/README.md"}, {"resource": "repository:src/kaggriculture/env.rs"}, {"resource": "repository:src/kaggriculture/opponents.rs"}, {"resource": "repository:src/kaggriculture/bindings.rs"}, {"resource": "repository:src/kaggriculture/observe.rs"}, {"resource": "repository:src/kaggriculture/mod.rs"}, {"resource": "repository:src/kaggriculture/opponent_env_tests.rs"}, {"resource": "repository:src/kaggriculture/env_tests.rs"}, {"resource": "repository:Cargo.toml"}, {"resource": "repository:Cargo.lock"}, {"resource": "repository:pyproject.toml"}, {"resource": "repository:justfile"}, {"resource": "repository:Dockerfile.kaggle"}, {"resource": "repository:scripts/build_kaggle_submission.sh"}, {"resource": "repository:python/owl/rs.pyi"}, {"resource": "repository:python/owl/game.py"}, {"resource": "repository:python/owl/kaggriculture/config.py"}, {"resource": "repository:python/owl/kaggriculture/env.py"}, {"resource": "repository:python/owl/kaggriculture/telemetry.py"}, {"resource": "repository:python/owl/train/ppo.py"}, {"resource": "repository:scripts/run_ppo.py"}, {"resource": "repository:tests/kaggriculture/test_opponent_mix.py"}, {"resource": "repository:tests/owl/kaggriculture/test_opponents.py"}, {"resource": "repository:tests/scripts/test_run_ppo.py"}, {"resource": "repository:tests/kaggriculture/test_env.py"}, {"resource": "repository:docs/rl-api-specs.md"}, {"resource": "repository:docs/kaggriculture-contract.md"}, {"resource": "repository:docs/rules-parity-coverage.md"}, {"resource": "repository:docs/containerization.md"}, {"resource": "repository:README.md"}, {"resource": "repository:ops/opponent-mix-2026-09-30/baseline_digest.py"}, {"resource": "repository:ops/opponent-mix-2026-09-30/baseline-digest-25412a7.json"}, {"resource": "repository:ops/opponent-mix-2026-09-30/mutations.py"}, {"resource": "repository:ops/opponent-mix-2026-09-30/mutations.log"}, {"resource": "repository:ops/opponent-mix-2026-09-30/prepare.log"}, {"resource": "repository:ops/opponent-mix-2026-09-30/post-digest.json"}, {"resource": "repository:cookbook/references/snapshot-view-isolates-byte-exact-evaluation-opponents.md"}, {"resource": "repository:cookbook/references/kaggriculture-parity-summary-maps-tested-and-untested-layers.md"}, {"resource": "repository:configs/kaggriculture_4rank_vs_cha22.yaml"}, {"resource": "repository:configs/kaggriculture_2rank_vs_cha22.yaml"}, {"resource": "repository:configs/kaggriculture_4rank_margin.yaml"}, {"resource": "repository:tests/kaggriculture/test_configs.py"}, {"resource": "repository:tests/scripts/test_run_ppo.py"}, {"resource": "repository:ops/cha22-anchor-2026-09-30/run-statement.md"}, {"resource": "repository:ops/cha22-anchor-2026-09-30/prepare.log"}, {"resource": "repository:cookbook/references/cha22-opponent-imports-a-view-adapted-closure-with-light-parity.md"}, {"resource": "repository:ops/opponent-mix/review-r1.md"}, {"resource": "repository:ops/opponent-mix/r1_mutations.py"}, {"resource": "repository:ops/opponent-mix/r1-mutations.log"}, {"resource": "repository:ops/opponent-mix/digest-environment.log"}, {"resource": "repository:ops/opponent-mix/prepare.log"}]
---

# Train PPO against a fixed opponent with a learner mask

## Decision

The owner, verbatim, on 2026-09-30, in order:

1. **"OK, for fixed bot, we can use cha22 (check ~/kaggriculture-v2)."**
2. **"can we acceleerate this setup?"**
3. **"implement the new rewrad first before we revisit the cha22 anchor setup."**
4. After the new reward was implemented and its run launched: **"is anchor thing ready?"**

The owner adopted training against a fixed bot, with cha22 as that bot, and set
the order: the new reward (term M,
[[replace-the-reward-with-half-cash-difference-and-half-terminal-sign|term M Decision]])
before the anchor setup.

**Interpretation, not owner adoption.** The orchestrating agent read the words
as follows; the owner has not confirmed these readings:
- "anchor thing" is the cha22 fixed-opponent learnability setup: PPO against a
  fixed bot instead of mirror self-play. It tests whether this RL pipeline can
  improve on BC at all. The orchestrator relayed the owner's conclusion as "the
  agent can learn from BC, but not our RL pipeline". This note has not seen that
  sentence at its source.
- "implement the new rewrad first" means the anchor setup uses term M. So this
  branch starts from `kg/rebuild-reward-margin` (`25412a7`).
- The work splits into Track A (this bot-agnostic mechanism) and Track B (the
  cha22 import). Track B's stopped work is the unreviewed WIP commit `6653148` on
  `kg/rebuild-cha22-opponent`. Track A does not depend on it.

## Design (implementer choices, `kg/rebuild-opponent-mix`)

- **Config.** `env.opponent_mix = {bot: <registry key>, fraction: f}`, with
  `0 < f <= 1`. The keys come from the native registry
  (`owl.rs.kaggriculture_opponent_bots()`: `starter`, `r04`, `ecobot`, `e776`,
  and `cha22` since the anchor merge below).
  `f × n_envs` must be a whole number of at least one; the mix never rounds
  silently. The first that many envs of each rank host the bot, and the rest
  stay self-play. When `opponent_mix` is None (the default) it is omitted from
  the dump, so `config.yaml` and `v3/config_sha256` are unchanged.
- **Seat.** In env `e`'s `k`-th game the learner plays seat `(e + k) mod 2`.
  `k` is 0 at construction and increases by one at every reset, truncation and
  auto-reset. The adapter publishes `learner_mask` (`bool [E,2]`).
- **Native step.** `opponents_rs` gains `HostedSeat`: an engine-less controller
  view refreshed from the host's snapshot, reusing `SeatController`'s
  lifecycle checks. Each hosted env keeps one per game. Inside the step's
  worker, a clone acts on the pre-step snapshot, so a failed batch keeps the
  committed controller. The bot seat's transport must be the absent program
  (length 0, zero tokens); anything else fails the batch. The learner's
  decoded program executes unchanged. Seeds, observations and rewards are
  those of self-play, and no identity enters any tensor.
- **Trainer (`scripts/run_ppo.py` → `PPOTrainer`, the one canonical path).**
  - The rollout forward runs on learner rows only (`forward_learner_rows`, a
    `[rows, 1]` batch; rows are encoded independently), so bot seats cost no
    rollout forward pass.
  - Bot rows are stored as the absent program with zero log-probabilities and
    zero values. The buffer stores the per-step learner mask.
  - `_apply_learner_mask` ANDs it into the value, policy and entity masks.
    These masks weight the policy, entropy, teacher-KL, value and
    teacher-value terms, advantage normalization, return and
    explained-variance telemetry, and every denominator.
  - GAE runs per seat column on that seat's own rewards. A column's seat
    changes only across a `done`, so no bootstrap crosses a seat change.
  - Replay and teacher inputs mark bot rows not playing, so replay admits the
    absent program. The update still encodes those rows (see gaps).
- **Telemetry.** Per update:
  - `train/win_rate_vs_bot` (a draw scores one half), `train/own_bank_mean_vs_bot`,
    `train/opponent_bank_mean_vs_bot`, `train/margin_mean_vs_bot` and
    `train/bank_games_vs_bot`, gathered over ranks;
  - the self-play bank keys cover self-play games only.

  At every `checkpoint_freq`, `_evaluate_against_bot` plays one game per env,
  with every env hosting the bot and each seat covered for even `n_envs`. It
  logs `eval/*_vs_bot` and `eval/{bank_games,win_rate}_vs_bot_seat_{0,1}`.
  Promotion still reads `eval/win_rate_against_last_best` alone. The last-best
  evaluation env never hosts the bot. The bot's name appears only as the W&B
  summary labels `opponent_mix/bot` and `opponent_mix/fraction`.
- **Stateless policy.** The bot key never reaches observations, embeddings,
  heads, losses, rewards, normalization, checkpoints or checkpoint selection
  ([[the-policy-is-stateless-and-observation-only|stateless Decision]]); the
  bot keeps its own scripted state.
- **Kaggle build.** The root crate's controllers sit behind the default Cargo
  feature `fixed-opponents`, with an uninhabited stand-in when it is off. The
  Kaggle submission build (`Dockerfile.kaggle`,
  `scripts/build_kaggle_submission.sh`) passes `--no-default-features` and
  asserts an empty registry. EcoBot and E776 carry no redistribution license
  (`opponents_rs/README.md`). `just rs-lint` lints both configurations.

## Checks of this version

- **None-mix byte identity.** On the pre-change tree `25412a7`,
  `ops/opponent-mix-2026-09-30/baseline_digest.py` recorded two digests:
  - the native digest `257eae38…`: two games' every output byte through nine
    steps, one truncation and two auto-resets;
  - the tiny two-update CPU trainer digest `3ffd53a0…` (metrics and final
    weights).

  Both digests reproduce on this branch
  (`baseline-digest-25412a7.json`, `post-digest.json`). The native one is a
  golden test (`test_self_play_native_env_is_byte_identical_to_the_pre_mix_tree`).
  The trainer one depends on the CPU thread count and stays a receipt.
  `3ffd53a0…` reproduces on `0f70773` with `OMP_NUM_THREADS=2` (torch 2.9.0,
  Accelerate BLAS, Apple M5, Python 3.12.13). One thread gives `2fd98367…`,
  and four threads or an unset variable give `efe76677…`, the value review
  r1 saw on both base and HEAD (`ops/opponent-mix/digest-environment.log`).
  The base was not rebuilt and re-run at two threads for this note. A test also shows
  the self-play trainer never calls a mix helper.
- **Bot actions and learner actions.** `opponent_env_tests.rs` replays every
  bot in both seats on an independent kernel plus `HostedSeat` reference, and
  every state snapshot agrees. A two-bot counterfactual diverges, so the
  learner seat is never played by the bot. `opponents_rs/tests/hosted.rs`
  reproduces `play_match` for every bot in both seats.
- **Learner mask.** `test_learner_mask_excludes_scripted_rows_from_every_loss_term`
  scrambles the bot rows' rewards, values, log-probabilities and observations
  after collection. Every loss term (policy, value, entropy, teacher KL,
  teacher value) is active, and the metrics and weights are unchanged. With
  the mask removed, they change. Covered at fractions 1.0 and 0.5.
- **Other tests.** Seat alternation and auto-reset controller restarts (native
  and adapter), absent-program admission with rollback, determinism across
  thread counts, identical step-zero observations for every bot, telemetry
  keys, checkpoints without opponent bytes, and a two-update fraction-1.0 run
  through `run_ppo.main()` with the bot evaluation at each checkpoint. The
  nine previously skipped learned-seat tests in `test_opponents.py` now run.
- **Mutations.** All 13 source mutations are killed in one run on the
  committed code (`ops/opponent-mix-2026-09-30/mutations.log`). The mutations
  cover:
  - the bot seat playing PASS;
  - a stale controller after auto-reset;
  - a seat rule without the episode;
  - an unchecked scripted transport;
  - a wrong terminal learner seat;
  - the loss mask removed, and the value mask alone removed;
  - the full-batch rollout forward;
  - a learner mask not refreshed after a step;
  - the self-play bank split, the config dump and the last-best eval env;
  - live replay rows.

  M3 first survived, because the seat test derived its expectation from the
  function under test; the test now asserts the flip itself.
- **Full `just prepare`** on `011eb05` exits 0 (`prepare.log`). It covers:
  - root Rust: 298 passed, 5 ignored (291 before);
  - engine: 41 + 9 + 22;
  - opponents: 12 + 2 + 5 + 5;
  - Python: 2,888 passed, 9 skipped (2,853 passed and 18 skipped on the base);
  - mypy, docs-lint and docs-fresh.

  Its peak RSS is 3.3 GB, as on earlier landings. Two earlier attempts failed
  on a clippy and four ruff findings in the new tests, and both were fixed.

## Review r1 and its follow-ups

An independent Claude review (a substitute for Codex under the owner's
approval, not a Codex verdict) approved `0f70773` with no P1
(`ops/opponent-mix/review-r1.md`). It re-ran the fast suites, regenerated the
Cha22 oracle games byte-identically and ran nine mutations. Two survived; the
follow-ups change tests, docs and receipts only, so the vs-cha22 run launched
from `0f70773` computes exactly what this tree computes:

- **P2-1.** `test_fixed_bot_evaluation_attributes_banks_to_the_learned_seat`
  runs `_evaluate_against_bot` on the real hosted-Cha22 env with seat-distinct
  terminal banks. It asserts the per-seat win rates (1 in seat 0, 0 in seat 1),
  the own and opponent bank means and the margin. M4 (own and opponent
  swapped) now fails it.
- **P3-1.** `test_forward_learner_rows_matches_the_full_batch_rows` now mixes
  hosted and self-play envs and plays four sampled steps first, so every
  learned row's value is distinct. M5 (values scattered to `rows.flip(0)`) now
  fails it. Both results: `ops/opponent-mix/r1-mutations.log`.
- **P3-2, P3-3, P3-4 and P3-5** are documented below and in
  `docs/rl-api-specs.md` and `opponents_rs/README.md`. The trainer digest's
  thread count is recorded above. No logged value changed.

## The cha22 anchor presets (merge of Track B, `kg/rebuild-opponent-mix`)

The orchestrating agent relayed the owner's "is anchor thing ready?" as a build
request. `kg/rebuild-cha22-opponent` (`ab09708`, the reviewed Cha22 import with
its WIP `6653148` in history,
[[../references/cha22-opponent-imports-a-view-adapted-closure-with-light-parity|Cha22 Reference]])
is merged here with `--no-ff`. The semantic resolution:

- `Game` keeps both sides' fields. The hosted view also carries the
  `configuration` that Cha22's closure reads, so `HostedSeat` can host Cha22.
- The registry has five keys. `opponents_rs/tests/hosted.rs` now requires every
  key to be hosted in each seat. It adds swapped Starter/R04 and E776/EcoBot
  games and a Cha22 mirror game; before, the test hosted each bot in one seat
  only, despite its name.
- The custody manifest is rehashed, and the READMEs, parity doc and cookbook
  entries of both sides are combined.

Presets (implementer choices where the owner gave no value):

- `configs/kaggriculture_4rank_vs_cha22.yaml` is
  `configs/kaggriculture_4rank_margin.yaml` plus
  `env.opponent_mix: {bot: cha22, fraction: 1.0}`. That is term M exactly
  (`econ_shaping` 0, `econ_bank_weight` 0, `econ_margin_weight` .5,
  `econ_margin_scale` 50,000, `econ_margin_cap` .5, `terminal_scale` .5),
  `muon_lr` 1e-4, `adamw_lr` 5e-6, `checkpoint_freq` 10M and `native_threads` 4.
- `configs/kaggriculture_2rank_vs_cha22.yaml` differs only by Isaiah's
  world-size division (`n_envs` 128, `segments_per_minibatch` 8).
- The documented warm start is the BC best
  (`/root/bc-best/checkpoint_bc_best.pt`, sha256 `fd854587…6f51`) with
  `--load-model-weights-mode model_only`. The fresh optimizer, LR warm-up and
  last_best teacher start at BC. This choice is not owner-given. It follows
  the question "can RL improve on BC", and the margin run's J/2 warm start
  would instead measure a policy that already had RL.

Checks: `test_vs_cha22_presets_are_the_margin_preset_against_cha22` asserts both
presets equal the margin preset apart from the mix (and the world-size
division). `test_cha22_anchor_two_update_run_through_main` runs Cha22 at
fraction 1.0 under term M through `run_ppo.main()` for two CPU updates of
three-step games. Every update logs 2 `train/bank_games_vs_bot`, 0 self-play
games, a win rate in [0, 1], a margin equal to own minus opponent bank and a
nonzero `train/reward_margin_abs_mean`. Each checkpoint's fixed-bot evaluation
covers both seats. No checkpoint contains the bytes `cha22`. Cha22 is
parity-qualified at the default configuration only; these three-step games are
a mechanics check, not a behavioural one. Full `just prepare` on the merge
tree: `ops/cha22-anchor-2026-09-30/prepare.log`. The run statement and launch
command are in `ops/cha22-anchor-2026-09-30/run-statement.md`.

## Gaps and reopening conditions

- **Cha22 throughput and the pod slot.** Cha22 is the largest bot (its 4.9 MB
  route tape is parsed once per process; its controller state is cloned per
  step), and its stepping cost at 64 envs per rank is unmeasured. The pod's four GPUs held the live run
  `M-margin-J2-4rank-20260930` at merge time. When to launch is the owner's
  call.
- **Nothing has been trained with a mix.** Whether PPO improves on BC against
  a fixed bot is the open question this mechanism exists to answer. No GPU
  run and no `nsys` profile has been made. The throughput of bot stepping,
  controller clones and the learner-row gather is unmeasured.
- **Update-phase cost.** The update and the teacher-target pass still encode
  scripted rows, so at fraction 1.0 half the update rows are masked work.
  Compacting them needs a seat-row rollout layout. Reopen if a profile shows
  the update dominating.
- **Qualification scope.** Bot behaviour is parity-qualified at the default
  game configuration only; the short-episode test configurations run the same
  controllers unqualified.
- **Build contexts.** The Docker dependency-fetch stages copy only the root
  manifest and not the path crates. This predates the change: `engine_rs` has
  the same gap. The submission build must keep `--no-default-features` while
  the license gap stands. The `kg/rebuild-7-4-ship` branch's own build lines
  are separate and still need the flag.
- **Evaluation cost and the collective timeout (review r1 P3-5).** The
  fixed-bot evaluation plays `env.n_envs` full games per checkpoint after the
  last-best evaluation. Both run on rank 0 before `broadcast_object`, while
  the other ranks wait under the default NCCL timeout of 10 minutes. The
  self-play run's last-best evaluation alone paused training about 33 s at
  10M steps (orchestrator's report, not re-measured here). The fixed-bot
  evaluation's wall time at the pod's `n_envs` is unmeasured. Check
  `time/eval_seconds + time/eval_vs_bot_seconds` at the first checkpoint
  interval against that timeout. Reopen if it comes within a few minutes of
  it. An odd `n_envs` covers seat 1 once more than seat 0.
- **Degenerate or bot-including telemetry (review r1 P3-2).** The values are
  unchanged and documented in `docs/rl-api-specs.md`:
  - at fraction 1.0 no segment has two learned seats, so
    `train/return_common_mean` and `train/return_zero_sum_abs_mean` log an
    empty-mask mean;
  - `perf/tokens_per_second` and `train/{1,2}p_rate` count scripted rows;
  - `train/reward_bank_mean` averages the bot's seat too, and is 0 under
    term M.

  Learner-masking them would change logged values, which is out of scope
  while the run is live.
- **Custody (review r1 P3-3).** The imported Cha22 headers cite
  `agents/cha22/main.py`, a path that is in neither repository.
  `opponents_rs/README.md` now maps it to the original's SHA-256. The
  Apache-2.0 text exists only as comment lines in
  `notices/cha22/UPSTREAM-SOURCE-COMMENTS.txt`. Before any redistribution,
  add the plain license text and a change statement.
