Reviewer: independent Claude subagent (substitute for Codex during its usage limit; owner-approved). Not a Codex verdict.

# Review r1: fixed-opponent PPO + Cha22 import + vs-cha22 presets

- **Target.** `kg/rebuild-opponent-mix` at HEAD `0f70773` (the merge commit: parents `927972e` Track A and `ab09708` Track B), compared with base `kg/rebuild-reward-margin` `25412a7`.
- **Build job status.** The in-progress `git merge --no-ff --no-commit kg/rebuild-cha22-opponent` in the task brief had already been committed as `0f70773` when this review started. The working tree was clean, `MERGE_HEAD` was absent, and no conflicted paths remained. So the reviewer had no merge to finish and made no commit. I reviewed the committed resolution instead.
- **Method.** I worked in scratch detached worktrees: `/private/tmp/oppmix-r1` at `0f70773` and `/private/tmp/oppmix-r1-base` at `25412a7`. Each got its own `uv sync --frozen` and `maturin develop` (debug). I did not edit `/Users/poonszesen/kg-v3-oppmix`; the only write there is this uncommitted report. I ran fast tests only.

## Owner requirement trace

| Owner, verbatim, 2026-09-30 | Where it lands | Checked |
|---|---|---|
| "OK, for fixed bot, we can use cha22 (check ~/kaggriculture-v2)." | `env.opponent_mix {bot: cha22, fraction: 1.0}` in `configs/kaggriculture_{4,2}rank_vs_cha22.yaml`; Cha22 native import in `opponents_rs/src/native_agents/cha22/` | yes (below) |
| "can we acceleerate this setup?" | Light 3-game parity corpus; rollout forward on learner rows only | parity re-run; forward-row test |
| "implement the new rewrad first before we revisit the cha22 anchor setup." | The presets are exactly `kaggriculture_4rank_margin.yaml` (term M) plus the mix | `test_vs_cha22_presets_are_the_margin_preset_against_cha22` passes and pins equality |
| "is anchor thing ready?" | Presets, run statement `ops/cha22-anchor-2026-09-30/run-statement.md` | Ready to launch on the evidence below; nothing trained |

## Checks run (this version, this machine)

| Check | Result |
|---|---|
| Fast Python suite `pytest tests -m "not slow"` | 2933 passed, 9 skipped |
| Targeted: `test_opponent_mix.py`, `test_configs.py`, `tests/owl/kaggriculture/test_opponents.py`, `test_run_ppo.py` | 341 passed (includes `test_cha22_anchor_two_update_run_through_main`) |
| `cargo test --offline --manifest-path opponents_rs/Cargo.toml --locked` | lib 22, hosted 2, lifecycle 6, oracle_parity 7: all pass |
| `cargo test --lib opponent_env_tests` (root crate) | 7 passed |
| `cargo test --lib` (root crate) | 291 passed, 7 failed. All 7 are Orbit Wars `rules_engine` fixture tests: "No replay parity fixtures found in tests/fixtures/orbit_wars_replays". Those fixtures are gitignored and absent in a fresh worktree. This is environmental, not a branch defect; `ops/cha22-anchor-2026-09-30/prepare.log` shows 298 passed where they exist |
| `scripts/check_engine_trim.py`, `scripts/check_opponent_import.py` | OK / passed |
| `scripts/check_doc_freshness.py`; cookbook lint hook on all 4 changed notes | pass; no lint output |
| **Cha22 parity re-run.** I regenerated all 3 oracle games from the pinned original `main.py` (sha256 `127ed3e6…`, `/private/tmp/cha22-oracle/src/main.py`) under CPython 3.11.15 and kaggle-environments 1.32.7, with `PYTHONHASHSEED=0` and `--preset cha22 --manifest` | All 3 `.jsonl.gz` files and `MANIFEST.json` are **byte-identical** to `opponents_rs/fixtures/oracle-cha22/` (sha256 `0bf36100…`, `1bf5c8d1…`, `8174aa5b…`). Rust `oracle_parity` passes on them. `tests/hosted.rs` shows that `HostedSeat` reproduces `play_match` for every key in both seats, including a Cha22 mirror, so the hosted Cha22 is the parity-qualified Cha22 |
| **None-mix byte identity.** I ran `ops/opponent-mix-2026-09-30/baseline_digest.py` on base `25412a7` and on HEAD `0f70773`, both built here | native `257eae38…` on both, matching the test constant and the receipt. trainer `efe76677…` on **both** base and HEAD, so they are identical. The committed receipts record trainer `3ffd53a0…` for both; see P3-4 |
| Determinism | Cha22 `HashMap`s are iterated only through explicit `order` vectors (`tail.rs:142-190`), with no RNG or clock. The native digest is stable across runs |

## Mutations (9 run, 7 caught, 2 survived)

Each mutation was applied by a script in the scratch worktree, tested, and then reverted with `git checkout`. The tree was clean afterwards.

| # | Mutation | Result |
|---|---|---|
| M1 | `_apply_learner_mask`: value mask not learner-masked | CAUGHT (`test_learner_mask_excludes_scripted_rows_from_every_loss_term[2-2]`) |
| M2 | Advantage normalization over both seats (`_policy_mask(_obs_index(segments.obs, idx))`) | CAUGHT (same test) |
| M3 | Train telemetry: own/opponent swapped in `split_fixed_opponent_games` | CAUGHT (`test_fixed_opponent_telemetry_helpers`) |
| M4 | Eval: own/opponent swapped in `_evaluate_against_bot` | **SURVIVED** (P2-1) |
| M5 | `forward_learner_rows`: values scattered to `rows.flip(0)` | **SURVIVED** (P3-1) |
| M6 | Python env: learner mask not refreshed after `step` | CAUGHT (`test_seats_alternate_and_auto_reset_restarts_the_bot`) |
| M7 | Self-play config dumps `opponent_mix: null` (config identity) | CAUGHT (`test_self_play_config_dumps_exactly_as_before_the_mix`) |
| R1 | Rust `learner_seat` ignores the episode (no alternation) | CAUGHT (`seats_alternate_by_env_and_episode_and_every_reset_restarts_the_bot`) |
| R2 | Rust: bot not re-created on auto-reset | CAUGHT (same test) |

## Review of the requested properties

- **Learner mask in every loss term.** `train_iteration` (`python/owl/train/ppo.py:806-812`) ANDs the learner mask into the value, policy and entity masks. These three feed:
  - the policy, entropy and teacher-KL terms, through `batch_policy_weight`;
  - the value and teacher-value terms, through `batch_value_weight` and `teacher_value_cross_entropy(value_mask=batch_value_mask)`, whose Kaggriculture override uses the mask (`python/owl/model/kaggriculture.py:900-921`);
  - the per-entity clip, through `_policy_entity_mask(batch_segment_obs)` on the learner view;
  - advantage normalization, through `batch_policy_mask`;
  - `player_step_total`, return/explained-variance telemetry, and the advantage mean/std.

  The teacher precompute and the replay both use `_learner_model_view`. The perturbation test scrambles the scripted rows' rewards, values, logp, obs and banks. All five loss terms are nonzero, and the metrics and weights stay bit-identical. M1 and M2 confirm the test's power.
- **GAE per learner seat.** GAE runs per `[env, seat]` column (`advantages.py`). A seat switch happens only at a reset or truncation, and there the `done` or truncation cut stops the recursion. The stored mask is the pre-step mask (`write_step(learner=self._learner)` runs before the refresh at `ppo.py:1137-1145`). `last_values` comes from a full-batch forward, and the rows it adds for scripted seats are masked out.
- **Bot never trained on.** Scripted rows are never forwarded at collection (`forward_learner_rows`). They store the absent program with zero logp and values, and the native env refuses any non-absent submission for them (`env.rs` transport check).
- **Seat alternation and bot reset.** `learner_seat = (env%2 + episode%2)%2`. `episode` increments on reset, truncation and auto-reset (`prepare_reset` `env.rs:468`, `prepare_step` `env.rs:701`), with a fresh `HostedSeat` each time. The failed-batch path clones the controller. R1 and R2 are caught.
- **No opponent identity in the model path.** The bot key reaches only:
  - config validation;
  - the native constructor;
  - `logger.set_summary("opponent_mix/bot")` (`scripts/run_ppo.py:519-523`).

  Promotion reads only the last-best win rate, and the last-best eval env is forced to `opponent_mix=None` (tested). The anchor test asserts that `b"cha22"` is absent from every checkpoint. Observations carry no identity (`test_observations_carry_no_opponent_identity`). The learned seat index is ordinary observation content, not identity.
- **vs-bot telemetry and eval.**
  - Train split by learned seat: correct, and M3 is caught.
  - Eval: I checked it by reading the code. It runs on rank 0 and uses the pre-step learner mask for attribution. With `env.reset()` the learner takes seat 1 in even envs, so 64 envs give 32 games per seat. Sampling is stochastic, like the last-best eval. Test coverage of the attribution is missing (P2-1).
- **Presets.** Reward = term M, pinned through equality with `kaggriculture_4rank_margin.yaml` and the `_MARGIN_SHAPING` check. The 2-rank preset differs from the 4-rank one only in `n_envs` and `segments_per_minibatch`. The warm start (BC best, `model_only`) is a launch argument recorded in the header and in the run statement. `_require_kaggriculture_teacher_source` accepts `--load-model-weights` as the last-best teacher seed (`scripts/run_ppo.py:1155-1182`), so the command as written launches.
- **License and NOTICE custody.**
  - `opponents_rs/notices/cha22/` holds NOTICE, the upstream README, the notebook page and the source comments. The manifest pins every imported or adapted file and every notice by sha256, and `check_opponent_import` passes.
  - The original `main.py` is pinned by hash and is not copied into the repo.
  - The Kaggle build drops the controllers (`--no-default-features` in `scripts/build_kaggle_submission.sh:187` and `Dockerfile.kaggle:65`).
  - Minor gaps are in P3-3.
- **Merge resolution.** No conflict markers remain in the 5 formerly conflicted files. Compared with `ab09708`, `opponents_rs` only adds the hosted view and registry key, and `hosted.rs` covers all 5 keys in both seats. Between `927972e` and `0f70773` there are no code changes under `src/`, `python/` or `engine_rs/src` (only `engine_rs/TRIM_MANIFEST.json`, which `check_engine_trim` passes).
- **Docs and cookbook.**
  - The Decision note, the new Reference note, both folder indexes and a prepended `cookbook/log.md` entry are present, and the lint hook reports nothing.
  - The owner is quoted verbatim, and the interpretation is labelled as the agent's.
  - `docs/rl-api-specs.md` and `docs/rules-parity-coverage.md` are updated, and `check_doc_freshness` passes.

## Findings

### P1

None.

### P2

- **P2-1. The fixed-bot eval's seat attribution is untested** (`scripts/run_ppo.py:2102-2103`).
  - **Problem.** Swapping `banks[seat]` and `banks[1 - seat]` (M4) passes all 341 targeted tests. `test_cha22_anchor_two_update_run_through_main` checks only ranges and `margin = own - opponent`, and both still hold after the swap.
  - **Why it matters.** The run statement names `eval/win_rate_vs_bot` and `eval/margin_mean_vs_bot` as the discriminating observation. A future regression here would silently invert the anchor's answer.
  - **Current code.** Correct by inspection.
  - **Fix.** Add a test with a stubbed model or a deterministic env where the learned seat's bank is known, for example a learner that submits the absent-equivalent weakest program against a bot with a known final bank. It should assert that `eval/own_bank_mean_vs_bot` equals the learner seat's terminal bank per seat.

### P3

- **P3-1. `test_forward_learner_rows_matches_the_full_batch_rows` cannot detect row misrouting of values** (`tests/kaggriculture/test_opponent_mix.py:416-434`). At the reset observation every row's value is identical (measured: all four equal `0.11597609…`), so M5 (`values[rows.flip(0)]`) survives. Compare after a few env steps, or use a model whose value depends on the row.
- **P3-2. Some telemetry is degenerate or includes the bot under a fixed-opponent mix.**
  - `train/return_common_mean` and `train/return_zero_sum_abs_mean` (`python/owl/train/ppo.py:857-864`) require both seats valid. Under fraction 1.0 that never happens, so they log an empty-mask mean. They are finite (asserted), but not meaningful.
  - `model_tokens` (`ppo.py:816`) and `_player_count_rates` count scripted rows that are never forwarded, which overstates tokens/s.
  - `reward_bank_mean` (`python/owl/kaggriculture/env.py:510`) averages both seats, including the bot's. It is 0 under term M, so there is no effect today.

  Document these, or learner-mask them.
- **P3-3. Custody nits.**
  - The imported Cha22 Rust headers point to `agents/cha22/main.py` (`opponents_rs/src/native_agents/cha22/mod.rs:2`, `market.rs:2`, `tail.rs:2`), a v2 path that does not exist in v3. They are byte-pinned imports, so fix the pointer in the NOTICE or README rather than in the files.
  - The Apache-2.0 license text is not included beside the derivative. That is acceptable for training-only, undistributed use, but record it if the code is ever redistributed.
- **P3-4. The trainer digest in the receipts does not reproduce** (`ops/opponent-mix-2026-09-30/post-digest.json` and `baseline-digest-25412a7.json`: `3ffd53a0…`). Today, on this Mac, both base and HEAD give `efe76677…`. The byte-identity claim still holds because base equals HEAD. The receipt should record the environment (torch/BLAS build, thread count) or be refreshed. The script's docstring already warns that the digest is CPU/BLAS dependent.
- **P3-5. The eval wall time before the rank-0 `broadcast_object` is unmeasured** (`scripts/run_ppo.py:661-675`). The vs-bot eval (64 games × 720 steps) now runs after the last-best eval while ranks 1-3 wait in the collective. The process group uses the default NCCL timeout (`python/owl/train/distributed.py:109`). The run statement lists eval wall time as unmeasured. Check the first checkpoint interval's `time/eval_vs_bot_seconds` against the collective timeout.

## Residual unknowns

- Cha22 throughput at 64 envs per rank, and whether the pipeline learns against Cha22, are unmeasured; nothing has been trained.
- Parity covers 3 default-config games only, as stated.
- The root-crate Orbit Wars fixture tests were not re-run here because their fixtures are absent.

VERDICT: APPROVE
