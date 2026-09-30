Reviewer: independent Claude subagent (substitute for Codex during its usage limit; owner-approved). Not a Codex verdict.

# Review r1: reward term M (cash-difference potential) on `kg/rebuild-reward-margin`

- Branch HEAD reviewed: `8b45577` (core `87beaf0` + docs/cookbook `8b45577`), base `kg/isaiah-gap-closure` `111ae7d`.
- Work done in a scratch detached worktree (`.../scratchpad/margin-review`, own `.venv`, own `maturin` build). `/Users/poonszesen/kg-v3-margin` was not edited except for this file.
- Owner request under review: "can we relaunch (before anchor) run, fix the reward, 0.5 Cash Diff (add this in) + 0.5 (Terminal loss 1/-1/0)?", "implement the new rewrad first before we revisit the cha22 anchor setup.", "can we accelerrate?".

## Bottom line

There are no P1 findings. The reward code is correct as specified, and it is safe to launch. The math, operation order, zero-sum property, telescoping, equal start, reset/auto-reset boundary, validation, `terminal_scale`, byte-identity with the term disabled, Python/Rust parity, and the preset (against J/2's effective config) all check out. The requested changes are to documentation, the cookbook and one test pin. None of them changes a reward bit, so they do not block the launch, only the landing. One of them (P2-1) is a semantic limit the owner's 0.5/0.5 split carries: the docs currently state it the wrong way round.

## Checks I ran (this version, scratch worktree at `8b45577`)

| Check | Result |
| --- | --- |
| `cargo test --lib kaggriculture::` | 134 passed, 3 ignored (includes all 7 new `margin_*` tests) |
| `cargo test --lib` (whole root crate) | 284 passed, 7 failed, 5 ignored. All 7 failures are Orbit `rules_engine` parity tests: "No replay parity fixtures found in tests/fixtures/orbit_wars_replays". The fixtures are git-ignored and absent from a fresh worktree, so this is unrelated to the change (284 + 7 = 291 matches the branch's prepare.log) |
| `pytest tests/kaggriculture -m "not slow"` | 1326 passed, 4 skipped (CUDA / BC env skips) |
| `cargo fmt --check`, `cargo clippy --all-targets -D warnings`, `ruff check`, `ruff format --check`, `mypy python/ scripts/` | all clean |
| Resolved preset vs J/2's effective config (`/Users/poonszesen/kg-v3-runs/control-J2-4rank-20260930/rundir/config.yaml`), recursive dict walk of `FullConfig.from_file(...).model_dump()` | Only differences: `econ_shaping` 0.2→0.0, the six new/absent bank+margin keys (bank 0/100000/0, margin .5/50000/.5), and `runtime.n_runtime_gpus` 1 vs 4. The last is filled by `run_ppo.py:1353` from `world_size` at launch, so it is not a preset difference. muon 1e-4, adamw 5e-6, checkpoint_freq 10M, native_threads 4, seed 0, model, PPO and teacher settings all match. |
| Equal start (engine) | `engine_rs/src/lib.rs:1245-1248` builds every farm with `money: starting_money`; `env.rs:501` reads `before_banks` from the game before the step; Rust `autoreset_never_spans_two_games_in_the_margin_term` asserts `[3000, 3000]` on the first transition of both games. Terminal banks are `farm.money` (`lib.rs:1489`), the same source as `seat_money()`, so the telescoped margin equals the evaluated final margin. |
| Telemetry cost | `margin_rewards(...).abs().mean()` on the preset's `[64, 2]` CPU pinned tensors: about 14 µs per step (Mac, 1 thread). That is negligible against a rollout step of tens of ms. |

## Mutations (8 run; 7 killed, 1 survived)

The harness copies the file, applies one exact-string edit, runs the command and restores the file. The worktree was clean after each mutation.

| # | Mutation | Test command | Outcome |
| --- | --- | --- | --- |
| M1 | Rust `margin_score`: drop `.max(-c_m)` (no lower clamp) | `cargo test --lib kaggriculture::env_tests::margin` | **killed**: `margin_score_is_linear_then_clamps_at_both_signs`, `margin_term_is_zero_sum_bit_for_bit`, `margin_episode_telescopes...` |
| M2 | Rust `transition`: pay the level `M(after)` without subtracting `M(before)` | `cargo test --lib kaggriculture::env_tests::` | **killed**: `margin_episode_telescopes_to_the_final_margin_score` (the autoreset test alone does not catch it) |
| M3 | Rust `terminal_scale` ignores `c_m` | same | **killed**: validation, before-f32 and telescope tests |
| M4 | Rust `validate` drops `c_m` from the active-cap budget | same | **killed**: `margin_validation_budget_and_terminal_scale` |
| M5 | `env.py` telemetry: signed mean instead of `abs().mean()` | `pytest test_rewards.py test_env.py test_configs.py` | **killed**: `test_real_binding_reports_the_margin_reward_abs_mean_per_step` |
| M6 | Python oracle `margin_score`: `w * (m / S)` (quotient first) | same | **SURVIVED**. See P3-1. |
| M7 | Preset `econ_margin_scale` 50000 → 40000 | same | **killed**: `test_margin_preset_is_j2_apart_from_the_reward` |
| M8 | Python oracle: before-margin sign flipped (`flip - banks`) | same | **killed**: `test_extreme_value_native_rewards_match_independent_oracle[margin-preset]` |

## Verified properties

- **Formula and order.** `reward.rs:135-143` computes `(w_m * margin / S_m).max(-c_m).min(c_m)`: product, then quotient, then clamp. The doc comment states that order. The Rust test at `w_m = 0.1, S_m = 3` pins the order bit for bit. Overflow gives ±inf, which clamps to ±c_m, and NaN is impossible because `S_m > 0` and the banks are validated finite.
- **Potential form and accumulation order.** In `reward.rs:179-190`, the sum `relative += bank; += margin` is formed in f64 and then rounded to f32 once. For the relative-only and relative+bank cases, it is the same expression tree as `111ae7d`, so the disabled path is byte-identical. Evidence: the Rust `disabled_margin_term_is_byte_identical_to_the_previous_reward` reimplements the pre-M formula, and `test_env_reference.py` replays the recorded 16-game native fixture's rewards exactly.
- **Zero-sum.** `M` is odd bit for bit (IEEE `*` and `/` round sign-symmetrically), `after[s]-after[1-s]` is the exact negation of the other seat's, and with the bank term off `relative` is antisymmetric too. So `r_0 = -r_1` exactly, including on terminal steps. The tests assert this in Rust and against the live binding.
- **Telescoping.** With equal reset banks, a game's economic sum telescopes to `M(final margin)`, up to one f32 rounding per step. The Rust and Python tests bound the drift by an explicit ulp budget over 719 steps that include a lead change and clamp saturation.
- **Validation and budget.** The rules match in Rust (`reward.rs:33-82`) and Python (`rewards.py:66-103`). Coefficients must be finite and nonnegative; `w_m > 0` requires `S_m > 0` and `c_m > 0`; `c_m` joins `((death+ineffective)+bank)+margin < 1`, summed in the same order in both. `terminal_scale` subtracts `c_m` last in both. A disabled term's scale and cap are inert, even when they would break the budget. The two sides raise different error messages for a config that violates two rules at once, but both reject it.
- **PyO3 and schema.** The dict is exactly 13 keys (`bindings.rs:331-345`), `rs.pyi` is updated, every preset states the term explicitly, and `git grep econ_bank_cap` and `git grep econ_margin_cap` list the same code, config and test files.
- **Stateless policy.** The term is computed from game banks inside the native transition and enters only `rewards`. There is no model input or checkpoint state and no opponent identity.

## Findings

### P2-1: Mid-game return-to-go under term M spans [−1.5, 1.5]; the docs claim the winner critic "can represent it"

- `docs/rl-api-specs.md:1473-1474` ("so unlike term A the zero-sum winner critic can represent it").
- `cookbook/decisions/replace-the-reward-with-half-cash-difference-and-half-terminal-sign.md:36` (same claim).
- Also relevant: `docs/kaggriculture-contract.md:236`, `configs/kaggriculture_4rank_margin.yaml:16`, `tests/kaggriculture/test_configs.py:442-452`.

The critic predicts return-to-go from the current state: `V(s_t) ≈ E[M(final) + ½·sign(final) − M(margin_t)]`, not the whole-game return. It outputs `2p − 1 ∈ (−1, 1)` (`python/owl/model/kaggriculture.py:1077-1084`). `M(margin_t)` ranges over [−c_m, c_m], so the return-to-go range is `2·c_m + terminal_scale = 1.5`.

Evidence from the oracle on the preset:
- A seat that trails by 30k at step t and wins by 25k gets a return-to-go of **+1.05** (−1.05 for the rival).
- The extreme swing from −50k to +50k gives **±1.5**.

Every earlier reward kept the return-to-go inside [−1, 1], because the penalties are monotone capped counters and term A's score changes stay within ±c_b. Term M is the first term whose potential is signed, so it is also the first to break that bound. The value target is then outside what the critic can output for states with a large current lead or deficit that later reverses, and the MSE fit is biased in exactly the comeback and collapse states the reward targets.

This is inherent to the owner's 0.5 / 0.5 split with a signed clamp. It is not a coding bug, and it probably affects few states, since typical final |margin| is 10-18k. The docs and the Decision currently state the opposite, and their "within [−1, 1]" holds only for whole-game returns from the reset state.

Requested:
- Replace "can represent it" with the accurate statement: the zero-sum structure is representable, but the magnitude of a mid-game return-to-go can reach `2·c_m + terminal_scale` (1.5 for the preset), which is outside the `2p − 1` range. Record it as a known, unmeasured limit with a reopening condition.
- Optionally (lean), log the fraction of segment returns with |R| > 1 as telemetry. Otherwise use the existing `train/explained_variance` plus terminal margins to judge it.
- Do not change the coefficients without the owner.

### P2-2: The governing cookbook References were not reconciled with term M

- `cookbook/references/reward-reuse-preserves-objective-and-critic-semantics.md:4` ("so nine are required"), `:17` ("the native ten-key dict") and `:21` ("still holds for the range").
- `cookbook/references/kaggriculture-configs-follow-isaiahs-scaling-6m-recipe.md:30` ("six, nine since owner term A").

Both notes cite `repository:` sources that this branch edits (`src/kaggriculture/reward.rs`, `python/owl/kaggriculture/rewards.py`, `env_tests.rs`, `env.py`, the configs and `test_configs.py`). They updated for term A but not for term M. Their current claims are now false: there are twelve coefficients and a thirteen-key native dict. The critic-range claim is contradicted by P2-1. The first note also says (line 135) "Dense bank-delta shaping ... must be named as a new v3 choice if introduced". The Decision does name term M, but the Reference does not link it.

The CLAUDE.md cookbook contract requires revising an existing durable concept when its claims or limits change, and reconciling its description and index together. Requested: add a short term M section to the reward Reference (coefficients, zero-sum, the P2-1 limit, and the new oracle rows), fix the counts in both notes, update their descriptions and `cookbook/references/index.md`, and add a log line. This is a landing blocker only.

### P3-1: The Python oracle's product-then-quotient order is not pinned (mutation M6 survived)

- `python/owl/kaggriculture/rewards.py:219`

Every Python margin case uses `w_m ∈ {0.5, 1e308}`. Multiplying by 0.5 is exact, and 1e308 saturates, so `w*(m/S)` and `(w*m)/S` give identical bits, and the quotient-first mutation passed all 262 tests. The Rust side pins the order with `w_m = 0.1, S_m = 3` (`env_tests.rs`, `margin_score_is_linear_then_clamps_at_both_signs`). The docstring claims native order, and term A's verify r2 added exactly this pin for `bank_score`.

Requested: add a Python row at `w_m = 0.1, S_m = 3` (or a live native game at a non-dyadic weight in `test_extreme_value_native_rewards_match_independent_oracle`) that compares bits with native. Live rewards are unaffected, because the oracle is telemetry and test only.

### P3-2: The Decision's inventory and checks need small follow-ups after this review

- `cookbook/decisions/replace-the-reward-with-half-cash-difference-and-half-terminal-sign.md`, "Checks" and "Gaps".

"No independent verification of this change has run yet" becomes stale once this review lands. When you apply P2-1, P2-2 and P3-1, cite this report (labelled as a Claude substitute, not Codex) and its mutation table.

Also add one gap to the Decision: the warm-started critic was trained on J/2's reward (terminal 0.75 plus capped penalties). Its value outputs are therefore miscalibrated for term M at the start of the relaunch, which confounds early learning curves in the same way as the reset LR schedule the Decision already names.

## Not findings (checked and accepted)

- **`runtime.n_runtime_gpus` 1 vs 4.** The trainer writes it from `world_size`, so it is not a preset difference.
- **The Python `margin_rewards` telemetry recompute.** It is exactly 0 and skipped with the term off, and costs about 14 µs per step with it on.
- **`train/return_common_mean` under term M.** It stays exactly 0, because per-step rewards are exact negations and segment sums over the same order preserve that.
- **Truncation.** The preset keeps `truncation_prob` 0, so no bootstrapped mid-game truncations arise. Where they do arise, the potential form is still correct: the critic supplies the rest.
- **The warm-start path.** It matches `J2-resume-r0208/run_resume.sh`, except that the relaunch uses the new preset in place of `kaggriculture_4rank.yaml -o ... econ_cap=0.2`. The J/2 `config.yaml` lacks the new keys, but `--load-model-weights` does not read the source's config (Decision, "Migration").

VERDICT: REQUEST CHANGES
