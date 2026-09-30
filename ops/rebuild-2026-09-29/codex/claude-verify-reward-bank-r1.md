Reviewer: independent Claude subagent (substitute for Codex during its usage limit; owner-approved). Not a Codex verdict.

# Verification r1: `kg/rebuild-reward-bank` (owner term A + recipe J halved)

- Branch head: `ab98e73` (`/Users/poonszesen/kg-v3-rewardA`); diff `kg/isaiah-gap-closure...HEAD`, merge base `4731c49`, 3 commits, 39 files.
- Scratch worktree: detached `ab98e73` at `<scratchpad>/verify-bank`, fresh `uv sync` venv (it has been removed).
- Owner decision checked against the transcript. After the options, the owner wrote verbatim "A is good + decrease the LR by half?". The agent then proposed a design, and the owner answered "Go ahead don't waste time."
- Nothing was launched, and no pod was touched.

## Checks I ran

| Check | Result |
|---|---|
| `cargo test --lib kaggriculture` (scratch, HEAD) | 123 passed, 0 failed, 3 ignored |
| `pytest -m "not slow"` on `test_rewards`, `test_native_env`, `test_env`, `test_env_reference`, `test_configs`, `test_training_smoke`, `test_game` | 626 passed, 0 skipped |
| Independent mutations (script and log beside this file) | 12 of 14 killed; 2 survived (findings 2, 3) |
| Textual preset diffs (comments stripped) | only the intended fields differ (see below) |

I did not rerun the full `just prepare`. The implementer's `ops/rebuild-2026-09-29/reward-bank/prepare.log` records it at exit 0 (Rust 280/5 ignored, Python 2,825/18 skipped).

### Mutations (`claude-verify-reward-bank-r1-mutations.{py,log}`)

| ID | Mutation | Result |
|---|---|---|
| V1 | Rust `bank_score` drops `.min(cap_b)` | KILLED (Rust) |
| V2 | Python `bank_score` drops the cap | KILLED: 5 test failures, including native-parity tests |
| V3 | Rust term made relative (own increment minus rival increment) | KILLED: 4 Rust failures |
| V4 | Python oracle made relative | KILLED |
| V5 | Off-by-one previous bank: `env.rs` passes the previous transition's cached before-bank | KILLED (Rust autoreset/env test) |
| V6 | Autoreset reset forgotten: `env.rs` carries the previous after-bank (correct on step 1, wrong after reset or truncation) | KILLED (Rust autoreset test) |
| V7 | Rust `terminal_scale` unchanged (bank cap ignored) | KILLED: 3 Rust failures |
| V8 | Python `terminal_scale` unchanged | KILLED: 6 failures, including native parity and autoreset |
| V9 | Python budget uses `>` instead of `>=` (edge .25 + .1 + .65 = 1.0) | KILLED |
| V10 | Rust budget counts an inactive bank cap | KILLED: 3 failures |
| V11 | Rust rounds the bank term to f32 separately (`relative as f32 + bank as f32`) | **SURVIVED**: Rust tests, rebuilt extension, and `test_rewards` + `test_native_env` (422 passed) |
| V12 | Rust drops the `banks_before` finiteness check | **SURVIVED** (Rust) |
| V13 | Adapter telemetry swaps before and after (sign flip) | KILLED |
| V14 | Python oracle ignores the `w_b > 0` gate | KILLED |

The task's five required mutations are V1/V2 (drop the cap), V3/V4 (make it relative), V5 (off-by-one previous bank), V6 (forget the autoreset reset) and V7/V8 (terminal scale unchanged). All of them are killed. The tree was restored and rebuilt clean afterwards (`tree restored: True`).

## What verifies

- **Reward math (Rust `reward.rs`, Python `rewards.py`).**
  - Per seat: `B = min(cap_b, (w_b * max(0, bank)) / S)`, in the same operation order in both languages. The seat adds only its own `B(after) - B(before)`. The rival's bank never enters (V3/V4 are killed).
  - The relative term plus the bank increment are summed in f64, rounded to f32 once, then the terminal is added in f64 and rounded again, the same schedule as before. V11 shows this schedule is not pinned by any test (finding 2).
- **First transition and autoreset.**
  - `env.rs` reads `before_banks` from the live game right before the kernel step, and `after_banks` before the auto-reset. The first transition of every game, including the very first after construction and every game after an autoreset or a truncation, therefore starts from the reset bank (3,000). No state is carried.
  - The Rust and Python autoreset tests are discriminating: a purchase leaves the first game below 3,000, and V5 and V6 are both killed.
- **Telescoping.** `gamma` is forced to 1.0 for Kaggriculture (`train/config.py`), so the shaping is exactly potential-based. A game's return is the relative penalty difference + `B(final) - B(3000)` + `terminal_scale × sign`. The 719-step Rust and Python walks pin this within an explicit rounding budget.
- **Budget and terminal scale.**
  - Enabled bank caps join the budget in the same order in both languages: `(death + ineffective) + bank`, which must be `< 1`.
  - `terminal_scale = 1 - death - ineffective - bank` (in the presets, 1 - .25 - .25 = 0.5, exact in binary64).
  - Complete-game returns stay in [-1, 1].
- **Validation.**
  - Every new field is required, finite and nonnegative. `w_b > 0` requires `S > 0` and `cap_b > 0`.
  - The binding requires exactly ten keys.
  - The Rust/Python admission predicate is equal on the paired table: 11 old rows, 1 strengthening row and 8 bank rows. The Python table also runs against the native constructor.
- **Nonfinite paths.** None is reachable. Coefficients are finite, `S > 0` when active, and banks are finiteness-checked. An overflowing product or quotient is `+inf` and saturates at `cap_b`, and `inf * 0` cannot occur. The only gap is test coverage of the `banks_before` check (finding 3).
- **`w_b = 0` is byte-identical.**
  - The disabled branch returns `relative as f32` exactly as before.
  - The git-tracked 16-game fixture (unchanged on the branch) replays bit-exactly with the term off.
  - The Rust and Python "legacy formula" tests compare bits across `banks_before`, inactive `S` and inactive `cap_b`.
- **Presets.** With comments stripped:
  - Each `*_bank.yaml` differs from its recipe-J base only in `econ_bank_weight` (0 -> 1.0), `econ_bank_cap` (0 -> 0.25), `muon_lr` (2e-4 -> 1e-4) and `adamw_lr` (1e-5 -> 5e-6).
  - The new `kaggriculture_4rank_bc_finetune.yaml` differs from `kaggriculture_4rank.yaml` only in the two LRs divided by 10.
  - The whole-config model-diff test pins all three.
- **Telemetry.**
  - `reward_bank_mean` is appended every step by `env.py`. `_mean_env_metrics` averages it over steps and ranks and logs it as `train/reward_bank_mean`.
  - `train/return_common_mean` is logged in `ppo.py` for Kaggriculture only.
  - The smoke test checks both through `train_iteration`. V13 is killed, and so were the implementer's P3/P4.
- **The critic's common-mode limit** is documented in the Decision, `docs/kaggriculture-contract.md` ("Rewards") and `docs/rl-api-specs.md`.
- **The Decision** quotes the owner verbatim ("A is good + decrease the LR by half?") in `decider` and in the body, and the cookbook note, index and log are updated together.

## Findings

### 1. (Blocking) The bank presets contradict the design the owner accepted, and the Decision misattributes the values

After "A is good + decrease the LR by half?", the agent proposed the formula `min(cap_b, w_b × max(0, bank)/S)`. It proposed the values in these words: "Proposed values: S = 100,000, w_b = 1.0, cap_b = 0.25", with the stated consequences "A 70k final bank earns **+0.175** and 100k+ earns the full **+0.25**" and "a shared slide into poverty now **costs both seats**". The owner answered "Go ahead don't waste time."

The literal values contradict those consequences. With `w_b = 1`, the score saturates at a bank of 25,000:

- `B(70k) = 0.25`, not 0.175.
- Every bank change above 25k pays exactly 0.

The consequences the owner accepted correspond to `w_b = 0.25` (score `= 0.25 × min(1, bank/100k)`): `B(70k) = .175` and `B(100k) = .25`.

This matters for the purpose of term A. Run J's economy sat at about 72–74k and slid through 68.4k, 52.6k and 17.9k to 5.7k. Under the shipped presets, the whole 73k -> 25k part of that slide is unpunished. The term only activates after the economy has already lost two thirds of its value.

The implementer noticed the 25k saturation and flagged it under "Consequence the owner may want to revisit". But two records state this incorrectly:

- The Decision says "The adopted values are term A with weight 1, scale 100,000 and cap .25".
- The Decision and all three preset headers say "These are the owner's values as given" / "the values are the owner's".

The owner gave no values. The agent proposed them after the owner's answer, and the owner accepted a proposal whose own numbers say 70k -> .175. Recording that as owner-given values breaks the cookbook rule "Never make an interpretation into owner adoption".

**Required edit (either option):**

- **(a) Preferred: match the accepted consequences.**
  - Set `econ_bank_weight: 0.25` in the three `*_bank.yaml` presets. Keep `S` at 100,000, `cap_b` at .25 and `terminal_scale` at .5.
  - Update `_BANK_SHAPING` and the bank-preset tests in `tests/kaggriculture/test_configs.py`, the preset headers, the Decision (values, the "25k" consequence paragraph, and the per-game range, which becomes [-.0075, +.2425]), the decisions index line and the log entry.
- **(b) Keep `w_b = 1`.** Then the owner must confirm it explicitly, because it contradicts the numbers they were shown.

Under either option, replace "the owner's values" with an accurate provenance: the agent proposed the values after the owner's answer, and the owner answered "Go ahead don't waste time." Quote the proposal's value line and its 70k/100k consequences. Launching the bank preset should wait on this edit, because it is the next run.

### 2. (Minor, test gap) The f64-before-f32 rounding schedule is not pinned

Mutation V11 (Rust rounds the relative and bank parts to f32 separately, then adds them in f32) survives:

- every Rust kaggriculture test;
- the rebuilt extension's `test_rewards.py` + `test_native_env.py` (422 passed), including the three live native bank games.

The live games almost never have a nonzero relative delta and a nonzero bank delta that disagree under double rounding. A discriminating deterministic case exists: seat 1 with relative `+0.2` (opponent drought: `0.2 × 1`) and own bank 3,000 -> 3,006. Then `f32(0.2 + 6e-5) = 0.20006` but `f32(0.2) + f32(6e-5) = 0.20006001`. Other cases: relative `.16`, bank 3,000 -> 3,003; relative `.08`, bank 3,000 -> 3,002.

**Fix:** add such a row to the Rust bank tests (`assert_eq!` on bits) and to the Python oracle test. This is what the "exact Python parity" claim rests on.

### 3. (Minor, test gap) The Rust `banks_before` finiteness check is unpinned

Mutation V12 (the check runs over `banks_after` twice) survives the Rust tests. The Python oracle's `banks_before` NaN check is tested. The Rust one is not. It is unreachable from the engine, because banks are finite. **Fix:** add one `cfg.transition(..., [NAN, 0.], [0., 0.], ...)` expecting `Err`.

### 4. (Minor, telemetry) The common-mode metric has no zero-sum counterpart

The Decision itself says `train/return_common_mean` equals `train/return_mean` whenever both seats are valid, which is always the case in Kaggriculture self-play. So it adds no new number. The reopen condition "if `train/return_common_mean` grows large relative to the zero-sum return" has no logged zero-sum return to compare against.

**Suggested:** also log the antisymmetric part, e.g. `train/return_zero_sum_abs_mean = mean(|R0 - R1| / 2)`, or the common-mode variance share. Not blocking.

### 5. (Notes, non-blocking)

- **Unmeasured telemetry cost.** `env.py` computes the float64 oracle every step, over `[n_envs, 2]`, with a finiteness scan, whenever `w_b > 0`. The cost is likely negligible but is unmeasured against the rollout step time. The repo requires end-to-end throughput claims to be measured, so treat any "costs nothing" wording as limited to `w_b = 0`.
- **Native threads.** The new 4-rank recipe-J base leaves `native_threads` at the ranked preset's value. Run J used `env.native_threads=4` as an override. The header discloses this, but a launcher reproducing J/2 must pass the override again.
- **Stale comment.** In `tests/kaggriculture/test_rewards.py`, the comment "The ten paired cases" still counts only the old table.
- **Truncated quote.** The Decision quotes option A with "..." in place of "e.g. Δbank / scale". That is acceptable, but after the finding-1 edit, the proposal it cites should be quoted in full.

## Summary

The code is correct and well tested:

- own-seat reward only;
- the cap binds;
- the first transition and autoreset start from the reset bank;
- exact telescoping at `gamma = 1`;
- matching Rust/Python admission;
- `terminal_scale` .5;
- a bit-identical `w_b = 0` path;
- no reachable nonfinite path;
- presets that differ only in the intended fields;
- telemetry keys present and logged.

One blocker remains. The shipped `econ_bank_weight: 1.0` saturates the score at 25k, which contradicts the accepted design ("70k earns +0.175, 100k+ earns +0.25"), leaves the observed 73k -> 25k slide unpunished, and is recorded as owner-given values when the owner gave none. Two small test gaps and one telemetry gap should be closed alongside.

VERDICT: REQUEST CHANGES
