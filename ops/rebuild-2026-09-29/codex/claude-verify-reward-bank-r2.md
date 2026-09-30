Reviewer: independent Claude subagent (substitute for Codex during its usage limit; owner-approved). Not a Codex verdict.

# Verification r2: `kg/rebuild-reward-bank` (owner term A + recipe J halved)

- **Branch.** Head `8aeba3d` (`/Users/poonszesen/kg-v3-rewardA`). Diff `kg/isaiah-gap-closure...HEAD`, merge base `4731c49`, 5 commits, 45 files.
- **Scratch worktrees.** Detached `8aeba3d` and `4731c49` under `<scratchpad>/verify-bank-{r2,base}`, each with a fresh `uv sync` venv and `maturin develop`. Both are removed.
- **Owner decision.** I checked it against the session transcript (`3f7ce81c…jsonl`, lines 9738–9763). Option A as offered: "A. Absolute bank shaping: a small per-step reward for own bank growth, e.g. Δbank / scale, capped like the other terms. It directly defends the economy. v2 had a bank-difference term." Owner, verbatim: "A is good + decrease the LR by half?". Next came the agent's value proposal. The owner's next message, "Go ahead don't waste time.", answered a later message that offered the J/2 control run.
- **Launches.** None. No pod was touched.

## Checks I ran (this version, 8aeba3d)

| Check | Result |
|---|---|
| `cargo test --workspace` (ignored Orbit fixtures copied in from the main checkout) | 283 passed, 0 failed, 5 ignored |
| `cargo test` for `engine_rs` and `opponents_rs` | 94 passed, 0 failed |
| `pytest -m "not slow" tests` (whole suite) | 2,828 passed, 18 skipped (flash-attn/CUDA, x86 quant, pod-bound parity, sibling repo) |
| **w_b = 0 cross-version byte identity** (new in r2) | HEAD vs merge base `4731c49`: 48 games × 719 steps from the tracked fixture tokens, each under a randomized relative-reward config. HEAD passes w_b 0 with random *inactive* S ∈ {0, 1, 1e5, 7.3} and cap_b ∈ {0, .25, .9}. The reward arrays are **byte-identical**: both npz sha256 `c033c1b7…a58ae2` |
| **Admission differential fuzz** (new in r2) | 60,000 random coefficient vectors (subnormals, f64 max, exact budget edges) through `KaggricultureRewardConfig` and the native constructor: **0 disagreements** (23,787 accepted) |
| **Rust ↔ Python reward differential** (new in r2) | 6 configs × 4,000 random transitions, with banks including −5000, −0.0, 3,000 ± a few, 25k, 70k, 100k, 1e300 and f64::MAX. 23,888 of 24,000 are bit-equal. The 112 differences are all a signed zero on one degenerate config (finding 2) |
| Parsed-YAML preset diff (independent of the pydantic test) | Only the intended fields differ (see Presets) |
| Independent mutations (23) | **21 killed, 2 survived**: one equivalent mutant and one test gap (finding 1) |

The probes and their receipts are in `claude-verify-reward-bank-r2-probes/`. The mutation script and log are in `claude-verify-reward-bank-r2-mutations.{py,log}`. I did not rerun `just prepare` as a whole (no ruff, mypy or docs-lint run by me). The implementer's `prepare-r1.log` records it at exit 0.

### Mutations

| ID | Mutation | Result |
|---|---|---|
| M1 | Rust `bank_score` drops `.min(cap_b)` (**drop the cap**) | KILLED (Rust) |
| M2 | Python oracle drops the cap | KILLED (native parity, `bank-cap-binds-near-the-start-bank`) |
| M3 | Rust own increment minus rival increment (**make it relative**) | KILLED (6 Rust failures) |
| M4 | Python oracle scores the bank *difference* to the rival (**relative**) | KILLED (native parity `bank-presets`) |
| M5 | `env.rs` passes the post-step bank as "before" (**off-by-one**, zero lag) | KILLED (Rust) |
| M6 | `env.rs` uses the previous step's published before-bank (**off-by-one**, t−2) | KILLED (Rust) |
| M7 | `env.rs` carries the previous after-bank across reset, correct on step 1 (**forget the autoreset reset**) | KILLED (Rust autoreset test) |
| M8 | Rust `terminal_scale` ignores cap_b (**terminal scale unchanged**) | KILLED (4 Rust failures) |
| M9 | Python `terminal_scale` ignores cap_b (**terminal scale unchanged**) | KILLED |
| M10 | Rust terminal sign from `banks_before` | KILLED |
| M11 | Rust score order `w_b * (bank / S)` instead of `(w_b * bank) / S` | **SURVIVED**: Rust, rebuilt extension, parity and env tests (finding 1) |
| M12 | Python oracle drops the negative-bank clamp | KILLED |
| M13 | Rust rounds the relative and bank parts to f32 separately (r1 V11) | KILLED (the r1 fix holds) |
| M14 | Rust skips the `banks_before` finiteness check (r1 V12) | KILLED (the r1 fix holds) |
| M15 | Rust validate drops the `cap_b > 0` requirement | KILLED |
| M16 | Python budget counts an inactive bank cap | KILLED |
| M17 | Adapter `reward_bank_mean` sums the seats instead of averaging them | KILLED |
| M18 | `return_zero_sum_abs_mean` drops the `/ 2` | KILLED |
| M19 | `return_common_mean` reads seat 0 only | KILLED |
| M20 | 8-rank bank preset w_b .25 → .2 | KILLED |
| M21 | 2-rank bank preset S 100,000 → 10,000 | KILLED |
| M22 | 4-rank bank preset `muon_lr` 1e-4 → 1.1e-4 (the log's label for M22 is garbled; this is the actual edit) | KILLED |
| M23 | Rust always adds the bank term (removes the `w_b > 0` gate) | SURVIVED, **equivalent** (see below) |

The task's five required mutation classes are drop the cap (M1/M2), make it relative (M3/M4), off-by-one previous bank (M5/M6), forget the autoreset reset (M7) and terminal scale unchanged (M8/M9). Every one of them is killed. After each mutant the tree was restored and the extension rebuilt (`tree clean: True`).

M23 is equivalent, so it is not a gap. With w_b = 0, `bank_score` returns +0.0 on both sides, and `relative` is never −0.0: each penalty delta is ≥ +0 by monotonicity, and `x − x = +0` in round-to-nearest. So `relative + (+0 − +0)` has the same bits as `relative`. The cross-version byte-identity probe confirms this independently.

## What verifies

- **Reward math against the spec.**
  - `B = min(cap_b, (w_b · max(0, bank)) / S)`, with the same operation order in Rust and Python. The differential probe confirms this at non-power-of-two `w_b` = .37 and S = 7.3.
  - Seat `s` adds only `B(after_s) − B(before_s)`. The sum is taken in f64 with the relative term and rounded to f32 once. On a terminal step it is promoted, `terminal_scale · sign(bank_after)` is added, and it is rounded again.
  - The rival's bank never enters (M3/M4/M10 are killed).
- **First transition and autoreset.** `env.rs` reads `before_banks` from the live game right before `step_with_market_metrics`, and `after_banks` right after it. Only then does the auto-reset replace the game.
  - The first transition after construction, after an autoreset, or after a truncation therefore starts from the reset bank.
  - No state is carried. M5, M6 and M7 are all killed.
- **Cap.** The score saturates at cap_b. With the presets, 3k scores .0075, 25k .0625, 70k .175, 73k .1825, 99,999 .2499975, and 100k+ .25. A per-game bank term lies in [−.0075, +.2425].
- **Telescoping.** `rl.gamma` must be 1.0 for Kaggriculture (`train/config.py`). At that gamma a game's bank return is exactly `B(final) − B(reset)`, which is potential-based. The 719-step Rust and Python walks pin it within the rounding budget.
- **Per-seat independence.** The own-seat-only tests kill M3/M4. The adapter metric averages over every seat (M17 is killed).
- **Admission predicate.**
  - Rust `validate()` and `KaggricultureRewardConfig` are equal on the 20-row paired table. The Python table also runs through the native constructor.
  - My 60k-case differential fuzz found no disagreement, including at the exact binary64 budget edges (e.g. `.25 + .1 + .65 = 1.0`, with the order `(death + ineffective) + bank`).
- **Validation.** All three fields are required, finite and nonnegative (pydantic `strict`, `allow_inf_nan=False`; Rust `is_finite() && >= 0`). `w_b > 0` requires `S > 0` and `cap_b > 0`. The binding requires exactly ten keys.
- **Terminal scale.** `1 − death − ineffective − bank` (only active caps count), in the same order in both languages. It is 0.5 in the presets, exact in binary64.
- **w_b = 0 regression.** The regression is byte-identical, shown three ways:
  - the tracked 16-game fixture replays bit-exactly at HEAD;
  - my cross-version probe (48 randomized-config games, HEAD vs merge base) matches byte for byte;
  - M23 is equivalent.
- **Presets** (parsed-YAML diff).
  - `*_bank.yaml` vs its recipe-J base: only `econ_bank_weight` 0 → .25, `econ_bank_cap` 0 → .25, `muon_lr` 2e-4 → 1e-4 and `adamw_lr` 1e-5 → 5e-6. S stays 100,000 and `checkpoint_freq` stays 10M.
  - The new `kaggriculture_4rank_bc_finetune.yaml` differs from `kaggriculture_4rank.yaml` only in the two LRs ÷ 10.
  - Every other Kaggriculture config adds only the explicit off triple.
  - `native_threads` stays 2. Run J passed `env.native_threads=4` as an override, and the README says a J/2 reproduction must pass it again.
- **Critic common-mode limit and telemetry.**
  - The limit is documented in the Decision, in `docs/kaggriculture-contract.md` ("Rewards") and in `docs/rl-api-specs.md`.
  - `reward_bank_mean` is appended every step by `env.py`. `_mean_env_metrics` logs it as `train/reward_bank_mean` over steps and ranks.
  - `train/return_common_mean` and the r1-requested `train/return_zero_sum_abs_mean` are logged in `ppo.py`, for Kaggriculture only.
  - The smoke test checks all three through `train_iteration`. M17, M18 and M19 are killed.
- **Decision provenance.**
  - `decider` and the body quote the owner verbatim ("A is good + decrease the LR by half?"). Option A is quoted in full, matching the transcript.
  - The body quotes the agent's value proposal verbatim. It correctly states that the owner neither gave nor confirmed the numbers, and that "Go ahead don't waste time." answered the control-run offer.
  - The r1 blocker is resolved: `w_b = .25` realises the proposal's stated consequences (70k scores .175, 100k+ scores .25), and the records no longer call the values the owner's.
  - The note, the decisions index and a prepended log entry were updated together. The older log entry, which still says "bank 1 / 100,000 / .25", is history, and the newer entry supersedes it.
- **Nonfinite paths.** None is reachable.
  - Coefficients are finite, `S > 0` when active, and both bank vectors are finiteness-checked in Rust (M14 is killed) and in Python.
  - An overflowing product or quotient is +inf and saturates at cap_b, and `0 · inf` cannot occur.
  - `|economic| < 1`, so the f32 cast cannot overflow. The final rewards are checked again.

## Findings

### 1. (Minor, test gap) The score's binary64 operation order is not pinned

M11 (Rust `w_b * (bank / S)`) survives every Rust test, the rebuilt extension's parity tests and the env tests. The docs and the Decision state "product, then quotient, then cap" as the parity contract. All the pinned rows use w_b ∈ {1, .25} (powers of two, for which both orders are exact) or saturate at the cap.

- **Effect on the presets:** none. At w_b = .25 the two orders are bit-identical.
- **Discriminating rows:** w_b = .1, S = 100,000.
  - `(0.1·70000)/1e5 = 0.07` but `0.1·(70000/1e5) = 0.06999999999999999`.
  - Similarly at banks 3,003, 3,006 and 12,345.
- **Suggested fix:** add one Rust `assert_eq!` on `bank_score(70_000.).to_bits()` at w_b .1, and the matching Python oracle row.

This does not block, because the shipped values are immune.

### 2. (Minor, note) Signed-zero divergence between the oracle and native on non-terminal steps

`transition_rewards` always adds `terminal * dones`, which is +0.0 on a non-terminal step. So an f32 economic value of −0.0 becomes +0.0 in the oracle, while native returns −0.0. This happens only when the f64 economic sum is a tiny negative that underflows f32.

- **Where it shows:** my differential run hit it only at w_b = 5e-324, S = 1e300 (112 of 4,000 rows for that config, every one a `0` vs `0x80000000` difference). Validation admits that config: it is the "inert bank term" the Decision already lists as a limit.
- **The class is pre-existing:** the admitted relative-only row with W = 1e-300 can produce the same −0.0. The bank term only adds a route to it.
- **Effect:** it is unreachable at the presets and numerically irrelevant, since ±0 are equal in every sum.
- **Suggested fix:** either scope the "bit-exact parity" wording to exclude the sign of zero, or make the oracle add the terminal only where `dones` is true.

### 3. (Minor, doc) Pre-change run configs no longer load

The three fields are required with no defaults. I checked this: `FullConfig.from_file` on the merge-base `kaggriculture_4rank.yaml` fails with three "Field required" errors. So a `config.yaml` written before this change no longer validates. That includes the BC best's sibling config, which `bc.py` writes as a PPO FullConfig, and J/2's `checkpoint_final.pt` directory. Such configs fail if they are used as `rl.teacher_init` (`_checkpoint_config_path`) or for a full resume.

- **Planned launch:** unaffected. It uses `--load-model-weights … model_only` with `teacher_init: null`, and that path reads no sibling config.
- **Remedy:** migration is semantics-preserving. Adding `econ_bank_weight: 0.0, econ_bank_scale: 100000.0, econ_bank_cap: 0.0` reproduces the pre-change reward exactly. That matches the repo's "explicit schema migration" rule.
- **Suggested fix:** one sentence in the Decision's limits or in the README.

### 4. (Non-blocking, evidence currency) The J/2 control run now exists

After this branch's records were written, the owner-approved J/2 control (4-rank, recipe J with the LR halved and no bank term; W&B `nw3klj2s`) completed. Its bank fell about 43% after the LR peak and plateaued at 47–53k through iteration 147. J fell to 5.7k.

- **What it changes:** it is the LR-only arm. The Decision's sentence "a run of the bank presets cannot attribute its outcome to either change alone" is now only partly true. Bank-preset versus J/2 isolates term A at the halved LR, with one seed each.
- **Where to record it:** that run's own episode. The Decision's limits can link it when this branch lands. Not needed for this code change.

### 5. (Notes)

- The Decision's "Adaptation inventory → Python" bullet still says `ppo.py` logs only `train/return_common_mean`. The zero-sum key is described elsewhere in the note. This is a cosmetic nit.
- The adapter's per-step float64 oracle cost is still unmeasured, and the docs now say so correctly.
- I could not independently exercise the cookbook-lint hook: my payload also passed a deliberately broken note, so the hook did not run on my input. I make no claim about lint beyond the implementer's `prepare-r1.log` (markdown docs-lint and docs-fresh).

## Summary

The r1 blocker is fixed, and so are the r1 test and telemetry gaps:

- the presets use w_b .25;
- the provenance is recorded accurately;
- the f32 single-rounding and the `banks_before` finiteness pins are in place (M13 and M14 are now killed);
- the zero-sum counterpart metric exists and is tested.

The reward math matches the spec on the first transition, autoreset, cap, telescoping and per-seat independence. Rust and Python admission are provably equal on the paired table and on a 60k-case fuzz. The terminal scale is 0.5. The w_b = 0 path is byte-identical to the merge base across randomized configs. The presets differ from their bases only in the intended four fields. The telemetry keys exist and are logged, the critic limit is documented, and the Decision quotes the owner verbatim. No nonfinite path is reachable.

What remains is minor: one unpinned operation order that the shipped values are immune to, a signed-zero oracle nit on a degenerate config, and a doc note about migrating pre-change configs.

VERDICT: APPROVE WITH EDITS
