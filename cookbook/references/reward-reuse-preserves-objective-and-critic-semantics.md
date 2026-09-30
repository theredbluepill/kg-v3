---
type: "Reference"
title: "Reward reuse preserves objective and critic semantics"
description: "Task 1.5 made the reward coefficients explicit; the owner's term A (2026-09-30) adds three more (own-bank weight, scale, cap), so nine are required. Python and native admission agree on every binary64 case of the shared table, now with eight bank rows. The independent f64/f32 oracle matches the recorded 11,504-transition native fixture (bit-exact replay with the bank term off) and six live games, three with the bank term. Stage 2 corrected the oracle to native order at overflowing inner sums. The bank term is not zero-sum, and the winner critic cannot represent its common mode. Older recipes stay historical."
tags: ["kaggriculture-v3", "adaptation", "rewards"]
status: "verified-scoped"
generated: {"by": "openai/codex; revised by anthropic/claude-opus-5-5", "at": "2026-09-30"}
sources: [{"resource": "reference-branch:kg/reference-2026-09-29/ops/default4096-2026-09-29/results.md"}, {"resource": "reference-branch:kg/reference-2026-09-29/configs/kaggriculture_2rank.yaml"}, {"resource": "reference-branch:kg/reference-2026-09-29/configs/kaggriculture_4rank.yaml"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/default4096-2026-09-29/plan.md"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/v3-port-checks.md"}, {"resource": "user-directive:2026-09-28:reuse-rewards-not-v2-model"}, {"resource": "external-repository:/Users/poonszesen/kaggriculture-v2/ops/myolie-dagger-2026-09-22/selfplay.py"}, {"resource": "reference-branch:kg/reference-2026-09-29/engine_rs/src/lib.rs"}, {"resource": "reference-branch:kg/reference-2026-09-29/engine_rs/src/ffi.rs"}, {"resource": "reference-branch:kg/reference-2026-09-29/python/owl/kaggriculture/env.py"}, {"resource": "reference-branch:kg/reference-2026-09-29/python/owl/model/kaggriculture.py"}, {"resource": "repository:ops/cookbook-setup-checks.md"}, {"resource": "reference-branch:kg/reference-2026-09-29/python/owl/kaggriculture/rewards.py"}, {"resource": "repository:python/owl/train/config.py"}, {"resource": "reference-branch:kg/reference-2026-09-29/tests/kaggriculture/test_env.py"}, {"resource": "reference-branch:kg/reference-2026-09-29/tests/kaggriculture/test_codec.py"}, {"resource": "repository:python/owl/kaggriculture/rewards.py"}, {"resource": "repository:python/owl/kaggriculture/config.py"}, {"resource": "repository:tests/kaggriculture/test_rewards.py"}, {"resource": "repository:tests/kaggriculture/test_configs.py"}, {"resource": "repository:configs/kaggriculture.yaml"}, {"resource": "repository:configs/kaggriculture_2rank.yaml"}, {"resource": "repository:configs/kaggriculture_4rank.yaml"}, {"resource": "repository:docs/rl-api-specs.md"}, {"resource": "repository:ops/rebuild-2026-09-29/briefs/1.4.md"}, {"resource": "repository:ops/rebuild-2026-09-29/briefs/1.5.md"}, {"resource": "repository:ops/rebuild-2026-09-29/stage1-adapter/results.md"}, {"resource": "repository:src/kaggriculture/reward.rs"}, {"resource": "repository:src/kaggriculture/env_tests.rs"}, {"resource": "repository:tests/kaggriculture/test_native_env.py"}, {"resource": "repository:tests/fixtures/kaggriculture_env_reference_v1.json"}, {"resource": "repository:ops/rebuild-2026-09-29/stage2-adapter/final-report.txt"}, {"resource": "repository:ops/rebuild-2026-09-29/stage2-adapter/claude-review/mutations.txt"}, {"resource": "repository:python/owl/kaggriculture/env.py"}, {"resource": "repository:configs/kaggriculture_4rank_bc_finetune_bank.yaml"}, {"resource": "repository:docs/kaggriculture-contract.md"}, {"resource": "repository:ops/rebuild-2026-09-29/reward-bank/prepare.log"}, {"resource": "repository:ops/rebuild-2026-09-29/reward-bank/prepare-r1.log"}, {"resource": "repository:ops/rebuild-2026-09-29/reward-bank/mutations-r1.log"}, {"resource": "repository:ops/rebuild-2026-09-29/codex/claude-verify-reward-bank-r1.md"}]
---

# Reward reuse preserves objective and critic semantics

## Owner term A — absolute own-bank shaping (2026-09-30)

The [[../decisions/add-absolute-own-bank-shaping-and-halve-the-recipe-j-learning-rates|term A Decision]] adds a non-zero-sum reward component. It quotes the owner ("A is good + decrease the LR by half?"), and it holds the design, the parameter values, the evidence and the critic limit. What changed in this Reference's scope:

- **Coefficients.** `econ_bank_weight`, `econ_bank_scale` and `econ_bank_cap` are required, finite and nonnegative in both `KaggricultureRewardConfig` and the native ten-key dict. A positive weight requires a positive scale and cap. The bank cap joins the active-cap budget, which must stay below 1, and it reduces `terminal_scale`.
- **Reward.** Each seat adds `B(own bank after) − B(own bank before)`, with `B = min(cap, w × max(0, bank) / S)`. The before bank is read before the step, and the after bank before auto-reset. The sum joins the relative term before the first f32 rounding. With the weight 0 nothing is added. The recorded 16-game fixture (made before the change) replays bit-exactly with the term off, and the reference policy module stays byte-identical for fixture custody.
- **Admission table.** The shared table now has eight bank rows after the original eleven. The rows are the same, in the same order, in the Rust table, the Python/native agreement test and Task 1.4's native Python table. They include the exact binary64 budget edge .25 + .1 + .65 = 1.0.
- **Oracle.** The oracle gains `bank_score` and `bank_rewards`, and `transition_rewards` now takes `banks_before`. Three new live 96-transition native games with the bank term match the oracle exactly: the preset values (`w_b` .25 since verify r1), a cap that binds near the start bank, and overflow saturation. Rust and Python rows now pin the single f32 rounding of the relative plus bank sum on inputs where separate rounding changes the bits, and Rust rejects a nonfinite `banks_before`; both were surviving mutations in the verification.
- **Critic semantics.** The earlier claim that the winner critic "is compatible with this recipe because `terminal_scale = 1 - caps` keeps returns in [-1, 1]" still holds for the range. But with the bank term on, part of the return is common to both seats, and a critic whose two seat values sum to 0 cannot represent it. `train/reward_bank_mean` and `train/return_common_mean` measure that part, and `train/return_zero_sum_abs_mean` (mean `|R_0 − R_1| / 2`) measures the part the critic can represent, for comparison. It is unmeasured on any run.

Checks are recorded in the Decision; full `just prepare` output is in `ops/rebuild-2026-09-29/reward-bank/prepare.log` and, for the verify-r1 fixes, `prepare-r1.log` beside it.

## Task 1.5 Stage 2 — native agreement

With Task 1.4 merged, `test_python_and_native_reward_admission_agree` runs all
ten shared cases plus the strengthening case against the real
`rs.KaggricultureEnv` constructor, and the Rust admission table in
`src/kaggriculture/env_tests.rs` and Task 1.4's native Python table carry the
same strengthening case. In Claude's review, Rust admission on raw event
weights, or without the ineffective-cap check, each failed the native agreement
test through the rebuilt extension. The oracle matches the recorded 16-game
native fixture (11,504 transitions, custody-validated loader, unchanged
one-f32-ULP allowance). Three live 96-transition games with extreme
coefficients match native rewards exactly.

One Stage 1 claim was wrong. The oracle rescaled W into the event weights when
the inner death sum overflowed, to avoid "premature saturation". The live tiny-W
game showed native rewards of ±.25 at step 71 against the oracle's ±1e-12:
native binary64 order keeps the overflowed sum infinite and saturates at the
cap. The rescaling is removed and the synthetic expectation corrected; native
production rewards and admission did not change. Changed paths:
`python/owl/kaggriculture/rewards.py`, `tests/kaggriculture/test_rewards.py`,
`test_native_env.py`, `src/kaggriculture/env_tests.rs`. Full `just prepare`
passes (Python 2,224 passed, 6 hardware skips). Trainer use of these rewards
waits for Task 3.1.

## Task 1.5 Stage 1 — explicit configuration and independent oracle

The owner-scoped Stage 1 replaces the Task 3.4 defaulted reward class with one
required six-coefficient definition in `owl.kaggriculture.rewards`. The existing
`reward_shaping` name and top-level `reward_mode='win_loss'` remain. Serialization
adds that mode to the exact native dict; no runtime stub-only TypedDict import,
compatibility alias or production Python reward calculation is introduced.

All six values are finite/nonnegative, inactive zero caps are allowed, and
active caps sum below one. Positive W requires positive death cap and at least
one positive binary64 product W times starvation/drought weight. Positive
ineffective weight requires positive ineffective cap. The ten shared admission
cases include two underflow cases; the native comparison is written but skipped
until Task 1.4. This intentionally strengthens the reference's combined rule.
Claude's Stage 1 review found no test separating the two: a mutant implementing
the combined rule passed all 65 reward/config tests. One extra Python/native
case, W .2 with both death weights 0 and ineffective .001/.1, now requires
rejection and kills that mutant.
The merged base actually checked raw death weights independently of ineffective
weight, so it was not precisely the reference's combined check as the prompt
suggested. The reviewed ABI is the implemented predicate.

The independent float64 functions evaluate capped cumulative S0/D1/I2 penalties,
rival-minus-own deltas and scaled terminal bank sign, rejecting malformed shapes,
nonfinite banks and decreasing counters. Counters 3..31 carry no hidden weights.
Disabled components short-circuit, avoiding 0*inf. An overflowing inner death
sum stays infinite and saturates at the cap even for tiny W, as native binary64
order does (Stage 2 correction below). The transition
oracle models economic f64→f32, terminal addition after promotion to f64, and
final f32 rounding. A literal adjacent-f32 discriminator checks that order.
Complete synthetic 719-transition paths cover both penalized seats, both winners,
ties and saturated caps; errors stay within the explicit sum of output-rounding
ULP budgets. Independent review found that scalar `torch.where` branches rounded
an overflowing coefficient's .7 cap through f32; a failing exact-value regression
now passes with dtype-preserving tensor branches. Bootstrapped partial returns
remain outside the bound.

Changed-path inventory: `python/owl/kaggriculture/rewards.py`, `config.py`,
`tests/kaggriculture/test_rewards.py`, `test_configs.py`, all three
`configs/kaggriculture*.yaml` (explicit ineffective cap .1, unchanged resolved
recipe), `docs/rl-api-specs.md`, the Stage 1 reconciliation in brief 1.5, this
Reference/index/log, and `ops/rebuild-2026-09-29/stage1-adapter/`. The
[[native-game-semantics-use-v3-owned-buffers|native boundary Reference]] inventories
adapter/stub/factory/codec consumers of this config.

Existing-concept search covered reward, economic penalties, underflow, terminal
scale, telescoping and truncation. Independent support is the reviewed Task
1.4 formula/rounding contract and hand-derived CPU examples; the reference
branch is read-only historical support. Reward/config tests passed 65 with 11
binding skips after the missing-module red and the missing-YAML-coefficient red;
with the review case they pass 66 with 12 binding skips.
Full Python preparation passed 1,705 tests with 22 skips. Receipts retain actual
commands, exits and intermediate lint failures, not inherited score evidence.

Future consequence: Stage 2 (above) executed the native admission and fixture
comparisons. Task 3.1 still owns runtime factory,
action transport and buffer-retention integration; Tasks 3.2/3.3 already supply
semantic guards but do not prove native execution. CUDA/pinned DMA and functional
pod smoke remain pending. This stage selects no new recipe or performance/learning
claim and runs no training, model diagnostic or GPU workload.

## Historical reference-branch scope

The following source audit and GPU recipe describe the pinned reference branch,
not this Stage 1 implementation. Older reward modes and test counts are not
current-tree qualification.

## What the source records

The owner permits reuse of v2 reward shaping while explicitly excluding its model implementation. The inspected `selfplay.py` source (lines87,365–432,3277–3380,4167–4201 at audit) defines these distinct recipes:

| Recipe | Exact meaning | Source defaults / scope |
| --- | --- | --- |
| `win_loss` | Terminal `sign(own_bank - other_bank)`: win +1, loss −1, draw 0 | Default terminal mode; no intermediate bank reward |
| `margin` | Terminal `(own_bank - other_bank) / 3000` | Unbounded; v2 uses a linear critic. Its replica-centered training alternative is a separate estimator, not the game reward |
| `win_share` | `(1-w)*sign(margin) + w*(2*share-1)`, `share=own/(own+other)` or .5 when total≤0 | `w=.1`; a bounded surrogate, not exactly the competition ranking |
| Optional own-bank target | `(1-w)*base + w*clip((own-T)/S,-1,1)` | Off when T≤0; w=.3, S=T when unspecified; not zero-sum |

For cumulative starvation deaths S, drought deaths D and ineffective commands I, the base economic penalty per seat is:

`P = min(c_d, W*(k_s*S + k_d*D)) + min(c_i, w_i*I)`

Every step receives `ΔP_other − ΔP_own`; terminal base reward is multiplied by `1-c_d-c_i`. Source defaults are W=0, w_i=0 (both off), configured death cap=.25, starvation weight4, drought weight1 and ineffective cap=.10. Effective c_d is zero unless W>0; effective c_i is zero unless w_i>0. The components are capped separately so early ineffective actions cannot exhaust the death budget. Native `ECON_FIELDS=32`; the first three counters are starvation, drought and ineffective unit commands (PASS is not a verb).

An optional third failure-money component is `min(c_f, k*money)`, with k=0 off and c_f=.10. Its per-counter money values come from an explicit external map; unknown names/negative or nonfinite values fail closed and absent names mean zero. This audit does not invent a cost table or select that optional extension.

V2's bounded-return proof uses gamma1, bounded terminal objectives and bootstrap projection into `[-1+P_own, 1-P_other]`. It does not establish that arbitrary PPO discount/critic settings preserve the same proof. V2's `value_of` bounds `win_loss`/`win_share` through `2*sigmoid(raw)-1`, but leaves margin linear. Dense bank-delta shaping is not the current terminal-margin formula and must be named as a new v3 choice if introduced.

## Interpretation and consequence

Port formulas through the existing v3 reward/data boundary; do not copy the v2 trainer or model. Compute economic increments and terminal bank outcome before native auto-reset, then reset the cumulative baseline for the next game. Preserve antisymmetry for the zero-sum options and report effective caps/weights in configuration and telemetry. A tanh critic is incompatible with unbounded raw margin unless the architecture explicitly supports a linear value mode; reject incompatible recipes rather than silently changing their meaning.

Evaluation uses raw final banks, independently of these shaping terms. Horizon truncation preserves current economic increments before reset.

## Historical reference-branch GPU recipe

Owner asks “你有沒有加econ shaping (0.2)?” before the half-hour run. The preceding verification run used unshaped terminal W/L/D; implementation availability did not mean it was enabled. Both GPU recipes now explicitly use W=.2, starvation/drought weights4/1, death cap.25, ineffective weight0, gamma1 and terminal scale.75. The .2 value is the event coefficient, not the cap or an extra own-bank bonus. Bank-based evaluation/promotion remains unchanged. A bounded two-GPU run confirmed nonzero shaping (max step reward .25, zero-sum mean 0); its receipt is `ops/default4096-2026-09-29/results.md`. That establishes execution, not learning benefit. The Isaiah-aligned critic (per-player winner softmax, value 2p − 1) is compatible with this recipe because `terminal_scale = 1 - caps` keeps returns in [-1, 1].

## Historical source checks and limits

This reference is independently source-inspected, not a v3 reward-performance result. The reference-branch reward module implemented terminal W/L/D, win-only, win/share, raw margin and opt-in separate capped death/ineffective penalties. Config enforces gamma1 for enabled economic shaping and compatible critic output mode; it does not import v2 projected-bootstrap bounds. Native reward/reset cases are included in the 20-test adapter suite. The final receipt reports 772 passing Python cases including shared-PPO integration and incumbent lifecycle checks; this closes source/check reconciliation without certifying reward effectiveness. Exact source SHA-256 is retained in the setup receipt. No coefficient, bank target, dense-delta formula, optional failure table or claimed learning gain is selected merely by documenting these source semantics.
