---
type: "Reference"
title: "Reward reuse preserves objective and critic semantics"
description: "Task 1.5 Stage 1 makes six reward coefficients explicit and checks binary64 admission, independent capped-reward math and f32 rounding; native reward agreement and trainer/pod integration remain pending, while older recipes stay historical."
tags: ["kaggriculture-v3", "adaptation", "rewards"]
status: "verified-scoped"
generated: {"by": "openai/codex", "at": "2026-09-29"}
sources: [{"resource": "reference-branch:kg/reference-2026-09-29/ops/default4096-2026-09-29/results.md"}, {"resource": "reference-branch:kg/reference-2026-09-29/configs/kaggriculture_2rank.yaml"}, {"resource": "reference-branch:kg/reference-2026-09-29/configs/kaggriculture_4rank.yaml"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/default4096-2026-09-29/plan.md"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/v3-port-checks.md"}, {"resource": "user-directive:2026-09-28:reuse-rewards-not-v2-model"}, {"resource": "external-repository:/Users/poonszesen/kaggriculture-v2/ops/myolie-dagger-2026-09-22/selfplay.py"}, {"resource": "reference-branch:kg/reference-2026-09-29/engine_rs/src/lib.rs"}, {"resource": "reference-branch:kg/reference-2026-09-29/engine_rs/src/ffi.rs"}, {"resource": "reference-branch:kg/reference-2026-09-29/python/owl/kaggriculture/env.py"}, {"resource": "reference-branch:kg/reference-2026-09-29/python/owl/model/kaggriculture.py"}, {"resource": "repository:ops/cookbook-setup-checks.md"}, {"resource": "reference-branch:kg/reference-2026-09-29/python/owl/kaggriculture/rewards.py"}, {"resource": "repository:python/owl/train/config.py"}, {"resource": "reference-branch:kg/reference-2026-09-29/tests/kaggriculture/test_env.py"}, {"resource": "reference-branch:kg/reference-2026-09-29/tests/kaggriculture/test_codec.py"}, {"resource": "repository:python/owl/kaggriculture/rewards.py"}, {"resource": "repository:python/owl/kaggriculture/config.py"}, {"resource": "repository:tests/kaggriculture/test_rewards.py"}, {"resource": "repository:tests/kaggriculture/test_configs.py"}, {"resource": "repository:configs/kaggriculture.yaml"}, {"resource": "repository:configs/kaggriculture_2rank.yaml"}, {"resource": "repository:configs/kaggriculture_4rank.yaml"}, {"resource": "repository:docs/rl-api-specs.md"}, {"resource": "repository:ops/rebuild-2026-09-29/briefs/1.4.md"}, {"resource": "repository:ops/rebuild-2026-09-29/briefs/1.5.md"}, {"resource": "repository:ops/rebuild-2026-09-29/stage1-adapter/results.md"}]
---

# Reward reuse preserves objective and critic semantics

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
Disabled components short-circuit; safe overflow handling avoids 0*inf and
premature saturation when tiny W rescales an overflowing inner sum. The transition
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

Future consequence: Stage 2 must execute paired native admission and the native
fixture oracle, confirm fixture custody/schema and extreme coefficient behavior,
and preserve the native reward authority. Task 3.1 still owns runtime factory,
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
