---
type: "Reference"
title: "Reward reuse preserves objective and critic semantics"
description: "Exact v2 W/L/D, bank-margin and capped-economic formulas with v3 critic/bootstrap compatibility limits."
tags: ["kaggriculture-v3", "adaptation", "rewards"]
status: "verified-scoped"
generated: {"by": "openai/codex", "at": "2026-09-28"}
sources: [{"resource": "repository:ops/v3-port-checks.md"}, {"resource": "user-directive:2026-09-28:reuse-rewards-not-v2-model"}, {"resource": "external-repository:/Users/poonszesen/kaggriculture-v2/ops/myolie-dagger-2026-09-22/selfplay.py"}, {"resource": "repository:engine_rs/src/lib.rs"}, {"resource": "repository:engine_rs/src/ffi.rs"}, {"resource": "repository:python/owl/kaggriculture/env.py"}, {"resource": "repository:python/owl/model/kaggriculture.py"}, {"resource": "repository:ops/cookbook-setup-checks.md"}, {"resource": "repository:python/owl/kaggriculture/rewards.py"}, {"resource": "repository:python/owl/train/config.py"}, {"resource": "repository:tests/kaggriculture/test_env.py"}, {"resource": "repository:tests/kaggriculture/test_codec.py"}]
---

# Reward reuse preserves objective and critic semantics

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

## Checks and limits

This reference is independently source-inspected, not a v3 reward-performance result. The v3 reward module now implements terminal W/L/D, win-only, win/share, raw margin and opt-in separate capped death/ineffective penalties. Config enforces gamma1 for enabled economic shaping and compatible critic output mode; it does not import v2 projected-bootstrap bounds. Native reward/reset cases are included in the 20-test adapter suite. The final receipt reports 772 passing Python cases including shared-PPO integration and incumbent lifecycle checks; this closes source/check reconciliation without certifying reward effectiveness. Exact source SHA-256 is retained in the setup receipt. No coefficient, bank target, dense-delta formula, optional failure table or claimed learning gain is selected merely by documenting these source semantics.
