**The brief still needs revision.** The quantity-low correction, event layout and coupled-Gumbel sequence are right, but the revised mask table introduces two indexing errors, and replay validation remains underspecified.

I treated v2 as superseding v1. Reference line numbers below refer to `kg/reference-2026-09-29`. This was read-only source inspection; no files changed or tests ran.

**1 — Not fully resolved: quantity-low is fixed, but two new table rows are wrong.**

[Brief lines 56–68](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/briefs/2.3-action-heads.md:56) correctly change the low mask to `unit_quantity_low[present, int(high == 0)]`. However, `unit_kind[live]` and `market_kind[available]` are incorrect for the declared tables: these are vocabulary vectors, not tables indexed by runtime booleans.

The complete derived table is below. `δ0(W)` means a width-`W` mask admitting only zero.

| Slot | Stored shape | Effective mask |
|---|---|---|
| 1 `unit_kind` | `[20]` | `where(unit_live[..., None], unit_kind, δ0(20))` |
| 3 `unit_item` | `[20,16]` | `unit_item[kind]` |
| 4 `unit_quantity_present` | `[20,2]` | `unit_quantity_present[kind]` |
| 5 `unit_quantity_high` | `[2,32]` | `unit_quantity_high[present]` |
| 6 `unit_quantity` | `[2,2,32]` | `unit_quantity_low[present, (high == 0).long()]` |
| 7 `market_kind` | `[8]` | `where(available[..., None], market_kind, δ0(8))`, then remove HIRE where capacity is exhausted |
| 8 `market_item` | `[8,16]` | `market_item[kind]` |
| 9 `market_quantity_high` | `[8,32]` | `market_quantity[kind]` |
| 10 `market_quantity` | `[8,32]` | `market_quantity[kind]` |

This follows the native `State::allows`/`advance` rules in `myolie_sampler.rs:147–258`, table extraction in `gpu_sampling_grammar.py:68–110`, and actual selection in reference `kaggriculture.py:578–675`.

The dependencies check out independently:

- Unit kinds are `1..18`. Item support is `1..12` for kinds 6/7, `1..5` for kind 8, otherwise `{0}`.
- Only kinds 6/7 allow `present=1`. Absent quantity forces both digits to zero; present quantity excludes low zero only when high is zero.
- Market item support is `1..5`, `{1,9}`, `10..12`, and `1..9` for kinds 3, 4, 5, and 6 respectively; otherwise `{0}`.
- Both market digits admit `0..31` for kinds 3–6, otherwise `{0}`. Rust converts kinds 4–6 to its quantity-bearing state after slot 8. The Python extractor explicitly verifies equal high/low support.

**Edit:** Replace the two kind rows, retain the native digit-support assertion, and add the requested quantity cases: omission, rejected explicit unit zero, `1/31/32/1023`, and accepted market zero.

**2 — Not fully resolved: replay enforcement is restored, but its complete contract is missing.**

[Lines 71–76](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/briefs/2.3-action-heads.md:71) correctly require `evaluate_actions` to reject malformed programs. This fixes the mistaken reliance on native `step`.

However, reference `_decode` performs three checks: support validity, exact reconstructed-length equality, and **full reconstructed-token equality** (`kaggriculture.py:768–783`). The brief should preserve these directly rather than approximate the last check with prose.

**Edit:** Specify:

- Check token/length shapes and `int64` dtypes before policy execution, as reference `_decode:747–755` does.
- Make every replay-derived gather, table lookup and embedding index safe before execution. Record invalid original values in device flags; use safe temporary indices without admitting or returning repaired actions. A host check after the kernels cannot prevent an earlier out-of-range access.
- Compare the entire canonical token tensor with the supplied tensor. This includes actor ordinals, reserved target, unit-frame market fields, market-frame unit fields, STOP bits and every padding element.
- For market queue STOP index `s`, require `lengths = where(live, actor_counts + s + 1, 0)`. Inactive rows require all-zero tokens and length zero.

**The single batched synchronization is sound**, provided it is implemented outside the compiled tensor policy and before evaluation returns to loss computation. `.all()` produces a device reduction; the host conversion/branch is the synchronization. Aggregate support, length and canonical-equality flags first, then perform one host transfer. The reference already transfers one compact flag vector through `_check_flags:121–126`, preserving useful error messages.

Keep `valid` on an explicitly typed internal policy result and consume it there. If sampling flags are meant to be public, define that API: current [`ModelOutput`](/Users/poonszesen/kg-v3/python/owl/model/base.py:55) has no validity field. Remove the unmeasured assertion that the synchronization is “cheap compared with the update.”

**3 — Not fully resolved: runtime arguments are present, but their application remains incomplete.**

[Line 77](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/briefs/2.3-action-heads.md:77) now names all four required inputs. It still omits live gating, frame placement and the forced sentinel.

**Edit:** Add these equations:

```text
unit_live[b,f] = live[b] & (f < actor_counts[b])
available[b,p] = live[b] & (p < order_limits[b])
market_frame[b,p] = actor_counts[b] + p
market_live[b,p] = live[b] & (p <= first_final_NONE[b])
```

Unavailable positions force `NONE`, including the sentinel at `p == order_limits`. Padded units use singleton-zero support. Inactive rows produce zero tokens, lengths, log-probabilities and entropies.

Also pin default `hire_limit=241`, count submitted HIREs regardless of execution success, and test mixed actor counts, order limits and inactive rows against native plans. These are reference behaviors at `kaggriculture.py:578,621,630,663–735`, not additional model choices.

**4 — Not fully resolved: the central replay-test mistake is fixed; requested coverage is incomplete.**

[Line 78](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/briefs/2.3-action-heads.md:78) correctly saves one sampling result and replays it unchanged, split and permuted. The explicit FP32 tolerance is appropriate.

**Edit:** Freeze weights and save/compare **per-slot log-probabilities and entropies**, plus their frame sums, restoring original row order after permutation. Frame sums alone can conceal offsetting slot errors. Add malformed-program rejection tests and the reference CPU `backend="eager", fullgraph=True` density/gradient test (`tests/kaggriculture/test_model.py:483–505`), keeping host validation outside the captured core.

Explicitly hand production compiler/BF16 rollout-versus-minibatch and chunk-boundary checks to Phase 6, and link Task 3.6’s pre-update log-ratio alarm. The eager test does not qualify the historical L6 failure.

**5 — Not fully resolved: initialization intent is correct; wiring is still ambiguous.**

[Line 79](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/briefs/2.3-action-heads.md:79) states the correct token standard deviation, output gain and Muon membership. But the brief still describes registering `nn.Embedding` modules. Isaiah’s [`_init_input_layer`](/Users/poonszesen/kg-v3/python/owl/model/stateless_transformer_v1.py:3125) initializes only `nn.Linear` and `nn.Parameter`.

**Edit:** Explicitly return `market_position.weight` and each slot embedding’s `.weight` from `get_input_layers`. Extend [`reset_parameters`](/Users/poonszesen/kg-v3/python/owl/model/kaggriculture.py:188) to initialize every actor `head.out` through `_init_linear(..., gain=0.01)`, giving orthogonal weights and zero bias. Listing outputs in `get_output_layers` alone does not extend the current critic-only reset.

Require reset and optimizer-membership tests. `actor_input_proj.weight` and head `.up.weight` remain Muon-eligible.

**6 — Resolved: event log-probability layout matches the reference.**

[Line 80](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/briefs/2.3-action-heads.md:80) correctly specifies:

```text
event: [E,2,252,12]
per_player_entity = event.sum(-1)
launch = zeros_like(per_player_entity)
```

This matches reference `kaggriculture.py:784–804`. Implicit slots 0/2/11 have zero density; an early sampled market `NONE` retains its density in **slot 7**.

**Completion edit:** Explicitly mirror the entropy layout: `entropies.event` has the same shape, its sum supplies `per_player_entity`, and `components[name] = event[..., slot]`.

**7 — Resolved: the coupled-Gumbel sequence is correct.**

[Lines 81–85](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/briefs/2.3-action-heads.md:81) correctly distinguish exclusive **raw-HIRE** counts for correction from exclusive **final-HIRE** counts for density, reusing the same perturbed scores. Together with “first final `NONE`,” this matches reference `kaggriculture.py:635–664`.

**Completion edits:** Say “eight independent Gumbels per market position, independent across positions,” avoiding the possible interpretation of one shared scalar. State that the first final `NONE` retains its kind probability, the forced sentinel has zero density, later positions are marginalized, and `EMPTY` consumes an order. Connect enumeration to the actual sampler/replay implementation and force correction outcomes of both `NONE` and `EMPTY`.

**Additional document edit:** Replace superseded v1 statements instead of leaving contradictory requirements in the operative brief—particularly lines 16, 22, 29, 32 and 39. Removing sampling synchronization is a throughput choice; it is not the L6 corruption remedy.

VERDICT: REVISE