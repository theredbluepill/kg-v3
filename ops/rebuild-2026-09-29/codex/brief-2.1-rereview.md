**Verdict: APPROVE WITH EDITS.** The revised design addresses the substantive v1 concerns. The remaining edits below make the typing, validation, and chunking requirements precise; none requires an architectural redesign.

Reviewed [brief v2](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/briefs/2.1-encoder.md) against the v1 review, accepted contract v4, and Isaiah’s code at `d8f91d6`. No files modified. Three isolated, cache-disabled mypy probes ran; no repository suites or GPU tests ran.

**Findings 1–10**

| # | Status | Reason |
|---|---|---|
| 1 | **Not fully resolved** | Three defaulted type variables, generic output dataclasses, parameterized teachers, and optimizer annotations address the main gaps. However, “token counting stays concrete for Orbit” must preserve its implementation—not its `obs: ObsBatch` signature. The signature must become `obs: ObsT`; see edit 1. |
| 2 | **Resolved** | Direct encoder construction, deferred factory/config registration, and explicit unfinished-method failures keep 2.1 within scope. |
| 3 | **Resolved** | The table enumerates all **29** observation fields with the correct names, dtypes, and shapes, including exact integer tensors, float64 banks, and the nested mask. All six stem widths are correct. Validation needs the clarification in edit 3. |
| 4 | **Resolved** | `B=2E`, independent seat rows, pre-trunk player addition, sequence masks, named readouts, and token counting are now explicit. Sequence length is correctly **709** with four scratch tokens. |
| 5 | **Resolved** | `Kmax=max(D,H)=512`, actual packed rows, conservative padded rows, and the `>=2**31` boundary match the intended GEMM guard. Newly proposed chunking needs edit 4. |
| 6 | **Resolved** | Stem inputs and token parameters are excluded from Muon; stem output weights and actor input projection weights remain eligible. The encoder-only test scope needs edit 5. |
| 7 | **Resolved** | The typed Isaiah trunk config and explicit initialization scheme address both concerns. Restrict the config-comparison test to shared fields, as in edit 6. |
| 8 | **Not fully resolved** | The strengthened tests cover nearly everything requested. Supported SwiGLU coverage remains unspecified, and the new chunking behavior needs additional bounded assertions; see edits 4 and 6. |
| 9 | **Resolved** | Invalid extras are removed; Python preparation, explicit typing checks, Rust regressions, mapped documentation, and cookbook records are included. |
| 10 | **Resolved** | The brief explicitly reuses Isaiah’s classes and packing helpers while rebuilding Kaggriculture assembly from the contract. The reference model remains an oracle. |

**Answers to the three v2 questions**

1. **Does the manifest match contract v4 exactly?**  
   **Yes, for the observation field/name/dtype/shape manifest.** This includes `[E,2,2,44]` player features, `[E,2,241,12]` inventory/rank tensors, and float64 `[E,2,2]` banks. The schema version remains **3**, despite document version v4. Channel semantics, applicability, and padding rules remain governed by the [accepted contract](/Users/poonszesen/kg-v3/docs/kaggriculture-contract.md:41).

2. **Can the three-parameter generic base leave `ppo.py` / `run_ppo.py` untouched in 2.1?**  
   **Yes, with edit 1.** Default arguments preserve bare Orbit annotations, while Kaggriculture is constructed directly. This establishes encoder-task isolation; it does not complete Kaggriculture rollout storage, training compile dispatch, or cached teacher integration.

3. **Is padded chunking safe with packed/flash dispatch?**  
   **Yes, when it slices complete batch rows after dispatch is selected.** Isaiah’s attention branches on `packed`, and its attention, LayerNorm, and MLP operations do not mix batch rows. Keeping `packed=None` preserves padded execution. Packed overflow should still fail, and forced-flash failure must not become a padded fallback. See [dispatch](/Users/poonszesen/kg-v3/python/owl/model/stateless_transformer_v1.py:644) and [attention](/Users/poonszesen/kg-v3/python/owl/model/stateless_transformer_v1.py:3064).

**Required edits, including new issues**

1. **Should-fix — Finish the generic token-count seam.**  
   Replace the brief’s token-count bullet with:

   > `count_non_masked_tokens(self, obs: ObsT)` remains non-abstract. Its default implementation narrows with `isinstance(obs, ObsBatch)` and returns the Orbit entity count; otherwise it raises `NotImplementedError` requiring an override. Kaggriculture overrides it.

   Retaining the current [Orbit-specific signature](/Users/poonszesen/kg-v3/python/owl/model/base.py:195) makes the Kaggriculture override incompatible. Isolated mypy probes confirmed the failure and the proposed fix. Add positive and mixed-game typing checks for this method.

2. **Should-fix, new — Specify compatible defaulted TypeVars.**  
   State `from typing_extensions import TypeVar` and use `TypeVar(..., default=...)`. The repository supports Python **3.11+**; stdlib defaulted TypeVars or newer generic syntax cannot be assumed. If adding a direct dependency, use `uv add` and include the resulting manifest/lockfile changes.

3. **Should-fix, new — Correct the validation API and category bounds.**  
   Rename the proposed instance `validate()` to **`validate_tensors()`**. Pydantic already defines `BaseModel.validate(cls, value) -> Self`; an instance `validate(self) -> None` fails mypy override checking, confirmed by an isolated probe.

   Also write every categorical bound as **`0 <= value < cardinality`**, including masked slots. Permit negative sentinels only in contract-defined exact channels, and signed values in their defined channels. The current upper-bound-only wording plus blanket negative allowance is ambiguous.

4. **Should-fix, new — Specify dispatch-preserving chunking.**  
   Add this algorithm to the L6 section:

   > Preserve Isaiah’s forced-flash eligibility check, then select packed/padded dispatch once. Guard outside the compiled callable. For padded input, use `rows_per_chunk=(2**31-1)//(sequence_length*Kmax)` and reject if zero. Slice complete batch rows and matching masks, call the selected eager/compiled trunk with `packed=None`, and concatenate in original order. Packed overflow raises without changing attention paths.

   Add tiny tests for strict per-chunk bounds, mask/order preservation, chunked versus unchunked numerical agreement, continued compiled-callable dispatch, and forced-ineligible flash failure before trunk execution.

5. **Should-fix, new — Make encoder-only optimizer acceptance executable.**  
   Specify **`get_output_layers()` returns `()` in 2.1**. It is [abstract in the base](/Users/poonszesen/kg-v3/python/owl/model/base.py:228), and the [optimizer calls it](/Users/poonszesen/kg-v3/python/owl/train/optimizer.py:263).

   Defer the actual actor-input-projection parameter-group assertion to Task 2.3, which introduces that projection. Keep its Muon inclusion rule documented now.

6. **Should-fix — Finish the test specification.**  
   Cover **SwiGLU** if the Kaggriculture config accepts it; otherwise explicitly reject it. Isaiah’s [FeedForward](/Users/poonszesen/kg-v3/python/owl/model/actor/common.py:11) has a distinct gate/value branch.

   Change “resolved trunk config equal to 6m’s except depth” to comparison of the **seven listed shared fields**, allowing the depth difference. Whole-config equality would also compare the deferred actor configuration: Isaiah’s 6m selects `discrete_targets`, while a fresh trunk config defaults to `pure`.