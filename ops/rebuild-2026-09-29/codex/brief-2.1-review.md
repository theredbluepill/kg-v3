Reviewed the latest brief, including its defaulted-TypeVar revision. No files changed; no test suite run.

1. **Blocker — `BaseModelAPI[ObsT, ActT]` alone does not complete the typed seam.**  
   **Q2:** Defaulted type variables can preserve existing bare Orbit annotations under strict mypy, but several members of [base.py](/Users/poonszesen/kg-v3/python/owl/model/base.py:47) remain Orbit-specific:

   - `ModelOutput.actions` and `ModelServingOutput.actions` require generic output dataclasses.
   - `action_spec` requires a third defaulted type parameter, or removal from the common interface.
   - `count_non_masked_tokens()` accesses `obs.entity_mask`, which Kaggriculture lacks.
   - Teacher arguments must use the current type parameters; bare `BaseModelAPI` means Orbit.
   - Cached teacher targets remain Orbit-specific until the planned Phase 4 refactor.

   **Brief edit:** Specify these changes explicitly. Add `python/owl/train/optimizer.py` to Files: its model arguments must become generic, because their proposed Orbit defaults reject Kaggriculture statically. Add type checks for forward/serve outputs, action-spec pairing and rejection of mixed-game arguments. Existing Orbit annotations need no wholesale rewrite. Making token counting abstract would additionally require updating test doubles in `tests/owl/train/{test_ppo,test_optimizer,test_distributed}.py`.

2. **Blocker — shared factory registration exceeds the encoder-only scope.**  
   [factory.py](/Users/poonszesen/kg-v3/python/owl/model/factory.py:12) accepts Orbit observation/action specs. [FullConfig](/Users/poonszesen/kg-v3/python/owl/train/config.py:17) immediately accesses actor and critic configuration, while this task defers those components.

   **Brief edit:** Prefer deferring shared `ModelConfig`/factory registration until Task 2.3 or 3.1. Instantiate the encoder directly now, and specify fail-fast behavior for unfinished policy/value methods.

   If registration remains in 2.1, add typed factory overloads, runtime rejection of mismatched game specs, and explicit Orbit narrowing. The affected existing files include `model/config.py`, `model/factory.py`, `train/config.py`, `agent/agent.py`, `scripts/run_ppo.py`, and potentially `model/__init__.py` for exports. Also name `train/utils.py`: its compile dispatcher currently accepts only `StatelessTransformerV1`; defining a Kaggriculture compile hook does not wire it into training.

3. **Should-fix — the schema is referenced, not enumerated; all proposed stem widths are correct.**  
   **Q1:** Exact field-list agreement cannot be approved because the brief contains no field list and `types.py` is not implemented. Contract v3 requires 29 top-level observation fields, including `action_mask`.

   The stem calculations are correct:

   | Stem | Channels | Width |
   |---|---|---:|
   | Tile | 15 floats + kind 6 + crop 6 + animal 4 + cell 100 + role 2 | **133** |
   | Actor | 26 floats (`2+12+12`) + slot 241 + cell 100 + role 4 | **371** |
   | Shop | type 8 + slot 8 | **16** |
   | Market | 2 floats + product 9 | **11** |
   | Player | `11+12+5+12+2+2` | **44** |
   | Global | 15 floats | **15** |

   **Brief edit:** Add the complete name/dtype/shape manifest, including exact inventory/rank/storage tensors, float64 banks, globals and nested action mask. Specify fixed one-hot class counts and conversion to the float-channel dtype. Exact integer fields must remain in the schema even when their scaled counterparts feed stems.

4. **Should-fix — sequence and player addition match; masks and flattening need precision.**  
   **Q3:** The proposed order matches contract v3 and [Isaiah’s encoding pattern](/Users/poonszesen/kg-v3/python/owl/model/stateless_transformer_v1.py:557). Adding each projected player-feature row to its corresponding learned player token **before the trunk** is correct.

   **Brief edit:** Define `B=2E`, with each seat flattened into an independent batch row. State the masks in sequence order:

   `actor_mask[482], true[200], shop_mask[8], true[9], s.expand(2), true[1+n], s[1], s.expand(2)`

   Here `s` is that row’s `still_playing`. Isaiah masks player, plan and critic tokens; the brief makes plan always-on and leaves critic masking ambiguous. Contract v3’s always-true `still_playing` makes these equivalent for valid current inputs, but the implementation contract should be explicit.

   Specify `T=705+n_scratch` (**709** at the proposed preset), `global_hidden[B,1,D]`, `board_hidden[B,n,D]`, and an override whose token count equals `token_mask.sum()`.

5. **Should-fix — the L6 bound is conservative, but its derivation is incorrect.**  
   **Q4:** Isaiah has separate `D→D` Q, K, V and output projections, not a combined `D→3D` projection. With `H=int(D*mlp_ratio)`, MLP projections are `D→H→D`; SwiGLU also uses separate gate/value matrices. See [attention](/Users/poonszesen/kg-v3/python/owl/model/stateless_transformer_v1.py:3053) and [FeedForward](/Users/poonszesen/kg-v3/python/owl/model/actor/common.py:11).

   **Brief edit:** Use `Kmax=max(D,H)` and reject `M*Kmax >= 2**31`. For this preset, `Kmax=512`, not 768. Define padded `M=B*T` and packed `M=x.shape[0]`; using padded capacity for both is acceptable if labelled conservative.

   Test `M_safe=(2**31-1)//Kmax`, the next integer, and equality where divisible. Use integer shape checks plus a stub proving rejection occurs before trunk execution. A complete meta-device encoder is unsuitable because packing performs data-dependent operations. Scope the claim to the recorded GEMM mechanism.

6. **Should-fix — the proposed Muon exclusion changes Isaiah’s rule.**  
   Isaiah’s `3D→D` actor input projections are **not** returned by [get_input_layers()](/Users/poonszesen/kg-v3/python/owl/model/stateless_transformer_v1.py:437). Adding the actor input projection there would incorrectly exclude its weight from Muon.

   **Brief edit:** Remove that exclusion. Keep stem input matrices, learned token parameters and final output layers excluded; retain the actor projection’s Muon eligibility. Test actual parameter-group identities. Replace “stem `.output` included” with “stem `.output.weight` included”: biases and normalization parameters use AdamW.

7. **Should-fix — shared config typing and initialization are underspecified.**  
   `ObservationInputStem`, `TransformerBlock` and `MultiHeadSelfAttention` accept `StatelessTransformerV1Config`. Matching attributes on an unrelated Kaggriculture config do not satisfy mypy; overriding its literal architecture discriminator through inheritance is also incompatible.

   **Brief edit:** Explicitly construct a typed Isaiah `trunk_cfg` from the validated shared fields, or introduce a small shared config protocol and add `stateless_transformer_v1.py` to Files. Pin GELU and production `force_flash_attn=True`; CPU-test overrides should be separate.

   Also specify Isaiah’s [reset scheme](/Users/poonszesen/kg-v3/python/owl/model/stateless_transformer_v1.py:385): hidden orthogonal gain `sqrt(2)`, stem-input gain `1`, residual output gain `1/sqrt(2*depth)`, zero linear biases, LayerNorm one/zero, and the stated token initialization. Reusing classes alone leaves default PyTorch initialization.

8. **Should-fix — strengthen test acceptance while keeping every new test cheap.**

   **Brief edit:** Amend the eight tests as follows:

   | Test | Required addition |
   |---|---|
   | 1. Types | Every dtype, shared leading dimensions, pinned size and categorical boundary; valid negative/sentinel and large scaled values must pass. |
   | 2. Topology | Exact stem widths, resolved configuration, block count and initialization. Inspect 6m structure without forwards; test initialization on a tiny model. |
   | 3. Fields/masks | Distinct markers plus an identity trunk to prove offsets and pre-trunk player addition; perturb masked shops as well as actors. |
   | 4. Stateless | Reject non-`None` state at every listed API; reject unexpected recurrent checkpoint keys. Separate encoder tests from deferred head behavior. |
   | 5. Seat isolation | Sound proposal; also require the changed row to respond, preventing constant-output false passes. |
   | 6. Guard | Apply finding 5, including packed/padded dispatch and supported activation branches. |
   | 7. Muon | Inspect actual groups without taking an optimizer step. |
   | 8. Regressions | Retain Orbit suites and project mypy. Tests themselves are not included in the current `py-static` target. |

   Add CPU-mocked flash/compile tests: pack once, unpack once, correct masks and `max_seqlen`, `dynamic=True`, requested compile mode, unchanged state-dict keys and correct dispatch. [Isaiah already has suitable examples](/Users/poonszesen/kg-v3/tests/owl/model/test_stateless_transformer_v1.py:1682).

   Use CPU, `E=1`, small width/depth and inference mode for behavioral checks, retaining all contractual entity slots. No large tensors, real compilation or six-layer numerical runs are needed. These tests establish wiring, not CUDA qualification.

9. **Should-fix — commands and documentation inventory need correction.**  
   This branch’s [pyproject.toml](/Users/poonszesen/kg-v3/pyproject.toml:31) has no `reference` extra, so the two listed pytest commands cannot run as written.

   **Brief edit:** Remove `--extra reference`, retain `just py-prepare`, and include the plan-required `cargo test` after Isaiah-file changes. Add the mapped `docs/model-architecture.md` update and required cookbook note/index/log record to Files.

10. **Note — no reference-model copying is proposed.**  
    The stated stems, role parameters and imported Isaiah blocks comply with rebuilding on Isaiah’s classes.

    **Brief edit:** Make that boundary explicit: reuse Isaiah’s layer classes and packing helpers; rebuild Kaggriculture assembly from contract fields. The reference model remains a semantic/test oracle, with no copied flat-offset encoder, custom trunk or v2 model implementation.

**Verdict: REVISE.**