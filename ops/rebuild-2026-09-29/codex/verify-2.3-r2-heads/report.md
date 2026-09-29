# Independent re-verification — Task 2.3 grammar action heads

Date: 2026-09-29. Reviewer: Codex. Branch: `kg/rebuild-heads`.

Reviewed `0e989a123343df11eb3782907188e5df77bdd36e..88f95f64df2164f4198335e6719a2075dee6d567` against the approved Task 2.3 brief v3, contract v4, the brief-review receipts, and the requested reference oracle at `65f0eac5bb00b18a9d3acce319c2a231cbd5dff0` (`kg/reference-2026-09-29`). Reference files: `python/owl/model/kaggriculture.py`, `python/owl/kaggriculture/gpu_sampling_grammar.py`, `engine_rs/src/myolie_sampler.rs`, and `tests/kaggriculture/test_model.py`.

Target: verify the heads' grammar, replay, architecture and overflow requirements and close every prior-round finding, without leaving tracked modifications. Completion condition: source/oracle audit complete, requested commands pass, at least two independent mutations fail their targeted tests, restored-source checks pass, and every tracked file matches its original SHA-256. All conditions were met. Two native subagents performed independent read-only grammar and architecture audits; neither found an actionable issue.

## Findings and previous-round closure

**No actionable findings. No fix requested.**

The specified previous-round report (`/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/codex/verify-2.3-r1.md`, fingerprint in `environment.json`) contains one low-severity documentation finding. It is resolved in all three locations:

- `python/owl/model/kaggriculture.py:525–530`
- `docs/model-architecture.md:744`
- `cookbook/references/kaggriculture-grammar-heads-sit-behind-isaiahs-actor-projection.md:32`

These now say sampling adds no **policy-validation** host synchronization, and identify retained packing and oversized-batch chunk planning as host-visible operations. No globally synchronization-free forward claim is made in those current descriptions. The approved brief's original sentence is historical specification wording; the correction implements the prior review's requested qualification.

## Verified requirements

| Area | Evidence and conclusion |
| --- | --- |
| Tables and supports | All eight tables match the Rust `State::allows`/`advance` rules and reference extractor by source comparison. Unit kinds 1–18, item subsets, absent quantity, explicit-unit-zero rejection, digits 1/31/32/1023 and accepted market zero are covered. The native-table equality test is explicitly skipped pending the binding. |
| Runtime context | Own actor counts, order limits, `still_playing`, frame placement and submitted-HIRE capacity follow the brief equations. Padding/inactive rows are canonical zero; lengths include STOP. Cash and successful execution do not enter syntax masks. |
| Coupled Gumbel | The actual sampling path creates per-entry independent Gumbels, computes exclusive raw-HIRE counts, and removes HIRE using the same perturbed scores. Density uses exclusive final-HIRE counts. Enumeration drives the actual coupling helper and replay core at budgets 0/1/2/3/10, covers NONE and EMPTY corrections, and checks individual program probabilities and total mass to 1e-9. |
| STOP | First final NONE retains its kind density; forced sentinel and later positions contribute zero density and entropy. EMPTY consumes an order. |
| Replay safety | Shape/dtype checks precede encode. Every replay-derived gather/table/embedding index is clamped before use, with original invalidity retained in flags. Observation-derived market frame indices stay within the fixed actor/frame envelope. Full canonical equality includes ordinals, reserved target, cross-kind fields, STOP, padding and inactive rows. |
| Replay validation | Support, length and canonical groups are computed inside `policy_core`. `check_replay_flags` reduces them and first failing rows into one compact transfer outside the core before returning to loss computation. `valid` remains internal; sampling does not call the host admission check. |
| Densities and gradients | Events are `[E,2,252,12]`, implicit slots 0/2/11 are zero, frame fields sum slots, launch fields are zero and entropy components map to slots. Saved sample whole/split/permuted replay covers dense, capacity, every STOP and mixed inactive cases. Fullgraph eager-backend and finite-difference gradient checks pass. |
| Trunk overflow | `trunk_gemm_width` equals the maximum actual Linear input/output width, verified by enumeration over varied configurations. Packed dispatch chunks at row boundaries with strict bounds; tests check distinct inputs, order, real packing/unpacking and unchunked equality. Only an individually unfittable row raises. |
| Head overflow | Chunking begins before actor projection and enforces `rows × 252 × max(3D,D,widest head) < 2^31`. Preset limit is 11,096 rows. Boundary ±1 and dispatch/equality tests cover log-probabilities, entropies, tokens, flags and an invalid later chunk. Production compilation stays trunk-only. |
| Workload handoff | Preset trunk bound is 5,915 rows. Documentation gives the rollout/minibatch/teacher arithmetic; the Task 3.4 startup assertion is recorded in the plan. |
| Topology/init/Muon | Isaiah's shared classes remain. Actor projection is 3D→D; nine heads are `OutputProjectionMLP`; prefix/position embeddings remain under `actor`. Output singular values, zero biases, input/hidden initialization and actual optimizer membership are checked. Projection/head `.up` weights are Muon; output heads, embeddings, norms and biases are AdamW. |
| Statelessness/isolation/budget | No carried state or opponent identity input is introduced. Seat perturbation changes that seat while leaving the other exactly unchanged; intervening unrelated calls do not change repeated results. Preset has 6,252,223 parameters, retaining Isaiah's width/head/ratio settings with depth eight. |

## Executed checks

CPU environment and source identities are recorded in `environment.json`. PyTorch 2.9.0; `OMP_NUM_THREADS=2` for pytest.

| Command | Result |
| --- | --- |
| `uv run pytest tests/kaggriculture -q` | **256 passed, 1 skipped**, 12.93 s |
| `uv run pytest tests/owl tests/scripts tests/tools -m 'not slow' -q` | **723 passed, 3 skipped**, 4.76 s |
| `uv run mypy python/owl scripts` | **52 source files clean** |
| After restoration: `uv run pytest tests -m 'not slow' -q` | **979 passed, 4 skipped**, 17.85 s |

`just py-prepare` was attempted and failed with exit 127 because `just` is unavailable. Its checks were run directly: Ruff import-order check, Ruff format in check mode, Python 3.11 syntax, Ruff lint, mypy over `python scripts` (53 files), the full restored-source non-slow suite and docs freshness. All passed. Formatting was checked without modifying files. Docs freshness reports no working-tree code changes; this is not an independent proof of all documentation claims.

Ancillary diff check: code/tests/docs/cookbook/configs pass `git diff --check 0e989a1..HEAD`. The unrestricted check reports only trailing whitespace in the already committed previous-round raw pytest mutation logs under `ops/rebuild-2026-09-29/codex/verify-2.3-heads/`; it does not pass globally. These are verbatim diagnostic artifacts, with no functional impact or requested edit. Full output is in `diff-check.log`.

## New mutation probes

Each mutation ran alone, with original source bytes restored in `finally` before the next. All failed through the expected targeted tests, not test collection.

| Mutation | Observed result |
| --- | --- |
| Replace `_safe_choice` clamped temporary index with the raw value | **5 failed, 2 passed**; malformed replay triggers unsafe indexing rather than `GrammarReplayError`. |
| Use inclusive raw-HIRE prefix counts | **3 failed, 2 passed**; budgets 1/2/3 disagree with enumerated/replayed density. |
| Remove `−1` from the head row-limit numerator, admitting equality with 2^31 | **3 failed**; the strict-bound and single-row rejection tests detect it. |

`mutations.json` records original/restored SHA-256 and exit codes. `run_mutations.py` is the reproducible probe driver; mutation logs and exact original bytes are adjacent. All **305 tracked files** match the pre-verification SHA-256 manifest; `git status --short` shows only the pre-existing untracked backup and this new untracked evidence directory. No tracked modification or staged change remains.

## Limits

- Native table extraction, recorded-reference-program decoding and replay integration await Task 1.2/1.4, as allowed by the approved brief. The present skip is table equality; recorded-program integration must still be added at that handoff.
- CPU `backend="eager", fullgraph=True` proves graph capture/density/gradient agreement, not production Inductor/CUDA/BF16 correctness or throughput. Phase 6 qualification and Task 3.6's first-minibatch alarm remain separate work.
- Market-position noise independence in `policy_core` is established by source inspection. The single-slot frequency test and coupling-helper enumeration do not independently test future accidental noise broadcasting at that call site. Current code uses independent noise correctly; no current defect is inferred.
- The remaining three skips are two unavailable FlashAttention CUDA tests and one unavailable quantized-backend test.

VERDICT: APPROVE
