# Independent verification — Task 2.3 grammar action heads

Date: 2026-09-29. Reviewer: Codex. Branch: `kg/rebuild-heads`.

Reviewed `0e989a123343df11eb3782907188e5df77bdd36e..5ec3af1e7b2f13f1884079bfaa990b016812c47b` against the approved Task 2.3 brief v3, its three prior review receipts, contract v4, and the requested reference oracles at `65f0eac5bb00b18a9d3acce319c2a231cbd5dff0` (`kg/reference-2026-09-29`). This is a CPU implementation verification, not native integration or GPU qualification.

No functional blocker was found. One low-severity documentation correction remains.

## Finding

**Low — qualify the no-host-sync claim.** `python/owl/model/kaggriculture.py:525` says `forward` has “no host synchronization”; `docs/model-architecture.md:744` repeats this, and `cookbook/references/kaggriculture-grammar-heads-sit-behind-isaiahs-actor-projection.md` says `forward` never syncs. The new oversized packed-trunk path explicitly transfers row counts with `token_mask.sum(dim=1).tolist()` at `python/owl/model/kaggriculture.py:484`. Retained packing also uses host-visible shape information. Therefore the claim is true of policy validation, not of the complete forward call.

**Fix:** say “sampling adds no policy-validation host synchronization,” and distinguish replay's single compact admission transfer from packing/chunk-planning synchronization. No sampler, replay or overflow-guard change is requested. This is a wording correction, not evidence of a measured performance regression.

## Contract review

| Area | Result and evidence |
| --- | --- |
| Mask tables | All eight table shapes and supports match the pinned Rust `State::allows`/`advance`, the reference extractor and brief §2. This includes unreachable-prefix zero placeholders, unit kinds 1–18, item subsets, omitted unit quantity, rejection of explicit unit zero, base-32 boundaries and acceptance of market zero. Native extraction is intentionally pending. |
| Runtime context | Own actor-mask counts, per-row order limits and liveness drive unit support, market placement and the forced sentinel. Capacity counts submitted HIREs, independently of cash/execution. Inactive tokens, lengths and densities are zero. |
| Coupled Gumbel | `gumbel_perturb` creates independent noise at every score element. The actual policy passes one perturbed score tensor into `couple_market_kinds`; its exclusive raw-HIRE prefix masks HIRE and reuses those scores. Replay density uses the exclusive final-HIRE prefix. Enumeration covers budgets 0/1/2/3/10 and both NONE and EMPTY correction outcomes, with total probability and individual market-kind program density checked to 1e-9. |
| STOP | The first final NONE retains slot-7 density; the forced sentinel has zero density. Later positions have zero log-probability/entropy, and lengths include STOP. EMPTY consumes a position. |
| Replay admission | Shape/dtype admission precedes encoding. Every replay-derived indexing value is clamped before table/gather/embedding use. `market_frame` is observation-derived and bounded by the 241-wide own actor mask. Support, length and full elementwise canonical equality remain separate device flags. The canonical check covers ordinals, reserved targets, cross-kind fields, STOP, padding and inactive rows. Invalid supplied values cannot be silently repaired and returned. |
| Replay synchronization | `check_replay_flags` reduces flags/first failing rows to one compact tensor and transfers it once outside `policy_core`, before returning to loss computation. Sampling does not invoke that check. The internal result owns `valid`; `ModelOutput` does not. This does not establish a globally synchronization-free encoder/forward. |
| Outputs and gradients | Event tensors are `[E,2,252,12]`; slots 0/2/11 have zero density; per-player/entity fields sum slots; launch fields are zero; entropy components match slots. Critic values share the encode. Saved-sample whole/split/permuted replay, fullgraph eager-backend density/gradient comparison and nonzero finite-difference gradient checks pass. |
| Trunk overflow | Width equals the maximum actual Linear input/output width for supported block configurations, checked by enumeration. Both padded and packed dispatch enforce the strict element bound. Packed tests verify distinct chunk inputs/masks, contiguous ordered coverage, strict bounds and unchunked equality; only an individually oversized row is rejected. |
| Head overflow | Chunks begin before actor projection and enforce `rows × 252 × max(3D,D,widest sampled head) < 2^31`. Tests cover numerical and dispatch boundaries ±1, log-probability/entropy/flag equality, deterministic tokens and a malformed row in a later chunk. Production compilation remains trunk-only. |
| Workload arithmetic | Independent arithmetic agrees with 5,915 trunk rows and 11,096 head rows at the preset. Documented rollout/minibatch headroom is correct. The Task 3.4 startup workload assertion handoff already exists in the base plan. |
| Topology/init/Muon | Isaiah's layer classes are retained. Actor projection is 3D→D; heads are nine `OutputProjectionMLP`s; embeddings exist only under the actor. Initialization tests check output singular values 0.01, zero biases, hidden/projection gains and embedding standard deviations. Optimizer membership checks include the projection and head `.up` in Muon, and head `.out`, embeddings, norms and biases in AdamW. |
| Statelessness/isolation/budget | No carried state or opponent-identity input was introduced. Seat perturbation changes that seat's density while leaving the other bitwise unchanged; intervening unrelated calls leave repeated results unchanged. Preset count is 6,252,223; Isaiah's width, head count and ratios are retained, with depth 8 rather than the 6m ladder's depth 6. |

## Executed checks

Environment: macOS arm64, Python 3.12.13, PyTorch 2.9.0, no CUDA. Exact source/environment identities are in `environment.json`.

| Command | Result |
| --- | --- |
| `uv run pytest tests/kaggriculture -q` | **256 passed, 1 skipped**, 15.03 s |
| `uv run pytest tests/owl tests/scripts tests/tools -m 'not slow' -q` | **723 passed, 3 skipped**, 4.76 s |
| `uv run mypy python/owl scripts` | **Success, 52 source files** |
| Post-restoration `uv run pytest tests -m 'not slow' -q` | **979 passed, 4 skipped**, 17.15 s |
| `git diff --check 0e989a1..HEAD` | Passed |

`just py-prepare` could not execute because `just` is not installed (exit 127). Its checks were invoked directly: Ruff import-order check, Ruff format in check mode (preserving the read-only review), Python 3.11 syntax check, full Ruff lint, `uv run mypy python scripts` (53 source files), the full non-slow Python suite above, and docs freshness. All passed. The docs-freshness tool found no working-tree source change; this is not an independent validation of all committed documentation claims.

The Kaggriculture skip is the named native-table comparison awaiting Task 1.2/1.4. The other skips are two unavailable FlashAttention CUDA tests and one unavailable quantized backend test. Logs are adjacent to this report.

## Independent mutation checks

Each mutation was applied alone to `python/owl/model/kaggriculture_actor.py`; its original bytes were restored in a `finally` block before the next test. All three mutations produced assertion/test failures and exit status 1, rather than a test-collection failure.

| Mutation | Targeted result |
| --- | --- |
| Replace `_safe_choice`'s clamped index with the raw index | **5 failed, 2 passed**: malformed replay reaches unsafe indexing instead of the expected `GrammarReplayError`. |
| Exclude slot 0 from full canonical equality | **2 failed, 7 passed**: unit-frame and market-frame actor-ordinal corruption is no longer rejected. |
| Replace the exclusive raw-HIRE prefix with an inclusive count | **3 failed, 2 passed**: budgets 1/2/3 no longer match enumerated/replayed program density. |

Original/restored actor SHA-256 and outcomes are in `mutations.json`. All **286 tracked files** were compared against the pre-mutation SHA-256 manifest and matched; the full suite then passed after restoration. No tracked modification or index change is left. This report and its evidence directory are untracked review artifacts.

## Limits

- Native-table extraction and recorded-program decoding/replay integration are expressly deferred by the approved brief to Task 1.2/1.4. The current skipped test is table equality; recorded-program integration still needs to be added at that handoff.
- CPU `backend="eager", fullgraph=True` establishes graph compatibility and gradient agreement, not Inductor/CUDA/BF16 correctness, throughput or historical L6 qualification. Those remain Phase 6 work, with the Task 3.6 first-minibatch alarm still a separate handoff.
- Market-position noise independence is established by source inspection here. The empirical test exercises one categorical slot, and exhaustive HIRE enumeration directly exercises the coupling helper; neither independently detects a future change that broadcasts a shared market noise vector inside `policy_core`. Current implementation is correct; this is an additional coverage opportunity, not a required fix for this verdict.

VERDICT: APPROVE WITH EDITS
