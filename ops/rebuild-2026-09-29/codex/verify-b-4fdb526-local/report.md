# Independent verification of lane B at 4fdb526

Reviewed HEAD `4fdb5264e7a4494693b8eb97687faf73e70a9d8d` against the Task 2.1 and 2.2 briefs, accepted contract v4, and plan I0/I0b/L6. No tracked modifications remain. This untracked receipt and its logs are verification artifacts, not an implementation adaptation.

The requested endpoint range is `1177a5be3ec89c907e37573eee2732bc822d0c90..4fdb5264e7a4494693b8eb97687faf73e70a9d8d`: `kg/isaiah-gap-closure` has advanced with lanes C/D. Its common ancestor with lane B is `2d9f1cd34a1b23470497c5d255220adc70b6f6df`. The two-dot diff therefore includes C/D work absent from lane B. Those absences are branch divergence, not deletions made by B's commits; this review does not qualify a combined merge result.

## Findings

1. **P2 / should fix — tile count bounds remain incomplete.** `python/owl/kaggriculture/types.py:112` permits −1 for every `tiles_int` channel. Independent probes set `yield_units`, `consecutive_unwatered`, and `consecutive_unfed` to −1 separately; all passed `check_contract`. These channels are counts, not date/deadline sentinels. Require channels 0, 5, and 6 to be nonnegative, retaining −1 for channels 1–4, and add negative-count tests. This is a residual of prior finding 1's channel-specific bounds requirement.

2. **P2 / should fix — the fixture still violates a contract-derived feature.** `tests/kaggriculture/conftest.py:154` encodes next-hire cost as `100 * (actors + hires)`. Contract player channel 10 requires the actual public-rule cost: the pinned engine at `kg/reference-2026-09-29:engine_rs/src/lib.rs:3792` and `:3821` uses `farmHandCostMult * fib(hires_today)`, with Fibonacci starting 1, 1. The default fixture has hires `[4,1]` and multiplier 100: expected costs are `[500,100]`, while its encoded costs are approximately `[600,200]`. Use the pinned formula and independently assert the channel. Also, `make_obs(order_limit=0)` is accepted at `conftest.py:203` and writes configuration maximum 0 at `:263`, outside the contract's 1–10 envelope; reject zero in this fixture or distinguish a per-turn limit from the configured maximum. Prior finding 7's original zero-fill/privacy/reserved/mask defects are fixed, but the broader contract-valid claim is not yet true.

3. **P2 / should fix — critic tests accept constant outputs.** `tests/kaggriculture/test_model_encoder.py:710`, `:723`, and `:762` check normalization/self-consistency, swaps without an asymmetric premise, and only the unaffected seat. A mutation replacing `return logits.log_softmax(dim=-1)` with `return torch.zeros_like(logits).log_softmax(dim=-1)` passes all five critic-focused tests: **5 passed, 28 deselected**. Use controlled distinct critic-token inputs and deterministic head weights to assert the independently expected nonuniform probabilities, then assert the changed seat's value responds while the other seat remains unchanged. The implementation currently computes the correct formula; this finding concerns missing regression protection.

## Prior findings 1–7

| Prior finding | Recheck |
|---|---|
| 1: leading dims and bounds | Exact `[E,2]`, inventory/storage/rank/global/order bounds repaired. Tile count lower bounds remain incomplete as above. |
| 2: packed/flash tests | Resolved: successful packing/unpacking, token masks and `max_seqlen`, strict boundary, pre-trunk rejection, compiled dispatch, stable checkpoint keys, and SiLU coverage. These are CPU/mocked checks. |
| 3: vacuous chunk masks | Resolved: distinct row masks, exact input/mask slices and order, strict bound and numerical agreement. Reusing the first chunk's masks now fails both parameterizations. |
| 4: vacuous offsets | Resolved: independent region markers assert each named readout and full sequence, including `still_playing=False` masks. |
| 5: absent typing regression | Resolved: executed persistent mypy probe validates positive types and exactly 11 marked mixed-game errors. |
| 6: remaining acceptance assertions | Resolved: orthogonal gains, attention and MLP residuals, token statistics, norm initialization, all policy/serving/evaluation hidden-state paths, exact optimizer memberships, and per-field dtype/shape/missing-field checks. |
| 7: contract-valid fixture | Original mask/zero-fill/privacy/reserved defects resolved; incorrect hire-cost formula and zero configuration maximum remain. |

## Design verification

The encoder reuses Isaiah's actual `ObservationInputStem`, `TransformerBlock`, final LayerNorm, packing helpers and initialization. Widths, token order, named offsets and observation-only seat-row isolation match the brief. Padded dispatch chunks complete rows and masks below the strict L6 bound; packed overflow rejects before executing the trunk, preserving dispatch. Action heads/combined policy evaluation/teachers are explicitly deferred, so this qualifies the implemented I0/I0b scope only.

The critic at `python/owl/model/kaggriculture.py:176` and `:453` is a shared `OutputProjectionMLP(trunk,1)` over the two critic-value tokens. It applies float32 winner log-softmax, and `compute_value` returns `2*exp(logp[...,0])-1` with shape `[E,2]`. Output gain is Isaiah's 1.0 (`:197`); the output projection is excluded from Muon (`:213`), while its hidden projection uses Muon. The unmasked two-player softmax is equivalent for valid v4 observations: both players are active and `still_playing` is always true after auto-reset. No critic implementation defect found.

## Executed checks

| Command | Result |
|---|---|
| `uv run pytest tests/kaggriculture -q` | **155 passed**, 13.80s; exit 0 |
| `uv run pytest tests/owl tests/scripts tests/tools -m "not slow" -q` | **723 passed, 3 skipped**, 25.31s; exit 0 |
| `uv run mypy python/owl scripts` | **50 source files, no issues**; exit 0 |
| `uvx --from rust-just just py-prepare` after restoring mutations | **878 passed, 3 skipped**, 11.97s test phase; formatting/lint/static/docs checks pass; exit 0. Its wider mypy invocation checks 51 source files. |

The skips are two unavailable flash-attn CUDA cases and one unavailable x86 quantization backend. Real CUDA/FlashAttention execution, generated compiled kernels, GPU throughput and combined-lane integration were not qualified. Docs freshness checks the worktree against HEAD, not the committed review range.

Mutation probes were sequential and restored in `finally`, byte-for-byte. Constant-critic mutation: five tests passed (gap above). Reused-chunk-mask mutation: two tests failed at the exact mask-slice assertion. The restored model SHA-256 is `92f4df615656f761262ecc62a1219336e5b7904bcde3a4876ccd2cf4bc109dee`. Final `git diff --exit-code HEAD` and staged diff are clean. `.codex-tmp/` was already untracked on entry; these verification receipts are the only new untracked directory.

VERDICT: APPROVE WITH EDITS
