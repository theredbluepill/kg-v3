# Independent verification of Task 3.1-rest, round 3

Branch: `kg/rebuild-trainer-model`.
Reviewed exactly `git diff e1458d2a717d9d731a367cbb78b98616ee6649f4...HEAD`, with HEAD `4cac1a18f2209c54d40bef80d44755334a1a1ed2`. The requested base is also the merge base. Entry tracked state was clean.

Findings: **none**. No severity/file/line/fix entries remain. Approval is scoped to this model-side change, not completion of the full Task 3.1 trainer integration.

## Prior findings

Read the requested external report at `/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/codex/verify-3.1-rest-r2.md` directly.

Its sole P3 finding is resolved:

- `cookbook/references/compiled-gemm-template-overflows-above-2-21-rows.md:51` now states that Kaggriculture is registered in the shared union/factory and the trainer compile helper dispatches to its guarded blocks-plus-final-norm region.
- The same paragraph links the Task 3.1 model-side Reference, retains the CPU recording-only evidence, and explicitly leaves FullConfig integration, integrated workloads and real Inductor/CUDA qualification open.
- The description, references index and newest cookbook log entry are reconciled with that correction.
- Both previously closed r1 wording findings remain fixed: config-class matching is distinguished from Isaiah's architecture-string matching, and skips are correctly split into three hardware/backend cases and one missing native grammar binding.

## Contract and code review

Compared the diff with `ops/rebuild-2026-09-29/plan.md` Task 3.1 and principles I0/I0b/I5/I8/I11, `ops/rebuild-2026-09-29/briefs/2.2-critic.md`, accepted contract v4 in `docs/kaggriculture-contract.md`, mapped model documentation, relevant cookbook decisions/References, and Isaiah's source at git object `32b3ec9`.

- Registration directly extends the shared ModelConfig and create_model implementation. Exhaustive class matching ends in assert_never; mismatched game specs fail explicitly. Models are returned directly. No compatibility shim or second trainer is added.
- TrunkCompileAPI is a nominal capability interface, not a forwarding wrapper. The existing compile helper dispatches through it. Orbit's recurrent rejection and model-level cross-attention/player-adapter restrictions remain intact. The implementation of Isaiah's stateless model changes only its import/base declaration.
- Kaggriculture compiles only `_forward_transformer_trunk`, consisting of blocks and final norm. `_run_trunk` remains before that callable, with guard/chunking. Stems, actor projection/heads and critic remain outside. MLP compilation retains Isaiah's existing selection logic.
- Critic masking uses dtype-minimum fill followed by log_softmax, matching Isaiah's masked winner-distribution formulation. Shared OutputProjectionMLP, per-token logits, self/opponent order, independent seat observations, `2*p(self)-1`, initialization and optimizer grouping remain covered by tests.
- The inactive-row uniform result is a documented deviation from Isaiah's preliminary no-live-player rejection. Contract v4 says still_playing is always true, so this synthetic edge case is outside the contracted environment output. Its reopening condition is explicitly recorded.
- FullConfig intentionally rejects Kaggriculture while its environment field is Orbit-only. Native environment integration, rollout storage and Kaggriculture observation mapping remain Task 3.1 work. The code and records consistently state this boundary.
- Cookbook adaptation inventory covers the changed implementation/tests/docs. Current registration and verification claims agree with source and receipts. Historical log entries were treated as historical evidence.

Two parallel read-only subagent reviews independently examined architecture/contracts and cookbook/prior-finding closure. They returned no actionable findings. Root performed source inspection, all execution and custody checks.

## Executed checks

Host: macOS 26.4 arm64; Python 3.12.13; torch 2.9.0; CUDA unavailable.

| Check | Result | Evidence |
|---|---|---|
| `uv run pytest tests/kaggriculture tests/owl tests/scripts tests/tools -m 'not slow' -q` | **1,324 passed, 4 skipped**, 25.18 s | `pytest.log` |
| Orbit/script/tool subset included in that successful run | **1,048 passed, 3 skipped**; independent collection identifies 1,051 cases | `orbit-collection.log`, `pytest.log` |
| `uv run mypy python/owl scripts` | **57 source files clean** | `mypy.log` |
| Post-restoration `uvx --from rust-just just py-prepare` | **Passed**; Ruff, syntax check, mypy, **1,324 passed, 4 skipped**, docs freshness | `py-prepare.log` |
| Additional `cargo test`, using existing local fixtures | **155 passed, 2 ignored** | `cargo-test-with-local-fixtures.log`, `cargo-fixture-inputs.json` |
| Three-dot diff whitespace check over code/tests/docs/cookbook/README | **Passed** | Executed `git diff --check ... -- python tests docs cookbook README.md` |

Pytest skips: one missing Kaggriculture native grammar binding, two unavailable FlashAttention/CUDA tests, one unavailable quantized backend.

The initial plain `cargo test` had **148 passed, 7 failed, 2 ignored**, with all seven failures explicitly reporting absent generated Orbit fixtures (`cargo-test.log`). Existing sibling-worktree fixtures were then reused: a temporary generation fixture copy and the read-only replay directory selected via ORBIT_WARS_PARITY_FIXTURE_DIR. Every fixture's path, size and SHA-256 was recorded. No fixture was downloaded or regenerated, no parity requirement was disabled, the rerun passed, and the temporary copy was removed. This proves parity against those existing fixtures, not fresh fixture generation against an external package.

## Non-vacuous tests and restoration

Two actual source mutations were applied sequentially to `python/owl/model/kaggriculture.py`, with byte restoration in finally blocks after each run:

1. Use raw logits instead of masked logits in the critic log_softmax. `test_winner_softmax_is_isaiahs_masked_softmax` failed at its probability assertion, with 4/12 elements mismatched and maximum absolute difference 0.493600279. The test injects distinct inactive logits so zeroed encoder tokens cannot conceal the missing mask.
2. Return the trunk before guard/chunking. `test_trunk_guard_stays_in_front_of_the_compiled_callable` failed because recorded calls were `[4]` instead of `[1, 1, 1, 1]`.

Both commands exited 1 from intended assertion failures, not import/collection errors. Logs and `mutations.json` retain the evidence. Original and restored SHA-256 for each mutation: `f91cd61d32327374ceb11ebd3921a771415e510df9ee11c2548ace771cb6bb08`.

After restoration, py-prepare passed. All **796 tracked files** match their entry SHA-256 values; unstaged and staged git diffs are empty. Only this untracked verification-evidence directory remains. See `tracked-before.json`, `tracked-after-check.json` and `identity.json`.

## Limits

No real Inductor/CUDA compilation, FlashAttention execution, integrated Kaggriculture PPO/BC/teacher workload, training, or performance qualification was attempted. Compile-boundary checks use recording stand-ins and tiny CPU tensors. Full trainer integration remains explicitly open.

VERDICT: APPROVE
