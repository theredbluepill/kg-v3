# Independent verification — Task 3.1-rest

Branch: `kg/rebuild-trainer-model`. HEAD: `b1da613a197a334547563111c1e11453f84551b8`.
Reviewed exactly `git diff e1458d2a717d9d731a367cbb78b98616ee6649f4...HEAD`;
the merge base is `e1458d2a717d9d731a367cbb78b98616ee6649f4`.

Scope: model-side registration, compile dispatch, and masked critic against rebuild
plan Task 3.1, `briefs/2.2-critic.md`, accepted contract v4, and Isaiah at `32b3ec9`.
This does not approve completion of the whole game seam: Kaggriculture environment
configuration, rollout storage and trainer integration remain open, as recorded.

## Findings

1. **P3 / low — `cookbook/references/kaggriculture-model-joins-isaiahs-factory-compile-and-masked-critic.md:53`.**
   “4 hardware skips” misclassifies the native grammar binding skip. The retained
   receipt and the independent rerun both show one unavailable Task 1.2/1.4 binding,
   two unavailable FlashAttention/CUDA cases, and one unavailable quantized backend.
   **Fix:** write “4 skips (3 hardware/backend, 1 unavailable native grammar binding)”
   or simply “4 skipped.”
2. **P3 / low — `cookbook/references/kaggriculture-model-joins-isaiahs-factory-compile-and-masked-critic.md:19`.**
   The sentence implies Isaiah also matches on config classes. His factory at
   `32b3ec9` actually matches `config.model_arch` strings. **Fix:** state that this
   change preserves Isaiah’s exhaustive dispatch and `assert_never`, while switching
   to config-class matching to narrow the game-specific types.

No functional defect found. Findings were not applied because this verification
must leave tracked files unchanged.

## Supported conclusions

- Registration follows Isaiah’s discriminated `ModelConfig` union and direct
  factory construction. Both game/spec mismatch directions fail explicitly. The
  nominal compile interface and factory validation helpers are not compatibility
  shim wrappers. Orbit constructors retain their arguments and behavior.
- `FullConfig` explicitly rejects Kaggriculture while its environment remains Orbit,
  before accessing Orbit-only model fields. The limitation is accurately recorded.
- The trunk target reaches exactly `_forward_transformer_trunk`, with `dynamic=True`;
  only the blocks and final norm are in that callable. `_run_trunk` retains guard,
  chunking and packed dispatch outside compilation. Stems, token assembly, packing,
  actor projection, action heads and critic remain eager. The MLP target compiles
  only block MLP modules. Recording tests inspect actual module execution during
  sampling, replay and value calls; the AST check additionally pins compile sites.
- The critic uses the critic-token mask, which expands each seat row’s
  `still_playing` flag over self/opponent. Its masked fill plus log-softmax matches
  Isaiah’s primitive and preserves live-row probabilities. Inactive rows return
  uniform probabilities/value zero; Isaiah’s full `_critic_logits` instead rejects
  an all-inactive row. That deviation is explicit in the records, and contract v4
  requires `still_playing` always true on the native path.
- Critic tests inject distinct nonzero hidden states into inactive rows, avoiding
  the accidental uniform result from zeroed encoder states. Hand-computed, token
  order, initialization, optimizer membership and seat-isolation tests also pass.
- The cookbook has a material adaptation inventory, index and log entry and does
  not claim completed trainer integration or real CUDA compilation. The two
  wording findings above should be corrected.

## Independent checks

- `uv run pytest tests/kaggriculture tests/owl tests/scripts tests/tools -m 'not slow' -q`:
  **1,324 passed, 4 skipped**, 28.54 s (`pytest.log`).
  The included `tests/owl tests/scripts tests/tools` selection is **1,048 passed,
  3 skipped**: independently collected 1,051 cases (`orbit-collection.log`), all
  included in the successful combined run. Kaggriculture contributes 276 passed,
  1 skipped.
- `uv run mypy python/owl scripts`: **success, 57 source files** (`mypy.log`).
- After restoring mutations, `uvx --offline --from rust-just just py-prepare`:
  **passes**, including format/lint/syntax/mypy and **1,324 passed, 4 skipped**,
  35.69 s (`py-prepare.log`). The docs-fresh invocation sees a clean working tree;
  branch-level documentation coverage was checked against the three-dot diff.
- Supplemental `cargo test --offline`: **148 passed, 7 failed, 2 ignored**
  (`cargo-test.log`). Every failure reports absent generation/replay parity fixtures
  under `tests/fixtures`; this is an environment verification gap, not a demonstrated
  regression. The reviewed diff changes no `src`, `engine_rs`, Cargo manifest/lock,
  or fixture-generation script. Fixtures were not regenerated or bypassed.
- Supplemental `git diff --check` reports trailing whitespace in retained raw
  pytest output at `ops/rebuild-2026-09-29/trainer-model/critic-red.log:3`.

## Mutation evidence and custody

Both mutations were applied individually to `python/owl/model/kaggriculture.py`,
run against the named regression, and restored in `finally` with exact-byte checks:

1. Return `F.log_softmax(logits)` instead of masked logits:
   `test_winner_softmax_is_isaiahs_masked_softmax` **fails** at its independent
   probability comparison (4/12 entries differ, max absolute difference 0.4936003).
2. Call the trunk before the padded guard/chunking:
   `test_trunk_guard_stays_in_front_of_the_compiled_callable` **fails**, recording
   `[4]` instead of four single-row calls.

Each source restoration has SHA-256
`f91cd61d32327374ceb11ebd3921a771415e510df9ee11c2548ace771cb6bb08`.
See `mutations.json` and `mutation-*.log`. The post-restoration full preparation
run is green. All **766 tracked files** have unchanged SHA-256 hashes, and tracked
Git status is empty (`tracked-before.json`, `tracked-after-check.json`). Only this
untracked verification evidence directory was added; no tracked adaptation or
cookbook edits were made.

All checks were local CPU checks. Trunk compilation used recording stand-ins;
real Inductor/CUDA/BF16/FlashAttention execution and performance are unverified.

VERDICT: APPROVE WITH EDITS
