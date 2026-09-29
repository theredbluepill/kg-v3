# Independent verification — Task 3.1-rest, round 2

Branch: `kg/rebuild-trainer-model`. HEAD: `aadba6da1c1c996db502bd1bc6b570d31e1ae958`.
Reviewed `git diff e1458d2a717d9d731a367cbb78b98616ee6649f4...HEAD` (three-dot).
The merge base is exactly `e1458d2a717d9d731a367cbb78b98616ee6649f4`.

Scope: model-side registration, trainer compile dispatch and critic masking against
plan Task 3.1, `ops/rebuild-2026-09-29/briefs/2.2-critic.md`, accepted contract v4,
and Isaiah's source at `32b3ec900ad406eedd965f53a1a0f4490d31c589`.
This does not certify completion of the whole trainer game seam.

## Finding

**P3 / low — `cookbook/references/compiled-gemm-template-overflows-above-2-21-rows.md:51`.**
The current Consequences section still says Kaggriculture is absent from
`ModelConfig`/the factory and that `configure_model_compile` rejects its trunk
target. This diff implements all three, so those current-state claims are stale.
The surrounding measured findings remain pinned to their historical versions.

**Fix:** replace the obsolete claims with a link to the Task 3.1 model-side
Reference and state that registration and guarded compile dispatch now exist.
Retain the accurate limits: full trainer integration and integrated rollout/PPO/
evaluation/BC/teacher workloads and real CUDA compilation remain unverified.
Reconcile the note/index/log together under the cookbook contract. No edit was
applied during verification.

No functional defect or compatibility shim wrapper was found.

## Round-one findings

Both findings in `/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/codex/verify-3.1-rest-r1.md`
are resolved:

- Reference line 19 explicitly distinguishes config-class matching from Isaiah's
  `config.model_arch` string matching, retaining exhaustive match/assert_never.
- Reference line 53 correctly separates three hardware/backend skips from the
  one unavailable native grammar binding. Fresh execution confirms those reasons.

The new Reference, its index/log entries, and updated encoder/heads notes accurately
scope the adaptation. The older GEMM Reference is the remaining exception above.

## Supported code conclusions

- The shared discriminated union and factory construct the correct model directly;
  both game/spec mismatch directions fail explicitly. Orbit constructors preserve
  their arguments and behavior. FullConfig rejects Kaggriculture before accessing
  Orbit-only model fields while its environment config remains Orbit-specific.
- The nominal TrunkCompileAPI replaces concrete-model admission. The actual
  Kaggriculture compiled target is only blocks plus final norm; guards, chunking,
  packing, stems, actor and critic remain outside it. MLP dispatch preserves
  Isaiah's module selection. No second trainer or observation compatibility shim
  was introduced.
- Critic masking follows Isaiah's masked-fill/log-softmax primitive, consumes the
  actual critic token mask, and computes self value as 2*p(self)-1. Existing tests
  cover shared-head topology, shape/range, hand-computed scores, token order,
  initialization, optimizer membership, hidden-state rejection and seat isolation.
- An all-inactive row yields uniform probabilities/value zero instead of Isaiah's
  full critic raising. This is explicitly documented and outside contract v4's
  native batches, whose still_playing is always true (contract line 166).
- The compile tests execute modules inside a recording compile boundary, rather
  than only checking method existence. The critic test injects distinct nonzero
  hidden states into finished rows so the old unmasked softmax cannot pass through
  encoder zeroing. The mutation results below independently demonstrate sensitivity.

## Fresh checks

- `uv run pytest tests/kaggriculture tests/owl tests/scripts tests/tools -m 'not slow' -q`:
  **1,324 passed, 4 skipped**, 26.47 s (`pytest.log`).
- The included `tests/owl tests/scripts tests/tools` selection contributes
  **1,048 passed, 3 skipped**. A separate collection found all **1,051 cases**
  (`orbit-collection.log`); all ran in the successful combined suite. Kaggriculture
  contributes 276 passed and one skipped case.
- `uv run mypy python/owl scripts`: **success, 57 source files** (`mypy.log`).
- After mutation restoration, `uvx --offline --from rust-just just py-prepare`:
  **passes**, including Ruff, Python syntax, mypy and **1,324 passed, 4 skipped**,
  25.11 s (`py-prepare.log`). Docs freshness sees no tracked working-tree changes;
  branch documentation was reviewed separately using the three-dot diff.
- No Rust test was rerun. This diff changes no Rust/Cargo files. Round one's missing
  parity-fixture limitation is not claimed resolved by these Python checks.

## Mutation checks and custody

Diagnostic: demonstrate that reverting masking or bypassing the guarded trunk
changes observable behavior caught by the focused regression. Each mutation was
applied separately and the file restored in a finally block; the stopping condition
was the intended test assertion failing, followed by restored full preparation.

1. Replace `F.log_softmax(masked_logits, dim=-1)` with unmasked logits:
   `test_winner_softmax_is_isaiahs_masked_softmax` fails its independent probability
   assertion, 4/12 elements different, max absolute difference 0.493600279.
2. Return the trunk result before padded chunk/guard admission:
   `test_trunk_guard_stays_in_front_of_the_compiled_callable` fails because the
   recorded calls are `[4]` instead of `[1, 1, 1, 1]`.

Both source restorations are byte-exact; SHA-256:
`f91cd61d32327374ceb11ebd3921a771415e510df9ee11c2548ace771cb6bb08`.
See `mutations.json` and `mutation-*.log`.

All **781 tracked files** retain their original content hashes, modes and symlink
targets; tracked Git status is empty (`tracked-before.json`, `tracked-after-check.json`).
Only this untracked verification evidence directory was added.

All execution was local CPU verification. Compile-boundary tests use recording
stand-ins. Real Inductor/CUDA/BF16/FlashAttention execution and throughput remain
unverified; full trainer integration is explicitly still open.

VERDICT: APPROVE WITH EDITS
