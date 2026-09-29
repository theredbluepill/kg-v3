# Independent Task 1.3 verification — 2026-09-29

Scope: `git diff kg/isaiah-gap-closure...HEAD`, against the Task 1.3 brief,
accepted contract (currently v4.1, retaining v4 observation schema 3), and plan
Task 1.3. This is an independent review receipt, not an implementation change.

- HEAD: `7b5eacd0d393ef65f6bec0998184ed3ae160d378`, branch `kg/rebuild-observe`.
- Base and merge base: `f464c3db45449ce362fde25c5d9b37f88ad20e5e`.
- Reference: `65f0eac5bb00b18a9d3acce319c2a231cbd5dff0`.
- Tracked tree was clean at entry and after verification. No fixes, commits,
  ref changes, or tracked receipt changes were made. New `codex-verify-*`
  artifacts in this directory are untracked.

## Findings

1. **P2 — `tests/kaggriculture/test_observe.py:285`: the optional pinned-memory
   probe can crash the mandatory Python suite.** A fresh file-level invocation
   passed its first two ordinary cases, then exited 139 at
   `torch.empty(1, pin_memory=True).fill_(0)`. The isolated `[True-1]` case
   reproduced that native segmentation fault. Catching `RuntimeError` cannot
   handle it. This occurs before the encoder call, so it is not evidence of an
   encoder memory-safety defect. Fix: skip pinned cases before allocation when
   CUDA is unavailable, matching `python/owl/rl.py:329`; retain the actual
   pinned writer/pointer assertions on supported CUDA. If broader backend
   support is intended, isolate the capability probe in a bounded subprocess.
   The present full Kaggriculture suite cannot be reported green on this host.
   Receipts: `codex-verify-shard-test_observe.*`,
   `codex-verify-pinned-repro.*`.

2. **P2 — `Cargo.toml:15`: the required grammar bridge migration was omitted.**
   This is the first production root-to-engine dependency. Contract
   `docs/kaggriculture-contract.md:296` and plan
   `ops/rebuild-2026-09-29/plan.md:194` require moving the kernel acceptance and
   replay-state tests into root integration at this edge. The standalone
   `engine_rs/tests/grammar_kernel.rs` and authored registrations remain.
   The explicit deferral in the cookbook Reference at lines 187–190 is accurate
   disclosure, but does not amend the governing requirement. Fix: move the
   coverage, remove the bridge and its allowlist/manifest registration,
   regenerate the manifest, and rerun the root/engine/trim checks.

3. **P3 — `docs/rl-api-specs.md:1026`: stale separate-feature-graph claim.**
   The root now shares the engine's Serde features, as documented correctly in
   `docs/rules-engine.md:226`. Fix this paragraph to describe the current graph
   and actual migration state.

4. **P3 — `docs/rules-parity-coverage.md:493`: current results link points to the
   superseded incomplete handoff.** `1.3/results.md` records the pre-merge R1
   and missing-schema failures; `1.3/claude-review.md` records the subsequent
   qualification. Fix the current-results link and label the earlier receipt
   historical. Preserve both receipts.

## Semantics, oracle, and custody

Read-only parallel reviews checked production encoder formulas and admission
against the contract and reference/kernel sources, and separately checked
oracle construction and documentation. No additional encoder semantic defect
was found. The reviewed surfaces include tile applicability and sentinels,
coordinate order, all actors, exact counts and insertion ranks, both-seat
private noninterference, hire costs, public configuration, shops, market
channels, masks, one-snapshot preparation, and all-environment preflight before
Python output publication.

The recorder calls the actual pinned `myolie_features::encode_invest`; expected
values are independent of tensor reconstruction. Export custody, strict
records, quotas, source identities, per-seat hashes, stream lengths and EOF are
checked. The current 512-state/1,024-perspective comparison passed at every
8,176-offset row. This verification reran custody and comparison, not full
reference regeneration. The cookbook's regeneration claim is supported by its
historical receipt and source audit, not a fresh regeneration in this review.

Fresh non-vacuity control: swap inventory and price in production
`write_shops_and_market`, leaving validation and reconstruction unchanged.
The oracle failed with exit 101 at
`record=official:95324500:0 seat=0 offset=889 field=market blocks:
actual=0.0025 recorded=1.0`. The source was restored in a `finally` block and
verified byte-for-byte. Both pre/post SHA-256 values are
`0d84eb9df01f699767fc674996e5eae14be31c09f70b058b5af35fce36365749`.
The full restored root suite then passed again. Receipts:
`codex-verify-mutation.json`, `codex-verify-mutant.*`,
`codex-verify-restored-root.*`.

There is no three-dot change under `engine_rs/` or to
`scripts/check_engine_trim.py`. The trim checker passes against its pinned
reference. Production Orbit rules, model, trainer, and PPO code are untouched
by this diff; the Orbit change is the reviewed test-only Number decoder.

## Fresh execution results

| Check | Result |
|---|---|
| `cargo test --manifest-path engine_rs/Cargo.toml --locked` | 87 passed: 41 unit, 18 grammar bridge, 9 RNG, 19 replay; no ignored tests |
| `uv run python scripts/check_engine_trim.py` | `engine trim manifest: OK`, also rerun after restoration |
| `cargo test --locked`, with `REQUIRE_PARITY_FIXTURES=1` | 244 passed, 4 ignored; repeated green after restoration |
| Root breakdown | 87 Kaggriculture, 95 RL, 62 Orbit rules tests passed |
| Fresh `uv run maturin develop --locked` | Passed; Python tests used the current native build |
| Kaggriculture Python, unchanged inputs in separate file processes | 311 passed, 1 existing missing-grammar-binding skip; 2 pinned cases excluded from the completed run after reproducing the probe crash |
| Observation plus custody without the 2 pinned cases | 100 passed, 2 deselected = 55 observation + 45 custody |
| All 22 `tests/owl/test_*.py` files, recursively, one process per file | 787 passed, 3 platform skips (2 CUDA attention, 1 x86 quantization) |
| `scripts/check_doc_freshness.py` | `No doc updates required`; this mechanical check does not resolve the prose findings above |
| Three-dot whitespace check excluding `ops/` | Passed; full check reports whitespace in frozen historical command logs |

The four ignored root tests are the two inherited expensive Orbit angle audits,
explicit corpus generation, and optimized timing. None is silently counted as
passed. Orbit parity fixtures were required, not allowed to disappear silently.

Execution used offline dependencies, two Cargo jobs, one Rust test thread,
two Rayon threads and one OMP/MKL thread. The original parallel engine command
hit the 1 GB guard; the equivalent serial run completed in 29.26 s with sampled
peak 418,578,432 bytes. The combined `uv run pytest tests/kaggriculture -q`
likewise hit the 1 GB guard; splitting by file preserved test inputs. Separate
observation execution then exposed the independently reproduced native probe
crash. The whole Kaggriculture command is therefore **not green**.

The sandbox disallowed uv's default writable cache; a temporary cache was used.
After explicitly rebuilding with installed maturin, `UV_NO_SYNC=true` reused the
existing locked environment because the fresh offline cache lacked a maturin
build-isolation package. No dependency or lockfile change was made.

Host: macOS 26.4 arm64, CPython 3.12.13, torch 2.9.0, no CUDA build. Existing
optimized timing remains a documented memory-limited pod handoff; it was not
reattempted. No performance qualification, CUDA/pinned qualification, or
Task 1.4 lifecycle/transition-safety claim is made.

The implementation semantics and retained Orbit tests support approval after
the listed edits. The pinned-test failure and triggered bridge migration must
be resolved before treating Task 1.3 as fully complete.

VERDICT: APPROVE WITH EDITS
