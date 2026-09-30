# Independent Task 1.3 verification — 2026-09-29

Reviewed HEAD `8247ee28e2bdd04bf30e535005619e4585478b12` on
`kg/rebuild-observe`, using `git diff kg/isaiah-gap-closure...HEAD`.
The base ref was `f4ecd90ec15cf09adcd20570aedf636324ffac30`; the three-dot
merge base was `f464c3db45449ce362fde25c5d9b37f88ad20e5e`.
Requirements: `ops/rebuild-2026-09-29/briefs/1.3.md`, contract v4 with the
accepted v4.1 grammar-bridge clarification, and plan Task 1.3. Two independent
read-only subreviews covered production semantics and oracle/docs custody.

## Finding

- **P3 — stale missing-corpus diagnosis**, `scripts/kaggriculture_observation_oracle/regenerate.py:628`.
  The direct `--check --output <missing-directory>` command reports
  `qualified observation corpus missing; R1 coverage dependency unresolved`.
  Missing files do not establish a quota failure; the qualified v2 corpus now
  exceeds the unchanged actor quota. The Rust diagnostic was corrected, but
  the Python entry point retains the obsolete attribution. Reproduction is in
  `missing-corpus.log`. **Fix:** name the missing manifest and the regeneration
  command, and add a Python missing-manifest regression. Also retire the same
  obsolete explanation in the ignored generator reason at
  `src/kaggriculture/oracle_corpus.rs:1394`. Reconcile the overbroad diagnostic
  retirement claim at
  `cookbook/references/structured-observations-preserve-legal-state-and-order.md:163`.
  No encoder semantic defect or blocking finding was identified.

## Executed checks

Commands used the brief's CPU/offline bounds: two Cargo jobs, one Rust test
thread, two Rayon threads, OMP/MKL one thread, 120 seconds and sampled process
group RSS below 1 GB per diagnostic. JSON receipts record actual wall time,
peak RSS, command and stop reason.

| Check | Result | Receipt |
| --- | --- | --- |
| `cargo test --manifest-path engine_rs/Cargo.toml --locked` | 69 passed: 41 unit, 9 RNG, 19 replay; none ignored | `engine.json`, `engine.log` |
| `uv run python scripts/check_engine_trim.py` | OK | `trim.json`, `trim.log` |
| `cargo test --locked` | 254 passed, 4 ignored | `root.json`, `root.log` |
| `uv run maturin develop` | Current-source extension build passed | `python-build.json`, `python-build.log` |
| `uv run pytest tests/kaggriculture -q` | Combined process stopped at the 1 GB bound; complete bounded coverage below | `kaggriculture.json`, `kaggriculture.log` |
| All `tests/kaggriculture` tests, bounded batches | 311 passed, 3 skipped | `shards-tests-kaggriculture.json`, `remaining.json` |
| All `tests/owl` files, bounded batches | 787 passed, 3 skipped | `shards-tests-owl.json` |
| Observation custody and engine-trim pytest files | 124 passed | `custody.json`, `custody.log` |
| Restored `uvx --offline --from rust-just just rs-prepare` | Passed trim, formatting, Clippy, root 254/4 ignored, engine 69, docs freshness | `restored-rs-prepare.json`, `restored-rs-prepare.log` |

The full Kaggriculture command stopped at 1,006,518,272 sampled bytes after
11.24 seconds. File-level sharding also exhausted the bound in the model-head
file. Splitting that file into its compiled-core test and all other tests
completed all 65 cases (1 + 64); no test was omitted. The six total Python
skips are the unavailable native grammar binding, two CUDA-only pinned cases,
two flash-attention CUDA cases and an unavailable x86 quantized backend.
Orbit root fixture generation and replay tests passed. Python totals above
are fresh runs, not copied from the earlier 1,437-test preparation receipt.

## Semantics and oracle discrimination

Reviewed all 29 named buffer mappings, dtypes, applicability/sentinels, exact
integer and bank preservation, insertion ranks, role order, own-private
isolation, config/hire rules, masks, array admission and preflight-before-write
against the contract, kernel and reference-branch source. The root's Orbit
float-fixture repair is test-only. No retained vendored kernel bytes changed;
the trim checker confirms the pinned inventory and authorized historical trim.
The grammar kernel tests now run in root, and the engine bridge is removed.

The recorder invokes actual pinned `myolie_features::encode_invest` at
`65f0eac5bb00b18a9d3acce319c2a231cbd5dff0`; its feature SHA-256 is
`24a7d09f9ddf8196aed7e5dc3b18d7590562370d697c8407f95b4cc8b4f9a6ab`.
The tensor-only reconstructor matches all 8,176 offsets across 512 states and
1,024 seat rows. Custody checks verify the 384 official / 96 seeded / 32 dense
partition, source hashes and unchanged quotas. There are six non-synthetic
states with more than 16 actors against quota four. Shed-order variation uses
the documented dense-state exception. The actual merged Python schema accepts
all 512 records.

One fresh production mutation swapped market inventory and price sources in
`write_shops_and_market`. Custody still passed, then the comparison failed with
exit 101 at `official:95324500:0`, seat 0, offset 889: actual 0.0025 versus
recorded 1.0. A `finally` block restored the source byte-for-byte; before and
after SHA-256 are
`0d84eb9df01f699767fc674996e5eae14be31c09f70b058b5af35fce36365749`.
The restored full Rust suite and preparation checks passed. See `mutation.py`,
`mutation.json`, `mutant.log` and `restored-rs-prepare.log`.

## Documentation, limits and worktree custody

Current docs distinguish observation preservation from engine differential
parity and label historical handoff receipts. Cookbook adaptation inventory,
index and log describe the implemented source and current qualification.
The one diagnostic inconsistency is listed above. Optimized timing remains
unmeasured, as accurately documented and allowed by the brief's Mac-to-pod
handoff. CUDA/pinned behavior, live lifecycle integration, huge-aggregate
transition safety and buffer reuse fences are not established by this review.
No performance claim or new full corpus regeneration is made.

Three-dot `git diff --check` reports whitespace in 85 frozen operational
receipt files; excluding `ops/`, the implementation, tests, docs and cookbook
diff passes. Raw receipts were preserved.

All 1,500 initially hashed tracked files match their original bytes. Both
unstaged and staged tracked diffs remain empty. Existing untracked receipts
were preserved; this review adds only untracked files in this directory.
`summary.json`, `identity.json` and the before/after status receipts record
the final evidence.

VERDICT: APPROVE WITH EDITS
