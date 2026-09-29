# Task 1.3 handoff — incomplete

Authority: the reviewed brief including R1–R5/Q1–Q6, plus the owner’s explicit schema, Mac, offline and Git constraints. Resumed from committed partial `32e2cdd6e2bbbddb6a847bc0c12b39b2d8a66a90`; all subsequent changes remain in this working tree. No Git write, vendored edit, network, training job or GPU diagnostic occurred.

**Native implementation is present; Task 1.3 is not complete.** The unchanged R1 generator cannot satisfy its actor quota, Task I’s actual schema is unmerged, and R4 optimized timing must run on the pod after the local memory stop.

## 1. Files changed

[`files-changed.md`](files-changed.md) lists every changed/added path since the recovery HEAD, one line per file with purpose, including all command receipts. The original committed A regression and its pre-dependency receipt remain part of the task evidence. [`final-source.sha256`](final-source.sha256) pins the 23 final code/dependency/doc/cookbook files separately from working receipts. No `engine_rs/` file is changed.

## 2. Final commands and actual results

Every build/test shell exported:

```sh
export TMPDIR=/Users/poonszesen/kg-v3-observe/.codex-tmp CARGO_BUILD_JOBS=2 CARGO_NET_OFFLINE=true UV_OFFLINE=true RAYON_NUM_THREADS=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RUST_TEST_THREADS=1
```

Each command below ran inside `python3 ops/rebuild-2026-09-29/1.3/bounded.py --name <receipt> -- ...` (120 seconds and 1,000,000,000 sampled RSS bytes). Full argv, exit, wall time and RSS are in [`final-commands.json`](final-commands.json) and the individual JSON/log pairs. No final command hit its guard.

| Command | Actual result | Wall seconds | Sampled peak bytes |
|---|---|---:|---:|
| `cargo test --locked --offline` | [233 passed, 1 failed (missing qualified corpus), 4 ignored; exit 101](final-01-root-tests.log) | 11.81 | 565,280,768 |
| `cargo test --locked --offline --manifest-path engine_rs/Cargo.toml` | [59 passed, 0 failed, 0 ignored; exit 0](final-02-engine-tests.log) | 17.66 | 551,272,448 |
| `uv run --offline python scripts/check_engine_trim.py` | [engine trim manifest: OK; exit 0](final-03-trim.log) | 25.09 | 747,159,552 |
| `uv run --offline pytest tests/kaggriculture/test_observe.py tests/tools/test_observation_oracle_custody.py -q` | [1 collection error (missing schema); no tests executed; exit 2](final-04-observe-custody-pytest.log) | 1.88 | 255,377,408 |
| `uv run --offline --extra reference pytest tests/owl tests/scripts tests/tools -m 'not slow' -q` | [undefined reference extra; pytest not started; exit 2](final-05-reference-fast-pytest.log) | 0.11 | 638,976 |
| `uv run --offline pytest tests/owl tests/scripts tests/tools -m 'not slow' -q` | [1,056 passed, 0 failed, 3 platform skips; exit 0](final-05b-fast-pytest-no-extra.log) | 18.88 | 614,907,904 |
| `uvx --offline --from rust-just just rs-prepare` | [233 passed, 1 corpus failure, 4 ignored; exit 101](final-06-rs-prepare.log) | 11.97 | 482,705,408 |
| `uvx --offline --from rust-just just py-prepare` | [1 schema collection error; no tests executed; exit 2](final-07-py-prepare.log) | 10.24 | 876,036,096 |
| `uvx --offline --from rust-just just prepare` | [233 passed, 1 corpus failure, 4 ignored; exit 101](final-08-prepare.log) | 13.50 | 483,131,392 |
| `uvx --offline --from rust-just just docs-fresh` | [No doc updates required; exit 0](final-09-docs-fresh.log) | 0.21 | 101,285,888 |
| `uv run --offline python ops/rebuild-2026-09-29/1.3/native_boundary_smoke.py` | [55 native checks; schema_checked=false; exit 0](final-10-native-boundary.log) | 1.05 | 75,382,784 |
| `python3 ops/rebuild-2026-09-29/1.3/final_audit.py` | [diff/whitespace, note lint/48 source paths, mutation receipts and vendor status pass; exit 0](final-11-source-audit.log) | 0.11 | 163,840 |

`rs-prepare` passes trim, root/engine formatting and root/engine Clippy before root tests fail. Its later engine-test/docs-fresh steps are not reached; they pass in separate commands above. `py-prepare` passes formatting (89 files unchanged), Python 3.11 syntax, Ruff and mypy (52 source files), then stops at collection. `prepare` also builds the extension and passes docs lint before the same root test failure; it never reaches its full Python suite. The broad no-extra fast run includes all 43 custody cases. Its three skips are two unavailable CUDA flash-attn cases and one unavailable x86 quantized backend.

Exact unresolved errors:

```text
qualified 512-state observation oracle missing: R1 non-synthetic >16-actor quota is 0 < 4; full reference comparison remains blocked
ModuleNotFoundError: No module named 'owl.kaggriculture'
error: Extra `reference` is not defined in the project's `optional-dependencies` table
```

The four ignored Rust diagnostics are the two inherited Orbit diagnostics, explicit corpus generation, and optimized timing. The missing-corpus comparison is **not ignored**. Missing schema is an unconditional import failure, not an optional dependency skip. The undefined extra is a command/branch mismatch, not a schema failure; no dependency was added to conceal it.

### Red/green receipts by task

| Task | Actual evidence |
|---|---|
| A | [`task-a.md`](task-a.md): literal decimal regression passes before dependencies, fails after feature unification (`invalid type: map, expected f64`), then Number repair passes. Root checkpoint 157 passed/2 ignored, trim OK. |
| B | [`task-b.md`](task-b.md): missing-API reds, then 11 boundary tests green; disabled length check fails and restoration passes. |
| C | [`task-c.md`](task-c.md): missing API red, nine config/four hire tests green (combined B/C 24). Rival-count mutation fails two tests; restored green. |
| D | [`task-d.md`](task-d.md): nine semantic failures plus independent constructor scan, then ten passes. Transposed coordinates fail; restored green. Scan: 2,880 states/576,000 tiles. |
| E | [`task-e.md`](task-e.md): twelve actor/storage/privacy semantic failures become passes (B–E 46). Reversed actor/shed ranks fail two tests; restored green. |
| F | [`task-f.md`](task-f.md): five context reds, no-op diagnostic red and separate fractional-bonus red become 54 B–F passes. 72 corruptions rejected; late-env failure, serial/two-worker equality and reuse pass. |
| G | [`task-g.md`](task-g.md): custody import/negative controls and eight producer stub failures become green; full generation emits 512 then fails unchanged R1. Nine source-custody regressions fail, then pass; final custody 43 green. Source-only reference recorder compiles with 125 original files unchanged. |
| H | [`task-h.md`](task-h.md): zero reconstructor yields two semantic failures; no-op recorded checks/decoder yield two more. Final eight reconstruction controls and five added-fact tests pass. All 16 temporary mutations fail, source is restored. Full reference comparison remains blocked. |
| I | [`task-i.md`](task-i.md): separate native missing-function AttributeError before implementation; 55 NumPy checks green plus build/Clippy/Ruff/stub mypy. Actual-schema collection fails, explicitly not counted as semantic red. Schema/corpus tests are authored but unexecuted. |
| J | [`task-j.md`](task-j.md): two-acquisition implementation fails (2 vs 1), single acquisition passes; stable allocations/282,246 bytes for sparse+dense rows. Optimized build stops for memory; no costs measured. Docs/cookbook checks pass; aggregate gates retain blockers above. |

Actual production/formula mutation episodes: B length admission, C rival hire count, D x/y transpose, E reversed ranks, H 16 boundary/formula cases, J duplicate snapshot. H covers summary, tile, first/last actors and inventory entries, storage, market, swapped shops, actor count, maintenance, suffix, ever/current fertilizer, dropped positive zero and constants. [`h13-mutations.json`](h13-mutations.json) records exact snippets, intended mismatch offsets and matching pre/post restoration hashes. No mutation remains. Incidental malformed test expectations, compile/style failures and failed early watchdog probes are retained and labelled separately; filenames containing “green” are not themselves evidence of success.

## 3. Oracle and fixture results

Actual producer output: **512 input records = 384 official + 96 seeded + 32 dense**, with 480 non-synthetic records. These headers represent 1,024 potential seat rows. **Zero full reference feature seat rows were recorded; zero offsets were compared against a frozen full corpus; mismatches are not computed.** Eight hand/stream reconstruction controls cover all 8,176 positions independently but do not replace the missing reference corpus. Full source-only recorder compilation is in `g-recorder-compile.json`.

R1 non-synthetic coverage, independently recounted in Python and matched to native generation:

| Quota | Actual | Required minimum | Result |
|---|---:|---:|---|
| `actor_gt16_states` | 0 | 4 | fail |
| `animal_COW` | 4,702 | 4 | pass |
| `animal_GOOSE` | 243 | 4 | pass |
| `animal_SHEEP` | 2,677 | 4 | pass |
| `both_hires_nonzero_states` | 379 | 8 | pass |
| `crop_CARROT` | 173 | 4 | pass |
| `crop_MELON` | 2,999 | 4 | pass |
| `crop_STRAWBERRY` | 7,934 | 4 | pass |
| `crop_TOMATO` | 373 | 4 | pass |
| `crop_WHEAT` | 8,169 | 4 | pass |
| `fert_current` | 1,058 | 4 | pass |
| `fert_expired` | 257 | 4 | pass |
| `reordered_inventories` | 1,199 | 8 | pass |
| `reordered_sheds` | 0 | 1 | explicit_dense_d31_exception |
| `shops_ge4_states` | 266 | 8 | pass |
| `tile_COOP` | 742 | 8 | pass |
| `tile_EMPTY` | 28,897 | 8 | pass |
| `tile_LOCKED` | 32,825 | 8 | pass |
| `tile_PASTURE` | 8,112 | 8 | pass |
| `tile_PLANT` | 19,648 | 8 | pass |
| `tile_WEED` | 5,776 | 8 | pass |
| `unfed` | 922 | 4 | pass |
| `unwatered` | 7,053 | 4 | pass |

Only the explicitly reviewed d30/d31 remove/reinsert pair supplies the shed-order exception; no exception or quota reduction covers the failed actor quota. Official selection maximum is 13 actors; seeded HIRE/day-reset upper bounds are [6,4,2,2,9,4]. Exact counts/action hashes/proof: `g-input-generation.json`, `g-input-audit.json`, `r1-blocker.md`, `oracle-results.json`.

R3 final compressed fixture sizes and SHA-256s: **unavailable**, because final publication stops before reference recording. No ≤8,388,608-byte budget pass is claimed. The unqualified raw header stream has 8,405,456 bytes and SHA-256 `a915294b5e9bc474529ae347b052b32488fddf78c84be027e493c21bbb3c8b53`; this is not a compressed qualified fixture hash.

Pinned official input fixtures (not new observation fixtures):

| Fixture | Compressed bytes | SHA-256 |
|---|---:|---|
| `engine_rs/fixtures/episode-95324500.jsonl.gz` | 247,813 | `47cdfa489b7a80edf8ec1361f2f55cd033c75824d624cd7c9e9daaa3137affd7` |
| `engine_rs/fixtures/episode-95901360.jsonl.gz` | 281,033 | `e80653f445570a3778a3fb9026a66614b1ecaa2ea417cf8358d3f1850d725281` |
| `engine_rs/fixtures/episode-95921764.jsonl.gz` | 264,735 | `bc3e01cd12ff70fd2f78bfbe7129d5caca468a6324c124c25c9efebc86fbd3f2` |
| `engine_rs/fixtures/episode-95990191.jsonl.gz` | 260,289 | `4bf1a3b09c644719c8b36a619289844e52c3458d0e25dc0a0a1429bc6eaa0d1b` |

## 4. Timing and resource handoff

Profile: root release, fat LTO, one codegen unit, CARGO_BUILD_JOBS=2. Actual build attempt stops at 53.8096 seconds when sampled process-group RSS reaches 1,052,393,472 bytes; SIGKILL/child exit −9, wrapper 137. Build did not complete and the test body did not run. Snapshot/validation/prepared-write/full-write phase costs are **not measured**. No debug timing substitutes. CPU brand query was sandbox-denied; platform/architecture and source fingerprints are in [`timing.json`](timing.json).

Claude pod command, unchanged from R4 (use an appropriate pod temporary directory and the same CPU/offline exports):

```sh
cargo test --release --locked --offline --lib kaggriculture::tests::measure_observe_cost -- --exact --ignored --nocapture
```

The authored test preserves one game, 20 warmups + 200 measured repetitions per phase for sparse `official:95324500:0` and dense `dense:0`, checks fixed bytes/allocations and records hashes and per-env/per-seat costs. One public deep snapshot still allocates; this is not a zero-copy or production-SPS claim. The Mac guard samples every 0.1 seconds, so the reported first over-limit sample can exceed the threshold.

## 5. Deviations and corrections

- Pinned PyInt has no Display implementation. Exact public Serialize→decimal parse is used only during config admission; no float intermediate, live-state JSON or kernel accessor. Tests cover exact i64 limits.
- G/H full qualification stops at the literal R1 contradiction; no substitute corpus, lowered quota, recipe change or missing-fixture skip.
- Task I actual-schema acceptance is unexecuted pending Task 2.1; separate NumPy checks are explicitly not schema qualification.
- R4 optimized costs move to the pod after the required memory stop; no reduced workload or debug substitute. A Clippy-only runtime profile-guard fix follows the stopped attempt and is recorded separately.
- The requested `--extra reference` does not exist here; exact failure is preserved and the broad suite is also run without it. No optional extra is invented.
- Internal existing just recipes omit some literal offline/locked flags; they ran unchanged under the required offline environment. Every direct Cargo invocation used `--locked --offline`; no lockfile changed during preparation.
- G review exposed late source hashing and an earlier Mac PID-count/watchdog bug. Both received attributable regression failures, repairs and green checks; earlier incomplete RSS observations are not recast as measured full runs.
- First-edit cookbook hooks required reading governing notes before retrying doc writes. They did not require user approval; the reads and edits were completed.

## 6. Open questions and unresolved items

- Task 2.1: merge the actual shared schema through Claude integration, then run Task I real-schema and full corpus tests; do not copy a schema.
- R1: Claude must review a recipe correction capable of four non-synthetic >16-actor states; current instructions prohibit an autonomous recipe/quota change.
- G/H: after that correction, generate the complete 512-record oracle, verify all 1,024 seat rows/all 8,176 offsets, compressed budget and final fixture SHA-256s.
- R4: run the exact optimized timing command on the pod; phase costs and snapshot-path performance qualification remain unknown.
- Q4/Task 7.5: the pinned official weed JSON-schema range is unavailable locally; current writer accepts finite nonnegative numeric values without an invented ≤1 cap.
- Task 1.4: separately establish transition-safe huge aggregates, game/seed/terminal rollback and reuse fences; this writer proves only output publication.
- Tooling: reconcile the requested nonexistent `reference` extra with the integration branch; broad fast tests already pass without it.
