Task 7.5 documentation is complete. Two test commands stopped at the RAM guard; their incomplete results are recorded explicitly.

Changed/added files:

- [docs/rules-parity-coverage.md](/Users/poonszesen/kg-v3-t75/docs/rules-parity-coverage.md:139): 122-line summary and minimal corrections.
- [claims.md](/Users/poonszesen/kg-v3-t75/ops/rebuild-2026-09-29/7.5/claims.md): 96 claim-ledger rows.
- [results.md](/Users/poonszesen/kg-v3-t75/ops/rebuild-2026-09-29/7.5/results.md): complete report and verification limits.

Additional receipts, each under `ops/rebuild-2026-09-29/7.5/`:

```text
bc-audit.txt
dev-build.json
dev-build.log
dev-dependencies.json
dev-dependencies.log
docs-fresh-final.json
docs-fresh-final.log
docs-fresh.json
docs-fresh.log
engine-tests-initial.log
engine-tests.json
engine-tests.log
installed-pin.txt
pytest.json
pytest.log
root-tests-after-setup.json
root-tests-after-setup.log
root-tests.json
root-tests.log
source-audit.txt
structure-check.txt
trim-after-setup.json
trim-after-setup.log
trim.log
```

Checks ran on **2026-09-29**, HEAD **`bde337465a9fa7c07bedded88d5d696d7cefb7ef`**:

| Command | Actual result |
|---|---|
| `cargo test --locked --offline --manifest-path engine_rs/Cargo.toml` | Exit 0; **69 passed, 0 failed, 0 ignored** |
| `cargo test --offline` | Initially exit 101: missing local Python. After setup, RAM guard stopped it: **−15**, no final counts |
| `uv run --offline python scripts/check_engine_trim.py` | Exit 0; manifest OK |
| `uv run --offline pytest tests/scripts/test_kaggriculture_parity.py tests/tools/test_check_engine_trim.py tests/kaggriculture -q` | RAM guard stopped it: **143**, no final counts |
| `uv run --offline python scripts/check_doc_freshness.py` | Exit 0; no updates required |
| `uv sync --offline --no-install-project` | Exit 0 |
| `uv run --offline --no-sync maturin develop` | Exit 0 |

Subsequent uv checks used `UV_NO_SYNC=true`. Guards measured individual process trees; the brief root/pytest overlap does **not** establish an aggregate-task RAM bound below 1 GB.

The “not tested” list, as written:

- D1/D2 agreement: Unicode-decimal quantities and unhashable fields remain known
  divergences. Their repros assert failures; they do not establish repaired parity.
- Exhaustive Python rules-path or malformed-input agreement. Model grammar actions
  cannot exercise D1/D2, and the generated policies/probes are a bounded sample.
- Kaggle framework behavior outside the interpreter: timeouts, agent errors and
  `INVALID` statuses.
- Strong-play worlds beyond the four official episodes, or a larger pod parity sweep.
- Direct equality of recorded RNG/shop schedule headers; replay tests check their
  effects through state instead.
- Full-season codec parity on the official action streams, or every actor/order/HIRE combination;
  selected replay actions and local support classes are the tested scope.
- Whole-observation or whole-snapshot Python parity from the Task 1.4 fixture;
  its reference is Rust and its comparisons target the native transition boundary.
- Complete historical observation-corpus source custody: three engine input hashes
  were omitted; generation-time dirty bytes and the full producer module inventory are absent.
- Task 7.1 opponents and their oracle in this integration. The recorded approval
  is on unmerged `kg/rebuild-7-1`.
- Task 7.3 replay export / Kaggle-episode round trip in this integration;
  it is unmerged, with no approving verdict in this tip's phase tracker.
- Task 7.4 Kaggriculture packaging; this tip records a brief under review, not implementation.
- Task 3.1 Kaggriculture rollout/mask/action mapping, trainer-level execution or
  learning qualification; canonical `scripts/run_ppo.py` still stops at that seam.
- CUDA/BF16 native-adapter parity, hardware table upload and pinned-memory DMA
  reuse-fence qualification. Separate GPU model diagnostics do not qualify these paths.
- Complete-update throughput. Task 1.4 measured observation/lifecycle components;
  Task 1.3's dedicated timing diagnostic remains incomplete.

Existing corrections in `docs/rules-parity-coverage.md`:

- `:5`: no summary pointer → added summary link.
- `:306`: `Game::new` → `Game::new_with_seed_decimal`.
- `:324`: grammar coverage “returns” → “returned”.
- `:515`: temporary bridge awaiting retirement → already retired into root tests.
- `:603`: nonexistent end-of-page Task 1.5 note → summary-row link.
- `:656`: observation phase costs unmeasured → Task 1.3 diagnostic incomplete; later Task 1.4 component measurements credited.
- `:687`: ten admission cases → ten plus Task 1.5 strengthening case.
- `:783`: Task 1.5 work described without completion status → CPU implementation merged; CUDA qualification remains open.

Left out: BC day-end attribution, mismatch cause and unaffected-label claims; completed current root/Python totals; fresh regeneration completion; later unmerged results. Orbit Wars text is byte-identical. No code, cookbook or Git-state changes.

VERDICT: DONE