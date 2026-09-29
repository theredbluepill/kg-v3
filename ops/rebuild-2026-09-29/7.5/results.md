# Task 7.5 result receipt

Date: 2026-09-29. Branch: `kg/rebuild-7-5`.
HEAD: `bde337465a9fa7c07bedded88d5d696d7cefb7ef`.

Target: add one evidence map immediately before the Kaggriculture rules-kernel
section, with a claim ledger, current bounded checks, and minimal repairs to
contradicted detailed prose. Stop when the map and ledger account for tested
layers, evidence limits and actual command outcomes. This is documentation work;
it does not qualify a trainer, model, GPU path or learning result.

Only `docs/rules-parity-coverage.md` and this `7.5/` evidence directory changed.
No code, fixture, config, lockfile, engine or cookbook edit was made. No Git write
was attempted. Claude owns the cookbook record. Read-only subagents audited BC,
grammar/observation and native/adapter evidence; their final review caught an
overbroad observation-timing gap and two ledger references, now corrected.

## Check environment and limits

Every shell exported:

```sh
export CARGO_BUILD_JOBS=2 CARGO_NET_OFFLINE=true UV_OFFLINE=true
export RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=1
export TMPDIR=/Users/poonszesen/kg-v3-t75/.codex-tmp
```

The environment was initially absent. `uv sync --offline --no-install-project`
installed only the existing locked dependencies; `uv run --offline --no-sync
maturin develop` built this source in development mode. Subsequent `uv run`
checks also exported `UV_NO_SYNC=true`, retaining that installed development
build. No dependency was added or fetched online.

The inline watchdog sampled each command's descendant-process resident memory
using macOS libproc every 0.25 seconds and stopped at 960 MiB or 120 seconds.
JSON sidecars record observed wall time, sampled peak, stop reason and process
status. Sampling can observe memory just above the threshold before termination.
The root retry and pytest overlapped briefly; guards measured each process tree,
not the combined memory of concurrent commands. These receipts therefore do not
establish an aggregate-task RAM bound below 1 GB. No stopped suite was forced or
rerun with a relaxed bound.

Two early monitor failures are kept visible:

- The first engine command completed its test summaries, but the watchdog failed
  because sandboxed `ps` was unavailable. `engine-tests-initial.log` lacks an
  authoritative process exit/resource receipt. The libproc-monitored rerun is
  the current engine result.
- The first trim invocation started uv's automatic native build. Its watchdog
  sent SIGTERM, then cleanup raised `PermissionError`; the chosen limit, peak
  and child exit were not saved. `trim.log` preserves only the launch/build
  output. It is not a passed trim check. The successful development build and
  later trim result have complete receipts.

No training, panel, GPU or separate model run occurred. No new `sweep.py` run
occurred. The requested pytest file contains live committed-fixture regeneration;
the selection was stopped without a final pytest summary.

## Actual commands

| Command | Result | Receipt |
| --- | --- | --- |
| `cargo test --locked --offline --manifest-path engine_rs/Cargo.toml` | Exit 0: 41 library + 9 RNG + 19 replay = 69 passed; 0 failed, 0 ignored; 0 doc tests | `engine-tests.log`, `.json` |
| `cargo test --offline` | Initial exit 101: missing `.venv/bin/python` | `root-tests.log`, `.json` |
| `cargo test --offline` after setup | SIGTERM, process status -15; 25.511 s; sampled peak 986,544 KiB; RSS guard; no final suite counts | `root-tests-after-setup.log`, `.json` |
| `uv run --offline python scripts/check_engine_trim.py` after setup | Exit 0: `engine trim manifest: OK` | `trim-after-setup.log`, `.json` |
| `uv run --offline pytest tests/scripts/test_kaggriculture_parity.py tests/tools/test_check_engine_trim.py tests/kaggriculture -q` | SIGTERM, shell status 143; 21.570 s; sampled peak 989,744 KiB; RSS guard; no final suite counts | `pytest.log`, `.json` |
| `uv run --offline python scripts/check_doc_freshness.py` | Exit 0: `No doc updates required`; no mapped follow-up edit | `docs-fresh-final.log`, `.json` (initial pass also retained as `docs-fresh.*`) |
| `uv sync --offline --no-install-project` | Exit 0; locked local dependency setup | `dev-dependencies.log`, `.json` |
| `uv run --offline --no-sync maturin develop` | Exit 0; 22.704 s; sampled peak 618,512 KiB | `dev-build.log`, `.json` |

`installed-pin.txt` independently reads installed package metadata and interpreter
bytes without importing or stepping Kaggle: version 1.32.7 and SHA-256 match.
`source-audit.txt` aggregates committed trace headers/manifests and records Git
ancestry. `bc-audit.txt` records the read-only branch check and pairing sums.

## Unverified and omitted

- BC mismatch cause, day-end attribution and unaffected-label claims. The sampled
  episode configurations/clock states are absent from the inspected tracked
  evidence; the receipt's pod archive paths are unavailable locally. Modulo-24
  arithmetic alone is insufficient. No remote archive access was attempted.
- Full current root/Python pass/fail/ignore/skip totals. Both commands stopped
  before summaries. Historical full-prepare totals remain historical.
- Fresh live-regeneration completion. The requested pytest did not finish.
- Later approval/results on unmerged branches. The summary uses this tip's
  dated tracker for 7.1/7.3/7.4 and only the expressly permitted BC evidence.
- A new weed-range parity claim. The locally installed JSON schema is available
  and declares a minimum of zero but no maximum; reading it is not a transition
  parity test, so it was not promoted into this coverage summary.

The BC branch had advanced to `954f640a6d396d5d012eccee0c21aa552ec76d14` during
inspection. Pairing JSON and preparer source remain byte-identical to requested
`933d661`; the summary pins that requested evidence commit. The later receipt
withdraws the unaffected-label claim. No branch checkout, write or execution was used.

## Final inventory and exact documentation limits

The generated inventory, copied not-tested list and correction locations below
are part of this receipt. Structural QA separately verifies the Orbit Wars body
is byte-identical, the new summary has one occurrence before the rules kernel,
the seven table rows and anchors exist, and the diff stays inside scope.

### Files changed or added

- `docs/rules-parity-coverage.md`: coverage map, opening pointer and minimal corrections.
- `ops/rebuild-2026-09-29/7.5/bc-audit.txt`
- `ops/rebuild-2026-09-29/7.5/claims.md`
- `ops/rebuild-2026-09-29/7.5/dev-build.json`
- `ops/rebuild-2026-09-29/7.5/dev-build.log`
- `ops/rebuild-2026-09-29/7.5/dev-dependencies.json`
- `ops/rebuild-2026-09-29/7.5/dev-dependencies.log`
- `ops/rebuild-2026-09-29/7.5/docs-fresh-final.json`
- `ops/rebuild-2026-09-29/7.5/docs-fresh-final.log`
- `ops/rebuild-2026-09-29/7.5/docs-fresh.json`
- `ops/rebuild-2026-09-29/7.5/docs-fresh.log`
- `ops/rebuild-2026-09-29/7.5/engine-tests-initial.log`
- `ops/rebuild-2026-09-29/7.5/engine-tests.json`
- `ops/rebuild-2026-09-29/7.5/engine-tests.log`
- `ops/rebuild-2026-09-29/7.5/installed-pin.txt`
- `ops/rebuild-2026-09-29/7.5/pytest.json`
- `ops/rebuild-2026-09-29/7.5/pytest.log`
- `ops/rebuild-2026-09-29/7.5/results.md`
- `ops/rebuild-2026-09-29/7.5/root-tests-after-setup.json`
- `ops/rebuild-2026-09-29/7.5/root-tests-after-setup.log`
- `ops/rebuild-2026-09-29/7.5/root-tests.json`
- `ops/rebuild-2026-09-29/7.5/root-tests.log`
- `ops/rebuild-2026-09-29/7.5/source-audit.txt`
- `ops/rebuild-2026-09-29/7.5/structure-check.txt`
- `ops/rebuild-2026-09-29/7.5/trim-after-setup.json`
- `ops/rebuild-2026-09-29/7.5/trim-after-setup.log`
- `ops/rebuild-2026-09-29/7.5/trim.log`

### Not tested (verbatim from the summary)

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


### Existing statements corrected

- `docs/rules-parity-coverage.md:5`: Opening paragraph lacked a map pointer -> added the Task 7.5 summary link.
- `docs/rules-parity-coverage.md:324`: Grammar coverage returns with Task 1.2 -> Grammar coverage returned with Task 1.2.
- `docs/rules-parity-coverage.md:307`: Game::new(config, seed, 2) -> Game::new_with_seed_decimal(config, seed, 2).
- `docs/rules-parity-coverage.md:515`: engine grammar include is temporary; Tasks 1.3/1.4 will retire it -> Task 1.3 retired it; acceptance tests run in root grammar_kernel_tests.rs.
- `docs/rules-parity-coverage.md:603`: see the Task 1.5 note at the end of this page -> link to the Task 1.5 summary row.
- `docs/rules-parity-coverage.md:687`: exact ten-case binary64 admission predicate -> ten cases plus the Task 1.5 strengthening case.
- `docs/rules-parity-coverage.md:656`: phase costs remain unmeasured -> Task 1.3 diagnostic incomplete; Task 1.4 component costs measured.
- `docs/rules-parity-coverage.md:783`: adapter/rewards/codec/table bridge and CUDA qualification belong to Task 1.5 -> CPU implementation merged; CUDA reuse-fence qualification remains open.

VERDICT: DONE

## Claude review (2026-09-29)

- Every cited test file, test function and receipt path in the new summary was
  checked to exist at `bde3374` (`episode_*`, `generated_fixtures_replay`,
  `env_directory_traces`, `compare_observation_oracle`,
  `test_every_frozen_oracle_record_passes_the_actual_schema`,
  `test_native_matches_training_batch_16_complete_games`,
  `test_project_environment_satisfies_the_engine_pin`, `load_pinned_kaggle`,
  `Game::new_with_seed_decimal`, and all 27 file/receipt paths). Spot-checked
  numbers against receipts: probe arithmetic (23,339 − 21,824 = 1,515) in
  `1.1b/verify-r1/sweep-summary.json`; 321/43 codec records and three 96-transition
  extreme-coefficient games in `stage2-adapter/native/results.md`; BC pairing
  5,743/5,752 and the nine private-only mismatches in
  `git show kg/rebuild-bc-now:ops/rebuild-2026-09-29/bc-a100-2026-09-29/pairing.json`.
- Codex's guarded root Rust and pytest runs stopped at its 960 MiB guard. Claude
  ran `uvx --from rust-just just py-prepare` (exit 0; 2,319 passed, 10 skipped;
  docs-fresh "No doc updates required"; `py-prepare.log`) and replaced those two
  rows in the doc's "Current checks" table. The root Rust suite was not rerun:
  `git diff --name-only 7f797a3 HEAD -- '*.rs' Cargo.toml Cargo.lock engine_rs`
  is empty, so the merge's `merge-env-adapter/prepare.log` (274 passed, 5 ignored)
  is cited as the latest full run.
- BC paragraph: added that the receipt itself calls the mismatch turns day ends,
  still unconfirmed here.
