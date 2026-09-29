# Task G — Reference oracle custody and fixed-recipe input producer

Target: implement reproducible, source-bound recording and independently checked
custody, then run the prescribed input recipe without weakening its quotas.
Stopping condition: reject the final corpus if any required non-synthetic quota
fails; do not build/record reference features from a failed generation.

## Implemented inventory

- `src/kaggriculture/oracle_corpus.rs` and its test-only module registration:
  strict record/source types; native official replay and ordered state checks;
  six literal seeded policies; 32 literal dense states; source-domain admission;
  checked gzip children; repeatable action streams, hashes and round trips.
- `scripts/kaggriculture_observation_oracle/record.rs`: standalone example for
  the complete pinned reference export, calling only its `encode_invest` for
  expected values. Errors remove only the raw output created by this process.
- `scripts/kaggriculture_observation_oracle/regenerate.py`: exact manifest
  admission, raw header-byte hashing, two seat hashes per record, source and
  coverage recounts, complete export verification, deterministic gzip, optional
  lossless byte-plane storage, and atomic fixture-directory installation after
  validation. Source identity is captured before producer execution and checked
  between phases and before installation; the exported recorder must match its
  initial source hash before and after execution. Subcommands use process-group
  timeout/cleanup and sampled RSS on Mac/Linux; generation remains offline.
- `tests/tools/test_observation_oracle_custody.py`: tiny custody, corruption,
  recipe typing, dense-pair, deterministic compression, byte-plane and driver
  controls. These fixtures are temporary test inputs, not qualified oracle data.
- `ops/rebuild-2026-09-29/1.3/`: bounded command logs/receipts, compact generation
  report, source hashes, and source-only recorder compile check.

The producer's `timing_header(false)` selects official episode 95324500 step 0;
`timing_header(true)` selects dense d=0. Both are available without a qualified
final fixture. All legacy offsets remain reserved for Task H's test reconstructor.

## Actual fixed-recipe result

`g25-full-inputs-r1` produces exactly 512 input records: 384 official, 96 seeded,
and 32 dense. Each official episode is replayed through all 719 transitions,
checking states and nested object order, and selects the prescribed 96 steps.
Each seed samples through step 94, then runs the last action at step 94 to
terminal step 95. The complete 95-line action stream is replayed from seed and
its final snapshot hash is checked. Six action files and raw headers stay in
`.codex-tmp/g-observation-inputs/`; only compact hashes/counts are retained in ops.

The 480 non-synthetic records contain **zero** states above 16 actors, below the
required four. Every other required non-synthetic quota passes except reordered
sheds, whose specified exception uses dense d=31. The Rust full-generation test
fails explicitly with `actor_gt16_states=0 < 4` after writing its actual coverage
report. Runtime: 15.93 seconds; sampled aggregate peak RSS: 65,716,224 bytes.
This confirms the earlier R1 arithmetic audit with actual engine execution.

Dense d=31 changes only the first key's insertion position in seat 0's farmer
inventory and shed relative to d=30, with identical values and remaining header.
The Python full-input/corpus validators independently enforce that exact pair.
Native sheds start with all 12 keys; the pinned engine replaces/decrements them
without removing keys. Accordingly the explicit dense order exception remains
visible; no synthetic state satisfies the non-synthetic actor quota.

No qualified `tests/fixtures/kaggriculture/observation-v3/` fixture is installed.
No full legacy comparison or schema corpus qualification is claimed. R1 is a
dependency failure, not an encoder semantic red.

## Checks and evidence

- `g1-custody-red`: the tests first fail because the driver module is absent.
- `g7-producer-red`: all eight producer tests fail with the initial stubs,
  including actual policy, comparator and quota assertions.
- `g24-producer-green`: all eight producer tests pass (5.71 seconds including
  compile, sampled peak 644,300,800 bytes). The intermediate g21 JSON macro
  syntax error is an incidental compile failure, not semantic evidence.
- Python review tests expose float-valued recipe integers, source/header step
  mismatches, and missing dense-pair enforcement before repairs. An exponent
  test's spelling typo was corrected separately; it is not an implementation red.
- `g19-custody-green`: 32 tiny custody tests pass. These include one-byte changes,
  ordered header changes, duplicate/missing records, swapped seats, wrong feature
  length, versions/keys/source tags, strict integer metadata, exact d30/d31 edits,
  deterministic gzip, shuffled seat-byte admission, source export mismatches,
  sampled memory-stop behavior, and refusing reference export after producer
  failure. Final source checks are recorded separately below.
- `g23-python-lint` and `g18-python-mypy`: targeted Ruff and mypy pass. Intermediate
  Python-only checks use `uv run --offline --no-sync` after the initial normal uv
  build so they cannot rebuild changing Rust sources concurrently.

Every recorded build/test uses the prescribed offline environment, two build
jobs/two Rayon threads, one Rust test/OMP/MKL thread, the worktree TMPDIR, and the
120-second/1-GB `bounded.py` wrapper. No model, training or GPU work occurred.

## Driver, resource-guard correction and source-only compile

The exact future regeneration command was run. `g26-full-driver-r1` fails before
exporting/building the reference, but exposed an incidental monitoring defect:
the driver interpreted Mac `proc_listpgrppids`' PID count as a byte count, reporting
zero internal RSS. Its outer wrapper did not include the new child process group;
that receipt therefore does not measure complete native-generation RSS. The
separate `g25` direct invocation supplies bounded native-run evidence.

`g27-rss-real-red` measures zero for a live child holding 32 MiB. Correcting the
return-unit interpretation makes `g28-rss-real-green` pass. The driver now sums
caller and child groups, kills the child process group on failure/timeout, and
uses a shared deadline five seconds inside the outer boundary. The parent's
deadline probe also exposed process-relative Mac Python 3.9 monotonic clocks;
`bounded.py` now passes a Unix deadline, converted once into the child's monotonic
clock. The failed monotonic probe is retained separately from the successful
Unix-deadline probe. A short live deadline test confirms cleanup occurs before
the outer limit.

`g31-driver-r1-fixed-watchdog` repeats the full driver with the corrected guard:
15.45 seconds, combined caller/child sampled peak 134,086,656 bytes, explicit R1
failure, no reference export or final fixture. `g30-input-audit` independently
recounts the actual 512-row stream, validates source order, dense pair and header
steps, and reproduces the quota failure. All six 95-row action files match their
recorded hashes. The 8,405,456 input bytes have SHA-256
`a915294b5e9bc474529ae347b052b32488fddf78c84be027e493c21bbb3c8b53`.
Compact evidence is `g-input-generation.json` and `g-input-audit.json`.

Parent explicitly authorized a separate **source-only** recorder compile after
the R1 result. `g32-recorder-source-check` compiled successfully; its first
receipt omitted the successful command's internal peak, so `g34` repeats with
that instrumentation. `g-recorder-compile.json` records all 125 original reference
file hashes unchanged before/after, the exact recorder hash and command, rustc
and Cargo versions. Export plus check takes 14.70 seconds; Cargo check takes 7.02
seconds, sampled caller/child peak 627,818,496 bytes. No headers are passed to
the recorder and no reference feature bytes are generated. This proves source
compilation against the original crate, not feature parity or corpus admission.

`g36-clippy-green` passes all-target Clippy. `g37-root-tests` passes 219 tests,
with three ignored (the two inherited tests and explicit corpus generation).
The exact producer used by g25 is preserved at
`.codex-tmp/g25-oracle-corpus.rs`; `g25-producer-source.sha256` pins
`5237cda4a89e3c3de12a3e4c153f7b8675e958252b9300818d1155f76b8e242a`.
Subsequent Clippy changes replace equivalent remainder predicates and change
only the order of two independent field assignments; the eight producer tests
and complete root suite pass afterward. Current source hashes are recorded
separately from that execution snapshot.

`g39-custody-final` runs the ordinary requested
`uv run --offline pytest tests/tools/test_observation_oracle_custody.py -q`:
34 pass, 18.04 seconds including the editable extension build, sampled outer
peak 746,782,720 bytes. Final source inventory is `g-source.sha256`; the exact
G producer at that handoff is `.codex-tmp/g-final-oracle-corpus.rs`. Source/build
ownership then passes to Task H. Consolidated `rs-prepare`, `py-prepare`,
documentation and cookbook checks belong to the parent task's final receipt;
the earlier targeted Ruff/mypy checks above are not presented as that final gate.

## Independent review correction: source identity across execution

Review found that the driver originally captured source hashes and root commit
only after producer/reference execution. A source change between phases could
therefore be labelled as the source that produced earlier bytes. The exported
recorder was also outside the original reference file set checked after builds,
so its copy could differ from the recorder hash placed in the manifest.

`g29-source-custody-red` records nine semantic failures before the production
repair: six source/commit changes and two corrupted recorder copies were accepted
instead of rejected, and the control showed identity capture after the producer.
These tests mock producer/reference execution and corpus admission to isolate
execution custody. Their temporary two-row input, sparse feature placeholder and
mocked admission are not a full-generation result or an exception to R1.

The repair snapshots commit and all declared source, engine and official-fixture
hashes into immutable tuples before invoking the producer. It rechecks them
before production, after production, after reference export/execution, and after
staged corpus validation immediately before rename. Manifest fields use the
captured hashes. The copied recorder must match its captured hash before its
command starts and remain equal afterward. A mismatch leaves no final directory.
The producer-failure test still verifies the actual failing command is the native
producer and that reference export never starts.

`g30-source-custody-green`: all nine controls pass. Final
`g36-source-custody-final-green`: **43 custody tests pass** in 0.39 seconds
(0.63-second bounded command; sampled peak 74,514,432 bytes). Targeted Ruff,
format-check and mypy on both Python files pass in
`g35-source-custody-ruff-green`, `g37-source-custody-format-check` and
`g34-source-custody-mypy`; mypy's sampled peak is 230,047,744 bytes. The initial
lint run found two regex-string markings and an existing long test command;
these formatting issues were repaired before the final checks.

All repair checks use `uv run --offline --no-sync` under the 120-second/1-GB
wrapper and prescribed exported limits, with no Rust rebuild, live game,
reference execution or fixture generation. Only the driver, custody tests and
this receipt changed in the correction. The R1 recipe, quotas, actual coverage
report and unqualified full-comparison status are unchanged. Consolidated prepare
and cookbook inventory remain the parent task's responsibility.
