# Task A — root dependency and Orbit fixture decoding

Scope: Task A of the reviewed 1.3 brief, including R5. No vendored engine file,
production Orbit rule or comparison tolerance may change. Parent owns the combined
cookbook and final preparation record at Task J.

## Diagnostic contract

Question: does real root/engine `serde_json/arbitrary_precision` feature unification
break tagged `RandomCall::Uniform` decimal decoding, and does a test-only Number
conversion restore it without weakening finite-number rejection?

Use literal low/high/value values 0.125/1.75/0.625. Require a pass before adding
dependencies, a failure after adding them with the unchanged decoder, and green
root tests plus the engine trim check after the repair. Stop a bounded diagnostic
at 120 seconds or 1 GB process-tree RSS. No training, model or GPU work.

All build/test shells export `CARGO_BUILD_JOBS=2`, `CARGO_NET_OFFLINE=true`,
`UV_OFFLINE=true`, `RAYON_NUM_THREADS=2`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`
and `RUST_TEST_THREADS=1`. Cargo dependency edits use `cargo add --offline` only;
Cargo tests use `--locked --offline`.

## Actual checks so far

- Read CLAUDE.md, cookbook index/latest log, native-buffer Reference and restart
  Decision, the reviewed Task A/R5 brief, and both rules docs. Searched the
  cookbook for source governance, `arbitrary_precision` and `fixture_float`.
  Read the reference branch's fixture decoder with read-only `git show`.
- The first-edit hook surfaced the native-buffer Reference after the first edit
  attempt. Re-read it and explicitly retried the same edit successfully.
- Added only `arbitrary_precision_uniform_fixture_float` before dependency edits.
  `a1-before-dependencies.log` records **1 passed, 0 failed**, 157 filtered out,
  compilation 2.07 seconds, test 0.00 seconds.
- Monitoring limitation for that first check: the temporary monitor's process
  enumeration used `ps`, which the sandbox denied. The Cargo child completed and
  its full passing output was retained, but that invocation has no measured RSS
  or wrapper exit/timing summary. Further builds paused to establish monitoring.
- An attempted `uv run --offline python` psutil probe unexpectedly invoked the
  project build. The probe's tool output reported `Building owl` and no numeric
  result. Later probes use `.venv/bin/python` directly: self RSS is available;
  process-child enumeration via psutil is denied by sandbox sysctl. This is not
  a successful monitored build or test receipt.

Completion/results will be updated after dependencies, attributable red and repair.
