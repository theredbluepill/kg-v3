# Task J — snapshot control, optimized-build stop and documentation

Planned workload and stopping conditions were recorded first in `timing-plan.md`.
The exactly-one-snapshot test was written before its acquisition seam. The seam
was deliberately called twice for the required negative control: `j1` fails
with left=2/right=1 (one test failed, 237 filtered). Removing the extra call makes
`j2` pass. Sparse official initial state and 241-actor dense state each reuse all
29 allocations through four poisoned writes, compare all 282,246 environment
bytes and run the diagnostic row validator. No mutation remains.

The ignored `measure_observe_cost` test is compiled but requires release mode at
runtime. It uses one game at a time, 20 warmups/200 measured repetitions for
snapshot, validation of an existing snapshot, prepared writing and complete
both-seat encoding. It excludes diagnostic check_row scans from measured work,
checks stable bytes/allocations, and reports source/header/config hashes, costs
per environment/seat and byte counts. Black-box/deadline/test-counter overhead
is explicitly included. It does not measure training or GPU throughput.

Actual `j5` command:

```sh
cargo test --release --locked --offline --lib kaggriculture::tests::measure_observe_cost -- --exact --ignored --nocapture
```

The fat-LTO, one-codegen-unit build at CARGO_BUILD_JOBS=2 is killed at the first
sample above 1,000,000,000 bytes: **1,052,393,472 bytes**, **53.8096 seconds**,
child exit -9 (wrapper exit 137). Build did not finish; test body and phase costs
are unexecuted. `timing.json` records this actual stop and the **same command for
Claude to run on the pod**. No second optimized attempt or debug timing was run.
The separate CPU-brand query is sandbox-denied; platform/machine are recorded,
and no CPU brand is inferred from historical notes.

`j3` formatting passes. `j4` Clippy catches a constant profile assertion; wrapping
the runtime profile guard in black_box resolves it, and `j6` all-target Clippy
passes. This startup-guard-only edit follows the stopped build; the receipt pins
the final source separately. No measured operation changed and no costs exist
to attribute to either version. Aggregate final tests exercise the restored
snapshot control again.

The API/rules/parity docs now describe the implemented boundary, L4 repair,
observation-vs-rules coverage, schema/corpus blocks and measured build limit.
The cookbook Reference/index/log were reconciled together; the original dirty
note and checksum are preserved as `cookbook-before-closeout.*`. Final command
outcomes and full changed-file inventory belong in `results.md` and
`files-changed.md`. Task I and full Task 1.3 completion remain blocked.
