# Task J timing statement — planned, not measured results

Question: for the approved public snapshot path, how much native preparation
time comes from snapshot acquisition, validation and writing both seat rows?
This diagnostic does not measure learner throughput or justify an engine edit.

Inputs are the exact producer recipes for `official:95324500:0` (sparse) and
`dense:0` (241 actors per farm). They remain reproducible even though R1 prevents
a qualified corpus from being installed. Actual source, header and config hashes
will accompany results. No qualified corpus hash may be invented.

Use one live game at a time, one CPU test thread, preallocated output storage,
20 warmups and 200 measured repetitions for each of four phases:

1. Acquire and drop a public `Game::snapshot()`.
2. Validate one existing snapshot through the same private validation function
   that `ObservationGame::prepare()` uses.
3. Write both seats from one already prepared observation.
4. Complete native snapshot, validation and both-seat writing. Do not include
   the test helper's diagnostic `check_row` scan inside this measured phase.

Use `black_box`, verify the final encoded bytes against the untimed baseline,
and check stable output addresses. The discriminating observations are separate
phase wall times, ns/environment and ns/seat, plus complete output byte counts.
Cloning snapshot state still allocates; stable output storage is a separate claim.

Only an optimized build supplies timing evidence. The root release profile is
fat LTO, nonincremental, one codegen unit. Record build time independently from
test-body and phase time. Every build/test shell exports the owner's offline
and thread bounds plus the worktree TMPDIR. The exact timing command is:

```sh
cargo test --release --locked --offline --lib kaggriculture::tests::measure_observe_cost -- --exact --ignored --nocapture
```

Stop the release build at 600 seconds or 1,000,000,000 bytes sampled process-group
RSS; hand the exact command to Claude for the pod if stopped. The diagnostic body
has a separate 120-second bound. Do not replace optimized costs with debug costs,
reduce repetitions, shrink inputs or run a model/trainer/GPU. No latency threshold
or end-to-end speed claim is implied. Task 1.4/6.1 must evaluate representative
complete work before reopening the pinned engine's public access API.
