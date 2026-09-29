# Task 1.4 bounded implementation

Target: exact reviewed native lifecycle ABI, transactional games/seeds/35 outputs,
terminal timing, reference equality, native grammar exports. Stop each diagnostic
at its first discriminating assertion or the 115 s / 960 MiB watchdog limit.
No training, model, GPU or live environment count above two. Rust tests serial.
Release builds get at most one attempt per requested check, 600 s / 960 MiB.

Inputs: kg/rebuild-env at e197528820ab7cfb429e21259000957370abf1c6;
reference 65f0eac5bb00b18a9d3acce319c2a231cbd5dff0, read-only.
Expected checks (not results): A retirement, B values, C ABI admission,
D rollback/arithmetic, E terminal/truncate/seeds, F native grammar, G oracle/custody.
Actual checks are the paired JSON/log receipts and results.md.

Placement verified: mod.rs is the only module root; lib.rs already registers it.
ObservationGame, ObsStaging and validated serial/parallel views match the merged
interfaces named by the owner. PreparedObservation internals remain private.
Task A already implemented by 1.3: existing retirement regression strengthened,
not duplicated; no missing-behavior red is claimed for existing behavior.
Required superpowers skill names are not installed in the skills catalog or the
searched local skill directories; native subagents and the explicit A–G TDD
instructions supply the workflow without that optional package.
