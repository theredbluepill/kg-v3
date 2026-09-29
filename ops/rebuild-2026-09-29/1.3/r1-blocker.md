# R1 quota feasibility (actual read-only check)

Question: can the fixed 512-state recipe meet every non-synthetic coverage quota?
Inputs: four pinned official gzip fixtures, observation steps 0–31, 344–375,
687–718; exact six profile D/M pairs and observation-corpus-v1 market schedule.
Stopping condition: one streaming scan, no games or reference build.

Command: `python3 ops/rebuild-2026-09-29/1.3/r1_audit.py`.
Result: 384 selected states, maximum 13 actors, zero >16-actor states. Full
counts are in `r1-audit.json`; other required categories already exceed their
quotas except reordered sheds (the brief explicitly allows dense d=31 there).

The native reset starts with no hands (`engine_rs/src/lib.rs:1263`), each
successful HIRE adds exactly one (`lib.rs:3832`), and end_of_day clears every
hand and resets hires_today (`lib.rs:4505`). Step invokes end_of_day before
publishing observations divisible by D (`lib.rs:1477`). The specified market
policy submits q in 0..min(M,2), with HIRE iff `(t+s+q)%8==0`.

At observation n, the upper bound on hands is the count of those HIRE positions
in t=floor(n/D)*D..n-1. Enumerating n=0..95 and both seats yields maximum actor
counts [6,4,2,2,9,4] in profiles 0..5, even assuming unlimited money and every
HIRE succeeds. Affordability can only lower those bounds. Thus the complete
non-synthetic selection can have exactly zero >16-actor states, below the
required four. Dense synthetic states cannot satisfy that quota.

No policy, selection, quota or contract was changed. Per R1, final corpus
qualification/publication must stop until Claude reviews a corrected recipe.
Continue independent encoder work; do not label a missing corpus a semantic red
or invent a full 512-state oracle pass. This also blocks full H and I corpus
checks, independently of Task I's missing shared schema.

Independent read-only review (`review_a`) reproduced every selected-state count
and the seeded bound with a separate turn-by-turn recurrence. It additionally
scanned all 2,880 snapshots / 576,000 tiles for R2 constructor keys, integer/i64
and boolean fields, crop/animal vocabulary, structure compatibility: zero
violations. Shape totals: EMPTY 73,459; LOCKED 210,575; PLANT 212,147; WEED 10,257;
bare PASTURE 4,593; bare COOP 970; COW 38,060; SHEEP 22,619; GOOSE 3,320.
The review reports 0.705 seconds and 13,484,032-byte peak RSS on Python 3.9.6.
This independent inline audit is supporting source evidence; the durable Rust
R2 regression still belongs to Task D and must execute against authored code.
