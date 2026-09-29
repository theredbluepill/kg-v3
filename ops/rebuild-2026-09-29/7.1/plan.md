# Task 7.1 bounded import check

Input: integration `b8747b6e8acece5f561d09a75bb914364a60ac05`, reference
`65f0eac5bb00b18a9d3acce319c2a231cbd5dff0`, the four byte-exact controllers,
and the frozen `engine_rs` crate. The requested placement is a standalone
edition-2024 crate with explicit engine re-exports.

Question: can the imported controllers compile against the frozen engine's
public API? A compiler error naming a required private item or absent method
discriminates an API boundary defect from dependency/cache or controller
behaviour problems. Run an isolated compile probe before implementing the
registry. Stop this path on an inaccessible engine item, as the task explicitly
requires; do not change engine visibility, copy accessors or substitute a Game.

No opponent game, training, GPU, model diagnostic or panel run is planned for
this admission check. Shells use the task's offline/thread/TMPDIR exports. The
probe is under `.codex-tmp`; compact source/hash/diagnostic receipts stay here.
The required final commands will record actual exit statuses, including absent
opponent artifacts after the stop. No missing file will be created merely to
turn those commands green. Existing regression suites are distinct from new
opponent qualification.

Completion: exact API blockers and source hashes are recorded, the frozen
engine is verified unchanged apart from permitted manifest bookkeeping, and
the coverage document and cookbook state what remains unimplemented.
