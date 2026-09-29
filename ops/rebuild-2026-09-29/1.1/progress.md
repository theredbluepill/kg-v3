# Task 1.1 execution ledger

Plan: `ops/rebuild-2026-09-29/briefs/1.1-rules-kernel.md` with
`1.1-review-claude.md` and the owner's implementation request.

Target: exact three-file rules kernel, exhaustive provenance, direct four-episode
replay parity, unchanged 155-pass/2-ignore root Rust suite, green Python and prepare.
No training, network, Git writes, or commit. All Cargo work uses three build jobs.

Preflight: branch is `kg/rebuild-codex`; three pre-existing dirty cookbook files
are preserved byte-exact under `before/` with baseline SHA-256 values.
The unchanged root test suite passes. Baseline tool/feature identity is recorded.

| Step/interface | Review |
| --- | --- |
| A checker/import | Python/pytest replaces Node under Claude's review; checker must fail before manifest exists. |
| A/B manifest/replay | Refresh authored replay hash only after formatting the new file; retained bytes stay pinned. |
| B replay/comparators | Run the broken private-order comparator red before the full comparator and replay pass. |
| A/C package/preparation | Separate manifest-path invocations preserve feature isolation; root prepare must include engine checks too. |
| C documentation/results | Update existing restart Decision, index and log; preserve older reference scope. |

Implementation choices: `non_engine_changes` records path/reason inventory in the
manifest, as Claude requests. Receipt files have no manifest hashes to avoid a
self-referential checksum cycle. The existing worktree is used as directed.

- A: complete; manifest passes, 41 retained library + 9 RNG tests pass.
- B: complete; broken comparator 0 passed/1 failed, corrected comparators 2 passed;
  corrupted replay snapshot 0 passed/1 failed, restored first episode passes;
  complete engine 57 passed/0 ignored, 2,876 transitions and 2,880 snapshots.
- C: complete; independent review clean after two fixes; final engine 57/0,
  checker passes, prepare root 155/2 ignored, engine 57/0, Python 769/3 skipped.
  Final documentation reconciles actual results; no commit or Git write.

Deviations supported by failed checks: raw cargo fmt --check proposes changes in
exactly lib.rs/econ_attrib.rs; raw Clippy reports 1 too_many_arguments,
4 collapsible_if, and 1 needless_range_loop (inline test). Preserve the hard
byte-level source contract: ignore only those two file paths in rustfmt.toml,
allow only those three Clippy names for the engine command, and re-deny them in
the authored replay crate. rs-format also runs the manifest checker so drift in
ignored source fails preparation. Receipts retain both raw failures. Reviewer
agrees the failures are style-only and asks the final authored inventory be exact.

The initial extraction of Appendix C matched too broadly and produced a syntax
error; the corrected receipt generator extracts by its Appendix C heading.
No rules bytes changed in that repair. Parallel materialization ran only after
checker missing-module red; missing-manifest red still ran before manifest creation.

