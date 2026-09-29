# Independent Task 1.2 review

Reviewed branch `kg/rebuild-grammar`, HEAD `7877c46e39687e4332fe3ad5fc65d3c76feb2125`.
Scope: `git diff kg/isaiah-gap-closure...HEAD`; merge base `90ed86c817051d657c8ea0ab596b0eecb56b5d53`.
Authority: Task 1.2 brief, contract v4 with expressly agreed v4.1 clarifications, and rebuild plan Task 1.2.
Reference: `65f0eac5bb00b18a9d3acce319c2a231cbd5dff0`.

No actionable findings. Severity/location/fix: not applicable.

Semantics inspected against reference sampler, codec, tables and kernel: slot supports and transitions, actor order, quantity omission and zero, EMPTY, final STOP, exclusive submitted-HIRE count, strict signed transport, zero padding and failure nonmutation. Docs/cookbook claims and deferred limits match implementation/evidence. All 12 retained engine files match the three-dot base; retained/excluded inventories and reference pin are unchanged.

| Check | Result | Evidence |
|---|---|---|
| `cargo test --manifest-path engine_rs/Cargo.toml --locked` | 77 passed, 0 ignored: 41 library + 18 grammar/kernel + 9 RNG + 9 replay | engine.log |
| `uv run python scripts/check_engine_trim.py` | engine trim manifest: OK | trim.log |
| `uv run pytest tests/kaggriculture -q` | exit 4, 0 tests; directory absent on this Rust-only grammar branch | kaggriculture-python.log |
| `cargo test --locked` | 164 passed, 2 existing ignores (155 retained Orbit + 9 new grammar) | root.log |
| `uv run pytest tests/owl tests/scripts tests/tools -m 'not slow' -q` | 1045 passed, 3 platform skips | python-regression.log |
| `uv run pytest tests/tools/test_check_engine_trim.py tests/tools/test_record_grammar_reference.py -q` | 87 passed (included in total above) | task12-python-tooling.log |
| `record_reference.py verify` with committed fixture/manifest and pinned scratch reference | exact compressed fixture and manifest equality; 320 scheduled programs, 44 controls, 64 real and 64 dense programs | oracle-verify.log |
| restored `just rs-prepare` | pass: formatting, both Clippy graphs, root 164/2, engine 77/0, trim and doc freshness | rs-prepare-restored-retry.log |

All builds used CARGO_BUILD_JOBS=2 and offline execution. The first uvx rs-prepare launch failed because its default tool directory was outside writable roots; retry used UV_TOOL_DIR under this receipt directory and passed. Doc freshness is a clean-worktree check, not independent evidence of the three-dot documentation review.

Mutation: changed only `src/kaggriculture/grammar.rs` HIRE guard from `<` to `<=`. Exact fixture-oracle test failed (exit 101; 0 passed/1 failed) at `grammar_tests.rs:1116`, admitting `mutation-hire-capacity-1-1`. Restored in a finally block. Before/restored SHA-256: `5b660b560731b15b746795fec3875eed911943038b3cea3d45ca912032a0426d`. Restored full Rust suites pass. See mutation.json, mutation-red.log and grammar.before.rs.

Nonblocking whitespace observation: three-dot `git diff --check` reports trailing whitespace in preserved raw failure logs/patch evidence and a final blank line in oracle/README.md; the recorded historical plain `git diff --check` only covered the then-working diff. These are not semantic defects; retained raw evidence was not rewritten.

Limits: no native Python binding/GPU/training qualification; root-engine feature unification (L4) remains deferred. Replay-rejection categories remain synthetic-only because all 5752 scanned real candidates were admitted. Dense coverage is synthetic. Missing tests/kaggriculture is explicitly reported, not counted as passing.

Final `git status --porcelain=v1`, unstaged diff and staged diff are empty. No tracked modification remains. Verification receipts are outside the repository.

VERDICT: APPROVE
