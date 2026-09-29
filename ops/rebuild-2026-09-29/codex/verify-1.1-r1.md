Verified HEAD **`0dc9bddb5369d988be230b022eff3eb2f363e624`** on `kg/rebuild-codex`.

The requested base currently resolves to `2d9f1cd34a1b23470497c5d255220adc70b6f6df`. Its apparent deletions in the two-dot diff are additions on the diverged base branch; Task 1.1 itself deletes no tracked files.

1. **Retained byte identity — PASS.** Independently read Git blobs from reference `65f0eac5bb00b18a9d3acce319c2a231cbd5dff0` and calculated SHA-256. All 125 reference hashes match the manifest: 12 retained, 113 excluded. These eight retained files are byte-identical:

   | Path within `engine_rs/` | SHA-256 |
   |---|---|
   | `src/py_random.rs` | `40ff0c5e5a79a8ad9d580af4957e028f95ff180603259d8b81be2b357cd73356` |
   | `src/econ_attrib.rs` | `861b7bbaf61ac065bb18797bc389c2acbf6e15c37decaeeeff7eeaeac71e05ac` |
   | `tests/py_random.rs` | `5d844a377db4db7817991c63440e30f7253a5799048e69f1ed8ba82b0beef35a` |
   | `LICENSE` | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |
   | `fixtures/episode-95324500.jsonl.gz` | `47cdfa489b7a80edf8ec1361f2f55cd033c75824d624cd7c9e9daaa3137affd7` |
   | `fixtures/episode-95901360.jsonl.gz` | `e80653f445570a3778a3fb9026a66614b1ecaa2ea417cf8358d3f1850d725281` |
   | `fixtures/episode-95921764.jsonl.gz` | `bc3e01cd12ff70fd2f78bfbe7129d5caca468a6324c124c25c9efebc86fbd3f2` |
   | `fixtures/episode-95990191.jsonl.gz` | `4bf1a3b09c644719c8b36a619289844e52c3458d0e25dc0a0a1429bc6eaa0d1b` |

   Historical provenance bytes remain intact with the trim appendix appended. Cargo changes remove only binary/cdylib targets, Rayon and its six-package dependency closure; remaining dependency versions/checksums are unchanged.

2. **Exactly seven `lib.rs` removals — PASS.** The complete diff contains only:

   ```text
   reference line 19: -pub mod ffi;
   reference line 21: -pub mod joint_matching;
   reference line 22: -pub mod myolie_features;
   reference line 23: -pub mod myolie_sampler;
   reference line 24: -pub mod native_agents;
   reference line 25: -pub mod policy_rows;
   reference line 27: -pub mod training;
   ```

   All are recorded in `VENDORED_FROM.md`. Reference hash: `a33c70ffb1f1ee887b4c51b66d8592dd7cab38f583cd3af06586356e27fd6d84`. Result: **185,626 bytes**, `c4b9bac5057be3a435d2f1035aae17bcd15e7f95ea8557322e4929877c8231fd`. No other differences.

3. **Excluded modules — PASS.** All 113 excluded paths are absent. The package contains exactly 14 non-build files: 12 retained files, manifest and replay test. No excluded-module references occur in retained source or tests; historical/exclusion documentation names them intentionally.

4. **Replay parity — PASS, mutation proven.** The harness constructs `Game` from configuration/seed and executes fixture actions. It compares resulting public/private state, numeric representation, recursive object ordering, statuses, rewards and terminal banks. Four episodes cover **2,876 transitions / 2,880 snapshots**.

   Changed only the first transition’s `expected.step` from **1 → 2** in episode 95324500. The targeted test exited **101**, reporting **0 passed / 1 failed**, with `public state … transition 0`. After restoration it passed **1/1**, and the manifest checker passed again.

   Requested `git checkout --` failed with exit **128** because the sandbox denied creation of `/Users/poonszesen/kg-v3/.git/worktrees/kg-v3-codex/index.lock`. Restored using the pinned HEAD blob instead; exact original fixture bytes and SHA-256 were confirmed.

5. **Checker — core enforcement passes; coverage gaps below.** It verifies the pinned reference, exhaustive inventories, hashes, declared byte reconstruction, exact seven-line trim, exact Cargo manifest changes and preserved historical provenance. Missing inputs and mismatches fail rather than silently passing. The **47 checker tests** meaningfully exercise corruption, undeclared edits, excluded files, malformed schemas and unsafe paths. They do not exercise the complete `check()` entry point.

6. **Formatting/workflow scope — PASS.** Both required Claude edits are reflected: Python/pytest tooling and separate engine preparation invocations. Existing root checks remain unchanged. Rustfmt excludes exactly `engine_rs/src/lib.rs` and `econ_attrib.rs`; three inherited Clippy allowances apply only to the engine command and are explicitly re-denied in authored replay code. Root/engine formatting checks and the configured engine Clippy command passed independently.

7. **Root Cargo / L4 — PASS.** Root `Cargo.toml`, `Cargo.lock` and `src/rules_engine/generation.rs` are byte-identical to the requested base. Feature-tree inspection confirms root `serde_json 1.0.149` has `default/std`; standalone engine `1.0.151` has `arbitrary_precision/preserve_order`. Separate package resolution and invocations soundly isolate L4. Reopen it when introducing a compiled root consumer.

All requested commands ran successfully with offline environment settings:

| Command | Result |
|---|---|
| `cargo test --manifest-path engine_rs/Cargo.toml --locked --offline` | **59 passed**, 0 failed, 0 ignored; 0 doctests |
| `uv run python scripts/check_engine_trim.py` | **PASS**, exit 0: `engine trim manifest: OK` |
| `uv run pytest tests/tools -q` | **49 passed**, 0 failed |
| `cargo test` | **155 passed**, 0 failed, **2 ignored** |

**Blocking findings:** None.

**Non-blocking edits:**

- Harden [the checker](/Users/poonszesen/kg-v3-codex/scripts/check_engine_trim.py:281) and add `check()`-level regression tests. Isolated-copy probes demonstrated that updated manifest declarations can authorize LICENSE changes or remove the trim provenance appendix. An eighth `lib.rs` deletion correctly fails. Committed license/provenance bytes are correct.
- Reconcile [the receipt](/Users/poonszesen/kg-v3-codex/ops/rebuild-2026-09-29/1.1/results.md:5): its opening still says **57 tests/uncommitted**, although its appended review records 59 tests. It also references an absent `replay-counts.json`.

Final `git status --porcelain` is **empty**; tracked and staged diffs are empty. No commits or branches were created. Temporary caches were preserved alongside the [mutation log](/private/tmp/task1.1-independent-verification-4wr_sft2/mutated-replay.log) outside the repository.

VERDICT: APPROVE WITH EDITS