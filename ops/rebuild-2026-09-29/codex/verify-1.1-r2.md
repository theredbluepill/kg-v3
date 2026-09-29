Verified HEAD **`5378763f69c9d39be0d6a0423efe33b2e615d44e`**, branch `kg/rebuild-codex`.

1. **Retained bytes — PASS.** Independently read Git blobs at reference `65f0eac5bb00b18a9d3acce319c2a231cbd5dff0`. All 125 manifest reference hashes match: 12 retained, 113 excluded. All eight unchanged files match byte-for-byte:

   | File | SHA-256 prefix |
   |---|---|
   | `src/py_random.rs` | `40ff0c5e5a79a8ad` |
   | `src/econ_attrib.rs` | `861b7bbaf61ac065` |
   | `tests/py_random.rs` | `5d844a377db4db78` |
   | `LICENSE` | `c71d239df91726fc` |
   | Episode 95324500 | `47cdfa489b7a80edf` |
   | Episode 95901360 | `e80653f445570a37` |
   | Episode 95921764 | `bc3e01cd12ff70fd` |
   | Episode 95990191 | `4bf1a3b09c644719` |

   [Full independently verified hashes](/private/tmp/task-1.1-verify-r2.1RkAyN/verified-hashes.json). Historical provenance is an exact prefix; its appended trim text exactly matches the brief.

2. **Seven `lib.rs` removals — PASS.** The complete diff removes only reference lines **19, 21–25, 27**: `ffi`, `joint_matching`, `myolie_features`, `myolie_sampler`, `native_agents`, `policy_rows`, and `training` declarations. No additions or other differences. Result: **185,626 bytes**, SHA-256 `c4b9bac5057be3a435d2f1035aae17bcd15e7f95ea8557322e4929877c8231fd`. All removals are recorded in provenance.

3. **Excluded modules — PASS.** All 113 excluded paths are absent. No excluded-module references remain in engine sources, tests, or Cargo dependencies. Inventory/provenance documents retain explanatory mentions.

4. **Replay strength — PASS.** The harness constructs `Game::new(config, seed, 2)`, executes fixture actions, and compares actual public/private state, recursive object-key order, statuses, rewards, terminal banks, and completion. Four fixtures exercise **2,876 transitions / 2,880 snapshots**.

   I changed episode 95324500’s first transition `expected.step` from **1 → 2**. The episode failed at transition 0: **0 passed, 1 failed**, exit 101. After restoration: **1 passed, 0 failed**. [Failure evidence](/private/tmp/task-1.1-verify-r2.1RkAyN/mutation-red.log).

   Requested `git checkout -- …` failed because the sandbox denied creating `/Users/poonszesen/kg-v3/.git/worktrees/kg-v3-codex/index.lock`. I restored the exact `git show HEAD:<path>` bytes instead and verified the original full hash.

5. **Checker and regressions — PASS.** The checker enforces exhaustive inventories, fixed retained/authored sets, hashes, exact edit reconstruction, seven removals, immutable retained files, restricted Cargo changes, and pinned append-only provenance. Invalid inputs fail nonzero. Independent temporary-copy tests disabling the license, lockfile, and provenance guards produced **3 failed / 52 deselected**, proving those regressions detect missing enforcement. The non-engine inventory is descriptive; root-file identity was checked separately.

6. **Formatting/workflow scope — PASS.** Claude’s required Python/pytest conversion and separate engine invocations are present. Root commands remain unchanged. Rustfmt excludes only `lib.rs` and `econ_attrib.rs`; both root and engine formatting checks pass. Strict engine Clippy independently reproduced **six inherited findings** across three lint categories. The scoped allowances pass and are explicitly re-denied in authored replay tests.

7. **Root Cargo / L4 — PASS.** Root `Cargo.toml`, `Cargo.lock`, and `generation.rs` match both the named base and Isaiah’s `32b3ec9` byte-for-byte. Cargo metadata confirms an independent engine workspace. Root Serde JSON enables only default/std; engine resolution retains arbitrary precision and insertion order. This soundly isolates L4 for Task 1.1; integration must revisit it. The engine lockfile removes only Rayon’s six-package closure and dependency edge, preserving remaining versions/checksums.

All requested commands ran successfully with offline environment settings:

| Command | Result |
|---|---|
| `cargo test --manifest-path engine_rs/Cargo.toml --locked --offline` | **59 passed, 0 failed, 0 ignored** |
| `uv run python scripts/check_engine_trim.py` | **PASS**, exit 0 |
| `uv run pytest tests/tools -q` | **57 passed, 0 failed** |
| `cargo test` | **155 passed, 0 failed, 2 ignored** |

**Blocking findings:** None.

**Non-blocking findings:** The named base currently resolves to `1177a5b`; merge-base is `ab0de01`. Consequently, the requested two-dot diff includes reversals of base-only model/trainer/documentation work. Those paths are unchanged on this branch relative to the merge-base. Replay qualification remains limited to the four pinned episodes plus unit/RNG coverage.

Final `git status --porcelain` is **empty**. Temporary output and the pre-existing Xcode cache were preserved outside the repository. No tracked changes, commits, or branches were left.

VERDICT: APPROVE