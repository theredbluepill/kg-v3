Task 1.2 is complete, including R1–R5 and all five agreed v4.1 clarifications. Changes remain unstaged for Claude; no Git metadata or retained engine bytes were changed.

1. Changed/added files

The [complete inventory](/Users/poonszesen/kg-v3-grammar/ops/rebuild-2026-09-29/1.2/file-inventory.md) describes all 136 paths individually. Main changes:

- Cargo.toml: promoted existing serde_json through Cargo.
- src/lib.rs and src/kaggriculture.rs: registered the compiled grammar.
- src/kaggriculture/grammar.rs: typed cursor, tables, strict encode/decode.
- src/kaggriculture/grammar_tests.rs: nine shared semantic tests.
- engine_rs/tests/grammar_kernel.rs: nine kernel tests, plus the shared tests.
- engine_rs/TRIM_MANIFEST.json: computed final hash and change registration.
- scripts/check_engine_trim.py: fixed two-test authored allowlist.
- Both tooling test files: inventory and oracle regressions.
- Grammar fixture and manifest: independent reference corpus and custody.
- Contract, RL API, parity docs and plan: clarified APIs, coverage and retirement.
- Cookbook Reference/index/log: adaptation inventory, actual checks and limits.
- ops/1.2/: recorder, harness, preserved controls, logs and receipts.

2. Commands and results

The [88-command ledger](/Users/poonszesen/kg-v3-grammar/ops/rebuild-2026-09-29/1.2/commands.md) includes exact commands, exit codes and test counts.

Final results, all exit 0:

- Root Cargo: 164 passed, 2 existing ignored.
- Engine Cargo: 77 passed, none ignored.
- Root grammar: 9 passed.
- Engine grammar/kernel: 18 passed.
- Tooling pytest: 87 passed.
- Relevant Python suite: 1,045 passed, 3 platform skips.
- rs-prepare, py-prepare, docs-fresh and full just prepare: passed.
- Trim checker, retained-byte guards, staging guard and git diff --check: passed.

Test-first evidence:

- 5.1: constants compile failure → pass; checker 14 failures → 62 passes.
- 5.2: explicit stubs failed; validator controls failed as intended → 25 passes.
- 5.3: transition stub failed → pass.
- 5.4: decoder’s three tests and encoder’s two tests failed against stubs → passed.
- 5.5: omitted HIRE guard failed → restored implementation passed.
- 5.6: renderer controls failed → restored replay/kernel tests passed. BUY_LAND→HIRE failed specifically on bank 2999 versus expected 2000.

3. Oracle facts

- 320 scheduled accepted programs: 256 synthetic, 64 real.
- 64 dense programs; 22 dense/full-ten-order programs at length 252.
- Full markets: 180 synthetic + 8 real.
- Additional controls: 43 rejections + 1 acceptance.
- All 5,752 replay candidates scanned; no replay rejection category occurred.
- Independent rerun reproduced fixture and manifest exactly.
- Eight recorded disagreements: four HIRE-capacity cases, three padding cases, one incomplete-prefix applicability difference.

Compressed: 199,448 bytes, SHA-256:
fa26a81fa21189b5329f04af069c5b70491927d1a9d7200d443cfdd89ec6ebd3

Uncompressed: 2,901,204 bytes, SHA-256:
36d87a61d174959545a85fdf74eef10fc917502420c07b427a8d2a871547542e

4. Kernel acceptance

Both seats pass dense 241-actor execution, successful 240→241 HIRE, subsequent capacity masking, actor-order checks, 140 unit/98 market matrices and meaningful transfer effects. All 256 synthetic programs execute on both seats. All 64 selected replay comparisons preserve snapshots, recursive key order, outcomes and counters.

5. Deviations

- The requested `reference` Python extra does not exist: exit 2 before collection. Identical selection without that extra passed.
- Shared tests needed formatting-compatible expressions for editions 2021/2024.
- A mistyped PLANT test token was diagnosed and corrected before recapturing its control.
- An incidental scratch Python bytecode file was removed; final scratch contains only the authorized FFI append.
- Unavailable Superpowers workflows were replaced by direct brief-driven TDD.

Full explanations are in [results.md](/Users/poonszesen/kg-v3-grammar/ops/rebuild-2026-09-29/1.2/results.md).

6. Open questions and risks

1. No Task 1.2 semantic question remains unresolved.
2. L4 must reopen at the first production engine dependency; retire the temporary include then.
3. Bindings, observation actor-slot integration and actual model sampling/replay remain downstream work.
4. This corpus does not qualify broader BC data or GPU/buffer behavior.
5. Claude owns review and commit. No training, network fetch or performance qualification ran.