# Task 1.2 — grammar implementation receipt

Status: complete. All Task 1.2 implementation items and required tests pass; full offline `just prepare` exits 0. Changes remain in the working tree for Claude.

## Scope and custody

Implement the reviewed `ops/rebuild-2026-09-29/briefs/1.2.md`, including Claude's
R1–R5 and the five agreed clarifications. C3 remains a reviewed semantic port.
Reference pin: `65f0eac5bb00b18a9d3acce319c2a231cbd5dff0`. Initial implementation
HEAD was `a88150c84a179b70514445ef0137767987940380`; after a usage interruption,
Claude committed the partial section 5.1 work as `cdd2617`. Codex resumed from
that clean checkpoint. Codex does not stage, commit, or write Git metadata.

The named Superpowers skills were not installed in the available skill roots;
the reviewed task-by-task brief is followed directly. AGENTS.md authorizes native
subagents. Independent checker/oracle/kernel tasks use disjoint files, and
Cargo builds are serialized between agents with two jobs, offline and oracle
test threads limited to one. No training, GPU, model run, network fetch or
throughput claim is part of this change.

Every bounded check is declared before launch in `checks.json` (question,
inputs, expected discriminator and stopping condition). `run_check.py` records
the actual process exit code and combined stdout/stderr in `logs/`. A failed
check is evidence to diagnose, not permission to change its expected behavior.
Read-only source inspection used `git show` on the pinned reference, `cat`,
`sed`, `rg`, Git identity/status reads, and local documentation; these source
reads do not claim execution results.

## Sequential test-first evidence

1. Section 5.1 constants: `constants-red` exits 101 with unresolved grammar
   symbols. Registering the module and exact typed constants/enums yields
   `constants-green`: exit 0, 1 passed. This initial section explicitly uses
   missing-module red as specified, before later error-returning stubs.
2. Section 5.1 checker: `checker-red` exits 1, 14 failed / 48 passed against
   the old one-test inventory. Failures show the intended second path rejected,
   omitted second path accepted, and old error label obstructing downstream
   attack assertions. Only the fixed allowlist and its label change;
   `checker-green` exits 0, 62 passed. `checker-current-inventory` exits 0,
   `engine trim manifest: OK`. The comment-only kernel-test placeholder is
   formatted and hashed; its final implementation hash must replace that value.
3. Section 5.2 recorder: `oracle-validator-red` exits 1 with 10 failures
   against explicit helper stubs. Those failures are fixture-setup failures,
   not evidence that deep validator checks ran. After implementation the suite
   grows to 25 passing tests. A targeted validator-only Err stub fails the valid
   fixture test (1 failed); a no-op validator fails 24 corruption/rejection tests
   (1 valid fixture passes). Both restore exact bytes, then 25 pass. Four initial
   Ruff invocations fail on diagnosed style/typing/import issues; final lint and
   format pass. `oracle-record-final` and `oracle-verify-final` exit 0 and
   independently reproduce all fixture and manifest bytes. No new grammar is
   called for expectations. The full oracle receipt is `oracle/README.md`.
4. Section 5.3: `transitions-red` exits 101 with the named table/reachable-state
   test failing against explicit error stubs; `transitions-green` exits 0,
   1 passed. All 964 reference-derived bits and independently literal supports
   agree. Exploration visits every admitted/rejected local edge for the stated
   shape equivalence classes, tests all A=1..241 ordinals and all market high
   digits. These are support-class checks, not all possible A/O/H combinations.
   A changes readiness/ordinal/HIRE inequality; O changes queue availability.
5. Section 5.4 decoder: `decode-red` exits 101, 3 failed / 2 passed; the explicit
   decoder error stub fails exact commands, malformed transport and fixture
   tests. `decode-green` exits 0, 5 passed. The 140 unit and 98 market cases are
   constructed from independent literal command matrices. An in-memory expected
   action mutation is rejected without changing frozen fixture bytes.
6. Section 5.4 encoder: `encode-red` exits 101, 2 failed against an explicit Err
   stub. `encode-green` exits 0, 2 passed. Full 3,024-token round trips include
   all 321 admitted fixture/control rows and 64 verbatim replay actions. Invalid
   keys/types/names/arities/quantities/queues/HIRE capacity leave output untouched.
7. Section 5.5: prior transitions already implement capacity. The required
   omitted-HIRE-guard control fails (exit 101, 1 failed), then exact restoration
   passes (exit 0, 1 passed). Each of five budgets enumerates 14³ coupled races;
   every canonical queue key/mass and both mass totals agree within 1e-12 with
   full-frame Rust-cursor sequential branching. A17/H16, A240/H241 and A241/H241
   boundaries and HIRE increment at slot11 are also checked.
8. Section 5.6 cursor replay: prior rendering is complete. The first negative
   and restored attempts both exit 101 before the intended discriminator because
   the new test mistyped PLANT as token17 (CARE). Source inspection of the pinned
   enum establishes PLANT8; only that input is corrected, not its expected action.
   `replay-walk-negative-corrected` then fails exact JSON equality (PASS vs PLANT)
   after a successful cursor walk. Exact source restoration passes
   `replay-walk-green-corrected` (1 passed), including all321 admitted fixture rows
   and22 dense/full-ten-order programs.
9. Section 5.6 kernel: tests are authored after section5.4, so the included
   decoder is already implemented. `kernel-initial` passes1. The required
   BUY_LAND→HIRE renderer mutation fails the bank-effect assertion first:
   actual2999 versus expected2000 (exit101,1failed). Restoration passes the
   whole integration binary,18passed. Before bytes and patches are retained
   under controls/; no retained kernel source is mutated.
10. Final read-only review finds a missing-sentinel coverage gap: earlier cases
    fail length/support before reaching the final cursor guard. Add valid-length
    PASS+HIRE with no sentinel; removing the final guard now fails the named
    malformed test (exit101,1failed), restoring it passes1. No production semantic
    change or weakened expectation is needed. A subsequent preparation check
    catches edition2021/2024 formatter differences in shared tests. Qualifying
    JSON macros and naming intermediate error/budget results makes both formatters
    agree without new exclusions or semantic changes.

## Oracle facts

- Scheduled accepted:320 =256 synthetic +64 real replay programs. All three
  reference decoders agree on every scheduled canonical action.
- Dense241:64 synthetic, zero real. Dense/full-ten-order/length252:22
  (16 schedule-forced +6 incidental). Full market:188 =180 synthetic +8 real;
  “full” here means the shape's O orders, including O=1.
- Additional controls:44 =43 v4 rejections +1 zero-padding acceptance. All
  5,752 replay seat candidates were scanned,1,438 per episode. No codec rejection
  or incomplete-layout category occurred; the seven absent categories are
  explicit in the manifest, so no rejected-replay action was fabricated.
- Compressed fixture:199,448 bytes; SHA-256
  `fa26a81fa21189b5329f04af069c5b70491927d1a9d7200d443cfdd89ec6ebd3`.
- Decompressed JSONL:2,901,204 bytes; SHA-256
  `36d87a61d174959545a85fdf74eef10fc917502420c07b427a8d2a871547542e`.
- Manifest:34,874 bytes; SHA-256
  `ac2e44f5f8578926a4f2a91932c35e4634d945b95192518d4e2d1c92d62c45ae`.
  All are within reviewed limits. Independent rerun reproduces fixture and
  manifest exactly, without overwriting committed expected data.
- Eight recorded differences: four capacity cases rejected by sampler/Python
  but accepted by training FFI; three padding mutations accepted as prefixes by
  the historical decoders but rejected by full-buffer v4; one incomplete
  missing-STOP prefix rejected by sampler with FFI/Python explicitly inapplicable.
  Every original verdict/error is preserved, not normalized to the new decoder.

## Kernel acceptance facts

All new kernel cases run on both seats where applicable. Explicit states are
constructed only through serialized official headers and public Game::from_header;
private fields and retained bytes are untouched.

- A241 with ten EMPTY orders and distinct STOP uses exactly252 frames, steps
  successfully and counts241 PASS commands; actor/private inventory count remains241.
- A240 plus one affordable HIRE steps to exactly241 actors/inventories and bank2999.
  The resulting plan masks HIRE at all ten market positions and forced sentinel.
  Encoding two HIREs at A240 rejects with `hire capacity`, preserving output.
- Farmer NORTH and three hands SOUTH/EAST/WEST yield positions [4,3], [4,5],
  [5,4], [3,4], proving frame/ordinal/engine order. Observation actor_slot mapping
  remains the Task1.3 side of this documented invariant.
- 140 unit and98 market cases per seat compare independently expected canonical
  JSON and exact direct-versus-decoded execution. Twenty meaningful PICKUP/PLACE
  cases exercise absent/1/31/32/1023 inventory effects. Market0/EMPTY no-ops retain
  syntax/order. Rich/poor games have equal plans and per-token masks; two admitted
  HIREs execute twice with funds and zero times without funds.
- All256 synthetic fixture programs additionally step each seat against recorded
  expected JSON and matching direct execution (512 comparisons).
- All64 selected real replay seat actions compare direct joint actions against
  the same actual seeded state with one seat replaced by new decode. Public/private
  snapshots, recursive object order, statuses/rewards/done/step, economic/attribution
  counters and terminal banks match. Grouping by four episodes avoids per-sample
  resets. These64 codec comparisons are distinct from the retained four full
  replay tests (2,876 transitions /2,880 snapshots).

## Deviations, diagnosis and review

1. Named Superpowers skills are unavailable; direct brief TDD replaces that tool
   workflow. Required tests/stubs/controls and receipts are retained.
2. Section5.2's original stub failures were in helper setup, so targeted validator
   Err/no-op controls were added before claiming validation discrimination.
3. Later tests reuse already complete production bodies as the brief permits;
   capacity, rendering and final-sentinel controls provide real red evidence.
4. Initial reference Python import generated one ignored bytecode cache. Only
   that generated file and empty directory were removed; sys.dont_write_bytecode
   prevents recurrence. Final scratch audits show only the authorized FFI append,
   with empty untracked/ignored inventories. No Git metadata/worktree operations
   or network fetch occurred. This incidental scratch write is explicitly recorded.
5. `uv run --offline --extra reference ...` exits2 before collection: the clean-base
   pyproject defines only flash-attn, not reference. The identical pytest selection
   without the nonexistent extra passes1,045 with3 platform skips. No dependency
   is added to make the command pass.
6. Shared tests initially format differently across editions; expressions/imports
   were rewritten to a common formatted form. Existing pinned-source exclusions
   and engine-only allowances remain exactly unchanged, and authored engine code
   re-denies the three allowed inherited Clippy lints.
7. One read-only oracle inspection initially treated the schema's source string
   as an object and exited1 TypeError; corrected inspection used the documented
   string discriminator. No expected data changed. Routine cat/sed/rg/git-show
   source reads are not semantic checks; bounded validation commands are logged.

Independent native agents reviewed grammar semantics and oracle custody. The
only concrete grammar findings (PLANT test typo and missing final-sentinel case)
were diagnosed and closed with discriminating reruns. Kernel test source was
reviewed against the required matrices/effects/dense/replay checks. The PR
checklist is applied to source/fixture scope, shared production logic, docs and
required preparation; no PR, stage, commit or merge is performed here.

## Changed-path inventory

`file-inventory.md` gives every changed/added path relative to pre-task a88150c,
including the partial cdd2617 commit and current worktree additions. Main files:

- Cargo.toml: existing serde_json promoted from dev to normal by Cargo. Cargo.lock
  stays unchanged; the separate engine metadata/lockfile stay byte-identical.
- src/lib.rs and src/kaggriculture.rs: compile/register the new grammar.
- src/kaggriculture/grammar.rs: typed plan/cursor, tables and strict encode/decode.
- src/kaggriculture/grammar_tests.rs: nine shared test-first semantic tests.
- engine_rs/tests/grammar_kernel.rs: nine kernel acceptance/comparator tests plus
  the shared grammar source/tests under the engine package.
- engine_rs/TRIM_MANIFEST.json: exact authored final SHA and non-engine inventory.
- scripts/check_engine_trim.py: fixed authored allowlist exactly two tests.
- tests/tools/test_check_engine_trim.py: test-first second-file/hash/omission attacks.
- tests/tools/test_record_grammar_reference.py: strict independent oracle tooling
  regressions (25 tests).
- tests/fixtures/kaggriculture/grammar-v4-reference.jsonl.gz and matching manifest:
  bounded source-pinned independent corpus and custody/disagreement inventory.
- docs/rl-api-specs.md: typed APIs/tables, admission, bindings and retirement boundary.
- docs/rules-parity-coverage.md: current coverage, denominators and limits.
- docs/kaggriculture-contract.md: agreed five v4.1 clarifications, no semantic change.
- ops/rebuild-2026-09-29/plan.md: correct source location and include retirement.
- Cookbook native semantics Reference, references/index.md and prepended log.md:
  one coherent adaptation inventory with actual checks and remaining gaps.
- ops/rebuild-2026-09-29/1.2/: receipt/check runner, mutation helper, declared checks,
  complete command logs, preserved stubs/controls, recorder/harness/original-source
  custody and oracle receipt. Scratch request/response files are intentionally ignored.

## Full validation commands

All 88 declared checks' exact argv, environment, question, inputs, expected
discriminator, stopping condition, actual exit code and log path is in
`checks.json`; `commands.md` renders those commands/results, including failures
and all preparation subcommand summaries. Logs preserve unmasked native output.
All Cargo/just invocations inherit CARGO_BUILD_JOBS=2, CARGO_NET_OFFLINE=true and
UV_OFFLINE=true. Oracle tests use one thread. No output is piped to mask failures.

Final commands pass: root Cargo **164 passed, 0 failed, 2 existing ignored**;
engine Cargo **77 passed, 0 failed, 0 ignored** (41 library +18 grammar/kernel
+9 RNG +9 replay), plus0 doctests. Root named grammar selection passes9, engine
selection passes18. Combined tooling pytest passes87. Relevant Python selection,
py-prepare and full prepare each report **1,045 passed, 3 platform skips**; the
skips are two unavailable flash-attn CUDA cases and one unavailable x86 quantized
backend. All1048 Python cases are collected, with no deselections.

`rs-prepare-common-format`, `py-prepare-final`, `docs-fresh-final`,
`trim-after-prepare` and `prepare-final` all exit0. The full prepare is the actual
requested command (`CARGO_BUILD_JOBS=2 uvx --offline --from rust-just just prepare`),
not a fallback. It builds the extension, formats/lints both Rust packages,
formats/lints/type-checks Python (51 mypy sources), runs full Rust/Python tests,
checks docs lint/freshness and the engine manifest. The authored test's final
formatted SHA-256 is
`416ed73aaa707bce0994bbdd46d97f9887a3990a5945304e365305051c5ba8c5`, directly
logged by shasum and computed into the manifest. Final guards pass (exit0) against both HEAD and pre-task a88150c retained
bytes, including license and fixtures; staged diff is empty and git diff --check
is clean. The final documentation lint also exits0 after verification-count edits.

Feature graphs are separately recorded: root serde_json1.0.149 has default/std
only; engine serde_json1.0.151 retains arbitrary_precision, preserve_order and
indexmap. No production root→engine edge or feature unification was introduced.



## Remaining qualification boundaries

L4 stays deferred to the first production root → engine dependency; the same
source is temporarily compiled in two separately resolved packages. The engine
test include is retired at that dependency edge in favor of root integration.
L6's historical failure is the Inductor GEMM overflow; checked i64 token
admission is a separate hazard. Native batch transactions, PyO3 buffers, device
table upload, actual model sampling/replay and GPU behavior remain later tasks.

## Open questions and unresolved risks

1. All five brief questions are resolved by Claude's reviewed recommendations
   and Codex agreement, recorded as v4.1 clarifications. No Task1.2 semantic
   decision remains for Claude. Claude still owns committing this un-staged work
   and reviewing the cross-stream handoff.
2. L4 remains intentionally deferred: the first production root→engine edge in
   1.3/1.4 must retire the temporary include/allowlist entry, move tests to root,
   preserve numeric features and recheck the float fixture's feature interaction.
3. The Task2.3 v3 table contract and zero-padding rule were read and match these
   eight tables. Actual bindings/device upload/model sampling and log-density
   replay still require their own integration tests; Claude's head implementation
   must retain the agreed post-STOP zeros. Native length0 must stay rejected at1.4.
4. The oracle is four known replay worlds plus synthetic support coverage.
   Zero rejected replay candidates do not prove the broader BC corpus is clean;
   admission/rejection denominators must be remeasured there. Dense states are
   explicit synthetic states, not claims about naturally reached games.
5. CPU syntax/kernel checks do not qualify native batch atomicity, PyO3 buffer
   lifetime, CUDA/BF16/model behavior, learning strength or the distinct L6 GEMM
   overflow repair. No training, GPU/network operation or throughput claim ran.

## Claude review (2026-09-29, at `75169be`)

Reviewer: Claude, against `briefs/1.2.md` (R1–R5, questions 1–5), contract
v4.1 and plan C3. Every changed path in `git diff kg/isaiah-gap-closure...HEAD`
was read. Verdict: **approve; no production defect found.**

- `grammar.rs` matches reference `myolie_sampler.rs` `allows`/`advance`
  slot by slot (including Product/Animal/Sell → Seed, the high-zero low digit
  rule, `A + h < H` and HIRE increment only at slot 11). It adds the checked
  i64 transport, zero-padding admission, strict encoder and table extraction
  specified in §2. No `crate::` path, `unsafe`, `unwrap` or `#[allow]`.
- Contract v4.1, `rl-api-specs.md`, `rules-parity-coverage.md` and the plan
  text agree with the implemented signatures and admission order.
- One mutation per oracle, then exact byte restore (`claude_review_mutations.py`,
  `logs/claude-review/`): 20 of 20 controls make their named test fail at
  an assertion, not at compilation. They cover all nine shared grammar tests,
  all nine kernel tests, the trim-checker tests and the recorder validator.
  Each restore is SHA-256 checked. The kernel BUY_LAND→HIRE control fails on
  the bank assertion (line 222) before JSON equality.
- `record_reference.py verify` against the scratch reference reproduced the
  fixture (compressed `fa26a81f…`, uncompressed `36d87a61…`) and every count
  (`logs/claude-review/oracle-verify.log`, exit 0).
- The brief's §6 Python command named a nonexistent `reference` extra. It is
  corrected in place to the selection Codex actually ran.
- After these edits, `CARGO_BUILD_JOBS=2 OMP_NUM_THREADS=2 uvx --from rust-just just prepare`
  exits 0 (`logs/claude-review/prepare.log`): root Rust 164 passed / 2 ignored;
  engine 41 + 18 + 9 + 9 passed; Python 1,045 passed / 3 skipped; trim
  checker OK; docs fresh. `git diff --check` is clean and pinned engine bytes
  are unchanged.

Residual risks (not Task 1.2 defects): the four traces contain no replay
codec rejection, so the encoder's rejection categories are checked only on
synthetic inputs; Task 5.1 must measure its own admission table. Most kernel
tests assert canonical JSON before effects, so a renderer fault fails there;
the effect assertions are independently discriminating only where a control
shows it (BUY_LAND bank effect). L4, bindings, device tables and GPU behavior
remain later tasks, as listed above.
