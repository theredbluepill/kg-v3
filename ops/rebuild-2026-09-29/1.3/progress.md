# Task 1.3 execution ledger

Plan: `ops/rebuild-2026-09-29/briefs/1.3.md`, including Claude R1–R5/Q1–Q6.
Branch checked: `kg/rebuild-observe`; initial tracked/untracked status clean.

## Planned expectations (not results)

Execute A→B→C→D→E→F→G→H→I→J, test first at each step.
A: pre-unification pass, post-unification decimal failure, Number repair green.
B: checked typed buffers. C: checked config/hire costs. D: strict tiles.
E: all actors/private ranks. F: complete context/check_row/transactions.
G: 512-state source-bound oracle, non-synthetic R1 quotas, ≤8 MiB compressed.
H: bitwise 8,176-offset reconstruction plus added-fact mutations.
I: implement native binding/tests/stub; real schema checks blocked on 2.1 merge.
J: snapshot counter mutation, optimized four-phase cost, docs/cookbook/final checks.

## Constraints and preflight

- User's explicit implementation scope supersedes the historical brief-only wording.
- User requires working-tree handoff; commit/rebase/merge checkpoints are not executed.
- Config wrapper→prepared snapshot→typed seat/batch buffers is the common B–F/I/J interface.
- G emits strict records consumed by H/I/J; no reduced quotas or missing-fixture skips.
- I imports the actual Task 2.1 schema; no substitute/fallback schema is authored.
- C/D/F validate before publication; E privacy checks require changed own-row controls.
- G's reference crate is exported into temporary storage; vendored kernel stays byte-exact.
- J's optimized-build budget is 600 seconds/1 GB; other diagnostics use 120 seconds/1 GB.
- No policy/model/PPO/reward/grammar code, training, GPU, network or Git mutations.
- Skill execution adapted to existing isolated checkout and user-required ops receipts.

## Actual progress

- Native I implementation passes 55 independent NumPy checks, Clippy, Ruff and
  stub mypy. Actual-schema collection remains blocked (task-i.md); no substitute
  schema is present. G source-custody review adds nine semantic reds then greens;
  final custody suite has 43 passes. Recipe/quotas remain unchanged.
- J counter red observes two snapshots vs one; restored single acquisition and
  stable sparse/dense output allocations pass. Release timing build hits the
  1-GB RSS stop after 53.81 seconds; no test body/phase costs run. Exact pod command
  is in timing.json/task-j.md. Documentation and cookbook adaptation are updated;
  all required aggregate commands follow, with actual results still to record.

- H implementation and independent controls are now available: eight reconstruction
  controls and five added-fact tests pass; all 16 temporary mutations fail at the
  intended assertions and restoration passes. The stream comparator remains a
  nonignored failure because no R1-qualified corpus exists. `task-h.md` separates
  hand-derived checks from the unexecuted 512-record reference comparison.
- I first tests are written against the actual (absent) Task 2.1 schema, with no
  fallback. A separate explicit-NumPy smoke captures the missing-binding
  AttributeError before implementation. Missing-schema collection is a dependency
  failure, not a semantic TDD red. Binding implementation is underway.

- Read complete reviewed brief, v4 contract, API spec, plan, CLAUDE.md, current rule docs,
  cookbook index/latest log, governing restart/scope/stateless/native-buffer notes.
- A delegated to native subagent `task_a`, serial implementation. Read-only independent
  source audit examines R1/R2/legacy constraints; no concurrent builds authorized.

- Recovery at committed `32e2cdd`: only A pre-dependency test existed. A now green:
  157 root passes/2 ignored; trim OK. Detailed receipts in task-a.md.
- B in progress, delegated `buffers_b`; no concurrent builds.
- R1 audit: 384 selected official records have maximum 13 actors and 0 >16 states.
  Fixed seeded policy has actor upper bounds [6,4,2,2,9,4] before daily hand reset.
  Required ≥4 non-synthetic >16 states is impossible without reviewed recipe changes.
  Preserve quota and stop final corpus installation; continue independent encoder work.
  Exact counts/script: r1-audit.json and r1_audit.py. Task I schema still absent.
- Necessary Task C API correction: pinned PyInt has no Display implementation
  (only PyIntError does). Its public Serialize implementation emits exact BigInt
  decimal as a JSON Number. Use constructor-only serde_json::to_string(&PyInt)
  then checked parse::<i64>(); no float intermediate, Debug parsing, new accessor,
  vendored edit or live-state serialization. Wrong ruling would affect config
  admission only; tests include i64 extremes and 2^63 rejection.

- B typed-buffer slice green: 11 tests, formatting and all-target Clippy pass;
  semantic length-check mutation fails and is restored. Independent review has no
  blocking B finding. C owns config/wrapper tests and implementation next.
- C green: nine config and four hire tests; combined B/C 24 pass. Rival-hire
  mutation produced two failures, restored green. All-target Clippy passes.
  D follows with strict tile shapes, fields and full official-state scan.
- D green: semantic red nine failures/one passing independent scan; restored ten
  tile tests and Clippy pass. Transpose mutation fails and is restored. Official
  R2 scan: 2,880 snapshots, 576,000 tiles. Independent C/D reviews have no blockers.
- E green: five actor, five storage and two privacy tests initially fail, then
  pass; all B–E tests total 46 passes. Reversed actor/shed ranks fail both rank
  regressions; restored suite and Clippy pass. F context/transactions follows.
- F green: five context semantic failures, diagnostic-validator red, then all
  54 B–F tests pass. check_row rejects 72 corruptions; fractional bonus red is
  repaired. Two-env late-error publication and serial/Rayon/reuse tests pass.
  Clippy passes. G owns deterministic inputs and custody next; R1 remains blocked.
- G producer: eight semantic stub failures become eight passes. Full native input
  generation writes all 512 records (480 non-synthetic), then fails only the R1
  actor quota: `actor_gt16_states=0 < 4`. g25: 15.93 s / 65,716,224-byte sampled
  peak. No final reference fixture is installed. Custody/driver source checks
  and the separate recorder compile are still being finalized.
- Watchdog correction: the driver initially interpreted macOS
  proc_listpgrppids's PID count as bytes. A live-child RSS probe exposes it;
  driver repair is under G. The outer monitor now exports a Unix deadline for
  nested cleanup. Missing-env red then green are recorded separately. A first
  attempted monotonic deadline failed because Mac system Python 3.9's clock
  origin is process-local; no successful monotonic-sharing claim is made.
- G completed controls: 34 Python custody tests, eight producer controls,
  full pre-H root suite (219 passed, three ignored), recorder compilation and
  Clippy pass. Full input generation and independent Python recount agree on
  all 512 headers and unchanged R1 failure. The corrected watchdog reports
  134,086,656 bytes caller+child-group peak for the bounded driver. See task-g.md
  for actual command/version/source receipts and earlier measurement limits.
- Independent G/H review confirms the reviewed reconstruction formulas and
  identifies a source-custody race: source identity was collected after execution.
  A targeted regression is being added before a pre-execution/installation
  identity guard. This does not change recipe, quotas or full-oracle status.

## Final handoff status (actual)

All eight requested end commands were executed. Root: 233 passed, one visible
missing-corpus failure, four ignored. Standalone engine: 59 passed, none ignored;
trim OK. Exact combined observe/custody pytest and py-prepare fail collection
for the absent Task 2.1 schema. The requested reference extra is undefined;
without it the broad fast suite passes 1,056 tests with three platform skips.
rs-prepare and prepare retain the root corpus failure after preceding checks
pass. docs-fresh and final 55-case native smoke pass. Source audit checks note
metadata/sources, 16 restored H mutations, no duplicate snapshot and no vendor
changes. Results/commands, complete file inventory, R1 counts/fixture hashes and
R4 pod handoff are recorded in results.md, files-changed.md, final-commands.json,
oracle-results.json and timing.json. Task I and Task 1.3 remain incomplete.
