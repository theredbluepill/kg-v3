# Independent Task 1.4 oracle / recorder verification

Scope: `e197528...ba9b59b` in `kg/rebuild-env`. Read the Task 1.4 oracle/ABI brief,
the current native-buffer Reference, the prior verification report and the
pinned reference TrainingBatch source. No tracked file was edited.

Target and stopping condition: verify recorder provenance and strict full-game
comparison, independently remove the 14 repaired inventory guards, perturb each
transition comparison/math oracle, and exercise custody, budget, publication and
watchdog boundaries. Stop each perturbation on the intended diagnostic; never
run a reference compilation or recording. Native replay perturbations use the
already-installed extension and mutate only copied test code. All mutation source
is isolated at `/private/tmp/kg-env-r2-oracle` and restored byte for byte.

## Prior findings

- **RESOLVED — P3 inventory guards.**
  `tests/tools/test_record_kaggriculture_env_reference.py:245` now contains 17
  coherent invalid-array cases; each refreshes array metadata and matches the
  specific semantic error. The nine old hash-corruption cases remain separate.
  All 14 require-call omissions fail a named current test; the source hash of
  the frozen fixture was not used as a mutation kill.
- **RESOLVED — P3 constructor status.**
  `docs/rl-api-specs.md:1060` now says the approved Q1 constructor refinement is
  incorporated by contract v4.2.

## New finding

**P3 — Exercise compressed-size and hash custody independently of size mismatch.**
`tests/tools/test_record_kaggriculture_env_reference.py:255` changes only declared
sizes, so deleting the sole compressed-size cap still reaches a later declared /
actual-size mismatch and passes the test's broad `size|budget` match. At line 273,
the hash test appends bytes without refreshing `compressed_bytes`, so it stops at
the size check before reaching the fixture hash. There is no negative expanded
hash case. Deleting each of the compressed cap, fixture hash and expanded hash
checks independently leaves **all 52 behavioral recorder tests passing**. The
frozen-source test is excluded because any code mutation changes the pinned
recorder hash regardless of its behavior.

Current production code correctly enforces all three. Coherent scratch probes
prove those guard omissions matter: a valid real NPZ above a deliberately lowered
`MAX_COMPRESSED` passes when the compressed cap is removed; a wrong fixture hash
with unchanged sizes passes when its check is removed; a wrong expanded hash
passes when its check is removed. Baseline rejects each at the specific guard.

**Fix:** use a valid small NPZ and lower `MAX_COMPRESSED` just below its actual
size, asserting `compressed size budget exceeded`; alter only the fixture hash
(or preserve/refresh size while corrupting the payload), asserting `fixture hash
differs` before NumPy load; alter only `expanded_sha256`, asserting `expanded
fixture hash differs`. Keep the generic size mismatch tests separately.

This finding does not reopen the resolved array-inventory issue and is not a
production admission defect. The expanded manifest cap omission is explicitly
**not** included: its coherent over-budget fixture is still rejected by the
separate ZIP expanded-total check. The current header-layout omission survives
the stock unexpected-member test because ZIP inventory rejects first; a coherent
header/metadata mismatch probe distinguishes its pre-NumPy guard using a load
sentinel, without making a claim that NumPy would accept malformed data.

## Fresh checks and mutations

- Full copied recorder suite: **53 passed** before mutation, **53 passed** after
  the inventory campaign, **53 passed** after all campaigns.
- Repaired inventory/hash guards: **14/14 omissions killed** by current named
  tests (`guard-removals.json`, `guard-00.log` through `guard-13.log`).
- Replay comparison/math perturbations: **7/7 killed**. Each of the six transition
  output changes reports game 0 / seed 17000 / step 0 / the intended field; the
  mathematical reward perturbation fails its separate assertion. These are
  scratch test/output perturbations, not seven rebuilt native mutants.
- Initial remaining-custody stock campaign: **3/9 killed, 6/9 survived**. Source
  identity, ZIP inventory and exported-source byte drift are killed. Size/hash
  survivors are investigated above rather than counted as kills.
- Coherent extra probes: **17/17 guard omissions distinguished**: six coverage /
  array-schema guards; five source/manifest/budget guards; three actual loader
  size/hash guards; fixed recipe, positive budget and Mac budget with an inert
  worker sentinel. No reference subprocess was launched.
- Coherent loader boundaries: three cap/hash omissions accept invalid input;
  expanded cap omission remains rejected by the independent ZIP total guard;
  header-layout omission reaches an inert NumPy-load sentinel. All outcomes are
  recorded in `loader-boundaries.json`.
- Publication: **2/2 premature-publication mutants killed** by the current
  failed-validation and failed-worker tests.
- Supervision: fixed-recipe omission reaches only an inert sentinel; omitting
  supervisor RSS changes the deterministic fake-process stop reason from memory
  to wall time. No real process was killed or started.
- Whole behavioral-suite confirmations: **52 passed / 1 frozen-source case
  deselected** independently under each of three cap/hash omissions.

Counts across campaigns overlap deliberately; do not add them as unique guards.

## Corpus and provenance

`corpus-summary.json` verifies the fixture through the actual current loader and
checks the full source identity against pinned Git objects and mutable recorder,
policy and Rust example bytes. The fixture contains **16 complete games**, seeds
17000–17015, **11,504 transitions**, **23,008 seat programs**, and **46,528 active
frames**. It is **310,365 compressed bytes / 17,201,128 expanded bytes**, below both
budgets. All seven required coverage values are positive in all 16 games.

The Rust recorder imports the actual pinned
`kaggriculture_engine::training::TrainingBatch`, calls `from_json` / `step`, and
records its output rewards, dones, banks, counters, seeds and terminal values.
Its source is added only to an exported reference tree; source export hashes are
checked after build and again before publication. Actor tokens come from the
pinned reference codec. The policy reads only the current step and current hands,
exactly matching the brief's fixed recipe and two reward configurations.

The exact replay compares float32/float64 bit representations (including signed
zero), all counter values, done/seed schedules and terminal metrics. Its scalar
math oracle has its own explained one-ULP allowance and does not relax the
recorded-reference bitwise assertions. Full replay baseline counts belong to the
parent's fresh full suite; no new reference regeneration is claimed here.

## Custody and harness limits

`restoration-custody.json` verifies all nine copied source/test files equal their
main-worktree counterparts. Both mutated copies (recorder and replay test) are
restored byte for byte. The main recorder SHA-256 remains
`156bee305dd86d0c8f33584dab0f6175810e52908723577418f6d9f5536f3499`.

Early scratch attempts inside `.codex-tmp` inherited Git's relative archive
prefix and failed before tests could exercise semantics; explicitly setting Git
environment variables alone did not eliminate the cwd prefix. Moving scratch
outside the worktree and using read-only Git environment fixed it. One initial
replay import lacked the copied `test_observe.py` helper. These harness failures
are preserved under `initial-harness/` and are not counted as test or mutation
results. No native build, fresh reference recording, training or GPU work occurred
in this lane.

Lane recommendation: APPROVE WITH EDITS for the new P3 test-custody finding.
