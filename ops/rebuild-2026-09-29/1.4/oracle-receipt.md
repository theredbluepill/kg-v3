# Task G reference oracle custody

## Planned diagnostic

Question: does the new native lifecycle reproduce every transition from the
actual reference `TrainingBatch` at
`65f0eac5bb00b18a9d3acce319c2a231cbd5dff0`, including its two reward rounding
points and terminal auto-reset?

The reviewed recipe is sixteen seeds 17000–17015, default configuration,
stride one, 719 transitions each and one live reference batch. The policy reads
only the current public clock and extant hands. Hires, placed GOOSE, sales and
positive starvation/drought/ineffective counters are required in every game.
The reference Rust example, added only to a temporary `git archive` export,
returns float bits and compact public policy input. The Python driver loads the
exported reference codec and packs only active token prefixes. It never imports
the root lifecycle or substitutes a simulator for `TrainingBatch`.

The one recording attempt includes archive export, offline locked debug build
and all sixteen complete games. The supervisor counts its own resident memory
and the whole worker process group (cargo/compiler/native children), stops at
115 seconds or 960 MiB on this Mac, and kills the worker group. A surrounding
bounded diagnostic gives the nested supervisor three seconds to clean up.
Workers publish only to `.codex-tmp/`; the supervisor publishes the complete
validated fixture pair only after a successful worker exit. Budget failure
leaves the full replay and native reward/done mutation PENDING (pod); there is no
reduced fixture or skipped missing-fixture assertion.

## Actual checks before recording

- `g-recorder-red`: one collection error, absent recorder `FileNotFoundError`;
  exit 2, 1.354 seconds, sampled peak 322,469,888 bytes.
- `g-replay-recorder-red`: one failed assertion, "Task 1.4 reference recorder is
  missing"; exit 1, 1.762 seconds, peak 305,954,816 bytes.
- `g-recorder-unit-first`: 29 passed, one frozen-fixture test deselected while
  the fixture had not been recorded; 5.368 seconds, peak 400,179,200 bytes.
- `g-recorder-full-missing-fixture`: 33 passed, one failed (required absent full
  fixture); 5.721 seconds, peak 397,344,768 bytes. The failure is deliberately
  loud, not skipped. Synthetic tests do not qualify reference/native parity.
- `g-recorder-lint-complete`: Ruff passes all three new Python paths, exit 0.
- `g-recorder-mypy-complete`: mypy passes both new script files, exit 0.
- `g-recorder-rustfmt`: Rust example formatting passes, exit 0. This is not a
  Rust compilation receipt.

The synthetic checks cover every per-game quota, full transition/seed inventory,
source and fixture drift, compression/expansion budgets, deterministic ZIP
ordering/timestamps, validation before NumPy load (`allow_pickle=False`), and
failure without final publication. A mocked watchdog proves that supervisor and
child resident memory are combined and the complete process group is killed.
Only immutable git objects are cached; mutable recorder/policy/Rust source and
the existing grammar corpus/manifest are rehashed on every load/publication.

Post-attempt custody review added rehashing of every original exported reference
file after compilation and again before fixture publication. The two clean/drift
tests first failed with missing `verify_export` (`g-export-drift-red`: two
failed, 34 deselected), then passed (`g-export-drift-green` and
`g-export-drift-confirm`: two passed, 34 deselected). The final confirmation used
all prescribed thread environment settings; the earlier green accidentally set
`OMP_NUM_THREADS=1` rather than 2. These file-only tests do not invoke OpenMP.
Ruff and two-script mypy both pass after this guard (`g-export-lint`,
`g-export-mypy`). The suite now contains 36 tests; the full fixture test remains
the required missing-fixture failure.

Formatting and typing were completed before the recording attempt to avoid
invalidating recorded producer hashes through later cosmetic edits. The actual
failed-attempt producer hashes are frozen in
`reference-attempt-source-custody.json`; the later export-drift guard's hashes
are separately recorded in `reference-pod-source-custody.json` as unattempted.

## Recording result

The single Mac recording attempt stopped while compiling the exported reference
engine, before the Rust recorder example could compile or any game could run.
The inner watchdog measured **1,012,252,672 bytes** combined supervisor/worker
process-group RSS against the 1,006,632,960-byte (960 MiB) cap, killed the worker
group and recorded exit -9 after **11.296769 seconds**. The enclosing diagnostic
returned exit 1 after **11.442614 seconds**. Its 58,654,720-byte measurement is
only the outer group; the inner receipt explicitly includes the separately
supervised worker and all compiler/native descendants.

Evidence: `g-reference-recording.{log,json}` and
`reference-recording-attempt.json`. Inspection confirmed that neither final
fixture nor final manifest exists. No second Mac build/record attempt ran.
`g-recorder-final-custody` again reports **33 passed, one failed**, solely the
required missing-full-fixture failure. This preserves the requested failure
instead of skipping unrecorded evidence.

PENDING (pod), exact recording command (run with the task's environment exports;
Claude may choose larger explicit positive watchdog budgets on Linux):

```sh
uv run --offline python scripts/record_kaggriculture_env_reference.py --reference 65f0eac5bb00b18a9d3acce319c2a231cbd5dff0 --games 16 --first-seed 17000 --max-live-envs 1 --max-seconds 115 --max-rss-mib 960 --output tests/fixtures/kaggriculture_env_reference_v1.npz
```

Games/transitions compared: **0 / 0**. No first-divergence or passing live-policy
coverage claim is available. Compressed/expanded fixture sizes and fixture
SHA-256s are unavailable because nothing was published. Compilation/execution of
the actual Rust example, all sixteen-game coverage assertions, full bitwise
native replay and its native reward/done mutation remain PENDING (pod).
