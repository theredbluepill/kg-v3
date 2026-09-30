# Observation oracle watchdog test isolation

During the source-verified full Python run, four existing tests failed in
`tests/tools/test_observation_oracle_custody.py` with
`shared oracle execution deadline expired before launch`. The retained log is
`python-final-deadline-fail.log`: 4 failed, 3,011 passed, 9 skipped in 219.40 seconds.

## Diagnosis and source identity

`scripts/kaggriculture_observation_oracle/regenerate.py` initializes its shared
`RUN_DEADLINE` at module import. Without an external deadline it permits about
110 seconds: the 115-second allowance minus five seconds for cleanup. Pytest
imports the test module, and therefore the oracle module, during collection.
Earlier tests can consume that entire allowance before these four tests begin.
Their intended RSS/stdout checks then fail before launching any subprocess.

This coupling exists in task base
`b2276bc5b70073b58a27f9e5fbd52473dbd48569`. Both files were byte-identical to
that base before this test-only repair; their last touching commit was
`b8747b6e8acece5f561d09a75bb914364a60ac05`.

| File | Base SHA-256 |
| --- | --- |
| `scripts/kaggriculture_observation_oracle/regenerate.py` | `e7b39e4e9bbe439e40d9818c3f0cde06e19479d6ed09f1badb919fff7e9a5472` |
| `tests/tools/test_observation_oracle_custody.py` | `ea647d1c0ca1a457fd0342d079b846f483b179b3c46a1a7a12cd269345554052` |

## Repair

A function-scoped autouse fixture gives each test a fresh default deadline
(`time.monotonic() + 115 - 5`) through pytest's reversible monkeypatch. Each
test represents an independent oracle invocation. The production script has
no named budget constants to reuse and remains byte-identical to the base,
including its shared deadline, RSS limit, process-group cleanup and wall limit.

The new `test_expired_shared_deadline_rejects_before_launch` deliberately
expires the deadline and forbids `Popen`, proving rejection still occurs
before launch. The existing test of a live child stopped by an 0.08-second
shared deadline remains intact. The original RSS tests remain intact too.

Updated test SHA-256 (format check made no change):
`7af794e766dd9df51b3d8d5578e5e920801c8593a624ee0d6e3070b63505ba5a`.

## Actual checks

- `.venv/bin/python -m pytest tests/tools/test_observation_oracle_custody.py -q`:
  exit 0, **54 passed in 0.77 seconds** (`oracle-test-isolation-tests.log`).
- The same command with `KG_BOUNDED_DEADLINE_UNIX=0`: exit 0, **54 passed in
  0.75 seconds** (`oracle-test-isolation-expired-import.log`). This deliberately
  makes the oracle's import-time deadline already expired without waiting;
  the test fixture restores per-test isolation, while both explicit deadline
  rejection tests still exercise production guards.
- Ruff 0.15.10 check and format check on the edited test file: exit 0; the file
  was already formatted (`oracle-test-isolation-ruff.log` and
  `oracle-test-isolation-format.log`). The initial `uvx --offline ruff` attempt
  could not write its sandboxed cache and exited 2; its output is retained in
  `oracle-test-isolation-uvx-attempt.log`. The checks used the existing cached
  Ruff binary directly.

The main task owns the full-suite rerun and cookbook record. No production
watchdog, corpus, fixture manifest, or native implementation changed here.
