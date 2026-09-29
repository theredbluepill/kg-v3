---
type: "Reference"
title: "Live differential parity checks the Rust kernel"
description: "Task 1.1b replays traces generated live from Kaggle's hash-pinned kaggriculture engine through the vendored Rust kernel: 8 committed games and a 40-game local sweep agree; 303 input probes find two malformed-input divergence classes (Unicode digits, unhashable items), recorded as expected failures."
tags: ["kaggriculture-v3", "adaptation", "parity", "rules-engine"]
status: "verified-scoped"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-09-29"}
sources:
  - resource: "user-directive:2026-09-29:add-a-parity-check-with-kaggle-environments"
  - resource: "external:pypi:kaggle-environments==1.32.7"
  - resource: "repository:scripts/kaggriculture_parity/generate_traces.py"
  - resource: "repository:scripts/kaggriculture_parity/sweep.py"
  - resource: "repository:engine_rs/tests/replay_parity.rs"
  - resource: "repository:engine_rs/fixtures/generated/MANIFEST.json"
  - resource: "repository:engine_rs/TRIM_MANIFEST.json"
  - resource: "repository:scripts/check_engine_trim.py"
  - resource: "repository:tests/scripts/test_kaggriculture_parity.py"
  - resource: "repository:tests/tools/test_check_engine_trim.py"
  - resource: "repository:docs/rules-parity-coverage.md"
  - resource: "repository:ops/rebuild-2026-09-29/1.1b/results.md"
  - resource: "repository:ops/rebuild-2026-09-29/1.1b/sweep-summary.json"
---

# Live differential parity checks the Rust kernel

The owner asked, on 2026-09-29: “can you add a parity check after your rust
engine, with kaggle envcironments? thanks a lot”. Task 1.1 had replayed only
four recorded official episodes; its limits named “a fresh differential run
against Python” as missing. This Reference records the check that fills it and
what it found. It serves the
[[../decisions/restart-the-port-from-isaiahs-clean-base|restart Decision]],
whose Task 1.1 rules kernel it exercises. Searches for parity, differential,
kaggle_environments, generated traces and malformed input found no existing
concept beyond that Decision and the coverage document.

## Mechanism

- **Oracle.** `generate_traces.py` runs `kaggle_environments.make("kaggriculture")`
  itself, only when the installed package is 1.32.7 and the engine file's
  SHA-256 equals the `engine_rs/Cargo.toml` pin. The project lock keeps
  kaggle-environments 1.29.0, which has no Kaggriculture environment. The
  generator runs in `uv run --isolated --no-project --with
  kaggle-environments==1.32.7`, and the project manifest and lock stay unchanged.
- **Format.** Traces use the official `kaggriculture-re-parity-v1` records. They
  add a `rejected` record for steps on which Python's interpreter raises, where
  Kaggle's `env.step` keeps its state. Rust must error without mutating state.
- **Inputs.** The inputs come from seeded random, edge-case, Kaggle built-in and
  mixed-seat policies across default, free-hire (250 actors), rich (market-loop
  escape) and custom configurations. There are also 303 one-command probes.
- **Comparator.** `replay_parity.rs` compares official, committed-generated and
  `KAGG_PARITY_TRACES` traces using the same code. It reports the first line,
  step, kind and field path, with expected and actual values.
- **Custody.** `engine_rs/fixtures/generated/MANIFEST.json` pins 15 traces
  (667,059 bytes); `check_engine_trim.py` validates it, and `TRIM_MANIFEST.json`
  pins the manifest.

## Findings

On this version (commits `33e1428`, `8ca378b`), 8 committed games (3,960
transitions) and a 40-game sweep (21,824 transitions) agree completely. That
includes 180 steps Python rejects and Rust also rejects without state change.
Of the 303 probes, 266 agree. The other 37 divergences fall into two classes, both
on malformed input:

- **D1:** Python `int()` accepts non-ASCII decimal digit strings (`"٣"`,
  `"３"`); Rust errors on them as unit counts and drops them as market
  quantities.
- **D2:** Python raises `TypeError` when an array or object reaches a dict
  lookup: a unit verb, a PLANT crop (also for missing hands), a PICKUP/PLACE
  item, or a BUY_SEED/BUY_ANIMAL item. Rust treats these as no-ops.

The vendored kernel bytes are pinned, so nothing was repaired. Seven minimized
repros are expected-failure fixtures with explicit reasons. Rust asserts the
exact divergence, so a change in either engine fails the test. Full-game
policies exclude D1/D2 inputs by default; with them restored, 8 of 12 games
diverge, all classified D1/D2. The model's grammar emits only ASCII names and
integers, so neither class is reachable from policy actions. An adapter that
accepts external actions would need explicit handling.

## Verification and limits

Checks run on this branch are recorded in `ops/rebuild-2026-09-29/1.1b/results.md`:
66/66 engine tests, the trim checker, the Python tests (including live
byte-identical regeneration of the committed set), perturbation tests, and
`just prepare`. Generated games use weak policies (most random seats go
bankrupt); only the four official episodes are competitive worlds. Kaggle
framework behavior outside the interpreter is not modeled. The 40-game, 303-probe
sweep is a bounded sample, and a pod-scale sweep remains open. Reopen D1/D2 if
the adapter must accept untrusted actions or if the pinned engine changes.
