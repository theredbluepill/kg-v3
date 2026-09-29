# Task 7.1 frozen-engine API probe

## Diagnostic expectation

The required standalone crate must re-export the existing engine types and
`fib`, then compile the four byte-exact pinned controllers without modifying
the engine. Source inspection found `fib` and `Game.config` private, plus absent
`Game` accessors. The bounded diagnostic question was whether an actual external
crate produces those errors. Stop on the first compiler result; execute no game
and make no implementation workaround. This expectation was stated before the
compile, not inferred from successful opponent behavior.

## Actual result

`cargo check --locked --offline --manifest-path
.codex-tmp/7.1-native-api-probe/Cargo.toml --all-targets` exited **101**. The
engine dependency compiled; the probe failed with **42 library errors and 51
library-test errors**. No tests or games executed. The full compiler output is
`native-api-probe-red.log`; `native-api-probe-red.exit` contains `101`.

| Required item | Frozen engine status | Callers |
| --- | --- | --- |
| `fib` | Private function, `src/lib.rs:3785`; E0603 | Starter |
| `Game.config` | Private field, `src/lib.rs:1190`; E0616 | Starter `hire_step`, pinned `starter.rs:174` |
| `Game::farms()` | Absent; E0599 | All four bots |
| `Game::privates()` | Absent; E0599 | All four bots |
| `Game::step_index()` | Absent; E0599 | All four bots |
| `Game::market()` | Absent; E0599 | R04, EcoBot, E776 |
| `Game::town()` | Absent; E0599 | R04, EcoBot, E776 |

`git grep` at the reference pin locates the five accessor definitions in excluded
`engine_rs/src/policy_rows.rs:1106–1118`. Restoring that historical module is
outside the requested scope. Private `fib` and `config` were available to
controllers nested inside the original engine crate; re-exporting engine types
from a separate crate does not grant access to them. Starter's historical helper
and inline tests remain part of its required byte-exact source.

The compiler also emits E0308/E0277 at E776 lines 562/565 while its `Game`
accessors are unresolved. This receipt does not attribute those as independent
controller defects. No compiler-suggested source changes were made.

All five scratch imports match the brief and pinned Git blobs byte for byte:

| Reference path suffix under `engine_rs/` | Bytes | SHA-256 |
| --- | ---: | --- |
| `src/native_agents/starter.rs` | 14,759 | `01b4de943e7ea9df425b97355f8451ce48547b3666cc1c3aa3faa5649eda22e0` |
| `src/native_agents/r04.rs` | 71,788 | `a80f130c636fc9def610027b6a5593c67db3a4ddb5d63ac972d80c4d2ba31431` |
| `src/native_agents/ecobot.rs` | 92,771 | `8794e7cf57ef6663c1578464d8e5a1d2749736bc21ba52d45a2301a58e338ce8` |
| `src/native_agents/e776.rs` | 38,299 | `02166970e2fc8345178d29ea863fe8532c9a4fa7d227651d0ebc57b0df82b9fe` |
| `fixtures/e776-kenjo-trace.json` | 117,954 | `da0d5d1bd326cb5bf068c2065ba1fe8f7e644107db806d7f9a1eae4dafd89692` |

`native-api-probe.json` preserves the exact authored probe sources,
package-manager-generated Cargo manifest/lock, versions, source hashes, and
compiler-log hash. The probe is ignored scratch work, not a production crate;
no `opponents_rs/` artifact was created. There is no green opponent test result
or parity claim. The user explicitly requires stopping this placement path if
an engine item is not public, so registry/lifecycle/visibility/determinism work
stopped here. No engine bytes were edited by this diagnostic.

## Reproduction

Run from the repository root. The setup block below is for a fresh checkout
where `.codex-tmp/7.1-native-api-probe` does not yet exist. In this checkout the
probe already exists; rerun only the following command after the same exports:

```sh
export CARGO_BUILD_JOBS=2 CARGO_NET_OFFLINE=true UV_OFFLINE=true
export RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=1
export TMPDIR=/Users/poonszesen/kg-v3-t71/.codex-tmp
cargo check --locked --offline --manifest-path .codex-tmp/7.1-native-api-probe/Cargo.toml --all-targets
```

Fresh-checkout setup (do not run over an existing scratch crate):

```sh
export CARGO_BUILD_JOBS=2 CARGO_NET_OFFLINE=true UV_OFFLINE=true
export RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=1
export TMPDIR=/Users/poonszesen/kg-v3-t71/.codex-tmp
set -e
cargo init --lib --edition 2024 --name kaggriculture-opponents-api-probe --vcs none .codex-tmp/7.1-native-api-probe
cargo add --offline --manifest-path .codex-tmp/7.1-native-api-probe/Cargo.toml kaggriculture-engine --path engine_rs
cargo add --offline --manifest-path .codex-tmp/7.1-native-api-probe/Cargo.toml serde_json --features arbitrary_precision,preserve_order
cargo add --offline --manifest-path .codex-tmp/7.1-native-api-probe/Cargo.toml indexmap@2.11 num-bigint@0.4 num-traits@0.2
mkdir -p .codex-tmp/7.1-native-api-probe/src/native_agents .codex-tmp/7.1-native-api-probe/fixtures
for bot in starter r04 ecobot e776; do
  git show 65f0eac5bb00b18a9d3acce319c2a231cbd5dff0:engine_rs/src/native_agents/$bot.rs > .codex-tmp/7.1-native-api-probe/src/native_agents/$bot.rs
done
git show 65f0eac5bb00b18a9d3acce319c2a231cbd5dff0:engine_rs/fixtures/e776-kenjo-trace.json > .codex-tmp/7.1-native-api-probe/fixtures/e776-kenjo-trace.json
uv run --offline python - <<'PY'
import json
from pathlib import Path

receipt = json.loads(Path('ops/rebuild-2026-09-29/7.1/native-api-probe.json').read_text())
probe = Path(receipt['probe_path'])
for name, source in receipt['probe_support'].items():
    if name.startswith('src/'):
        (probe / name).write_text(source['contents'])
    else:
        assert (probe / name).read_text() == source['contents'], name
PY
cargo check --locked --offline --manifest-path .codex-tmp/7.1-native-api-probe/Cargo.toml --all-targets
```

The initial setup attempt used `cargo add --path ../../engine_rs` from the repo
root; Cargo resolved that relative to the working directory and refused the
nonexistent `/Users/engine_rs`. The corrected command above used `--path
engine_rs`; the generated manifest correctly records `../../engine_rs` relative
to the probe manifest. No offline dependency was missing and no network was used.
