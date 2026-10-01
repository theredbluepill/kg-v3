# Native-step parity and benchmark receipt

The bounded question is whether moving independent native environment work into
Rayon and reusing staging changes any published byte or seed outcome, and whether
the same 720 native calls become cheaper. Completion is an exact old-source
golden match at 1, 4 and 8 native threads and an equal-work local before/after
timing. This is a CPU component check, not complete learner-update throughput.

## Pristine baseline custody

- Source: `b2276bc5b70073b58a27f9e5fbd52473dbd48569`, exported by
  `git archive` into `/private/tmp/kg-sps-baseline-b2276bc` before implementation
  edits. Source files are independent of the concurrently edited worktree.
- Build: the existing local environment's `maturin build --release`, in the
  pristine export, with its own Cargo target. It completed in 54.68 seconds;
  `baseline-build.log` retains the command output. The resulting abi3 wheel's
  `owl/rs.abi3.so` was extracted into that export's `python/owl/`.
- Runtime: Python 3.12.13, NumPy 2.4.4, Torch 2.9.0; macOS 26.4 arm64,
  Apple M5, 10 cores (4 performance, 6 efficiency), 24 GB memory. Hardware was
  read with `system_profiler SPHardwareDataType -json`, selecting chip/model/core
  count/memory fields only.
- `native-golden.json` and `baseline-benchmark.json` include the actual loaded
  extension SHA-256, all native source-file hashes, lockfile/config hashes,
  helper hash and absolute binary path. No binary is committed.
- The recorder compares every captured native source hash with the immutable
  base Git object before permitting `record`. `verify` never rewrites the golden.

The initial attempt to sample BUY_PRODUCT uniformly over all products was
rejected by the existing native encoder. The sampler was corrected to the
grammar's WHEAT/FERTILIZER support before any successful golden was published.

## Parity scope

`tests/kaggriculture/native_step_oracle.py` samples canonical JSON commands and
encodes them with the existing native grammar. Each environment has its own
Python `random.Random(8128 + env_index)`, independent of processing order. The
engine starts at seed 941003 with stride 7; the initial seeds are 941003,
941010, 941017 and 941024. Configuration is the default 720-step game envelope;
reward is the current 0.25 own bank /150k + 0.25 cash difference /100k + 0.5
terminal sign recipe. Random actions are grammar legal; economic execution can
be ineffective, as allowed by that grammar.

The golden covers 1,440 calls across four environments: eight complete games.
The game's `episodeSteps=720` includes its initial state, so natural terminations
occur at native calls 719 and 1,438. Two additional calls after the second reset
are included. Maximum actor count in the recorded trajectory is 10.

Every initial/step/reset publication contributes raw array bytes to a separate
SHA-256 per each of the 29 observation fields and six transition outputs
(rewards, dones, bank before/after, economic counters before/after). Action tokens
and lengths are also hashed. Metrics and terminal diagnostics use explicit
type tags and little-endian binary64 encoding for floats. Every step also hashes
the complete state snapshot and native seed stream, so auto-reset RNG outputs,
terminal records and game counters are covered. Signed zero is distinguished.

After the full games, the same recording covers three masks (alternating rows,
all false, complementary alternating rows), each `truncate_envs` publication and
its next successful step, followed by full `reset`. Native truncation has no
separate `truncated` output tensor: the selected observations, preserved
transition outputs, seed stream and terminal-record invalidation are all hashed.
The PPO-owned truncation flag is outside this unchanged native interface.

`test_native_step_parity.py` compares threads 1/4/8 independently against the
same old-source golden. It also compares malformed last-peer rollback and each
subsequent good call with an unaffected control, and checks 722 calls of a
four-environment batch against independent one-environment streams stepped in
reverse order. The latter crosses auto-reset and compares every output byte,
metrics, snapshots, terminal records and current seeds.

Pristine-source test result: **7 passed in 7.55s** (`baseline-pytest.log`).
The initial `optimized-parity.log` is superseded: its copied console launcher
pointed to the original environment. Qualification uses the seven parity cases
in the source-verified full/focused runs (`python-final-tests.log` and
`python-native-final-tests.log`) with `runtime_probe.py` asserting the interpreter
and extension inside pytest. The standalone `optimized-verify-{1,4,8}.json`
receipts bind the same golden match to each actual loaded binary and source
inventory; these are correctness checks, with no timing claims.
`final-verify-4.json` repeats the exact match on the final release rebuild after
the equivalent Clippy loop cleanup; the full/focused runs cover all three counts.

## Local microbenchmark

The benchmark creates 20 environments and four native threads and sums elapsed
`perf_counter_ns` around exactly 720 calls to `rs.KaggricultureEnv.step` per
replicate. It includes native validation, GIL transitions, simulation, reward,
observation encoding, output publication, metrics-dict assembly and one
auto-reset. Constructor/import/build, legal action generation, action hashes,
final-state hashes, Python reward telemetry, GPU/model/learner work are outside
the timed spans. Caller output arrays are reused throughout. There is no empty
loop subtraction. Five independent equal-seed replicates are retained.

Baseline step seconds: **0.789712273, 0.801221569, 0.811460921, 0.938839863,
0.911192448**. Median **0.811460921 s** for 14,400 environment transitions,
or **17,745.77 environment transitions/s**. Maximum actor count: 13. The retained
series was rerun after golden capture completed; no agent build/test ran during
these measurements. Scheduling, thermal and power-state noise are not controlled.

Optimized step seconds: **0.681750035, 0.658316131, 0.712675411, 0.660819773,
0.786570719**. Median **0.681750035 s**, or **21,122.11 environment
transitions/s**: 15.98% less timed native work and 1.190x the component rate in
these separate five-replicate series. `optimized-benchmark.json` retains the
series and source inventory. The action hash and final-state/seed hash equal the
baseline exactly.

An additional five-pair experiment alternated separate baseline/candidate
processes, reversing order on odd pairs to expose drift. `paired_benchmark.py`
is the reproducible driver; `paired-benchmark.json` preserves both identities,
all ten samples and their execution order.

| Pair | Execution order | Baseline seconds | Optimized seconds |
| --- | --- | ---: | ---: |
| 0 | Baseline, optimized | 1.388226045 | 2.742900827 |
| 1 | Optimized, baseline | 1.272358415 | 1.488603691 |
| 2 | Baseline, optimized | 1.700539185 | 0.837861480 |
| 3 | Optimized, baseline | 1.141821576 | 0.742962592 |
| 4 | Baseline, optimized | 0.983318624 | 0.917302201 |

The paired-series medians are 1.272358415 and 0.917302201 s (27.91% less
time, 1.387x rate), but the optimized path loses two of five pairs and both arms
vary substantially. The agent held its builds/tests during timing; unrelated
host load and CPU scheduling/power state were not isolated. These numbers
support a local reduction in median component time on this workload with
substantial measurement noise, not a stable speedup magnitude or a per-sample
non-regression guarantee. All ten samples have identical action and final-state
hashes. No result or outlier was discarded from the paired series.

After the final release rebuild, five candidate repeats followed by five
pristine-baseline repeats measured candidate seconds **1.216313878, 1.190349540,
1.203040680, 1.475719549, 1.164478214**, baseline seconds **1.985459686,
1.762235097, 1.790876246, 1.828217771, 2.135753602**. Medians are **1.203040680**
and **1.828217771** seconds (34.20% less time, 1.5197× rate in this series).
`final-benchmark.json` and `final-baseline-benchmark.json` retain final identities
and unchanged action/output hashes. No builds/tests overlapped. The large drift
across all three retained series reinforces the measurement-noise limit; this
does not establish a stable percentage improvement or H200 gain.

The benchmark records both action-sequence and final-state/seed hashes. Equal
hashes are required across replicates; before/after comparison must also retain
those equal hashes. Dense 241-actor games, pinned CUDA DMA, learner SPS, H200/Linux
and any shared-host training effects remain outside this measurement.

## Reproduction

To rebuild a pristine reference from this checkout, use a fresh temporary tree
and the already installed locked environment (offline commands require cached
Cargo dependencies):

```sh
task_repo="$PWD"
task_python="$PWD/.venv/bin/python"
task_maturin="$PWD/.venv/bin/maturin"
baseline_tree="$(mktemp -d /private/tmp/kg-native-baseline.XXXXXX)"
git archive b2276bc5b70073b58a27f9e5fbd52473dbd48569 | tar -x -C "$baseline_tree"
(
  cd "$baseline_tree"
  "$task_maturin" build --release --offline --locked --interpreter "$task_python" --out "$baseline_tree/wheels"
)
"$task_python" - "$baseline_tree" <<'PY'
from pathlib import Path
import sys
import zipfile

root = Path(sys.argv[1])
wheels = list((root / "wheels").glob("*.whl"))
assert len(wheels) == 1, wheels
with zipfile.ZipFile(wheels[0]) as wheel:
    (root / "python/owl/rs.abi3.so").write_bytes(wheel.read("owl/rs.abi3.so"))
PY
PYTHONPATH="$baseline_tree/python" "$task_python" "$task_repo/ops/sps-2026-10-01/native_benchmark.py" record --output /private/tmp/repeated-golden.json
```

Build the candidate in this worktree with
`uv run --no-sync maturin develop --release --skip-install --offline` after
installing the locked Python dependencies. This matches the final local build's
offline installation substitution; the normal network-enabled command can omit
`--skip-install --offline`.

Use the desired source tree's `python/` directory in `PYTHONPATH` and an
environment with this repository's locked dependencies:

```sh
PYTHONPATH=/private/tmp/kg-sps-baseline-b2276bc/python .venv/bin/python ops/sps-2026-10-01/native_benchmark.py record --output /tmp/repeated-golden.json
PYTHONPATH=python .venv/bin/python ops/sps-2026-10-01/native_benchmark.py verify --threads 4 --output /tmp/verified.json
PYTHONPATH=python .venv/bin/python ops/sps-2026-10-01/native_benchmark.py benchmark --output /tmp/benchmark.json
PYTHONPATH=python .venv/bin/python -m pytest tests/kaggriculture/test_native_step_parity.py -q
.venv/bin/python ops/sps-2026-10-01/paired_benchmark.py --baseline-python-source /private/tmp/kg-sps-baseline-b2276bc/python --candidate-python-source python --pairs 5 --output /tmp/paired.json
```
