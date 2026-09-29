# Stage 2 native grammar tables

Target: admit only the Task 1.4 grammar ABI and use its eight tables by default
at model construction. Stop when the strict admission tests, native equality
and model native-loader/default/injection probes pass. This is a CPU unit-test
check, not a model diagnostic, trainer run or throughput measurement.

Changed paths:

- `python/owl/kaggriculture/gpu_grammar.py`: constants-first native loader with
  exact version/names/widths, keys, NumPy bool dtype, shape and C-layout checks;
  calls the existing conversion helper with the requested device.
- `python/owl/model/kaggriculture.py`: native tables become the default;
  explicit table injection and non-persistent actor buffers remain.
- `tests/kaggriculture/test_gpu_grammar.py`: removes the NotImplementedError skip.
- `tests/kaggriculture/test_native_tables.py`: equality, ABI corruption, binding
  failure propagation, ordered admission/device forwarding, native default
  called once and explicit-injection bypass tests.
- `docs/model-architecture.md`: native default/validation/lifetime; startup
  guard names the remaining Task 3.1 storage/action-mapping blocker.

Every test/check command below inherited `OMP_NUM_THREADS=2`,
`CARGO_BUILD_JOBS=2`, `CARGO_NET_OFFLINE=true`, `UV_OFFLINE=true`.
Actual subprocess exits are recorded in the named JSON files, separate from
the receipt wrapper's exit; no shell output pipeline masks them.

| Receipt | Command | Exit | Result |
|---|---|---:|---|
| `red.log`, `red.json` | `uv run --offline pytest tests/kaggriculture/test_native_tables.py tests/kaggriculture/test_gpu_grammar.py -q` | 1 | 22 failed, 24 passed, 0 skipped |
| `format.log`, `format.json` | `uv run --offline ruff format python/owl/kaggriculture/gpu_grammar.py python/owl/model/kaggriculture.py tests/kaggriculture/test_native_tables.py tests/kaggriculture/test_gpu_grammar.py` | 0 | 1 file reformatted, 3 unchanged |
| `lint.log`, `lint.json` | `uv run --offline ruff check python/owl/kaggriculture/gpu_grammar.py python/owl/model/kaggriculture.py tests/kaggriculture/test_native_tables.py tests/kaggriculture/test_gpu_grammar.py` | 0 | all checks passed |
| `green.log`, `green.json` | `uv run --offline pytest tests/kaggriculture/test_native_tables.py tests/kaggriculture/test_gpu_grammar.py -q` | 0 | 46 passed, 0 failed, 0 skipped |

Red ran after adding the tests/unskip and before implementing the loader/default.
The native equality/admission tests hit the existing NotImplementedError; the
model-default test independently failed with zero native table calls instead
of one. The injection test already passed. Green runs the real binding and
compares every native table bit against the independent expected-table oracle.
The red command triggered `uv run`'s permitted automatic debug native rebuild;
no Rust source was changed by this subtask. Green pytest body elapsed 0.05 s.

No table-task deviation from the brief. Beyond value equality, strict ABI
admission rejects boolean versions and float widths rather than treating them
as integer constants. CPU string and torch.device forms are checked. Hardware
upload is not qualified here; buffers use the model's existing `.to(device)`
path. Parent owns canonical prepare, full-suite and final integrity receipts.
No cookbook, protected Orbit tests, git state, dependency or vendored engine
files were changed by this subtask.
