# Task 1.2 section 5.2 — independent reference oracle receipt

The reviewed fixed corpus was recorded from reference `65f0eac5bb00b18a9d3acce319c2a231cbd5dff0` and reproduced by a second native recording. No rebuilt grammar was imported or called to produce expectations. Native calls ran only `ffi::task12_oracle::record_requests`, offline with two Cargo jobs and one Rust test thread. No training, GPU, network fetch or throughput experiment ran.

## Corpus and custody

```json
{
  "sampled": {
    "count": 256,
    "dense_241": 64,
    "full_market": 180,
    "dense_full10": 22
  },
  "replay": {
    "count": 64,
    "dense_241": 0,
    "full_market": 8,
    "dense_full10": 0
  }
}
```

All 320 scheduled programs agree across the actual reference sampler decoder, training FFI decoder and file-loaded Python codec. Eight shapes × 32 fixed PRNG seeds give 256 synthetic programs; four episodes × four strata × four earliest admitted seat programs give 64 real programs. The full-market schedule forces all odd sample indices to consume their complete order budget. Dense examples are synthetic.

All 5,752 real candidate seat actions were scanned (1,438 per episode): zero codec rejections and zero incomplete-layout exclusions. Therefore no replay_rejected record is emitted; all seven planned admission categories are explicitly absent in the manifest. This says nothing about the much larger historical BC dataset.

44 mutation/control records are additional: 40 malformed/capacity candidates plus four padding controls. v4 rejects 43 and accepts the zero-padding control. The 320 scheduled denominator excludes that accepted control.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Compressed fixture | 199448 | `fa26a81fa21189b5329f04af069c5b70491927d1a9d7200d443cfdd89ec6ebd3` |
| Decompressed JSONL | 2901204 | `36d87a61d174959545a85fdf74eef10fc917502420c07b427a8d2a871547542e` |
| Manifest | 34874 | `ac2e44f5f8578926a4f2a91932c35e4634d945b95192518d4e2d1c92d62c45ae` |

Source, source-blob, recorder, harness, append patch, four input traces, Cargo lock and toolchain hashes are recorded in the manifest. `oracle-sha256-final` logs directly computed output/source hashes. Deterministic gzip uses empty filename and mtime 0. `oracle-record-final` and `oracle-verify-final` both exit 0; verify independently reruns all native plan/decode requests and checks byte equality plus manifest equality, without replacing the fixture.

Scratch before bytes and exact append patch are preserved in `scratch/`. The only tracked scratch source change is original `ffi.rs` plus the byte-exact authored harness. Cargo targets live outside scratch at `.codex-tmp/grammar-reference-build-1.2`; request/response working artifacts are ignored by `scratch/.gitignore`. Initial codec loading created one Python bytecode cache; `oracle-scratch-bytecode-clean` removes exactly that generated file/empty directory, then `sys.dont_write_bytecode=True` prevents recurrence. `oracle-scratch-final-state` checks exact source concatenation and no codec cache.

## Decoder disagreement inventory

| Case | Sampler | Training FFI | Python codec | v4 |
|---|---|---|---|---|
| 23-token missing STOP candidate | Reject | Not applicable: incomplete FFI prefix | Not applicable: incomplete frames | Reject |
| HIRE shapes 1/10/1, 17/10/16, 240/10/241 with two hires, 241/10/241 | Reject | Accept | Reject | Reject |
| Nonzero / out-of-width / negative padding | Accept prefix | Accept prefix | Accept prefix | Reject full buffer |
| Zero padding control | Accept | Accept | Accept | Accept |

The manifest retains all eight actual acceptance/action/applicability disagreements with full original outcomes. Differing historical error wording alone is not called a decoder disagreement. Every fixture row still preserves its full original decoder errors.

## Tests and limitations

Initial tests preceded implementation and gave 10 failures against explicit error stubs; these failures occurred in fixture-helper setup, so they are not represented as deep validator evidence. Later targeted controls establish reachability and discrimination: an explicit validator-only error stub fails the valid-fixture test (1 failure), while a temporary no-op validator makes 24 rejection tests fail (1 valid fixture test passes). Both controls restore the recorder byte-exactly in finally blocks. Final restored suite: 25 passed.

The 25 tests cover deterministic compression, valid recorded fixture, missing/extra program/header/shape/table/oracle fields, duplicate record ids and JSON keys, stale source/recorder/harness/append/Cargo/toolchain hashes, changed expected actions, quotas, size and corruption. Hash checks establish identity, not semantic truth; native re-recording supplies the independent expectation check. Complete kernel transition acceptance, production strict encoder/decoder and GPU behavior are outside this oracle-only section.

The no-op-control `uv run` observed an editable owl rebuild after concurrent root source changes; it inherited the two-job/offline environment. Subsequent validator-only proof used installed `.venv/bin/pytest` to avoid unrelated editable rebuilds. No expected fixture data was changed to obtain a passing implementation.

## Logged command inventory

All check statements, exact argv, environment, real exit codes and durations are in `../checks.json`; full output is in `../logs/`. The following is the oracle-owned subset.

### oracle-validator-red — exit 1

```text
uv run --offline pytest tests/tools/test_record_grammar_reference.py -q
```

### oracle-validator-green-first — exit 0

```text
uv run --offline pytest tests/tools/test_record_grammar_reference.py -q
```

### oracle-python-format — exit 1

```text
uvx --offline ruff check --fix ops/rebuild-2026-09-29/1.2/oracle/record_reference.py tests/tools/test_record_grammar_reference.py
```

### oracle-python-format2 — exit 0

```text
uvx --offline ruff format ops/rebuild-2026-09-29/1.2/oracle/record_reference.py tests/tools/test_record_grammar_reference.py
```

### oracle-harness-format — exit 0

```text
rustfmt --edition 2024 ops/rebuild-2026-09-29/1.2/oracle/reference_harness.rs
```

### oracle-validator-green — exit 0

```text
uv run --offline pytest tests/tools/test_record_grammar_reference.py -q
```

### oracle-lint-postformat — exit 1

```text
uvx --offline ruff check ops/rebuild-2026-09-29/1.2/oracle/record_reference.py tests/tools/test_record_grammar_reference.py
```

### oracle-format-final — exit 0

```text
uvx --offline ruff format ops/rebuild-2026-09-29/1.2/oracle/record_reference.py tests/tools/test_record_grammar_reference.py
```

### oracle-lint-final — exit 1

```text
uvx --offline ruff check ops/rebuild-2026-09-29/1.2/oracle/record_reference.py tests/tools/test_record_grammar_reference.py
```

### oracle-record-first — exit 0

```text
uv run --offline python ops/rebuild-2026-09-29/1.2/oracle/record_reference.py record --reference-root /Users/poonszesen/kg-v3-grammar/.codex-tmp/grammar-reference-1.2 --output tests/fixtures/kaggriculture/grammar-v4-reference.jsonl.gz --manifest tests/fixtures/kaggriculture/grammar-v4-reference.manifest.json
```

### oracle-scratch-bytecode-clean — exit 0

```text
python3 -c 'from pathlib import Path; p=Path(".codex-tmp/grammar-reference-1.2/python/owl/kaggriculture/__pycache__/actor_codec.cpython-312.pyc"); print(p, p.stat().st_size); p.unlink(); p.parent.rmdir()'
```

### oracle-lint-fix2 — exit 1

```text
uvx --offline ruff check --fix ops/rebuild-2026-09-29/1.2/oracle/record_reference.py tests/tools/test_record_grammar_reference.py
```

### oracle-format-hardening — exit 0

```text
uvx --offline ruff format ops/rebuild-2026-09-29/1.2/oracle/record_reference.py
```

### oracle-validator-hardening — exit 0

```text
uv run --offline pytest tests/tools/test_record_grammar_reference.py -q
```

### oracle-format-tests-final — exit 0

```text
uvx --offline ruff format tests/tools/test_record_grammar_reference.py
```

### oracle-lint-clean — exit 0

```text
uvx --offline ruff check ops/rebuild-2026-09-29/1.2/oracle/record_reference.py tests/tools/test_record_grammar_reference.py
```

### oracle-record-final — exit 0

```text
uv run --offline python ops/rebuild-2026-09-29/1.2/oracle/record_reference.py record --reference-root /Users/poonszesen/kg-v3-grammar/.codex-tmp/grammar-reference-1.2 --output tests/fixtures/kaggriculture/grammar-v4-reference.jsonl.gz --manifest tests/fixtures/kaggriculture/grammar-v4-reference.manifest.json
```

### oracle-verify-final — exit 0

```text
uv run --offline python ops/rebuild-2026-09-29/1.2/oracle/record_reference.py verify --reference-root /Users/poonszesen/kg-v3-grammar/.codex-tmp/grammar-reference-1.2 --fixture tests/fixtures/kaggriculture/grammar-v4-reference.jsonl.gz --manifest tests/fixtures/kaggriculture/grammar-v4-reference.manifest.json
```

### oracle-tooling-final — exit 0

```text
uv run --offline pytest tests/tools/test_record_grammar_reference.py -q
```

### oracle-scratch-final-state — exit 0

```text
python3 -c 'import pathlib,subprocess; root=pathlib.Path(".codex-tmp/grammar-reference-1.2"); print(subprocess.check_output(["git","status","--short","--untracked-files=all"],cwd=root,text=True)); before=pathlib.Path("ops/rebuild-2026-09-29/1.2/oracle/scratch/ffi.rs.before").read_bytes(); append=pathlib.Path("ops/rebuild-2026-09-29/1.2/oracle/reference_harness.rs").read_bytes(); assert (root/"engine_rs/src/ffi.rs").read_bytes()==before+append; assert not list((root/"python/owl/kaggriculture").glob("__pycache__/actor_codec.*.pyc")); print("exact append and no codec bytecode: OK")'
```

### oracle-sha256-final — exit 0

```text
shasum -a 256 ops/rebuild-2026-09-29/1.2/oracle/record_reference.py ops/rebuild-2026-09-29/1.2/oracle/reference_harness.rs ops/rebuild-2026-09-29/1.2/oracle/scratch/ffi.rs.before ops/rebuild-2026-09-29/1.2/oracle/scratch/append.patch tests/fixtures/kaggriculture/grammar-v4-reference.jsonl.gz tests/fixtures/kaggriculture/grammar-v4-reference.manifest.json
```

### oracle-validator-negative-control — exit 1

```text
python3 -c 'from pathlib import Path; import subprocess,sys; p=Path("ops/rebuild-2026-09-29/1.2/oracle/record_reference.py"); before=p.read_bytes(); s=before.decode(); marker="    exact(manifest, MANIFEST_KEYS, \"manifest\")"; assert marker in s; p.write_text(s.replace(marker,"    return\n"+marker,1)); result=None
try:
 result=subprocess.run(["uv","run","--offline","pytest","tests/tools/test_record_grammar_reference.py","-q"],check=False)
finally:
 p.write_bytes(before); print("validator source restored byte-exactly", flush=True)
assert result is not None
sys.exit(result.returncode)'
```

### oracle-validator-restored-green — exit 0

```text
uv run --offline pytest tests/tools/test_record_grammar_reference.py -q
```

### oracle-validator-explicit-error-control — exit 1

```text
python3 -c 'from pathlib import Path; import subprocess,sys; p=Path("ops/rebuild-2026-09-29/1.2/oracle/record_reference.py"); before=p.read_bytes(); s=before.decode(); marker="    exact(manifest, MANIFEST_KEYS, \"manifest\")"; assert marker in s; p.write_text(s.replace(marker,"    raise ValueError(\"explicit validator error stub control\")\n"+marker,1)); result=None
try:
 result=subprocess.run([".venv/bin/pytest","tests/tools/test_record_grammar_reference.py::test_fixture_validation_and_compression","-q"],check=False)
finally:
 p.write_bytes(before); print("validator source restored byte-exactly", flush=True)
assert result is not None
sys.exit(result.returncode)'
```

### oracle-validator-explicit-error-restored — exit 0

```text
.venv/bin/pytest tests/tools/test_record_grammar_reference.py -q
```

