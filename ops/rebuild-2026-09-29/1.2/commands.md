# Task 1.2 — complete bounded command ledger

All commands below inherit `CARGO_BUILD_JOBS=2 CARGO_NET_OFFLINE=true UV_OFFLINE=true`.
Each check was declared before launch in `checks.json`. Exit codes are direct process results;
negative controls and diagnosed intermediate failures remain visible. Exact output, including
all child commands of preparation and native oracle recording, is retained in the linked logs.
Routine read-only source inspection (`git show`, `rg`, `cat`, `sed`, Git status/log and JSON reads)
is distinguished in results.md from semantic verification. No stage/commit/worktree command ran.

## constants-red — exit 101

```text
cargo test --locked --lib grammar_constants -- --test-threads=1
```

Result: No test denominator (compile/tool/check command); see log for diagnostics.

Log: [logs/constants-red.log](logs/constants-red.log)

## checker-placeholder-format — exit 0

```text
rustfmt --edition 2024 engine_rs/tests/grammar_kernel.rs
```

Result: No test denominator (compile/tool/check command); see log for diagnostics.

Log: [logs/checker-placeholder-format.log](logs/checker-placeholder-format.log)

## checker-placeholder-hash — exit 0

```text
shasum -a 256 engine_rs/tests/grammar_kernel.rs
```

Result: No test denominator (compile/tool/check command); see log for diagnostics.

Log: [logs/checker-placeholder-hash.log](logs/checker-placeholder-hash.log)

## constants-green — exit 0

```text
cargo test --locked --lib grammar_constants -- --test-threads=1
```

Result: test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 157 filtered out; finished in 0.00s

Log: [logs/constants-green.log](logs/constants-green.log)

## checker-red — exit 1

```text
uv run --offline pytest tests/tools/test_check_engine_trim.py -q
```

Result: 14 failed, 48 passed in 2.12s

Log: [logs/checker-red.log](logs/checker-red.log)

## checker-green — exit 0

```text
uv run --offline pytest tests/tools/test_check_engine_trim.py -q
```

Result: 62 passed in 1.92s

Log: [logs/checker-green.log](logs/checker-green.log)

## checker-current-inventory — exit 0

```text
uv run --offline python scripts/check_engine_trim.py
```

Result: engine trim manifest: OK

Log: [logs/checker-current-inventory.log](logs/checker-current-inventory.log)

## oracle-validator-red — exit 1

```text
uv run --offline pytest tests/tools/test_record_grammar_reference.py -q
```

Result: 10 failed in 0.03s

Log: [logs/oracle-validator-red.log](logs/oracle-validator-red.log)

## resumed-source-identity — exit 0

```text
python3 -c 'import subprocess; commands=[["git","branch","--show-current"],["git","status","--short"],["git","log","-3","--oneline"],["git","rev-parse","kg/reference-2026-09-29"]]; [subprocess.run(c,check=True) for c in commands]'
```

Result: No test denominator (compile/tool/check command); see log for diagnostics.

Log: [logs/resumed-source-identity.log](logs/resumed-source-identity.log)

## oracle-validator-green-first — exit 0

```text
uv run --offline pytest tests/tools/test_record_grammar_reference.py -q
```

Result: 10 passed in 2.40s

Log: [logs/oracle-validator-green-first.log](logs/oracle-validator-green-first.log)

## oracle-python-format — exit 1

```text
uvx --offline ruff check --fix ops/rebuild-2026-09-29/1.2/oracle/record_reference.py tests/tools/test_record_grammar_reference.py
```

Result: No test denominator (compile/tool/check command); see log for diagnostics.

Log: [logs/oracle-python-format.log](logs/oracle-python-format.log)

## oracle-python-format2 — exit 0

```text
uvx --offline ruff format ops/rebuild-2026-09-29/1.2/oracle/record_reference.py tests/tools/test_record_grammar_reference.py
```

Result: No test denominator (compile/tool/check command); see log for diagnostics.

Log: [logs/oracle-python-format2.log](logs/oracle-python-format2.log)

## oracle-harness-format — exit 0

```text
rustfmt --edition 2024 ops/rebuild-2026-09-29/1.2/oracle/reference_harness.rs
```

Result: No test denominator (compile/tool/check command); see log for diagnostics.

Log: [logs/oracle-harness-format.log](logs/oracle-harness-format.log)

## oracle-validator-green — exit 0

```text
uv run --offline pytest tests/tools/test_record_grammar_reference.py -q
```

Result: 18 passed in 4.74s

Log: [logs/oracle-validator-green.log](logs/oracle-validator-green.log)

## oracle-lint-postformat — exit 1

```text
uvx --offline ruff check ops/rebuild-2026-09-29/1.2/oracle/record_reference.py tests/tools/test_record_grammar_reference.py
```

Result: No test denominator (compile/tool/check command); see log for diagnostics.

Log: [logs/oracle-lint-postformat.log](logs/oracle-lint-postformat.log)

## oracle-format-final — exit 0

```text
uvx --offline ruff format ops/rebuild-2026-09-29/1.2/oracle/record_reference.py tests/tools/test_record_grammar_reference.py
```

Result: No test denominator (compile/tool/check command); see log for diagnostics.

Log: [logs/oracle-format-final.log](logs/oracle-format-final.log)

## oracle-lint-final — exit 1

```text
uvx --offline ruff check ops/rebuild-2026-09-29/1.2/oracle/record_reference.py tests/tools/test_record_grammar_reference.py
```

Result: No test denominator (compile/tool/check command); see log for diagnostics.

Log: [logs/oracle-lint-final.log](logs/oracle-lint-final.log)

## oracle-record-first — exit 0

```text
uv run --offline python ops/rebuild-2026-09-29/1.2/oracle/record_reference.py record --reference-root /Users/poonszesen/kg-v3-grammar/.codex-tmp/grammar-reference-1.2 --output tests/fixtures/kaggriculture/grammar-v4-reference.jsonl.gz --manifest tests/fixtures/kaggriculture/grammar-v4-reference.manifest.json
```

Result: test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 120 filtered out; finished in 1.00s; test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 120 filtered out; finished in 0.17s

Log: [logs/oracle-record-first.log](logs/oracle-record-first.log)

## oracle-scratch-bytecode-clean — exit 0

```text
python3 -c 'from pathlib import Path; p=Path(".codex-tmp/grammar-reference-1.2/python/owl/kaggriculture/__pycache__/actor_codec.cpython-312.pyc"); print(p, p.stat().st_size); p.unlink(); p.parent.rmdir()'
```

Result: No test denominator (compile/tool/check command); see log for diagnostics.

Log: [logs/oracle-scratch-bytecode-clean.log](logs/oracle-scratch-bytecode-clean.log)

## oracle-lint-fix2 — exit 1

```text
uvx --offline ruff check --fix ops/rebuild-2026-09-29/1.2/oracle/record_reference.py tests/tools/test_record_grammar_reference.py
```

Result: No test denominator (compile/tool/check command); see log for diagnostics.

Log: [logs/oracle-lint-fix2.log](logs/oracle-lint-fix2.log)

## oracle-format-hardening — exit 0

```text
uvx --offline ruff format ops/rebuild-2026-09-29/1.2/oracle/record_reference.py
```

Result: No test denominator (compile/tool/check command); see log for diagnostics.

Log: [logs/oracle-format-hardening.log](logs/oracle-format-hardening.log)

## oracle-validator-hardening — exit 0

```text
uv run --offline pytest tests/tools/test_record_grammar_reference.py -q
```

Result: 18 passed in 4.81s

Log: [logs/oracle-validator-hardening.log](logs/oracle-validator-hardening.log)

## oracle-format-tests-final — exit 0

```text
uvx --offline ruff format tests/tools/test_record_grammar_reference.py
```

Result: No test denominator (compile/tool/check command); see log for diagnostics.

Log: [logs/oracle-format-tests-final.log](logs/oracle-format-tests-final.log)

## oracle-lint-clean — exit 0

```text
uvx --offline ruff check ops/rebuild-2026-09-29/1.2/oracle/record_reference.py tests/tools/test_record_grammar_reference.py
```

Result: No test denominator (compile/tool/check command); see log for diagnostics.

Log: [logs/oracle-lint-clean.log](logs/oracle-lint-clean.log)

## oracle-record-final — exit 0

```text
uv run --offline python ops/rebuild-2026-09-29/1.2/oracle/record_reference.py record --reference-root /Users/poonszesen/kg-v3-grammar/.codex-tmp/grammar-reference-1.2 --output tests/fixtures/kaggriculture/grammar-v4-reference.jsonl.gz --manifest tests/fixtures/kaggriculture/grammar-v4-reference.manifest.json
```

Result: test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 120 filtered out; finished in 0.94s; test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 120 filtered out; finished in 0.17s

Log: [logs/oracle-record-final.log](logs/oracle-record-final.log)

## oracle-verify-final — exit 0

```text
uv run --offline python ops/rebuild-2026-09-29/1.2/oracle/record_reference.py verify --reference-root /Users/poonszesen/kg-v3-grammar/.codex-tmp/grammar-reference-1.2 --fixture tests/fixtures/kaggriculture/grammar-v4-reference.jsonl.gz --manifest tests/fixtures/kaggriculture/grammar-v4-reference.manifest.json
```

Result: test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 120 filtered out; finished in 0.93s; test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 120 filtered out; finished in 0.17s

Log: [logs/oracle-verify-final.log](logs/oracle-verify-final.log)

## oracle-tooling-final — exit 0

```text
uv run --offline pytest tests/tools/test_record_grammar_reference.py -q
```

Result: 25 passed in 6.56s

Log: [logs/oracle-tooling-final.log](logs/oracle-tooling-final.log)

## serde-promote-add — exit 0

```text
cargo add --offline serde_json@1.0.149
```

Result: No test denominator (compile/tool/check command); see log for diagnostics.

Log: [logs/serde-promote-add.log](logs/serde-promote-add.log)

## serde-promote-remove-dev — exit 0

```text
cargo remove --offline --dev serde_json
```

Result: No test denominator (compile/tool/check command); see log for diagnostics.

Log: [logs/serde-promote-remove-dev.log](logs/serde-promote-remove-dev.log)

## oracle-scratch-final-state — exit 0

```text
python3 -c 'import pathlib,subprocess; root=pathlib.Path(".codex-tmp/grammar-reference-1.2"); print(subprocess.check_output(["git","status","--short","--untracked-files=all"],cwd=root,text=True)); before=pathlib.Path("ops/rebuild-2026-09-29/1.2/oracle/scratch/ffi.rs.before").read_bytes(); append=pathlib.Path("ops/rebuild-2026-09-29/1.2/oracle/reference_harness.rs").read_bytes(); assert (root/"engine_rs/src/ffi.rs").read_bytes()==before+append; assert not list((root/"python/owl/kaggriculture").glob("__pycache__/actor_codec.*.pyc")); print("exact append and no codec bytecode: OK")'
```

Result: No test denominator (compile/tool/check command); see log for diagnostics.

Log: [logs/oracle-scratch-final-state.log](logs/oracle-scratch-final-state.log)

## oracle-sha256-final — exit 0

```text
shasum -a 256 ops/rebuild-2026-09-29/1.2/oracle/record_reference.py ops/rebuild-2026-09-29/1.2/oracle/reference_harness.rs ops/rebuild-2026-09-29/1.2/oracle/scratch/ffi.rs.before ops/rebuild-2026-09-29/1.2/oracle/scratch/append.patch tests/fixtures/kaggriculture/grammar-v4-reference.jsonl.gz tests/fixtures/kaggriculture/grammar-v4-reference.manifest.json
```

Result: No test denominator (compile/tool/check command); see log for diagnostics.

Log: [logs/oracle-sha256-final.log](logs/oracle-sha256-final.log)

## oracle-validator-negative-control — exit 1

```text
python3 -c 'from pathlib import Path; import subprocess,sys; p=Path("ops/rebuild-2026-09-29/1.2/oracle/record_reference.py"); before=p.read_bytes(); s=before.decode(); marker="    exact(manifest, MANIFEST_KEYS, \"manifest\")"; assert marker in s; p.write_text(s.replace(marker,"    return\n"+marker,1)); result=None
try:
 result=subprocess.run(["uv","run","--offline","pytest","tests/tools/test_record_grammar_reference.py","-q"],check=False)
finally:
 p.write_bytes(before); print("validator source restored byte-exactly", flush=True)
assert result is not None
sys.exit(result.returncode)'
```

Result: 24 failed, 1 passed in 4.47s

Log: [logs/oracle-validator-negative-control.log](logs/oracle-validator-negative-control.log)

## oracle-validator-restored-green — exit 0

```text
uv run --offline pytest tests/tools/test_record_grammar_reference.py -q
```

Result: 25 passed in 7.71s

Log: [logs/oracle-validator-restored-green.log](logs/oracle-validator-restored-green.log)

## oracle-validator-explicit-error-control — exit 1

```text
python3 -c 'from pathlib import Path; import subprocess,sys; p=Path("ops/rebuild-2026-09-29/1.2/oracle/record_reference.py"); before=p.read_bytes(); s=before.decode(); marker="    exact(manifest, MANIFEST_KEYS, \"manifest\")"; assert marker in s; p.write_text(s.replace(marker,"    raise ValueError(\"explicit validator error stub control\")\n"+marker,1)); result=None
try:
 result=subprocess.run([".venv/bin/pytest","tests/tools/test_record_grammar_reference.py::test_fixture_validation_and_compression","-q"],check=False)
finally:
 p.write_bytes(before); print("validator source restored byte-exactly", flush=True)
assert result is not None
sys.exit(result.returncode)'
```

Result: 1 failed in 0.17s

Log: [logs/oracle-validator-explicit-error-control.log](logs/oracle-validator-explicit-error-control.log)

## oracle-validator-explicit-error-restored — exit 0

```text
.venv/bin/pytest tests/tools/test_record_grammar_reference.py -q
```

Result: 25 passed in 7.70s

Log: [logs/oracle-validator-explicit-error-restored.log](logs/oracle-validator-explicit-error-restored.log)

## transition-stub-hash — exit 0

```text
shasum -a 256 src/kaggriculture/grammar.rs ops/rebuild-2026-09-29/1.2/controls/transitions-stub.rs.txt
```

Result: No test denominator (compile/tool/check command); see log for diagnostics.

Log: [logs/transition-stub-hash.log](logs/transition-stub-hash.log)

## transitions-red — exit 101

```text
cargo test --locked --lib grammar_tables_match_reference_and_reachable_states -- --test-threads=1
```

Result: test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 158 filtered out; finished in 0.00s

Log: [logs/transitions-red.log](logs/transitions-red.log)

## transitions-green — exit 0

```text
cargo test --locked --lib grammar_tables_match_reference_and_reachable_states -- --test-threads=1
```

Result: test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 158 filtered out; finished in 0.92s

Log: [logs/transitions-green.log](logs/transitions-green.log)

## decode-red — exit 101

```text
cargo test --locked --lib grammar_ -- --test-threads=1
```

Result: test result: FAILED. 2 passed; 3 failed; 0 ignored; 0 measured; 157 filtered out; finished in 1.00s

Log: [logs/decode-red.log](logs/decode-red.log)

## decode-green — exit 0

```text
cargo test --locked --lib grammar_ -- --test-threads=1
```

Result: test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 157 filtered out; finished in 1.05s

Log: [logs/decode-green.log](logs/decode-green.log)

## encode-red — exit 101

```text
cargo test --locked --lib grammar_encode -- --test-threads=1
```

Result: test result: FAILED. 0 passed; 2 failed; 0 ignored; 0 measured; 162 filtered out; finished in 0.00s

Log: [logs/encode-red.log](logs/encode-red.log)

## encode-green — exit 0

```text
cargo test --locked --lib grammar_encode -- --test-threads=1
```

Result: test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 162 filtered out; finished in 0.42s

Log: [logs/encode-green.log](logs/encode-green.log)

## hire-negative — exit 101

```text
python3 ops/rebuild-2026-09-29/1.2/mutate_check.py hire-guard src/kaggriculture/grammar.rs 'token != 1 || self.shape.actors + u16::from(self.hires) < self.shape.hire_limit' true -- cargo test --locked --lib grammar_coupled_hire_exact_distribution -- --test-threads=1
```

Result: test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 164 filtered out; finished in 0.00s

Log: [logs/hire-negative.log](logs/hire-negative.log)

## hire-green — exit 0

```text
cargo test --locked --lib grammar_coupled_hire_exact_distribution -- --test-threads=1
```

Result: test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 164 filtered out; finished in 0.03s

Log: [logs/hire-green.log](logs/hire-green.log)

## replay-walk-negative — exit 101

```text
python3 ops/rebuild-2026-09-29/1.2/mutate_check.py replay-render src/kaggriculture/grammar.rs 'let mut command = vec![Value::from(kind.as_str())];' 'let mut command = vec![Value::from(if kind == UnitKind::Plant { "PASS" } else { kind.as_str() })];' -- cargo test --locked --lib grammar_replays_reference_program -- --test-threads=1
```

Result: test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 165 filtered out; finished in 0.00s

Log: [logs/replay-walk-negative.log](logs/replay-walk-negative.log)

## replay-walk-green — exit 101

```text
cargo test --locked --lib grammar_replays_reference_program -- --test-threads=1
```

Result: test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 165 filtered out; finished in 0.00s

Log: [logs/replay-walk-green.log](logs/replay-walk-green.log)

## replay-walk-negative-corrected — exit 101

```text
python3 ops/rebuild-2026-09-29/1.2/mutate_check.py replay-render-corrected src/kaggriculture/grammar.rs 'let mut command = vec![Value::from(kind.as_str())];' 'let mut command = vec![Value::from(if kind == UnitKind::Plant { "PASS" } else { kind.as_str() })];' -- cargo test --locked --lib grammar_replays_reference_program -- --test-threads=1
```

Result: test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 165 filtered out; finished in 0.00s

Log: [logs/replay-walk-negative-corrected.log](logs/replay-walk-negative-corrected.log)

## replay-walk-green-corrected — exit 0

```text
cargo test --locked --lib grammar_replays_reference_program -- --test-threads=1
```

Result: test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 165 filtered out; finished in 0.54s

Log: [logs/replay-walk-green-corrected.log](logs/replay-walk-green-corrected.log)

## missing-sentinel-negative — exit 101

```text
python3 ops/rebuild-2026-09-29/1.2/mutate_check.py final-sentinel src/kaggriculture/grammar.rs 'if state.is_some() { return Err("program lacks distinct final STOP".into()); }' '' -- cargo test --locked --lib grammar_rejects_malformed_i64 -- --test-threads=1
```

Result: test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 165 filtered out; finished in 0.01s

Log: [logs/missing-sentinel-negative.log](logs/missing-sentinel-negative.log)

## oracle-readonly-final-audit — exit 0

```text
python3 -B -c 'from pathlib import Path; import collections,gzip,hashlib,importlib.util,json,subprocess; p=Path("ops/rebuild-2026-09-29/1.2/oracle/record_reference.py"); spec=importlib.util.spec_from_file_location("readonly_oracle_audit",p); assert spec is not None and spec.loader is not None; m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); f=Path("tests/fixtures/kaggriculture/grammar-v4-reference.jsonl.gz"); mp=f.with_name("grammar-v4-reference.manifest.json"); manifest=json.loads(mp.read_text()); payload=gzip.decompress(f.read_bytes()); m.validate_fixture(payload,manifest); rows=[json.loads(l) for l in payload.splitlines()][1:]; print("counts",json.dumps(manifest["counts"],sort_keys=True)); print("full_market_by_source",json.dumps(dict(collections.Counter(r["source"] for r in rows if r["source"] in {"sampled","replay"} and r["length"]==r["shape"]["actors"]+r["shape"]["order_limit"]+1)),sort_keys=True)); print("manifest_bytes",mp.stat().st_size); print("disagreement_ids",json.dumps([d["id"] for d in manifest["disagreements"]])); print("hashes",json.dumps({str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [p,Path("ops/rebuild-2026-09-29/1.2/oracle/reference_harness.rs"),Path("tests/tools/test_record_grammar_reference.py"),f,mp]},sort_keys=True)); ref=Path(".codex-tmp/grammar-reference-1.2"); status=subprocess.check_output(["git","status","--short","--untracked-files=all"],cwd=ref,text=True); print("scratch_status",repr(status)); assert status==" M engine_rs/src/ffi.rs\n"; original=Path("ops/rebuild-2026-09-29/1.2/oracle/scratch/ffi.rs.before").read_bytes(); harness=Path("ops/rebuild-2026-09-29/1.2/oracle/reference_harness.rs").read_bytes(); assert (ref/"engine_rs/src/ffi.rs").read_bytes()==original+harness; assert not list(ref.rglob("*.pyc")); receipt=json.loads(Path("ops/rebuild-2026-09-29/1.2/checks.json").read_text()); verified=[r for r in receipt if r["id"]=="oracle-verify-final"]; assert len(verified)==1 and verified[0]["exit_code"]==0; print("independent native verify receipt: exit0; strict final provenance and exact append: OK")'
```

Result: No test denominator (compile/tool/check command); see log for diagnostics.

Log: [logs/oracle-readonly-final-audit.log](logs/oracle-readonly-final-audit.log)

## oracle-scratch-ignored-audit — exit 0

```text
python3 -B -c 'from pathlib import Path; import subprocess; root=Path(".codex-tmp/grammar-reference-1.2"); untracked=subprocess.check_output(["git","ls-files","--others","--exclude-standard"],cwd=root,text=True); ignored=subprocess.check_output(["git","ls-files","--others","--ignored","--exclude-standard"],cwd=root,text=True); print("untracked",repr(untracked)); print("ignored",repr(ignored)); assert not untracked and not ignored'
```

Result: No test denominator (compile/tool/check command); see log for diagnostics.

Log: [logs/oracle-scratch-ignored-audit.log](logs/oracle-scratch-ignored-audit.log)

## missing-sentinel-green — exit 0

```text
cargo test --locked --lib grammar_rejects_malformed_i64 -- --test-threads=1
```

Result: test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 165 filtered out; finished in 0.01s

Log: [logs/missing-sentinel-green.log](logs/missing-sentinel-green.log)

## authored-rust-format — exit 0

```text
cargo fmt
```

Result: No test denominator (compile/tool/check command); see log for diagnostics.

Log: [logs/authored-rust-format.log](logs/authored-rust-format.log)

## kernel-initial — exit 0

```text
cargo test --manifest-path engine_rs/Cargo.toml --offline --locked --test grammar_kernel decoded_programs_feed_kernel -- --test-threads=1
```

Result: test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 17 filtered out; finished in 0.00s

Log: [logs/kernel-initial.log](logs/kernel-initial.log)

## ops-helpers-format — exit 0

```text
uvx --offline ruff format ops/rebuild-2026-09-29/1.2/run_check.py ops/rebuild-2026-09-29/1.2/mutate_check.py
```

Result: No test denominator (compile/tool/check command); see log for diagnostics.

Log: [logs/ops-helpers-format.log](logs/ops-helpers-format.log)

## kernel-renderer-red — exit 101

```text
python3 ops/rebuild-2026-09-29/1.2/mutate_check.py kernel-buy-land src/kaggriculture/grammar.rs 'let kind = MarketKind::try_from(frame[Slot::MarketKind as usize])?;' 'let kind = MarketKind::try_from(frame[Slot::MarketKind as usize])?; let kind = if kind == MarketKind::BuyLand { MarketKind::Hire } else { kind };' -- cargo test --manifest-path engine_rs/Cargo.toml --offline --locked --test grammar_kernel decoded_programs_feed_kernel -- --test-threads=1
```

Result: test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 17 filtered out; finished in 0.00s

Log: [logs/kernel-renderer-red.log](logs/kernel-renderer-red.log)

## ops-helpers-lint — exit 0

```text
uvx --offline ruff check --fix ops/rebuild-2026-09-29/1.2/run_check.py ops/rebuild-2026-09-29/1.2/mutate_check.py
```

Result: No test denominator (compile/tool/check command); see log for diagnostics.

Log: [logs/ops-helpers-lint.log](logs/ops-helpers-lint.log)

## kernel-green — exit 0

```text
cargo test --manifest-path engine_rs/Cargo.toml --offline --locked --test grammar_kernel -- --test-threads=1
```

Result: test result: ok. 18 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 13.57s

Log: [logs/kernel-green.log](logs/kernel-green.log)

## kernel-format — exit 0

```text
rustfmt --edition 2024 engine_rs/tests/grammar_kernel.rs
```

Result: No test denominator (compile/tool/check command); see log for diagnostics.

Log: [logs/kernel-format.log](logs/kernel-format.log)

## kernel-clippy — exit 0

```text
cargo clippy --manifest-path engine_rs/Cargo.toml --offline --all-targets --locked -- -D warnings -A clippy::too_many_arguments -A clippy::collapsible_if -A clippy::needless_range_loop
```

Result: No test denominator (compile/tool/check command); see log for diagnostics.

Log: [logs/kernel-clippy.log](logs/kernel-clippy.log)

## final-branch — exit 0

```text
git branch --show-current
```

Result: No test denominator (compile/tool/check command); see log for diagnostics.

Log: [logs/final-branch.log](logs/final-branch.log)

## final-reference-pin — exit 0

```text
git rev-parse kg/reference-2026-09-29
```

Result: No test denominator (compile/tool/check command); see log for diagnostics.

Log: [logs/final-reference-pin.log](logs/final-reference-pin.log)

## root-serde-features — exit 0

```text
cargo tree --offline --locked -e features -i serde_json
```

Result: No test denominator (compile/tool/check command); see log for diagnostics.

Log: [logs/root-serde-features.log](logs/root-serde-features.log)

## engine-serde-features — exit 0

```text
cargo tree --manifest-path engine_rs/Cargo.toml --offline --locked -e features -i serde_json
```

Result: No test denominator (compile/tool/check command); see log for diagnostics.

Log: [logs/engine-serde-features.log](logs/engine-serde-features.log)

## final-kernel-hash — exit 0

```text
shasum -a 256 engine_rs/tests/grammar_kernel.rs
```

Result: No test denominator (compile/tool/check command); see log for diagnostics.

Log: [logs/final-kernel-hash.log](logs/final-kernel-hash.log)

## register-final-kernel-hash — exit 0

```text
python3 -c 'from pathlib import Path; import hashlib,json; p=Path('"'"'engine_rs/TRIM_MANIFEST.json'"'"'); m=json.loads(p.read_text()); row=next(r for r in m['"'"'authored'"'"'] if r['"'"'path'"'"']=='"'"'engine_rs/tests/grammar_kernel.rs'"'"'); row['"'"'sha256'"'"']=hashlib.sha256(Path(row['"'"'path'"'"']).read_bytes()).hexdigest(); print(row); p.write_text(json.dumps(m,indent=2)+'"'"'\n'"'"')'
```

Result: No test denominator (compile/tool/check command); see log for diagnostics.

Log: [logs/register-final-kernel-hash.log](logs/register-final-kernel-hash.log)

## final-trim — exit 0

```text
uv run --offline python scripts/check_engine_trim.py
```

Result: engine trim manifest: OK

Log: [logs/final-trim.log](logs/final-trim.log)

## final-tool-pytest — exit 0

```text
uv run --offline pytest tests/tools/test_check_engine_trim.py tests/tools/test_record_grammar_reference.py -q
```

Result: 87 passed in 9.72s

Log: [logs/final-tool-pytest.log](logs/final-tool-pytest.log)

## final-root-grammar — exit 0

```text
cargo test --locked --lib grammar_ -- --test-threads=1
```

Result: test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured; 157 filtered out; finished in 1.94s

Log: [logs/final-root-grammar.log](logs/final-root-grammar.log)

## final-engine-grammar — exit 0

```text
cargo test --manifest-path engine_rs/Cargo.toml --locked --test grammar_kernel -- --test-threads=1
```

Result: test result: ok. 18 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 13.45s

Log: [logs/final-engine-grammar.log](logs/final-engine-grammar.log)

## final-orbit-python — exit 2

```text
uv run --offline --extra reference pytest tests/owl tests/scripts tests/tools -m 'not slow' -q
```

Result: No test denominator (compile/tool/check command); see log for diagnostics.

Log: [logs/final-orbit-python.log](logs/final-orbit-python.log)

## final-orbit-python-existing-deps — exit 0

```text
uv run --offline pytest tests/owl tests/scripts tests/tools -m 'not slow' -q
```

Result: 1045 passed, 3 skipped in 39.06s

Log: [logs/final-orbit-python-existing-deps.log](logs/final-orbit-python-existing-deps.log)

## final-root-cargo — exit 0

```text
cargo test --locked
```

Result: test result: ok. 164 passed; 0 failed; 2 ignored; 0 measured; 0 filtered out; finished in 0.98s

Log: [logs/final-root-cargo.log](logs/final-root-cargo.log)

## final-engine-cargo — exit 0

```text
cargo test --manifest-path engine_rs/Cargo.toml --locked
```

Result: test result: ok. 41 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.11s; test result: ok. 18 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 6.25s; test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s; test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 2.53s; test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

Log: [logs/final-engine-cargo.log](logs/final-engine-cargo.log)

## rs-prepare-first — exit 1

```text
uvx --offline --from rust-just just rs-prepare
```

Result: engine trim manifest: OK

Log: [logs/rs-prepare-first.log](logs/rs-prepare-first.log)

## shared-format-root — exit 0

```text
cargo fmt
```

Result: No test denominator (compile/tool/check command); see log for diagnostics.

Log: [logs/shared-format-root.log](logs/shared-format-root.log)

## shared-format-engine-check — exit 0

```text
cargo fmt --manifest-path engine_rs/Cargo.toml --check
```

Result: No test denominator (compile/tool/check command); see log for diagnostics.

Log: [logs/shared-format-engine-check.log](logs/shared-format-engine-check.log)

## rs-prepare-common-format — exit 0

```text
uvx --offline --from rust-just just rs-prepare
```

Result: engine trim manifest: OK; test result: ok. 164 passed; 0 failed; 2 ignored; 0 measured; 0 filtered out; finished in 0.96s; test result: ok. 41 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.09s; test result: ok. 18 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 6.32s; test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s; test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 2.45s; test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s; No doc updates required

Log: [logs/rs-prepare-common-format.log](logs/rs-prepare-common-format.log)

## py-prepare-final — exit 0

```text
uvx --offline --from rust-just just py-prepare
```

Result: Success: no issues found in 51 source files; ======================= 1045 passed, 3 skipped in 19.24s =======================; No doc updates required

Log: [logs/py-prepare-final.log](logs/py-prepare-final.log)

## docs-fresh-final — exit 0

```text
uvx --offline --from rust-just just docs-fresh
```

Result: No doc updates required

Log: [logs/docs-fresh-final.log](logs/docs-fresh-final.log)

## trim-after-prepare — exit 0

```text
uv run --offline python scripts/check_engine_trim.py
```

Result: engine trim manifest: OK

Log: [logs/trim-after-prepare.log](logs/trim-after-prepare.log)

## prepare-final — exit 0

```text
uvx --offline --from rust-just just prepare
```

Result: engine trim manifest: OK; Success: no issues found in 51 source files; test result: ok. 164 passed; 0 failed; 2 ignored; 0 measured; 0 filtered out; finished in 0.99s; test result: ok. 41 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.09s; test result: ok. 18 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 6.15s; test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s; test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 2.37s; test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s; ======================= 1045 passed, 3 skipped in 19.08s =======================; No doc updates required

Log: [logs/prepare-final.log](logs/prepare-final.log)

## engine-bytes-guard — exit 0

```text
git diff --exit-code -- engine_rs/src/lib.rs engine_rs/src/py_random.rs engine_rs/src/econ_attrib.rs engine_rs/Cargo.toml engine_rs/Cargo.lock engine_rs/VENDORED_FROM.md src/rules_engine/generation.rs
```

Result: No test denominator (compile/tool/check command); see log for diagnostics.

Log: [logs/engine-bytes-guard.log](logs/engine-bytes-guard.log)

## engine-bytes-from-task-start — exit 0

```text
python3 -c 'import json,subprocess,sys; m=json.load(open('"'"'engine_rs/TRIM_MANIFEST.json'"'"')); paths=[r['"'"'path'"'"'] for r in m['"'"'retained'"'"']]+['"'"'src/rules_engine/generation.rs'"'"']; sys.exit(subprocess.run(['"'"'git'"'"','"'"'diff'"'"','"'"'--exit-code'"'"','"'"'a88150c'"'"','"'"'--'"'"',*paths],check=False).returncode)'
```

Result: No test denominator (compile/tool/check command); see log for diagnostics.

Log: [logs/engine-bytes-from-task-start.log](logs/engine-bytes-from-task-start.log)

## final-staging-guard — exit 0

```text
git diff --cached --exit-code
```

Result: No test denominator (compile/tool/check command); see log for diagnostics.

Log: [logs/final-staging-guard.log](logs/final-staging-guard.log)

## final-docs-lint — exit 0

```text
uvx --offline --from rust-just just docs-lint
```

Result: No test denominator (compile/tool/check command); see log for diagnostics.

Log: [logs/final-docs-lint.log](logs/final-docs-lint.log)

## final-whitespace — exit 0

```text
git diff --check
```

Result: No test denominator (compile/tool/check command); see log for diagnostics.

Log: [logs/final-whitespace.log](logs/final-whitespace.log)

## final-status — exit 0

```text
git status --short --untracked-files=all
```

Result: No test denominator (compile/tool/check command); see log for diagnostics.

Log: [logs/final-status.log](logs/final-status.log)
