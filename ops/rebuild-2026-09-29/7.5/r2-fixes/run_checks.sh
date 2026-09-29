#!/usr/bin/env bash
# Task 7.5 r2 fix checks. Mac, CPU only; each command measured with /usr/bin/time -l
# (max RSS of the largest single child process, not a process-tree sum).
set -u
cd "$(git rev-parse --show-toplevel)"
export CARGO_BUILD_JOBS=2 OMP_NUM_THREADS=2
D=ops/rebuild-2026-09-29/7.5/r2-fixes
run() {
  name=$1; shift
  echo "== $name: $*" | tee "$D/$name.log"
  /usr/bin/time -l "$@" >>"$D/$name.log" 2>&1
  echo "exit $?" >>"$D/$name.log"
}
run engine-fmt cargo fmt --manifest-path engine_rs/Cargo.toml --check
run engine-clippy cargo clippy --offline --manifest-path engine_rs/Cargo.toml --all-targets --locked -- -D warnings -A clippy::too_many_arguments -A clippy::collapsible_if -A clippy::needless_range_loop
run engine-tests cargo test --locked --offline --manifest-path engine_rs/Cargo.toml
run opponents-tests cargo test --offline --manifest-path opponents_rs/Cargo.toml --locked
run engine-trim uv run --offline --no-sync python scripts/check_engine_trim.py
run opponent-import uv run --offline --no-sync python scripts/check_opponent_import.py
run pytest-parity-custody uv run --offline --no-sync pytest -q -m "not slow" tests/scripts/test_kaggriculture_parity.py tests/tools/test_check_engine_trim.py tests/tools/test_check_opponent_import.py tests/owl/kaggriculture/test_opponents.py
run pytest-env uv run --offline --no-sync pytest -q -m "not slow" tests/kaggriculture/test_env_reference.py tests/kaggriculture/test_native_env.py tests/kaggriculture/test_env.py
run docs-fresh uv run --offline --no-sync python scripts/check_doc_freshness.py
