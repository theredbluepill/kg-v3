#!/bin/sh
# Direct just prepare equivalents: just/uvx are not installed in this worktree,
# and network is unavailable. Use already cached executables and dependencies.
set -eu
export UV_CACHE_DIR=/private/tmp/kg-sps-uv-cache
export UV_NO_SYNC=1
export UV_OFFLINE=1
export OMP_NUM_THREADS=2
export CARGO_BUILD_JOBS=2
ruff=/Users/poonszesen/.cache/uv/binaries-v0/ruff/0.15.10/aarch64-apple-darwin/ruff
markdown=/Users/poonszesen/.cache/uv/archive-v0/P95gkolteNpZraf1z4YTp/bin/pymarkdown
set -x
uv run maturin develop --skip-install --offline
uv run python scripts/check_engine_trim.py
uv run python scripts/check_opponent_import.py
cargo fmt
cargo fmt --manifest-path engine_rs/Cargo.toml --check
cargo fmt --manifest-path opponents_rs/Cargo.toml --check
"$ruff" check python scripts tests --select I --fix
"$ruff" format python scripts tests
cargo clippy --all-targets --offline -- -D warnings
cargo clippy --no-default-features --offline -- -D warnings
cargo clippy --manifest-path engine_rs/Cargo.toml --all-targets --offline --locked -- -D warnings -A clippy::too_many_arguments -A clippy::collapsible_if -A clippy::needless_range_loop
cargo clippy --manifest-path opponents_rs/Cargo.toml --all-targets --offline --locked -- -D warnings
uv run python scripts/check_python_311_syntax.py
"$ruff" check python scripts tests
"$markdown" scan ./*.md
"$markdown" scan --recurse python scripts tests src docs
uv run python -m mypy python scripts
cargo test --offline
cargo test --manifest-path engine_rs/Cargo.toml --offline --locked
cargo test --manifest-path opponents_rs/Cargo.toml --offline --locked
uv run python -m pytest tests
uv run python scripts/check_doc_freshness.py
