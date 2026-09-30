#!/bin/bash
set -euo pipefail
export PATH=/root/.cargo/bin:$PATH
export CARGO_BUILD_JOBS=8
export CARGO_TARGET_DIR=/root/ship/target
export PYO3_PYTHON=/root/ship/venv/bin/python
unset RUSTFLAGS CARGO_ENCODED_RUSTFLAGS
cd /root/ship/src
date -u +%FT%TZ
rustc --version; cargo --version
/root/ship/venv/bin/maturin --version
/root/ship/venv/bin/maturin build --release --compatibility linux -i /root/ship/venv/bin/python --out /root/ship/wheels
date -u +%FT%TZ
