#!/bin/bash
set -euxo pipefail
cd /root/kg-v3
if ! command -v rustup >/dev/null && [ ! -x /root/.cargo/bin/rustup ]; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none
fi
source /root/.cargo/env
rustup show active-toolchain || rustup toolchain install
rustc -V; cargo -V
uv --version
time UV_LINK_MODE=copy uv sync --frozen --group dev --extra flash-attn
time uv run maturin develop --release
echo SETUP_DONE
