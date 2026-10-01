#!/usr/bin/env bash
# Idempotent one-time setup of the Linux anchor-eval harness on a CPU pod.
# usage: setup_pod.sh            (pod address and pins in pod.env)
#
#   uv + rustup (minimal)                      /root/.local/bin, /root/.cargo/bin
#   Python 3.11 venv: torch 2.6.0+cpu (PyTorch CPU index), kaggle-environments 1.32.7,
#     numpy 2.4.6, pyyaml 6.0.3                /root/anchor-eval/venv
#   fixed-shop kaggle_environments (PYTHONPATH, as on the Mac)
#                                              /root/anchor-eval/kenv-fixedshop
#   v2 cha22_agent / v56_agent built from `git archive $V2_COMMIT engine_rs`
#                                              /root/anchor-eval/bin
#   harness + Linux anchor wrappers            /root/anchor-eval/{harness,anchors}
# Fails if the engine or binary SHA-256 differ from the pins in pod.env.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/pod.env"
R=/root/anchor-eval
mkdir -p "$INPUT_CACHE"
KENV_TGZ="$INPUT_CACHE/kenv-fixedshop.tar.gz"
V2_TGZ="$INPUT_CACHE/engine_rs-${V2_COMMIT:0:8}.tar.gz"
[[ -f "$KENV_TGZ" ]] || { echo "missing $KENV_TGZ (tar of the fixed-shop kenv dir, see README)" >&2; exit 1; }
[[ -f "$V2_TGZ" ]] || git -C "$V2_REPO" archive --format=tar.gz -o "$V2_TGZ" "$V2_COMMIT" engine_rs

pod "mkdir -p $R/setup $R/bin $R/harness $R/anchors $R/pkgs $R/games"
pod_put "$KENV_TGZ" "$V2_TGZ" "$R/setup/"
pod "set -euo pipefail; export PATH=/root/.local/bin:/root/.cargo/bin:\$PATH; cd $R
command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh) > setup/uv-install.log 2>&1
command -v cargo >/dev/null || (curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal) > setup/rustup.log 2>&1
if ! venv/bin/python -c 'import torch, numpy, yaml, kaggle_environments as k; assert torch.__version__ == \"2.6.0+cpu\" and numpy.__version__ == \"2.4.6\" and k.__version__ == \"1.32.7\"' 2>/dev/null; then
  uv venv -q --python 3.11 venv
  uv pip install -q --python venv/bin/python torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
  uv pip install -q --python venv/bin/python kaggle-environments==1.32.7 numpy==2.4.6 pyyaml==6.0.3
fi
rm -rf kenv-fixedshop && tar -xzf setup/kenv-fixedshop.tar.gz
echo '$KENV_ENGINE_SHA256  kenv-fixedshop/kaggle_environments/envs/kaggriculture/kaggriculture.py' | sha256sum -c --quiet
if ! (cd bin && printf '%s  cha22_agent\n%s  v56_agent\n' $CHA22_AGENT_SHA256 $V56_AGENT_SHA256 | sha256sum -c --quiet 2>/dev/null); then
  rm -rf v2-engine && mkdir v2-engine && tar -xzf setup/$(basename "$V2_TGZ") -C v2-engine
  (cd v2-engine/engine_rs && cargo build --release --locked --bin v56_agent --bin cha22_agent) > setup/cargo-build.log 2>&1
  cp v2-engine/engine_rs/target/release/{cha22,v56}_agent bin/
  (cd bin && printf '%s  cha22_agent\n%s  v56_agent\n' $CHA22_AGENT_SHA256 $V56_AGENT_SHA256 | sha256sum -c)
fi
"
pod_put -r "$HERE/pod/harness" "$HERE/pod/anchors" "$R/"
pod "set -e; cd $R; export PATH=/root/.local/bin:/root/.cargo/bin:\$PATH
PYTHONPATH=$R/kenv-fixedshop venv/bin/python -c 'import sys, platform, torch, numpy, kaggle_environments as k; print(\"python\", platform.python_version(), \"torch\", torch.__version__, \"numpy\", numpy.__version__, \"kaggle_environments\", k.__version__, k.__file__)'
uv --version; rustc --version; cargo --version
sha256sum kenv-fixedshop/kaggle_environments/envs/kaggriculture/kaggriculture.py bin/cha22_agent bin/v56_agent
nproc; free -g | sed -n 2p"
