#!/usr/bin/env bash
# Plan 6.3b: set up a fresh 8x RTX PRO 6000 pod for Kaggriculture v3 (DRAFT).
#
# Run ON THE POD from the repository checkout, after stage_from_mac.sh has
# delivered the source bundle, the BC best and the W&B credential:
#   cd "$KG_REPO" && bash ops/rebuild-2026-09-29/8rank-run/setup.sh
#
# It installs nothing system-wide except rustup and uv when they are missing,
# never changes the driver, CUDA or security settings, and never prints the
# W&B credential. It stops at the first failed gate. Receipts go to
# $KG_RECEIPTS/setup (default /workspace/kg-v3-receipts/setup).
#
# Gates, in order:
#   1. host facts (GPUs, driver, vCPUs, cgroup quota, memory, disk, topology);
#   2. the NVIDIA driver is the compile gate's probed 595.91.07 on all 8 GPUs;
#      another driver stops here: repeat the ATEN-only GEMM A/B first
#      (ops/rebuild-2026-09-29/run-statements/aten-gemm-ab.md,
#      cookbook/decisions/kaggriculture-compiles-gemms-with-cublas-only.md);
#   3. rustup (toolchain from rust-toolchain.toml) and uv;
#   4. uv sync --frozen --group dev --extra flash-attn;
#   5. maturin release build (run_ppo asserts a release build);
#   6. the compile-stack check (torch 2.9.0, triton 3.5.0, driver);
#   7. flash-attn on sm_120 on every GPU (check_flash_all_gpus.py) and the
#      flash-attn CUDA tests;
#   8. the W&B credential (cookbook/workflows/install-the-wandb-credential-before-any-pod-launch.md, step 2);
#   9. the BC best's SHA-256;
#  10. whether nsys is present (recorded; a missing nsys limits the
#      all-reduce claim and is not a gate).
set -euo pipefail

EXPECTED_GPUS=8
PROBED_DRIVER=595.91.07
BC_SHA256=fd8545872aca59c70e273e9655055e1104cd719588364f463ae87b0d488e6f51
REPO=$(git rev-parse --show-toplevel)
PKG=ops/rebuild-2026-09-29/8rank-run
BC_BEST=${KG_BC_BEST:-/workspace/bc-best/checkpoint_bc_best.pt}
RECEIPTS=${KG_RECEIPTS:-/workspace/kg-v3-receipts}/setup
mkdir -p "$RECEIPTS"
cd "$REPO"

step() { printf '\n== %s (%s)\n' "$1" "$(date -u +%FT%TZ)" | tee -a "$RECEIPTS/setup.log"; }
fail() { printf 'SETUP STOP: %s\n' "$1" | tee -a "$RECEIPTS/setup.log" >&2; exit 1; }

step "1. host facts"
{
  echo "repo=$REPO"
  echo "head=$(git rev-parse HEAD)"
  echo "porcelain_lines=$(git status --porcelain | wc -l)"
  echo "nproc=$(nproc)"
  echo "cgroup_cpu_max=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo unavailable)"
  free -g
  df -h / /workspace 2>/dev/null || df -h /
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,pcie.link.gen.max,pcie.link.width.max --format=csv
} > "$RECEIPTS/host.txt" 2>&1
cat "$RECEIPTS/host.txt"
nvidia-smi topo -m > "$RECEIPTS/topo.txt" 2>&1 || true
cat "$RECEIPTS/topo.txt"

step "2. driver gate"
mapfile -t drivers < <(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
[ "${#drivers[@]}" -eq "$EXPECTED_GPUS" ] \
  || fail "expected $EXPECTED_GPUS GPUs, nvidia-smi lists ${#drivers[@]}"
for d in "${drivers[@]}"; do
  if [ "$(echo "$d" | tr -d ' ')" != "$PROBED_DRIVER" ]; then
    fail "NVIDIA driver $d is not the probed $PROBED_DRIVER. run_ppo's compile gate \
(python/owl/model/compile_gemm.py) will refuse it. Do not change the driver: repeat \
the ATEN-only GEMM A/B on this stack first (ops/rebuild-2026-09-29/run-statements/aten-gemm-ab.md, \
cookbook/decisions/kaggriculture-compiles-gemms-with-cublas-only.md), or pick a pod with $PROBED_DRIVER."
  fi
done
echo "driver $PROBED_DRIVER on all $EXPECTED_GPUS GPUs" | tee -a "$RECEIPTS/setup.log"

step "3. rustup and uv"
if ! command -v rustup >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path --default-toolchain none
fi
export PATH="$HOME/.cargo/bin:$PATH"
rustup show active-toolchain || rustup toolchain install "$(sed -n 's/^channel = "\(.*\)"/\1/p' rust-toolchain.toml)"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
{ rustc --version; cargo --version; uv --version; } | tee "$RECEIPTS/toolchain.txt"

step "4. uv sync (frozen, dev group, flash-attn extra)"
# flash-attn builds without isolation (pyproject no-build-isolation-package),
# so torch must be installed first; the Phase 6.0 setup ran the same two steps.
uv sync --frozen --group dev 2>&1 | tail -20 | tee "$RECEIPTS/uv-sync.log"
uv sync --frozen --group dev --extra flash-attn 2>&1 | tail -20 | tee -a "$RECEIPTS/uv-sync.log"
uv run --no-sync python -c 'import flash_attn, torch; print("flash_attn", flash_attn.__version__, "torch", torch.__version__)' \
  | tee -a "$RECEIPTS/uv-sync.log"
sha256sum uv.lock pyproject.toml Cargo.lock | tee "$RECEIPTS/lockfiles.sha256"

step "5. maturin release build"
uv run --no-sync maturin develop --release 2>&1 | tail -5 | tee "$RECEIPTS/maturin.log"
uv run --no-sync python -c 'from owl.rs import assert_release_build; assert_release_build(); print("release build ok")' \
  | tee -a "$RECEIPTS/maturin.log"

step "6. compile-stack check"
uv run --no-sync python - <<'PY' | tee "$RECEIPTS/compile-stack.txt"
from owl.model.compile_gemm import check_compile_stack, installed_compile_stack
print(check_compile_stack(installed_compile_stack()))
PY

step "7. flash-attn on sm_120, every GPU"
uv run --no-sync python "$PKG/check_flash_all_gpus.py" --expect-gpus "$EXPECTED_GPUS" \
  | tee "$RECEIPTS/flash-all-gpus.json"
CUDA_VISIBLE_DEVICES=0 uv run --no-sync pytest -q tests/owl/model/test_attn.py 2>&1 \
  | tail -5 | tee "$RECEIPTS/test-attn.log"
grep -q " skipped" "$RECEIPTS/test-attn.log" && fail "flash-attn CUDA tests skipped on a GPU host"

step "8. W&B credential (never printed)"
[ -f "$HOME/.netrc" ] || fail "no ~/.netrc: install it from the Mac with stage_from_mac.sh (credential Workflow step 1)"
[ "$(stat -c %a "$HOME/.netrc")" = 600 ] || fail "~/.netrc is not mode 600"
[ -z "${WANDB_MODE:-}" ] || fail "WANDB_MODE is set in the environment; unset it"
uv run --no-sync python -c 'import wandb; print("wandb.Api ok, entity:", wandb.Api(timeout=30).default_entity)' \
  | tee "$RECEIPTS/wandb-check.txt"

step "9. BC best"
[ -f "$BC_BEST" ] || fail "BC best missing at $BC_BEST (copy_bc_best.sh)"
echo "$BC_SHA256  $BC_BEST" | sha256sum -c - | tee "$RECEIPTS/bc-best.sha256.check"

step "10. nsys (recorded, not a gate)"
{ command -v nsys && nsys --version; } > "$RECEIPTS/nsys.txt" 2>&1 \
  || echo "nsys absent: the all-reduce share cannot use a timeline; see run-statement-6.3b.md" \
  | tee "$RECEIPTS/nsys.txt"

step "SETUP PASS"
sha256sum "$RECEIPTS"/* > "$RECEIPTS.SHA256SUMS" 2>/dev/null || true
