#!/bin/bash
# Sprint pod bootstrap: runs ON the new pod, idempotent, fail-fast.
#
#   bash /root/sprint-kit/bootstrap.sh --bundle /root/sprint.bundle \
#        [--sha 07c8fc9972297f28e0414da752700194a90d4fc3] [--expect-gpus 8] [--repo /root/kg-v3]
#        [--allow-unprobed-driver]
#
# --sha defaults to the pinned sprint source 07c8fc99 (integration main whose preset hash
# b7fa7f9d... matches the live run pcy5knet). Another --sha is a source change: launch.sh
# still refuses a preset whose hash differs unless --allow-recipe-drift (owner's yes).
# Steps (each safe to rerun): GPU inventory and driver gate -> repo from the git
# bundle (detached at --sha; must contain 07c8fc99 and the live anchor 0f70773)
# -> rustup -> uv 0.9.0 -> uv sync --frozen --group dev --extra flash-attn
# (UV_LINK_MODE=copy) -> uv run maturin develop --release -> version, release
# build and compile-stack checks with .venv/bin/python -> /root/sweep-cache ->
# CPU/NUMA topology (non-fatal: a failure is recorded and points to launch.sh --numa off)
# -> W&B credential check (never reads the key).
# The W&B key is NOT handled here: mac_side.md step 3 pipes only the
# api.wandb.ai netrc entry over stdin.
# Receipts: /root/receipts/sprint-bootstrap/<UTC stamp>/ (log, versions, topology).
set -euo pipefail

PINNED_SHA=07c8fc9972297f28e0414da752700194a90d4fc3   # the sprint source (mac_side.md step 1)
BUNDLE="" SHA=$PINNED_SHA EXPECT_GPUS=8 REPO=/root/kg-v3 ALLOW_UNPROBED_DRIVER=0
MAIN_FLOOR=07c8fc9972297f28e0414da752700194a90d4fc3   # integration main containing the recipe
ANCHOR=0f707731                                          # live run pcy5knet's code (/root/kg-v3-anchor)
PROBED_DRIVER_FALLBACK=595.91.07
UV_VERSION=0.9.0
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

usage() { sed -n '2,20p' "$0"; exit 2; }
while [ $# -gt 0 ]; do
  case "$1" in
    --bundle) BUNDLE=$2; shift 2 ;;
    --sha) SHA=$2; shift 2 ;;
    --expect-gpus) EXPECT_GPUS=$2; shift 2 ;;
    --repo) REPO=$2; shift 2 ;;
    --allow-unprobed-driver) ALLOW_UNPROBED_DRIVER=1; shift ;;
    -h|--help) usage ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done
[ -n "$BUNDLE" ] || { echo "need --bundle" >&2; usage; }
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "--sha must be a full 40-hex commit id, got '$SHA'" >&2; exit 2; }
[[ "$EXPECT_GPUS" =~ ^[1-9][0-9]*$ ]] || { echo "--expect-gpus must be a positive integer" >&2; exit 2; }

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
R=/root/receipts/sprint-bootstrap/$STAMP
mkdir -p "$R"
exec > >(tee -a "$R/bootstrap.log") 2>&1
step() { echo; echo "== $(date -u +%FT%TZ) $*"; }
die() { echo "BOOTSTRAP FAILED: $*" >&2; exit 1; }
trap 'echo "BOOTSTRAP FAILED at line $LINENO (exit $?); fix and rerun (idempotent)" >&2' ERR

step "1. GPUs and driver"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found: not a GPU pod or no driver in the container"
nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,power.limit,driver_version --format=csv | tee "$R/gpus.csv"
nvidia-smi > "$R/nvidia-smi.txt"
nvidia-smi topo -m > "$R/nvidia-topo.txt" 2>&1 || true
N_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
[ "$N_GPUS" = "$EXPECT_GPUS" ] || die "found $N_GPUS GPUs, expected $EXPECT_GPUS (pass --expect-gpus N if the owner's pod differs)"
DRIVERS=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sort -u)
[ "$(echo "$DRIVERS" | wc -l)" = 1 ] || die "mixed driver versions: $DRIVERS"
DRIVER=$DRIVERS
MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1)
APPS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
echo "gpus=$N_GPUS driver=$DRIVER min_memory_mib=$MEM_MIB compute_apps=$APPS"
[ "$APPS" = 0 ] || echo "WARNING: $APPS compute app(s) already on the GPUs; launch.sh will refuse until they are gone"

step "2. Repository from the bundle (detached at $SHA)"
command -v git >/dev/null || die "git not found"
[ -f "$BUNDLE" ] || die "bundle $BUNDLE not found (mac_side.md step 2 copies it)"
sha256sum "$BUNDLE" | tee "$R/bundle.sha256"
if [ ! -d "$REPO/.git" ]; then
  [ ! -e "$REPO" ] || die "$REPO exists but is not a git checkout; move it away first"
  git init -q "$REPO"
fi
cd "$REPO"
# verify needs a repository (it checks the bundle's prerequisites against it; a full bundle has none)
git bundle verify "$BUNDLE" > "$R/bundle-verify.txt" 2>&1 || { cat "$R/bundle-verify.txt"; die "git bundle verify failed"; }
git fetch -q "$BUNDLE" '+refs/*:refs/sprint-bundle/*'
git cat-file -e "$SHA^{commit}" 2>/dev/null || die "commit $SHA is not in the bundle"
git merge-base --is-ancestor "$MAIN_FLOOR" "$SHA" || die "$SHA does not contain integration main $MAIN_FLOOR"
git merge-base --is-ancestor "$ANCHOR" "$SHA" || die "$SHA does not contain the live run's anchor $ANCHOR"
HEAD_NOW=$(git rev-parse -q --verify HEAD || echo none)
if [ "$HEAD_NOW" != "$SHA" ]; then
  # Only untracked build products may be present; tracked edits refuse.
  [ -z "$(git status --porcelain --untracked-files=no 2>/dev/null)" ] || die "$REPO has tracked changes; refusing to switch to $SHA"
  git -c advice.detachedHead=false checkout -q --detach "$SHA"
fi
git rev-parse HEAD | tee "$R/git_head.txt"
echo "tracked changes: $(git status --porcelain --untracked-files=no | wc -l)" | tee -a "$R/git_head.txt"
[ -f configs/kaggriculture_4rank_margin.yaml ] || die "configs/kaggriculture_4rank_margin.yaml missing at $SHA"

step "3. Compile-stack driver gate (python/owl/model/compile_gemm.py at $SHA)"
PROBED=$(sed -n 's/^[[:space:]]*nvidia_drivers=(\(.*\)),*$/\1/p' python/owl/model/compile_gemm.py | tr -d '"'"'"' ' | tr ',' ' ' | xargs)
[ -n "$PROBED" ] || { echo "could not read the probed driver list; assuming $PROBED_DRIVER_FALLBACK"; PROBED=$PROBED_DRIVER_FALLBACK; }
echo "probed drivers: $PROBED; installed: $DRIVER"
DRIVER_OK=0
for d in $PROBED; do [ "$d" = "$DRIVER" ] && DRIVER_OK=1; done
if [ $DRIVER_OK = 0 ]; then
  cat <<EOF

*** DRIVER GATE WOULD REFUSE THE LAUNCH ***
run_ppo calls check_compile_stack() at startup for the compiled Kaggriculture trunk
(rl.model_compile=trunk, the recipe) and raises:
  "unprobed NVIDIA driver $DRIVER for compiled Kaggriculture regions (probed: $PROBED) ..."
The 4-GPU pod abl4mvr5w1mmn4 passed because its driver IS 595.91.07 (env/pod/hardware.txt).
There is no override flag. Options, for the owner to choose (do NOT upgrade/downgrade drivers):
  A. Replace the pod with one whose driver is $PROBED (same GPU type / data center as
     abl4mvr5w1mmn4, RTX PRO 6000 Blackwell Server Edition), then rerun this script. Recipe-identical.
  B. Qualify this driver: repeat the ATEN-only GEMM A/B (ops/rebuild-2026-09-29/results.md,
     "ATEN-only GEMM A/B") on this pod, add "$DRIVER" to KAGGRICULTURE_PROBED_COMPILE_STACK on a
     branch with its cookbook record, re-bundle, rerun this script. Costs the A/B time.
  C. Run eager: launch.sh --extra 'rl.model_compile=none' skips the gate, but it is a recipe
     change (throughput and eager-vs-compiled numerics) and needs the owner's yes.
To finish the build anyway (for B or C), rerun with --allow-unprobed-driver.
EOF
  [ "$ALLOW_UNPROBED_DRIVER" = 1 ] || exit 3
  echo "continuing because --allow-unprobed-driver was given; launch.sh will still refuse without an eager override"
fi

step "4. Rust toolchain"
if ! command -v rustup >/dev/null && [ ! -x /root/.cargo/bin/rustup ]; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none
fi
# shellcheck disable=SC1091
source /root/.cargo/env
rustup show active-toolchain >/dev/null 2>&1 || rustup toolchain install
rustc -V; cargo -V

step "5. uv"
if ! command -v uv >/dev/null; then
  curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | sh
  export PATH="/root/.local/bin:$PATH"
fi
command -v uv >/dev/null || die "uv install failed"
uv --version
for tool in rsync; do
  command -v "$tool" >/dev/null || { echo "installing $tool (the Mac copy-off needs it)"; apt-get update -qq && apt-get install -y -qq "$tool"; }
done

step "6. uv sync (frozen, dev + flash-attn)"
time UV_LINK_MODE=copy uv sync --frozen --group dev --extra flash-attn

step "7. maturin develop --release"
time uv run maturin develop --release
sha256sum python/owl/rs.abi3.so | tee "$R/rs.abi3.so.sha256"

step "8. Environment checks (.venv/bin/python; the launch uses .venv/bin/torchrun, no uv run)"
.venv/bin/python - "$EXPECT_GPUS" <<'EOF' | tee "$R/versions.json"
import json, sys
import torch, triton, flash_attn
import owl.rs
from owl.model.compile_gemm import check_compile_stack, installed_compile_stack
owl.rs.assert_release_build()
out = {
    "python": sys.version.split()[0],
    "torch": torch.__version__, "torch_cuda": torch.version.cuda,
    "triton": triton.__version__, "flash_attn": flash_attn.__version__,
    "cuda_devices": torch.cuda.device_count(),
    "device0": torch.cuda.get_device_name(0),
    "capability0": ".".join(map(str, torch.cuda.get_device_capability(0))),
    "owl_rs": owl.rs.__file__, "owl_rs_release_build": True,
}
assert out["cuda_devices"] == int(sys.argv[1]), out
try:
    rep = check_compile_stack(installed_compile_stack())
    out["compile_stack"] = {"torch": rep.torch, "triton": rep.triton, "nvidia_driver": rep.nvidia_driver}
except RuntimeError as error:
    out["compile_stack_refused"] = str(error)
print(json.dumps(out, indent=1))
EOF
grep -q '"compile_stack_refused"' "$R/versions.json" && echo "compile stack REFUSED (see step 3 options)" || echo "compile stack accepted"
UV_LINK_MODE=copy uv sync --frozen --group dev --extra flash-attn --dry-run 2>&1 | tail -3 | tee "$R/uv-dry-run.txt"

step "9. Caches and dirs"
mkdir -p /root/sweep-cache/inductor /root/sweep-cache/triton /root/runs /root/receipts
df -h / | tee "$R/disk.txt"
free -g | tee "$R/memory.txt"

step "10. CPU / NUMA topology"
{ lscpu | grep -E '^(CPU\(s\)|Model name|Socket|NUMA)'; nproc; } | tee "$R/cpu.txt"
for n in /sys/devices/system/node/node*; do [ -e "$n/cpulist" ] && echo "$(basename "$n") cpus $(cat "$n/cpulist")"; done | tee -a "$R/cpu.txt"
# Non-fatal: a topology error (e.g. a cpuset covering one NUMA node only) must not skip step 11.
TOPO_OK=1 TOPO_RC=0
KG_NT_NATIVE_THREADS=4 .venv/bin/python "$KIT/main_probe_auto.py" --topology "$EXPECT_GPUS" > "$R/topology.json" 2> "$R/topology.err" || TOPO_RC=$?
if [ "$TOPO_RC" = 0 ]; then
  cat "$R/topology.json"
  grep -q '"oversubscribed": true' "$R/topology.json" && echo "WARNING: some node has fewer CPUs than ranks x (native_threads + 2); see mac_side.md" || echo "no node oversubscribed at native_threads=4"
else
  TOPO_OK=0
  cat "$R/topology.err"
  echo "WARNING: topology discovery FAILED (exit $TOPO_RC; recorded in $R/topology.err)."
  echo "  launch.sh with the default --numa cpu will refuse. Fix the cause, or launch with --numa off"
  echo "  (ranks run unbound: a throughput difference from pcy5knet, not a recipe change; tell the owner)."
fi

step "11. W&B credential (the key is never read or printed here)"
if [ -f /root/.netrc ]; then
  echo "netrc mode $(stat -c %a /root/.netrc)"
  [ "$(stat -c %a /root/.netrc)" = 600 ] || die "/root/.netrc must be mode 600"
  [ -z "${WANDB_MODE:-}" ] || die "WANDB_MODE is set ($WANDB_MODE); unset it"
  .venv/bin/python -c "import wandb; print('wandb.Api ok, entity:', wandb.Api(timeout=30).default_entity)" | tee "$R/wandb.txt"
else
  echo "no /root/.netrc yet: run mac_side.md step 3 before launch.sh (launch refuses without it)"
fi

step "BOOTSTRAP DONE head=$(git rev-parse --short HEAD) pinned=$([ "$(git rev-parse HEAD)" = "$PINNED_SHA" ] && echo yes || echo NO) driver=$DRIVER gate_ok=$DRIVER_OK topology_ok=$TOPO_OK receipts=$R"
