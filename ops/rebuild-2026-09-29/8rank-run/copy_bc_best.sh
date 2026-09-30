#!/usr/bin/env bash
# Copy the BC best from the Mac's durable copy to a pod and verify it (DRAFT).
# Run ON THE MAC:
#   bash ops/rebuild-2026-09-29/8rank-run/copy_bc_best.sh <pod-ssh-target>
#
# Source: $KG_BC_LOCAL (default ~/kg-v3-runs/bc-best, outside any repository;
# copied from /tmp/kg-v3-bc-best on 2026-09-30 and checked against
# ~/kg-v3-runs/bc-best.SHA256SUMS). Target: $KG_POD_BC_DIR on the pod
# (default /workspace/bc-best). <pod-ssh-target> comes from the live pod read
# and is never written into tracked files.
#
# Alternative: network volume 4llk4uaf20 (EU-RO-1, "kggriv2-shared-1tb") holds
# the original at /workspace/kg-v3-bc-2026-09-29/run/bc-20260929-142216/checkpoint_bc_best.pt.
# It is reachable only from a pod created in EU-RO-1 with that volume attached;
# then set KG_BC_BEST to that path (after the same SHA-256 check) instead of
# running this script.
set -euo pipefail

POD=${1:?usage: copy_bc_best.sh <pod-ssh-target>}
SRC=${KG_BC_LOCAL:-$HOME/kg-v3-runs/bc-best}
DEST=${KG_POD_BC_DIR:-/workspace/bc-best}
SHA=fd8545872aca59c70e273e9655055e1104cd719588364f463ae87b0d488e6f51

echo "$SHA  $SRC/checkpoint_bc_best.pt" | shasum -a 256 -c -
ssh "$POD" "mkdir -p '$DEST'"
scp -q "$SRC/checkpoint_bc_best.pt" "$SRC/checkpoint_bc_best.json" "$SRC/bc_result.json" \
  "$SRC/bc_config.yaml" "$SRC/config.yaml" "$POD:$DEST/"
ssh "$POD" "cd '$DEST' && echo '$SHA  checkpoint_bc_best.pt' | sha256sum -c -"
echo "BC best on the pod at $DEST/checkpoint_bc_best.pt (SHA-256 verified on both hosts)"
