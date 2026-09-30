#!/usr/bin/env bash
# Stage a fresh 8-GPU pod from the Mac (DRAFT): source, BC best, W&B credential.
# Run ON THE MAC from a clean checkout of the commit to run:
#   bash ops/rebuild-2026-09-29/8rank-run/stage_from_mac.sh <pod-ssh-target>
#
# 1. The source travels as a git bundle of HEAD (nothing is pushed) and is
#    cloned to $KG_POD_REPO (default /workspace/kg-v3) at that commit.
#    The ignored Orbit Wars fixtures are not needed for training.
# 2. The BC best: copy_bc_best.sh.
# 3. The W&B credential: step 1 of
#    cookbook/workflows/install-the-wandb-credential-before-any-pod-launch.md,
#    verbatim: only the api.wandb.ai netrc entry, through stdin, mode 600, never
#    printed and never overwriting an existing ~/.netrc.
# Then, on the pod: cd $KG_POD_REPO && bash ops/rebuild-2026-09-29/8rank-run/setup.sh
set -euo pipefail

POD=${1:?usage: stage_from_mac.sh <pod-ssh-target>}
REMOTE_REPO=${KG_POD_REPO:-/workspace/kg-v3}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$(git -C "$HERE" rev-parse --show-toplevel)"
[ -z "$(git status --porcelain)" ] || { echo "commit first: the bundle carries HEAD only" >&2; exit 1; }
COMMIT=$(git rev-parse HEAD)
BUNDLE=$(mktemp -d)/kg-v3.bundle
git bundle create "$BUNDLE" HEAD
scp -q "$BUNDLE" "$POD:/tmp/kg-v3.bundle"
ssh "$POD" "set -eu
  if [ -e '$REMOTE_REPO' ]; then echo '$REMOTE_REPO exists; refusing to overwrite' >&2; exit 1; fi
  git clone -q /tmp/kg-v3.bundle '$REMOTE_REPO'
  git -C '$REMOTE_REPO' checkout -q '$COMMIT'
  git -C '$REMOTE_REPO' rev-parse HEAD"

bash "$HERE/copy_bc_best.sh" "$POD"

set -o pipefail
uv run python scripts/export_wandb_netrc_entry.py \
  | ssh "$POD" 'set -eu; umask 077
      if [ -e "$HOME/.netrc" ]; then
        echo "pod ~/.netrc exists; merge the api.wandb.ai entry by hand" >&2; exit 1
      fi
      tmp="$HOME/.netrc.wandb.$$"; cat > "$tmp"
      if [ ! -s "$tmp" ]; then
        rm -f "$tmp"; echo "empty credential; nothing installed" >&2; exit 1
      fi
      chmod 600 "$tmp"; mv "$tmp" "$HOME/.netrc"'
echo "staged $COMMIT on $REMOTE_REPO; next: setup.sh on the pod"
