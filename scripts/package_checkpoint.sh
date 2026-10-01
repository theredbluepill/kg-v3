#!/usr/bin/env bash
# Package one PPO checkpoint as a Kaggriculture Kaggle submission, from the Mac,
# reusing the cached Linux x86-64 native module instead of rebuilding it.
#
# Stages (each timed and recorded in OUT_DIR/PACKAGE.md):
#   venv       builder venv (CPython 3.11, torch 2.6.0, numpy 2.4.6), made once
#   preflight  clean tree; native sources identical to the cached module's
#              source commit; cached module present with the recorded SHA-256;
#              local Kaggle image id equal to the runtime receipt's
#   build      scripts/build_kaggriculture_submission.py with both receipts and
#              the checkpoint's expected SHA-256 (and --final-turn-liquidation
#              when given, which bakes rule 1 on in the packaged main.py)
#   verify     fresh extract; every file re-hashed against the inner manifest;
#              archive weights compared tensor by tensor with checkpoint["model"];
#              packaged main.py's rule 1/rule 2 wiring checked against the flag
#   image      strict self-play episode of the extracted archive in the local
#              Kaggle image (amd64 emulated, no network, 1.6 CPUs, 6.5 GB):
#              40 turns by default (--episode-steps), 720 with --full-episode
# Nothing here uploads or submits; submission is the owner's decision.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/package_checkpoint.sh CHECKPOINT CONFIG OUT_DIR [options]

Options:
  --full-episode   run the full 720-turn Kaggle-image episode (about 6 minutes
                   emulated) instead of the short bounded one
  --episode-steps N
                   bound for the short episode (default 40)
  --seed N         episode seed (default 7)
  --final-turn-liquidation
                   bake rule 1 (final-turn liquidation) on in the packaged
                   main.py; rule 2 stays on its environment switch (off)
  --allow-fixed-opponents
                   accept a cached module whose manifest records the
                   fixed-opponent controllers compiled in (see README)
  -h, --help       show this help

Environment:
  KG_PACKAGE_VENV  builder venv (default ~/.cache/kg-v3/package-venv-py311-torch2.6.0)
EOF
}

positional=()
full_episode=0
episode_steps=40
seed=7
allow_fixed_opponents=0
final_turn_liquidation=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h | --help) usage; exit 0 ;;
    --full-episode) full_episode=1; shift ;;
    --episode-steps) episode_steps="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --allow-fixed-opponents) allow_fixed_opponents=1; shift ;;
    --final-turn-liquidation) final_turn_liquidation=1; shift ;;
    -*) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) positional+=("$1"); shift ;;
  esac
done
if [[ ${#positional[@]} -ne 3 ]]; then
  usage >&2
  exit 2
fi

abspath() { (cd "$(dirname "$1")" && printf '%s/%s\n' "$(pwd -P)" "$(basename "$1")"); }
now() { perl -MTime::HiRes=time -e 'printf "%.3f\n", time'; }
die() { echo "package_checkpoint: $*" >&2; exit 1; }

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
CHECKPOINT="$(abspath "${positional[0]}")"
CONFIG="$(abspath "${positional[1]}")"
OUT_DIR="${positional[2]}"
[[ -f "$CHECKPOINT" ]] || die "checkpoint not found: $CHECKPOINT"
[[ -f "$CONFIG" ]] || die "config not found: $CONFIG"
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd -P)"
ARCHIVE="$OUT_DIR/submission.tar.gz"
[[ ! -e "$ARCHIVE" ]] || die "$ARCHIVE already exists; use a new OUT_DIR"

CACHE_MANIFEST="$REPO/native-cache/manifest.json"
RUNTIME_RECEIPT="$REPO/ops/rebuild-2026-09-29/7.4/kaggle-runtime-receipt.json"
HARNESS="$REPO/scripts/kaggle_local_episode.py"
VENV="${KG_PACKAGE_VENV:-$HOME/.cache/kg-v3/package-venv-py311-torch2.6.0}"
PY="$VENV/bin/python"
STAGES=()
stage_start=0
begin() { stage_start="$(now)"; echo "== $1" >&2; }
end() { STAGES+=("$1=$(perl -e "printf '%.1f', $(now) - $stage_start")"); }
jget() { "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1]))
for k in sys.argv[2].split("."): d=d[k]
print(json.dumps(d) if isinstance(d,(dict,list,bool)) else d)' "$1" "$2"; }
t_total="$(now)"

# ---- builder venv ----------------------------------------------------------
begin venv
if ! "$PY" -c 'import torch, numpy; assert torch.__version__ == "2.6.0" and numpy.__version__ == "2.4.6"' 2>/dev/null; then
  echo "creating builder venv $VENV" >&2
  uv venv -q --python 3.11 "$VENV"
  uv pip install -q --python "$PY" torch==2.6.0 numpy==2.4.6
fi
end venv

# ---- preflight -------------------------------------------------------------
begin preflight
cd "$REPO"
dirty="$(git status --porcelain --untracked-files=all)"
[[ -z "$dirty" ]] || die "working tree is not clean:
$dirty"
HEAD_COMMIT="$(git rev-parse HEAD)"
SO_SHA="$(jget "$CACHE_MANIFEST" module.sha256)"
SO_SOURCE="$(jget "$CACHE_MANIFEST" module.source_commit)"
SO_CACHE="$REPO/$(jget "$CACHE_MANIFEST" module.cache_path)"
SO_SEED="$REPO/$(jget "$CACHE_MANIFEST" module.seed_from)"
SO_OPPONENTS="$(jget "$CACHE_MANIFEST" module.fixed_opponents_compiled_in)"
git cat-file -e "$SO_SOURCE^{commit}" 2>/dev/null \
  || die "cached module's source commit $SO_SOURCE is not in this repository"
native_paths=()
while IFS= read -r p; do native_paths+=("$p"); done \
  < <("$PY" -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["native_source_paths"]))' "$CACHE_MANIFEST")
changed="$(git diff --name-only "$SO_SOURCE" HEAD -- "${native_paths[@]}")"
maturin_same="$("$PY" - "$SO_SOURCE" <<'EOF'
import subprocess, sys, tomllib
def table(rev):
    text = subprocess.run(["git", "show", f"{rev}:pyproject.toml"], check=True,
                          capture_output=True, text=True).stdout
    return tomllib.loads(text)["tool"]["maturin"]
print("yes" if table(sys.argv[1]) == table("HEAD") else "no")
EOF
)"
[[ "$maturin_same" == yes ]] || changed="${changed}${changed:+$'\n'}pyproject.toml [tool.maturin]"
if [[ -n "$changed" ]]; then
  die "native sources differ from ${SO_SOURCE:0:8}, the cached owl.rs module's source commit:
$changed
The .so must be rebuilt for Linux x86-64 (see README, Kaggle packaging) and
native-cache/manifest.json replaced before packaging from $HEAD_COMMIT."
fi
if [[ "$SO_OPPONENTS" == true && $allow_fixed_opponents -ne 1 ]]; then
  die "the cached module has the fixed-opponent controllers (EcoBot, E776, Cha22)
compiled in; opponents_rs/README.md and Cargo.toml say the Kaggle build drops
them (--no-default-features). Rebuild without them, or pass
--allow-fixed-opponents to package this module anyway (recorded in PACKAGE.md)."
fi
if [[ ! -f "$SO_CACHE" ]]; then
  [[ -f "$SO_SEED" ]] || die "cached module missing at $SO_CACHE and no seed at $SO_SEED"
  echo "seeding native cache from $SO_SEED" >&2
  cp "$SO_SEED" "$SO_CACHE.tmp"
  mv "$SO_CACHE.tmp" "$SO_CACHE"
fi
actual_so_sha="$(shasum -a 256 "$SO_CACHE" | cut -d' ' -f1)"
[[ "$actual_so_sha" == "$SO_SHA" ]] || die "cached module SHA-256 $actual_so_sha != manifest $SO_SHA"
IMAGE="$(jget "$RUNTIME_RECEIPT" image)"
IMAGE_ID="$(jget "$RUNTIME_RECEIPT" image_id)"
local_image_id="$(docker image inspect "$IMAGE" --format '{{.Id}}' 2>/dev/null)" \
  || die "Kaggle image $IMAGE is not available to docker"
[[ "$local_image_id" == "$IMAGE_ID" ]] \
  || die "local $IMAGE is $local_image_id, not the runtime receipt's $IMAGE_ID"
CKPT_SHA="$(shasum -a 256 "$CHECKPOINT" | cut -d' ' -f1)"
CONFIG_SHA="$(shasum -a 256 "$CONFIG" | cut -d' ' -f1)"
"$PY" -c 'import json,sys; m=json.load(open(sys.argv[1])); json.dump(m["receipt"], open(sys.argv[2],"w"), indent=2)' \
  "$CACHE_MANIFEST" "$OUT_DIR/native-module-receipt.json"
end preflight

# ---- build -----------------------------------------------------------------
begin build
build_flags=()
[[ $final_turn_liquidation -eq 1 ]] && build_flags+=(--final-turn-liquidation)
PYTHONPATH="$REPO/python" "$PY" "$REPO/scripts/build_kaggriculture_submission.py" \
  --checkpoint "$CHECKPOINT" --config "$CONFIG" \
  --native-module "$SO_CACHE" \
  --native-receipt "$OUT_DIR/native-module-receipt.json" \
  --runtime-receipt "$RUNTIME_RECEIPT" \
  --expected-checkpoint-sha256 "$CKPT_SHA" \
  ${build_flags[@]+"${build_flags[@]}"} \
  --output "$ARCHIVE" > "$OUT_DIR/build.json"
ARCHIVE_SHA="$(shasum -a 256 "$ARCHIVE" | cut -d' ' -f1)"
ARCHIVE_BYTES="$(stat -f %z "$ARCHIVE")"
echo "archive $ARCHIVE" >&2
echo "  sha256 $ARCHIVE_SHA  bytes $ARCHIVE_BYTES" >&2
end build

# ---- verify ----------------------------------------------------------------
begin verify
AGENT="$OUT_DIR/agent"
rm -rf "$AGENT"
mkdir -p "$AGENT"
tar -xzf "$ARCHIVE" -C "$AGENT"
"$PY" - "$AGENT" "$CHECKPOINT" "$CKPT_SHA" "$SO_SHA" "$HEAD_COMMIT" "$final_turn_liquidation" > "$OUT_DIR/verify.json" <<'EOF'
import hashlib, json, sys
from pathlib import Path
import torch
agent, checkpoint, ckpt_sha, so_sha, head, ft = Path(sys.argv[1]), *sys.argv[2:]
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
manifest_path = agent / "manifest.json"
manifest = json.loads(manifest_path.read_text())
on_disk = {p.relative_to(agent).as_posix() for p in agent.rglob("*") if p.is_file()}
listed = set(manifest["files"])
problems = []
if on_disk - listed != {"manifest.json"} or listed - on_disk:
    problems.append(f"file set mismatch: extra {sorted(on_disk - listed - {'manifest.json'})}, missing {sorted(listed - on_disk)}")
bad = [n for n, e in manifest["files"].items() if (agent / n).is_file() and (sha(agent / n) != e["sha256"] or (agent / n).stat().st_size != e["bytes"])]
problems += [f"hash/size mismatch: {n}" for n in bad]
checks = {
    "checkpoint.original_sha256": (manifest["checkpoint"]["original_sha256"], ckpt_sha),
    "native_module.sha256": (manifest["native_module"]["sha256"], so_sha),
    "source.commit": (manifest["source"]["commit"], head),
}
problems += [f"{k}: {a} != {b}" for k, (a, b) in checks.items() if a != b]
main_py = (agent / "main.py").read_text()
rule1_baked = main_py.count("final_turn_liquidation=True,") == 1 and "final_turn.enabled_from_env()" not in main_py
rule1_env = main_py.count("final_turn_liquidation=final_turn.enabled_from_env(),") == 1
rule2_env = main_py.count("block_late_investments=late_invest.enabled_from_env(),") == 1
if not (rule1_baked if ft == "1" else rule1_env):
    problems.append(f"main.py rule 1 wiring does not match --final-turn-liquidation={ft}")
if not rule2_env:
    problems.append("main.py rule 2 is not on its environment switch")
if manifest["entrypoint"]["final_turn_liquidation"] != ("baked on" if ft == "1" else "environment (off)"):
    problems.append(f"manifest entrypoint.final_turn_liquidation={manifest['entrypoint']['final_turn_liquidation']!r}")
original = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
slim = torch.load(agent / "models/primary/checkpoint.pt", map_location="cpu", weights_only=True)
if set(slim) != {"model"}:
    problems.append(f"slim checkpoint keys {sorted(slim)} != ['model']")
slim = slim["model"]
if set(slim) != set(original):
    problems.append(f"state key mismatch: {sorted(set(slim) ^ set(original))[:10]}")
unequal = [k for k in sorted(set(slim) & set(original))
           if slim[k].dtype != original[k].dtype or slim[k].shape != original[k].shape
           or not torch.equal(slim[k], original[k])]
problems += [f"tensor differs: {k}" for k in unequal[:20]]
report = {
    "ok": not problems,
    "manifest_sha256": sha(manifest_path),
    "files_rehashed": len(listed),
    "tensors_compared": len(set(slim) & set(original)),
    "tensors_equal": len(set(slim) & set(original)) - len(unequal),
    "parameters": int(sum(t.numel() for t in slim.values())),
    "slim_sha256": manifest["checkpoint"]["slim_sha256"],
    "config_sha256": manifest["checkpoint"]["config_sha256"],
    "final_turn_liquidation": manifest["entrypoint"]["final_turn_liquidation"],
    "block_late_investments": manifest["entrypoint"]["block_late_investments"],
    "problems": problems,
}
print(json.dumps(report, indent=2))
sys.exit(0 if report["ok"] else 1)
EOF
MANIFEST_SHA="$(jget "$OUT_DIR/verify.json" manifest_sha256)"
end verify

# ---- Kaggle-image episode ---------------------------------------------------
begin image
steps_args=(--episode-steps "$episode_steps")
episode_label="steps$episode_steps"
if [[ $full_episode -eq 1 ]]; then
  steps_args=()
  episode_label="full"
fi
mkdir -p "$OUT_DIR/validation"
RECEIPT_NAME="kaggle-image-episode-self-seed$seed-$episode_label.json"
DOCKER_CMD=(docker run --rm --platform linux/amd64 --network none --cpus=1.6 --memory=6.5g
  -v "$AGENT:/kaggle_simulations/agent:ro"
  -v "$HARNESS:/harness/kaggle_local_episode.py:ro"
  -v "$OUT_DIR/validation:/out"
  "$IMAGE" python /harness/kaggle_local_episode.py
  --agent-dir /kaggle_simulations/agent --seed "$seed" --opponent self
  ${steps_args[@]+"${steps_args[@]}"} --replay-dir /out/replays --receipt "/out/$RECEIPT_NAME")
image_status=0
"${DOCKER_CMD[@]}" > "$OUT_DIR/validation/docker-episode.log" 2>&1 || image_status=$?
end image

CMD_FLAGS=""
[[ $full_episode -eq 1 ]] && CMD_FLAGS+=" --full-episode"
[[ $full_episode -eq 1 ]] || CMD_FLAGS+=" --episode-steps $episode_steps"
CMD_FLAGS+=" --seed $seed"
[[ $allow_fixed_opponents -eq 1 ]] && CMD_FLAGS+=" --allow-fixed-opponents"
[[ $final_turn_liquidation -eq 1 ]] && CMD_FLAGS+=" --final-turn-liquidation"
# ---- receipt -----------------------------------------------------------------
TOTAL="$(perl -e "printf '%.1f', $(now) - $t_total")"
"$PY" - "$OUT_DIR" "$RECEIPT_NAME" "$image_status" <<EOF
import json, sys
from pathlib import Path
out, receipt_name, image_status = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
verify = json.loads((out / "verify.json").read_text())
receipt_path = out / "validation" / receipt_name
episode = json.loads(receipt_path.read_text()) if receipt_path.is_file() else None
stages = dict(s.split("=") for s in "${STAGES[*]}".split())
lines = [
    "# Kaggle package receipt",
    "",
    "Built by \`scripts/package_checkpoint.sh\`. Nothing was uploaded or submitted.",
    "",
    "| Item | Value |",
    "| --- | --- |",
    "| Command | \`scripts/package_checkpoint.sh $CHECKPOINT $CONFIG $OUT_DIR$CMD_FLAGS\` |",
    "| Archive | \`submission.tar.gz\`, $ARCHIVE_BYTES bytes |",
    "| Archive sha256 | \`$ARCHIVE_SHA\` |",
    "| Inner manifest.json sha256 | \`$MANIFEST_SHA\` |",
    "| Checkpoint | \`$CHECKPOINT\` |",
    "| Checkpoint sha256 | \`$CKPT_SHA\` |",
    f"| Slim checkpoint sha256 | \`{verify['slim_sha256']}\` |",
    "| Config | \`$CONFIG\`, sha256 \`$CONFIG_SHA\` |",
    "| Source | commit \`$HEAD_COMMIT\`, clean tree |",
    "| Native module | \`$SO_SHA\` from native-cache (source commit \`$SO_SOURCE\`, native sources identical at HEAD); fixed-opponent controllers compiled in: $SO_OPPONENTS |",
    f"| Endgame rules in main.py | rule 1 final-turn liquidation: {verify['final_turn_liquidation']}; rule 2 late-investment filter: {verify['block_late_investments']} |",
    "| Builder | \`$PY\` |",
    "| Kaggle image | \`$IMAGE\` (\`$IMAGE_ID\`), linux/amd64 emulated, --network none --cpus=1.6 --memory=6.5g |",
    "",
    "## Checks",
    "",
    f"- Verify: ok={verify['ok']}; {verify['files_rehashed']} files re-hashed against the inner manifest; "
    f"{verify['tensors_equal']}/{verify['tensors_compared']} model tensors equal to checkpoint['model'] "
    f"({verify['parameters']:,} parameters); problems: {verify['problems'] or 'none'}.",
]
if episode is None:
    lines.append(f"- Kaggle-image episode: FAILED (docker exit {image_status}, no receipt); see validation/docker-episode.log.")
else:
    s = episode["summary"]
    seats = "; ".join(
        f"seat {k}: {v['calls']} calls, {v['exceptions']} exceptions, {v['invalid_raw_actions']} invalid raw, "
        f"{v['default_pass_returns']} default passes, turn 0 {v['turn0_duration_s']:.2f} s, "
        f"steady p99 {v['steady_duration_s'].get('p99', float('nan')):.3f} s, max {v['steady_duration_s'].get('max', float('nan')):.3f} s"
        for k, v in s["seats"].items())
    lines.append(
        f"- Kaggle-image episode (strict, self-play, seed {episode['seed']}, episode_steps {episode['episode_steps']}): "
        f"qualified={episode['qualified']}; {s['recorded_steps']} steps, {s['bad_status_count']} bad statuses, "
        f"final {s['final_statuses']}; {seats}; banks {s['final_banks']}; wall {episode['wall_s']} s; "
        f"owl.rs loaded from {episode['loaded_from']['owl.rs']}. Receipt: validation/{receipt_name}.")
lines += [
    "",
    "## Stage wall times (s)",
    "",
    "| " + " | ".join(stages) + " | total |",
    "|" + " --- |" * (len(stages) + 1),
    "| " + " | ".join(stages.values()) + " | $TOTAL |",
    "",
    "## Limits",
    "",
    "- Packaging and legality evidence only, not strength. Episode timings are emulated amd64 on the Mac, not Kaggle hardware.",
    "- That Kaggle production runs this image is not established (see the runtime receipt).",
    "- The native module was not rebuilt here; its custody is native-cache/manifest.json.",
]
if "$SO_OPPONENTS" == "true":
    lines.append("- The native module has the fixed-opponent controllers compiled in (default cargo features); the repo's Kaggle build drops them.")
if episode is not None and episode["episode_steps"] is not None:
    lines.append(f"- Bounded episode ({episode['episode_steps']} turns); run with --full-episode for a 720-turn game.")
(out / "PACKAGE.md").write_text("\n".join(lines) + "\n")
EOF

echo "stages: ${STAGES[*]} total=$TOTAL" >&2
echo "archive: $ARCHIVE"
echo "sha256:  $ARCHIVE_SHA"
echo "bytes:   $ARCHIVE_BYTES"
echo "receipt: $OUT_DIR/PACKAGE.md"
if [[ $image_status -ne 0 ]]; then
  tail -20 "$OUT_DIR/validation/docker-episode.log" >&2
  die "Kaggle-image episode failed (exit $image_status)"
fi
