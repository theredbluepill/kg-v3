#!/usr/bin/env bash
# One command: package a checkpoint on the Mac (Linux Kaggle tarball), ship it to the
# CPU eval pod, play the 48-game fixed-shop anchor panel there, copy receipts back.
#
# usage: eval_ckpt.sh CKPT CONFIG LABEL [--rule1-off] [--package PKG_OUT_DIR]
#   LABEL         game label prefix, e.g. 70M -> games 70M-<anchor>-s<seed>-seat<seat>
#   --rule1-off   package without --final-turn-liquidation; output folder LABEL
#                 (default: rule 1 baked on, output folder LABEL-ft-on, like the Mac arms)
#   --package D   reuse an existing package_checkpoint.sh OUT_DIR (submission.tar.gz,
#                 verify.json) instead of building; CKPT/CONFIG are then only checked
#                 against its manifest
# Env: PARALLEL (default 30), SEEDS, POD_* (pod.env), KG_PKG_REPO (packager worktree).
#
# Rule 2 is always off. Nothing is committed, pushed, uploaded or submitted.
# Outputs: $KG_PKG_REPO/artifacts/anchor-eval/<OUT>-<ckpt sha12>/  (package, ignored dir)
#          pod /root/anchor-eval/{pkgs,games}/<OUT>...
#          ../games-fixedshop-linux/<OUT>/{receipts,logs,replays,run.log,package/,eval.json,summary.md}
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/pod.env"
now() { perl -MTime::HiRes=time -e 'printf "%.1f\n", time'; }
die() { echo "eval_ckpt: $*" >&2; exit 1; }

pos=(); rule1=1; pkg_in=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --rule1-off) rule1=0; shift ;;
    --package) pkg_in="$2"; shift 2 ;;
    -h | --help) sed -n 2,19p "$0"; exit 0 ;;
    -*) die "unknown option $1" ;;
    *) pos+=("$1"); shift ;;
  esac
done
[[ ${#pos[@]} -eq 3 ]] || die "usage: eval_ckpt.sh CKPT CONFIG LABEL [--rule1-off] [--package DIR]"
CKPT=${pos[0]}; CONFIG=${pos[1]}; LABEL=${pos[2]}
[[ -f "$CKPT" && -f "$CONFIG" ]] || die "checkpoint or config not found"
[[ "$LABEL" =~ ^[A-Za-z0-9.]+$ ]] || die "LABEL must be [A-Za-z0-9.]+ (it prefixes game labels)"
OUT=$LABEL; [[ $rule1 -eq 1 ]] && OUT=$LABEL-ft-on
CKPT_SHA=$(shasum -a 256 "$CKPT" | cut -d' ' -f1)
CONFIG_SHA=$(shasum -a 256 "$CONFIG" | cut -d' ' -f1)
LOCAL=$ANCHOR_GAMES/games-fixedshop-linux/$OUT
[[ ! -e "$LOCAL/receipts" ]] || die "$LOCAL already has receipts; pick another LABEL or move it"
t0=$(now)

# ---- 1. package (Mac) ---------------------------------------------------------
if [[ -n "$pkg_in" ]]; then
  PKG=$(cd "$pkg_in" && pwd -P)
else
  PKG=$KG_PKG_REPO/artifacts/anchor-eval/$OUT-${CKPT_SHA:0:12}
  if [[ ! -f "$PKG/submission.tar.gz" ]]; then
    flags=(); [[ $rule1 -eq 1 ]] && flags+=(--final-turn-liquidation)
    "$KG_PKG_REPO/scripts/package_checkpoint.sh" "$CKPT" "$CONFIG" "$PKG" ${flags[@]+"${flags[@]}"} >&2
  fi
fi
python3 - "$PKG" "$CKPT_SHA" "$CONFIG_SHA" "$rule1" <<'EOF' || die "package check failed"
import json, sys
pkg, ckpt, cfg, rule1 = sys.argv[1:]
v = json.load(open(f"{pkg}/verify.json"))
m = json.load(open(f"{pkg}/agent/manifest.json"))
want = "baked on" if rule1 == "1" else "environment (off)"
errs = [e for e, bad in [
    (f"verify not ok: {v['problems']}", not v["ok"]),
    (f"checkpoint sha {m['checkpoint']['original_sha256']} != {ckpt}", m["checkpoint"]["original_sha256"] != ckpt),
    (f"config sha {v['config_sha256']} != {cfg}", v["config_sha256"] != cfg),
    (f"rule 1 {v['final_turn_liquidation']!r} != {want!r}", v["final_turn_liquidation"] != want),
    (f"rule 2 {v['block_late_investments']!r} not env-switched", v["block_late_investments"] != "environment (off)"),
] if bad]
print("\n".join(errs), file=sys.stderr)
sys.exit(1 if errs else 0)
EOF
ARCHIVE_SHA=$(shasum -a 256 "$PKG/submission.tar.gz" | cut -d' ' -f1)
MANIFEST_SHA=$(shasum -a 256 "$PKG/agent/manifest.json" | cut -d' ' -f1)
t_pkg=$(now)

# ---- 2. ship ----------------------------------------------------------------
R=/root/anchor-eval
RPKG=$R/pkgs/$OUT-${ARCHIVE_SHA:0:12}
pod "set -e; cd $R; test -x venv/bin/python; echo '$KENV_ENGINE_SHA256  kenv-fixedshop/kaggle_environments/envs/kaggriculture/kaggriculture.py' | sha256sum -c --quiet
cd bin; printf '%s  cha22_agent\n%s  v56_agent\n' $CHA22_AGENT_SHA256 $V56_AGENT_SHA256 | sha256sum -c --quiet" \
  || die "pod not set up (or pins differ); run setup_pod.sh"
pod_put -r "$HERE/pod/harness" "$HERE/pod/anchors" "$R/"
pod "mkdir -p $RPKG"
pod_put "$PKG/submission.tar.gz" "$RPKG/"
pod "set -e; cd $RPKG; echo '$ARCHIVE_SHA  submission.tar.gz' | sha256sum -c --quiet
rm -rf agent; mkdir agent; tar -xzf submission.tar.gz -C agent
echo '$MANIFEST_SHA  agent/manifest.json' | sha256sum -c --quiet
G=$R/games/$OUT/receipts
if ls \$G/*.json >/dev/null 2>&1; then
  grep -h '\"manifest_sha256\"' \$G/*.json | sort -u | grep -qv '$MANIFEST_SHA' && { echo 'pod games/$OUT holds receipts of another package' >&2; exit 1; }
fi; true"
t_ship=$(now)

# ---- 3. games (pod, detached; resumable: existing receipts are kept) -----------
pod "mkdir -p $R/games/$OUT; cd $R/games/$OUT; PARALLEL=${PARALLEL:-30} ${SEEDS:+SEEDS='$SEEDS'} LIQ=$rule1 \
  setsid nohup $R/harness/run_games.sh $RPKG/agent $OUT $LABEL >> run.log 2>&1 < /dev/null & echo launched"
echo "games running on pod: $R/games/$OUT/run.log" >&2
while true; do
  sleep 30
  last=$(pod "tail -1 $R/games/$OUT/run.log") || { echo "poll failed; retrying" >&2; continue; }
  done_n=$(pod "ls $R/games/$OUT/receipts 2>/dev/null | wc -l") || done_n='?'
  echo "$(date -u +%FT%TZ) receipts=$done_n last: $last" >&2
  [[ "$last" == '{"receipts"'* ]] && break
done
t_games=$(now)

# ---- 4. copy back ---------------------------------------------------------------
mkdir -p "$LOCAL/package"
# one tar stream (scp -r of ~150 small files took ~4 min over this link)
pod "tar -C $R/games/$OUT -cf - receipts logs replays run.log" | tar -C "$LOCAL" -xf -
[[ $(ls "$LOCAL/receipts" | wc -l) -eq $(pod "ls $R/games/$OUT/receipts | wc -l") ]] || die "copy-back incomplete"
cp "$PKG/PACKAGE.md" "$PKG/verify.json" "$PKG/build.json" "$PKG/agent/manifest.json" "$LOCAL/package/"
t_back=$(now)
python3 "$HERE/summarize.py" "$LOCAL" --json "$LOCAL/summary.json" | tee "$LOCAL/summary.md"
python3 - "$LOCAL/eval.json" <<EOF
import json, sys
s = json.load(open("$LOCAL/summary.json"))["summary"]["ALL"]
games_wall = [l for l in open("$LOCAL/run.log") if l.startswith("end ")][-1].split("wall_s=")[1].strip()
json.dump({
  "label": "$LABEL", "out": "$OUT", "rule1": "baked on" if $rule1 else "off", "rule2": "off",
  "checkpoint": "$CKPT", "checkpoint_sha256": "$CKPT_SHA", "config": "$CONFIG", "config_sha256": "$CONFIG_SHA",
  "package_dir": "$PKG", "archive_sha256": "$ARCHIVE_SHA", "manifest_sha256": "$MANIFEST_SHA",
  "pod": "$POD_HOST:$POD_PORT", "pod_package": "$RPKG", "pod_games": "$R/games/$OUT",
  "parallel": int("${PARALLEL:-30}"),
  "wall_s": {"package": round($t_pkg - $t0, 1), "ship": round($t_ship - $t_pkg, 1),
             "games_poll": round($t_games - $t_ship, 1), "games_pod_run_log": int(games_wall),
             "copy_back": round($t_back - $t_games, 1), "total": round($t_back - $t0, 1)},
  "games": s["games"], "qualified": s["qualified"],
}, open(sys.argv[1], "w"), indent=1)
print(open(sys.argv[1]).read())
EOF
