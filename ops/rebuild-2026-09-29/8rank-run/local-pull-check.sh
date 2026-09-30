# Fake-rsync harness for pull_from_pod.sh (Mac, 2026-09-30; r1 fix check). It maps
# "fakepod:" to a scratch directory and gives each pod-side write a fresh mtime,
# because openrsync skips same-size files whose mtime second is unchanged.
# Output: local-pull-check.out.

set -u
S=/private/tmp/claude-501/-Users-poonszesen-kg-v3/3f7ce81c-f27d-4f24-bb54-c9e0de2d998a/scratchpad/pulltest
PKG=/Users/poonszesen/kg-v3-8rankprep/ops/rebuild-2026-09-29/8rank-run
rm -rf "$S"; mkdir -p "$S/bin" "$S/pod/out/main-x/20260930-000000" "$S/pod/receipts/main/main-x"
cat > "$S/bin/rsync" <<EOF
#!/bin/bash
args=(); for a in "\$@"; do case "\$a" in fakepod:*) args+=("$S/pod\${a#fakepod:}");; *) args+=("\$a");; esac; done
exec /usr/bin/rsync "\${args[@]}"
EOF
chmod +x "$S/bin/rsync"
RUN=$S/pod/out/main-x/20260930-000000; REC=$S/pod/receipts/main/main-x
final_list() { (cd "$S/pod/out/main-x" && find . -type f -name 'checkpoint_*.pt' -exec shasum -a 256 {} + | sed 's#  \./#  #') > "$REC/checkpoints_final.sha256"; }
N=10
stamp() { N=$((N + 1)); touch -t "2026093001$N" "$@"; }
run() { stamp "$RUN"/*.pt "$REC"/* 2>/dev/null || true; KG_DURABLE_LOCAL=$S/mac KG_POD_RECEIPTS=/receipts PATH="$S/bin:$PATH" bash "$PKG/pull_from_pod.sh" fakepod "/out/main-x" > "$S/out.txt" 2>&1; st=$?; sed "s#$S#S#; s#^20[0-9T:-]*Z #<t> #" "$S/out.txt"; echo "exit=$st"; }
echo "== N no watchdog list yet, final mode"; run
head -c 3000 /dev/urandom > "$RUN/checkpoint_000000016384.pt"
head -c 3000 /dev/urandom > "$RUN/checkpoint_last_best.pt"
d1=$(shasum -a 256 "$RUN/checkpoint_000000016384.pt" | cut -d' ' -f1)
b1=$(shasum -a 256 "$RUN/checkpoint_last_best.pt" | cut -d' ' -f1)
printf '%s  %s\n' "$d1" 20260930-000000/checkpoint_000000016384.pt "$b1" 20260930-000000/checkpoint_last_best.pt > "$REC/checkpoints.sha256"
echo "== C1 clean first pull"; run
head -c 3000 /dev/urandom > "$RUN/checkpoint_last_best.pt"
b2=$(shasum -a 256 "$RUN/checkpoint_last_best.pt" | cut -d' ' -f1)
echo "$b2  20260930-000000/checkpoint_last_best.pt" >> "$REC/checkpoints.sha256"
head -c 3000 /dev/urandom > "$RUN/checkpoint_final.pt"
final_list
echo "== C2 clean final pull, two last_best versions, final list"; run
cp "$REC/checkpoints.sha256" "$S/wd.bak"
echo "== A wrong digest in the watchdog list"; printf '%s  %s\n' "$(printf '0%.0s' $(seq 64))" 20260930-000000/checkpoint_000000016384.pt > "$REC/checkpoints.sha256"; run
cp "$S/wd.bak" "$REC/checkpoints.sha256"
echo "== B pod file changed after the final list (final-list mismatch)"; echo x >> "$RUN/checkpoint_final.pt"; run
final_list
echo "== D a last_best version the Mac never pulled"; echo "$(printf 'ab%.0s' $(seq 32))  20260930-000000/checkpoint_last_best.pt" >> "$REC/checkpoints.sha256"; run
echo "== E superseded digest of a non-last_best path, newest line matches"; cp "$S/wd.bak" "$REC/checkpoints.sha256"; echo "$(printf 'cd%.0s' $(seq 32))  20260930-000000/checkpoint_final.pt" >> "$REC/checkpoints.sha256"; echo "$(shasum -a 256 "$RUN/checkpoint_final.pt" | cut -d' ' -f1)  20260930-000000/checkpoint_final.pt" >> "$REC/checkpoints.sha256"; run
