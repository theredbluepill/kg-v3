set -u
R=/workspace/kg-v3-rebuild/runs/flash-attn-setup-2026-09-29
W=$R/wheel/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
I=/workspace/kg-v3-rebuild/.venv/lib/python3.12/site-packages/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so
O=$R/wheel_member_compare.txt
T=$(mktemp -d /tmp/fa-wheel-cmp.XXXXXX)
{
echo "# flash_attn_2_cuda wheel-member vs installed .so comparison (Phase 6.0 follow-up; post-run)"
echo "date_utc_start: $(date -u +%FT%TZ)"
echo "host: $(hostname)  (pod w7ia3zvxqsvs3g)"
echo "wheel: $W"
echo "installed: $I"
echo "tempdir: $T (removed afterwards)"
echo
echo "\$ sha256sum \"\$W\""
sha256sum "$W"
echo "\$ unzip -l \"\$W\" 'flash_attn_2_cuda*.so'"
unzip -l "$W" 'flash_attn_2_cuda*.so'
echo "\$ unzip -q \"\$W\" 'flash_attn_2_cuda*.so' -d \"\$T\"; echo exit=\$?"
unzip -q "$W" 'flash_attn_2_cuda*.so' -d "$T"; echo "exit=$?"
echo "\$ ls -l \"\$T\" \"\$I\""
ls -l "$T" "$I"
echo "\$ sha256sum \"\$T\"/flash_attn_2_cuda*.so \"\$I\""
sha256sum "$T"/flash_attn_2_cuda*.so "$I"
echo "\$ cmp \"\$T\"/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so \"\$I\"; echo exit=\$?   # exit=0 = byte-identical"
cmp "$T"/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so "$I"; echo "exit=$?"
echo "\$ grep flash_attn_2_cuda .venv/lib/python3.12/site-packages/flash_attn-2.8.3.dist-info/RECORD"
grep flash_attn_2_cuda /workspace/kg-v3-rebuild/.venv/lib/python3.12/site-packages/flash_attn-2.8.3.dist-info/RECORD
echo "date_utc_end: $(date -u +%FT%TZ)"
} > "$O" 2>&1
rm -rf "$T"
echo "tempdir removed: $([ -e "$T" ] && echo no || echo yes)"
sha256sum "$O"
