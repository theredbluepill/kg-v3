# flash-attn on the pod — receipts (Phase 6.0, 2026-09-29 06:16–06:27Z)

Run statement: `../run-statements/pod-flash-attn-setup.md`. Summary and interpretation: `../results.md`, section "Phase 6.0 — flash-attn on the pod".

## Identity

- Pod `w7ia3zvxqsvs3g` (RUNNING, $4.18/h; 2× RTX PRO 6000 Blackwell Server Edition, cc 12.0, driver 595.91.07). GPU 0 only (`CUDA_VISIBLE_DEVICES=0`). Left running and idle. The idle checks at 06:16:25, 06:24:26, 06:26:39 and 06:28:14Z are retained in `post-run/operator_transcript_excerpts.txt`. A fresh read-only check at 06:36:54Z is in `post-run/git_idle_state_post_run.txt`. Every check shows 0 MiB and 0 % on both GPUs, with no python process running.
- Source: `kg/isaiah-gap-closure` @ `69397da3d1559be4fbb525636122d8a9bc614a49`, transferred as `git bundle create v3.bundle HEAD` (sha256 `149763ad2a727cce47242cfe840db7224495ff2d43b79f72466e87b01848c6fd`, 6.9 MB; pod copy at `/workspace/transfer-flash-attn-2026-09-29/v3.bundle`, same hash). Cloned to `/workspace/kg-v3-rebuild`. `pod/source_commit.txt` records only the commit hash. The detached HEAD and clean tree are backed by other receipts:
  - the `git checkout` output at 06:18:38Z and the tracked-file `git status` count of 0 at 06:26:39Z (`post-run/operator_transcript_excerpts.txt`);
  - a fresh post-run receipt at 06:36:54Z (`post-run/git_idle_state_post_run.txt`): HEAD `69397da`, `symbolic-ref` exit 1 (detached), `git status --porcelain` empty, `runs/` gitignored.

  "Nothing pushed; no credentials copied" is **operator-reported**. The only receipt is the clone's `git remote -v`, which shows just the bundle path.

  `/workspace/kg-v3` was checked read-only at 06:38Z (`post-run/untouched_paths_post_run*.txt`):
  - no mtime changes;
  - 19,796 `.venv` entries have a new ctime. All of them are hard-linked regular files: none has link count 1, and none is a non-regular entry. The sampled files share an inode with the new venv (link count 3), which fits uv hard-linking from its cache. Their mtimes are unchanged, but the two venvs share inodes;
  - `/workspace/gemm-limits-src-1ddc71d` is unchanged.
- Environment: `/workspace/kg-v3-rebuild/.venv`, uv 0.9.0 (pod system), Python 3.12.3, rustc 1.97.0-nightly (e9e32aca5 2026-04-17) = `nightly-2026-04-18` already on the pod.

## Versions actually installed (`pod/versions.json`)

| Component | Version |
|---|---|
| torch | 2.9.0+cu128 (CUDA 12.8, cuDNN 91002, cxx11abi True, arch list incl. sm_120) |
| triton | 3.5.0 |
| flash-attn | 2.8.3 (einops 0.8.2) |
| nvcc (not used; no source build) | 12.8.93 (`/usr/local/cuda-12.8`) |
| maturin | 1.13.1 |

## flash-attn wheel custody

- The lock's sdist (`flash_attn-2.8.3.tar.gz`, sha256 `1e71dd64…0370d`) was built with `FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE` and no build isolation (pyproject). Its `setup.py` downloaded the prebuilt release wheel `flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl` (`pod/02_uv_sync_flash.log`: "Guessing wheel URL"; 7.5 s; no compile).
- Release asset digest (GitHub API): `sha256:4e2f9e39313266b1544b68138b15b91ee6221eccf14f7902b7c6620351340810`, 253,780,426 bytes, asset updated 2025-12-17T09:32:21Z (the v2.8.3 release itself is dated 2025-08-14; the torch 2.9 wheels were added later, so the URL resolves to a mutable release asset). A re-download on the pod (`runs/flash-attn-setup-2026-09-29/wheel/`, 243 MB, not copied here) has that same sha256.
- Installed `flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so`: sha256 `8ca052bf2d3f53baa629e22749b9622a95273c5bffb5f06cd24768ef63f65807`, 997,961,816 bytes — byte-identical to the `.so` inside the release wheel. Receipt: `pod/wheel_member_compare.txt`, produced post-run at 06:37Z by `post-run/wheel_member_compare.sh`. The script extracts the member from the retained wheel into a temp dir, which it removes afterwards. Both files hash to `8ca052bf…5807`, `cmp` exits 0, and the `dist-info/RECORD` entry agrees. The receipt is also kept on the pod in the run dir.
- sm_120 support: flash-attn 2.8.3 `setup.py` defaults `FLASH_ATTN_CUDA_ARCHS="80;90;100;120"` and emits `arch=compute_120,code=sm_120` for CUDA ≥ 12.8. `cuobjdump --list-elf` on the installed `.so` lists 72 cubins each for sm_80, sm_90, sm_100 and **sm_120** (`pod/cuobjdump_list_elf.txt`); no PTX (`pod/cuobjdump_list_ptx.txt`).
- `dist-info/RECORD` sha256 `817df8e9…540e`.

## Rust extension

`uv sync --frozen --group dev --extra flash-attn` installed the project (editable, 20 s), then `uv run --frozen --group dev --extra flash-attn maturin develop` (justfile `build`; dev profile, unoptimized) built `python/owl/rs.abi3.so`, sha256 `35239d1b6c8c0e68d97bdf5c8a2e102caa8ca77176c26ff8e1e076c517e693b8`. `owl`, `owl.rs`, `owl.model.kaggriculture` import from `/workspace/kg-v3-rebuild` (`pod/05_import_check.log`). The first `uv run --frozen pytest` reinstalled the editable `owl` once ("Uninstalled 1 / Installed 1"); `owl.rs` hash unchanged afterwards (`pod/09_owl_rs_after_pytest.log`).

## Files

- `smoke_flash.py` is the smoke script, sha256 `0e1db99db9a9649015315b09a79eac8306df83edeed4a9f59a7168ff5823dee3`. The same bytes ran on the pod (sha256 check at 06:24:26Z in the transcript excerpts).
  - **Relocation:** on the pod it sits in the run dir, `runs/flash-attn-setup-2026-09-29/smoke_flash.py`. Locally it is one level above `pod/`, so the local manifest lists it as `../smoke_flash.py`.
- `pod/01…04_*.log` — sync/build logs; `pod/05_import_check.log`; `pod/06_smoke_flash.log` (includes autotune output); `pod/07`, `pod/08` — `tests/owl/model/test_attn.py` (7 passed, 0 skipped); `pod/10_pytest_kaggriculture.log` (161 passed).
- `pod/smoke/smoke_flash.json` — all measurements; `pod/smoke/kernels_*.txt` — CUDA kernel names per profiled forward.
- `pod/MANIFEST.sha256` was regenerated locally on 2026-09-29 after Codex review r1. It excludes itself. The original had recorded the empty-file hash for its own entry.
  - It covers every local `pod/` payload file, including the new `wheel_member_compare.txt`, plus `../smoke_flash.py` and `../post-run/*`. Check it with `cd pod && shasum -a 256 -c MANIFEST.sha256`.
  - The wheel (`wheel/…whl`, sha256 `4e2f9e39…0810`) is remote-only and is not listed.
  - The pod-side `MANIFEST.sha256` was left unchanged. It still carries the stale self-entry and does not list `wheel_member_compare.txt`.
- `post-run/` holds receipts added after Codex review r1:
  - `operator_transcript_excerpts.txt`, extracted by `extract_operator_transcript.py`, contains the timestamped operator-session calls and results: pre-run statement `Write` 06:17:58.822Z, content sha256 equal to the committed pre-run text; pre-check; first pod mutation 06:18:04Z; idle re-checks. The pod SSH host and port are redacted. The source is a local harness transcript outside git, not tamper-evident; its sha256 at extraction is recorded.
  - `git_idle_state_post_run.txt` covers git and idle state, and `untouched_paths_post_run.txt` / `untouched_paths_post_run_followup.txt` cover the untouched-path mtime/ctime checks. All three are read-only and post-run.
  - `wheel_member_compare.sh` is the script behind `pod/wheel_member_compare.txt`.
- Not copied (large, on the pod only): `wheel/…whl` (243 MB, sha256 above), `inductor_cache/` (55 MB, 1,106 files).
