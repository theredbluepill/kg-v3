# flash-attn on the pod — receipts (Phase 6.0, 2026-09-29 06:16–06:27Z)

Run statement: `../run-statements/pod-flash-attn-setup.md`. Summary and interpretation: `../results.md`, section "Phase 6.0 — flash-attn on the pod".

## Identity

- Pod `w7ia3zvxqsvs3g` (RUNNING, $4.18/h; 2× RTX PRO 6000 Blackwell Server Edition, cc 12.0, driver 595.91.07). GPU 0 only (`CUDA_VISIBLE_DEVICES=0`). Left running and idle (06:26:39Z: 0 MiB / 0 % on both GPUs, no python process).
- Source: `kg/isaiah-gap-closure` @ `69397da3d1559be4fbb525636122d8a9bc614a49`, transferred as `git bundle create v3.bundle HEAD` (sha256 `149763ad2a727cce47242cfe840db7224495ff2d43b79f72466e87b01848c6fd`, 6.9 MB; pod copy at `/workspace/transfer-flash-attn-2026-09-29/v3.bundle`, same hash). Cloned to `/workspace/kg-v3-rebuild`, detached at that commit, clean tree (`pod/source_commit.txt`). Nothing pushed; no credentials copied. `/workspace/kg-v3` and its `.venv` were not touched.
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
- Installed `flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so`: sha256 `8ca052bf2d3f53baa629e22749b9622a95273c5bffb5f06cd24768ef63f65807`, 997,961,816 bytes — byte-identical to the `.so` inside the release wheel.
- sm_120 support: flash-attn 2.8.3 `setup.py` defaults `FLASH_ATTN_CUDA_ARCHS="80;90;100;120"` and emits `arch=compute_120,code=sm_120` for CUDA ≥ 12.8. `cuobjdump --list-elf` on the installed `.so` lists 72 cubins each for sm_80, sm_90, sm_100 and **sm_120** (`pod/cuobjdump_list_elf.txt`); no PTX (`pod/cuobjdump_list_ptx.txt`).
- `dist-info/RECORD` sha256 `817df8e9…540e`.

## Rust extension

`uv sync --frozen --group dev --extra flash-attn` installed the project (editable, 20 s), then `uv run --frozen --group dev --extra flash-attn maturin develop` (justfile `build`; dev profile, unoptimized) built `python/owl/rs.abi3.so`, sha256 `35239d1b6c8c0e68d97bdf5c8a2e102caa8ca77176c26ff8e1e076c517e693b8`. `owl`, `owl.rs`, `owl.model.kaggriculture` import from `/workspace/kg-v3-rebuild` (`pod/05_import_check.log`). The first `uv run --frozen pytest` reinstalled the editable `owl` once ("Uninstalled 1 / Installed 1"); `owl.rs` hash unchanged afterwards (`pod/09_owl_rs_after_pytest.log`).

## Files

- `smoke_flash.py` — smoke script, sha256 `0e1db99db9a9649015315b09a79eac8306df83edeed4a9f59a7168ff5823dee3` (same bytes ran on the pod).
- `pod/01…04_*.log` — sync/build logs; `pod/05_import_check.log`; `pod/06_smoke_flash.log` (includes autotune output); `pod/07`, `pod/08` — `tests/owl/model/test_attn.py` (7 passed, 0 skipped); `pod/10_pytest_kaggriculture.log` (161 passed).
- `pod/smoke/smoke_flash.json` — all measurements; `pod/smoke/kernels_*.txt` — CUDA kernel names per profiled forward.
- `pod/MANIFEST.sha256` — hashes of everything in the pod run dir except the Inductor cache.
- Not copied (large, on the pod only): `wheel/…whl` (243 MB, sha256 above), `inductor_cache/` (55 MB, 1,106 files).
