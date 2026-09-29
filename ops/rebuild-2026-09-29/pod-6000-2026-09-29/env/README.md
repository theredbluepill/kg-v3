# Pod aki4vy8kpfldpa environment receipt (2026-09-29, 15:53-15:59Z)

Status: **pending Codex review** (Codex is at its usage limit until Oct 6). Nothing
here is independently verified.

This is environment setup, not an experiment. No training or benchmark ran, so no
run statement or W&B run was made. The only GPU work was the small flash-attn vs
SDPA correctness check below.

## Identity

- Pod `aki4vy8kpfldpa` ("distinctive_purple_gazelle"), $4.18/h. The container
  started at about 15:51Z (the pod list shows 23:51 GMT+8; the processes date from 15:51).
- Source: `kg/isaiah-gap-closure` @ `994818b87041426c6fc442a85fb06932937d58a7`.
  That is the merge of integration tip `5b43062` into `kg/merge-7-1`, and it was the
  branch tip when this worktree was created. The branch `kg/rebuild-pod-6000-evidence`
  was created from it in `/Users/poonszesen/kg-v3-pod6000`.
- Transfer: `git bundle create pod6000.bundle kg/rebuild-pod-6000-evidence`, sha256
  `632edb268034439aedff8e57a61fa0f1a6dd841059a0e060e62b39ae95ce5a26` (24 MB). It was
  cloned to `/root/kg-v3` and checked out detached at `994818b`. `pod/git_state.txt`
  shows a detached HEAD, 0 porcelain lines, and the bundle as the only remote.
- The gitignored Orbit fixtures (`tests/fixtures/generation/`,
  `tests/fixtures/orbit_wars_replays/`) were copied from `/Users/poonszesen/kg-v3/tests/fixtures`
  to both the worktree and the pod.

## Setup (`pod/setup.sh`, `pod/setup.log`)

- rustup 1.29.1 was installed with the minimal profile and no default toolchain.
  `rust-toolchain.toml` then installed `nightly-2026-04-18`, which is rustc
  1.97.0-nightly (e9e32aca5 2026-04-17) with LLVM 22.1.2 and cargo 1.97.0-nightly.
- `UV_LINK_MODE=copy uv sync --frozen --group dev --extra flash-attn` took 45.6 s
  (uv 0.9.0, Python 3.12.3). flash-attn 2.8.3 was built from the lock's sdist with
  `FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE`, so it fetched a prebuilt wheel and compiled
  nothing. There is no `nvcc` on this pod.
- `uv run maturin develop --release` (maturin 1.13.1) took 28.4 s. It produced
  `python/owl/rs.abi3.so` (sha256 `5ad74acd…de69`, mtime 15:55:52Z), and
  `owl.rs.assert_release_build()` passes (`pod/owl_release.txt`). The first later
  `uv run --frozen` printed "Uninstalled 1 package / Installed 1 package" without naming
  the package; that it was the editable `owl` is inferred, not observed.
  The `.so` mtime did not change, and `uv sync --dry-run` then reported "Would make no
  changes". An earlier pod showed the same behaviour (`../../flash-attn-setup-2026-09-29/README.md`).

## Verified versions (`pod/verify_env.json`, `pod/verify_env.py`)

| Component | Value |
|---|---|
| torch | 2.9.0+cu128 (CUDA 12.8, cuDNN 91002, arch list includes sm_120) |
| triton | 3.5.0 |
| flash-attn | 2.8.3; `flash_attn_2_cuda` .so sha256 `8ca052bf…5807`, the same bytes as the wheel whose custody `flash-attn-setup-2026-09-29` records |
| owl.rs | imports from `/root/kg-v3/python/owl/rs.abi3.so`, release build |
| wandb | 0.26.1; `wandb.Api()` authenticated from `/root/.netrc`, default entity `spoon` (only the entity was printed) |

flash-attn check: bf16 `flash_attn_func` compared with `F.scaled_dot_product_attention`
at B=4, L=256, H=8, D=64 on both GPUs, both sm_120. Max abs error was 9.77e-4
non-causal and 1.95e-3 causal on each GPU, under the 2e-2 tolerance, with finite outputs.

## Hardware (`pod/hardware.txt`)

- 2x NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB each, 600 W power
  limit, 2430 MHz max SM clock, PCI 03:00.0 and 04:00.0. Driver 595.91.07, host
  CUDA 13.2.
- Topology: GPU0-GPU1 `PIX`. Both GPUs have CPU affinity 0-63,128-191 on NUMA node 0.
- CPU: 2x AMD EPYC 9575F, 2 NUMA nodes, `nproc` = 256. RAM: 1,511 GiB. Disk: 500 GB
  overlay at `/`. No network volume.
- OS: Ubuntu 24.04.3 LTS, kernel 6.8.0-138-generic, git 2.43.0.
- After setup, both GPUs showed 0 MiB and 0 % use (15:59:13Z).

## Lock and source hashes (`pod/hashes.sha256`)

- `uv.lock` `f3311c31…07f5`
- `Cargo.lock` `01f02e3c…a352`
- `pyproject.toml` `6912a9e9…abb01`
- `Cargo.toml` `893f97a1…f4d5`
- `rust-toolchain.toml` `eb5146fd…a6d`

## BC best checkpoint (bulk data stays on the pod at `/root/bc-best/`)

`bc-best-local.sha256` holds the hashes taken on the Mac from `/tmp/kg-v3-bc-best`.
On the pod, `sha256sum -c` gives OK for all four files:

- `checkpoint_bc_best.pt`: `fd8545872aca59c70e273e9655055e1104cd719588364f463ae87b0d488e6f51`
  (51,911,591 B). This matches the expected hash.
- `checkpoint_bc_best.json`: `545867732db0…35a7`
- `shards-top1/manifest.json`: `ba5fe1c41774…3036`
- `shards-top1/validation/2026-09-22-112208626.npz`: `ab4e62616071…17a3`

Only one validation shard was copied, as the task asked. The manifest is 1.4 MB and
probably lists more shards than are on the pod. Any consumer must accept the subset.

## Limits

- The full `just prepare` and `pytest` suites were not run on the pod. The import,
  release-build and flash-attn checks are the only functional checks.
- Nothing was pushed. No credentials were copied or printed.
