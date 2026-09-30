# Pod abl4mvr5w1mmn4 environment receipt (2026-09-30, 23:54Z Sep 29 to 00:03Z)

This is environment setup, not an experiment. No training or benchmark ran, so there is
no run statement and no W&B run. The only GPU work was the small flash-attn vs SDPA
correctness check below. Nothing here has been independently reviewed.

## Identity

- Pod `abl4mvr5w1mmn4` ("official_indigo_ocelot"), $8.36/h. The container's PID 1
  started at 2026-09-29 23:54:27Z (07:54 GMT+8, matching the pod list).
- Source: integration tip `kg/isaiah-gap-closure` @
  `7e87f5420b3696141ea41f6453d4fafb2748eea0` ("Record the BC W&B landing ...").
  The branch `kg/rebuild-pod4-evidence` was created from it in `/Users/poonszesen/kg-v3-pod4`.
- Transfer: `git bundle create pod4.bundle kg/rebuild-pod4-evidence`, sha256
  `4077681770d49c8b8594baaa8d5e54039b98a701c0434d072b6e26362a0d4fdb` (26 MB), matched on
  the pod. It was cloned to `/root/kg-v3` and checked out detached at `7e87f54`.
  `pod/git_state.txt` shows a detached HEAD, a clean tree (0 porcelain lines) and the
  bundle as the only remote.
- The gitignored Orbit fixtures (`tests/fixtures/generation/`,
  `tests/fixtures/orbit_wars_replays/`) were copied from
  `/Users/poonszesen/kg-v3-int/tests/fixtures` to the worktree and, as a tarball
  (sha256 `546b03c4…1c93`), to the pod. macOS AppleDouble `._*` files from the tarball
  were deleted on the pod before `git_state.txt` was taken.

## Setup (`pod/setup.sh`, `pod/setup.log`)

- rustup 1.29.1 was installed with the minimal profile and no default toolchain;
  `rust-toolchain.toml` then auto-installed `nightly-2026-04-18` (rustc 1.97.0-nightly
  e9e32aca5 2026-04-17, cargo 1.97.0-nightly eb94155a9 2026-04-09).
- `UV_LINK_MODE=copy uv sync --frozen --group dev --extra flash-attn` took 2m43.8s
  (uv 0.9.0, Python 3.12.3; slower than pod 6000's 45.6 s because of package downloads).
  flash-attn 2.8.3 was built from the lock's sdist path, which fetched a prebuilt wheel.
- `uv run maturin develop --release` (maturin 1.13.1) took 35.9 s and produced
  `python/owl/rs.abi3.so`, sha256 `5ad74acde8162e800763987a83e9aca4952b603cdadf9fe52eb0cf947a49de69`.
  That is byte-identical to the pod 6000 build of the earlier source `994818b`.
- As on earlier pods, the first `uv run --frozen` printed "Uninstalled 1 package /
  Installed 1 package" (`pod/verify_env.err`); afterwards `uv sync --dry-run` reported
  "Would make no changes", and the release-build assertion ran after that reinstall.

## Verified versions (`pod/verify_env.json`, `pod/verify_env.py`)

| Component | Value |
|---|---|
| torch | 2.9.0+cu128 (CUDA 12.8, cuDNN 91002, arch list includes sm_120) |
| triton | 3.5.0 |
| flash-attn | 2.8.3; `flash_attn_2_cuda` .so sha256 `8ca052bf…5807`, the same bytes pod 6000 recorded |
| owl.rs | `/root/kg-v3/python/owl/rs.abi3.so`; `owl.rs.assert_release_build()` passed |
| wandb | 0.26.1; `wandb.Api()` authenticated from `/root/.netrc`, default entity `spoon` (only the entity was printed) |

flash-attn check: the script asserts 4 visible devices, then compares bf16
`flash_attn_func` with `F.scaled_dot_product_attention` at B=4, L=256, H=8, D=64 on every
GPU (all sm_120). Max abs error was 9.77e-4 non-causal and 1.95e-3 causal on each of the
4 GPUs, under the 2e-2 tolerance, with finite outputs.

## Hardware (`pod/hardware.txt`)

- 4x NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB each, 600 W power limit,
  2430 MHz max SM clock, PCI 06:00.0, 77:00.0, F4:00.0, F5:00.0. Driver 595.91.07 (the
  compile gate's probed driver), host CUDA 13.2.
- Topology: GPU0-GPU1 `NODE` (NUMA 0, CPUs 0-63,128-191); GPU2-GPU3 `PIX` (NUMA 1,
  CPUs 64-127,192-255); every cross pair is `SYS`. No NVLink. GPU1 shares a `PIX` switch
  with the two RoCE NICs.
- CPU: 2x AMD EPYC 9535 64-Core, 2 NUMA nodes, `nproc` = 256. RAM 1,511 GiB, no swap.
  Disk: 500 GB overlay at `/`. No network volume.
- OS: Ubuntu 24.04.3 LTS, kernel 6.8.0-137-generic, git 2.43.0.
- After setup all 4 GPUs showed 0 MiB and 0 % use (00:01Z).

## Lock and source hashes (`pod/hashes.sha256`; the Mac worktree gives the same five)

- `uv.lock` `f3311c31…07f5`
- `Cargo.lock` `01f02e3c…a352`
- `pyproject.toml` `6912a9e9…abb01`
- `Cargo.toml` `893f97a1…f4d5`
- `rust-toolchain.toml` `eb5146fd…a6d`

The lock hashes are the same as pod 6000's at `994818b`.

## BC best checkpoint (bulk data stays on the pod at `/root/bc-best/`)

The source was the durable Mac copy `/Users/poonszesen/kg-v3-runs/bc-best/`, which
already existed, so no new copy was made. `bc-best-local.sha256` holds the hashes taken
on the Mac; on the pod `sha256sum -c` gives OK for both files:

- `checkpoint_bc_best.pt`: `fd8545872aca59c70e273e9655055e1104cd719588364f463ae87b0d488e6f51`
  (51,911,591 B), matching the expected hash.
- `checkpoint_bc_best.json`: `545867732db0…35a7`

The BC shards were not copied; the PPO warm start reads only the checkpoint.

## Limits

- `just prepare` and `pytest` were not run on the pod. The import, release-build,
  flash-attn and W&B-auth checks are the only functional checks.
- No 4-rank NCCL/process-group check ran here; the first 4-rank run is the first test of it.
- Nothing was pushed. No credentials were copied or printed.
