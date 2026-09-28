# GPU verification environment

Owned benchmark pod: `6mlzh6v4c89w2x`, `kg-v3-sps-2gpu-20260929`.
Two NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs, each 97887 MiB;
64 reported vCPUs. Pod GPU rate read back as $4.18/hour, plus disk.
50 GB ephemeral container disk, no volume. No existing pod changed.
Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, selected from existing
account inventory. Ubuntu24.04.3, host kernel6.8.0-134-generic, driver595.91.07.
The driver reports CUDA13.2 capability; the locked Torch wheel uses CUDA12.8.

Deployment uses a git bundle of upstream HEAD plus an overlay of tracked and
untracked nonignored working files. No remote git push or project commit.
`uv sync --locked` installs Torch2.9.0+cu128 and project dependencies.
Rust installed through official rustup installer, minimal profile,
`nightly-2026-04-18` (`rustc1.97.0-nightly e9e32aca5 2026-04-17`).
Explicit `uv run maturin develop --release` selects release native code.
Credentials are not transferred into the source archive; benchmark debug logs
are local evidence, not a claim of W&B synchronization.

Nsight installation follows the official installation guide's CLI-only option:
<https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html>.
The image's CUDA apt repository update failed a mirror metadata size mismatch;
no CUDA or driver packages were installed. The official devtools repository's
`Packages.gz` identified the standalone CLI package:
`https://developer.download.nvidia.com/devtools/repos/ubuntu2404/amd64/NsightSystems-linux-cli-public-2026.5.1.161-3889610.deb`.
SHA256 `61829db6392e5c293ada1319df86a97c3356bc810335a1975cb551fcfe3eca08`
was verified before `dpkg -i /tmp/nsys.deb`.
Version: NVIDIA Nsight Systems2026.5.1.161-265138896106v0.
Installed profile help supports `--kill=none` and `--wait=all`.
`nsys status -e`: kernel paranoid4; perf_event_open/sampling unavailable.
Security settings remain unchanged; capture disables CPU sampling/context switches.

An initial source extraction raced the still-running SCP and failed EOF;
after SCP completed, extraction was repeated successfully before dependency
installation or execution. The failure did not produce a benchmark result.
