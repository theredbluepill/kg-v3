---
type: "Reference"
title: "Pod v3 environment runs flash-attn 2.8.3 forward on sm_120"
description: "Phase 6.0 installed flash-attn 2.8.3 (prebuilt torch 2.9 wheel, 72 sm_120 cubins) in a separate pod venv; the varlen kernel's max error against an fp32 SDPA reference equals BF16 SDPA's (0.00359), and the forced packed flash path runs in eager and compiled trunk forwards. Forward only, trunk numerics unqualified; the wheel is a mutable release asset outside uv.lock, and the old venv's files are now hard-linked, with sampled files sharing inodes with the new venv."
tags: ["kaggriculture-v3", "cuda", "environment"]
status: "verified-scoped"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-09-29"}
sources: [{"resource": "repository:ops/rebuild-2026-09-29/results.md"}, {"resource": "repository:ops/rebuild-2026-09-29/run-statements/pod-flash-attn-setup.md"}, {"resource": "repository:ops/rebuild-2026-09-29/flash-attn-setup-2026-09-29/README.md"}, {"resource": "repository:ops/rebuild-2026-09-29/flash-attn-setup-2026-09-29/smoke_flash.py"}, {"resource": "repository:ops/rebuild-2026-09-29/flash-attn-setup-2026-09-29/pod/versions.json"}, {"resource": "repository:ops/rebuild-2026-09-29/flash-attn-setup-2026-09-29/pod/wheel_member_compare.txt"}, {"resource": "repository:ops/rebuild-2026-09-29/flash-attn-setup-2026-09-29/pod/cuobjdump_list_elf.txt"}, {"resource": "repository:ops/rebuild-2026-09-29/flash-attn-setup-2026-09-29/pod/07_pytest_test_attn.log"}, {"resource": "repository:ops/rebuild-2026-09-29/flash-attn-setup-2026-09-29/post-run/untouched_paths_post_run.txt"}, {"resource": "repository:ops/rebuild-2026-09-29/flash-attn-setup-2026-09-29/post-run/untouched_paths_post_run_followup.txt"}, {"resource": "repository:ops/rebuild-2026-09-29/flash-attn-setup-2026-09-29/pod/MANIFEST.sha256"}, {"resource": "repository:uv.lock"}, {"resource": "pod-artifact:w7ia3zvxqsvs3g:/workspace/kg-v3-rebuild/.venv"}, {"resource": "external-url:https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3"}]
---

# Pod v3 environment runs flash-attn 2.8.3 forward on sm_120

Phase 6.0 of the rebuild set up a **separate v3 environment** on pod `w7ia3zvxqsvs3g`, and the model's forced packed FlashAttention path **runs forward on sm_120**. This clears the "no flash-attn on the pod" blocker for the forward path only. Trunk numerics, backward, rollout, PPO and evaluation are not qualified here.

## Environment

- Checkout: `/workspace/kg-v3-rebuild`, from `kg/isaiah-gap-closure` @ `69397da` via a git bundle, detached and clean (post-run receipt).
- Venv: `/workspace/kg-v3-rebuild/.venv`, built with `uv sync --frozen --group dev --extra flash-attn`.
- Installed: torch 2.9.0+cu128, triton 3.5.0, flash-attn 2.8.3 (`pod/versions.json`).
- flash-attn came from a **prebuilt release wheel**, `flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl`, which the lock's sdist downloads with `FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE`. No compile ran. The installed `.so` holds 72 sm_120 cubins alongside sm_80/90/100 (`cuobjdump --list-elf`).
- The Rust extension is a **dev-profile** (unoptimized) `maturin develop` build. Build `--release` before any environment-throughput measurement.

## What ran

- **Kernel against SDPA.** `flash_attn_varlen_func` on BF16 packed q/k/v (8 heads × 32, 256 sequences, 173,518 tokens) matches an fp32 per-sequence SDPA reference as closely as BF16 SDPA does (max |Δ| 0.00359 for both). Flash and BF16 SDPA differ by at most 2⁻⁸ = 0.00390625. The magnitude of the element with that largest difference was not retained, so this is not expressed in BF16 spacings. No element falls outside `0.02 + 0.02|ref|`.
- **Model trunk.** With the preset `configs/model/kaggriculture.yaml` (`force_flash_attn: true`) under bf16 autocast, the packed flash path ran in both the **eager** and the **compiled** (`max-autotune-no-cudagraphs`, dynamic) trunk forward. The profiler records `flash::flash_fwd_kernel` in both, with one `pack_sequence` call per forward.
- **Tests on the pod.** `tests/owl/model/test_attn.py` passed 7 of 7, including the flash-backend test that skips on the Mac; `tests/kaggriculture` passed 161 of 161.

## Limits

- **Forward only.** The flash backward kernels and compiled backward through the packed trunk were not exercised. At `69397da` the actor heads did not exist (the critic head did), and the model was not wired into `run_ppo`.
- **Trunk numerics are not qualified.** In every trunk comparison (compiled vs eager flash, flash vs padded SDPA), 0.017–0.021 % of present-token elements fall outside `0.02 + 0.02|ref|`, with max |Δ| up to 0.087. BF16 rounding over 8 layers is a plausible explanation, not a proven one. The discriminating check, an fp32 reference with autocast and TF32 off that retains outlier magnitudes and locations, has not run.
- The timings in the smoke are single profiled forwards, not throughput evidence.

## Hazards

- **Wheel custody.** The torch 2.9 wheel is a release asset added on 2025-12-17, after the v2.8.3 tag. `setup.py` resolves it by URL, and `uv.lock` pins only the sdist hash. A rebuilt pod could therefore resolve a different kernel **without any lock change**. Recheck both digests after any rebuild:
  - wheel sha256 `4e2f9e39313266b1544b68138b15b91ee6221eccf14f7902b7c6620351340810` (equals the GitHub release digest at install time);
  - installed `flash_attn_2_cuda` `.so` sha256 `8ca052bf2d3f53baa629e22749b9622a95273c5bffb5f06cd24768ef63f65807`, byte-identical to the wheel member (`cmp` exit 0).
- **Hard-linked venvs.** After the install, 19,796 files in `/workspace/kg-v3/.venv` had a new ctime and are hard-linked regular files. Cross-venv inode identity was checked for two sampled files only (`smmap/buf.py` and `bin/maturin`, link count 3; `post-run/untouched_paths_post_run_followup.txt`). uv hard-linking from its cache fits this but is inferred. Treat every such file as possibly shared: an in-place edit of a shared file in either venv changes the other. Avoid in-place edits of installed files, or create venvs with `uv --link-mode=copy`.
- **Operator-reported custody.** "No credentials copied" and "nothing pushed" have no retained receipt. The only evidence is the clone's remote list, which shows just the bundle path.

## Consequences

- Phases 6.1–6.4 use this venv or reproduce both digests above. Any pod rebuild verifies the release-asset digest before trusting the kernel.
- 6.1 must still cover flash backward, rollout, the PPO update and evaluation, and should run the fp32 discriminating trunk check.
- Earlier "no flash-attn on the pod" statements, including the FlashAttention limit in the [[compiled-gemm-template-overflows-above-2-21-rows|compiled-GEMM Reference]], predate this setup. That note's revision belongs to the ATEN implementation lane.

## Verification

- Codex reviewed the receipts twice: `verify-flash-attn-r1` (APPROVE WITH EDITS, edits applied in `78df33c`) and `verify-flash-attn-r2` (APPROVE). The reports are local working transcripts in `ops/rebuild-2026-09-29/codex/`, not committed.
- On this checkout, `shasum -a 256 -c MANIFEST.sha256` in `flash-attn-setup-2026-09-29/pod/` reports 30 of 30 files OK (2026-09-29). This checks local custody of the receipts, not the pod's live venv.
- The wheel-member receipt (`pod/wheel_member_compare.txt`) and the cubin listing are the retained evidence for the digest and sm_120 claims.

Concept search before writing: the [[compiled-gemm-template-overflows-above-2-21-rows|compiled-GEMM Reference]] (FlashAttention limit), the [[../decisions/restart-the-port-from-isaiahs-clean-base|restart Decision]] (flash-attn pin unchanged), the historical [[shared-ppo-adapts-game-batches-without-a-second-loop|shared-PPO Reference]] (reference-branch FlashAttention use) and the [[../decisions/start-multi-gpu-qualification-with-two-ranks|multi-GPU Decision]] (host facts live in ops). None records a current-tree flash-attn environment. The [[kaggriculture-encoder-reuses-isaiah-stateless-layers|encoder Reference]] leaves CUDA flash execution to Phase 6; this note is that first, forward-only step. Full evidence: `ops/rebuild-2026-09-29/results.md`, "Phase 6.0 — flash-attn on the pod".
