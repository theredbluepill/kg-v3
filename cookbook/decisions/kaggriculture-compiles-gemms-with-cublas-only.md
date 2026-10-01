---
type: "Decision"
title: "Kaggriculture compiles GEMMs with cuBLAS only"
description: "To keep the compiled-GEMM CUDA crash from recurring, every compiled Kaggriculture region runs with Inductor's max_autotune_gemm_backends set to \"ATEN\" (cuBLAS) instead of Isaiah's default \"ATEN,TRITON,CPP\", after a startup check rejects an unprobed torch, triton or NVIDIA driver (probed: 595.91.07 on RTX PRO 6000 sm_120; 570.211.01 on H200 sm_90, correctness half only). The owner asked for the outcome; the implementer chose the setting from a measured A/B (+4.5–6.5 % model-only update wall). Enforcement is CPU-tested only, and the overflow guards stay."
tags: ["kaggriculture-v3", "cuda", "compile", "decisions", "adaptation"]
status: "verified-scoped"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-09-29"}
decider: "Owner: Also can we ensure under any circumstance, the CUDA crash we faced earlier on the compiler issue will never be met in our v3?"
sources: [{"resource": "user-directive:2026-09-29:ensure-the-compiler-crash-never-recurs"}, {"resource": "rebuild-model-branch:9bdd82d:ops/rebuild-2026-09-29/results.md#L259-L303"}, {"resource": "rebuild-model-branch:9bdd82d:ops/rebuild-2026-09-29/aten-gemm-ab-2026-09-29/README.md"}, {"resource": "rebuild-model-branch:9bdd82d:ops/rebuild-2026-09-29/aten-gemm-ab-2026-09-29/codex-verify-r1.md"}, {"resource": "rebuild-model-branch:9bdd82d:ops/rebuild-2026-09-29/run-statements/aten-gemm-ab.md"}, {"resource": "local-untracked:kg-v3/ops/rebuild-2026-09-29/codex/verify-aten-ab-r2.md"}, {"resource": "local-untracked:kg-v3/ops/rebuild-2026-09-29/codex/verify-aten-r1.md"}, {"resource": "local-untracked:kg-v3/ops/rebuild-2026-09-29/codex/verify-aten-r2.md"}, {"resource": "repository:ops/h200-driver-gate-2026-10-01/README.md"}, {"resource": "repository:ops/h200-driver-gate-2026-10-01/run-statement.md"}, {"resource": "repository:python/owl/model/compile_gemm.py"}, {"resource": "repository:python/owl/train/utils.py"}, {"resource": "repository:python/owl/model/kaggriculture.py"}, {"resource": "repository:python/owl/model/stateless_transformer_v1.py"}, {"resource": "repository:python/owl/model/base.py"}, {"resource": "repository:scripts/run_ppo.py"}, {"resource": "repository:python/owl/train/logging.py"}, {"resource": "repository:tests/conftest.py"}, {"resource": "repository:tests/kaggriculture/conftest.py"}, {"resource": "repository:tests/kaggriculture/test_compile_gemm_backends.py"}, {"resource": "repository:tests/kaggriculture/test_model_compile.py"}, {"resource": "repository:tests/kaggriculture/test_model_encoder.py"}, {"resource": "repository:tests/kaggriculture/test_model_heads.py"}, {"resource": "repository:tests/scripts/test_run_ppo.py"}, {"resource": "repository:ops/rebuild-2026-09-29/aten-wiring/py-prepare.log"}, {"resource": "repository:ops/rebuild-2026-09-29/aten-wiring/py-prepare-r1.log"}, {"resource": "repository:README.md"}, {"resource": "repository:docs/model-architecture.md"}, {"resource": "repository:docs/kaggriculture-model.md"}]
---

# Kaggriculture compiles GEMMs with cuBLAS only

## Decider and scope of the decision

The owner asked, after the compiled-GEMM crash ([[compiled-gemm-template-overflows-above-2-21-rows|compiled-GEMM Reference]]):

> Also can we ensure under any circumstance, the CUDA crash we faced earlier on the compiler issue will never be met in our v3?

Source: `user-directive:2026-09-29:ensure-the-compiler-crash-never-recurs`, the owner's message in the Claude Code rebuild session.

The owner asked for the outcome, not for a specific mechanism. **The setting below is the implementer's measured choice.** The owner did not choose `"ATEN"` and did not review the cost.

## Rule

- **cuBLAS only.** Every compiled Kaggriculture region runs with `torch._inductor.config.max_autotune_gemm_backends = "ATEN"`, so compiled `mm`/`addmm`/`bmm` lower to extern cuBLAS calls. Isaiah's torch 2.9.0 default is `"ATEN,TRITON,CPP"`. Every public compile entry point applies the setting before `torch.compile`: `configure_model_compile` for the `mlp` target, and `KaggricultureTransformer.compile_transformer_trunk` itself for the `trunk` target, whether `configure_model_compile` or a caller invokes it directly. The optional `compile_actor_heads` entry point now claims the same setting itself, including with an eager trunk; its per-call actor check covers lazy specialization. The default-off rollout extensions and their CPU/GPU verification boundary are recorded in [[../references/rollout-optimizations-preserve-sampling-behind-default-off-switches|the rollout optimization Reference]]. The claim lives in `python/owl/model/compile_gemm.py`.
- **Every compile mode.** The task asked for max-autotune modes. The setting applies to every mode, because `TORCHINDUCTOR_MAX_AUTOTUNE` can switch on the Triton GEMM templates under any mode. When max-autotune is off, the templates are not candidates, so the setting changes nothing.
- **Probed stack only.** Before the setting, a check rejects any version the A/B did not probe. The probed versions live in one constant, `KAGGRICULTURE_PROBED_COMPILE_STACK` in `python/owl/model/compile_gemm.py`: torch 2.9.0, any local build suffix such as `+cu128`; triton 3.5.0; NVIDIA driver 595.91.07 or 570.211.01.
  - GPU hosts (`torch.cuda.is_available()`) read the driver with `nvidia-smi`. A missing `nvidia-smi`, a missing triton or any unprobed driver fails.
  - An installed triton is checked on every host, with or without CUDA.
  - Hosts without CUDA, such as the owner's Mac, skip the driver check and record why; they cannot report a driver. They skip the triton check, with a recorded reason, only when triton is not installed, since Inductor emits no Triton kernels there.
  - `run_ppo` repeats the check before it creates the run directory.
- **Enforced at compile and at every call.**
  - A direct `KaggricultureTransformer.compile_transformer_trunk` call runs the same stack check and claim as `configure_model_compile`. Setting `"ATEN"` by hand does not skip them.
  - Once a region is compiled, `_run_trunk` re-checks the value before each trunk call. Inductor compiles lazily and recompiles on new dynamic shapes, reading the config at that moment, not at `torch.compile`.
- **Orbit unchanged, one game per process.** Isaiah's Orbit models keep the backends they find. The setting is process-global, so the first compile claims it for one game. `StatelessTransformerV1.compile_transformer_trunk` claims Orbit itself, like the configured paths. Compiling the other game in the same process raises, in either order and through any entry point, before any value changes. A lock serializes the claim's check and write, so this also holds when threads claim concurrently.
- **Telemetry.** `run_ppo` prints the claim and records it as W&B summary fields: `compile_gemm_game`, `compile_gemm_backends`, `compile_stack_torch`, `compile_stack_triton` and `compile_stack_nvidia_driver`. `MetricLogger.set_summary` now accepts strings.
- **The guards stay.** `_run_trunk`'s overflow guard and chunking, and the head-extent guard, stay in place. They remain until the real-trunk backward above the bound and the production compile path are verified, as the A/B recommended.

## Evidence

The ATEN-only GEMM A/B ran at `e1458d2` on pod `w7ia3zvxqsvs3g`. It is recorded on branch `kg/rebuild-model` at `9bdd82d` in `ops/rebuild-2026-09-29/results.md` lines 259–303 and in `aten-gemm-ab-2026-09-29/`. Codex verification r1 requested edits; r2 (`verify-aten-ab-r2.md`) gave **APPROVE**.

- **Correctness above the int32 bound** (lines 269–274, guard bypassed, compared element-wise with eager). Every case that failed under the default backends was correct under ATEN-only:
  - The real packed trunk up to 8,387,470 tokens had 0 wrong tokens.
  - Linear 768→256 and 512→256 had 0 bad rows.
  - The synthetic MLP forward and backward at 4,194,305 rows were clean.
  - The default-backend controls in the same run reproduced 2 and 4,665 wrong trunk tokens, plus an illegal memory access in autotune.
- **Kernel evidence** (lines 276–281):
  - The ATEN-only caches contain zero `triton_tem_` definitions or launches.
  - Every autotune line logs `num_triton_choices: 0`.
  - Decompose-K and persistent-TMA are excluded by source gating on `TRITON` in the backend list.
- **Cost** (lines 283–293, component timing conditional on the synthetic A/B/C/D schedule, not end-to-end SPS). The model-only update wall rose by **+6.5 %** at mid density (4.667 → 4.970 s) and **+4.5 %** at dense (8.561 → 8.944 s). The ceiling fell from 1,755 to 1,648 and from 957 to 916 SPS/rank.
  - Most of the cost is in the small no-grad calls (A +15–18 %, D +17–26 %). That is consistent with lost prologue/epilogue fusion, which is an inference; no timeline was captured.
  - Peak memory is about the same, and cold compile is faster.

### H200 / driver 570.211.01 (2026-09-30, correctness half only)

For the owner's 8×H200 pod ("please setup this 8-gpu pod real quick and notice increased vram & h200"; "Keep the current recipe"), the correctness half and the template census were repeated on NVIDIA H200 (sm_90, 143,771 MiB), driver 570.211.01, torch 2.9.0+cu128, triton 3.5.0, flash-attn 2.8.3, GPU 0 only, source `07c8fc9`. Run statement, outputs and adaptations: `ops/h200-driver-gate-2026-10-01/`.
- Real packed trunk, guard bypassed, at 4,194,305 / 4,198,400 / 4,194,444 / 8,387,470 tokens: 0 wrong tokens, 0 non-finite, max |Δ| ≤ 0.1728, one trunk call each.
- Linear 768→256 at 2,796,203 rows and 512→256 at 4,198,401 and 8,388,608 rows: 0 bad rows, max |Δ| 0.0. MLP 256→512→256 forward and backward at 4,194,305 rows: forward and dX clean, parameter-gradient relative max ≤ 0.0026.
- Every ATEN cache has 0 `triton_tem_` definitions or launches and no decompose-K, persistent-TMA or contiguous-subgraph text; all 10 ATEN autotune lines log `num_triton_choices: 0`.
- The default-backend Linear 768→256 control reproduced the fault on sm_90 (illegal memory access during Triton mm autotuning), so the A/B still discriminates there.
- Adaptation: at `07c8fc9` the trunk compile itself runs the stack check, so the trunk probe replaced `check_compile_stack` with an explicit, flagged recorder. The claim still set `"ATEN"`, and the per-call re-check still ran.
- Not repeated on H200: the timing half, the default-backend trunk control, GPUs 1–7 individually, and multi-rank runs.

## Checks of this version (owner's Mac, CPU, recording stand-ins)

- `tests/kaggriculture/test_compile_gemm_backends.py` (55 tests) covers:
  - `"ATEN"` is in place at every compile call, for both targets and all four modes.
  - Orbit keeps `"ATEN,TRITON,CPP"` and never reads the Kaggriculture stack.
  - The `none` target claims nothing.
  - Both cross-game orders raise through every pair of entry points (configured trunk, configured mlp, direct trunk; 18 cases). The Orbit-first order leaves the value unchanged, and nothing of the second game compiles.
  - A direct Kaggriculture trunk compile claims `"ATEN"` with the checked stack. With `"ATEN"` preset by hand and an unprobed torch, it raises before compiling and claims nothing.
  - A direct Orbit trunk compile claims Orbit without reading the Kaggriculture stack.
  - An Orbit claim from another thread while Kaggriculture reads its stack raises and leaves the Kaggriculture claim and `"ATEN"` in place.
  - A reset after compile stops the next trunk call before any block runs, for both targets.
  - The single constant is 2.9.0, 3.5.0 and drivers 595.91.07 and 570.211.01. Either driver, alone or together, passes the check (added with the H200 gate; `tests/kaggriculture/conftest.py` now stands in a single-driver host).
  - Unprobed torch (2.10.0, 2.9.1, 2.8.0), triton (3.4.0, 3.6.0) and drivers are rejected.
  - A GPU host without triton or without a driver reading is rejected.
  - The skip reasons are explicit when CUDA is absent.
  - An unprobed torch fails before compiling and leaves the claim and the value unchanged.
- `tests/scripts/test_run_ppo.py`:
  - `main` rejects an unprobed torch before any run directory, env or model.
  - The startup check prints the checked stack, and it skips Orbit and `model_compile: none`.
  - The session records the claim as summaries.
- Task 3.1's compile tests (`tests/kaggriculture/test_model_compile.py`) and the direct-compile tests in the encoder and heads suites pass with the probed stack injected. A root `tests/conftest.py` isolates the process-global claim and value for each test.
- Four mutations failed their intended tests (run over `tests/kaggriculture`, `tests/scripts/test_run_ppo.py` and `tests/owl/train/test_config.py`; 555 pass unmutated):
  - not setting `"ATEN"`: 18 failures;
  - dropping the `_run_trunk` re-check: 2;
  - dropping the cross-game check: 2;
  - dropping the torch-version check: 5.
- After Codex verification r1 (`verify-aten-r1.md`, REJECT: direct compiles bypassed the claim; the non-CUDA triton prose was wrong), two more mutations failed the new tests: dropping the claim in `KaggricultureTransformer.compile_transformer_trunk` (21 of 54) and in `StatelessTransformerV1.compile_transformer_trunk` (16 of 54).
- `OMP_NUM_THREADS=2 uvx --from rust-just just py-prepare` passed after the r1 fixes: Ruff, mypy (62 source files), 1,516 passed with 5 skips, and docs freshness (`ops/rebuild-2026-09-29/aten-wiring/py-prepare-r1.log`; the first wiring run is `py-prepare.log`, 1,496 passed).
- Codex verification r2 (`verify-aten-r2.md`, APPROVE WITH EDITS) reproduced a concurrent Orbit claim landing while Kaggriculture read its stack, then being overwritten. The new concurrency test failed on that code (the Orbit thread got a claim instead of an error) and passes with the lock. `py-prepare` after the r2 fixes: `ops/rebuild-2026-09-29/aten-wiring/py-prepare-r2.log`.

## Limits — what this does not guarantee

"Never under any circumstance" is not established. The checks above establish only the following: on the probed stack, compiled Kaggriculture regions cannot select Inductor's Triton GEMM templates, which are the only mechanism attributed to the crash.

- **No GPU run of this code.** The A/B applied the setting through a wrapper at `e1458d2`. This wiring has run only on CPU with stand-ins. `run_ppo` still stops at `require_orbit_env` for Kaggriculture, so the production compile path is unreachable and unverified.
- **Unmeasured paths** (A/B limits, lines 261 and 295–303):
  - FlexAttention builds its own Triton templates regardless of this setting. The model does not use it.
  - Static-shape compiles and `bmm` numerics were not measured.
  - The real-trunk backward was not element-compared above the bound.
  - Non-GEMM Triton kernels rely on Inductor's own 32-bit-indexing guards.
- **Two stacks, one full A/B.** The full A/B covers one GPU model (RTX PRO 6000, driver 595.91.07), one run per arm, with inputs up to 2³² elements. The H200 / 570.211.01 stack has the correctness half and census only (one run, GPU 0): its compiled cost is unmeasured.
- **Orbit.** Isaiah's Orbit path keeps the vulnerable default. It is outside v3's Kaggriculture scope, and this change does not protect it.
- **The driver gate is conservative.** The crash mechanism is Inductor codegen, not the driver. A pod with another driver fails at startup until the A/B is repeated there.

## Reopen when

- A torch, triton or driver upgrade is proposed. Rerun the A/B and both GEMM probes on the new stack, then add it to `KAGGRICULTURE_PROBED_COMPILE_STACK`.
- End-to-end SPS shows that the cost matters. A measured alternative, such as Triton templates limited to shapes below the bound, would need its own A/B.
- The model adopts FlexAttention or static-shape compiles, or a compiled region adds `bmm`.
- The real-trunk backward above the bound and the production compile path are verified. That is also the condition for retiring the guards.
