# Codex brief: cut the Python/GPU half of the rollout step, behind switches

Owner (verbatim, 2026-10-01): "please accelerate SPS boost with CODEX", "please accelerate on SPS diagnosis".

## Context
- Live run: 8x H200, 20 envs/rank (40 seat rows x 709 padded tokens per forward), horizon 720, self-play, bf16 autocast, trunk compiled with `max-autotune-no-cudagraphs` (dynamic=True, GEMMs cuBLAS-only), flash-attn packed path. Rollout is 27 s of a 32 s iteration; ~9-12 s of it is the non-native part (forward dispatch, syncs, copies). GPU util 8-14%.
- Full code map with file:line references: `ops/sps-2026-10-01/code-map.md` (read it first; steps 1-3 and 5-9 are yours). Key facts: ~490 eager aten ops in the actor heads per step (`kaggriculture_actor.py:263-502`, 9 sequential head stages, no syncs, static shapes); `_compiled_actor_core` hook exists but is never set (`python/owl/model/kaggriculture.py:320, 1088`); 3 syncs in `build_packed_sequence` (`stateless_transformer_v1.py:3182-3186`); 2 blocking D2H copies into pageable memory in `_actions_to_cpu` (`ppo.py:2817-2832`); per-step CPU reward telemetry with `isfinite().all()` in `env.step` (`env.py:499-524`, `rewards.py:178-233`).

## Target (each behind its own config switch, default OFF, so the live recipe is unchanged unless enabled)
1. **Compile the actor policy core** (`rl.model_compile` gains an option, or a new knob e.g. `rl.compile_actor_heads: bool`): set `_compiled_actor_core` via `torch.compile` (same GEMM backend claim rules as the trunk — see `python/owl/model/compile_gemm.py` and `cookbook/decisions/kaggriculture-compiles-gemms-with-cublas-only.md`; the claim must cover the heads too). Consider `mode="max-autotune-no-cudagraphs"` first; CUDA graphs only if shapes are static and it is clean. Must keep sampling semantics: same Gumbel noise consumption (RNG call order), same masks, same logp/entropy.
2. **Remove the 3 packing syncs in the rollout forward**: e.g. compute `max_seqlen` from a known bound (padded token count) instead of `.item()`, drop/defer the `.all()` validation on the hot rollout path (keep it in debug/tests), avoid `nonzero` host sync if possible (flash-attn varlen needs cu_seqlens on device; max_seqlen may be an upper bound — verify flash-attn semantics allow an upper bound).
3. **Pinned, non-blocking action D2H**: preallocate pinned CPU buffers for tokens/lengths, `copy_(non_blocking=True)` + a CUDA event synchronize right before the native step needs them.
4. **Move the per-step reward telemetry validation** (`isfinite().all()` etc.) out of the per-step hot path (vectorize once per rollout or keep only when a debug flag is set), keeping the logged metric values identical.

## Hard requirements
- With every switch OFF, behaviour is byte-identical to today (tests must show this).
- With switches ON: CPU parity tests must show identical actions, logp, entropy, values for fixed seeds on CPU (eager vs new path where applicable; for the compiled heads, compare to eager within tight tolerance on CPU inductor, and state that the GPU check will be done on the diagnostic H200 pod). Syncs removal and pinned D2H must be exactly identical in outputs.
- Add a small benchmark script (`scripts/bench_rollout_step.py` or under `ops/sps-2026-10-01/`) that times the rollout step (forward + decode + D2H, and optionally with the env) at 40 seat rows on whatever device is available, with each switch on/off, so we can run it on the H200 diagnostic pod. It must not need W&B.
- Follow repo CLAUDE.md: `uv run ruff format`, `uv run ruff check`, `uv run mypy python/ scripts/`, `uv run pytest` on the touched areas (model, actor, ppo, env, compile_gemm, run_ppo/config tests), `uv run python scripts/check_doc_freshness.py`; update mapped docs (`docs/model-architecture.md`, `docs/rl-api-specs.md`, README config section) if you add knobs. `just` may be missing: run the commands directly.
- Record the adaptation in the cookbook (note + folder index.md + prepended cookbook/log.md entry) with evidence and limits.
- Commit on branch `kg/sps-python` with message ending `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Do NOT push. Do NOT touch any remote machine. Do not change Rust code (another job owns `src/`).

## Report (the -o file)
Switches added (names, defaults), changes (files), parity evidence, benchmark numbers on this Mac (CPU), test results, what still needs GPU verification, limits, and a final line `VERDICT: DONE` or `VERDICT: BLOCKED <reason>`.
