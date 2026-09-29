# Model-only SPS ceiling probe (COMPONENT measurement), 2026-09-29 06:45–06:49Z

**This is a component measurement, not end-to-end SPS.** It is an upper bound on env steps/s per rank *if the engine, host↔device copies, GAE, logging and the DDP all-reduce cost nothing*. Nothing here shows the trainer reaches these numbers.

- Pre-run statement: `../run-statements/model-sps-ceiling.md`, committed in `ed61770` at 06:44:12Z, before launch at 06:45:05Z.
- Source: `e1458d2a717d9d731a367cbb78b98616ee6649f4` (integration HEAD, heads merged), transferred by bundle (sha256 `4e56cc81…d025`) into `/workspace/kg-v3-rebuild`. Receipts: `pod/receipts/git_checkout.txt` and `git_post.txt` (HEAD `e1458d2`, `git status --porcelain` empty at 06:49:16Z; `runs/` is gitignored). There were no Rust, lock or pyproject changes since `69397da`, so no rebuild. `owl` imports from `/workspace/kg-v3-rebuild/python` (`pod/receipts/import_paths.txt`).
- Stack: torch 2.9.0+cu128, triton 3.5.0, flash-attn 2.8.3, RTX PRO 6000 Blackwell Server Edition (94.97 GiB visible to torch), GPU 0 only, 6,252,223 params.
- Script: `bench_model_sps.py` (sha256 `4a7ede9a…a08a`; the pod copy `pod/bench_model_sps.py` is identical). Driver: `pod/run_all.sh`. Per-density logs and JSON are in `pod/`.
- Idle receipts: `pod/receipts/idle_{nvidia_smi,ps}_{pre,prelaunch,post}.txt`. Every capture shows 0 MiB on both GPUs, no compute apps and no python/torchrun/run_ppo/cargo/pip/maturin process. The pod was left running and idle.

## Setup

- Model: preset `configs/model/kaggriculture.yaml`.
- Precision: fp32 params under `torch.autocast(bfloat16)`, with TF32 set by `configure_torch()`.
- Compilation: the trunk is compiled with `mode="max-autotune-no-cudagraphs"` and `dynamic=True`; the heads run eager. Flash is forced.
- Observations: `make_obs` for 128 envs, tiled along the env dim to reach 1,024 and 16,384 rows. The grammar tables are the synthetic `expected_grammar_tables`.
- Timing: CUDA events over 20 timed iterations, after the first call and 5 warmup iterations.

## Results (medians in ms; p90 is within 0.5 % of the median everywhere)

| density | tokens/row | A fwd 256 | B train 1,024 | C teacher 16,384 | D value 256 | update wall (s) | ceiling SPS/rank | ≈2 ranks |
|---|---|---|---|---|---|---|---|---|
| sparse (1/1 actors, 1 shop) | 222 | 11.09 | 147.47 | 851.98 | 7.48 | 3.929 | 2,085 | ~4,170 |
| mid (40/40 actors, 4 shops) | 303 | 12.87 | 177.56 | 987.72 | 9.50 | 4.662 | 1,757 | ~3,514 |
| dense (241/241 actors, 8 shops) | 709 | 25.41 | 326.58 | 1,690.58 | 22.04 | 8.564 | 957 | ~1,913 |

**Formulas.**
- update wall = 64·t_A + t_C + 16·t_B + t_D.
- env steps per update per rank = 128 × 64 = 8,192.
- ceiling SPS per rank = 8,192 / update wall.
- Two ranks give about 2× that. This assumes the DDP all-reduce of 6.25 M params is small; that assumption was not measured.

**Share of update wall.** 16·B takes 60–61 %, C takes 20–22 %, 64·A takes 18–19 % and D takes 0.2–0.3 %, at every density.

**Flash path.**
- `use_flash_attn` returned True on every call in every workload.
- Trunk `pack_sequence` calls per call were as predicted: A, B and D made 1 each. C made 1 at sparse (3,637,248 packed tokens, under the padded bound), 2 at mid (4,194,126 + 770,226) and 3 at dense (4,193,735 + 4,193,735 + 3,228,786).
- The head row chunks in C were 2 (11,096 rows per chunk).

**First call (compile/autotune).**
- The sparse process ran on a cold cache: A took 20.1 s and B 40.1 s.
- The mid and dense processes reused the run-local Inductor/Triton cache: A took 7.4 s, and B took 11.1 s at mid and 8.0 s at dense.
- D reused A's no-grad graph (≤ 0.02 s). C took 0.9–1.8 s.

**Peak memory.**

| density | A | B | C | D |
|---|---|---|---|---|
| sparse | 0.59 | 17.06 | 32.13 | 0.51 |
| mid | 0.59 | 20.92 | 30.74 | 0.51 |
| dense | 0.97 | **40.28** | 31.93 | 0.86 |

Values are `max_memory_allocated` in GiB. Against 94.97 GiB, the highest allocated peak is 42.4 % (B at dense), which is below the 85 % rule.

**Reserved memory is not below 85 %.** The caching allocator's `max_memory_reserved` reached 84.99 GiB (89.5 %) during dense C. It was 76.3 GiB (80 %) at mid and 72.4 GiB at sparse. That happened because C ran after B in the same process and reused its cached blocks, so this is fragmentation and carryover, not live tensors. A real trainer that runs teacher precompute after updates in one process could show about 87 GB in `nvidia-smi`. Mitigations are to run C before B, call `torch.cuda.empty_cache()` between phases, or use `expandable_segments`; none of these was tested.

## Limits

- **Component only.** Engine stepping, obs/action host↔device copies, GAE, logging, W&B, DDP all-reduce and the rollout's Python loop are excluded.
- **(C) is a proxy.** The Kaggriculture model has no `compute_teacher_distillation_targets`. C is `evaluate_actions` under `no_grad` over 16,384 rows in one call. The real path would chunk by `teacher_segments_per_minibatch` and may produce full per-slot distributions instead of chosen-action log-probs.
- **(B) is PPO-shaped, not the `run_ppo` loss.** It uses a scalar per-row ratio, not per_player clipping, and no teacher KL. It uses Muon with default `MuonConfig()` and `clip_grad_norm_(10)`. It runs one epoch (16 minibatches) per update.
- **Uniform densities.** Every env in a batch has the same density; real rollouts mix densities within a batch. Batches are tiled copies of 128 envs, and the synthetic grammar tables and random weights change the sampled programs, not the tensor shapes.
- **No Nsight Systems.** `nsys` is not installed on the pod (checked 06:41:21Z; nothing installed), so no timeline was captured. Phase attribution inside a step (stems vs trunk vs heads vs optimizer, and host gaps) is **unresolved**. Two indirect signals exist. First, D (encode + critic) costs 67–87 % of A, so the heads cost about 3.4–3.6 ms per 256-row sampling call regardless of density. Second, the host wall median equals the CUDA-event median within 0.03 ms, so each call is fully synchronous (`evaluate_actions` has a replay-flag host sync by design).
- The later density processes had a warm compile cache, so their compile times are not cold-start numbers.

## Large files kept on the pod only (not copied, > 1 MB)

- `/workspace/kg-v3-rebuild/runs/model-sps-ceiling-2026-09-29/v3-e1458d2.bundle`: 8,696,441 B, sha256 `4e56cc819f5972eaeca79e1b8357bc2f21d3e23439169c9a628566c73166d025`.
- `inductor_cache/` (150,591,125 B) and `triton_cache/` (95,711,340 B): 4,715 files in total. Their per-file sha256 manifest is `pod/receipts/compile_caches.sha256` (791,346 B, sha256 `0cd548eb8692369583333b6f05e5919576d636830e7a1b486ee3d4a171060043`).

Local copy manifest: `MANIFEST.sha256`.
