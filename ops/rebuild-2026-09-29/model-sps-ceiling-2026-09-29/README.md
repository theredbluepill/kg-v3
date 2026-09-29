# Model-only SPS ceiling probe (COMPONENT measurement), 2026-09-29 06:45–06:49Z

**This is a component measurement, not end-to-end SPS.** The numbers are **conditional estimates for the specified A/B/C/D synthetic schedule**: model-only env steps/s per rank *if the engine, host↔device copies, GAE, logging and the DDP all-reduce cost nothing, and if the eventual trainer's model work matches A/B/C/D*. B and C are surrogate workloads (see Limits), so their costs are not proven bounds on the eventual trainer. Nothing here shows the trainer reaches these numbers.

**Timing evidence only, not numerical qualification.** The script retained no finite-loss or finite-gradient check and no sampling-versus-replay equality check (log-probs from A against `evaluate_actions` in B). At `e1458d2`, `owl.train.utils.configure_model_compile` with `model_compile="trunk"` still rejects the Kaggriculture model (it requires `StatelessTransformerV1`); the probe calls `model.compile_transformer_trunk(...)` directly, so the compiled path timed here is not yet reachable from `scripts/run_ppo.py` config.

*Revised 2026-09-29 after Codex review `../codex/verify-sps-ceiling-r1.md` (APPROVE WITH EDITS, findings 1–6). Retained raw results, logs and receipts are unchanged; only this README's wording and the `MANIFEST.sha256` line for this README changed.*

- Pre-run statement: `../run-statements/model-sps-ceiling.md`, committed in `ed61770` at 06:44:12Z, before launch at 06:45:05Z.
- Source: `e1458d2a717d9d731a367cbb78b98616ee6649f4` (integration HEAD, heads merged), transferred by bundle (sha256 `4e56cc81…d025`) into `/workspace/kg-v3-rebuild`. Receipts: `pod/receipts/git_checkout.txt` and `git_post.txt` (HEAD `e1458d2`, `git status --porcelain` empty at 06:49:16Z; `runs/` is gitignored). There were no Rust, lock or pyproject changes since `69397da`, so no rebuild. `owl` imports from `/workspace/kg-v3-rebuild/python` (`pod/receipts/import_paths.txt`).
- Stack: torch 2.9.0+cu128, triton 3.5.0, flash-attn 2.8.3, RTX PRO 6000 Blackwell Server Edition (94.97 GiB visible to torch), GPU 0 only, 6,252,223 params.
- Script: `bench_model_sps.py` (sha256 `4a7ede9a…a08a`; the pod copy `pod/bench_model_sps.py` is identical). Driver: `pod/run_all.sh`. Per-density logs and JSON are in `pod/`.
- Idle receipts: `pod/receipts/idle_{nvidia_smi,ps}_{pre,prelaunch,post}.txt`. The `pre` (06:44:23Z) and `post` (06:49:16Z) captures show 0 MiB and 0 % on both GPUs. The `prelaunch` capture (06:45:04Z) shows **563 MiB on both GPUs** (P0), 0 % utilization and no running compute processes; it is not a 0 MiB capture. Launch followed one second later (06:45:05Z), although the run statement's safety rule said to wait if GPU memory was in use. No capture shows a compute app or a python/torchrun/run_ppo/cargo/pip/maturin process. The pod was left running and idle.
- Stopping: `pod/run_all.sh` runs the three densities sequentially with `set -u` (no `set -e`), each under its own `timeout 1500` (25 min). It does **not** enforce the statement's first-failure stop (it would continue to the next density after a failure) or its 45-minute aggregate limit (the three timeouts allow up to 75 min). All three densities exited 0; actual execution was 06:45:05Z–06:48:52Z, **227 s** (`pod/driver.log`), so these gaps did not affect the timings.

## Setup

- Model: preset `configs/model/kaggriculture.yaml`.
- Precision: fp32 params under `torch.autocast(bfloat16)`, with TF32 set by `configure_torch()`.
- Compilation: the trunk is compiled with `mode="max-autotune-no-cudagraphs"` and `dynamic=True`; the heads run eager. Flash is forced.
- Observations: `make_obs` for 128 envs, tiled along the env dim to reach 1,024 and 16,384 rows. The grammar tables are the synthetic `expected_grammar_tables`.
- Timing: CUDA events over 20 timed iterations, after the first call and 5 warmup iterations.

## Results (medians in ms over 20 timed CUDA-event samples)

| density | tokens/row | A fwd 256 | B train 1,024 | C teacher 16,384 | D value 256 | update wall (s) | ceiling SPS/rank | ≈2 ranks |
|---|---|---|---|---|---|---|---|---|
| sparse (1/1 actors, 1 shop) | 222 | 11.09 | 147.47 | 851.98 | 7.48 | 3.929 | 2,085 | ~4,170 |
| mid (40/40 actors, 4 shops) | 303 | 12.87 | 177.56 | 987.72 | 9.50 | 4.662 | 1,757 | ~3,514 |
| dense (241/241 actors, 8 shops) | 709 | 25.41 | 326.58 | 1,690.58 | 22.04 | 8.564 | 957 | ~1,913 |

**Variability.** p90/median is within +0.14 % for every workload except sparse B, whose p90 is **148.678 ms vs 147.472 ms median (+0.818 %)**; its last six samples ran at 148.2–148.8 ms. Dense B contains one tail sample of **545.106 ms** (sample 11 of 20) against a **326.579 ms** median (+66.9 %); the other 19 are 326.1–327.1 ms, so its p90 (327.007 ms) and median do not move, but its mean is 337.5 ms. The cause of that tail sample is not known. The script's `update_wall_s_p90` in the JSON is the weighted sum of component p90s; it is **not** a measured whole-update p90, and no whole update was timed.

**Formulas.**
- update wall = 64·t_A + t_C + 16·t_B + t_D (medians).
- env steps per update per rank = 128 × 64 = 8,192.
- ceiling SPS per rank = 8,192 / update wall.
- Two ranks give about 2× that. This assumes the DDP all-reduce of 6.25 M params is small; that assumption was not measured.

**Share of update wall.** 16·B takes 60–61 %, C takes 20–22 %, 64·A takes 18–19 % and D takes 0.2–0.3 %, at every density. This is a share of the synthetic schedule; it does not attribute B's cost to any sub-phase (stems, trunk, heads, backward, Muon).

**Flash path.**
- `use_flash_attn` returned True on every call in every workload.
- Trunk `pack_sequence` calls per call were **counted** by a wrapper and were as predicted: A, B and D made 1 each. C made 1 at sparse (3,637,248 packed tokens; 3,637,248 × 512 = 1.862 × 10⁹ < 2³¹, i.e. under the **packed**-token bound), 2 at mid (4,194,126 + 770,226) and 3 at dense (4,193,735 + 4,193,735 + 3,228,786).
- The head row chunks in C were **calculated from source**, not counted: `ceil(16,384 / head_rows_per_chunk(config))` = 2 with 11,096 rows per chunk (`info.C_head_chunks` in the JSON).

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

Values are `max_memory_allocated` in GiB. The allocated peak is **40.275 GiB** (B at dense), 42.4 % of 94.97 GiB. The plan defines the 85 % memory rule on `max_memory_allocated` (`../plan.md`, "Resource fit"), so for these component workloads the allocated peak is within that rule. This does not qualify complete-trainer memory fit, which also includes rollout buffers, engine state, teacher cache, evaluation and DDP.

**Reserved peak is a separate observation.** The caching allocator's `max_memory_reserved` reached **84.994 GiB (89.5 %)** during dense C. It was 76.3 GiB (80 %) at mid and 72.4 GiB at sparse. That is not the plan's allocated-memory rule, but it is what `nvidia-smi` would show. **Hypothesis (not tested):** C ran after B in the same process and the allocator kept B's cached blocks, so the reserved peak reflects carryover and fragmentation rather than live tensors. No allocator snapshot or phase-order control (C before B, or `torch.cuda.empty_cache()` between phases) was run, so this attribution is unsupported. Candidate mitigations (run C before B, `empty_cache()` between phases, `expandable_segments`) are likewise untested.

## Limits

- **Component only.** Engine stepping, obs/action host↔device copies, GAE, logging, W&B, DDP all-reduce and the rollout's Python loop are excluded.
- **(C) is a proxy.** The Kaggriculture model has no `compute_teacher_distillation_targets`. C is `evaluate_actions` under `no_grad` over 16,384 rows in one call. The real path would chunk by `teacher_segments_per_minibatch` and may produce full per-slot distributions instead of chosen-action log-probs.
- **(B) is PPO-shaped, not the `run_ppo` loss.** It clips one scalar joint ratio per row; because each row is one seat, that is per-player joint clipping (the earlier "not per_player" wording was wrong). It has no teacher KL. It uses Muon with default `MuonConfig()` and `clip_grad_norm_(10)`. It runs one epoch (16 minibatches) per update.
- **Compile path not reachable from config.** `configure_model_compile` rejects the model at `e1458d2`; the probe compiles the trunk directly.
- **No numerical checks.** No finite-loss/gradient or sampling-versus-replay equality check was retained; this is timing evidence only.
- **Uniform densities.** Every env in a batch has the same density; real rollouts mix densities within a batch. Batches are tiled copies of 128 envs, and the synthetic grammar tables and random weights change the sampled programs, not the tensor shapes.
- **No Nsight Systems timeline.** `nsys` absence on the pod is **operator-reported** (pre-check at 06:41:21Z: `which nsys` empty, no `/usr/local/cuda*/bin/nsys`, no `/opt/nvidia/nsight-systems*`); no command-output receipt of that check was retained. Nothing was installed and no timeline was captured. Phase attribution inside a step (stems vs trunk vs heads vs backward vs optimizer, and host gaps) is **unresolved**. A − D (3.4–3.6 ms per 256-row call) is the **incremental cost of the sampling path** over the value-only path; it is not an isolated measurement of head kernels. B's dominance of the update wall does not isolate heads backward or Muon. The host wall median is within 0.03 ms of the CUDA-event median, but the harness calls `end.synchronize()` before stopping the host clock, so that agreement says nothing about whether the model calls are intrinsically synchronous.
- The later density processes had a warm compile cache, so their compile times are not cold-start numbers.

## Large files kept on the pod only (not copied, > 1 MB)

- `/workspace/kg-v3-rebuild/runs/model-sps-ceiling-2026-09-29/v3-e1458d2.bundle`: 8,696,441 B, sha256 `4e56cc819f5972eaeca79e1b8357bc2f21d3e23439169c9a628566c73166d025`.
- `inductor_cache/` (150,591,125 B) and `triton_cache/` (95,711,340 B): 4,715 files in total. Their per-file sha256 manifest is `pod/receipts/compile_caches.sha256` (791,346 B, sha256 `0cd548eb8692369583333b6f05e5919576d636830e7a1b486ee3d4a171060043`).

Local copy manifest: `MANIFEST.sha256`.
