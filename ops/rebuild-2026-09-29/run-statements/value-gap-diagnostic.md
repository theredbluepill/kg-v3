# Run statement — value gap between no-grad sampling and grad-enabled replay (diagnostic, 2 GPUs, ≤ 45 min)

**Label: diagnostic.** No model, loss, reward or learning code changes. No timing is measured.

## Question and hypotheses

- **Observation being explained** (`results.md` on `kg/rebuild-gpu-checks`, "GPU checks bundle (component)", check 3): at `8fde43c`, values from grad-enabled replay (`evaluate_actions`) differ from no-grad sampling (`model(obs)`) by up to **0.0154–0.0195** on the [−1, 1] scale in every process (mid/dense × ATEN/default), while event log-probs differ by ≤ 4.1e-4 and no-grad `compute_value` equals the sampled values exactly.
- **Question (owner):** is that gap inherent to BF16 + `torch.compile` (and so also present in Isaiah's model), or does something Kaggriculture-specific in the grad path cause it?
- **H1.** Dynamo guards on grad mode, so no-grad calls and grad-enabled calls run **different compiled graphs** of the trunk (an inference graph; an AOTAutograd forward of the joint training graph). They fuse and round differently in BF16. The critic head (`.out` orthogonal gain **1.0**) passes the resulting trunk noise to the value. The actor heads (`.out` gain **0.01**) shrink it ~100× in the logits. This is inherent to BF16 + compile and is also present in Isaiah's `stateless_transformer_v1`.
- **H2.** Something Kaggriculture-specific in the grad path differs, for example the critic path, masking, or packing/chunking under grad.

## Inputs and code path

- **Source:** pod checkout `/workspace/kg-v3-rebuild` stays at `8fde43cd7408c9c4f9147eeb8f916bd08f8266ed` (`kg/rebuild-trainer-model`, tree `70d50fa3…`). Pre-check 10:36:59Z: HEAD `8fde43c`, porcelain empty. HEAD, tree, porcelain and `git check-ignore` of the run dir are recorded pre and post (`receipts/git_pre.txt`, `git_post.txt`). No transfer, no checkout change, no Rust rebuild (`python/owl/rs.abi3.so` sha256 `35239d1b…` recorded pre/post).
- **Environment:** `/workspace/kg-v3-rebuild/.venv` unchanged (torch 2.9.0+cu128, triton 3.5.0, flash-attn 2.8.3; driver 595.91.07; 2× RTX PRO 6000 Blackwell Server Edition, sm_120). No installs; no driver, CUDA or torch changes. `/workspace/kg-v3` and both venvs' files are not touched (they share uv hard links).
- **Kaggriculture model:** preset `configs/model/kaggriculture.yaml` (width 256, depth 8, 8 heads, `force_flash_attn: true`), fresh weights with `torch.manual_seed(0)`, `.train()`, fp32 parameters. Observations: `tests/kaggriculture/conftest.py::make_obs` at **mid** density (40 own / 40 rival actors, 4 shops; 303 tokens per row) at **256 rows** (128 envs × 2 seats) and **1,024 rows** (512 envs). Sampling seed `torch.manual_seed(1)` before each `model(obs)`.
- **Isaiah control model:** `configs/scaling_6m.yaml` as `FullConfig` (model `configs/model/stateless_transformer_6m.yaml`: width 256, depth 6, 8 heads, `force_flash_attn: true`; env `entity_based_ext_v2`, 320 entities, `discrete_targets`, `two_player_weight 0.75`; `rl.model_compile` defaults `"trunk"` / `"max-autotune-no-cudagraphs"`, `rl.dtype bfloat16`). Built with `owl.model.create_model`, then `reset_parameters()` as `run_ppo` does for a fresh launch, seed 0, `.train()`. His head gains from source: `_ACTOR_HEAD_INIT_GAIN = 0.01`, `_CRITIC_HEAD_INIT_GAIN = 1.0` (`python/owl/model/stateless_transformer_v1.py:125–126`, Isaiah's commit `99d995d1`). His forward, `evaluate_actions`, critic and trunk compile are unchanged by v3, apart from dispatching the compile through `TrunkCompileAPI` (`6485adb`).
- **Isaiah states:** real Orbit Wars observations from his Rust `owl.rl.VectorizedEnv` on the pod (`n_envs` = 256 and 1,024 rows; 2 or 4 players per row). After `reset()`, **24 steps** with actions sampled by the same fresh model (no_grad, BF16 autocast, as his rollout). The CPU pre-check at 10:40:14Z built the env from `configs/scaling_6m.yaml` and reset it (4 envs, players per row [2, 4, 2, 2]). The first F process saves the final observations (`obs/orbit_obs_{256,1024}.pt`); later F processes load them, so every F setting sees identical states.
- **Paths compared (both models):** **S** = `model(obs)` under `torch.no_grad()` + autocast (the rollout path, `ppo.py` `_collect_rollout`). **R** = `model.evaluate_actions(obs, S.actions)` with grad enabled + autocast (the PPO path).
- **Settings:** `configure_torch()` as `run_ppo` (TF32 on). BF16 = fp32 parameters under `torch.autocast(bfloat16)`. Trunk compiled through the registered path `configure_model_compile(model, model_compile="trunk", mode="max-autotune-no-cudagraphs")` → `compile_transformer_trunk` (`dynamic=True`); heads, stems and critic eager. The GEMM backend is set per process by `gemm_backend_wrap.py` (unchanged from the ATEN A/B and the GPU checks, sha256 `5a070ef6…`): **ATEN-only** (as v3 uses) or the torch default `"ATEN,TRITON,CPP"`. The value is recorded at start and end.

## Cases and stages

Every stage is a fresh subprocess with its own `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR`, and `TORCH_LOGS=recompiles`. GPU 0 runs the Kaggriculture stream and GPU 1 the Isaiah stream, in parallel. No timing is taken, so the streams may share the host.

| stage (GPU) | case | precision | trunk | backends | extras |
|---|---|---|---|---|---|
| `kgA_bf16_comp_aten` (0) | A, D, E | BF16 | compiled | ATEN | hidden, gain swap |
| `kgB_bf16_eager` (0) | B, D, E | BF16 | eager | (ATEN record) | hidden, gain swap |
| `kgC_fp32_comp_aten` (0) | C, E | fp32 (autocast off, TF32 off) | compiled | ATEN | hidden |
| `kgC_fp32_eager` (0) | C, E | fp32 | eager | (ATEN record) | hidden |
| `kgA_bf16_comp_default` (0) | A, D, E | BF16 | compiled | default | hidden, gain swap |
| `kgC_fp32_comp_default` (0, optional) | C | fp32 | compiled | default | hidden |
| `isF_eager` (1) | F | BF16 | eager | (ATEN record) | generates states |
| `isF_comp_default` (1) | F | BF16 | compiled | default | |
| `isF_comp_aten` (1) | F | BF16 | compiled | ATEN | |

- **fp32 attention path.** flash-attn needs BF16/FP16 inputs, so the fp32 stages set `force_flash_attn: false`. There the trunk takes the padded SDPA path, not packed flash. This is a stated difference from the BF16 stages.
- **Per row count (256, 1,024), every stage — the pair:** gaps R − S in values; the critic logit difference `log p(self) − log p(opp)`; event log-probs (Kaggriculture `event`, Isaiah `per_player_entity` and the per-player sum); the per-row joint log-ratio; entropies. Isaiah's gaps are over present players (`still_playing`).
- **Supplementary cells (every stage):**
  - `evaluate_actions` under no_grad vs S. This is the same grad mode with different code. Isaiah's `evaluate_actions` computes values via `log_softmax(...).exp()` while his forward uses `softmax`, a head-formula difference that exists regardless of grad mode.
  - `compute_value` with grad vs S, and under no_grad vs S.
  - A second grad replay vs the first, and a second no-grad sample with the same seed vs S (determinism).
- **Case D (actor head gain 1.0):** after the default-gain cells, the same in-memory model's actor head `.out` layers (`model.actor.get_output_layers()`, 9 layers) are re-initialized with `_init_linear(gain=1.0)` under seed 7. The critic head, trunk and stems are untouched, and the stage records that `critic_head.out` is not among the swapped layers. The pair is then repeated at both row counts. The compiled trunk is reused because its parameters are unchanged.
- **Case E (trunk hidden states):** `encode_observations` under no_grad vs with grad, compared on present tokens: trunk input, all tokens, the **critic-value tokens**, the **plan token** and the own-actor tokens. Each path is also compared with an fp32 eager padded-SDPA reference trunk (autocast off, TF32 off, same trunk input, 128-row chunks). Statistics: max, mean, RMS and p99 |Δ|, relative mean and max, and the exact-equal fraction.
  - **Head swap:** the same critic head, under no_grad, applied to each path's hidden states. It is applied in BF16 autocast and in fp32. This separates trunk noise from head arithmetic.
  - Isaiah's analogue (F) records the all-token, critic-value and plan hidden differences and the head swap.

## Pre-declared discriminating observations

Notation, per stage and row count: **V** = max |R − S| of values; **V̄** = mean |R − S|; **L** = max |R − S| of event log-probs (nats); **HS** = head-swap value max |Δ|.

0. **Reproduction (A):** `kgA_bf16_comp_aten` and `kgA_bf16_comp_default` show V in [0.005, 0.05] at both row counts. If V < 0.005 in both, the gap is **not reproduced**, and the other predictions are reported without attribution.
1. **B (eager BF16).** H1: V ≤ 1e-3 **and** V ≤ 0.1 × V(A-ATEN) at the same row count; bit-identical is expected, since both grad modes run the same eager kernels. H2: V ≥ 0.25 × V(A-ATEN).
2. **C (fp32).** H1: compiled fp32 V ≤ 1e-3 and eager fp32 V ≤ 1e-4. H2: either V ≥ 0.25 × V(A-ATEN).
3. **D (actor gain 1.0).** H1: compiled `L_gain1 / L_gain0.01 ≥ 10` at both row counts, with `L_gain1 ≥ 5e-3` nats, so the log-prob gap grows toward the value-gap scale. The eager BF16 gain-1.0 L stays ≤ 1e-3. Against H1: the compiled ratio stays < 3.
4. **E (hidden states), compiled BF16 ATEN.** H1 requires all of:
   - (i) at the critic tokens, grad-vs-nograd mean |Δ| ≤ 2 × the no-grad path's mean |Δ| vs fp32. The paths differ at the scale of BF16 rounding, not more.
   - (ii) the grad path's mean |Δ| vs fp32 lies within [0.5, 2] × the no-grad path's. Neither graph is anomalous.
   - (iii) HS (BF16 head) ≥ 0.5 × V. The gap enters through the trunk hidden states, not the head arithmetic.
   - (iv) critic-token grad-vs-nograd `rel_mean` ≤ 3 × the all-token `rel_mean`. The noise is not critic-token specific.

   H2 indications: (iii) fails, (iv) fails, or the grad path's error vs fp32 exceeds 2 × the no-grad path's at the critic tokens.
5. **F (Isaiah control).** H1: his compiled HS ≥ 0.25 × Kaggriculture's compiled-ATEN HS at the same row count, under at least one backend setting, and his eager HS ≤ 1e-3. The pair V is reported too, with his head-formula difference (supplementary cell) reported separately. H2: his compiled HS < 0.25 × Kaggriculture's at both row counts and both backends, while A reproduces.

- **Supplementary expectations (not decision rules):**
  - Kaggriculture `evaluate_actions` under no_grad equals S exactly: the same inference graph and the same head code.
  - `compute_value` with grad matches R, not S.
  - Repeats are bit-identical.
  - Compiled stages log a Dynamo recompile triggered by the grad-mode guard.
- **Mixed outcomes** are reported as such, prediction by prediction, and the unresolved part is named. Intermediate values (between an H1 and an H2 bound) are "inconclusive" for that prediction.

## Stopping rule, limits, idle rule

- **First unexpected failure stops the driver.** An unexpected failure is any stage with nonzero exit, a missing or errored result record, a step not `ok`, a wrong or missing backend record, any non-finite value in a recorded comparison, `use_flash_attn` not all true (BF16) or not all false (fp32), or the compiled-trunk call count zero on a compiled stage or nonzero on an eager one. The other stream's running stage is terminated and nothing further starts. **Gap sizes are results, never failures.**
- **Hard 45-min aggregate limit.** The driver's internal deadline is 42 min: no stage starts with < 90 s left, and each subprocess's timeout is the remaining budget. The optional `kgC_fp32_comp_default` starts only with ≥ 10 min left; otherwise it is skipped and logged, which is not a failure. The driver runs under `timeout -s TERM -k 20 2640` (44 min + 20 s grace).
- **Process-group cleanup.** Every stage runs in its own session. On SIGTERM/SIGINT/SIGHUP and at exit, the driver's handler terminates every started stage group: SIGTERM, ≤ 8 s wait, SIGKILL, ≤ 4 s wait. The GPU checks driver's pattern (`kg/rebuild-gpu-checks` `6392160`) left one gap, found by Codex review `verify-gpu-bundle-r2` finding 3: a signal between `Popen` and registration in the main thread. It is closed here structurally:
  - Stages are spawned **only from worker threads**, and the Python-level handler runs only in the main thread, which never spawns.
  - Spawn and registration happen under one non-reentrant lock that the handler also takes before it snapshots the stage groups.
  - A worker that finds cleanup already started does not spawn.

  A local CPU test (dummy sleep stages including a SIGTERM-ignoring grandchild; signals mid-stage and during rapid spawning; ordinary completion) must pass before launch. Its output is committed with the scripts.
- **Idle rule.** Immediately before the driver starts, GPUs 0 and 1 must each show no compute processes and 0 % utilization. Baseline driver-context memory is recorded, not treated as busy. If not idle, poll every 15 s for up to 600 s, then do not run. Raw receipts: `receipts/idle_nvidia_smi_{pre,prelaunch,post}.txt` and `idle_ps_*.txt`, plus a post-copy capture. Pre-check 10:36:59Z: both GPUs 0 MiB, 0 %, no compute apps, no python/torchrun process.
- **nsys:** not used (no timing). The 10:36:59Z pre-check found none (`which nsys` rc 1); `receipts/nsys_check.txt` records it. No install.

## Artifacts, budget, safety

- **Pod run dir:** `/workspace/kg-v3-rebuild/runs/value-gap-2026-09-29/` (gitignored `runs/`): scripts, per-stage logs, JSONL, backend records, receipts, Orbit Wars observation files, compile caches.
- **Local:** `ops/rebuild-2026-09-29/value-gap-2026-09-29/` gets scripts, logs, JSONL/JSON, receipts and a summary. No binaries; no file > 1 MB (larger files are listed with sha256 in a manifest). A section "Value gap diagnostic" is appended to `ops/rebuild-2026-09-29/results.md` on branch `kg/rebuild-value-gap`.
- **Budget and safety:**
  - ≤ 45 min aggregate driver wall on the already-running pod `w7ia3zvxqsvs3g` ($4.18/h, ≈ $3.1 max). No new billable resource.
  - The pod is never stopped, restarted or deleted; it is left running and idle.
  - No credentials are copied and nothing is pushed.

## Limits known before the run

- One stack (torch 2.9.0+cu128, triton 3.5.0, flash-attn 2.8.3, sm_120).
- Fresh weights: head gains at init. A trained critic's logit scale, and a trained actor's, may amplify or shrink the noise differently.
- Kaggriculture observations are synthetic, mid density only. Isaiah's are real, but from 24 steps of a fresh policy.
- The fp32 cases use padded SDPA instead of packed flash, so C changes the attention kernel along with the precision.
- Case D mutates the measured model in place, after its default-gain cells.
- The Isaiah model differs in depth, token set and player count, so F compares magnitudes, not identical computations.
- The generated code of the two graphs is not read; the compile caches are retained on the pod.
- No cookbook note is written by this run.
