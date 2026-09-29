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

## Pre-launch addendum — scripts and local checks (committed before launch)

- **Scripts** (local `ops/rebuild-2026-09-29/value-gap-2026-09-29/scripts/`, copied unchanged to the pod run dir; `receipts/scripts.sha256` re-records them at launch):
  - `common.py 69f42d8d…` and `kg_gap.py 6776ddcd…` (cases A–E). `is_gap.py 4ec0937a…` (case F).
  - `gemm_backend_wrap.py 5a070ef6…`, unchanged.
  - `driver.py 896966c3…` and `launch.sh c85bbe38…`.
  - `test_driver_cleanup.py 80ebeacd…`, the local cleanup test, which also provides the self-test dummy stage.
  - `common.py` and the stage scripts were adapted from the GPU checks' `common.py`/`c3_smoke.py` pattern, with the same model construction, registered compile path and `make_obs`.
- **Cleanup test** (`pre-launch/driver_cleanup_test_local.txt`, macOS, Python 3.9, emulated `timeout -s TERM` to the driver's group): **ALL PASS**.
  - Mid-stage SIGTERM: rc 143 8.1 s after the signal. Two groups were terminated, one needing SIGKILL; zero survivors.
  - 8 spawn-storm offsets (0.4–3.1 s, 20–156 live groups whose leaders had exited): rc 143 within 0.13 s, zero survivors. Every stage that wrote its PIDs was in cleanup's registered snapshot.
  - Ordinary completion: rc 0, zero survivors.

  It has not run under GNU `timeout` or on the pod.
- **Code-path dry run** (`pre-launch/dryrun_local.txt`, `VGAP_DRYRUN=1` on the local `8fde43c` worktree, CPU, rows 4/8, no compile, no flash): all steps `ok` for a BF16 "compiled" stand-in with hidden and gain swap, fp32 eager with hidden, and the Isaiah eager stage with state generation. The driver's judge accepted them except for the expected `use_flash_attn` flag, which is false on CPU. **Not evidence.**

## Amendment 1 — attempt 2 (written after attempt 1 stopped, before relaunch; attempt 1's gap numbers not yet read)

**Attempt 1 (10:47:44–10:49:20Z, driver 95.9 s, exit 3).** Launch-time `git_pre.txt` recorded HEAD `8fde43c` and empty porcelain, and the idle gate passed at 0 s. Stage outcomes:

- `isF_eager`, `isF_comp_default`, `kgA_bf16_comp_aten` and `kgB_bf16_eager` passed.
- `kgC_fp32_comp_aten` exited 1. Its first no-grad sample compiled and ran. The grad-enabled `evaluate_actions` then triggered the expected Dynamo recompile, logged as `GLOBAL_STATE changed: grad_mode`. Compiling that training graph failed in Inductor codegen: `InductorError: AssertionError: -704388212327147/1000000000000000` in `tiling_utils.analyze_memory_coalescing` → `extract_normalized_read_writes` → sympy `Expr.is_constant()`. That call evaluates the symbolic index at random points (`_random`), and a torch integer sympy function then asserts on a non-integer value.
- The driver stopped as the stopping rule requires. It terminated the running `isF_comp_aten` (rc −15), started nothing further, and the atexit cleanup found no live group.

Attempt 1 is recorded as it stands: a stop on an Inductor compile error in the fp32 compiled padded-SDPA training graph. That path is not the production BF16 flash path.

**Change.** Only the following changes; thresholds, cases and metrics are unchanged.

1. The two fp32 compiled stages pass `--no-coalesce-tiling`, which sets `torch._inductor.config.triton.coalesce_tiling_analysis = False` before compiling. That skips the failing analysis. Torch's own config comment says the analysis "does not yet apply to dynamic shapes", and every compile here is dynamic. The value is recorded in each result (`inductor_coalesce_tiling_analysis`). BF16 stages are unchanged and keep the default.
2. GPU 0's order becomes `kgA_bf16_comp_aten`, `kgB_bf16_eager`, `kgC_fp32_eager`, `kgA_bf16_comp_default`, `kgC_fp32_comp_aten`, `kgC_fp32_comp_default` (optional). The amended fp32 compiled stages run last, so a repeat failure cannot pre-empt the other cases. GPU 1 is unchanged.
3. Budget: the internal deadline is 40 min and `launch.sh` uses `timeout -s TERM -k 20 2520`. The aggregate therefore stays ≤ 45 min: 95.9 s + 2,520 s + 20 s = 2,636 s.

Scripts: `kg_gap.py 9e1686c4…`, `driver.py bef5b7fb…`, `launch.sh eb3a2ff0…`; the others are unchanged. The local cleanup test was rerun on the amended driver and passed (`pre-launch/driver_cleanup_test_local_attempt2.txt`).

**Attempt 2** reruns every stage from the start in the same run dir, with fresh caches and freshly generated Orbit Wars states. Attempt 1's outputs, caches, states and scripts move to `attempt1/` on the pod before the relaunch.

## Amendment 2 — attempt 3 (written after attempt 2 stopped, before relaunch; no gap numbers read from attempts 1–2)

**Attempt 2 (10:52:29–10:53:41Z, driver 72.3 s, exit 3).** The idle gate passed at 0 s. Stage outcomes:

- `isF_eager`, `kgA_bf16_comp_aten` and `kgB_bf16_eager` passed.
- `kgC_fp32_eager` failed with `torch.OutOfMemoryError`. It was trying to allocate 710 MiB with 92.37 GiB allocated by PyTorch. The failure came at 1,024 rows, inside the supplementary grad-enabled `compute_value`, while the grad replay's graph was still alive. The 256-row pair and hidden cells had completed.
- For scale, the BF16 stages' peak allocation at 1,024 rows was 56.8 GiB (compiled) and 71.5 GiB (eager). This is a planning error in the fp32 memory estimate: fp32 padded SDPA roughly doubles the activations, and the supplementary cells hold two grad graphs at once.
- The driver stopped as required. It terminated `isF_comp_default` (rc −15) and started nothing further.

Attempt 2 is recorded as it stands.

**Change.** Only the following changes; thresholds and metrics are unchanged.

1. The three fp32 stages run at **256 rows only** (`--rows 256`). Prediction 2 (C) is therefore evaluated at 256 rows only, against A at 256 rows. Every BF16 and Isaiah stage keeps 256 and 1,024 rows.
2. Budget: the internal deadline is 38 min and `launch.sh` uses `timeout -s TERM -k 20 2400`. The aggregate therefore stays ≤ 45 min: 95.9 + 72.3 + 2,400 + 20 = 2,588 s.

Scripts: `driver.py 68516ad9…` and `launch.sh bd98fe95…`; the others are unchanged from Amendment 1. The local cleanup test was rerun and passed (`pre-launch/driver_cleanup_test_local_attempt3.txt`).

**Attempt 3** reruns every stage from the start, with fresh caches and states. Attempt 2's files move to `attempt2/` on the pod.

## Amendment 3 — rerun of the two compiled Isaiah stages (written after attempt 3 completed and its results were read)

**Attempt 3 (10:58:27–11:02:23Z, driver 236.5 s, exit 0).** The idle gate passed at 0 s, and every stage passed the driver's criteria. Post receipts show HEAD `8fde43c`, empty porcelain and idle GPUs.

**Instrumentation defect found after the run (information error, mine).**
- `is_gap.py` replaced `stateless_transformer_v1.use_flash_attn` with a closure that appends to a list.
- Isaiah's attention calls `use_flash_attn` **inside** the compiled trunk (`stateless_transformer_v1.py:3100–3102`). Dynamo therefore guarded on the list's length and recompiled on every call.
- Each compiled F stage compiled 8 graphs, one per call of the eight 256-row calls, and then logged `torch._dynamo hit config.recompile_limit (8)`. **All 1,024-row F cells ran eagerly**, so their gap of 2.4e-7 is not a compiled measurement.
- The 256-row F cells are valid compiled comparisons. S ran graph 0/0 (no grad) and R ran graph 0/1 (grad, recompiled on `grad_mode`).
- `kg_gap.py` is not affected. It wraps `kaggriculture.use_flash_attn`, which is called only outside the compiled trunk, and every compiled Kaggriculture stage logged exactly one recompile (`grad_mode`) and no limit.
- The driver's judge missed the defect because the compiled-call counter counts calls into the compiled callable, including those Dynamo runs eagerly.

**Change (Amendment 3).**
1. `is_gap.py` no longer wraps `use_flash_attn`. Flash use is established by `force_flash_attn: true`: his trunk raises on any CUDA call that cannot use flash (`_requires_flash_attn`, lines 727–735 and 3100–3105). A probe `use_flash_attn(bf16 CUDA tensor)` is also recorded.
2. The driver fails any compiled stage whose log contains `recompile_limit`.
3. Only `isF_comp_default` (GPU 0) and `isF_comp_aten` (GPU 1) are rerun, in parallel (`VGAP_STAGESET=frerun`). They run in the sub-dir `frerun/` with fresh caches, on **attempt 3's saved states** (`frerun/obs` → `../obs`, sha256 in `large_files.sha256`), so they compare against attempt 3's `isF_eager` and Kaggriculture stages on identical inputs. The rerun also repeats the valid 256-row cells.
4. Budget: internal deadline 14 min, `timeout -s TERM -k 20 960`. The aggregate is 95.9 + 72.3 + 236.5 + 960 + 20 = 1,385 s ≤ 45 min.

Thresholds and prediction 5's rule are unchanged. Prediction 5 is evaluated on the rerun's cells. Attempt 3's F cells are retained and reported as the 256-row repeat and the invalid 1,024-row cells.

Scripts: `is_gap.py 6ea882a6…`, `driver.py fe37b01d…`, `launch.sh fff11d40…` (run dir, timeout and stage set taken from the environment); the others are unchanged. The local cleanup test was rerun and passed (`pre-launch/driver_cleanup_test_local_attempt3_frerun.txt`).
