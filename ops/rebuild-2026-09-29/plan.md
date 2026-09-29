# Kaggriculture Rebuild on Isaiah's Clean Base — Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the Kaggriculture port on Isaiah's `32b3ec9` so it follows his `StatelessTransformerV1` layer topology and `scaling_6m` recipe from the first commit. Then run one BC, verify on 2- and 4-rank RTX PRO 6000, and evaluate across opponents. Claude and Codex share the work.

**Architecture:** Isaiah's infrastructure is the base. It stays unchanged except for deliberate, tested seams. Game-specific work lives in two places: a trimmed, provenance-pinned rules kernel, and a v3-owned native environment that writes named per-entity observation tensors into caller-owned buffers, the same pattern as Isaiah's `src/rl`. The model reuses Isaiah's layer classes; only input channel widths and the grammar action heads are game-specific. The prior port on `kg/reference-2026-09-29` serves as a reference and test oracle, not as a source to copy.

**Tech Stack:** Rust (PyO3/maturin, rayon), Python 3.12, PyTorch 2.9 (BF16, Inductor, FlashAttention), pydantic configs, pytest, cargo, W&B, torchrun/NCCL on RunPod RTX PRO 6000, Codex CLI 0.158 (`codex exec`).

**Spec (owner decisions, in `cookbook/decisions/`):** `restart-the-port-from-isaiahs-clean-base`, `the-policy-is-stateless-and-observation-only` (Isaiah's stateless model is the reference, with the same layer topology), `recipe-choices-align-to-isaiah-without-owner-escalation`, `v3-reuses-isaiah-infrastructure-with-kaggriculture-semantics` (Isaiah's principles, one trainer, v3 Rust I/O, no v2 model), `start-multi-gpu-qualification-with-two-ranks`, `throughput-means-correct-complete-work`, `evaluation-preserves-generality-and-evidence`, `diagnose-the-mechanism-before-spending-on-runs`. The previous plan, `kg/reference-2026-09-29:ops/gap-closure-2026-09-29/plan.md`, supplies task designs where they still apply.

## Global Constraints

- Base: Isaiah's `32b3ec9` plus the carried cookbook and agent setup (`552c930`). Integration branch: `kg/isaiah-gap-closure`. Local `main` stays at `32b3ec9` until the owner says otherwise. No force-push without the owner's explicit request.
- One trainer: extend `scripts/run_ppo.py` and `python/owl/train/ppo.py`; never add a second PPO loop.
- v3-owned Rust I/O with caller-owned buffers; no v2 model, collector or trainer code; no ctypes path.
- Isaiah's stateless model is the reference, with the same layer topology (see "Isaiah's principles"). Only game I/O differs. No between-turn state, no opponent identity in inputs, losses, rewards, normalization or selection; the critic reads only its own seat's observation.
- Isaiah's Orbit path keeps working. After every task that touches an Isaiah file, run `uv run --extra reference pytest tests/owl tests/scripts tests/tools -m "not slow" -q` and `cargo test`.
- After Python edits run `just py-prepare`; after Rust edits run `just rs-prepare`; before a merge run `just prepare` and complete `docs/pr-checklist.md`. (`just` runs as `uvx --from rust-just just …` on the owner's Mac.)
- **On the owner's Mac: unit tests and bounded tiny checks only.** A "tiny check" means a tiny model, ≤ 2 environments, ≤ 2 updates, < 1 GB RAM and < 2 min. Anything larger runs on the pod.
- **Pod:** 2- and 4-rank RTX PRO 6000 are authorized for verification. Before each run: read live price and state, check the pod isn't running someone else's learner, and write a run statement. Only Claude operates the pod.
- Every material adaptation gets a cookbook record (note + folder index + prepended log).
- Recipe differences resolve toward Isaiah; residual differences are recorded, not escalated.

### Resource fit on 2× RTX PRO 6000 (owner, 2026-09-29)

Owner: "when fitting in GPUs, make sure we fits the GPU resource, since we are doing on RTX 6000 while Isaiah is doing in Tufa's B200 clusters". Isaiah's `scaling_6m` targets 1× B200 (192 GB). His `winner_ce_6m_4x5090` is the same recipe fitted to 4× RTX 5090 (32 GB, the same compute-capability-12.x generation as ours). That's the precedent: keep the global configuration identical (I2) and fit only the per-rank shapes and the spm/accumulation split (I1).

- **Hardware, from the pod read:** 2× RTX PRO 6000 Blackwell Server Edition, 97,887 MiB each, cc 12.0, 188 SMs, driver 595.91.07; 64 vCPU; 50 GB pod disk (≈25 GB free).
- **Memory target:** peak `torch.cuda.max_memory_allocated` ≤ 85% of 97,887 MiB per rank. It is measured at the **densest** states (late-game BC policy, maximum actors and tokens ≈ 720), not early sparse states. It covers rollout, update, teacher precompute and cache, evaluation, and compile/autotune transients. Per-rank shapes (128 envs, spm 8) equal Isaiah's 5090 per-rank shapes; our tokens are about 2× Orbit's (~720 vs ~337), so the expected peak is well under 96 GB. The 6.1 smoke must confirm it. If it doesn't fit, move to spm 4 / accumulation 2 (the same product, I1) before touching anything else, and never change the global batch.
- **Compiler limit:** the L6 bound is independent of GPU type; rows × tokens × inner dim < 2³¹ on every compiled GEMM (guard and chunking in 2.1).
- **Throughput:** the RTX PRO 6000 delivers much less than a B200, so wall-clock is longer. The learning trajectory is unchanged, because the schedule is defined in optimizer steps and env steps. Report measured SPS and an ETA for the planned env-step budget, and never shrink the workload to hit a time target (throughput Decision).
- **CPU:** native env threads per rank × 2 ranks + dataloader/compile workers ≤ 64 vCPU. Measure native step time at 128 envs/rank and pick `native_threads` from the measurement.
- **FlashAttention on cc 12.0:** Isaiah's 4× 5090 recipe forces FlashAttention with the same lock (`flash-attn 2.8.3`, torch 2.9.0), so it's expected to work. The 6.1 smoke confirms the FlashAttention path actually executes.
- **Disk:** checkpoints are small (~6–10M params). Nsight traces and the Inductor cache are large, so traces go to the pod volume with a custody manifest, and old pod run directories are removed only with the owner's OK.

### Run statement (before every pod run)

`ops/rebuild-2026-09-29/run-statements/<name>.md` must contain:
- the question or mechanism tested;
- exact inputs: config, source commit, checkpoint hashes;
- the observation that would discriminate between hypotheses;
- the stopping condition and budget (wall time and $);
- where the artifacts land.

## How we use the reference branch

Read reference files with `git show kg/reference-2026-09-29:<path>` (the tag `kg-reference-2026-09-29` is the same commit). Every component gets exactly one disposition:

- **V — vendor, trimmed and pinned.** Only for code that must stay faithful to an external truth (Kaggle's Python game engine). Retained files keep their reference bytes, checked by a SHA-256 manifest. Whatever training doesn't need stays out. Parity tests prove the behavior.
- **P — port after review.** Small, audited game-semantics code. Read it line by line, rename v2 names to v3 names, add types, and adapt the reference tests as regression tests.
- **R — rebuild, with the reference as oracle.** A new implementation in Isaiah's style. The reference produces recorded outputs (fixtures) that the new code must match or provably preserve.
- **X — reference only.** Read it for lessons and evidence; don't bring it over. Cite it as `reference-branch:kg/reference-2026-09-29/<path>`.

| # | Reference component | Disp. | Why | Proof on the clean base |
|---|---|---|---|---|
| C1 | Rules kernel: `engine_rs/src/lib.rs` (Game/Config/step), `py_random.rs`, `econ_attrib.rs` (economic counters used by rewards), plus the parity tests and fixtures they need | V | Must match Kaggle's Python engine exactly (targets `kaggle-environments==1.32.7`, Python engine SHA in `engine_rs/Cargo.toml` metadata); rewriting rules only adds risk | SHA-256 manifest shows the retained files byte-identical to the reference; the engine unit and replay-parity suites pass; `VENDORED_FROM.md` provenance carries over with a trim note |
| C2 | `engine_rs/src/native_agents/` (22 scripted bots), `joint_matching.rs`, `policy_rows.rs`, `src/bin/*`, `examples/*`, bot fixtures | X now; a few bots V in Phase 7 | Training doesn't need them; bots matter only as evaluation opponents | Phase 7 imports the selected bots with their own manifest |
| C3 | `engine_rs/src/myolie_sampler.rs`: native conditional grammar (masks, finite-state transitions, canonical decoding) | P, renamed to a v3 `grammar` module | Game I/O semantics, already proven by the coupled-Gumbel HIRE enumeration | Port its enumeration and replay tests; decode recorded reference programs identically |
| C4 | `engine_rs/src/myolie_features.rs` (flat v2 "observation-v5" vector) and the reference model's magic-offset slicing (`features[:, 1027:3827]` …) | R | Isaiah feeds named per-entity tensors into stems. Hard-coded offsets are fragile, and the reference review found the flat encoding loses inventory insertion order and some public configuration facts | Information-preservation oracle: on recorded states, every reference feature value is reconstructible from the new tensors, and the lost facts are now present |
| C5 | `engine_rs/src/ffi.rs` (48 C-ABI functions), `engine_rs/src/training.rs`, `src/kaggriculture.rs` (PyO3) | R, in Isaiah's `src/rl` PyO3 style | Training needs only reset/step/observe/mask/reward/auto-reset; the C ABI served v2's ctypes path and bots | Lifecycle tests: 719-step termination and auto-reset, seat isolation, rollback on error, disjoint rank seed streams. Seeded trajectories equal the reference `TrainingBatch` on rewards, dones and banks |
| C6 | `python/owl/kaggriculture/actor_codec.py` (554 lines, a copied demo codec with dynamic typing) | R | Duplicates the grammar in Python; the Rust grammar (C3) becomes the single source of truth | The reference codec is the oracle on recorded programs |
| C7 | `python/owl/kaggriculture/gpu_sampling_grammar.py` (device mask tables) | P | Model-side sampling support with sound validation | Port its tests |
| C8 | `python/owl/kaggriculture/rewards.py` | P | Audited formulas; `terminal_scale` keeps returns in [-1, 1] | Port its tests; add the return-bound test |
| C9 | `python/owl/kaggriculture/{types,env,native_bridge}.py`, `python/owl/game.py` | R | The new observation contract; a thin env shaped like Isaiah's `VectorizedEnv` | New tests plus trajectory equivalence (C5) |
| C10 | Trainer edits in the reference (`ppo.py`, `run_ppo.py`, `train/config.py`, `distributed.py`, `logging.py`, `utils.py`, generic typing churn) | R: reapply deliberately, one tested change per lesson | The reference diff mixes semantics with typing churn; only the needed seams come back | Isaiah's suites plus a named test per change |
| C11 | `python/owl/model/kaggriculture.py` | R on Isaiah's topology; the grammar heads are redesigned from the reference's batched-head design | Owner's topology directive | Topology test against Isaiah's 6m model, plus the reference's sampling/replay density tests |
| C12 | BC: `ops/bc-bootstrap-2026-09-29/{select_replays,prepare,train_bc,evaluate_bc}.py` | P for replay selection and admission; R for `prepare` (new observation) and `train_bc` (new model) | The data source and selection were audited; the observation and model are new | Same episode selection seeds, split identity and admission counts as the reference receipt |
| C13 | `scripts/benchmark_kaggriculture.py` | P (Phase 6) | Sound harness: denominators, hashes, telemetry | Port its tests |
| C14 | `configs/kaggriculture*.yaml`, `configs/model/kaggriculture.yaml` | R from Isaiah's `scaling_6m` plus his multi-GPU rule | Alignment Decision | Workload test against `configs/scaling_6m.yaml` |
| C15 | Build/infra: Cargo workspace member, `pyproject.toml` reference extra with the `kaggle-environments` pin, `justfile`, `Dockerfile*` | P, minimal, via `cargo add` / `uv add` | Needed seams only | Isaiah's suites; Orbit fixture parity under 1.32.7 |
| C16 | Docs: `kaggriculture-model.md`, `coordination-system-design.md`, `upstream-orbit-wars.md` | R (model docs), X (U/A design, README copy) | Written for the rebuilt design | `just docs-fresh` |
| C17 | `ops/*` receipts, W&B runs, BC checkpoints on the pod | X | Evidence and data custody | Cited as `reference-branch:` |

## Lessons from the reference (requirements)

| # | Lesson (reference evidence) | Where it lands |
|---|---|---|
| L1 | Evaluation must decide wins by raw final banks, not shaped return (a .8 economic cap could invert the real winner) | 3.2 |
| L2 | Artificial truncation keeps the real economic reward earned on that transition and adds the critic bootstrap | 3.2 |
| L3 | Rank seed streams use `seed+rank` with stride `world_size` on every reset (a `rank*n_envs` offset collided) | 1.4 |
| L4 | Vendoring the engine turns on `serde_json/arbitrary_precision` for the whole crate, which breaks a test-only float decode in Isaiah's `src/rules_engine/generation.rs` fixtures | 1.1 |
| L5 | A 4,096-game rank-0 evaluation exceeded the 600 s NCCL broadcast; Isaiah's cadence (128 envs/rank) avoids it, and idle GPUs during evaluation are not a hang | 3.3, 6.x |
| L6 | **Resolved (Task 0.2):** the CUDA illegal memory access is a Torch 2.9 Inductor max-autotune GEMM template overflowing 32-bit offsets above 2^21 rows, which silently corrupted rollout activations before faulting (cookbook `references/compiled-gemm-template-overflows-above-2-21-rows`) | 2.1 guard, 3.4 cadence, 3.6 alarm |
| L7 | W&B marks crashed runs "finished", and cleanup delays the traceback | 0.3 |
| L8 | Self-play collapsed to zero banks through market behavior; BC bootstraps from public replays | Phase 5 |
| L9 | BC overfit after about 14.6k updates (held-out NLL 0.761 → 1.954); select by held-out NLL | 5.2 |
| L10 | The 4096 / spm 1 / accumulation 2 cadence ran 2,048 optimizer steps per iteration against Isaiah's 16 | 3.4 |
| L11 | Muon exclusions and FlashAttention differed from Isaiah; both follow automatically from his classes and preset | 2.1, 3.4 |
| L12 | Every evaluation reused one seed | 3.3 |
| L13 | The flat features lost inventory insertion order and some public configuration facts | 1.3 |
| L14 | Heavy local runs overloaded the owner's Mac | Global Constraints |
| L15 | Component speed ≠ end-to-end speed; the async-GPU error surfaced far from its cause | 6.x run statements |

## Isaiah's principles (from his `AGENTS.md`, `README.md`, config headers and `docs/model-architecture.md` at `32b3ec9`)

| # | Principle | Kaggriculture form | Tasks |
|---|---|---|---|
| I0 | Stateless transformer: one encode per current observation, no hidden state, named token sequence, `evaluate_actions` returns log-probs, entropy and values from one encode, stateless teachers | Same | 2.x, 4.x |
| I0b | **Same layer topology:** `ObservationInputStem` stems (hidden `embed_dim·mlp_ratio`); separate per-role token parameters (player, board scratch, actor plan, per-player critic value); `TransformerBlock` trunk and final LayerNorm; `OutputProjectionMLP` critic head per critic-value token; `3D → D` actor input projection over [entity, player, plan] | Same classes. Categorical fields enter the stems as one-hot channels; only channel widths and the grammar action heads are game-specific | 2.1–2.3 |
| I1 | 16 optimizer steps per iteration; `spm × grad_accum` = 16; the split comes from a memory smoke ("grad_accum only as needed") | Start at spm 8 / accum 1 on 2 ranks; the smoke may move to 4 / 2 | 3.4, 6.1 |
| I2 | The same global config on any hardware: divide per-rank `n_envs` and spm by world size | 2 × 128, 4 × 64 envs | 3.4 |
| I3 | The LR schedule is defined over env steps; a batch change rescales warmup and decay | Any forced deviation rescales them in the YAML | 3.4, 6.1 |
| I4 | Model ladder: GELU, `mlp_ratio 2.0`, `head_dim 32`, scale width×depth; 6m = 256×6 (5,679,118 params in Orbit) | Our trunk sits on the ladder | 2.4 |
| I5 | Winner-distribution critic (value 2p − 1, gamma 1), value distillation on winner probabilities | Per-seat (self, opponent) softmax from the seat's own observation | 2.2 |
| I6 | Last-best teacher: KL(teacher ‖ student), small coefficients, the teacher shares the student's architecture and is refreshed in place, cached path bit-for-bit equal to the combined path | Per-slot categorical KL over the grammar heads | Phase 4 |
| I7 | Eval against last best, 70% promotion, separate current/best checkpoints, 8 replays exported | Raw-bank winners; Kaggriculture replay format | 3.3, 7.3 |
| I8 | Throughput defaults: compiled GAE/loss, trunk `max-autotune-no-cudagraphs`, BF16, TF32, forced FlashAttention | Verified active in Phase 6 | 3.4, 6.x |
| I9 | Parity-first engine; `docs/rules-parity-coverage.md` is the source of truth | Kaggriculture parity documented the same way | 1.1, 7.5 |
| I10 | Docs are contracts: read the mapped doc first, update it with the change, `just docs-fresh` | Same | all |
| I11 | Refactor rather than add compatibility shims; fail fast; strict persisted schemas; `isinstance` narrowing instead of `getattr` | Same | all |
| I12 | `just py-prepare` / `rs-prepare` / `prepare`, `uv add` / `cargo add`, tracked lockfiles, PR checklist, merge commits | Same | all |

## Working with Codex

- **Worktrees:** Codex works in its own worktree and branch, so the two agents never share a checkout. Before launch, Claude runs `uv sync` and `cargo fetch` there, so Codex needs no network for dependencies.

```bash
git worktree add ../kg-v3-codex -b kg/rebuild-codex kg/isaiah-gap-closure
(cd ../kg-v3-codex && uv sync --extra reference && cargo fetch)
codex exec -C ../kg-v3-codex -s workspace-write \
  -o ops/rebuild-2026-09-29/codex/<task>.md "<task prompt>"
```

- **Task prompt contents:** the plan path and task id; the Global Constraints (quote the Mac and pod rules); the task's reference files with their dispositions; required tests; and "report changed files, commands run with their results, and open questions".
- **Contract first:** Task 0.1 fixes the observation, action and env contract before the streams diverge. Changing it later needs both agents to agree, recorded in the contract doc.
- **Streams:**

| Stream | Owner | Tasks |
|---|---|---|
| Engine and native env | Codex | 0.3, 1.1–1.5, 5.1, 7.1, 7.3, 7.5 |
| Model, trainer, teacher, configs | Claude | 0.1, 2.x, 3.x, Phase 4, 5.2, 7.2, 7.4, Phase 8 |
| Pod operation | Claude (Codex writes scripts and reviews receipts) | 0.2, 5.2, Phase 6 |

- **Cross-review:** every task is reviewed by the other agent before it merges into `kg/isaiah-gap-closure` with a regular merge commit. Claude reviews with superpowers:requesting-code-review; Codex reviews with `codex exec -C <worktree> -s read-only "Review <base>..<head> against rebuild plan task <id>: correctness, contract, tests, Isaiah topology"`. Findings are fixed before the merge.
- **Task briefs:** each subsystem's code-level steps depend on reading its reference code. So before coding, the implementer writes `ops/rebuild-2026-09-29/briefs/<task>.md` (a TDD plan in the superpowers:writing-plans format: failing tests, implementation, commands, commit), and the other agent reviews it. The tasks below fix scope, interfaces, reference inputs and acceptance.

## Execution order

| Phase | Tasks | Owner | Depends on |
|---|---|---|---|
| 0 Contract and diagnostics | 0.1 contract, 0.2 CUDA reproduction on the reference, 0.3 W&B status | Claude / Claude+Codex / Codex | — |
| 1 Engine and native env | 1.1–1.5 | Codex | 0.1 |
| 2 Model on Isaiah's topology | 2.1–2.5 | Claude | 0.1 (synthetic tensors until 1.5) |
| 3 Trainer integration | 3.1–3.5 | Claude | 1.5, 2.3 |
| 4 Teacher distillation | 4.1–4.4 | Claude | 3.x |
| 5 BC (required) | 5.1 data, 5.2 train | Codex / Claude | 1.5, 2.x, 3.5 |
| 6 GPU verification | 6.1–6.4 | Claude | 4, 5.2 |
| 7 Evaluation and packaging | 7.1–7.5 | mixed | 6.2 |
| 8 Docs and closeout | 8.1–8.2 | Claude | all |

---

## Phase 0 — Contract and diagnostics

### Task 0.1: Observation, action and environment contract

**Owner:** Claude writes it; Codex reviews. **Files:** create `docs/kaggriculture-contract.md` and add a Kaggriculture section to `docs/rl-api-specs.md`. **Reference inputs (read, don't copy):** `engine_rs/src/myolie_features.rs` (which legal public and own-private facts exist, and their normalizations), `python/owl/model/kaggriculture.py` `_encode` (how the flat vector was grouped), `python/owl/kaggriculture/types.py`, `engine_rs/src/myolie_sampler.rs` (grammar slots and widths), `engine_rs/src/training.rs` (lifecycle), and `ops/coordination-design-2026-09-29/` (the missing-facts finding).

- [x] Define `KaggricultureObsBatch` as named per-entity tensors, following Isaiah's `ObsBatch`. Candidate groups from the reference: tiles, own/rival actors, storage, shops, products, globals. Give each group its float channels and its categorical index fields (tile kind, cell, actor index, role, product index) as integer tensors, plus entity masks, `still_playing`, and the action mask. State each channel's meaning and normalization.
- [x] Information audit table: every reference feature range (e.g. `features[1027:3827]`, the tile maintenance block) maps to a new field, or is recorded as intentionally dropped with a reason. Add the facts L13 found missing.
- [x] Action contract: 252 frames × 12 slots, the slot widths, the STOP, EMPTY, NONE and HIRE semantics, and the grammar mask API taken from C3.
- [x] Env contract: `reset`/`step` signatures, caller-owned pinned buffers, auto-reset, terminal metrics (raw banks per seat), reward configuration, and seed streams (L3).
- [x] Acceptance: Codex's review is recorded in the doc; both streams code against this file.

### Task 0.2: Reproduce the CUDA illegal memory access on the reference branch (pod) — DONE, root cause in `results.md`

**Owner:** Codex writes `ops/rebuild-2026-09-29/cuda_repro.py`; Claude runs it. It uses the reference branch checkout on the pod and the BC best checkpoint (SHA-256 `ffd7d9e4…`, identical to its pre-PPO copy; confirm the path on the pod before running).

- [x] Run statement. Question: which hypothesis holds — H1, a dense-state index overflow in compiled kernels; H2, an async copy from reused native buffers; or H3, some other kernel?
- [x] Reproduce and localize: a `CUDA_LAUNCH_BLOCKING=1` run of the crashed run's own **compiled** config (fault after 3 iterations in an Inductor Triton GEMM template), then a synthetic compiled-versus-eager probe (`int32_probe.py`) across the 2²¹-row boundary. The planned eager-first order, synchronous-copy run and `compute-sanitizer` step were not needed once the blocking run named the kernel. Budget: ≤ 2 GPU-hours.
- [x] Record the root cause (or "not reproduced", with the conditions tried) in `ops/rebuild-2026-09-29/results.md`, and turn it into a design requirement for 1.4 (buffers) and 2.3 (index validation).

### Task 0.3: Report failed runs as failed

**Owner:** Codex; Claude reviews. `logging.py`, `run_ppo.py` and `distributed.py` are still Isaiah's files on the clean base, so the previous plan's Task 1.1 applies unchanged. It covers the tests and the code: `MetricLogger.close(*, exit_code: int = 0)`, `WandbLogger.close` → `finish(exit_code=...)`, `run_ppo._logger_session`, and a rank-tagged traceback printed before `destroy_process_group`. Read it with `git show kg/reference-2026-09-29:ops/gap-closure-2026-09-29/plan.md`.

- [x] Follow that task's five steps exactly; Isaiah's suites must pass.

## Phase 1 — Engine and native environment (Codex)

### Task 1.1: Vendor the trimmed rules kernel (C1)

- [x] Add `engine_rs/` as a workspace crate with `lib.rs`, `py_random.rs` and `econ_attrib.rs` kept byte-identical to the reference. Remove `mod` lines only for excluded modules (C2, C4's `myolie_features`, C5's `ffi`), and record every edit to a retained file in `engine_rs/VENDORED_FROM.md`.
- [x] Bring only the fixtures the retained tests need; list them with their hashes.
- [x] Write `engine_rs/TRIM_MANIFEST.json`: retained files with reference SHA-256, excluded files with reasons.
- [x] L4: handle the `arbitrary_precision` feature interaction. Either keep the engine out of the root crate's feature graph, or apply the reference's test-only `fixture_float` decode in `src/rules_engine/generation.rs`, and record which.
- [x] Acceptance: `cargo test --manifest-path engine_rs/Cargo.toml --lib --locked` passes, `cargo test` passes (Isaiah's 155), and the manifest check passes.

### Task 1.2: Port the grammar kernel (C3)

- [x] `src/kaggriculture/grammar.rs`, reviewed C3 port from `myolie_sampler.rs`: typed plan/cursor, direct factored tables, checked i64 canonical decode and strict encode. Drop the binary runtime graph. The authored standalone engine test temporarily includes this one source; retained kernel bytes remain pinned. See the reviewed `briefs/1.2.md`.
- [x] At the first production root → engine edge (1.3/1.4), move kernel acceptance/replay-state tests into a root integration test, delete `engine_rs/tests/grammar_kernel.rs` and its allowlist/manifest registration, and reopen L4 with the test-only float-fixture repair. Keep one integration route. Done in Task 1.3 (verification round 1 fix): `src/kaggriculture/grammar_kernel_tests.rs`; receipts in `1.3/r1-fixes/`.
- [x] Port the coupled-Gumbel HIRE enumeration (3 positions, 8 kinds, budgets 0/1/2/3/10) and the replay tests.
- [x] Oracle: 320 recorded reference programs (256 synthetic, 64 real), including 64 dense 241-actor programs, agree exactly. Independent recording, strict codec, dense execution and replay-state evidence: `1.2/results.md`.

### Task 1.3: Structured observation encoder (C4)

- [x] `src/kaggriculture/observe.rs` (v3-owned) writes the 0.1 contract's tensors into caller-owned buffers for both seats, never exposing the rival's private state.
- [x] Oracle: record reference `myolie_features` outputs on ≥ 500 states (early, mid and late game; dense actors) into a fixture. For every contract mapping, reconstruct the reference values from the new tensors and assert equality within float tolerance. Also assert that the L13 facts are present.

### Task 1.4: Native environment lifecycle (C5)

- [ ] A PyO3 `KaggricultureEnv` in Isaiah's `src/rl` style: batched reset/step, auto-reset, rewards via 1.5's reward config, lazily built terminal metrics, rayon threads, transactional step with rollback on error, and seed streams per L3.
- [ ] From 0.2 (L6): no host buffer the trainer may still be copying from is overwritten by the next step. Test that the step-t buffers stay intact after step t+1, or document the double-buffer and event contract.
- [ ] Oracle: seeded fixed-action trajectories (≥ 16 games to termination) match the reference `TrainingBatch` on rewards, dones and final banks.

### Task 1.5: Python adapter and game seam

- [ ] `python/owl/kaggriculture/types.py`, `env.py` (C9), `rewards.py` (C8, ported with tests), `codec.py` (C6, rebuilt over the Rust grammar with the reference codec as oracle), and `gpu_grammar.py` (C7, ported).
- [ ] `python/owl/game.py` factory seam (C10) with Isaiah's Orbit tests unchanged.
- [ ] Acceptance: the Kaggriculture env tests pass, and Isaiah's suites pass.

## Phase 2 — Model on Isaiah's topology (Claude; synthetic tensors until 1.5 lands)

### Task 2.1: Encoder with Isaiah's classes

- [x] `python/owl/model/kaggriculture.py`: one `ObservationInputStem` per entity group, fed [float channels ‖ one-hot categorical fields]. Per-role parameters `player_tokens[2,D]` (self, opponent), `board_tokens[n_scratch,D]`, `actor_plan_tokens[1,D]` (only this seat acts in its observation), `critic_value_tokens[2,D]`. `TransformerBlock` × depth and `final_norm`. Named encoded fields, never positional slices.
- [x] Topology test: build Isaiah's 6m model (`configs/scaling_6m.yaml`) and ours. The shared roles must use identical classes (`ObservationInputStem`, `TransformerBlock`, `LayerNorm`, `OutputProjectionMLP`, the `3D→D` `Linear`); token parameters must be separate per role; `nn.Embedding` may appear only under `actor.`.
- [x] Stateless tests: outputs depend only on the current observation; hidden-state keys are rejected.
- [x] L6 guard: the compiled trunk fails fast (or chunks) when `rows × tokens × max_inner_dim ≥ 2^31`; boundary test with a stub trunk.
- [x] Muon: `get_input_layers` returns each stem's `.input` and the token parameters; `get_output_layers` returns `critic_head.out` and the head `.out`s. Isaiah's rule is then satisfied by construction (L11).

### Task 2.2: Critic

- [x] `critic_head = OutputProjectionMLP(cfg, 1)` applied per critic-value token → logits `[B, 2]` → softmax → value = 2p(self) − 1, with log-probabilities exposed for value distillation. The config requires `value_mode=win_loss` and gamma 1.

### Task 2.3: Grammar action heads

- [x] `actor_input_proj: Linear(3D, D)` over [entity, player, plan]. An `actor` module holds the game-specific heads, redesigned from the reference's batched heads: `OutputProjectionMLP` per slot, conditioning embeddings, market positions, and the HIRE-capacity coupled Gumbel sampler.
- [x] From 0.2 (L6): every index the heads use is validated on device before gather (fail fast, never clamp).
- [x] Port the reference density tests: sampling log-prob equals evaluation log-prob; Gumbel frequencies match categorical probabilities; finite-difference policy gradient; full 241-actor and full-market capacity; the private-perspective isolation test.

### Task 2.4: Model size on Isaiah's ladder — owner answer needed

- [x] The reference test `test_default_model_obeys_owner_parameter_budget` claims a 6–10M owner budget with no recorded source. Isaiah's 6m is 5.68M. Ask the owner once: if there's no budget, use 256×6 with `mlp_ratio` 2; if the budget applies, use the nearest on-ladder point inside it. Record the answer with the owner's quote.

### Task 2.5: Model docs

- [x] `docs/kaggriculture-model.md`: a conformance table against Isaiah's `docs/model-architecture.md` (same / game form / deviation), and a parameter count.

## Phase 3 — Trainer integration (Claude; Codex reviews). One tested change per item on Isaiah's files.

### Task 3.1: Game seam in the trainer

- [ ] Rollout storage and observation/action mapping for Kaggriculture batches. Refactor Isaiah's helpers into schema-generic ones (I11); don't wrap them in shims. Isaiah's suites must pass.

### Task 3.2: Game semantics

- [x] L1: raw-bank evaluation outcome. L2: truncation keeps the economic reward. Also joint per-player clipping and the value-mode guards. One named test each.

### Task 3.3: Evaluation

- [x] L12: `_evaluation_seed(base_seed, env_steps)` (reproducible and different per evaluation; check the native seed type's range). Isaiah's default evaluation count. Promotion telemetry (`eval/promoted`, `eval/promotion_threshold`, `eval/games`). Orbit is unaffected.

### Task 3.4: Configs from Isaiah's recipe
- [x] (From the GEMM-limit audit) startup workload assertion: at config load, compute rows per forward for rollout (n_envs × 2), minibatch (spm × horizon × 2), teacher chunk (min(teacher_spm, n_envs) × horizon × 2), eval and BC batch, and assert each is bounded by the model's trunk and head chunking limits; record the headroom in the run log.

- [x] `configs/kaggriculture_2rank.yaml` (128 envs/rank, spm 8, accum 1), `configs/kaggriculture_4rank.yaml` (64/4/1), `target_kl: null`, and the `scaling_6m` optimizer, scheduler, PPO coefficients, compile settings and 20M checkpoint cadence. Economic shaping 0.2 is the owner's choice. `configs/model/kaggriculture_gpu.yaml` forces FlashAttention; a CPU preset and `configs/kaggriculture.yaml` cover local tests.
- [x] Workload test: global envs, optimizer steps per iteration, global segments per step and transitions per iteration all equal `scaling_6m`; the optimizer config is equal too. (The previous plan's Task 2.1 has the test code.)

### Task 3.6: First-minibatch log-ratio alarm (L6)

- [x] In `PPOTrainer._update`, before the first optimizer step of each update, compute the mean log-ratio of the first minibatch. If |mean| > 0.05 nats (the plan's value; GPU/BF16 replay noise is unmeasured, so this threshold is not yet qualified), raise with the rollout batch shape and token count. Test with a model whose sampling and replay paths are deliberately made to disagree.

### Task 3.5: Bounded local functional check

- [ ] Tiny model, 2 envs, 2 updates, CPU: finite losses, a checkpoint written, and an evaluation plus promotion branch exercised through `test_last_best`-style tests. This is the only local "run".

## Phase 4 — Teacher distillation (Claude)

The previous plan's Tasks 3.2–3.5 carry over, adjusted to this model. They cover: per-slot policy distributions with `slot_kl` (KL(teacher ‖ student) over masked categoricals); the `TeacherTargets` protocol refactor of Isaiah's cached targets (`.index` / `.concat`); Kaggriculture teacher targets; cached path bit-for-bit equal to the combined path; value distillation on the per-seat winner distribution (mean over seats); last-best refresh and resume tests; and the `scaling_6m` teacher coefficients (0.005 / 0.005, `teacher_segments_per_minibatch` 128).

- [ ] 4.1 distributions and KL, 4.2 targets and cache, 4.3 model methods and trainer wiring, 4.4 configs. (4.1–4.3 approved, not merged: 8fde43c; staged merge a424d8c on kg/merge-teacher)

## Phase 5 — BC (required)

### Task 5.1: Data preparation for the new observation (Codex)

- [ ] Port `select_replays.py` (P): the same source volume, seeds, per-day counts and episode-identity split. Rebuild `prepare.py` (R) on the explicit-state native constructor, emitting the 0.1 contract's tensors. Acceptance: episode IDs, split and admission and rejection counts match the reference receipt (158,772 admitted / 22,416 rejected turns for the 252-episode slice), or the differences are explained.

### Task 5.2: BC training on the pod (Claude)

- [ ] Rebuild `train_bc.py` (R) on the new model, reusing Isaiah's optimizer factory, BF16 and compile. Two-rank RTX PRO 6000. Keep the best checkpoint by held-out NLL and stop on sustained held-out degradation (L9). Run statement first. Record the checkpoint SHA-256, the NLL curve and the stopping step.

## Phase 6 — GPU verification (pod, RTX PRO 6000; Claude operates, Codex reviews receipts)

- [ ] **6.0 FlashAttention on the pod (blocking for 6.1–6.4):** (approved, not merged: 78df33c) the GEMM-limit probe found that the pod venv has **no `flash-attn` package**, and the pod's run config had `force_flash_attn: false` (`results.md`, "GEMM limits at our shapes"). Install or build flash-attn 2.8.3 on the pod with `uv sync --extra flash-attn`, and set `force_flash_attn: true`. Verify that the real FlashAttention varlen path runs (import, kernel in use, compiled vs eager on the packed trunk) before any qualification or throughput claim.
- [ ] **6.1 Memory smoke, 2 ranks:** one full iteration with the teacher on and a forced evaluation at dense BC positions. Record peak memory per phase against the ≤ 85% target, teacher cache bytes, native step time and the chosen `native_threads`, confirm the FlashAttention path ran, and record the spm/accum split decision (I1/I3; see "Resource fit").
- [ ] **6.2 Complete-work run, 2 ranks, from the BC best:** 30 min bounded. Report game and learner-seat SPS over complete iterations, 16 optimizer steps per iteration, teacher telemetry, W&B status and whether the L6 fault is absent. Optionally an Nsight capture of one post-warmup iteration.
- [ ] **6.3 Four ranks:** the same denominators for 15 min.
- [ ] **6.4 Orbit end-to-end:** `configs/scaling_6m.yaml`, one GPU, 2 iterations, with a forced evaluation.

## Phase 7 — Evaluation and packaging

- [ ] **7.1 (Codex) Opponents:** import 3–5 native bots, chosen for different play styles (e.g. v43, farm2945, cha22, starter), as V with their own manifest and parity checks.
- [ ] **7.2 (Claude) Panel script:** win rate, bank margin, seeds, both seats, denominators, legality, completion and runtime, labelled selection vs held-out.
- [ ] **7.3 (Codex) Replay export:** Kaggle episode format, round-tripped through the native engine from the seed header; 8 replays per evaluation.
- [ ] **7.4 (Claude) Kaggle agent packaging:** validated on one bounded local episode.
- [ ] **7.5 (Codex) Parity docs:** `docs/rules-parity-coverage.md` gains a Kaggriculture section stating what is tested and what isn't.

## Phase 8 — Docs and closeout

- [ ] **8.1** Rewrite the cookbook References for the rebuilt tree; they currently describe the reference branch. Record the deliberate differences (truncation reward, raw-bank winners, per-evaluation seeds) in `docs/rl-api-specs.md`.
- [ ] **8.2** Results inventory, `just prepare`, PR checklist, merge. The owner decides any push, force-push or submission.

## Open question for the owner

- Model size (Task 2.4): did you set a 6–10M parameter budget?

## Self-review

- **Owner directives covered:** clean base and reference branch (C1–C17, dispositions); same layer topology (I0b, 2.1–2.3); extra BC (Phase 5); 2/4-rank RTX PRO 6000 (Phase 6); Codex (Working with Codex, streams); Isaiah's principles and stateless model (I0–I12).
- **No blind copying:** only C1 (the rules kernel) keeps reference bytes, and it's trimmed and hash-checked. The grammar, rewards, mask tables and benchmark are reviewed ports. The observation, env, codec, model, trainer seams, configs and BC training are rebuilt, with the reference as oracle.
- **Every reference lesson (L1–L15) maps to a task.**
