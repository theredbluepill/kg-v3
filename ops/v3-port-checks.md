# Kaggriculture v3 port receipt

Date: 2026-09-28. This is a source-scoped implementation receipt, not a competitive
training result or a GPU throughput qualification.

## Source and adaptation boundary

- Imported Isaiah Pressman's starter history at
  `32b3ec900ad406eedd965f53a1a0f4490d31c589`. `origin` is
  `git@github.com:theredbluepill/kg-v3.git`, `remote.pushDefault=origin`; `upstream`
  retains the starter. No commit or remote push was made by this work.
- V2 remained read-only. `engine_rs/V3_IMPORT.json` records source checkout/commit,
  dirty-source file hashes, vendored fixtures, adaptations and verification.
- `scripts/run_ppo.py` and `owl.train.ppo.PPOTrainer` remain canonical. The new
  game boundary uses PyO3 caller-owned arrays, GIL release and bounded native
  Rayon workers. There is no active ctypes or v2 trainer path.
- New stems/heads and within-turn decoder use Isaiah's transformer/attention
  components. No v2 model implementation is retained. V2 token grammar and reward
  semantics are references. Default size is **8,294,450 parameters**, including
  5,528,320 in the starter transformer trunk. Each legal private seat observation
  is encoded separately; no inter-turn state or opponent identity enters policy.
- New observation/action types preserve the current v3 native/PPO buffer contract:
  two seats, 8176 features per seat, up to 241 actors, 252 frames and 12 conditional
  token slots. Shared PPO preserves the exact game schema through rollouts,
  segmentation, transfers, updates and checkpointing.
- Three review repairs preserve raw-bank evaluation outcomes when shaping changes
  returns, current economic reward at truncation, and disjoint rank seed streams
  across automatic resets. Categorical sampling/evaluation shares the same
  grammar and density. Model hot-path changes preserve zero gradients for unused
  forced heads so distributed reductions remain defined.
- Terminal W/L/D, own-minus-opponent bank/3000 and win-share modes are explicit.
  Optional economic potential shaping defaults off, retains source caps, requires
  gamma 1 and matching critic range. It is not an inherited optimal recipe.
- Training-only installs omit the reference-game dependency tree. The optional
  `reference` extra pins Kaggle environments 1.32.7 and supplies fixture tooling;
  `just` tests and fixture regeneration request it. Lockfiles were changed by
  package-manager commands. Native build cache keys include vendored Rust source.

## Previous-best lifecycle

The cookbook reviewer compared AST function bodies to imported starter HEAD:
`_run_training_loop`, `_resolve_resume_launch`, `_next_periodic_checkpoint_step`
and `_validate_last_best_run_id` are unchanged. Initial incumbent creation,
**70% promotion**, numbered/incumbent/final checkpoint roles, run-ID pairing,
optimizer restoration and resume cadence remain in the shared lifecycle.

`tests/kaggriculture/test_last_best.py` executes real native two-seat evaluation
and PPO updates. Three tests cover .699/.7 retention/promotion, incumbent bytes,
checkpoint pairing and resume cadence. It starts a logical counter at 996 to
cross a valid 1000-step checkpoint boundary after four actual transitions; the
selection scalar is controlled only after actual evaluation to cover branches.
The new configs enable a 100,000-global-transition checkpoint cadence. Evaluation
keeps the starter's `n_envs` game count; a small development denominator does not
qualify generality or competitive selection reliability.

## Executed checks

Local host: Apple M5 / macOS arm64, CPython 3.12.13, Torch 2.9.0. Native extension
was built as a release PyO3 wheel with maturin and stable Rust 1.94.1. The
repository's pinned nightly installation was initially partial and missing its
rustc-driver library. Rustup repaired it to `1.97.0-nightly (e9e32aca5 2026-04-17)`.
The complete `rs-prepare` then also passed on the original pinned toolchain,
including all 155 tests, Clippy, formatting and doc freshness. Both stable and
pinned checks are retained; Linux/CUDA builds are a separate boundary.
The installer's subsequent optional Rustup self-update failed to set permissions
on a missing `rustup-init` file. This did not undo the toolchain repair: final
`rustc`, `cargo` and `rustup` version commands all succeeded. The final installer
log is retained separately from its earlier snapshot.

| Check | Actual result |
| --- | --- |
| `just py-prepare` equivalents via `uvx --from rust-just just py-prepare` | **772 passed, 3 skipped**; Ruff, Python 3.11 syntax, strict mypy (58 source files), doc freshness passed |
| `just rs-prepare`, original pinned nightly and stable toolchains | **155 passed, 0 failed, 2 existing ignored**; fmt, all-target Clippy with warnings denied, doc freshness passed |
| Vendored engine Rust suite | **120 passed**, including native serial/parallel comparisons and source fixtures |
| Focused model tests | **19 passed**, including density, legality, isolation, 241-actor capacity, categorical support/frequencies, projection parity and finite gradients for all parameters on active/inactive inputs |
| Focused native Python tests | **20 passed**, including buffer lifetime, transactional errors, economic shaping and rank seed streams |
| Real single-process CLI | One actual native→model→PPO update and checkpoint, 4 global transitions, tiny diagnostic model |
| Real two-process full-model CLI | **3 PPO updates, 96 global transitions, 192 seat turns**, full 8.29M model, FP32, Gloo, 8 environments/rank and horizon 2; finite logged metrics and final checkpoint |
| Benchmark guard | CUDA benchmark `--help` works; execution without 2 CUDA devices rejects before launch; summary denominator/warmup test passes |
| Locked project environment | Native release build and default `uv sync --locked` completed; **232 Kaggriculture/shared-trainer tests passed**, strict mypy passed all 58 source files, real command-line PPO update/checkpoint succeeded |
| Repository hygiene | `uv lock --check`, `git diff --check`, secret-pattern filename scan pass; local W&B credential file is mode 0600 and outside git |

The full Python suite used a separate CPython 3.12 environment with the proper
maturin wheel's native extension, current repository Python source and cached
Torch 2.9.0. Subsequently the normal project `.venv` finished syncing its lockfile
and passed the additional 232 tests, all-source mypy and actual PPO CLI check
above. Locked-tool compatibility required naming a grammar validation scalar
separately from its NumPy array and configuring the pytest repository import
path; neither changes game behavior. Exact retained command logs are under
`ops/port-evidence/` with SHA-256 custody. The three Python skips are two CUDA FlashAttention checks and
one unavailable x86 quantized backend check. No relevant failing test was hidden
by skipping missing fixtures: inherited Orbit fixtures were generated, downloaded
replay parity ran, and a test-only serde tagged-number parser was repaired for
feature-unified `arbitrary_precision`; its oracle assertions were preserved.

The full-model distributed diagnostic used the original entrypoint:

```sh
OWL_ALLOW_CPU_DDP=1 GLOO_SOCKET_IFNAME=lo0 python -m torch.distributed.run \
  --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29619 \
  scripts/run_ppo.py configs/kaggriculture_2rank.yaml runs --log-mode debug \
  --max-env-steps 96 -o rl.horizon=2 rl.model_compile=none rl.dtype=float32 \
  env.pin_memory=false rl.checkpoint_freq=null
```

`OWL_ALLOW_CPU_DDP=1` is an explicit diagnostic opt-in. Default distributed
launches still require CUDA; no implicit CPU fallback conceals missing GPUs.
The first macOS standalone rendezvous failed IPv6 name resolution; the explicit
IPv4 rendezvous above succeeded. This is CPU correctness evidence, not CUDA SPS.

## Performance evidence and remaining gate

Paired/interleaved native-adapter CPU checks (eight environments, one private
native thread, five blocks of 800 batch steps including resets) measured:

| Fixed-action workload | Before | After |
| --- | --- | --- |
| PASS, one farmer per seat | 17,957 game steps/s | 19,050 game steps/s |
| Move plus two market orders | 13,875 game steps/s | 14,535 game steps/s |

Ninety paired output checkpoints matched. Reusing action/mask storage and skipping
disabled reward arithmetic explains the scoped adapter change. Native transaction
cloning/staging remains for rollback correctness. These fixed-action measurements
exclude policy inference/PPO and do not cover high actor populations.

The 8.29M model CPU profiler at eight environments found scalar extractions
305→12 at reset and 680→27 at a synthetic 16-actor observation. Evaluation scalar
extractions fell 8→1. Mean forward latency changed only modestly (152.53→149.69 ms
at reset, 189.96→186.38 ms at 16 actors; two warmups/five samples, one CPU thread).
RNG draws differ after mathematically equivalent Gumbel-max sampling, and the
measurements are not a same-actions GPU comparison. Remaining per-frame STOP
synchronization, compact validation/context transfers, grammar misses, long
sequences, compiled cold/warm behavior and collectives require a CUDA timeline.

The owner recommends **two 5090s or two PRO6000s first**, then four/eight-rank
scaling. Live RunPod reads advertised LOW two-PRO6000 capacity in US-NE-1 and then
EUR-IS-1, but both corresponding create calls returned HTTP 400, “There are no
longer any instances available with the requested specifications.” Two 5090s
were unavailable. **No pod was created, no GPU compute started, and existing pods
were untouched.** A final `list_pods` read confirmed only the two pre-existing
EXITED pods (`0yihpugnavlg7e`, `p0wdnd40cbmrej`). No GPU SPS, Nsight trace, BF16/compiled GPU run or scaling result
is claimed.

`scripts/benchmark_kaggriculture.py` provides the bounded CUDA-only next check. It
launches the canonical trainer, records source/config/device identity and complete
loop wall-time SPS after warmup, and separately retains inherited host phase
timers. Its optional `--profile` captures an Nsight Systems timeline, including
startup; profiler overhead must be reported separately. Checkpoint evaluation is
disabled only inside this measurement harness. Normal training keeps it enabled.

## Cookbook and limitations

Every material adaptation maps to the cookbook's starter/history, native I/O,
model/head, shared-PPO and reward References and adopted Decisions. Lifecycle
scripts and Base rendering are tested; actual harness discovery/trust remains
unverified as stated in `ops/cookbook-setup-checks.md`.

This port does not qualify competitive learning, complete future-engine parity,
Kaggle submission packaging, CUDA throughput, FlashAttention or four/eight-rank
scaling. Inherited Orbit serving/benchmark/submission surfaces explicitly reject
Kaggriculture inputs until adapted. Teacher distillation, Orbit replay export and
per-entity PPO clipping are rejected on the initial Kaggriculture path; previous-
best evaluation/promotion remains active independently of distillation.
