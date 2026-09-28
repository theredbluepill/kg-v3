# Kaggriculture v3

V3 adapts [Isaiah Pressman's Orbit Wars starter](https://github.com/IsaiahPressman/kaggle-orbit-wars/tree/32b3ec900ad406eedd965f53a1a0f4490d31c589)
to Kaggriculture. Its history and license are retained. `origin` is
`theredbluepill/kg-v3`; `upstream` is the starter.

The training path remains **Rust/PyO3 caller-owned buffers → shared transformer →
original PPOTrainer → original optimizer/distributed/checkpoint lifecycle**.
The reusable Kaggriculture rules kernel comes from v2 and is adapted to this
boundary. There is one training entrypoint, `scripts/run_ppo.py`.

Start with the [cookbook](cookbook/index.md) and [agent contract](AGENTS.md).
Every adaptation has a cookbook record with source, consequence, checks and gaps.

## Setup

Install Rust, uv and just, then install the pinned toolchain:

```sh
rustup toolchain install nightly-2026-04-18 --component rustfmt
uv sync --locked
uv run maturin develop --release
```

The native extension includes the vendored `engine_rs/` crate. No separate v2
checkout, ctypes loader, training workspace or external snapshot is required.
The optional reference Python game is pinned to `kaggle-environments==1.32.7` to match the
Rust kernel's recorded compatibility target. This pin does not establish complete
parity with every possible game state or future competition-engine version.
Native training does not import that package. Use `uv sync --extra reference` for
reference-engine work; `just py-prepare` and fixture regeneration request this
extra automatically. This keeps JAX/OpenSpiel and other reference-game packages
out of the native training install.

## Training

```sh
uv run python scripts/run_ppo.py configs/kaggriculture.yaml runs --log-mode wandb
```

This starter config uses two environments and FP32 without compilation, with a
64-step rollout and joint turn PPO clipping. These are functional development
settings, not a measured production throughput optimum. W&B uses project `kg-v3`.
Choose the actual batch size and CUDA settings from measurements of complete
iterations, including collection, and follow the cookbook's Nsight Systems rule.
`env.native_threads` controls native environment parallelism; deterministic native
seeds are offset by rank. Existing torchrun/DDP launch and checkpoint/resume
support remains in the shared entrypoint.

The default model has **8,294,450 parameters**. Previous-best evaluation,
the original **70% promotion threshold**, checkpoint naming and resume pairing
remain in the shared starter loop. The Kaggriculture configs checkpoint every
100,000 global game transitions. Evaluation compares raw terminal banks; optional
economic shaping cannot change the game winner used for promotion. Evaluation
currently uses the starter's `env.n_envs` game denominator; the small development
batch is not sufficient evidence of general playing strength.

Start GPU performance qualification on two 5090s or two RTX PRO6000s:

```sh
uv run python scripts/benchmark_kaggriculture.py --ranks 2
uv run python scripts/benchmark_kaggriculture.py --ranks 2 --profile
```

The harness launches the original trainer with `configs/kaggriculture_2rank.yaml`,
excludes three warmup updates, measures five complete updates, and writes source
hashes, config/hardware metadata, logs and an SPS summary under `runs/perf-*`.
Its optional Nsight Systems run is separate evidence with profiler overhead.
It rejects CPU execution. Inspect `observed_game_sps` for complete loop wall time;
the inherited host phase timers need an Nsight timeline for CUDA attribution.
No CUDA SPS result has been obtained yet: two provisioning attempts failed with
no stock after live catalog reads advertised availability. No pod was created.

For a training run, preserving periodic evaluation and previous-best promotion:

```sh
uv run torchrun --standalone --nproc_per_node=2 scripts/run_ppo.py \
  configs/kaggriculture_2rank.yaml runs --log-mode wandb
```

The four-rank config is retained for subsequent scaling checks. Both GPU configs
use BF16 and compiled starter transformer blocks; hardware qualification remains
pending. `OWL_ALLOW_CPU_DDP=1` explicitly enables Gloo for local correctness
diagnostics, without changing the default CUDA requirement for distributed runs.

For a bounded local integration check (not a competitive training result):

```sh
uv run python scripts/run_ppo.py configs/kaggriculture.yaml runs --log-mode debug \
  --max-env-steps 4 -o rl.horizon=2 model.embed_dim=32 model.depth=1 model.n_heads=4
```

Run the game adapter and shared training tests:

```sh
uv run pytest tests/kaggriculture tests/owl/train
```

Rewards default to terminal win/loss/draw. `margin` uses own-minus-opponent bank
divided by 3000 and requires the unbounded `model.value_mode=margin` critic;
`win_share` uses the recorded W/L/D plus bank-share mixture. Optional economic
potential shaping defaults off and requires gamma 1. Exact formulas and bounds
are recorded in the [reward contract](cookbook/references/reward-reuse-preserves-objective-and-critic-semantics.md).

## Adaptation boundary

- Rust writes each seat's legal observation into `[env,2,8176]` feature buffers.
  The model encodes each seat independently; opposing private inventories and
  policy identity never enter its attention sequence.
- The stateless model reuses the starter's transformer blocks and attention.
  Kaggriculture stems, actor pointers and native-grammar action heads replace
  planets, fleets and launch heads on the active path. Action-prefix state resets
  each observation; it is not memory carried between turns.
- Actions carry up to 252 frames of 12 conditional tokens, covering 241 actor
  commands, market orders and STOP. Sampling and PPO evaluation use the same
  grammar/probability path. The rollout keeps original sampled densities.
- Shared PPO storage preserves the actual game schema through time/segment
  transforms, CPU/CUDA transfers and evaluation. Terminal rewards survive
  native auto-reset; horizon boundaries bootstrap the critic.
- Teacher distillation, Orbit replay export, per-frame PPO clipping and the Orbit
  distributional winner loss are rejected for this initial Kaggriculture path.
  Orbit checkpoints are not compatible with the new model.

See [native RL contract](docs/rl-api-specs.md) and
[model adaptation](docs/kaggriculture-model.md). CUDA throughput, full competition
parity, opponent-league strength and Kaggle submission qualification require their
own evidence; local correctness checks do not establish them.

The inherited Orbit modules/configs/tests remain for regression coverage and reuse;
`configs/kaggriculture.yaml` selects the active v3 game. Historical Orbit setup,
submission packaging and fixture recipes are in
[the preserved upstream documentation](docs/upstream-orbit-wars.md). Those
submission commands still target Orbit Wars and must not be used as a qualified
Kaggriculture submission path.
