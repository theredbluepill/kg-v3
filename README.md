# OWL: Orbit Wars (reinforcement) Learning

## Setup instructions

1. Install prerequisites:
   - [Rust](https://www.rust-lang.org/tools/install)
   - [uv](https://docs.astral.sh/uv/getting-started/installation/)
   - [just](https://github.com/casey/just)
2. Install the repository's pinned Rust toolchain and `rustfmt` component:

```sh
rustup toolchain install nightly-2026-04-18 --component rustfmt
```

3. Generate replay fixtures:

```sh
scripts/regenerate_test_fixtures.sh
```

4. `just prepare` should run without errors. If mapped docs are genuinely still
   current after a small code change, rerun it as `DOCS_CURRENT=1 just prepare`
   to acknowledge that review.

## Containerized builds

See `docs/containerization.md` for building a Docker image with the locked Python
dependencies, Rust toolchain, compiled extension module, and example Slurm launch
patterns.

## Kaggle submission build

Build a Kaggle-compatible image from the current worktree and create
`artifacts/submission.tar.gz` with:

```sh
just kaggle-submission runs/20260505-120000/checkpoint_last_best.pt
```

Pass a submission name to write `artifacts/<name>.tar.gz` instead:

```sh
just kaggle-submission runs/20260505-120000/checkpoint_last_best.pt my-run
```

Pass a quantization format as the third argument to quantize the packaged
model weights during submission generation:

```sh
just kaggle-submission runs/20260505-120000/checkpoint_last_best.pt my-run fp4
```

Pass an optional fallback checkpoint after the existing arguments to package a
second, faster model:

```sh
just kaggle-submission runs/20260505-120000/checkpoint_last_best.pt my-run fp4 \
  --fallback-checkpoint runs/20260501-090000/checkpoint_last_best.pt
```

The submission recipe runs `just prepare`, rebuilds the `orbit-wars:kaggle`
image with Buildx zstd layer compression, compiles the Rust extension inside
Kaggle's Python image, and packages `python/owl`, `python/main.py` or `main.py`,
the requested model bundle, and its adjacent `config.yaml` under
`models/primary/`. If a fallback checkpoint is provided, its model bundle and
adjacent `config.yaml` are packaged under `models/fallback/`.
Rebuilding the image during submission generation keeps the packaged Python code
aligned with the current checkout. The packaged checkpoint keeps the original
checkpoint contents only for validated model state-dict tensor weights needed by
the Kaggle agent; malformed model entries fail during packaging rather than at
agent startup. To store packaged models below fp32 precision, pass a
quantization format such as
`fp8_e4m3fn`, `fp4_e2m1fn_x2_scaled_block16`, or
`nf5_g128_lsq_policy_last_fp8`. Lower-bit normal-float formats `nf4_g128_lsq`,
`nf3_nf4_structured_3p5`, and `nf3_g128_lsq` are also supported; unique prefixes
such as `fp4` are accepted.
The default `fp32` leaves checkpoint weights unchanged. The Kaggle agent
loads quantized slim checkpoints by dequantizing and copying one tensor at a
time into the model, avoiding a full fp32 state-dict copy during startup.
The checked-in `python/owl/agent/agent_config.yaml`
sets `int8_quantization: always`, which converts loaded `nn.Linear` layers to
PyTorch dynamic int8 CPU inference while keeping final actor/critic output heads
in fp32. Set it to `2p` or `4p` to use int8 only for that game size, or `never`
to disable serving-time quantization and use fp32 CPU inference. The packaged
Kaggle agent is CPU-only; model serving, action expansion, and Kaggle action
conversion all keep tensors on CPU. Set
`lora_mode` to control whether LoRA adapters are dequantized and folded into
regular `nn.Linear` weights for every game (`always`) or only for two-player or
four-player games (`2p` / `4p`); folding happens before int8 inference
quantization. Set
`fallback_min_overage_time` in `python/owl/agent/agent_config.yaml` to switch to
the fallback model when remaining overage time drops below that threshold;
`null` disables fallback routing even if the fallback model is packaged. A
packaged fallback config is validated during startup, but the fallback model
weights are loaded on the second observed turn rather than during initial agent
construction. The delayed load still happens even if remaining overage time has
already fallen below the fallback threshold, preferring to risk that one timeout
over giving up fallback for the rest of the game.

The Kaggle observation encoder filters fleets smaller than the configured
`min_fleet_size` while encoding observations. This intentionally trades a small
amount of board-state detail for lower inference latency, because many tiny
fleets can increase entity count enough to trigger fallback routing. To avoid
marking a still-alive player as gone, if a player has no current planets and
all of their fleets are below that threshold, the encoder keeps that player's
single largest fleet in the encoded observation.

## Orbit Wars reference

The Rust rules engine targets the installed `kaggle-environments` Orbit Wars
implementation. Resolve the local module and gameplay prose paths with:

```sh
uv run python -c 'from importlib import import_module; from pathlib import Path; m = import_module("kaggle_environments.envs.orbit_wars.orbit_wars"); print(Path(m.__file__).resolve()); print(Path(m.__file__).with_name("README.md").resolve())'
```

## PPO training configs

Training presets live in `configs/`:

- `baseline.yaml`: PPO with last-best teacher stabilization and the 6m GELU
  stateless transformer preset,
  discrete-target actions, `max_entities=256`, one PPO epoch per rollout,
  larger rollout/minibatch sizing, Muon/AdamW optimizer rates, periodic
  checkpoints every 20M environment steps, `torch.compile` default mode for PPO
  tensor helpers, compiled transformer trunk with
  `max-autotune-no-cudagraphs`, and bfloat16 autocast by default.
- `baseline_adam.yaml`: Adam optimizer variant with explicit optimizer
  settings, including `1e-4` learning rate, `(0.9, 0.999)` betas, `1e-5`
  epsilon, no weight decay, and the same warmup/cosine scheduler shape.
- `baseline_adamw.yaml`: AdamW optimizer variant matching `baseline_adam.yaml`
  except for decoupled `0.01` weight decay.
- `model/stateless_transformer_21m_swiglu.yaml`: larger SwiGLU stateless
  transformer config with an inline discrete-target actor override using eight
  action mixtures.
- `model/stateless_transformer_6m.yaml`, `model/stateless_transformer_11m.yaml`,
  and `model/stateless_transformer_21m_gelu.yaml`: GELU variants of the
  stateless transformer presets.
- `model/stateless_transformer_28m.yaml`: larger SwiGLU stateless transformer
  preset with a discrete-target actor.
- `model/stateless_transformer_12m.yaml`,
  `model/stateless_transformer_25m.yaml`, and
  `model/stateless_transformer_50m.yaml`: a GELU `mlp_ratio=2.0`
  discrete-target ladder scaling width at a fixed `head_dim=32`
  (`320x10`, `448x11`, `512x19`).
- `model/stateless_transformer_200m_d38.yaml`,
  and `model/stateless_transformer_200m_d60.yaml`: depth-specific 200M stateless
  transformer presets with discrete-target actors.
- `kaggriculture_2rank.yaml`, `kaggriculture_4rank.yaml`, `kaggriculture_8rank.yaml`
  and the local CPU `kaggriculture.yaml`: Kaggriculture on the `scaling_6m`
  recipe, with per-rank `n_envs` and `segments_per_minibatch` divided by the
  world size (128/8, 64/4 and 32/2) so the global batch is unchanged. Their `env`
  section is `KaggricultureEnvConfig` (`owl.kaggriculture.config`), which
  `FullConfig` selects when `env.obs_spec.obs_spec` is `kaggriculture`; it
  requires the `kaggriculture_transformer` model and the Kaggriculture training
  guards (`rl.gamma=1.0`, `rl.value_loss: mse`, `rl.ppo_clip_mode: per_player`).
  Their teacher settings are `scaling_6m`'s (`last_best`, KL and value
  coefficients `0.005`, `teacher_segments_per_minibatch: 128`, not divided by
  the world size, as in Isaiah's per-rank configs).
  `run_ppo.py` prints their GEMM workload headroom (or rejects a workload the
  model cannot chunk) and the teacher-target cache bytes per rank (1,674,575,872,
  837,287,936 and 418,643,968 B) before creating the run directory. The
  canonical PPO collector, rollout buffer and update loop support both games
  through typed observation, action and mask mapping. Kaggriculture uses the
  native adapter with caller-owned buffers. The presets set
  `rl.eval_replay_games: 0` until Task 7.3 adds Kaggriculture replay export
  (restoring `scaling_6m`'s 8); a positive value fails at startup before
  creating the run directory, environment or model. The GPU presets (and
  `kaggriculture_1gpu_eager.yaml`) set `rl.checkpoint_freq: 10_000_000`, half of
  `scaling_6m`'s 20M by the owner's decision: each interval writes a checkpoint
  and runs the last-best evaluation (promotion at win rate >= 0.7), about every
  610 iterations of 16,384 env steps.
- `kaggriculture_{2,4,8}rank_bc_finetune.yaml` ("recipe J"): the ranked
  presets with both learning rates divided by 10 (`muon_lr` 0.0002, `adamw_lr`
  1e-5) and nothing else changed, for PPO from the BC best with the BC critic
  head kept (`--load-model-weights-mode model_only`). A 2-rank ablation from the
  BC best found that the full learning rates collapsed the self-play economy
  and that LR / 10 did not; the BC critic head gave a mean explained variance of
  0.84 against <= 0.29 for a fresh head. Banks rising above the BC level is not
  established, and no 8-rank run has used the preset. The 4- and 8-rank presets
  keep the same global workload, so the per-step learning rate carries over by
  construction. The 4-rank main run on this recipe (W&B `gq94cyyp`) collapsed
  the bank economy just after the LR warm-up peak.
- `kaggriculture_{2,4,8}rank_bc_finetune_bank.yaml` (owner decision
  2026-09-30, "A is good + decrease the LR by half?"): recipe J with both
  learning rates halved (`muon_lr` 0.0001, `adamw_lr` 5e-6) and the absolute
  own-bank reward term on (`econ_bank_weight` 0.25, `econ_bank_scale` 100,000,
  `econ_bank_cap` 0.25, so a 70k bank scores .175, the score saturates at
  100k, and `terminal_scale` is 0.5; agent-proposed values, see the Decision).
  Every other Kaggriculture config sets the term explicitly off (weight 0),
  which leaves rewards bit-identical. The trainer logs `train/reward_bank_mean`,
  `train/return_common_mean` and `train/return_zero_sum_abs_mean`; the zero-sum
  winner critic cannot represent the term's common mode. Run J used
  `env.native_threads=4` as a launch override, so a J/2 reproduction must pass
  it again. No run has used these presets. The three bank fields are required,
  so a pre-change `config.yaml` (for example a BC checkpoint's sibling config
  used as `rl.teacher_init`) must add `econ_bank_weight: 0.0`,
  `econ_bank_scale: 100000.0` and `econ_bank_cap: 0.0`, which reproduces its
  rewards exactly.
- `kaggriculture_4rank_margin.yaml` (owner decision 2026-09-30, "0.5 Cash Diff
  (add this in) + 0.5 (Terminal loss 1/-1/0)"): J/2's effective config
  (`kaggriculture_4rank.yaml` with `muon_lr` 0.0001, `adamw_lr` 5e-6,
  `checkpoint_freq` 10M and `native_threads` 4) with only the reward changed:
  starvation/drought shaping off (`econ_shaping` 0), term A off, and the
  zero-sum cash-difference term M on (`econ_margin_weight` 0.5,
  `econ_margin_scale` 50,000, `econ_margin_cap` 0.5, so `terminal_scale` is
  0.5). A game's return is `.5 · clamp(final margin / 50,000, −1, 1) + .5 ·
  sign(final margin)`. The 50,000 scale is agent-proposed, see the Decision.
  Launch it as a warm start from J/2's `checkpoint_final.pt` with
  `--load-model-weights-mode model_and_optimizer`. Every other config sets term
  M off (weight 0), which leaves rewards bit-identical; the trainer logs
  `train/reward_margin_abs_mean`. A pre-change `config.yaml` must add
  `econ_margin_weight: 0.0`, `econ_margin_scale: 50000.0` and
  `econ_margin_cap: 0.0`.
- `kaggriculture_4rank_bank_critic.yaml` (owner 2026-09-30, "per-player critic
  might be the way out?", reward numbers agent-proposed and owner-approved
  with "OK go ahead."): the margin preset with the reward 0.25 × own bank
  (term A, scale 150,000, cap .25) + 0.25 × cash difference (term M, scale
  100,000, cap .25) + 0.5 × terminal sign, and
  `model: kaggriculture_critic_offset`, whose per-seat critic offset head
  (`model.critic_offset`, zero-initialized) lets the value represent the own-bank
  term's common mode. Launch from a checkpoint without the head (the BC best or
  J/2) with `--load-model-weights-mode model_only`: the loader accepts exactly
  the missing `critic_offset_head.*` keys, while `model_and_optimizer` fails
  because the optimizer groups differ. The trainer adds
  `train/value_offset_mean`, `train/value_offset_abs_mean` and
  `train/ev_common`. `-o model.critic_offset_detach_trunk=true` keeps the
  offset's gradient out of the trunk. No run has used it.
- `kaggriculture_4rank_vs_cha22.yaml` and its 2-rank twin
  `kaggriculture_2rank_vs_cha22.yaml` (the cha22 anchor setup; owner
  2026-09-30, "OK, for fixed bot, we can use cha22" and "implement the new
  rewrad first before we revisit the cha22 anchor setup."): the margin preset
  (term M, J/2's halved LRs, `checkpoint_freq` 10M, `native_threads` 4) with
  `env.opponent_mix: {bot: cha22, fraction: 1.0}`, so every env trains against
  the fixed bot Cha22. The 2-rank twin divides `n_envs` and
  `segments_per_minibatch` by the world size only. Launch from the BC best
  with `--load-model-weights <checkpoint_bc_best.pt> --load-model-weights-mode
  model_only`. No run has used them.

The training entrypoint configures PyTorch for TF32 matmul/conv precision and
cuDNN benchmarking before constructing the environment, model, and optimizer.
Fresh launches explicitly reset model parameters before optimizer construction.
When `--load-model-weights` is set, the fresh trainer then replaces those
parameters from the checkpoint. Set
`--load-model-weights-mode model_and_optimizer` to also load optimizer
moment/momentum state while keeping the fresh optimizer hyperparameters and
scheduler state. `--load-model-weights-mode model_fresh_critic_head`
(Kaggriculture only) loads every model tensor except the critic head
(`critic_head.*`), which keeps the fresh launch's initialization; the optimizer
starts fresh as in `model_only`. The checkpoint must still hold every model
tensor, and any checkpoint key or model tensor the trainer does not save is
rejected (also by the `teacher_init` loader). The main rank records the
checkpoint's resolved path, SHA-256 and load mode in the run directory's
`warm_start.json` and as `warm_start/*` metric-run summary keys.
Resume launches load checkpoint weights and optimizer state without resetting
the model first.
Set `model.lora` on stateless transformer configs to run PPO as a LoRA
fine-tune. LoRA freezes the base model, wraps selected linear projections, and
trains only the low-rank adapter parameters. `rank` is required; optional fields
include `alpha_scale` (the LoRA update scale, default `1.0`), `target_modules`,
and `target_block_count` for final-block-only adaptation. Set `target_value_head` / `target_policy_head` to also wrap the
critic and actor heads (set `target_modules` to `[]` to adapt only the heads).
Set `roundtrip_quantization` to a supported checkpoint quantization format to
quantize and dequantize the frozen base weights before adapter training, so
fresh LoRA fine-tuning sees the same base-weight numerics used after final
checkpoint quantization.
LoRA presets under `configs/model/lora/` can be selected with overrides such as
`-o model.lora=2p_200m_qv_r16`.
Checkpoint extraction and Kaggle submission packaging accept separate base-model
and LoRA adapter quantization formats. When base quantization is enabled and no
LoRA format is provided, adapter tensors default to fp16. Packaged LoRA adapters
are folded into the base model before inference int8 quantization/emulation.
Recurrent models do not support LoRA. Fresh LoRA launches can use
`--load-model-weights` with a
non-LoRA base checkpoint; missing LoRA adapter tensors are initialized from the
config while base tensors are loaded from the checkpoint. Fresh LoRA launches
must use `--load-model-weights-mode model_only`; resume existing LoRA runs to
restore optimizer state. `teacher_mode: last_best` also works with LoRA: the
self-play opponent is built with the student's adapter architecture and seeded
from the (possibly non-LoRA) `teacher_init` checkpoint, so it can be refreshed
in place each time the student wins.
Optimizer configs may set `lr_schedule.schedule` to
`linear_warmup_cosine_decay` for warmup followed by cosine decay, or `cosine`
for a repeating LambdaLR multiplier that moves from `1.0` to `lr_min_ratio`
halfway through `full_cycle_steps` optimizer steps, then back to `1.0` at the
end of the cycle. The
schedule name selects the accepted scheduler fields; unrelated scheduler fields
are rejected.
Training `EnvConfig.n_envs` must be even. PPO updates run
`rl.ppo_epochs` full-shuffle passes over rollout segments, grouped by
`rl.segments_per_minibatch`; set `rl.gradient_accumulation_steps` above `1` to
accumulate multiple minibatches before each optimizer step. `EnvConfig.n_envs`
must be divisible by
`rl.segments_per_minibatch * rl.gradient_accumulation_steps`. In distributed PPO
launches, `EnvConfig.n_envs`, rollout horizon, minibatch segment width, and
gradient accumulation are per GPU. Checkpoint cadence, `--max-env-steps`, W&B
step values, and `train/env_steps` are counted across all ranks. Resume launches
with a different GPU count derive an equivalent per-rank config by scaling
`env.n_envs` and
`rl.segments_per_minibatch * rl.gradient_accumulation_steps` by
`saved_gpus / current_gpus`. The derived config keeps
`rl.segments_per_minibatch` at or below the saved value, so the per-minibatch
training batch does not increase; resume fails if the scaled values are
fractional or config-invalid.
When `env.reward_mode` is `win_only`, set `model.value_mode` to `win_only` so
critic values train against raw winner probabilities; other reward modes require
the default `win_loss` value mode.
`rl.value_loss` selects the critic objective. The default `mse` regresses the
scalar value toward the GAE return. `winner_ce` instead trains the
winner-probability softmax as a classifier, using the categorical cross-entropy
toward a distributional GAE(lambda) winner target (the same lambda-return
recursion carried on the per-player winner distribution, resolving to the
terminal winner distribution and bootstrapping the critic's distribution on
time-limit truncation). It requires `env.reward_mode: win_only` (hence
`model.value_mode: win_only`), `model.critic_mode: softmax`, `rl.gamma: 1.0`, and
`rl.vf_clip_coef: null`; see `configs/winner_ce_6m.yaml`.
When `rl.normalize_advantages` is enabled under distributed PPO, advantage mean
and variance are computed over the masked global minibatch across ranks.
`rl.eval_replay_games` must be no larger than `env.n_envs` because evaluation
samples replay games from the same vectorized eval batch.
`rl.ppo_clip_mode` defaults to `per_player`, which clips the summed per-player
joint action log-probability. Set it to `per_entity` to clip each controllable
action entity independently before summing those clipped policy-loss terms back
to the player-step.
A Kaggriculture environment requires `env.reward_mode: win_loss`, `rl.gamma:
1.0`, `rl.value_loss: mse` and `rl.ppo_clip_mode: per_player`: its per-seat
critic reads `2p(self) - 1`, economic shaping bounds complete-episode returns
only undiscounted, and a seat's turn is one joint autoregressive action. With
`rl.truncation_step`, an Orbit cut transition's reward is zeroed, while a
Kaggriculture cut keeps the economic reward earned on that transition; both
bootstrap from the critic's value of the cut state.
`rl.initial_stagger: true` (Kaggriculture only, default false and then omitted
from the config dump) cuts each env's first game through the same path, at a
step drawn from `env.seed` and the global env index. Later games run to their
natural end, so every rollout mixes game phases and carries game ends. See
`docs/rl-api-specs.md` ("Staggered game phases") and the
`configs/kaggriculture_{4,2}rank_bank_critic_credit.yaml` presets. Those
presets pair the stagger with 256-step segments at `gae_lambda: 1.0`.
`rl.first_minibatch_logratio_limit` (default `0.05` nats) is a correctness
alarm. Before the first optimizer step of each update, the policy-weighted mean
log-ratio of the first minibatch (replayed versus rollout log-probs, reduced
across ranks) must stay within the limit. Otherwise training raises a
`RuntimeError` that reports the rollout batch shape and every observation
tensor shape, action-mask tensors included, with parameters still unchanged. Set it to `null` to disable the check. The limit
uses the same units as `rl.ppo_clip_mode`'s log-ratio: under `per_player` it
bounds the joint action (entity log-probs are summed, so a coherent drift of
`d` nats on each of `K` acting entities reads as `K * d`); under `per_entity`
it bounds the mean per-entity log-ratio (the same drift reads as `d`). The
`0.05` default has not yet been measured against GPU BF16/compile replay noise;
the rebuild's Phase 6 GPU qualification measures that margin.
PPO supports `pure`, `discrete_targets`, and `discrete_target_bins` action specs
when the `StatelessTransformerV1` actor discriminator matches the environment
action spec. The current discrete-target actor requires
`max_per_planet_launches: 1`; the target-bin actor requires matching `n_bins`.
Both discrete target specs default to `targeting_mode: full_mask`; set
`stop_bad_launch` or `anything_goes` to expose loose target masks while
controlling whether sun-crossing decoded launches are replaced with no-ops.
`RecurrentTransformerV1` supports only `discrete_targets` with
`launch_mode: binary` and `max_per_planet_launches: 1`.
Set `rl.teacher_mode` to `fixed` or `last_best` to add student-teacher
stabilization losses. `fixed` requires `rl.teacher_init`, while `last_best`
uses the current last-best snapshot. On randomly initialized fresh launches
with no `teacher_init`, the last-best teacher losses stay disabled until the
current model first replaces `checkpoint_last_best.pt`; fresh launches from
`--load-model-weights` use that starting checkpoint as the initial last-best
teacher. Fixed teachers do not seed `checkpoint_last_best.pt`; win-rate
evaluation against last-best follows the same checkpoint lifecycle as a run
without a teacher. `rl.teacher_init` points at a training checkpoint whose
adjacent `config.yaml` is used to construct the teacher model before loading
weights. A fresh Kaggriculture launch with `teacher_mode: last_best` must name
its teacher checkpoint: `--load-model-weights CHECKPOINT` (the BC best seeds the
student and the last-best teacher) or `-o rl.teacher_init=CHECKPOINT`;
`run_ppo.py` rejects it before the workload check otherwise (pass
`-o rl.teacher_mode=null` to train without a teacher). No Kaggriculture config
carries a checkpoint path. Orbit launches keep the scratch behavior above.
For a fixed teacher, the architecture may differ from the student. A last-best
teacher always uses the student's architecture so it can be refreshed in place.
Observation specs must match except for `max_entities`, where the teacher model
uses the student capacity for rollout tensors; action specs must match exactly. Actor
factorization details such as discrete-target launch mode or target-bin count
must be compatible. Teacher models must be stateless; recurrent teachers are
rejected because PPO teacher inference runs only from stored rollout segments.
The frozen teacher trunk runs once per iteration in a chunked `no_grad` pass
after rollout (chunk size
`rl.teacher_segments_per_minibatch`, default `32` segments); its distribution
targets are cached and each update minibatch consumes them without re-running
the teacher trunk. `teacher/cache_bytes` logs this rank's cached target bytes
each iteration (0 without an active teacher). For Orbit, the cached action-KL path supports only the `discrete_targets`
actor without player-count adapters (a fixed teacher must also match the
student's launch mode); `KaggricultureTransformer` supports both cached paths.
Value distillation does not use the actor KL path, but
the trainer still requires matching action specs and non-adapter models. The
model owns the per-state value cross-entropy reduction
(`teacher_value_cross_entropy`): Orbit's joint winner distribution gives one CE
per state, while Kaggriculture averages each live seat's CE. Stateless student
models receive neither `hidden_state` nor `dones` on the teacher paths.
`rl.teacher_kl_coef` and `rl.teacher_value_coef` weight the action KL and
per-state winner-distribution cross-entropy stabilization losses; both default
to `0.001`. `rl.teacher_schedule.mode` defaults to `none`; set it to
`linear_decay` with `decay_steps` and `decay_min_ratio` to linearly decay both
teacher coefficients by optimizer step down to the configured minimum ratio.

Run a preset with:

```sh
uv run python scripts/run_ppo.py configs/baseline.yaml runs --log-mode debug --max-env-steps 16
```

Fresh launches accept `-o`/`--overrides field.path=value`; when provided, rank 0
prints the flattened override list before loading the config.

Every `run_ppo` launch logs to W&B project `kg-v3` (any game) by default. W&B
takes the entity from `WANDB_ENTITY` or the key's default entity. Before loading
the config, rank 0 checks telemetry: `--log-mode wandb` with the default
`--wandb-mode online` fails fast with `MissingWandbCredentialsError` unless
`WANDB_API_KEY` is set (neither blank nor padded) or the `NETRC` file (default
`~/.netrc`) has a password for the `WANDB_BASE_URL` host (default
`api.wandb.ai`; an http(s) URL without embedded credentials). The error names
the fix: install the credential with
`cookbook/workflows/install-the-wandb-credential-before-any-pod-launch.md`,
which copies only that entry with `scripts/export_wandb_netrc_entry.py`. In
either W&B mode (online or offline) the same startup check also rejects, before
the config, env or model exist, a set `WANDB_MODE` that differs from
`--wandb-mode` (wandb 0.26.1 would silently let the flag win, or reject an
empty value), and two settings that `wandb.init` would otherwise reject later:
a `WANDB_BASE_URL` that is set but empty, is not such a URL, or fails the
installed wandb's own `Settings` validation (for example `https://wandb.ai` or
`http://api.wandb.ai`; the error withholds the value); and a set
`WANDB_API_KEY` that is blank, padded or fails that validation. Other W&B
settings are not pre-checked. Once the run
starts, the logger stops if W&B reports a mode other than the requested one, so
the receipt records the mode the run actually used. Running without live
telemetry takes an explicit flag, either `--wandb-mode offline` (metrics stay in
the run directory's `wandb/` folder until `wandb sync`) or `--log-mode debug`
(stdout only); online failures never switch to offline. Either prints a
`W&B TELEMETRY OUTAGE` banner to stderr at startup, and an offline run also
names its `wandb/` folder once the run directory exists. `--wandb-mode offline`
requires `--log-mode wandb`.

Each launch or resume appends one record to the run directory's
`attempts.jsonl`. The record holds `attempt`, `experiment_id`
(`--experiment-id`, default the run directory name; resumes keep it), `job_type`
`ppo`, `source_commit` (`git HEAD`, `-dirty` for tracked edits;
`--source-commit` only where the checkout has no git metadata, and rejected when
it disagrees with git), every attempt's commit so far, `config_sha256` (canonical
JSON of the resolved config), `telemetry_mode` (`wandb-online`, `wandb-offline`
or `disabled`), the W&B project, entity, run ID and URL, `start_env_steps` and
`started_at` (a timezone-aware ISO time). Resume needs the run's `attempts.jsonl` and validates every field,
the attempt order, a constant experiment ID and job type, and the source-commit
history. The W&B run is named `ppo-<run-directory-name>`, uses the experiment
ID as its group, `ppo` as its job type, and tags `kaggriculture-v3`, `ppo` and
the game (`kaggriculture` or `orbit`). Training and evaluation metrics use the
same logger calls. It stores `v3.experiment_id` and `v3.job_type` in its config,
and the attempt's `v3/*` fields in its summary.

A resume passes the saved W&B run ID with `resume="must"`, online only: resume
launches reject `--wandb-mode offline` before allocating anything, because wandb
0.26.1 ignores `resume` offline and would start a separate same-ID segment
rather than continue the run. To resume a run whose earlier attempts were
offline, first `wandb sync` its `wandb/offline-run-*` folders; otherwise the
online `wandb.init` fails because the run does not exist remotely.

For a Kaggriculture launch on a suitable training host:

```sh
torchrun --nproc-per-node 8 scripts/run_ppo.py \
  configs/kaggriculture_8rank_bc_finetune.yaml runs \
  --load-model-weights BC_BEST_CHECKPOINT \
  --load-model-weights-mode model_only --wandb-mode online
```

`model_only` keeps the whole BC model, critic head included (see the BC warm
start section below). `configs/kaggriculture.yaml` is the tiny local CPU
config for the same launch without `torchrun`.

The fresh Kaggriculture `last_best` launch names its teacher checkpoint
(`--load-model-weights` or `-o rl.teacher_init=CHECKPOINT`; see the teacher
paragraph below).

Kaggriculture startup requires `env.seed` in `[0, 2**61)` and bounds the launch
so every rollout seed remains below `2**62`, apart from the evaluation band.
The budget reserves construction and trainer-reset seeds, allows both an
auto-reset and a truncation per environment step, and includes a full update
of stopping-point overshoot. An excessive `--max-env-steps` fails before run
allocation; omitting it uses the safe ceiling. The admitted environment-step
counter also remains in `_evaluation_seed`'s `[0, 2**61)` domain. A
Kaggriculture launch that keeps a checkpoint's `env_steps` (a resume, or a
fresh launch with `--load-model-weights` in any mode) starts its rollout
seeds at `env.seed + 4 * env_steps`, past every seed that checkpoint trained on
under the same `env.seed`, so it does not replay the earlier launches' worlds.
Startup reads that step from the checkpoint before allocation (a memory-mapped
metadata read), and fails if the trainer's later full load finds a different
`env_steps`. Such a launch whose saved step leaves no seed budget fails at
startup.

`rl.model_compile` defaults to `trunk`, which compiles the stateless
self-attention transformer trunk as one dynamic-shape callable after
FlashAttention packing and before unpacking, using
`rl.model_compile_mode: max-autotune-no-cudagraphs`. Set
`rl.model_compile=mlp` to compile each shared or per-player-count adapter
transformer-block MLP in place with `dynamic=True`, keeping attention packing
and flash-attn calls eager while allowing Inductor to optimize the FFN path. Set
`rl.model_compile=none` for short CPU smoke tests or compile-debugging runs.
The trunk mode dispatches through the model's `TrunkCompileAPI`
(`StatelessTransformerV1`, `KaggricultureTransformer`) and never compiles the
whole model; it rejects cross-attention observations, recurrent models, and
player-count adapter trunk blocks.
Compiling a `KaggricultureTransformer` (either target, any mode, through
`configure_model_compile` or a direct `compile_transformer_trunk` call) first
checks the probed compile stack (`KAGGRICULTURE_PROBED_COMPILE_STACK` in
`python/owl/model/compile_gemm.py`: torch 2.9.0, triton 3.5.0, NVIDIA driver
595.91.07). An installed triton is always checked; on hosts without CUDA the
driver check is skipped, and so is the triton check when triton is not
installed, each with a printed reason. It then sets
`torch._inductor.config.max_autotune_gemm_backends = "ATEN"` so compiled GEMMs
lower to cuBLAS instead of Inductor's Triton GEMM templates (cookbook decision
`kaggriculture-compiles-gemms-with-cublas-only`). Orbit compiles keep the
backends they find. The setting is process-global, so every compile entry point
claims it for its game, and a process that compiles one game refuses to compile
the other, whichever entry point either compile uses. `run_ppo` repeats the stack check before
creating the run directory, prints the claimed backends and stack, and records
them as `compile_gemm_*` and `compile_stack_*` W&B summary fields.

Fresh launches can also initialize the model from an existing full training
checkpoint without resuming the optimizer, scheduler, config, or W&B run:

```sh
uv run python scripts/run_ppo.py configs/baseline.yaml runs \
  --load-model-weights runs/20260505-120000/checkpoint_last_best.pt
```

This loads only `checkpoint["model"]` plus the `env_steps`,
`player_step_total`, `total_games_played`, and `total_active_entities` logging
counters. Optimizer steps, target-KL counters, optimizer state, scheduler state,
and checkpoint config are fresh for the new run.
Pass `--load-model-weights-mode model_and_optimizer` to additionally load the
checkpoint optimizer moment/momentum state. That mode still keeps optimizer
steps, target-KL counters, scheduler state, checkpoint config, and W&B run state
fresh; optimizer hyperparameters such as LR and weight decay come from the fresh
config, not from the checkpoint optimizer param groups.

PPO run directories save `config.yaml` alongside checkpoints. The saved config
includes `runtime.n_runtime_gpus`; resume uses it to keep the effective rollout
and optimizer-step batch shape equivalent when the current launch uses a
different number of ranks, and fails when no exact derived config exists.
Checkpoints save model, optimizer, scheduler, environment-step metadata,
optimizer-step metadata, player-step metadata, plus the W&B run ID used for
resume. Resume training by passing either a run directory or a checkpoint file
as the only positional path:

```sh
uv run python scripts/run_ppo.py runs/20260505-120000
uv run python scripts/run_ppo.py runs/20260505-120000/checkpoint_00_020_000_000.pt
```

Directory resume compares `checkpoint_final.pt` with the latest numbered
checkpoint when both exist, loads the one with more saved environment steps, and
never treats `checkpoint_last_best.pt` as the primary training checkpoint. When
`checkpoint_final.pt` is absent, directory resume loads the latest numbered
checkpoint. File resume loads `config.yaml` from the checkpoint's parent
directory. Both resume modes require the associated
`checkpoint_last_best.pt` and W&B logging so the saved run ID can be resumed.
Checkpoints do not save the Rust environment state or current observation, so
resumed runs continue from a fresh environment batch rather than acting as exact
simulator snapshots. Periodic checkpoint names use grouped zero-padded
environment-step labels such as `checkpoint_00_022_000_000.pt`. At each
periodic checkpoint, the current model is evaluated against the last-best
snapshot using sampled policy actions across `env.n_envs` games using the
configured `env.two_player_weight`, with current and last-best seats randomly
shuffled across active player slots for each eval game, and logs
`eval/win_rate_against_last_best` plus terminal environment metrics under
`eval/`. When the current model reaches at least 70% eval win rate, the
last-best snapshot is replaced and also saved as `checkpoint_last_best.pt`. If
the first periodic checkpoint finds no `checkpoint_last_best.pt`, the starting
last-best model is saved there first, using its starting environment-step count.
Each evaluation also logs `eval/games` (games scored), `eval/promoted` (1 when
the snapshot was replaced, else 0; logged only after the promotion's
checkpoint write and barrier complete) and `eval/promotion_threshold` (0.7).
Kaggriculture evaluation games are won by the higher raw final bank (equal
banks draw), never by the shaped training return, and log
`eval/candidate_bank`, `eval/last_best_bank` and `eval/candidate_bank_margin`
from the candidate's seat. Learner-perspective raw-bank telemetry follows
(`docs/rl-api-specs.md`): each evaluation adds `eval/bank_games` and the mean
and p10/p50/p90 of `eval/own_bank_*`, `eval/opponent_bank_*` and the signed
`eval/margin_*` (candidate minus last-best, unlike the seat-0 `eval/margin_0`); each training update adds `train/bank_games`,
`train/own_bank_{mean,p10,p50,p90}` over both learner seats,
`train/margin_abs_{mean,p50}`, `train/winner_bank_mean`,
`train/loser_bank_mean` and `train/draw_rate` over its completed games. The
training margin is absolute because a signed self-play margin is identically
zero. These keys are telemetry only and absent for Orbit. Each Kaggriculture
evaluation seeds its games with `_evaluation_seed(base_seed, env_steps)`, a
reproducible seed that differs per evaluation step within a run (the seed ranges of different evaluations or
runs are not guaranteed disjoint), in the non-negative int64 band
`[2**62, 2**62 + 2**61)`. `_create_eval_env` now constructs an independent native
Kaggriculture adapter through `owl.game.create_env`, with rank 0, world size 1
and the evaluation transfer device. Policy evaluation uses the same typed
observation/action mapper as PPO, and the adapter fences device reads before
reusing its buffers. Orbit evaluation environments stay unseeded.

Fixed-opponent PPO (`env.opponent_mix: {bot: <registry key>, fraction: f}`,
Kaggriculture only) trains against a scripted `opponents_rs` bot instead of
mirror self-play in the first `f * env.n_envs` envs of each rank (a whole
number, at least one). The learned seat alternates by env index and episode.
The bot acts natively inside the step. The learner mask keeps the bot's seat
out of every loss term, advantage normalization and denominator, and the
rollout forward skips its rows. Each update logs `train/win_rate_vs_bot`,
`train/own_bank_mean_vs_bot`, `train/margin_mean_vs_bot` and
`train/bank_games_vs_bot`. Each checkpoint evaluation adds a fixed-bot
evaluation in both seats, `eval/*_vs_bot`. Promotion still reads the last-best
win rate only, and the bot's name is only the W&B summary label
`opponent_mix/bot`. Absent (the default) is pure self-play, byte-identical to
before; details in `docs/rl-api-specs.md`, "Fixed-opponent collection". For
example, `-o env.opponent_mix.bot=r04 env.opponent_mix.fraction=1.0`. The
registry keys are `starter`, `r04`, `ecobot`, `e776` and `cha22`; the
`*_vs_cha22.yaml` presets above host Cha22 in every env.
For Orbit, set `rl.eval_replay_games` to a positive count to save random eval replay
samples from the weighted eval game set under
`eval_replays/<checkpoint-name>/` in the run directory. The sampled game
ordinals are selected up front rather than taking the first games to finish.
Each sampled eval game is written as its own JSONL file.

`--log-mode wandb` (the default) publishes every run, Kaggriculture or Orbit,
to the W&B project `kg-v3` under its experiment ID and game tag; the launch
section above covers the credential check, `--wandb-mode offline` (rejected for
resume launches), the outage banner and the `attempts.jsonl` receipt.
Exceptions escaping the training logger session, including `KeyboardInterrupt`
and `SystemExit`, close W&B with exit code 1; normal completion closes it with
exit code 0. The distributed session prints and flushes a rank-tagged traceback
to stderr before destroying the process group, so process-group teardown cannot
delay that failure report.

Training logs terminal environment metrics under `train/` when episodes finish
during a rollout, including game length, per-player win rates, launch density,
planet occupancy for 2-player and 4-player games, max-entity overflow counts,
terminal ship counts, completed game counts, planet captures, launch and fleet-size statistics,
neutral planet/comet undershot rates, full-length game rate, cumulative active
player-step totals, and fleet/ship losses in combat, the sun, or out of bounds.
Rollout observation mix is logged as `train/1p_rate`, `train/2p_rate`,
`train/3p_rate`, and `train/4p_rate` from `obs.still_playing` alive counts.
Planet occupancy is reported at terminal as
`train/terminal_planet_occupancy_rate_2p` and
`train/terminal_planet_occupancy_rate_4p`.
Policy logs include total entropy plus policy-specific component means such as
`policy/launch_entropy`, `policy/target_entropy`,
`policy/fleet_size_full_entropy`, and `policy/angle_and_size_entropy`.
Teacher runs additionally log `teacher/kl`, `teacher/value_cross_entropy`,
weighted loss terms, and per-action KL components such as
`teacher/launch_kl`, `teacher/target_kl`, or `teacher/fleet_size_full_kl`.
Teacher precompute timing is logged as `time/teacher_seconds` and
`perf/teacher_sps`; both are `0.0` when no teacher precompute runs.
Iteration throughput is logged as `perf/steps_per_second`, plus
`perf/tokens_per_second` for unmasked model tokens and
`perf/active_entities_per_second` for action-taking source entities. These
rates use total iteration time, including rollout, teacher precompute, and PPO
update time. The active-entity count is also accumulated as
`train/total_active_entities` and saved in checkpoints.

## Kaggriculture BC warm start

`scripts/train_bc.py` behavior-clones public replays into the Kaggriculture
policy before PPO (rebuild plan Task 5.2). It is offline supervised training,
not a second PPO loop, and launches like `run_ppo.py` (torchrun for several
GPUs):

```bash
torchrun --nproc-per-node 2 scripts/train_bc.py configs/bc/kaggriculture_2rank.yaml \
  --data <task-5.1-dataset> --output-dir runs/bc [--experiment-id <id>]
torchrun --nproc-per-node 2 scripts/train_bc.py runs/bc/<run> --data <dataset>  # restart
```

W&B uses `run_ppo`'s one path and credential check (see "PPO training configs"):
project `kg-v3`, job type `bc`, run name `bc-<run dir>`, grouped by
`--experiment-id` (default: the run directory's name) and tagged
`kaggriculture-v3`, `bc`, `kaggriculture`. Online is the default; without a key
rank 0 fails with `MissingWandbCredentialsError` before any config or data
load. `--wandb-mode offline` or `--log-mode debug` is an explicit outage with
the loud banner. Each attempt writes the shared `attempts.jsonl` receipt, and
`telemetry_mode` also appears in `bc_attempts.jsonl`, the best-checkpoint
record and `bc_result.json`. A restart continues the saved W&B run, so it
requires online W&B and a run that was not launched with `--log-mode debug`.

A restart is a new attempt: `bc_attempts.jsonl` gains a record with the
checkout's own source (`git`, or `--source-commit` on a checkout without git
metadata; a value that disagrees with git is rejected), the parent `bc_state.pt`
SHA-256, every earlier attempt's source, the experiment id, the settings hash
(`bc_config_sha256`: the BC config without its `ppo_config` path, plus the PPO
config's content) and the telemetry mode; it must agree with `attempts.jsonl`
on every earlier attempt. The best-checkpoint record and `bc_result.json` carry
the attempt that wrote them. The restart must keep the saved trajectory's
settings (the BC config except `max_steps`, and the whole PPO config); only
`max_steps` in the run's `bc_config.yaml` may be raised. BC run directories
created before this receipt existed have no `attempts.jsonl` and cannot restart.

A BC config (`BCConfig`, `owl.train.bc`) names the PPO config it warm-starts
(`ppo_config`); the model, `rl.dtype` (BF16 autocast) and `rl.model_compile`
(through `configure_model_compile`, so the Kaggriculture cuBLAS-only claim
applies) come from it, and the optimizer is built by `create_optimizer` /
`create_lr_scheduler`. The GEMM workload check covers the BC microbatch and
validation forwards. Data are Task 5.1 `kaggriculture-bc-shard-v1` shards
(`owl.kaggriculture.bc_data`: `manifest.json` plus one compressed `.npz` per
episode, episode-level `train`/`validation` split). Loading verifies every
shard's SHA-256, schema id and full contract, and each rank keeps rows
`[rank::world_size]` of every episode in host memory (exact integers stored as
range-checked int32, tokens as int16, gathered back as int64).

The loss is each seat's teacher-forced program NLL divided by its length
(`evaluate_actions`, whose replay validation admits the recorded programs),
plus `value_coef` times the winner cross-entropy against the episode's raw
final banks. The critic is trained, not frozen. Each rank draws its training
rows from a permutation seeded by `(seed, epoch, rank)`, so a restart from
`bc_state.pt` repeats the uninterrupted run's rows, updates, evaluation steps and
stopping step on a deterministic device. Every `eval_interval_steps` all
validation rows are evaluated; the lowest held-out NLL is saved as
`checkpoint_bc_best.pt` with exactly `run_ppo.py`'s checkpoint keys (`env_steps`
0), beside the PPO `config.yaml`, so PPO can start from it with
`--load-model-weights`, which also seeds the last-best teacher that a fresh
Kaggriculture `teacher_mode: last_best` launch requires. Start PPO from it with
a `*_bc_finetune.yaml` preset and `--load-model-weights
<run>/checkpoint_bc_best.pt --load-model-weights-mode model_only`, which keeps
the BC critic head. The handoff first chose `model_fresh_critic_head`, because
every BC game was the imitated team's win and the BC critic saturates
(|value| > 1 - 2e-6 on 97% of one held-out game's seat values). A 2-rank
ablation at LR / 10 contradicted that choice: the BC head's mean explained
variance was 0.84 at two seeds against <= 0.29 for the fresh head, at equal
trunk movement. `model_fresh_critic_head` stays available as the diagnostic
comparison. Training stops after
`patience_evals` evaluations without an improvement of more than `min_delta`
over the last such improvement, at `max_steps` or at `--max-runtime-hours`;
`min_delta` sets only that patience count, and every strict new minimum still
replaces the best checkpoint. A budget or runtime stop between scheduled
evaluations evaluates once more; that evaluation can replace the best checkpoint
but does not count toward patience, so a resumed run keeps the uninterrupted
cadence and its best is never higher in NLL than the uninterrupted run's. `bc_history.jsonl` holds the NLL curve,
`checkpoint_bc_best.json` the best checkpoint's SHA-256 and step, and
`bc_result.json` the stopping reason. W&B runs go to project `kg-v3` with
`job_type` `bc`.

## Replay capture

`scripts/benchmark_checkpoints.py` can save replay JSONL samples with
`--save-replay-games N`, split across 2-player and 4-player benchmark games
according to `--two-player-weight`. Files are written under `--replay-dir`,
defaulting to `replays/benchmark_checkpoints`.
Each sampled benchmark game is written as its own JSONL file.
For GPU-friendly int8 quality checks, pass `--int8-emulation [none|a|b|both]` to
emulate x86 int8 quantization of non-output `nn.Linear` weights and activations
for one or both checkpoints while leaving final actor/critic output heads
unquantized. This emulates x86 int8 numeric degradation on `--device`; it does
not measure real int8 kernel throughput.
Open `tools/orbit_wars_replay_viewer.html` in a browser and choose
a saved `.jsonl` file or Kaggle episode replay `.json` file to play back a
game.

Replay rows contain raw Rust environment snapshots for one completed game:
board constants, step/config values, outer-slot owner IDs, player maps, action
entity slots, planets, fleets, comets, rewards, dones, and model assignments.
The Python replay recorder samples game ordinals randomly before rollout and
uses Rust terminal snapshots captured before vectorized env auto-reset.

## Orbit Wars replay parity

Replay parity tests use compact Kaggle episode transition fixtures. The
`replay-*.jsonl` files are intentionally ignored by Git because full episodes can
be large.

The current reference episodes are:

- `75930761` (2-player game)
- `75926553` (4-player game)

If fixture files are missing, download them directly into the test fixture
directory:

```sh
scripts/regenerate_test_fixtures.sh
```

The regeneration script requires Kaggle API credentials configured for the local
user.

### Replay parity workflow

The fixture shape is JSONL with one row per transition: episode id, player
count, step, normalized numeric player action triples from
`steps[t][player].action`, per-player Kaggle `status` and `reward`, the input
observation from `steps[t - 1][0].observation`, and the expected state from
`steps[t][0].observation`. `cargo test` discovers all `replay-*.jsonl` files in
the fixture directory and fails by default if none are present.

Supported test environment variables:

- `ORBIT_WARS_PARITY_FIXTURE_DIR`: directory containing extracted JSONL parity
  fixtures.
- `REQUIRE_PARITY_FIXTURES=0`: skip replay parity, and skip generation parity
  only when generation fixtures are missing. Missing fixtures fail by default.

When the upstream rules change without replacing the reference episode set,
keep the replay test code stable. When replacing episodes, download replacement
JSONL fixtures, update `REQUIRED_REPLAY_COVERAGE` in
`src/rules_engine/replay_tests.rs` with each episode's id, player count, and row
count, and update the episode lists in this README, `docs/rules-engine.md`, and
`docs/rules-parity-coverage.md` before running the parity tests.

## Generation parity

Map generation, reset/home assignment, comet path generation, and comet ship
sampling are checked against fixtures produced by the Python reference
implementation under recorded random streams. Regenerate those fixtures after
intentional upstream rule changes:

```sh
uv run python scripts/generate_reference_fixtures.py
```

The generated fixture is written to
`tests/fixtures/generation/reference_generation.json` and ignored by Git.

## Updating tests after Python rule changes

When the official Orbit Wars environment changes, update the Rust parity tests
in this order:

1. Update the installed `kaggle-environments` package to the latest version.
2. Regenerate all test fixtures. With no arguments, the script uses the current
   reference episodes listed above:

```sh
scripts/regenerate_test_fixtures.sh
```

To switch replay episodes, pass the replacement Kaggle episode IDs:

```sh
scripts/regenerate_test_fixtures.sh NEW_EPISODE_ID_1 NEW_EPISODE_ID_2
```

The script removes outdated `replay-*.jsonl` files before downloading the new
set, and rewrites `tests/fixtures/generation/reference_generation.json` from the
installed Python environment. Replay tests also validate the documented
reference episode player counts and row counts, so update the required coverage
in `src/rules_engine/replay_tests.rs` when replacing the episode set.

3. Update the documented episode IDs in this README, `docs/rules-engine.md`,
   and `docs/rules-parity-coverage.md`.

4. Run the full checks with the new fixtures present:

```sh
just prepare
```

5. Fix any failing Rust parity tests by matching the updated Python behavior,
   then rerun `just prepare`.

Generation and replay fixtures remain ignored by Git; keep only the episode IDs
and setup commands in source control.
