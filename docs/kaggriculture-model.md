# Kaggriculture transformer

`kaggriculture_transformer` adapts the starter's stateless transformer to the
Kaggriculture native observation and action contract. It uses the existing PPO
trainer, GAE, optimizer, distribution, checkpoint and logging paths.

The transformer implementation comes from the Isaiah Pressman starter. The
Kaggriculture stems, categorical heads, prefix decoder and critic here are new
code. V2 supplies the legal feature-coordinate contract and native action grammar;
its model implementations (`isaiah_tokens.py`, `common.py`, `entity_encoder.py`)
and training loop are not imported or copied.

The default preset uses width 256, seven blocks, eight attention heads and an
MLP ratio of four: **8,294,450 parameters**, measured from `model.parameters()`
under pinned Torch 2.9.0. The starter trunk blocks contain 5,528,320 of those
parameters. Both the config class and `configs/model/kaggriculture.yaml` use this
size, within the owner's requested 6–10 million parameter range.

Each game supplies two independent legal perspectives `[batch, 2, 8176]`.
The model flattens the game and seat dimensions for encoding; attention never
mixes seats. A seat sees its own inventory and storage plus public rival state.
Opponent names, policy identity, league class and historical hidden state are
absent. Public actor ordinals distinguish units; they do not identify opponents.

The current observation supplies 200 tile tokens with position, kind and all
14 maintenance fields; up to 241 own and 241 rival actor tokens; twelve product
tokens; global, storage and ordered-shop tokens; learned scratch tokens; and
one policy and one critic token. Own actor inventories include every actor,
including those beyond v2's historical sixteen-actor checkpoint limit. Public
clock, remaining-time configuration, hiring, land and capacity fields enter the
global stem. Duplicated legacy actor/inventory coordinates are superseded by
the complete current-observation extension. Actor padding is masked.

The reusable trunk is the starter's `TransformerBlock`: pre-norm attention,
separate q/k/v projections, GELU/SiLU/SwiGLU feed-forward blocks, residual scaling,
SDPA fallback and optional packed FlashAttention. Trunk compilation remains
available. Input stems/embeddings and actor/critic heads declare their optimizer
roles through the starter API so Muon's input/output exclusions remain explicit.

The policy emits the native grammar's twelve categorical slots per frame:

| Slot | Vocabulary |
| --- | ---: |
| unit_actor | 241 |
| unit_kind | 20 |
| unit_target | 128 |
| unit_item | 16 |
| unit_quantity_present | 2 |
| unit_quantity_high | 32 |
| unit_quantity | 32 |
| market_kind | 8 |
| market_item | 16 |
| market_quantity_high | 32 |
| market_quantity | 32 |
| stop | 2 |

The representational capacity is 252 frames: all 241 actors, ten ordered market
slots, and a distinct STOP. Runtime work ends at the actual STOP. A configured
smaller `max_decode_frames` fails before decoding if it cannot represent the
observed actor count plus the complete market queue and STOP. Nothing is dropped.

The decoder reads the policy token, the current own actor token or market
context, and the already emitted current-turn prefix. A frame-history GRU is
reset to zero at every observation; it records action ordering within a turn
and never introduces state between turns. Native Rust grammar tables supply
the same tensor masks/transitions for sampling and teacher-forced evaluation.
Forced slots have zero log probability and entropy. The grammar restricts
syntax and configured representation, not profitability or execution success.
Explicit zero market quantities and empty market slots survive unchanged.
The retained evaluator boundary admits explicit transfer quantities 1–1023;
omitting a transfer quantity remains a distinct action.

For PPO, active slot log probabilities sum to a frame density
`[..., 2, 252]`, and the existing per-player loss sums frames into the full-turn
joint density. Entropies are conditional on the sampled/teacher-forced prefix;
their sum is the sampled-prefix estimate for the autoregressive distribution.
Padding contributes zero. Invalid choices, mismatched lengths and tokens after
STOP fail explicitly.

Each seat has an independent scalar critic. `value_mode=win_loss` applies tanh
for win/loss or bounded win-share rewards; its reported win probability is
`(value + 1) / 2`. `win_only` uses sigmoid. `margin` returns an unbounded linear
value for the bank-margin objective `(own bank - rival bank) / 3000`; its
auxiliary sigmoid is only an API diagnostic, not a calibrated win probability.
The two seats' predictions are never normalized together. All modes use MSE
value loss and preserve private-observation isolation. Winner cross-entropy,
teacher distillation, recurrence across observations, Orbit action heads and
LoRA are unsupported by this initial adaptation.

Focused tests live in `tests/kaggriculture/test_model.py`: native-oracle grammar
admission, sample/evaluation densities, policy finite differences and trunk
updates, 17/241-actor and full market capacity, private-perspective isolation,
checkpoint restoration, invalid inputs and segment-major evaluation. CUDA
performance and FlashAttention behavior need measurement on NVIDIA hardware;
no throughput claim follows from CPU correctness checks.

The focused suite passed all 19 cases on 2026-09-28 with Torch 2.9.0 and the
proper maturin-built PyO3 extension. Targeted Ruff and model-module mypy also
passed. These checks establish the stated CPU contracts, not gameplay strength
or CUDA/DDP performance.

The canonical PPO CLI also completed three updates with the full default model
on two CPU Gloo ranks: 96 global environment steps and 192 seat turns. This
checks distributed training correctness through the starter's real rollout and
update path; it does not establish GPU throughput.

The hot decoder projects each observation's actor and market contexts once,
before the frame loop. Repeated grammar bindings and constant index tensors
are cached. Slots 0, 2 and 11 have singleton native support, verified when a
grammar enters the cache, so their logits and random draws are unnecessary.
Their checkpoint parameters remain present with explicit zero gradients to
participate in DDP. Other slots sample categoricals with Gumbel-max; a fixed-seed
100,000-draw frequency test checks that distribution and exact masked support.
Active and wholly inactive evaluation tests check that every parameter receives
a finite gradient. No neural operation loops over individual observations.

A bounded CPU comparison used the actual 8,294,450-parameter model, eight native
environments (sixteen seat perspectives), Torch 2.9.0, macOS arm64, one CPU
thread, two warmups and five timed calls. Rollout timing includes stochastic
sampling with gradients disabled; evaluation builds the autograd graph but
does not run backward. Both include observation encoding and decoding.

| Observation scenario | Rollout before → after, mean ms | Evaluation before → after, mean ms |
| --- | ---: | ---: |
| Native reset, one actor per seat | 152.53 → 149.69 | 157.79 → 157.42 |
| Synthetic sixteen actors per seat | 189.96 → 186.38 | 195.85 → 193.95 |

The CPU profiler counted `aten::_local_scalar_dense` calls dropping from
305 to 12 and 680 to 27 during rollout, and from eight to one during evaluation.
`aten::arange` calls dropped from twelve to two during rollout and from
300/660 to two during evaluation. These operation counts demonstrate removal
of repeated categorical checks and index allocations. They do not measure
CUDA synchronization directly. The RNG algorithm changes sampled programs;
the two runs had the same maximum decoded lengths (12 and 27), but different
mean lengths. The sixteen-actor input is synthetic and is not a gameplay trace.
These short timings show modest CPU changes, not a throughput guarantee.

One live-STOP scalar check remains per sampled frame; evaluation reads its
maximum action length once. Compact validation vectors and observation context
still transfer to the host. Grammar cache misses still validate native plans
and upload tables. The reusable packed attention path has its own dynamic
packing costs. Before any throughput claim, profile the actual four-rank,
eight-environment-per-rank GPU job, including long actor/market programs,
grammar-shape churn, rollout transfer, backward, optimizer and collective
communication. Measure cold/warm trunk compilation separately. CPU checks do
not establish those GPU results.
