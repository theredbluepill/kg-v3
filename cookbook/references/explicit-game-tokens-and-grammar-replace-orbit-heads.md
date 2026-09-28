---
type: "Reference"
title: "Explicit game tokens and grammar replace Orbit heads"
description: "Retain starter transformer blocks with private tokens, native grammar and joint turn density; reduce repeated decoder work with scoped correctness and CPU evidence."
tags: ["kaggriculture-v3", "adaptation", "model"]
status: "verified-scoped"
generated: {"by": "openai/codex", "at": "2026-09-28"}
sources: [{"resource": "repository:tests/kaggriculture/test_model.py"}, {"resource": "repository:ops/v3-port-checks.md"}, {"resource": "user-directive:2026-09-28:record-every-adaptation"}, {"resource": "repository:python/owl/model/kaggriculture.py"}, {"resource": "repository:python/owl/model/config.py"}, {"resource": "repository:python/owl/model/factory.py"}, {"resource": "repository:python/owl/kaggriculture/types.py"}, {"resource": "repository:python/owl/kaggriculture/gpu_sampling_grammar.py"}, {"resource": "repository:configs/model/kaggriculture.yaml"}, {"resource": "repository:docs/kaggriculture-model.md"}, {"resource": "repository:docs/model-architecture.md"}]
---

# Explicit game tokens and grammar replace Orbit heads

## What the sources record

`python/owl/model/kaggriculture.py` retains the starter TransformerBlock attention, pack/unpack sequence utilities, pre-normalization, feed-forward and initialization infrastructure. The owner explicitly prohibits taking the v2 model implementation. No v2 `isaiah_tokens`, `common` or entity encoder model implementation is imported; the game stems, twelve heads, current-action-prefix decoder and seat critic are new v3 code. V2 contributes schema/codec meanings, not neural implementation. It replaces Orbit Wars stems and heads with Kaggriculture feature processing, registered as `model_arch=kaggriculture_transformer`. Each seat is encoded independently from its legal 8,176-feature perspective: 200 tiles, own/rival actor tokens, own-only inventory and storage, products, shops, clock/economic suffix, and learned scratch/policy/critic tokens. No opponent-identity label is an input and no hidden state persists between observations.

Twelve categorical heads follow the same native grammar in sampling and likelihood evaluation. Full action capacity is 241 actors plus ten market orders and STOP, giving 252 frames. Prefix legality and token vocabulary widths are explicit; illegal replayed actions fail clearly. The model returns joint turn log probability to the shared per-player PPO path, with component entropy available for diagnostics. An own-seat critic is computed without shared normalization across private seat views. Its explicit value mode uses tanh for bounded win/loss or win/share, sigmoid for win-only, and a raw linear output for unbounded margin. The margin mode's auxiliary sigmoid winner-probability API is uncalibrated and does not become a win-probability claim.

The current starter-sized configuration uses width256, depth7, eight heads and MLP ratio4; actual model construction counts **8,294,450 parameters**, including5,528,320 in retained starter transformer blocks. The size test targets6–10million parameters. No throughput optimum follows.

The decoder's GRU summarizes only previously selected frames **inside the current turn**, resetting for every observation. This is a scoped adaptation interpretation: it supports autoregressive current-action coordination without reintroducing between-turn memory. The current v3 owner instruction does not independently ban every recurrent module by name; the imported stateless discipline prohibits carried temporal state. This distinction is explicit rather than silently claiming literal adoption of all old Myolie implementation language.

## Interpretation and consequence

Reuse the transformer/training infrastructure while giving the game its actual observation and action semantics. Private perspectives must remain separate even when both seats share weights and are batched together. Preserve sampling/evaluation density agreement, native grammar transitions, action capacity and function of all slot heads. Architecture expressiveness does not establish that PPO learns investment payback, allocation, sale timing or maintenance priorities.

## Checks and remaining gaps

Source inspection confirms the declared token/decoder boundary. The model implementer reports **19/19 CPU model tests** using the properly built PyO3 wheel with Torch2.9: native/Python grammar masks and quantities, sampling/evaluation logp and entropy, finite-difference gradients and trunk updates,17/241 actors and252 frames, private-seat isolation, padding, checkpoint reload, sequence dimensions and bounded/unbounded critic modes. Scoped mypy/Ruff pass. Additional cases cover 100,000 categorical draws with exact masks, factored projection parity, cached grammar binding and finite gradients/DDP parameter participation for active and inactive branches. The final shared receipt reports 772 Python tests passed and a full-model two-process CPU diagnostic of three updates/96 game transitions. These checks qualify source-scoped status.

Legacy Orbit stateless/recurrent models, Agent and checkpoint benchmark explicitly reject Kaggriculture schemas/actions before Orbit conversion; this preserves useful starter code without claiming a Kaggriculture submission/benchmark path. The implementer reports230 legacy cases plus four direct rejection checks; exact commands/source versions are retained in the final port receipt. No CUDA throughput, FlashAttention/DDP qualification, full-capacity learning or competitive strength has been established.


## Avoid repeated work in the autoregressive decoder

The decoder replaces per-slot sampled Categorical validation/sampling with masked Gumbel-max while preserving the categorical law and exact log-probability/entropy computation; a 100,000-draw regression checks frequencies and legality. It factors frame-invariant linear projections, reuses native grammar bindings and caches ranges instead of rebuilding equivalent tables at every slot. Zero-contribution parameter participation keeps inactive action branches compatible with the shared DDP path. These are new v3 implementation refinements, not v2 neural code reuse.

The implementer reports an Apple-M5/macOS-arm64 CPU audit with eight environments, Torch2.9, one intra-op thread, two warmups and five repeats. Native-reset rollout/evaluation means move152.53→149.69ms and157.79→157.42ms; a synthetic16-actor case moves189.96→186.38ms and195.85→193.95ms. Timings are small local differences, not a GPU SPS claim. The stronger bounded trace result is scalar extraction count305→12 (native reset) and680→27 (16actors) during rollout,8→1 in evaluation; range creation drops12→2 in rollout and300/660→2 in evaluation. Sampling consumes different random numbers, so sampled actions are not asserted identical; maximum frame lengths remain12/27. Remaining per-frame STOP scalar checks, maximum-length evaluation and validation transfers still need CUDA profiling. Native grammar cache misses have setup cost. Evidence and exact sources are retained in `ops/port-evidence/`, pinned by `SHA256SUMS` and linked by the final port receipt.
