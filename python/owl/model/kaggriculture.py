"""Current-observation transformer and native-grammar Kaggriculture policy.

Each seat is encoded independently. Decoder recurrence records only the action
prefix in the current turn; no hidden state survives a game observation.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Self, cast

import torch
from pydantic import Field, model_validator
from torch import nn
from torch.nn import functional as F

from owl.config import BaseConfig
from owl.kaggriculture.gpu_sampling_grammar import GrammarBatch
from owl.kaggriculture.native_bridge import MyolieSampler
from owl.kaggriculture.types import (
    MAX_ACTORS,
    MAX_FRAMES,
    SLOT_WIDTHS,
    KaggricultureActionConfig,
    KaggricultureActions,
    KaggricultureObsBatch,
    KaggricultureObsConfig,
)
from owl.model.attn import use_flash_attn
from owl.model.base import (
    BaseModelAPI,
    InputLayer,
    ModelActionEntropies,
    ModelActionLogProbs,
    ModelEvaluation,
    ModelHiddenState,
    ModelOutput,
)
from owl.model.stateless_transformer_v1 import (
    PackedSequence,
    StatelessTransformerV1Config,
    TransformerBlock,
    pack_sequence,
    unpack_sequence,
)

SLOT_NAMES = (
    "unit_actor",
    "unit_kind",
    "unit_target",
    "unit_item",
    "unit_quantity_present",
    "unit_quantity_high",
    "unit_quantity",
    "market_kind",
    "market_item",
    "market_quantity_high",
    "market_quantity",
    "stop",
)
_FORCED_SLOTS = (0, 2, 11)


class KaggricultureActorConfig(BaseConfig):
    action_spec: Literal["kaggriculture"] = "kaggriculture"


class KaggricultureTransformerConfig(BaseConfig):
    model_arch: Literal["kaggriculture_transformer"] = "kaggriculture_transformer"
    embed_dim: int = Field(default=256, ge=1)
    depth: int = Field(default=7, ge=1)
    n_heads: int = Field(default=8, ge=1)
    mlp_ratio: float = Field(default=4.0, gt=0)
    activation: Literal["gelu", "silu", "swiglu"] = "gelu"
    n_scratch_tokens: int = Field(default=4, ge=0)
    force_flash_attn: bool = False
    max_decode_frames: int = Field(default=MAX_FRAMES, ge=1, le=MAX_FRAMES)
    actor: KaggricultureActorConfig = Field(default_factory=KaggricultureActorConfig)
    value_mode: Literal["win_loss", "win_only", "margin"] = "win_loss"
    critic_mode: Literal["independent"] = "independent"
    lora: None = None
    player_count_adapters_enabled: Literal[False] = False
    player_count_adapter_blocks: Literal[0] = 0

    @model_validator(mode="after")
    def _validate_dimensions(self) -> Self:
        if self.embed_dim % self.n_heads:
            raise ValueError("n_heads must evenly divide embed_dim")
        if int(self.embed_dim * self.mlp_ratio) < 1:
            raise ValueError("embed_dim * mlp_ratio must be at least 1")
        return self


@dataclass
class _Encoded:
    own_actors: torch.Tensor
    plan: torch.Tensor
    critic: torch.Tensor
    leading_shape: tuple[int, ...]
    shapes: list[tuple[int, int, int]]
    active: torch.Tensor
    actor_counts: torch.Tensor


def _check_flags(checks: tuple[tuple[torch.Tensor, str], ...]) -> None:
    """Transfer one compact validation vector, not a scalar per condition."""
    flags = torch.stack([flag for flag, _ in checks]).cpu().tolist()
    for flag, (_, message) in zip(flags, checks, strict=True):
        if not flag:
            raise ValueError(message)


def _categorical_choice(log_probs: torch.Tensor) -> torch.Tensor:
    """Exact categorical sampling via Gumbel-max without multinomial checks."""
    noise = torch.empty_like(log_probs).exponential_()
    noise.clamp_min_(torch.finfo(log_probs.dtype).tiny).log_()
    return (log_probs - noise).argmax(dim=-1)


class KaggricultureTransformer(
    BaseModelAPI[KaggricultureObsBatch, KaggricultureActions]
):
    tile_scale: torch.Tensor
    suffix_scale: torch.Tensor
    cell_slots: torch.Tensor
    tile_sides: torch.Tensor
    actor_slots: torch.Tensor
    zero_category: torch.Tensor

    def __init__(
        self,
        config: KaggricultureTransformerConfig,
        *,
        obs_spec: KaggricultureObsConfig,
        action_spec: KaggricultureActionConfig,
    ) -> None:
        super().__init__()
        if obs_spec.observation_version != 2:
            raise ValueError("Kaggriculture transformer requires observation_version=2")
        self.config, self.obs_spec = config, obs_spec
        self.action_spec = action_spec
        self.hire_limit = action_spec.hire_limit
        width = config.embed_dim
        hidden = int(width * config.mlp_ratio)

        def stem(channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, width)
            )

        self.tile_stem = stem(16)  # two legacy scalars plus 14 maintenance fields
        self.tile_kind = nn.Embedding(29, width)
        self.cell = nn.Embedding(100, width)
        self.actor_stem = stem(15)  # present, x, y, and own inventory (zero for rival)
        self.actor_index = nn.Embedding(MAX_ACTORS, width)
        self.roles = nn.Embedding(4, width)  # own/rival tiles and own/rival actors
        self.global_stem = stem(82)  # globals, remaining public fields, suffix, counts
        self.storage_stem = stem(17)
        self.shops_stem = stem(64)
        self.product_stem = stem(5)
        self.product_index = nn.Embedding(12, width)
        self.special_tokens = nn.Parameter(
            torch.empty(config.n_scratch_tokens + 2, width)
        )
        trunk_config = StatelessTransformerV1Config(
            embed_dim=width,
            depth=config.depth,
            n_heads=config.n_heads,
            mlp_ratio=config.mlp_ratio,
            activation=config.activation,
            force_flash_attn=config.force_flash_attn,
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(trunk_config) for _ in range(config.depth)
        )
        self.final_norm = nn.LayerNorm(width)
        self.frame_input = nn.Linear(2 * width, width)
        self.frame_norm = nn.LayerNorm(width)
        self.prefix_norm = nn.LayerNorm(width)
        self.frame_history = nn.GRUCell(width, width)
        self.slot_embeddings = nn.ModuleList(
            nn.Embedding(size, width) for size in SLOT_WIDTHS
        )
        self.heads = nn.ModuleList(nn.Linear(width, size) for size in SLOT_WIDTHS)
        self.critic_head = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1)
        )
        self.register_buffer(
            "tile_scale",
            torch.tensor(
                [
                    1,
                    1,
                    1 / 8,
                    1,
                    1 / 2,
                    1 / 30,
                    1 / 720,
                    1 / 30,
                    1 / 30,
                    1 / 2,
                    1,
                    1,
                    1,
                    1 / 4,
                ]
            ),
            persistent=False,
        )
        self.register_buffer(
            "suffix_scale",
            torch.tensor(
                [
                    1,
                    1 / 720,
                    1 / 30,
                    1 / 24,
                    1 / 720,
                    1 / 24,
                    1 / 240,
                    1,
                    1 / 15,
                    1 / 100,
                    1 / 10,
                ]
            ),
            persistent=False,
        )
        self.register_buffer(
            "cell_slots", torch.arange(100).repeat(2), persistent=False
        )
        self.register_buffer(
            "tile_sides", torch.arange(2).repeat_interleave(100), persistent=False
        )
        self.register_buffer("actor_slots", torch.arange(MAX_ACTORS), persistent=False)
        self.register_buffer(
            "zero_category", torch.arange(max(SLOT_WIDTHS)) == 0, persistent=False
        )
        self._native_sampler: MyolieSampler | None = None
        self._grammar_cache: OrderedDict[
            tuple[str, tuple[tuple[int, int, int], ...]], GrammarBatch
        ] = OrderedDict()
        self._trunk_forward: Callable[
            [torch.Tensor, torch.Tensor | None, PackedSequence | None], torch.Tensor
        ] = self._run_trunk
        self.reset_parameters()

    def get_input_layers(self) -> tuple[InputLayer, ...]:
        return (
            self.tile_stem,
            self.tile_kind,
            self.cell,
            self.actor_stem,
            self.actor_index,
            self.roles,
            self.global_stem,
            self.storage_stem,
            self.shops_stem,
            self.product_stem,
            self.product_index,
            self.special_tokens,
            self.slot_embeddings,
        )

    def get_output_layers(self) -> tuple[nn.Module, ...]:
        return (self.frame_input, self.frame_history, self.heads, self.critic_head)

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=self.config.embed_dim**-0.5)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.special_tokens, std=self.config.embed_dim**-0.5)
        self.frame_history.reset_parameters()
        for block in self.blocks:
            block = cast(TransformerBlock, block)
            gain = 1 / math.sqrt(2 * self.config.depth)
            nn.init.orthogonal_(block.attn.out.weight, gain=gain)
            nn.init.orthogonal_(block.mlp.down.weight, gain=gain)
        for head in self.heads:
            nn.init.orthogonal_(cast(nn.Linear, head).weight, gain=0.01)
        nn.init.orthogonal_(cast(nn.Linear, self.critic_head[-1]).weight, gain=1)

    def _run_trunk(
        self, x: torch.Tensor, mask: torch.Tensor | None, packed: PackedSequence | None
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, mask, packed)
        return self.final_norm(x)

    def compile_transformer_trunk(self, *, mode: str) -> int:
        self._trunk_forward = torch.compile(self._run_trunk, mode=mode, dynamic=True)
        return 1

    def _encode(self, obs: KaggricultureObsBatch) -> _Encoded:
        if obs.features.shape[-2:] != (2, 8176):
            raise ValueError("expected independent feature perspectives [..., 2, 8176]")
        leading = tuple(obs.features.shape[:-1])
        if obs.still_playing.shape != leading or obs.context.shape != (
            *leading[:-1],
            4,
        ):
            raise ValueError(
                "observation context or active-seat shape does not match features"
            )
        features = obs.features.reshape(-1, 8176)
        raw_counts = features[:, 1025:1027]
        counts = raw_counts.long()
        context_tensor = obs.context.reshape(-1, 4)
        context = context_tensor.cpu().tolist()
        shapes = [
            (int(row[1 + seat]), int(row[3]), self.hire_limit)
            for row in context
            for seat in range(2)
        ]
        if any(not 1 <= order_limit <= 10 for _, order_limit, _ in shapes):
            raise ValueError("market-order capacity must be between 1 and 10")
        if any(
            actors + orders + 1 > self.config.max_decode_frames
            for actors, orders, _ in shapes
        ):
            raise ValueError(
                "max_decode_frames cannot represent all actors, market orders and STOP"
            )
        own_pad = rival_pad = max(shape[0] for shape in shapes)
        count = features.shape[0]
        dtype = self.special_tokens.dtype
        old_tiles = features[:, 16:616].reshape(count, 200, 3)
        maintenance = features[:, 1027:3827].reshape(count, 200, 14) * self.tile_scale
        kinds = (old_tiles[..., 0] * 40).round().long()
        public_counts = context_tensor[:, 1:3].to(features.device)
        expected_counts = torch.stack(
            (public_counts, public_counts.flip(-1)), dim=1
        ).reshape(-1, 2)
        _check_flags(
            (
                (
                    (
                        (raw_counts == counts) & (counts >= 1) & (counts <= MAX_ACTORS)
                    ).all(),
                    "observations must contain between 1 and 241 actors per farm",
                ),
                (
                    ((features[:, 1024] == 1) & (features[:, 8165] == 1)).all(),
                    "complete current-observation feature schemas are required",
                ),
                (
                    (raw_counts == expected_counts).all(),
                    "context actor counts disagree with perspective features",
                ),
                (
                    ((kinds >= 0) & (kinds < 29)).all(),
                    "tile kind is outside the observation vocabulary",
                ),
            )
        )
        tiles = (
            self.tile_stem(torch.cat((old_tiles[..., 1:], maintenance), -1).to(dtype))
            + self.tile_kind(kinds)
            + self.cell(self.cell_slots)
            + self.roles(self.tile_sides)
        )
        actors = features[:, 3827:5273].reshape(count, 2, MAX_ACTORS, 3)
        inventory = features[:, 5273:8165].reshape(count, MAX_ACTORS, 12) / 32
        own_present = self.actor_slots[None, :] < counts[:, :1]
        inventory = inventory.masked_fill(~own_present.unsqueeze(-1), 0)
        actor_tokens, actor_masks = [], []
        for side, capacity in enumerate((own_pad, rival_pad)):
            raw = actors[:, side, :capacity].clone()
            raw[..., 1:] = raw[..., 1:] / 10
            held = (
                inventory[:, :capacity]
                if side == 0
                else inventory.new_zeros(count, capacity, 12)
            )
            indices = self.actor_slots[:capacity]
            actor_tokens.append(
                self.actor_stem(torch.cat((raw, held), -1).to(dtype))
                + self.actor_index(indices)
                + self.roles.weight[2 + side]
            )
            actor_masks.append(indices.unsqueeze(0) < counts[:, side : side + 1])
        global_features = torch.cat(
            (
                features[:, :16],
                features[:, 907:960],
                features[:, 8165:] * self.suffix_scale,
                raw_counts / MAX_ACTORS,
            ),
            -1,
        )
        market = F.pad(features[:, 889:907].reshape(count, 2, 9), (0, 3))
        storage = features[:, 872:889]
        seeds = F.pad(storage[:, :5], (0, 7))
        products = torch.stack(
            (
                market[:, 0],
                market[:, 1],
                storage[:, 5:],
                seeds,
                inventory.sum(dim=1),
            ),
            -1,
        )
        world = torch.stack(
            (
                self.global_stem(global_features.to(dtype)),
                self.storage_stem(storage.to(dtype)),
                self.shops_stem(features[:, 960:1024].to(dtype)),
            ),
            dim=1,
        )
        product_tokens = (
            self.product_stem(products.to(dtype)) + self.product_index.weight
        )
        special = self.special_tokens.unsqueeze(0).expand(count, -1, -1)
        x = torch.cat((tiles, *actor_tokens, world, product_tokens, special), dim=1)
        # Embeddings stay FP32 under autocast; attention needs the active mixed
        # precision dtype to select the starter's packed FlashAttention path.
        if torch.is_autocast_enabled(x.device.type):
            x = x.to(torch.get_autocast_dtype(x.device.type))
        fixed_tail = 15 + self.special_tokens.shape[0]
        mask = torch.cat(
            (
                torch.ones(count, 200, dtype=torch.bool, device=x.device),
                *actor_masks,
                torch.ones(count, fixed_tail, dtype=torch.bool, device=x.device),
            ),
            dim=1,
        )
        packed = None
        if use_flash_attn(x):
            x, packed = pack_sequence(x, mask, max_seqlen=x.shape[1])
            x = self._trunk_forward(x, None, packed)
            x = unpack_sequence(x, packed)
        else:
            if self.config.force_flash_attn and x.device.type == "cuda":
                raise RuntimeError(
                    "force_flash_attn requires CUDA fp16/bf16 and flash-attn"
                )
            x = self._trunk_forward(x, mask, None)
        return _Encoded(
            own_actors=x[:, 200 : 200 + own_pad],
            plan=x[:, -2],
            critic=x[:, -1],
            leading_shape=leading,
            shapes=shapes,
            active=obs.still_playing.reshape(-1),
            actor_counts=counts[:, 0],
        )

    def count_non_masked_tokens(self, obs: KaggricultureObsBatch) -> torch.Tensor:
        counts = obs.features[..., 1025:1027].sum(dtype=torch.int64)
        fixed = 217 + self.config.n_scratch_tokens
        return counts + obs.still_playing.numel() * fixed

    def _grammar(self, encoded: _Encoded) -> GrammarBatch:
        key = (str(encoded.plan.device), tuple(sorted(set(encoded.shapes))))
        if key not in self._grammar_cache:
            if self._native_sampler is None:
                self._native_sampler = MyolieSampler()
            for shape in key[1]:
                plan = self._native_sampler.plan(*shape)
                if any(
                    slot in _FORCED_SLOTS and len(allowed) != 1
                    for _, allowed, _, _, slot in plan.nodes
                ):
                    raise ValueError("native forced-slot support must be singleton")
            self._grammar_cache[key] = GrammarBatch(
                self._native_sampler, key[1], encoded.plan.device
            )
            if len(self._grammar_cache) > 4:
                self._grammar_cache.popitem(last=False)
        self._grammar_cache.move_to_end(key)
        grammar = self._grammar_cache[key]
        if grammar.shapes != tuple(encoded.shapes):
            grammar = grammar.rebind(encoded.shapes)
            self._grammar_cache[key] = grammar
        return grammar

    def _decoder_inputs(self, encoded: _Encoded) -> tuple[torch.Tensor, torch.Tensor]:
        """Project the fixed observation once, before the action-prefix loop."""
        width = encoded.plan.shape[-1]
        weight = self.frame_input.weight
        plan = F.linear(encoded.plan, weight[:, :width], self.frame_input.bias)
        actor = F.linear(encoded.own_actors, weight[:, width:]) + plan[:, None, :]
        market = F.linear(encoded.plan, weight[:, width:]) + plan
        return actor, market

    def _decode(
        self,
        encoded: _Encoded,
        *,
        actions: KaggricultureActions | None,
        deterministic: bool,
    ) -> tuple[KaggricultureActions, ModelActionLogProbs, ModelActionEntropies]:
        grammar = self._grammar(encoded)
        count, width = encoded.plan.shape
        device = encoded.plan.device
        nodes = torch.where(encoded.active, grammar.starts, -1)
        counts = encoded.actor_counts
        lengths = torch.zeros(count, dtype=torch.int64, device=device)
        expected_tokens: torch.Tensor | None = None
        expected_lengths: torch.Tensor | None = None
        if actions is not None:
            if actions.tokens.shape != (*encoded.leading_shape, MAX_FRAMES, 12):
                raise ValueError("actions must have shape [..., 2, 252, 12]")
            if actions.lengths.shape != encoded.leading_shape:
                raise ValueError("action lengths do not match observation perspectives")
            if (
                actions.tokens.dtype != torch.int64
                or actions.lengths.dtype != torch.int64
            ):
                raise ValueError("grammar action tokens and lengths must use int64")
            expected_tokens = actions.tokens.reshape(count, MAX_FRAMES, 12)
            expected_lengths = actions.lengths.reshape(count)
            _check_flags(
                (
                    (
                        (
                            (expected_lengths >= 0) & (expected_lengths <= MAX_FRAMES)
                        ).all(),
                        "action lengths are outside frame capacity",
                    ),
                    (
                        (expected_lengths.gt(0) == encoded.active).all(),
                        "only active seats must have a nonempty action program",
                    ),
                )
            )
        frames, frame_logp, frame_entropy = [], [], []
        history = encoded.plan.new_zeros(count, width)
        last_frame = max(actors + orders + 1 for actors, orders, _ in encoded.shapes)
        if expected_lengths is not None:
            last_frame = int(expected_lengths.max().item())
        valid_choices = torch.ones(count, dtype=torch.bool, device=device)
        row_indices = torch.arange(count, device=device)
        actor_inputs, market_input = self._decoder_inputs(encoded)
        # These slots always have singleton native support. Keep their existing
        # checkpoint parameters and DDP hooks while skipping useless logits/RNG.
        forced_zeros = (
            {
                slot: (
                    cast(nn.Linear, self.heads[slot]).weight.sum()
                    + cast(nn.Linear, self.heads[slot]).bias.sum()
                )
                * 0
                for slot in _FORCED_SLOTS
            }
            if torch.is_grad_enabled()
            else {
                slot: encoded.plan.new_zeros((), dtype=torch.float32)
                for slot in _FORCED_SLOTS
            }
        )
        for ordinal in range(last_frame):
            live = nodes >= 0
            if expected_tokens is None and not bool(live.any()):
                break
            source = actor_inputs[:, min(ordinal, actor_inputs.shape[1] - 1)]
            source = torch.where((ordinal < counts).unsqueeze(-1), source, market_input)
            base = self.frame_norm(source + history)
            prefix = torch.zeros_like(base)
            tokens, logps, entropies = [], [], []
            for slot, (head, embedding) in enumerate(
                zip(self.heads, self.slot_embeddings, strict=True)
            ):
                mask = grammar.options(nodes, slot)
                original_choice: torch.Tensor | None = None
                if expected_tokens is not None:
                    original_choice = expected_tokens[:, ordinal, slot]
                    in_range = (original_choice >= 0) & (
                        original_choice < SLOT_WIDTHS[slot]
                    )
                    choice = original_choice.clamp(0, SLOT_WIDTHS[slot] - 1)
                    valid_choices = valid_choices & in_range & mask[row_indices, choice]
                    # Invalid prefixes still fail below; a safe dummy support
                    # prevents NaNs while collecting validation on the device.
                    dummy = self.zero_category[: SLOT_WIDTHS[slot]]
                    mask = mask | (~mask.any(-1, keepdim=True) & dummy)
                if slot in _FORCED_SLOTS:
                    if expected_tokens is None:
                        choice = mask.long().argmax(-1)
                    selected = forced_zeros[slot].expand(count)
                    entropy = selected
                else:
                    logits = head(
                        self.prefix_norm(base + prefix / math.sqrt(slot + 1))
                    ).float()
                    log_probs = F.log_softmax(
                        logits.masked_fill(~mask, -torch.inf), dim=-1
                    )
                    if expected_tokens is None:
                        choice = (
                            log_probs.argmax(-1)
                            if deterministic
                            else _categorical_choice(log_probs)
                        )
                    selected = log_probs.gather(1, choice.unsqueeze(-1)).squeeze(-1)
                    entropy = -(log_probs.exp() * log_probs.masked_fill(~mask, 0)).sum(
                        -1
                    )
                tokens.append(choice)
                logps.append(selected)
                entropies.append(entropy)
                prefix = prefix + embedding(choice)
                nodes = grammar.advance(
                    nodes, choice if original_choice is None else original_choice, slot
                )
            history = self.frame_history(prefix / math.sqrt(12), history)
            frames.append(torch.stack(tokens, -1))
            frame_logp.append(torch.stack(logps, -1))
            frame_entropy.append(torch.stack(entropies, -1))
            lengths = lengths + live.to(torch.int64)
        if frames:
            tokens_tensor = F.pad(
                torch.stack(frames, 1), (0, 0, 0, MAX_FRAMES - len(frames))
            )
            logp = F.pad(
                torch.stack(frame_logp, 1), (0, 0, 0, MAX_FRAMES - len(frames))
            )
            entropy = F.pad(
                torch.stack(frame_entropy, 1), (0, 0, 0, MAX_FRAMES - len(frames))
            )
        else:
            tokens_tensor = torch.zeros(
                count, MAX_FRAMES, 12, dtype=torch.int64, device=device
            )
            logp = encoded.plan.new_zeros(count, MAX_FRAMES, 12)
            if torch.is_grad_enabled():
                decoder_modules = (
                    self.frame_input,
                    self.frame_norm,
                    self.prefix_norm,
                    self.frame_history,
                    self.slot_embeddings,
                    self.heads,
                )
                # A wholly inactive rank still participates in every DDP hook.
                for module in decoder_modules:
                    for parameter in module.parameters():
                        logp = logp + parameter.sum() * 0
            entropy = torch.zeros_like(logp)
        checks = [
            (valid_choices.all(), "action violates native grammar"),
            (
                (nodes == -1).all(),
                "action program did not terminate within its representational capacity",
            ),
        ]
        if expected_lengths is not None:
            checks.append(
                (
                    (lengths == expected_lengths).all(),
                    "action lengths disagree with native grammar STOP positions",
                )
            )
        if expected_tokens is not None:
            checks.append(
                (
                    (tokens_tensor == expected_tokens).all(),
                    "action tokens after STOP must be zero padding",
                )
            )
        _check_flags(tuple(checks))
        shape = (*encoded.leading_shape, MAX_FRAMES)
        logp, entropy = logp.reshape(*shape, 12), entropy.reshape(*shape, 12)
        joint_logp, joint_entropy = logp.sum(-1), entropy.sum(-1)
        return (
            KaggricultureActions(
                tokens=tokens_tensor.reshape(*shape, 12),
                lengths=lengths.reshape(encoded.leading_shape),
            ),
            ModelActionLogProbs(
                launch=torch.zeros_like(joint_logp),
                event=logp,
                per_player_entity=joint_logp,
            ),
            ModelActionEntropies(
                launch=torch.zeros_like(joint_entropy),
                event=entropy,
                per_player_entity=joint_entropy,
                components={
                    name: entropy[..., slot] for slot, name in enumerate(SLOT_NAMES)
                },
            ),
        )

    def _values(self, encoded: _Encoded) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.critic_head(encoded.critic).float().squeeze(-1)
        if self.config.value_mode == "margin":
            # Bank margin is unbounded. This auxiliary sigmoid is only API
            # diagnostic compatibility, not a calibrated win probability.
            values, probabilities = logits, logits.sigmoid()
        elif self.config.value_mode == "win_only":
            values = probabilities = logits.sigmoid()
        else:
            values = logits.tanh()
            probabilities = (values + 1) / 2
        return (
            values.reshape(encoded.leading_shape),
            probabilities.reshape(encoded.leading_shape),
        )

    @staticmethod
    def _require_stateless(hidden_state: ModelHiddenState | None) -> None:
        if hidden_state is not None:
            raise ValueError(
                "Kaggriculture model carries no state between observations"
            )

    def forward(
        self,
        obs: KaggricultureObsBatch,
        *,
        deterministic: bool = False,
        hidden_state: ModelHiddenState | None = None,
    ) -> ModelOutput:
        self._require_stateless(hidden_state)
        encoded = self._encode(obs)
        actions, logp, entropy = self._decode(
            encoded, actions=None, deterministic=deterministic
        )
        values, probabilities = self._values(encoded)
        return ModelOutput(actions, logp, entropy, values, probabilities)

    def evaluate_actions(
        self,
        obs: KaggricultureObsBatch,
        actions: KaggricultureActions,
        *,
        hidden_state: ModelHiddenState | None = None,
        dones: torch.Tensor | None = None,
    ) -> ModelEvaluation:
        del dones
        self._require_stateless(hidden_state)
        encoded = self._encode(obs)
        _, logp, entropy = self._decode(encoded, actions=actions, deterministic=False)
        values, probabilities = self._values(encoded)
        return ModelEvaluation(logp, entropy, values, probabilities)

    def compute_value(
        self,
        obs: KaggricultureObsBatch,
        *,
        hidden_state: ModelHiddenState | None = None,
    ) -> torch.Tensor:
        self._require_stateless(hidden_state)
        values, _ = self._values(self._encode(obs))
        return values
