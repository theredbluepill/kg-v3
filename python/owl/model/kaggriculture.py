"""Kaggriculture model on Isaiah's ``StatelessTransformerV1`` layer topology.

The encoder reuses Isaiah's classes unchanged: ``ObservationInputStem`` per
entity group, separate learned per-role tokens, ``TransformerBlock`` trunk with a
final ``LayerNorm``, his flash-packing path, compile hook and initialization.
Only the game I/O differs: stem input widths (float channels plus one-hot
categorical fields from ``docs/kaggriculture-contract.md``) and the grammar
action heads (``owl.model.kaggriculture_actor``) behind Isaiah's ``3D -> D``
actor input projection. Each seat row is encoded and decided independently.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal, Self, TypeVar

import torch
import torch.nn.functional as F
from pydantic import (
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)
from torch import nn

from owl.config import BaseConfig
from owl.kaggriculture import types as kt
from owl.kaggriculture.gpu_grammar import GrammarTables, native_grammar_tables
from owl.model.actor.common import OutputProjectionMLP
from owl.model.attn import use_flash_attn
from owl.model.base import (
    BaseModelAPI,
    InputLayer,
    ModelActionEntropies,
    ModelActionKLDivergences,
    ModelActionLogProbs,
    ModelEvaluation,
    ModelHiddenState,
    ModelOutput,
    ModelTeacherEvaluation,
    TrunkCompileAPI,
)
from owl.model.compile_gemm import (
    claim_gemm_backends,
    require_compiled_gemm_backends,
)
from owl.model.kaggriculture_actor import (
    POLICY_SLOTS,
    GrammarContext,
    GrammarPolicyResult,
    KaggricultureGrammarActor,
    check_replay_flags,
    sample_policy_exponentials,
)
from owl.model.kaggriculture_teacher import (
    GrammarSignature,
    KaggricultureTeacherTargets,
    slot_frames,
)
from owl.model.stateless_transformer_v1 import (
    _ACTOR_HEAD_INIT_GAIN,
    _CRITIC_HEAD_INIT_GAIN,
    ObservationInputStem,
    PackedSequence,
    StatelessTransformerV1Config,
    TransformerBlock,
    _expand_tokens,
    _init_input_layer,
    _init_linear,
    _init_module,
    _requires_flash_attn,
    pack_sequence,
    pack_sequence_from_cpu_mask,
    unpack_sequence,
)
from owl.model.teacher_targets import TeacherTargets

_T = TypeVar("_T")

KAGGRICULTURE_TRANSFORMER: Literal["kaggriculture_transformer"] = (
    "kaggriculture_transformer"
)

# Compiled mm/addmm templates in torch 2.9 can form the A-load offset in int32
# (measured: corruption once rows x input width > 2**31). Backward reads forward
# outputs as inputs, so compiled regions keep rows x max(in, out) < 2**31
# (cookbook reference compiled-gemm-template-overflows-above-2-21-rows).
_GEMM_ELEMENT_LIMIT = 2**31
_OVERFLOW_REFERENCE = (
    "cookbook/references/compiled-gemm-template-overflows-above-2-21-rows.md"
)

# Every compiled Kaggriculture region lowers its GEMMs to extern cuBLAS only, so
# Inductor never emits the Triton mm/addmm/bmm (or decompose-K, persistent-TMA)
# templates that wrapped the int32 A-load offset. ``configure_model_compile``
# sets it before compiling; the model refuses to compile, or to call a compiled
# region, under any other value (cookbook decision
# kaggriculture-compiles-gemms-with-cublas-only).
_TILE_STEM_WIDTH = (
    kt.TILE_FLOAT_CHANNELS
    + len(kt.TILE_KINDS)
    + len(kt.CROPS)
    + len(kt.ANIMALS)
    + kt.CELLS
    + len(kt.TILE_ROLES)
)
_ACTOR_STEM_WIDTH = (
    kt.ACTOR_FLOAT_CHANNELS + kt.MAX_ACTORS + kt.CELLS + len(kt.ACTOR_ROLES)
)
_SHOP_STEM_WIDTH = len(kt.SHOP_TYPES) + kt.SHOP_SLOTS
_MARKET_STEM_WIDTH = kt.MARKET_FLOAT_CHANNELS + kt.PRODUCT_COUNT
_ENTITY_TOKENS = kt.ACTOR_SLOTS + kt.TILES + kt.SHOP_SLOTS + kt.PRODUCT_COUNT
_PLAN_TOKENS = 1


class KaggricultureTransformerConfig(BaseConfig):
    model_arch: Literal["kaggriculture_transformer"] = KAGGRICULTURE_TRANSFORMER
    embed_dim: int = Field(default=256, ge=1)
    depth: int = Field(default=8, ge=1)
    n_heads: int = Field(default=8, ge=1)
    mlp_ratio: float = Field(default=2.0, gt=0.0)
    activation: Literal["gelu", "silu"] = "gelu"
    n_scratch_tokens: int = Field(default=4, ge=0)
    force_flash_attn: bool = False
    # Per-seat critic offset (owner, 2026-09-30: "per-player critic might be
    # the way out?"): V = 2 p(self) - 1 + o(own critic token), with o's output
    # layer zero-initialised. Off (the default) builds no head and dumps as
    # before the fields existed.
    critic_offset: bool = False
    # The offset head reads the critic token detached, so its value-loss
    # gradient never reaches the trunk. Requires critic_offset.
    critic_offset_detach_trunk: bool = False

    @model_validator(mode="after")
    def _validate_critic_offset(self) -> Self:
        if self.critic_offset_detach_trunk and not self.critic_offset:
            raise ValueError(
                "model.critic_offset_detach_trunk requires model.critic_offset"
            )
        return self

    @model_serializer(mode="wrap")
    def _omit_absent_critic_offset(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        # Configs without the head dump exactly as before the fields existed,
        # so their config.yaml and v3/config_sha256 identity are unchanged.
        data: dict[str, Any] = handler(self)
        if not self.critic_offset:
            del data["critic_offset"]
            del data["critic_offset_detach_trunk"]
        return data

    def trunk_config(self) -> StatelessTransformerV1Config:
        """Isaiah's typed trunk config built from the shared fields."""
        return StatelessTransformerV1Config(
            embed_dim=self.embed_dim,
            depth=self.depth,
            n_heads=self.n_heads,
            mlp_ratio=self.mlp_ratio,
            activation=self.activation,
            n_scratch_tokens=self.n_scratch_tokens,
            force_flash_attn=self.force_flash_attn,
        )


@dataclass(frozen=True)
class KaggricultureEncoded:
    hidden: torch.Tensor
    token_mask: torch.Tensor
    own_actor_hidden: torch.Tensor
    player_hidden: torch.Tensor
    global_hidden: torch.Tensor
    board_hidden: torch.Tensor
    actor_plan_hidden: torch.Tensor
    critic_value_hidden: torch.Tensor
    critic_value_mask: torch.Tensor


def sequence_length(config: KaggricultureTransformerConfig) -> int:
    """Tokens per seat row: entities, players, global, board, plan, critic."""
    return (
        _ENTITY_TOKENS
        + kt.PLAYERS
        + 1
        + config.n_scratch_tokens
        + _PLAN_TOKENS
        + kt.PLAYERS
    )


def trunk_gemm_width(config: KaggricultureTransformerConfig) -> int:
    """Max GEMM operand/output width in the compiled trunk.

    The maximum of ``max(in_features, out_features)`` over every ``nn.Linear``
    in a ``TransformerBlock``: ``D -> D`` attention projections and the
    ``D -> H -> D`` MLP (``H = int(D * mlp_ratio)``). A test enumerates the
    blocks' Linear layers so a new block layout cannot silently exceed it.
    """
    return max(config.embed_dim, int(config.embed_dim * config.mlp_ratio))


def rows_per_chunk(*, tokens: int, width: int) -> int:
    """Largest padded row count whose GEMMs stay below the overflow limit."""
    return (_GEMM_ELEMENT_LIMIT - 1) // (tokens * width)


def packed_row_chunks(row_tokens: list[int], *, width: int) -> list[tuple[int, int]]:
    """Split rows at row boundaries so each chunk's packed tokens x width < 2**31.

    Greedy in row order; raises only when a single row cannot fit.
    """
    chunks: list[tuple[int, int]] = []
    start, packed = 0, 0
    for row, tokens in enumerate(row_tokens):
        if tokens * width >= _GEMM_ELEMENT_LIMIT:
            raise ValueError(
                f"row {row} packs {tokens} tokens x {width}, at or above the 2**31 "
                f"GEMM limit; see {_OVERFLOW_REFERENCE}"
            )
        if (packed + tokens) * width >= _GEMM_ELEMENT_LIMIT:
            chunks.append((start, row))
            start, packed = row, 0
        packed += tokens
    if row_tokens:
        chunks.append((start, len(row_tokens)))
    return chunks


def head_gemm_width(config: KaggricultureTransformerConfig) -> int:
    """Max GEMM operand/output width in the action heads.

    ``3D -> D`` input projection, ``D -> D`` head ``up`` and ``D -> width``
    head ``out``.
    """
    widest_head = max(kt.SLOT_WIDTHS[slot] for slot in POLICY_SLOTS)
    return max(3 * config.embed_dim, config.embed_dim, widest_head)


def head_rows_per_chunk(config: KaggricultureTransformerConfig) -> int:
    """Largest row count with rows x MAX_FRAMES x head width < 2**31.

    The heads run eager in production; the chunking keeps a future compile of
    the head core from silently corrupting outputs (about 11,096 rows at D=256).
    """
    rows = (_GEMM_ELEMENT_LIMIT - 1) // (kt.MAX_FRAMES * head_gemm_width(config))
    if rows <= 0:
        raise ValueError(
            f"one row of {kt.MAX_FRAMES} frames x {head_gemm_width(config)} reaches "
            f"the 2**31 GEMM limit; see {_OVERFLOW_REFERENCE}"
        )
    return rows


def _one_hot(index: torch.Tensor, classes: int, like: torch.Tensor) -> torch.Tensor:
    return F.one_hot(index, classes).to(dtype=like.dtype)


class KaggricultureTransformer(
    BaseModelAPI[
        kt.KaggricultureObsBatch,
        kt.KaggricultureActions,
        kt.KaggricultureActionConfig,
    ],
    TrunkCompileAPI,
):
    def __init__(
        self,
        config: KaggricultureTransformerConfig,
        *,
        obs_spec: kt.KaggricultureObsConfig,
        action_spec: kt.KaggricultureActionConfig,
        grammar_tables: GrammarTables | None = None,
    ) -> None:
        """Build the encoder, critic and grammar action heads.

        ``grammar_tables`` defaults to the validated native tables. They are
        loaded once, then move with the model as non-persistent actor buffers.
        """
        super().__init__()
        self.config = config
        self.obs_spec = obs_spec
        self.action_spec = action_spec
        trunk = config.trunk_config()
        width = config.embed_dim
        self.tile_proj = ObservationInputStem(_TILE_STEM_WIDTH, trunk)
        self.actor_proj = ObservationInputStem(_ACTOR_STEM_WIDTH, trunk)
        self.shop_proj = ObservationInputStem(_SHOP_STEM_WIDTH, trunk)
        self.market_proj = ObservationInputStem(_MARKET_STEM_WIDTH, trunk)
        self.player_feature_proj = ObservationInputStem(
            kt.PLAYER_FEATURE_CHANNELS, trunk
        )
        self.global_proj = ObservationInputStem(kt.GLOBAL_FEATURE_CHANNELS, trunk)
        self.player_tokens = nn.Parameter(torch.empty(kt.PLAYERS, width))
        self.board_tokens = nn.Parameter(torch.empty(config.n_scratch_tokens, width))
        self.actor_plan_tokens = nn.Parameter(torch.empty(_PLAN_TOKENS, width))
        self.critic_value_tokens = nn.Parameter(torch.empty(kt.PLAYERS, width))
        self.blocks = nn.ModuleList(
            TransformerBlock(trunk) for _ in range(config.depth)
        )
        self.final_norm = nn.LayerNorm(width)
        # Isaiah's critic: one logit per critic-value token (self, opponent).
        self.critic_head = OutputProjectionMLP(trunk, 1)
        # Isaiah's actor input projection over [entity | self player | plan];
        # the grammar heads are the only game-specific actor part.
        self.actor_input_proj = nn.Linear(3 * width, width)
        self.actor = KaggricultureGrammarActor(
            trunk,
            native_grammar_tables() if grammar_tables is None else grammar_tables,
        )
        self._compiled_transformer_trunk: (
            Callable[
                [torch.Tensor, torch.Tensor | None, PackedSequence | None],
                torch.Tensor,
            ]
            | None
        ) = None
        # Opt-in compiled heads; eager remains the default.
        self._compiled_actor_core: Callable[..., GrammarPolicyResult] | None = None
        self._rollout_token_mask_cpu: torch.Tensor | None = None
        # True once a trunk or actor region is compiled;
        # every later compiled dispatch re-checks GEMM backends, because Inductor
        # compiles lazily and recompiles on new dynamic shapes.
        self.compiled_regions_require_gemm_backends = False
        # Per-seat critic offset head (model.critic_offset), registered last.
        # It reads only each row's own (self) critic-value token, so it sees the
        # row's current observation and nothing else; its output layer starts
        # at zero, so the value equals the winner critic's until it trains.
        self.critic_offset_head: OutputProjectionMLP | None = (
            OutputProjectionMLP(trunk, 1) if config.critic_offset else None
        )
        self.reset_parameters()

    # --- Isaiah's parameter conventions ---------------------------------------

    def reset_parameters(self) -> None:
        self.apply(_init_module)
        for layer in self.get_input_layers():
            _init_input_layer(layer)
        residual_gain = 1.0 / math.sqrt(2.0 * self.config.depth)
        for module in self.blocks:
            assert isinstance(module, TransformerBlock)
            _init_linear(module.attn.out, gain=residual_gain)
            _init_linear(module.mlp.down, gain=residual_gain)
        for layer in self.get_output_layers():
            assert isinstance(layer, nn.Linear)
            if (
                self.critic_offset_head is not None
                and layer is self.critic_offset_head.out
            ):
                continue
            gain = (
                _CRITIC_HEAD_INIT_GAIN
                if layer is self.critic_head.out
                else _ACTOR_HEAD_INIT_GAIN
            )
            _init_linear(layer, gain=gain)
        self.zero_critic_offset_output()

    def zero_critic_offset_output(self) -> None:
        """Zero the offset head's output layer, so every offset is exactly 0.

        The hidden layer keeps its initialization, so the head can train (an
        all-zero head would have no gradient). No-op without the head.
        """
        head = self.critic_offset_head
        if head is None:
            return
        with torch.no_grad():
            head.out.weight.zero_()
            _require(head.out.bias).zero_()

    def optional_state_keys(self) -> frozenset[str]:
        """The ``critic_offset_head.*`` keys, which a checkpoint may omit.

        A checkpoint from before the head (BC best, a flag-off run) loads into
        a model with the head only if it omits all of them; the loader then
        calls ``reset_optional_state``, so the offset starts at exactly 0.
        """
        head = self.critic_offset_head
        if head is None:
            return frozenset()
        return frozenset(f"critic_offset_head.{key}" for key in head.state_dict())

    def reset_optional_state(self) -> None:
        self.zero_critic_offset_output()

    def get_input_layers(self) -> tuple[InputLayer, ...]:
        return (
            self.tile_proj.input,
            self.actor_proj.input,
            self.shop_proj.input,
            self.market_proj.input,
            self.player_feature_proj.input,
            self.global_proj.input,
            self.player_tokens,
            self.board_tokens,
            self.actor_plan_tokens,
            self.critic_value_tokens,
            *self.actor.get_input_layers(),
        )

    def get_output_layers(self) -> tuple[nn.Module, ...]:
        layers = (self.critic_head.out, *self.actor.get_output_layers())
        if self.critic_offset_head is None:
            return layers
        # Excluded from Muon and int8 like every head output; reset_parameters
        # zeroes it instead of applying a gain.
        return (*layers, self.critic_offset_head.out)

    # --- encoder ---------------------------------------------------------------

    def count_non_masked_tokens(self, obs: kt.KaggricultureObsBatch) -> torch.Tensor:
        rows = obs.still_playing.numel()
        playing = obs.still_playing.sum(dtype=torch.int64)
        always_on = rows * (
            kt.TILES + kt.PRODUCT_COUNT + 1 + self.config.n_scratch_tokens
        )
        return (
            obs.actor_mask.sum(dtype=torch.int64)
            + obs.shop_mask.sum(dtype=torch.int64)
            + (kt.PLAYERS + _PLAN_TOKENS + kt.PLAYERS) * playing
            + always_on
        )

    def encode_observations(
        self, obs: kt.KaggricultureObsBatch
    ) -> KaggricultureEncoded:
        x, token_mask = self._assemble_tokens(obs)
        x = self._run_trunk(x, token_mask)
        x = x.masked_fill(~token_mask.unsqueeze(-1), 0.0)
        player_start = _ENTITY_TOKENS
        global_start = player_start + kt.PLAYERS
        board_start = global_start + 1
        plan_start = board_start + self.config.n_scratch_tokens
        critic_start = plan_start + _PLAN_TOKENS
        return KaggricultureEncoded(
            hidden=x,
            token_mask=token_mask,
            own_actor_hidden=x[:, : kt.MAX_ACTORS],
            player_hidden=x[:, player_start:global_start],
            global_hidden=x[:, global_start:board_start],
            board_hidden=x[:, board_start:plan_start],
            actor_plan_hidden=x[:, plan_start:critic_start],
            critic_value_hidden=x[:, critic_start : critic_start + kt.PLAYERS],
            critic_value_mask=token_mask[:, critic_start : critic_start + kt.PLAYERS],
        )

    def _assemble_tokens(
        self, obs: kt.KaggricultureObsBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = obs.still_playing.numel()
        tiles_float = obs.tiles_float.reshape(rows, kt.TILES, -1)
        tile_in = torch.cat(
            (
                tiles_float,
                _one_hot(
                    obs.tile_kind.reshape(rows, -1), len(kt.TILE_KINDS), tiles_float
                ),
                _one_hot(obs.tile_crop.reshape(rows, -1), len(kt.CROPS), tiles_float),
                _one_hot(
                    obs.tile_animal.reshape(rows, -1), len(kt.ANIMALS), tiles_float
                ),
                _one_hot(obs.tile_cell.reshape(rows, -1), kt.CELLS, tiles_float),
                _one_hot(
                    obs.tile_role.reshape(rows, -1), len(kt.TILE_ROLES), tiles_float
                ),
            ),
            dim=-1,
        )
        actors_float = obs.actors_float.reshape(rows, kt.ACTOR_SLOTS, -1)
        actor_in = torch.cat(
            (
                actors_float,
                _one_hot(obs.actor_slot.reshape(rows, -1), kt.MAX_ACTORS, actors_float),
                _one_hot(obs.actor_cell.reshape(rows, -1), kt.CELLS, actors_float),
                _one_hot(
                    obs.actor_role.reshape(rows, -1), len(kt.ACTOR_ROLES), actors_float
                ),
            ),
            dim=-1,
        )
        shop_in = torch.cat(
            (
                _one_hot(
                    obs.shop_type.reshape(rows, -1), len(kt.SHOP_TYPES), actors_float
                ),
                _one_hot(obs.shop_slot.reshape(rows, -1), kt.SHOP_SLOTS, actors_float),
            ),
            dim=-1,
        )
        market_float = obs.market_float.reshape(rows, kt.PRODUCT_COUNT, -1)
        market_in = torch.cat(
            (
                market_float,
                _one_hot(
                    obs.market_product.reshape(rows, -1), kt.PRODUCT_COUNT, market_float
                ),
            ),
            dim=-1,
        )
        global_x = self.global_proj(obs.global_features.reshape(rows, -1)).unsqueeze(1)
        dtype = global_x.dtype
        players = _expand_tokens(self.player_tokens, rows, dtype=dtype) + (
            self.player_feature_proj(obs.player_features.reshape(rows, kt.PLAYERS, -1))
        )
        x = torch.cat(
            (
                self.actor_proj(actor_in),
                self.tile_proj(tile_in),
                self.shop_proj(shop_in),
                self.market_proj(market_in),
                players,
                global_x,
                _expand_tokens(self.board_tokens, rows, dtype=dtype),
                _expand_tokens(self.actor_plan_tokens, rows, dtype=dtype),
                _expand_tokens(self.critic_value_tokens, rows, dtype=dtype),
            ),
            dim=1,
        )
        return x, self._observation_token_mask(obs)

    def _observation_token_mask(self, obs: kt.KaggricultureObsBatch) -> torch.Tensor:
        rows = obs.still_playing.numel()
        playing = obs.still_playing.reshape(rows, 1)
        device = obs.still_playing.device
        return torch.cat(
            (
                obs.actor_mask.reshape(rows, -1),
                torch.ones((rows, kt.TILES), dtype=torch.bool, device=device),
                obs.shop_mask.reshape(rows, -1),
                torch.ones((rows, kt.PRODUCT_COUNT), dtype=torch.bool, device=device),
                playing.expand(rows, kt.PLAYERS),
                torch.ones(
                    (rows, 1 + self.config.n_scratch_tokens),
                    dtype=torch.bool,
                    device=device,
                ),
                playing,
                playing.expand(rows, kt.PLAYERS),
            ),
            dim=1,
        )

    @contextmanager
    def rollout_packing(
        self,
        obs_host: kt.KaggricultureObsBatch,
        *,
        learner_mask: torch.Tensor | None = None,
    ) -> Iterator[None]:
        """Scope one rollout observation's host mask to its model forward.

        The CPU mask owns a fresh concatenation, so the native environment may
        reuse its observation buffers after this forward. Selection matches
        ``forward_learner_rows``' flattened seat order. No mask survives exit,
        including exceptions; update and teacher forwards keep normal packing.
        """
        if self._rollout_token_mask_cpu is not None:
            raise RuntimeError("rollout_packing contexts cannot be nested")
        if any(
            field.device.type != "cpu"
            for field in (
                obs_host.actor_mask,
                obs_host.shop_mask,
                obs_host.still_playing,
            )
        ):
            raise ValueError("rollout_packing requires CPU observation masks")
        mask = self._observation_token_mask(obs_host)
        if learner_mask is not None:
            if learner_mask.device.type != "cpu" or learner_mask.dtype != torch.bool:
                raise ValueError("rollout_packing learner_mask must be CPU bool")
            if learner_mask.shape != obs_host.still_playing.shape:
                raise ValueError("rollout_packing learner_mask shape must match seats")
            mask = mask[learner_mask.reshape(-1)]
        self._rollout_token_mask_cpu = mask
        try:
            yield
        finally:
            self._rollout_token_mask_cpu = None

    def _run_trunk(self, x: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        """Isaiah's dispatch, guarded against the compiled-GEMM overflow."""
        should_use_flash = use_flash_attn(x)
        if (
            _requires_flash_attn(x, force_flash_attn=self.config.force_flash_attn)
            and not should_use_flash
        ):
            raise RuntimeError(
                "force_flash_attn=True requires CUDA fp16/bf16 tensors "
                "and the flash-attn package"
            )
        if self.compiled_regions_require_gemm_backends:
            require_compiled_gemm_backends()
        trunk = self._compiled_transformer_trunk or self._forward_transformer_trunk
        width = trunk_gemm_width(self.config)
        if should_use_flash:
            return self._run_packed_trunk(trunk, x, token_mask, width)
        chunk = rows_per_chunk(tokens=x.shape[1], width=width)
        if chunk <= 0:
            raise ValueError(
                f"a single padded row of {x.shape[1]} tokens x {width} reaches the "
                f"2**31 GEMM limit; see {_OVERFLOW_REFERENCE}"
            )
        if x.shape[0] <= chunk:
            return trunk(x, token_mask, None)
        return torch.cat(
            [
                trunk(x[start : start + chunk], token_mask[start : start + chunk], None)
                for start in range(0, x.shape[0], chunk)
            ],
            dim=0,
        )

    def _run_packed_trunk(
        self,
        trunk: Callable[
            [torch.Tensor, torch.Tensor | None, PackedSequence | None], torch.Tensor
        ],
        x: torch.Tensor,
        token_mask: torch.Tensor,
        width: int,
    ) -> torch.Tensor:
        """Packed dispatch, chunked at row boundaries below the GEMM limit.

        The padded bound needs no sync; only a batch whose padded size could
        overflow transfers its per-row token counts to plan the chunks.
        """
        rows, tokens = x.shape[:2]
        host_mask = self._rollout_token_mask_cpu
        if host_mask is not None:
            if torch.is_grad_enabled():
                raise RuntimeError("rollout_packing requires a no-grad rollout forward")
            if host_mask.shape != token_mask.shape:
                raise ValueError("rollout_packing host mask shape differs from forward")
            if rows * tokens * width < _GEMM_ELEMENT_LIMIT:
                packed_x, packed = pack_sequence_from_cpu_mask(x, host_mask)
                return unpack_sequence(trunk(packed_x, None, packed), packed)
            row_tokens = [int(n) for n in host_mask.sum(dim=1).tolist()]
            outputs = []
            for start, stop in packed_row_chunks(row_tokens, width=width):
                packed_x, packed = pack_sequence_from_cpu_mask(
                    x[start:stop], host_mask[start:stop]
                )
                outputs.append(unpack_sequence(trunk(packed_x, None, packed), packed))
            return torch.cat(outputs, dim=0)
        if rows * tokens * width < _GEMM_ELEMENT_LIMIT:
            packed_x, packed = pack_sequence(x, token_mask, max_seqlen=tokens)
            return unpack_sequence(trunk(packed_x, None, packed), packed)
        row_tokens = [int(n) for n in token_mask.sum(dim=1).tolist()]
        outputs = []
        for start, stop in packed_row_chunks(row_tokens, width=width):
            packed_x, packed = pack_sequence(
                x[start:stop], token_mask[start:stop], max_seqlen=tokens
            )
            outputs.append(unpack_sequence(trunk(packed_x, None, packed), packed))
        return torch.cat(outputs, dim=0)

    def _forward_transformer_trunk(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor | None,
        packed: PackedSequence | None,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, token_mask, packed)
        return self.final_norm(x)

    def compile_transformer_trunk(self, *, mode: str) -> int:
        """Compile the blocks and final norm only; ``_run_trunk`` calls it.

        The overflow guard and chunking in ``_run_trunk`` stay in front of the
        compiled callable; this does not compile stems, heads or critic. Called directly
        or through ``configure_model_compile``, it first claims the process's
        GEMM backends for Kaggriculture: the probed-stack check, then cuBLAS
        only (``COMPILED_GEMM_BACKENDS``); an Orbit claim raises.
        """
        claim_gemm_backends("kaggriculture")
        self._compiled_transformer_trunk = torch.compile(
            self._forward_transformer_trunk, mode=mode, dynamic=True
        )
        self.compiled_regions_require_gemm_backends = True
        return 1

    def compile_actor_heads(self, *, mode: str) -> int:
        """Compile the grammar core, retaining eager RNG and head chunk guards.

        The process claim and per-call check cover heads even with an eager
        trunk. Static shapes suit rollout seat batches; other batch sizes and
        replay/teacher modes specialize separately. Random exponential draws
        stay eager to preserve sampling call order and RNG state.
        """
        if mode not in ("default", "max-autotune-no-cudagraphs"):
            raise ValueError(
                "actor heads require compile mode 'default' or "
                "'max-autotune-no-cudagraphs'; CUDA graph modes are not qualified"
            )
        claim_gemm_backends("kaggriculture")
        self._compiled_actor_core = torch.compile(
            self.actor.policy_core, mode=mode, fullgraph=True, dynamic=False
        )
        self.compiled_regions_require_gemm_backends = True
        return 1

    # --- policy / value API (Tasks 2.2 and 2.3) --------------------------------

    @staticmethod
    def _require_stateless(hidden_state: ModelHiddenState | None) -> None:
        if hidden_state is not None:
            raise ValueError(
                "KaggricultureTransformer is stateless; hidden_state must be None"
            )

    def forward(
        self,
        obs: kt.KaggricultureObsBatch,
        *,
        deterministic: bool = False,
        hidden_state: ModelHiddenState | None = None,
    ) -> ModelOutput[kt.KaggricultureActions]:
        """Sample one grammar program per seat row.

        Sampling adds no policy-validation host synchronization: the replay
        check runs only in ``evaluate_actions``. By default, packed encoding
        reads device mask values on the host. ``rollout_packing`` instead uses
        the current host observation's mask, including overflow chunk planning.
        """
        self._require_stateless(hidden_state)
        encoded = self.encode_observations(obs)
        result = self._policy(
            encoded, self._grammar_context(obs), None, deterministic=deterministic
        )
        actions, log_probs, entropies = self._policy_outputs(result, obs)
        values, winner_log_probs, value_offsets = self._value_parts(encoded, obs)
        return ModelOutput[kt.KaggricultureActions](
            actions=actions,
            log_probs=log_probs,
            entropies=entropies,
            values=values,
            winner_probabilities=winner_log_probs.exp(),
            value_offsets=value_offsets,
        )

    def _require_stateless_replay(
        self, hidden_state: ModelHiddenState | None, dones: torch.Tensor | None
    ) -> None:
        self._require_stateless(hidden_state)
        if dones is not None:
            raise ValueError(
                "KaggricultureTransformer is stateless; dones must be None"
            )

    def evaluate_actions(
        self,
        obs: kt.KaggricultureObsBatch,
        actions: kt.KaggricultureActions,
        *,
        hidden_state: ModelHiddenState | None = None,
        dones: torch.Tensor | None = None,
    ) -> ModelEvaluation:
        """Teacher-forced replay through the sampling path, with admission.

        Rejects malformed programs itself (PPO and BC replay never call the
        native ``step``): Python shape/dtype checks before any kernel, then the
        support, length and canonical flag groups with one host transfer.
        """
        self._require_stateless_replay(hidden_state, dones)
        _check_action_layout(actions, obs.still_playing.shape)
        encoded = self.encode_observations(obs)
        result = self._policy(
            encoded, self._grammar_context(obs), actions, deterministic=False
        )
        check_replay_flags(result.valid)
        return self._evaluation_from(result, encoded, obs)

    def _evaluation_from(
        self,
        result: GrammarPolicyResult,
        encoded: KaggricultureEncoded,
        obs: kt.KaggricultureObsBatch,
    ) -> ModelEvaluation:
        """The replay ``ModelEvaluation``, shared by the plain and teacher paths."""
        _, log_probs, entropies = self._policy_outputs(result, obs)
        values, winner_log_probs = self._values(encoded, obs)
        return ModelEvaluation(
            log_probs=log_probs,
            entropies=entropies,
            values=values,
            winner_probabilities=winner_log_probs.exp(),
            winner_log_probabilities=winner_log_probs,
        )

    # --- teacher distillation (Phase 4) ------------------------------------------

    def grammar_signature(self) -> GrammarSignature:
        """The tables' construction-time digest and the live ``hire_limit``."""
        return GrammarSignature(
            tables_sha256=self.actor.tables_digest,
            hire_limit=self.action_spec.hire_limit,
        )

    def compute_teacher_distillation_targets(
        self,
        obs: kt.KaggricultureObsBatch,
        actions: kt.KaggricultureActions,
        *,
        compute_action_kl: bool = True,
        compute_value: bool = True,
    ) -> KaggricultureTeacherTargets:
        """Frozen-teacher targets in the observation lead layout, under no_grad.

        One encode through ``_run_trunk``'s guards; the heads run through
        ``_policy``'s row chunking with ``collect_logits``. The replay is
        admitted under this model's own grammar (one host sync); a grammar that
        differs but still admits the program is caught by the student through
        ``grammar``.
        """
        _check_action_layout(actions, obs.still_playing.shape)
        lead = tuple(obs.still_playing.shape)
        slot_logits: dict[int, torch.Tensor] | None = None
        winner_probabilities: torch.Tensor | None = None
        with torch.no_grad():
            encoded = self.encode_observations(obs)
            if compute_action_kl:
                result = self._policy(
                    encoded,
                    self._grammar_context(obs),
                    actions,
                    deterministic=False,
                    collect_logits=True,
                )
                check_replay_flags(result.valid)
                slot_logits = {
                    slot: logits.reshape(*lead, *logits.shape[1:])
                    for slot, logits in _require(result.slot_logits).items()
                }
            if compute_value:
                winner_probabilities = (
                    self._winner_log_probabilities(encoded)
                    .exp()
                    .reshape(*lead, kt.PLAYERS)
                )
        return KaggricultureTeacherTargets(
            slot_logits=slot_logits,
            winner_probabilities=winner_probabilities,
            grammar=self.grammar_signature(),
        )

    def supports_cached_teacher_distillation(self) -> bool:
        return True

    def supports_cached_value_distillation(self) -> bool:
        # Value distillation always targets the masked winner softmax; the
        # optional critic offset is outside it and is trained by the value
        # loss alone.
        return True

    def evaluate_actions_with_cached_teacher(
        self,
        obs: kt.KaggricultureObsBatch,
        actions: kt.KaggricultureActions,
        teacher_targets: TeacherTargets,
        *,
        hidden_state: ModelHiddenState | None = None,
        dones: torch.Tensor | None = None,
        compute_teacher_action_kl: bool = True,
        compute_teacher_value: bool = True,
    ) -> ModelTeacherEvaluation:
        """Replay evaluation plus the per-slot KL against cached teacher logits.

        Admission runs before any kernel: stateless checks, the target type,
        required targets, the grammar signature (replay admission cannot see a
        teacher grammar that differs but admits the program), slot keys,
        dtypes and shapes. The student encodes once; its ``ModelEvaluation``
        is exactly ``evaluate_actions``'s.
        """
        self._require_stateless_replay(hidden_state, dones)
        if not isinstance(teacher_targets, KaggricultureTeacherTargets):
            raise TypeError(
                "KaggricultureTransformer needs KaggricultureTeacherTargets, got "
                f"{type(teacher_targets).__name__}"
            )
        lead = tuple(obs.still_playing.shape)
        _check_action_layout(actions, lead)
        teacher_logits = (
            self._admit_cached_slot_logits(teacher_targets, lead)
            if compute_teacher_action_kl
            else None
        )
        teacher_winner: torch.Tensor | None = None
        if compute_teacher_value:
            teacher_winner = teacher_targets.winner_probabilities
            if teacher_winner is None:
                raise ValueError("cached teacher value targets are missing")
            if tuple(teacher_winner.shape) != (*lead, kt.PLAYERS):
                raise ValueError(
                    "cached teacher winner_probabilities must have shape "
                    f"{(*lead, kt.PLAYERS)}, got {tuple(teacher_winner.shape)}"
                )
        encoded = self.encode_observations(obs)
        return self._teacher_evaluation(
            obs, actions, encoded, teacher_logits, teacher_winner
        )

    def _admit_cached_slot_logits(
        self, targets: KaggricultureTeacherTargets, lead: tuple[int, ...]
    ) -> dict[int, torch.Tensor]:
        """Validate cached slot logits and view them in row layout (no copy)."""
        slot_logits = targets.slot_logits
        if slot_logits is None:
            raise ValueError("cached teacher action targets are missing")
        if targets.grammar != self.grammar_signature():
            raise ValueError(
                "cached teacher targets were computed under a different grammar "
                f"(teacher {targets.grammar}, student {self.grammar_signature()}); "
                "replay admission cannot detect this, so the KL would compare "
                "different supports"
            )
        if set(slot_logits) != set(POLICY_SLOTS):
            raise ValueError(
                f"cached teacher slot keys must be {sorted(POLICY_SLOTS)}, "
                f"got {sorted(slot_logits)}"
            )
        rows = math.prod(lead)
        flat: dict[int, torch.Tensor] = {}
        for slot in POLICY_SLOTS:
            logits = slot_logits[slot]
            if logits.dtype not in (torch.float32, torch.float64):
                raise ValueError(
                    f"cached teacher slot {slot} logits must have dtype float32 "
                    f"or float64, got {logits.dtype}"
                )
            frames, width = slot_frames(slot), kt.SLOT_WIDTHS[slot]
            if tuple(logits.shape) != (*lead, frames, width):
                raise ValueError(
                    f"cached teacher slot {slot} logits must have shape "
                    f"{(*lead, frames, width)}, got {tuple(logits.shape)}"
                )
            flat[slot] = logits.reshape(rows, frames, width)
        return flat

    def evaluate_actions_with_teacher(
        self,
        obs: kt.KaggricultureObsBatch,
        actions: kt.KaggricultureActions,
        teacher: BaseModelAPI[
            kt.KaggricultureObsBatch,
            kt.KaggricultureActions,
            kt.KaggricultureActionConfig,
        ],
        *,
        hidden_state: ModelHiddenState | None = None,
        dones: torch.Tensor | None = None,
        compute_teacher_action_kl: bool = True,
        compute_teacher_value: bool = True,
    ) -> ModelTeacherEvaluation:
        """Combined path: one student pass plus a no-grad teacher pass.

        The teacher's row-layout logits go straight into the student's core,
        without the lead reshape or a ``TeacherTargets`` round trip; the result
        is bit-for-bit equal to the cached path. Admission (teacher type,
        ``action_spec``, grammar signature and per-table equality) runs before
        any kernel.
        """
        self._require_stateless_replay(hidden_state, dones)
        if not isinstance(teacher, KaggricultureTransformer):
            raise ValueError(
                "teacher must be a KaggricultureTransformer, got "
                f"{type(teacher).__name__}"
            )
        if teacher.action_spec != self.action_spec:
            raise ValueError(
                f"teacher action_spec {teacher.action_spec} must match the "
                f"student's {self.action_spec}"
            )
        if teacher.grammar_signature() != self.grammar_signature():
            raise ValueError(
                f"teacher grammar {teacher.grammar_signature()} differs from the "
                f"student's {self.grammar_signature()}"
            )
        teacher_tables = teacher.actor.tables().as_dict()
        for name, table in self.actor.tables().as_dict().items():
            if not torch.equal(teacher_tables[name], table):
                raise ValueError(
                    f"teacher grammar table {name} differs from the student's"
                )
        lead = tuple(obs.still_playing.shape)
        _check_action_layout(actions, lead)
        encoded = self.encode_observations(obs)
        teacher_logits: dict[int, torch.Tensor] | None = None
        teacher_winner: torch.Tensor | None = None
        if compute_teacher_action_kl or compute_teacher_value:
            with torch.no_grad():
                teacher_encoded = teacher.encode_observations(obs)
                if compute_teacher_action_kl:
                    teacher_result = teacher._policy(
                        teacher_encoded,
                        teacher._grammar_context(obs),
                        actions,
                        deterministic=False,
                        collect_logits=True,
                    )
                    check_replay_flags(teacher_result.valid)
                    teacher_logits = _require(teacher_result.slot_logits)
                if compute_teacher_value:
                    teacher_winner = (
                        teacher._winner_log_probabilities(teacher_encoded)
                        .exp()
                        .reshape(*lead, kt.PLAYERS)
                    )
        return self._teacher_evaluation(
            obs, actions, encoded, teacher_logits, teacher_winner
        )

    def _teacher_evaluation(
        self,
        obs: kt.KaggricultureObsBatch,
        actions: kt.KaggricultureActions,
        encoded: KaggricultureEncoded,
        teacher_logits: dict[int, torch.Tensor] | None,
        teacher_winner: torch.Tensor | None,
    ) -> ModelTeacherEvaluation:
        """Student replay with the per-slot KL, assembled for PPO."""
        result = self._policy(
            encoded,
            self._grammar_context(obs),
            actions,
            deterministic=False,
            teacher_logits=teacher_logits,
        )
        check_replay_flags(result.valid)
        student = self._evaluation_from(result, encoded, obs)
        action_kl: ModelActionKLDivergences | None = None
        if teacher_logits is not None:
            lead = tuple(obs.still_playing.shape)
            event = _require(result.kl).reshape(*lead, kt.MAX_FRAMES, kt.ACTION_SLOTS)
            per_frame = event.sum(dim=-1)
            action_kl = ModelActionKLDivergences(
                launch=torch.zeros_like(per_frame),
                event=event,
                per_player_entity=per_frame,
                components={
                    kt.SLOT_NAMES[slot]: event[..., slot] for slot in POLICY_SLOTS
                },
                target=None,
            )
        return ModelTeacherEvaluation(
            student=student,
            action_kl=action_kl,
            teacher_winner_probabilities=teacher_winner,
            student_winner_log_probabilities=student.winner_log_probabilities,
        )

    def teacher_value_cross_entropy(
        self,
        student_winner_log_probabilities: torch.Tensor,
        teacher_winner_probabilities: torch.Tensor,
        *,
        value_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Per-seat CE over (self, opponent), averaged over live seats per state.

        Each seat carries its own full winner distribution, so a mean keeps one
        CE per state (Isaiah's scale for ``teacher_value_coef``); a sum would
        double it. A non-live seat is excluded; a state with no live seat gives
        0, and PPO's state weight is 0 there too.
        """
        expected = (*value_mask.shape, kt.PLAYERS)
        for name, tensor in (
            ("student winner log-probabilities", student_winner_log_probabilities),
            ("teacher winner probabilities", teacher_winner_probabilities),
        ):
            if tuple(tensor.shape) != expected:
                raise ValueError(
                    f"{name} must have shape {expected}, got {tuple(tensor.shape)}"
                )
        per_seat = (
            -teacher_winner_probabilities.detach() * student_winner_log_probabilities
        ).sum(dim=-1)
        live = value_mask.to(dtype=per_seat.dtype)
        return (per_seat * live).sum(dim=-1) / live.sum(dim=-1).clamp_min(1.0)

    @staticmethod
    def _grammar_context(obs: kt.KaggricultureObsBatch) -> GrammarContext:
        """Own actors are the first ``MAX_ACTORS`` actor slots, in frame order."""
        return GrammarContext(
            actor_counts=obs.actor_mask[..., : kt.MAX_ACTORS]
            .sum(dim=-1, dtype=torch.int64)
            .reshape(-1),
            order_limits=obs.order_limits.reshape(-1),
            live=obs.still_playing.reshape(-1),
        )

    def _policy(
        self,
        encoded: KaggricultureEncoded,
        context: GrammarContext,
        actions: kt.KaggricultureActions | None,
        *,
        deterministic: bool,
        teacher_logits: dict[int, torch.Tensor] | None = None,
        collect_logits: bool = False,
    ) -> GrammarPolicyResult:
        """Run the heads in row chunks below the head GEMM-extent limit.

        ``teacher_logits`` (row layout, keyed by ``POLICY_SLOTS``) is sliced
        per chunk; ``slot_logits`` and ``kl`` are joined across chunks.
        """
        rows = context.live.shape[0]
        tokens = (
            None
            if actions is None
            else actions.tokens.reshape(rows, kt.MAX_FRAMES, kt.ACTION_SLOTS)
        )
        lengths = None if actions is None else actions.lengths.reshape(rows)
        chunk = head_rows_per_chunk(self.config)
        results = [
            self._policy_chunk(
                encoded,
                context,
                tokens,
                lengths,
                slice(start, min(start + chunk, rows)),
                deterministic=deterministic,
                teacher_logits=teacher_logits,
                collect_logits=collect_logits,
            )
            for start in range(0, max(rows, 1), chunk)
        ]
        if len(results) == 1:
            return results[0]
        slot_logits: dict[int, torch.Tensor] | None = None
        if collect_logits:
            slot_logits = {
                slot: torch.cat([_require(r.slot_logits)[slot] for r in results])
                for slot in POLICY_SLOTS
            }
        return GrammarPolicyResult(
            tokens=torch.cat([r.tokens for r in results]),
            lengths=torch.cat([r.lengths for r in results]),
            log_probs=torch.cat([r.log_probs for r in results]),
            entropies=torch.cat([r.entropies for r in results]),
            valid=torch.cat([r.valid for r in results]),
            slot_logits=slot_logits,
            kl=(
                None
                if teacher_logits is None
                else torch.cat([_require(r.kl) for r in results])
            ),
        )

    def _policy_chunk(
        self,
        encoded: KaggricultureEncoded,
        context: GrammarContext,
        tokens: torch.Tensor | None,
        lengths: torch.Tensor | None,
        rows: slice,
        *,
        deterministic: bool,
        teacher_logits: dict[int, torch.Tensor] | None = None,
        collect_logits: bool = False,
    ) -> GrammarPolicyResult:
        unit_input, market_input = self._actor_inputs(encoded, rows)
        core = self._compiled_actor_core or self.actor.policy_core
        if self._compiled_actor_core is not None:
            if self.compiled_regions_require_gemm_backends:
                require_compiled_gemm_backends()
            exponentials = (
                sample_policy_exponentials(unit_input)
                if tokens is None and not deterministic
                else None
            )
        else:
            exponentials = None
        return core(
            unit_input,
            market_input,
            context.actor_counts[rows],
            context.order_limits[rows],
            context.live[rows],
            self.action_spec.hire_limit,
            None if tokens is None else tokens[rows],
            None if lengths is None else lengths[rows],
            deterministic,
            (
                None
                if teacher_logits is None
                else {slot: logits[rows] for slot, logits in teacher_logits.items()}
            ),
            collect_logits,
            exponentials,
        )

    def _actor_inputs(
        self, encoded: KaggricultureEncoded, rows: slice
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``actor_input_proj`` outputs: unit ``[B, 241, D]``, market ``[B, D]``."""
        entity = encoded.own_actor_hidden[rows]
        player = encoded.player_hidden[rows, 0]  # this seat's own player token
        plan = encoded.actor_plan_hidden[rows, 0]
        actors = entity.shape[1]
        unit_input = self.actor_input_proj(
            torch.cat(
                (
                    entity,
                    player[:, None, :].expand(-1, actors, -1),
                    plan[:, None, :].expand(-1, actors, -1),
                ),
                dim=-1,
            )
        )
        # No market entity exists: the plan hidden stands in for it.
        market_input = self.actor_input_proj(torch.cat((plan, player, plan), dim=-1))
        return unit_input, market_input

    @staticmethod
    def _policy_outputs(
        result: GrammarPolicyResult, obs: kt.KaggricultureObsBatch
    ) -> tuple[kt.KaggricultureActions, ModelActionLogProbs, ModelActionEntropies]:
        lead = tuple(obs.still_playing.shape)
        shape = (*lead, kt.MAX_FRAMES, kt.ACTION_SLOTS)
        event = result.log_probs.reshape(shape)
        entropy = result.entropies.reshape(shape)
        frame_log_probs = event.sum(dim=-1)
        frame_entropies = entropy.sum(dim=-1)
        return (
            kt.KaggricultureActions(
                tokens=result.tokens.reshape(shape),
                lengths=result.lengths.reshape(lead),
            ),
            ModelActionLogProbs(
                launch=torch.zeros_like(frame_log_probs),
                event=event,
                per_player_entity=frame_log_probs,
            ),
            ModelActionEntropies(
                launch=torch.zeros_like(frame_entropies),
                event=entropy,
                per_player_entity=frame_entropies,
                components={
                    name: entropy[..., slot] for slot, name in enumerate(kt.SLOT_NAMES)
                },
            ),
        )

    def _values(
        self, encoded: KaggricultureEncoded, obs: kt.KaggricultureObsBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(value [env, seat], winner log-probs [env, seat, 2])``.

        The value is ``2 p(self) - 1``, plus the seat's critic offset when the
        model has the head.
        """
        values, winner_log_probs, _ = self._value_parts(encoded, obs)
        return values, winner_log_probs

    def _value_parts(
        self, encoded: KaggricultureEncoded, obs: kt.KaggricultureObsBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """``(value, winner log-probs, offset [env, seat] or None)``."""
        winner_log_probs = self._winner_log_probabilities(encoded).reshape(
            *obs.still_playing.shape, kt.PLAYERS
        )
        values = 2.0 * winner_log_probs[..., 0].exp() - 1.0
        offsets = self._critic_offsets(encoded)
        if offsets is None:
            return values, winner_log_probs, None
        offsets = offsets.reshape(values.shape)
        return values + offsets, winner_log_probs, offsets

    def _critic_offsets(self, encoded: KaggricultureEncoded) -> torch.Tensor | None:
        """Per-row offset from the row's own critic-value token, FP32 ``[rows]``.

        The head's Linear layers run in the autocast dtype; the output is cast.

        Only token 0 (self) is read: no opponent token and no other row. A
        non-live row gets exactly 0, so its value stays the winner critic's.
        With ``critic_offset_detach_trunk`` the token is detached, so the value
        loss reaches the trunk only through the winner head.
        """
        head = self.critic_offset_head
        if head is None:
            return None
        own = encoded.critic_value_hidden[:, 0]
        if self.config.critic_offset_detach_trunk:
            own = own.detach()
        offsets = head(own).float().squeeze(-1)
        return offsets.masked_fill(~encoded.critic_value_mask[:, 0], 0.0)

    def compute_value(
        self,
        obs: kt.KaggricultureObsBatch,
        *,
        hidden_state: ModelHiddenState | None = None,
    ) -> torch.Tensor:
        """Value per seat, ``[env, seat]``: ``2 p(self) - 1`` plus any offset."""
        self._require_stateless(hidden_state)
        values, _ = self._values(self.encode_observations(obs), obs)
        return values

    def winner_log_probabilities(self, obs: kt.KaggricultureObsBatch) -> torch.Tensor:
        """Winner distribution over (self, opponent) from each seat's own view.

        Shape ``[env, seat, 2]``; index 0 is this seat winning.
        """
        encoded = self.encode_observations(obs)
        return self._winner_log_probabilities(encoded).reshape(
            *obs.still_playing.shape, kt.PLAYERS
        )

    def _winner_log_probabilities(self, encoded: KaggricultureEncoded) -> torch.Tensor:
        """Isaiah's masked winner softmax over the critic-value tokens.

        As ``StatelessTransformerV1._critic_distillation``: logits of masked
        tokens are filled with the dtype minimum before ``log_softmax``, so
        ``exp`` equals Isaiah's ``masked_softmax(logits, still_playing)``. The
        mask is the critic tokens' own token mask (the row's ``still_playing``
        on both tokens); an all-masked row gets Isaiah's uniform result.
        """
        logits = self.critic_head(encoded.critic_value_hidden).float().squeeze(-1)
        masked_logits = logits.masked_fill(
            ~encoded.critic_value_mask, torch.finfo(logits.dtype).min
        )
        return F.log_softmax(masked_logits, dim=-1)


def _require(value: _T | None) -> _T:
    if value is None:
        raise RuntimeError("a requested policy-core output is missing")
    return value


def _check_action_layout(
    actions: kt.KaggricultureActions, lead: torch.Size | tuple[int, ...]
) -> None:
    """Python-side shape/dtype admission before any policy kernel (no sync)."""
    expected = (*lead, kt.MAX_FRAMES, kt.ACTION_SLOTS)
    if tuple(actions.tokens.shape) != expected:
        raise ValueError(
            f"action tokens must have shape {expected}, "
            f"got {tuple(actions.tokens.shape)}"
        )
    if tuple(actions.lengths.shape) != tuple(lead):
        raise ValueError(
            f"action lengths must have shape {tuple(lead)}, "
            f"got {tuple(actions.lengths.shape)}"
        )
    if actions.tokens.dtype != torch.int64 or actions.lengths.dtype != torch.int64:
        raise ValueError(
            "action tokens and lengths must be int64, got "
            f"{actions.tokens.dtype} and {actions.lengths.dtype}"
        )
