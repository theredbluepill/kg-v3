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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from pydantic import Field
from torch import nn

from owl.config import BaseConfig
from owl.kaggriculture import types as kt
from owl.kaggriculture.gpu_grammar import GrammarTables, expected_grammar_tables
from owl.model.actor.common import OutputProjectionMLP
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
from owl.model.kaggriculture_actor import (
    POLICY_SLOTS,
    GrammarContext,
    GrammarPolicyResult,
    KaggricultureGrammarActor,
    check_replay_flags,
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
    unpack_sequence,
)

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
    ]
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

        ``grammar_tables`` defaults to ``expected_grammar_tables()`` until the
        native binding (Task 1.2/1.4) exists; see ``owl.kaggriculture.gpu_grammar``.
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
            expected_grammar_tables() if grammar_tables is None else grammar_tables,
        )
        self._compiled_transformer_trunk: (
            Callable[
                [torch.Tensor, torch.Tensor | None, PackedSequence | None],
                torch.Tensor,
            ]
            | None
        ) = None
        # Heads run eager in production; tests swap in a captured core here.
        self._compiled_actor_core: Callable[..., GrammarPolicyResult] | None = None
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
            gain = (
                _CRITIC_HEAD_INIT_GAIN
                if layer is self.critic_head.out
                else _ACTOR_HEAD_INIT_GAIN
            )
            _init_linear(layer, gain=gain)

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
        return (self.critic_head.out, *self.actor.get_output_layers())

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
        playing = obs.still_playing.reshape(rows, 1)
        device = x.device
        token_mask = torch.cat(
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
        return x, token_mask

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

    @staticmethod
    def _run_packed_trunk(
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
        self._compiled_transformer_trunk = torch.compile(
            self._forward_transformer_trunk, mode=mode, dynamic=True
        )
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
        check runs only in ``evaluate_actions``. The encode is not sync-free;
        packed trunk dispatch sizes its packing on the host and, for an
        oversized batch, transfers per-row token counts to plan chunks.
        """
        self._require_stateless(hidden_state)
        encoded = self.encode_observations(obs)
        result = self._policy(
            encoded, self._grammar_context(obs), None, deterministic=deterministic
        )
        actions, log_probs, entropies = self._policy_outputs(result, obs)
        values, winner_log_probs = self._values(encoded, obs)
        return ModelOutput[kt.KaggricultureActions](
            actions=actions,
            log_probs=log_probs,
            entropies=entropies,
            values=values,
            winner_probabilities=winner_log_probs.exp(),
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
        self._require_stateless(hidden_state)
        if dones is not None:
            raise ValueError(
                "KaggricultureTransformer is stateless; dones must be None"
            )
        _check_action_layout(actions, obs.still_playing.shape)
        encoded = self.encode_observations(obs)
        result = self._policy(
            encoded, self._grammar_context(obs), actions, deterministic=False
        )
        check_replay_flags(result.valid)
        _, log_probs, entropies = self._policy_outputs(result, obs)
        values, winner_log_probs = self._values(encoded, obs)
        return ModelEvaluation(
            log_probs=log_probs,
            entropies=entropies,
            values=values,
            winner_probabilities=winner_log_probs.exp(),
            winner_log_probabilities=winner_log_probs,
        )

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
    ) -> GrammarPolicyResult:
        """Run the heads in row chunks below the head GEMM-extent limit."""
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
            )
            for start in range(0, max(rows, 1), chunk)
        ]
        if len(results) == 1:
            return results[0]
        return GrammarPolicyResult(
            tokens=torch.cat([r.tokens for r in results]),
            lengths=torch.cat([r.lengths for r in results]),
            log_probs=torch.cat([r.log_probs for r in results]),
            entropies=torch.cat([r.entropies for r in results]),
            valid=torch.cat([r.valid for r in results]),
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
    ) -> GrammarPolicyResult:
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
        core = self._compiled_actor_core or self.actor.policy_core
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
        )

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
        """``(2 p(self) - 1 [env, seat], winner log-probs [env, seat, 2])``."""
        winner_log_probs = self._winner_log_probabilities(encoded).reshape(
            *obs.still_playing.shape, kt.PLAYERS
        )
        return 2.0 * winner_log_probs[..., 0].exp() - 1.0, winner_log_probs

    def compute_value(
        self,
        obs: kt.KaggricultureObsBatch,
        *,
        hidden_state: ModelHiddenState | None = None,
    ) -> torch.Tensor:
        """``win_loss`` value ``2 * p(self) - 1`` per seat, shape ``[env, seat]``."""
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
        logits = self.critic_head(encoded.critic_value_hidden).float().squeeze(-1)
        return logits.log_softmax(dim=-1)


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
