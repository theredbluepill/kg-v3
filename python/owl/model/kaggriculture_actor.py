"""Kaggriculture grammar action heads (rebuild Task 2.3).

``KaggricultureGrammarActor`` holds the only game-specific actor parts on top of
Isaiah's actor input projection: a ``source_norm``, market queue-position
embeddings, within-frame prefix embeddings and one ``OutputProjectionMLP`` per
sampled slot. Frames are batched: every unit frame (one per own actor) and every
market queue position is decided in parallel; only the fixed within-frame slot
stages are unrolled, and the prefix state resets for every observation.

Sampling is exact Gumbel-max. Market HIRE capacity is an exact parallel
correction: the raw HIRE prefix decides where HIRE is masked, and the argmax is
re-taken over the same perturbed scores. Replay (teacher-forced evaluation) runs
the same code path and computes three device flag groups (support, length,
canonical equality) with safe indices; the host check lives in the model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from owl.kaggriculture import types as kt
from owl.kaggriculture.gpu_grammar import (
    MARKET_HIRE,
    MARKET_NONE,
    GrammarTables,
    grammar_tables_digest,
    validate_grammar_tables,
)
from owl.model.actor.common import OutputProjectionMLP, categorical_kl_from_logits

SLOT = {name: index for index, name in enumerate(kt.SLOT_NAMES)}
UNIT_POLICY_SLOTS = (1, 3, 4, 5, 6)
MARKET_POLICY_SLOTS = (7, 8, 9, 10)
POLICY_SLOTS = (*UNIT_POLICY_SLOTS, *MARKET_POLICY_SLOTS)
CONDITIONING_SLOTS = (1, 3, 4, 5, 7, 8, 9)
MARKET_POSITIONS = kt.MAX_ORDER_LIMIT + 1  # queue positions plus the sentinel
FLAG_GROUPS = ("support", "length", "canonical")


@dataclass(frozen=True)
class GrammarContext:
    """Per-row runtime grammar context (``B`` = seat rows)."""

    actor_counts: torch.Tensor  # int64 [B], own actors
    order_limits: torch.Tensor  # int64 [B]
    live: torch.Tensor  # bool [B], still_playing


@dataclass(frozen=True)
class GrammarPolicyResult:
    """Internal typed policy result; ``valid`` never reaches ``ModelOutput``.

    ``valid[:, g]`` is flag group ``FLAG_GROUPS[g]`` per row; sampling is valid
    by construction and reports all-true flags.

    Teacher distillation (Phase 4) fields, ``None`` unless requested:
    ``slot_logits[k]`` holds each policy slot's density-dtype logits under the
    replay-conditioned mask, filled with ``finfo(dtype).min`` outside it
    (``[B, 241, W_k]`` unit slots, ``[B, 11, W_k]`` market slots); ``kl`` is
    the per-slot ``KL(teacher || student)`` weighted by the same liveness as
    ``log_probs`` and placed in the same ``[B, 252, 12]`` frame layout.
    """

    tokens: torch.Tensor  # int64 [B, 252, 12]
    lengths: torch.Tensor  # int64 [B]
    log_probs: torch.Tensor  # float [B, 252, 12]
    entropies: torch.Tensor  # float [B, 252, 12]
    valid: torch.Tensor  # bool [B, 3]
    slot_logits: dict[int, torch.Tensor] | None = None
    kl: torch.Tensor | None = None


def _density_dtype(logits: torch.Tensor) -> torch.Tensor:
    """FP32 densities (FP64 kept for exactness tests)."""
    return logits if logits.dtype == torch.float64 else logits.float()


def _min_filled(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Isaiah's target-logit masking: the dtype minimum outside the support.

    Finite in FP32 and FP64, so ``softmax`` gives exactly 0 there without
    forming ``-inf`` (the density path keeps its ``-inf`` masking).
    """
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


def _slot_kl(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Liveness-weighted ``KL(teacher || student)`` in the student's dtype."""
    kl = categorical_kl_from_logits(teacher_logits, student_logits, mask)
    return kl.to(dtype=student_logits.dtype) * weight


def gumbel_perturb(masked_logits: torch.Tensor) -> torch.Tensor:
    """Add independent standard Gumbel noise (``-log Exp(1)``) to every entry."""
    noise = torch.empty_like(masked_logits).exponential_()
    tiny = torch.finfo(masked_logits.dtype).tiny
    return masked_logits - noise.clamp_min(tiny).log()


def gumbel_argmax(masked_logits: torch.Tensor) -> torch.Tensor:
    """Exact categorical sample over the last dim via Gumbel-max.

    ``masked_logits`` carries ``-inf`` outside the support; no multinomial.
    """
    return gumbel_perturb(masked_logits).argmax(dim=-1)


def couple_market_kinds(
    perturbed: torch.Tensor, actor_counts: torch.Tensor, hire_limit: int
) -> torch.Tensor:
    """Final market kinds from perturbed masked scores ``[B, P, 8]``.

    ``raw`` is the argmax at every position. The exclusive raw-HIRE prefix count
    decides where HIRE capacity (``actor_counts + prior < hire_limit``) is
    exhausted; there HIRE is masked and the argmax is re-taken over the same
    scores, so the corrected outcome can be any non-HIRE kind, including NONE
    and EMPTY. Once the raw and final HIRE prefixes differ, capacity is already
    exhausted under both, so the correction equals the sequential sampler.
    """
    raw = perturbed.argmax(dim=-1)
    raw_hires = (raw == MARKET_HIRE).long()
    prior = raw_hires.cumsum(dim=-1) - raw_hires
    can_hire = actor_counts[:, None] + prior < hire_limit
    hire_column = (
        torch.arange(perturbed.shape[-1], device=perturbed.device) == MARKET_HIRE
    )
    blocked = (~can_hire)[..., None] & hire_column
    return perturbed.masked_fill(blocked, -torch.inf).argmax(dim=-1)


def _delta0(width: int, device: torch.device) -> torch.Tensor:
    """Singleton-zero support of width ``width``."""
    return torch.arange(width, device=device) == 0


def _safe_choice(
    raw: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clamped index for every gather/lookup, and whether ``raw`` is admitted.

    The clamped value is only a temporary index: its validity flag goes to the
    support group, and the canonical comparison rejects any repaired value.
    """
    width = mask.shape[-1]
    safe = raw.clamp(0, width - 1)
    in_range = (raw >= 0) & (raw < width)
    admitted = in_range & mask.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return safe, admitted


def _masked_density(
    logits: torch.Tensor, mask: torch.Tensor, choice: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Masked log-softmax: chosen log-prob, entropy and admission."""
    log_probs = F.log_softmax(logits.masked_fill(~mask, -torch.inf), dim=-1)
    admitted = mask.gather(-1, choice.unsqueeze(-1)).squeeze(-1)
    selected = log_probs.gather(-1, choice.unsqueeze(-1)).squeeze(-1)
    entropy = -(log_probs.exp() * log_probs.masked_fill(~mask, 0.0)).sum(dim=-1)
    # A rejected replay raises after the core; keep -inf out of the graph.
    return torch.where(admitted, selected, 0.0), entropy, admitted


class KaggricultureGrammarActor(nn.Module):
    table_unit_kind: torch.Tensor
    table_unit_item: torch.Tensor
    table_unit_quantity_present: torch.Tensor
    table_unit_quantity_high: torch.Tensor
    table_unit_quantity_low: torch.Tensor
    table_market_kind: torch.Tensor
    table_market_item: torch.Tensor
    table_market_quantity: torch.Tensor

    def __init__(self, trunk_config: Any, tables: GrammarTables) -> None:
        super().__init__()
        width = trunk_config.embed_dim
        self.embed_dim = width
        self.source_norm = nn.LayerNorm(width)
        self.market_position = nn.Embedding(MARKET_POSITIONS, width)
        self.slot_embeddings = nn.ModuleDict(
            {
                kt.SLOT_NAMES[slot]: nn.Embedding(kt.SLOT_WIDTHS[slot], width)
                for slot in CONDITIONING_SLOTS
            }
        )
        self.heads = nn.ModuleDict(
            {
                kt.SLOT_NAMES[slot]: OutputProjectionMLP(
                    trunk_config, kt.SLOT_WIDTHS[slot]
                )
                for slot in POLICY_SLOTS
            }
        )
        validate_grammar_tables(tables)
        # Host identity of the tables, taken once: the buffers are never
        # reassigned, and teacher targets compare it without a device sync.
        self.tables_digest = grammar_tables_digest(tables)
        # Non-persistent: tables come from the grammar, never from checkpoints.
        for name, table in tables.as_dict().items():
            self.register_buffer(f"table_{name}", table.clone(), persistent=False)

    # --- Isaiah's parameter conventions ---------------------------------------

    def get_input_layers(self) -> tuple[nn.Parameter, ...]:
        """Token-like weights: Isaiah's ``_init_input_layer`` takes parameters."""
        weights = [self.market_position.weight]
        for embedding in self.slot_embeddings.values():
            assert isinstance(embedding, nn.Embedding)
            weights.append(embedding.weight)
        parameters: list[nn.Parameter] = []
        for weight in weights:
            assert isinstance(weight, nn.Parameter)
            parameters.append(weight)
        return tuple(parameters)

    def get_output_layers(self) -> tuple[nn.Linear, ...]:
        return tuple(
            head.out
            for head in self.heads.values()
            if isinstance(head, OutputProjectionMLP)
        )

    def max_head_width(self) -> int:
        return max(kt.SLOT_WIDTHS[slot] for slot in POLICY_SLOTS)

    # --- policy core ------------------------------------------------------------

    def tables(self) -> GrammarTables:
        return GrammarTables(
            unit_kind=self.table_unit_kind,
            unit_item=self.table_unit_item,
            unit_quantity_present=self.table_unit_quantity_present,
            unit_quantity_high=self.table_unit_quantity_high,
            unit_quantity_low=self.table_unit_quantity_low,
            market_kind=self.table_market_kind,
            market_item=self.table_market_item,
            market_quantity=self.table_market_quantity,
        )

    def market_base(self, market_input: torch.Tensor) -> torch.Tensor:
        """``[B, D]`` market input plus queue-position embeddings, normalized."""
        position = self.market_position.weight.to(dtype=market_input.dtype)
        return self.source_norm(market_input[:, None, :] + position[None, :, :])

    def slot_logits(self, slot: int, hidden: torch.Tensor) -> torch.Tensor:
        return _density_dtype(self.heads[kt.SLOT_NAMES[slot]](hidden))

    def _embed(self, slot: int, choice: torch.Tensor) -> torch.Tensor:
        embedding = self.slot_embeddings[kt.SLOT_NAMES[slot]]
        assert isinstance(embedding, nn.Embedding)
        return embedding(choice)

    def policy_core(
        self,
        unit_input: torch.Tensor,
        market_input: torch.Tensor,
        actor_counts: torch.Tensor,
        order_limits: torch.Tensor,
        live: torch.Tensor,
        hire_limit: int,
        supplied_tokens: torch.Tensor | None,
        supplied_lengths: torch.Tensor | None,
        deterministic: bool,
        teacher_logits: dict[int, torch.Tensor] | None = None,
        collect_logits: bool = False,
    ) -> GrammarPolicyResult:
        """Sample (``supplied_tokens is None``) or replay one row chunk.

        ``unit_input [B, 241, D]`` and ``market_input [B, D]`` are the outputs
        of the model's ``actor_input_proj``. Tensor-only; no host sync.

        ``collect_logits`` returns every policy slot's masked logits (the
        teacher side); ``teacher_logits`` (keyed by ``POLICY_SLOTS``, row
        layout of this chunk) adds the per-slot KL against them, evaluated at
        the same replayed prefix (the student side).
        """
        rows, actors, _ = unit_input.shape
        device = unit_input.device
        tables = self.tables()
        frames = torch.arange(actors, device=device)
        unit_live = live[:, None] & (frames[None, :] < actor_counts[:, None])
        positions = torch.arange(MARKET_POSITIONS, device=device)
        available = (
            live[:, None]
            & (positions[None, :] < order_limits[:, None])
            & (positions[None, :] < kt.MAX_ORDER_LIMIT)
        )
        market_frame = actor_counts[:, None] + positions[None, :]
        support = torch.ones(rows, dtype=torch.bool, device=device)

        # --- unit frames: kind, item, quantity present, high and low digits ---
        unit_base = self.source_norm(unit_input)
        unit_prefix = torch.zeros_like(unit_base)
        unit_choice: dict[int, torch.Tensor] = {}
        unit_logp: dict[int, torch.Tensor] = {}
        unit_entropy: dict[int, torch.Tensor] = {}
        collected: dict[int, torch.Tensor] = {}
        unit_kl: dict[int, torch.Tensor] = {}
        market_kl: dict[int, torch.Tensor] = {}
        delta_unit_kind = _delta0(tables.unit_kind.shape[0], device)
        for stage, slot in enumerate(UNIT_POLICY_SLOTS):
            if slot == SLOT["unit_kind"]:
                mask = torch.where(
                    unit_live[..., None], tables.unit_kind, delta_unit_kind
                )
            elif slot == SLOT["unit_item"]:
                mask = tables.unit_item[unit_choice[SLOT["unit_kind"]]]
            elif slot == SLOT["unit_quantity_present"]:
                mask = tables.unit_quantity_present[unit_choice[SLOT["unit_kind"]]]
            elif slot == SLOT["unit_quantity_high"]:
                mask = tables.unit_quantity_high[
                    unit_choice[SLOT["unit_quantity_present"]]
                ]
            else:
                mask = tables.unit_quantity_low[
                    unit_choice[SLOT["unit_quantity_present"]],
                    (unit_choice[SLOT["unit_quantity_high"]] == 0).long(),
                ]
            hidden = unit_base + unit_prefix / math.sqrt(stage + 1)
            logits = self.slot_logits(slot, hidden)
            if supplied_tokens is not None:
                raw = torch.where(unit_live, supplied_tokens[:, :actors, slot], 0)
                choice, admitted = _safe_choice(raw, mask)
            else:
                masked = logits.masked_fill(~mask, -torch.inf)
                choice = masked.argmax(-1) if deterministic else gumbel_argmax(masked)
                admitted = mask.gather(-1, choice.unsqueeze(-1)).squeeze(-1)
            logp, entropy, _ = _masked_density(logits, mask, choice)
            if collect_logits or teacher_logits is not None:
                floored = _min_filled(logits, mask)
                if collect_logits:
                    collected[slot] = floored
                if teacher_logits is not None:
                    unit_kl[slot] = _slot_kl(
                        teacher_logits[slot], floored, mask, unit_live
                    )
            support = support & admitted.all(dim=-1)
            unit_choice[slot] = choice
            unit_logp[slot] = logp * unit_live
            unit_entropy[slot] = entropy * unit_live
            if slot in CONDITIONING_SLOTS:
                unit_prefix = unit_prefix + self._embed(slot, choice).to(
                    dtype=unit_base.dtype
                )

        # --- market queue: kind (coupled HIRE capacity), item, quantity -------
        market_base = self.market_base(market_input)
        kind_width = tables.market_kind.shape[0]
        delta_market_kind = _delta0(kind_width, device)
        initial_mask = torch.where(
            available[..., None], tables.market_kind, delta_market_kind
        )
        kind_logits = self.slot_logits(SLOT["market_kind"], market_base)
        market_supplied: torch.Tensor | None = None
        if supplied_tokens is not None:
            market_supplied = supplied_tokens.gather(
                1, market_frame[..., None].expand(-1, -1, kt.ACTION_SLOTS)
            )
            raw_kind = market_supplied[..., SLOT["market_kind"]]
            kind = raw_kind.clamp(0, kind_width - 1)
            kind_in_range = (raw_kind >= 0) & (raw_kind < kind_width)
        else:
            # Eight independent Gumbels per position, independent across
            # positions; the correction reuses these perturbed scores.
            masked = kind_logits.masked_fill(~initial_mask, -torch.inf)
            perturbed = masked if deterministic else gumbel_perturb(masked)
            kind = couple_market_kinds(perturbed, actor_counts, hire_limit)
            kind_in_range = torch.ones_like(available)
        # Density uses the final HIRE prefix: exactly the sequential support up
        # to the first final NONE (STOP). Later positions are marginalized.
        final_hires = (kind == MARKET_HIRE).long()
        final_prior = final_hires.cumsum(dim=-1) - final_hires
        can_hire = actor_counts[:, None] + final_prior < hire_limit
        not_hire = torch.arange(kind_width, device=device) != MARKET_HIRE
        kind_mask = initial_mask & (can_hire[..., None] | not_hire)
        kind_logp, kind_entropy, kind_admitted = _masked_density(
            kind_logits, kind_mask, kind
        )
        stop_position = torch.where(
            kind == MARKET_NONE, positions[None, :], MARKET_POSITIONS
        ).amin(dim=-1)
        market_live = live[:, None] & (positions[None, :] <= stop_position[:, None])
        if collect_logits or teacher_logits is not None:
            # Logits are computed once per position: the final mask is the one
            # the density uses.
            floored = _min_filled(kind_logits, kind_mask)
            if collect_logits:
                collected[SLOT["market_kind"]] = floored
            if teacher_logits is not None:
                market_kl[SLOT["market_kind"]] = _slot_kl(
                    teacher_logits[SLOT["market_kind"]], floored, kind_mask, market_live
                )
        support = support & ((kind_admitted & kind_in_range) | ~market_live).all(dim=-1)
        market_choice = {SLOT["market_kind"]: kind}
        market_logp = {SLOT["market_kind"]: kind_logp * market_live}
        market_entropy = {SLOT["market_kind"]: kind_entropy * market_live}
        market_prefix = self._embed(SLOT["market_kind"], kind).to(
            dtype=market_base.dtype
        )
        for stage, slot in enumerate(MARKET_POLICY_SLOTS[1:], start=1):
            if slot == SLOT["market_item"]:
                mask = tables.market_item[kind]
            else:
                mask = tables.market_quantity[kind]
            hidden = market_base + market_prefix / math.sqrt(stage + 1)
            logits = self.slot_logits(slot, hidden)
            if market_supplied is not None:
                raw = torch.where(market_live, market_supplied[..., slot], 0)
                choice, admitted = _safe_choice(raw, mask)
            else:
                masked = logits.masked_fill(~mask, -torch.inf)
                choice = masked.argmax(-1) if deterministic else gumbel_argmax(masked)
                admitted = mask.gather(-1, choice.unsqueeze(-1)).squeeze(-1)
            logp, entropy, _ = _masked_density(logits, mask, choice)
            if collect_logits or teacher_logits is not None:
                floored = _min_filled(logits, mask)
                if collect_logits:
                    collected[slot] = floored
                if teacher_logits is not None:
                    market_kl[slot] = _slot_kl(
                        teacher_logits[slot], floored, mask, market_live
                    )
            support = support & (admitted | ~market_live).all(dim=-1)
            market_choice[slot] = choice
            market_logp[slot] = logp * market_live
            market_entropy[slot] = entropy * market_live
            if slot in CONDITIONING_SLOTS:
                market_prefix = market_prefix + self._embed(slot, choice).to(
                    dtype=market_base.dtype
                )

        # --- canonical token assembly -------------------------------------------
        unit_zero = torch.zeros_like(unit_choice[SLOT["unit_kind"]])
        market_zero = torch.zeros_like(kind)
        unit_tokens = torch.stack(
            [
                frames[None, :].expand(rows, -1)
                if slot == SLOT["unit_actor"]
                else unit_choice.get(slot, unit_zero)
                for slot in range(kt.ACTION_SLOTS)
            ],
            dim=-1,
        ).masked_fill(~unit_live[..., None], 0)
        market_tokens = torch.stack(
            [
                (positions[None, :] == stop_position[:, None]).long()
                if slot == SLOT["stop"]
                else market_choice.get(slot, market_zero)
                for slot in range(kt.ACTION_SLOTS)
            ],
            dim=-1,
        ).masked_fill(~market_live[..., None], 0)
        unit_zero_f = torch.zeros_like(unit_logp[SLOT["unit_kind"]])
        market_zero_f = torch.zeros_like(kind_logp)
        scatter = market_frame[..., None].expand(-1, -1, kt.ACTION_SLOTS)
        pad = (0, 0, 0, kt.MAX_FRAMES - actors)

        def place(
            unit: dict[int, torch.Tensor], market: dict[int, torch.Tensor]
        ) -> torch.Tensor:
            unit_stack = torch.stack(
                [unit.get(slot, unit_zero_f) for slot in range(kt.ACTION_SLOTS)], -1
            )
            market_stack = torch.stack(
                [market.get(slot, market_zero_f) for slot in range(kt.ACTION_SLOTS)],
                -1,
            )
            return F.pad(unit_stack, pad).scatter(1, scatter, market_stack)

        tokens = F.pad(unit_tokens, pad).scatter(1, scatter, market_tokens)
        log_probs = place(unit_logp, market_logp)
        entropies = place(unit_entropy, market_entropy)
        lengths = torch.where(live, actor_counts + stop_position + 1, 0)
        if supplied_tokens is not None:
            if supplied_lengths is None:
                raise ValueError("replay needs supplied lengths with supplied tokens")
            length_ok = lengths == supplied_lengths
            canonical_ok = (tokens == supplied_tokens).flatten(1).all(dim=-1)
            valid = torch.stack((support, length_ok, canonical_ok), dim=-1)
        else:
            valid = torch.stack(
                (support, torch.ones_like(support), torch.ones_like(support)), dim=-1
            )
        return GrammarPolicyResult(
            tokens=tokens,
            lengths=lengths,
            log_probs=log_probs,
            entropies=entropies,
            valid=valid,
            slot_logits=collected if collect_logits else None,
            kl=place(unit_kl, market_kl) if teacher_logits is not None else None,
        )


_FLAG_MEANINGS = {
    "support": "a chosen value is outside its prefix-dependent grammar mask",
    "length": "lengths disagree with the reconstructed STOP frame",
    "canonical": (
        "tokens differ from the canonical program (actor ordinals, reserved "
        "target, cross-kind fields, STOP bits, padding or inactive rows)"
    ),
}


class GrammarReplayError(ValueError):
    """A replayed program failed one or more flag groups."""

    def __init__(self, groups: tuple[str, ...], first_rows: tuple[int, ...]) -> None:
        self.groups = groups
        self.first_rows = first_rows
        details = "; ".join(
            f"{group} (first row {row}: env {row // kt.PLAYERS}, seat "
            f"{row % kt.PLAYERS}): {_FLAG_MEANINGS[group]}"
            for group, row in zip(groups, first_rows, strict=True)
        )
        super().__init__(
            f"evaluate_actions rejected the supplied program; failing groups: "
            f"{', '.join(groups)} — {details}"
        )


def check_replay_flags(valid: torch.Tensor) -> None:
    """Host admission of a replay: one compact transfer, then branch.

    ``valid`` is ``[rows, 3]`` (``FLAG_GROUPS`` order). Runs outside the tensor
    core and before the result reaches loss computation.
    """
    if valid.shape[0] == 0:
        return
    summary = torch.cat((valid.all(dim=0).long(), (~valid).long().argmax(dim=0)))
    flags = summary.cpu().tolist()
    groups = len(FLAG_GROUPS)
    failing = tuple(
        (name, int(row))
        for name, ok, row in zip(
            FLAG_GROUPS, flags[:groups], flags[groups:], strict=True
        )
        if not ok
    )
    if failing:
        raise GrammarReplayError(
            tuple(name for name, _ in failing), tuple(row for _, row in failing)
        )
