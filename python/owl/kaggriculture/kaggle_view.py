"""One live seat's Kaggle observation, encoded by the training ``write_seat``.

A Kaggle agent sees only its own seat: the shared public state plus its own
``private``. This module builds the native seat view from that observation and
the Kaggle configuration, then fills a reusable ``[1, 1]`` observation batch
through ``owl.rs.encode_kaggriculture_seat_into``, the same Rust ``write_seat``
the training environment uses. There is no Python encoder.

Kaggle's local loader adds ``__raw_path__`` to the configuration, which the
native configuration envelope rejects, so the game configuration is rebuilt
from an explicit allowlist. ``player``, ``remainingOverageTime`` and any seed
never reach the model; the seat only selects which row is written.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from owl import rs
from owl.kaggriculture.types import KaggricultureGameConfig, KaggricultureObsBatch

# The pinned kaggriculture.json configuration keys plus Kaggle's framework keys
# (actTimeout, runTimeout, seed); these are exactly the game config's aliases.
GAME_CONFIGURATION_KEYS: frozenset[str] = frozenset(
    field.alias if field.alias is not None else name
    for name, field in KaggricultureGameConfig.model_fields.items()
)
PUBLIC_KEYS = ("step", "day", "hour", "farms", "market", "town")


def game_configuration(
    configuration: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Return the allowlisted game configuration and the dropped key names.

    Insertion order is preserved; values are passed through unchanged for the
    native envelope to admit.
    """
    kept = {
        key: value
        for key, value in configuration.items()
        if key in GAME_CONFIGURATION_KEYS
    }
    dropped = tuple(key for key in configuration if key not in kept)
    return kept, dropped


def seat_view_json(
    observation: Mapping[str, Any], configuration: Mapping[str, Any]
) -> tuple[str, int, tuple[str, ...]]:
    """Build ``(view_json, seat, dropped_config_keys)`` from one Kaggle call.

    Reads only the shared public keys and this seat's own ``private``. JSON is
    written without ``sort_keys``: inventory and shed ranks follow insertion
    order.
    """
    seat = observation["player"]
    if type(seat) is not int or seat not in (0, 1):
        raise ValueError(f"observation player must be 0 or 1, got {seat!r}")
    game, dropped = game_configuration(configuration)
    view = {
        "configuration": game,
        "public": {key: observation[key] for key in PUBLIC_KEYS},
        "private": observation["private"],
    }
    return json.dumps(view, allow_nan=False, separators=(",", ":")), seat, dropped


@dataclass(frozen=True)
class SeatArrays:
    """Persistent NumPy views of one ``[1, 1]`` batch, built once."""

    tile_kind: NDArray[np.int64]
    tile_crop: NDArray[np.int64]
    tile_animal: NDArray[np.int64]
    tile_cell: NDArray[np.int64]
    tile_role: NDArray[np.int64]
    tiles_int: NDArray[np.int64]
    tiles_float: NDArray[np.float32]
    actor_slot: NDArray[np.int64]
    actor_cell: NDArray[np.int64]
    actor_role: NDArray[np.int64]
    actor_mask: NDArray[np.bool_]
    actor_inventory: NDArray[np.int64]
    actor_inventory_rank: NDArray[np.int64]
    actors_float: NDArray[np.float32]
    player_features: NDArray[np.float32]
    storage_counts: NDArray[np.int64]
    storage_rank: NDArray[np.int64]
    banks: NDArray[np.float64]
    shop_type: NDArray[np.int64]
    shop_slot: NDArray[np.int64]
    shop_mask: NDArray[np.bool_]
    market_product: NDArray[np.int64]
    market_float: NDArray[np.float32]
    market_int: NDArray[np.int64]
    global_features: NDArray[np.float32]
    globals_int: NDArray[np.int64]
    still_playing: NDArray[np.bool_]
    order_limits: NDArray[np.int64]
    can_act: NDArray[np.bool_]

    @classmethod
    def of(cls, obs: KaggricultureObsBatch) -> SeatArrays:
        if tuple(obs.still_playing.shape) != (1, 1):
            raise ValueError("seat arrays require a [1, 1] observation batch")
        return cls(
            tile_kind=obs.tile_kind.numpy(),
            tile_crop=obs.tile_crop.numpy(),
            tile_animal=obs.tile_animal.numpy(),
            tile_cell=obs.tile_cell.numpy(),
            tile_role=obs.tile_role.numpy(),
            tiles_int=obs.tiles_int.numpy(),
            tiles_float=obs.tiles_float.numpy(),
            actor_slot=obs.actor_slot.numpy(),
            actor_cell=obs.actor_cell.numpy(),
            actor_role=obs.actor_role.numpy(),
            actor_mask=obs.actor_mask.numpy(),
            actor_inventory=obs.actor_inventory.numpy(),
            actor_inventory_rank=obs.actor_inventory_rank.numpy(),
            actors_float=obs.actors_float.numpy(),
            player_features=obs.player_features.numpy(),
            storage_counts=obs.storage_counts.numpy(),
            storage_rank=obs.storage_rank.numpy(),
            banks=obs.banks.numpy(),
            shop_type=obs.shop_type.numpy(),
            shop_slot=obs.shop_slot.numpy(),
            shop_mask=obs.shop_mask.numpy(),
            market_product=obs.market_product.numpy(),
            market_float=obs.market_float.numpy(),
            market_int=obs.market_int.numpy(),
            global_features=obs.global_features.numpy(),
            globals_int=obs.globals_int.numpy(),
            still_playing=obs.still_playing.numpy(),
            order_limits=obs.order_limits.numpy(),
            can_act=obs.action_mask.can_act.numpy(),
        )


def encode_seat_into(view_json: str, seat: int, out: SeatArrays) -> None:
    """Fully overwrite ``out`` with one seat's row; a rejected view writes nothing."""
    rs.encode_kaggriculture_seat_into(
        view_json,
        seat,
        tile_kind=out.tile_kind,
        tile_crop=out.tile_crop,
        tile_animal=out.tile_animal,
        tile_cell=out.tile_cell,
        tile_role=out.tile_role,
        tiles_int=out.tiles_int,
        tiles_float=out.tiles_float,
        actor_slot=out.actor_slot,
        actor_cell=out.actor_cell,
        actor_role=out.actor_role,
        actor_mask=out.actor_mask,
        actor_inventory=out.actor_inventory,
        actor_inventory_rank=out.actor_inventory_rank,
        actors_float=out.actors_float,
        player_features=out.player_features,
        storage_counts=out.storage_counts,
        storage_rank=out.storage_rank,
        banks=out.banks,
        shop_type=out.shop_type,
        shop_slot=out.shop_slot,
        shop_mask=out.shop_mask,
        market_product=out.market_product,
        market_float=out.market_float,
        market_int=out.market_int,
        global_features=out.global_features,
        globals_int=out.globals_int,
        still_playing=out.still_playing,
        order_limits=out.order_limits,
        can_act=out.can_act,
    )
