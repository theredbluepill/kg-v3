"""Kaggriculture tensors — `docs/kaggriculture-contract.md` v4 in code.

Every tensor has leading dims ``[env, seat]``; each row is that seat's legal view.
``check_contract`` checks the full contract (dtypes, shapes, categorical ranges,
finiteness) and is for environment admission and tests, not the model hot path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from pydantic import BaseModel, Field

from owl.config import BaseConfig

PLAYERS = 2
CELLS = 100
TILES = 2 * CELLS
MAX_ACTORS = 241
ACTOR_SLOTS = 2 * MAX_ACTORS
SHOP_SLOTS = 8
MAX_FRAMES = 252
ACTION_SLOTS = 12

TILE_KINDS = ("EMPTY", "LOCKED", "WEED", "PLANT", "COOP", "PASTURE")
CROPS = ("NONE", "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("NONE", "GOOSE", "COW", "SHEEP")
ITEMS = (
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG",
    "MILK", "WOOL", "FERTILIZER", "GOOSE", "COW", "SHEEP",
)  # fmt: skip
PRODUCTS = ITEMS[:9]
SHOP_TYPES = (
    "BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP",
    "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE",
)  # fmt: skip
TILE_ROLES = ("OWN_FARM", "RIVAL_FARM")
ACTOR_ROLES = ("OWN_FARMER", "OWN_HAND", "RIVAL_FARMER", "RIVAL_HAND")

UNIT_KINDS = (
    "NONE", "PASS", "NORTH", "SOUTH", "EAST", "WEST", "PICKUP", "PLACE", "PLANT",
    "WATER", "HARVEST", "DROP", "BUILD_COOP", "BUILD_PASTURE", "FEED", "FERTILIZE",
    "COLLECT_FERTILIZER", "CARE", "DIG",
)  # fmt: skip
MARKET_KINDS = (
    "NONE", "HIRE", "BUY_LAND", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL",
    "SELL", "EMPTY",
)  # fmt: skip
ACTION_ITEMS = ("NONE", *ITEMS)
SLOT_NAMES = (
    "unit_actor", "unit_kind", "unit_target", "unit_item", "unit_quantity_present",
    "unit_quantity_high", "unit_quantity", "market_kind", "market_item",
    "market_quantity_high", "market_quantity", "stop",
)  # fmt: skip
SLOT_WIDTHS = (241, 20, 128, 16, 2, 32, 32, 8, 16, 32, 32, 2)

ITEM_COUNT = len(ITEMS)
PRODUCT_COUNT = len(PRODUCTS)
STORAGE_COUNTS = ITEM_COUNT + len(CROPS) - 1
TILE_INT_CHANNELS = 7
TILE_FLOAT_CHANNELS = 15
ACTOR_FLOAT_CHANNELS = 2 + 2 * ITEM_COUNT
PLAYER_FEATURE_CHANNELS = 44
MARKET_FLOAT_CHANNELS = 2
GLOBAL_FEATURE_CHANNELS = 15
GLOBAL_INT_CHANNELS = 16


class KaggricultureObsConfig(BaseConfig):
    obs_spec: Literal["kaggriculture"] = "kaggriculture"
    schema_version: Literal[3] = 3


class KaggricultureActionConfig(BaseConfig):
    action_spec: Literal["kaggriculture"] = "kaggriculture"
    hire_limit: int = Field(default=MAX_ACTORS, ge=1, le=MAX_ACTORS)


@dataclass(frozen=True)
class KaggricultureActionMask:
    """Potential command positions.

    Prefix-dependent vocabulary masks come from the native grammar.
    """

    can_act: torch.Tensor


@dataclass
class KaggricultureActions:
    tokens: torch.Tensor
    lengths: torch.Tensor


_I64, _F32, _F64, _BOOL = torch.int64, torch.float32, torch.float64, torch.bool

# field: (dtype, trailing shape, exclusive categorical upper bound or None)
_SCHEMA: dict[str, tuple[torch.dtype, tuple[int, ...], int | None]] = {
    "tile_kind": (_I64, (TILES,), len(TILE_KINDS)),
    "tile_crop": (_I64, (TILES,), len(CROPS)),
    "tile_animal": (_I64, (TILES,), len(ANIMALS)),
    "tile_cell": (_I64, (TILES,), CELLS),
    "tile_role": (_I64, (TILES,), len(TILE_ROLES)),
    "tiles_int": (_I64, (TILES, TILE_INT_CHANNELS), None),
    "tiles_float": (_F32, (TILES, TILE_FLOAT_CHANNELS), None),
    "actor_slot": (_I64, (ACTOR_SLOTS,), MAX_ACTORS),
    "actor_cell": (_I64, (ACTOR_SLOTS,), CELLS),
    "actor_role": (_I64, (ACTOR_SLOTS,), len(ACTOR_ROLES)),
    "actor_mask": (_BOOL, (ACTOR_SLOTS,), None),
    "actor_inventory": (_I64, (MAX_ACTORS, ITEM_COUNT), None),
    "actor_inventory_rank": (_I64, (MAX_ACTORS, ITEM_COUNT), None),
    "actors_float": (_F32, (ACTOR_SLOTS, ACTOR_FLOAT_CHANNELS), None),
    "player_features": (_F32, (PLAYERS, PLAYER_FEATURE_CHANNELS), None),
    "storage_counts": (_I64, (STORAGE_COUNTS,), None),
    "storage_rank": (_I64, (ITEM_COUNT,), None),
    "banks": (_F64, (PLAYERS,), None),
    "shop_type": (_I64, (SHOP_SLOTS,), len(SHOP_TYPES)),
    "shop_slot": (_I64, (SHOP_SLOTS,), SHOP_SLOTS),
    "shop_mask": (_BOOL, (SHOP_SLOTS,), None),
    "market_product": (_I64, (PRODUCT_COUNT,), PRODUCT_COUNT),
    "market_float": (_F32, (PRODUCT_COUNT, MARKET_FLOAT_CHANNELS), None),
    "market_int": (_I64, (PRODUCT_COUNT, MARKET_FLOAT_CHANNELS), None),
    "global_features": (_F32, (GLOBAL_FEATURE_CHANNELS,), None),
    "globals_int": (_I64, (GLOBAL_INT_CHANNELS,), None),
    "still_playing": (_BOOL, (), None),
    "order_limits": (_I64, (), None),
}


class KaggricultureObsBatch(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    tile_kind: torch.Tensor
    tile_crop: torch.Tensor
    tile_animal: torch.Tensor
    tile_cell: torch.Tensor
    tile_role: torch.Tensor
    tiles_int: torch.Tensor
    tiles_float: torch.Tensor
    actor_slot: torch.Tensor
    actor_cell: torch.Tensor
    actor_role: torch.Tensor
    actor_mask: torch.Tensor
    actor_inventory: torch.Tensor
    actor_inventory_rank: torch.Tensor
    actors_float: torch.Tensor
    player_features: torch.Tensor
    storage_counts: torch.Tensor
    storage_rank: torch.Tensor
    banks: torch.Tensor
    shop_type: torch.Tensor
    shop_slot: torch.Tensor
    shop_mask: torch.Tensor
    market_product: torch.Tensor
    market_float: torch.Tensor
    market_int: torch.Tensor
    global_features: torch.Tensor
    globals_int: torch.Tensor
    still_playing: torch.Tensor
    order_limits: torch.Tensor
    action_mask: KaggricultureActionMask

    def check_contract(self) -> None:
        """Check contract v4; raise ``ValueError`` naming the first bad field."""
        lead = self.still_playing.shape
        if len(lead) < 1 or lead[-1] != PLAYERS:
            raise ValueError(
                f"still_playing must end in the seat axis ({PLAYERS}), "
                f"got {tuple(lead)}"
            )
        # Iterating the pinned schema's known field names is the sanctioned
        # dynamic-access case (AGENTS.md).
        for name, (dtype, trailing, bound) in _SCHEMA.items():
            tensor: torch.Tensor = getattr(self, name)
            if tensor.dtype != dtype:
                raise ValueError(f"{name} must be {dtype}, got {tensor.dtype}")
            if tuple(tensor.shape) != (*lead, *trailing):
                raise ValueError(
                    f"{name} must have shape {(*lead, *trailing)}, "
                    f"got {tuple(tensor.shape)}"
                )
            if (
                bound is not None
                and tensor.numel()
                and (int(tensor.min()) < 0 or int(tensor.max()) >= bound)
            ):
                raise ValueError(f"{name} categories must lie in [0, {bound})")
            if dtype in (_F32, _F64) and not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{name} contains non-finite values")
        can_act = self.action_mask.can_act
        if can_act.dtype != _BOOL or tuple(can_act.shape) != (*lead, MAX_FRAMES):
            raise ValueError(
                f"action_mask.can_act must be bool {(*lead, MAX_FRAMES)}, "
                f"got {can_act.dtype} {tuple(can_act.shape)}"
            )
