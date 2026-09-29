"""Tiny synthetic Kaggriculture batches that satisfy contract v4."""

from __future__ import annotations

import pytest
import torch
from owl.kaggriculture import types as kt


def make_obs(
    envs: int = 1, *, own_actors: int = 2, rival_actors: int = 1
) -> kt.KaggricultureObsBatch:
    """A valid batch with every contract field, a few present actors and shops."""
    lead = (envs, kt.PLAYERS)
    generator = torch.Generator().manual_seed(7)

    def ints(*shape: int, high: int) -> torch.Tensor:
        return torch.randint(0, high, (*lead, *shape), generator=generator)

    def floats(*shape: int) -> torch.Tensor:
        return torch.randn((*lead, *shape), generator=generator)

    actor_mask = torch.zeros((*lead, kt.ACTOR_SLOTS), dtype=torch.bool)
    actor_mask[..., :own_actors] = True
    actor_mask[..., kt.MAX_ACTORS : kt.MAX_ACTORS + rival_actors] = True
    shop_mask = torch.zeros((*lead, kt.SHOP_SLOTS), dtype=torch.bool)
    shop_mask[..., :3] = True
    return kt.KaggricultureObsBatch(
        tile_kind=ints(kt.TILES, high=len(kt.TILE_KINDS)),
        tile_crop=ints(kt.TILES, high=len(kt.CROPS)),
        tile_animal=ints(kt.TILES, high=len(kt.ANIMALS)),
        tile_cell=torch.arange(kt.CELLS).repeat(2).expand(*lead, kt.TILES).clone(),
        tile_role=torch.cat(
            (
                torch.zeros(kt.CELLS, dtype=torch.int64),
                torch.ones(kt.CELLS, dtype=torch.int64),
            )
        )
        .expand(*lead, kt.TILES)
        .clone(),
        tiles_int=ints(kt.TILES, kt.TILE_INT_CHANNELS, high=5) - 1,
        tiles_float=floats(kt.TILES, kt.TILE_FLOAT_CHANNELS),
        actor_slot=ints(kt.ACTOR_SLOTS, high=kt.MAX_ACTORS),
        actor_cell=ints(kt.ACTOR_SLOTS, high=kt.CELLS),
        actor_role=ints(kt.ACTOR_SLOTS, high=len(kt.ACTOR_ROLES)),
        actor_mask=actor_mask,
        actor_inventory=ints(kt.MAX_ACTORS, kt.ITEM_COUNT, high=40),
        actor_inventory_rank=ints(kt.MAX_ACTORS, kt.ITEM_COUNT, high=kt.ITEM_COUNT + 1),
        actors_float=floats(kt.ACTOR_SLOTS, kt.ACTOR_FLOAT_CHANNELS),
        player_features=floats(kt.PLAYERS, kt.PLAYER_FEATURE_CHANNELS),
        storage_counts=ints(kt.STORAGE_COUNTS, high=500),
        storage_rank=ints(kt.ITEM_COUNT, high=kt.ITEM_COUNT + 1),
        banks=torch.rand((*lead, kt.PLAYERS), generator=generator, dtype=torch.float64)
        * 1e4,
        shop_type=ints(kt.SHOP_SLOTS, high=len(kt.SHOP_TYPES)),
        shop_slot=torch.arange(kt.SHOP_SLOTS).expand(*lead, kt.SHOP_SLOTS).clone(),
        shop_mask=shop_mask,
        market_product=torch.arange(kt.PRODUCT_COUNT)
        .expand(*lead, kt.PRODUCT_COUNT)
        .clone(),
        market_float=floats(kt.PRODUCT_COUNT, kt.MARKET_FLOAT_CHANNELS),
        market_int=ints(kt.PRODUCT_COUNT, kt.MARKET_FLOAT_CHANNELS, high=300),
        global_features=floats(kt.GLOBAL_FEATURE_CHANNELS),
        globals_int=ints(kt.GLOBAL_INT_CHANNELS, high=720),
        still_playing=torch.ones(lead, dtype=torch.bool),
        order_limits=torch.full(lead, 10, dtype=torch.int64),
        action_mask=kt.KaggricultureActionMask(
            can_act=torch.ones((*lead, kt.MAX_FRAMES), dtype=torch.bool)
        ),
    )


@pytest.fixture
def obs() -> kt.KaggricultureObsBatch:
    return make_obs()
