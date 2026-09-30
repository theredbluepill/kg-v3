"""Tiny synthetic Kaggriculture batches that satisfy contract v4.

``make_obs`` builds one small game per env (two farms, each player's actors,
sheds, shops and market) and then writes each seat's legal view of it, so the
rows obey the contract's zero-fill, privacy and mask rules rather than only its
dtypes and ranges.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pytest
import torch
from owl.kaggriculture import types as kt
from owl.model import compile_gemm

_EPISODE_STEPS = 720
_TURNS_PER_DAY = 24
_SHED_CAPACITY = 100
_PLANT = kt.TILE_KINDS.index("PLANT")
_COOP = kt.TILE_KINDS.index("COOP")
_PASTURE = kt.TILE_KINDS.index("PASTURE")
_EMPTY = kt.TILE_KINDS.index("EMPTY")
_LOCKED = kt.TILE_KINDS.index("LOCKED")
_GOOSE = kt.ANIMALS.index("GOOSE")
_COW = kt.ANIMALS.index("COW")
_SHEEP = kt.ANIMALS.index("SHEEP")
_OWN_PRIVATE_PLAYER_CHANNELS = slice(11, 42)
_SHED_ITEMS = kt.ITEM_COUNT


PROBED_GPU_STACK = compile_gemm.InstalledCompileStack(
    torch=f"{compile_gemm.KAGGRICULTURE_PROBED_COMPILE_STACK.torch}+cu128",
    triton=compile_gemm.KAGGRICULTURE_PROBED_COMPILE_STACK.triton,
    cuda_available=True,
    # One host reports one driver; the first probed one stands in for it.
    nvidia_drivers=compile_gemm.KAGGRICULTURE_PROBED_COMPILE_STACK.nvidia_drivers[:1],
)


@pytest.fixture
def probed_compile_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read this host as the probed GPU stack, so compile tests are hermetic.

    Without it the stack check reads the real host: skipped checks on the Mac,
    and a rejection on a GPU host whose driver was never probed.
    """
    monkeypatch.setattr(
        compile_gemm, "installed_compile_stack", lambda: PROBED_GPU_STACK
    )


@dataclass(frozen=True)
class _Player:
    actor_cells: torch.Tensor  # [n]
    inventory: torch.Tensor  # [n, I]
    inventory_rank: torch.Tensor  # [n, I]
    storage: torch.Tensor  # [17]
    storage_rank: torch.Tensor  # [I]
    bank: float
    hires_today: int
    tile_kind: torch.Tensor  # [100]
    tile_crop: torch.Tensor
    tile_animal: torch.Tensor
    tiles_int: torch.Tensor  # [100, 7]
    tiles_float: torch.Tensor  # [100, 15]
    public: torch.Tensor  # player_features channels 0..10
    private: torch.Tensor  # player_features channels 11..41


def _per_env(value: int | Sequence[int], envs: int, name: str) -> list[int]:
    counts = [value] * envs if isinstance(value, int) else list(value)
    if len(counts) != envs:
        raise ValueError(f"{name} needs one count per env ({envs}), got {counts}")
    return counts


def _ranked_keys(generator: torch.Generator, counts: torch.Tensor) -> torch.Tensor:
    """IndexMap-style ranks: present keys get 1..k in a random insertion order."""
    present = torch.rand(counts.shape, generator=generator) < 0.6
    counts.mul_(present)
    rank = torch.zeros_like(counts)
    for row in range(counts.shape[0]):
        order = torch.randperm(counts.shape[1], generator=generator)
        keys = [int(k) for k in order if present[row, k]]
        for position, key in enumerate(keys, start=1):
            rank[row, key] = position
    return rank


def _farm(
    generator: torch.Generator, *, step: int, day: int
) -> tuple[torch.Tensor, ...]:
    """One farm's 100 tiles with the contract's applicability rules."""

    def randint(low: int, high: int) -> int:
        return int(torch.randint(low, high, (), generator=generator))

    kind = torch.randint(0, len(kt.TILE_KINDS), (kt.CELLS,), generator=generator)
    crop = torch.zeros(kt.CELLS, dtype=torch.int64)
    animal = torch.zeros(kt.CELLS, dtype=torch.int64)
    ints = torch.zeros((kt.CELLS, kt.TILE_INT_CHANNELS), dtype=torch.int64)
    floats = torch.zeros((kt.CELLS, kt.TILE_FLOAT_CHANNELS))
    for cell in range(kt.CELLS):
        tile_kind = int(kind[cell])
        if tile_kind == _PLANT:
            crop[cell] = randint(1, len(kt.CROPS))
            yield_units, planted = randint(0, 8), randint(0, day + 1)
            lifespan = -1 if randint(0, 2) else step + randint(-20, 100)
            fertilized = -1 if randint(0, 2) else randint(0, day + 10)
            unwatered, watered = randint(0, 4), randint(0, 2)
            ints[cell] = torch.tensor(
                [yield_units, planted, lifespan, fertilized, 0, unwatered, 0]
            )
            floats[cell, :9] = torch.tensor(
                [
                    yield_units / 8,
                    watered,
                    unwatered / 8,
                    (day - planted) / 30,
                    float(lifespan >= 0),
                    (lifespan - step) / _EPISODE_STEPS if lifespan >= 0 else 0.0,
                    float(fertilized >= 0),
                    float(fertilized >= day),
                    max(0, fertilized - day) / 30,
                ]
            )
        elif tile_kind in (_COOP, _PASTURE):
            choices = (0, _GOOSE) if tile_kind == _COOP else (0, _COW, _SHEEP)
            animal[cell] = choices[randint(0, len(choices))]
            if animal[cell]:
                yield_units, placed, unfed = (
                    randint(0, 8),
                    randint(0, day + 1),
                    randint(0, 4),
                )
                ints[cell] = torch.tensor([yield_units, 0, 0, 0, placed, 0, unfed])
                floats[cell, 0] = yield_units / 8
                floats[cell, 9:] = torch.tensor(
                    [
                        (day - placed) / 30,
                        unfed / 8,
                        randint(0, 2),
                        randint(0, 2),
                        randint(0, 2),
                        randint(0, 4) / 8,
                    ]
                )
    return kind, crop, animal, ints, floats


_FARM_HAND_COST_MULT = 100


def _next_hire_cost(hires_today: int) -> int:
    """Engine rule: ``farmHandCostMult * fib(hires_today)``, fib = 1, 1, 2, 3, 5."""
    a, b = 1, 1
    for _ in range(hires_today):
        a, b = b, a + b
    return _FARM_HAND_COST_MULT * a


def _player(generator: torch.Generator, *, actors: int, step: int, day: int) -> _Player:
    if not 1 <= actors <= kt.MAX_ACTORS:
        raise ValueError(f"actor count must be in [1, {kt.MAX_ACTORS}], got {actors}")
    inventory = torch.randint(0, 40, (actors, kt.ITEM_COUNT), generator=generator)
    inventory_rank = _ranked_keys(generator, inventory)
    storage = torch.randint(0, 50, (kt.STORAGE_COUNTS,), generator=generator)
    shed = storage[:_SHED_ITEMS].unsqueeze(0)
    storage_rank = _ranked_keys(generator, shed)[0]  # zeroes absent shed keys
    kind, crop, animal, tiles_int, tiles_float = _farm(generator, step=step, day=day)
    bank = float(torch.rand((), generator=generator, dtype=torch.float64) * 1e4)
    hires = int(torch.randint(0, 5, (), generator=generator))
    quadrants = torch.randint(0, 2, (4,), generator=generator).float()
    public = torch.cat(
        (
            torch.tensor([bank / 200_000, actors / kt.MAX_ACTORS]),
            quadrants.sum().reshape(1) / 4,
            quadrants,
            torch.tensor(
                [
                    float((kind == _EMPTY).sum()) / 100,
                    float((kind == _LOCKED).sum()) / 100,
                    hires / 240,
                    _next_hire_cost(hires) / 200_000,
                ]
            ),
        )
    )
    used = int(storage[:_SHED_ITEMS].sum())
    private = torch.cat(
        (
            storage[:_SHED_ITEMS] / 100,
            storage[_SHED_ITEMS:] / 32,
            storage_rank / 12,
            torch.tensor([used, _SHED_CAPACITY - used]) / _SHED_CAPACITY,
        )
    ).float()
    return _Player(
        actor_cells=torch.randint(0, kt.CELLS, (actors,), generator=generator),
        inventory=inventory,
        inventory_rank=inventory_rank,
        storage=storage,
        storage_rank=storage_rank,
        bank=bank,
        hires_today=hires,
        tile_kind=kind,
        tile_crop=crop,
        tile_animal=animal,
        tiles_int=tiles_int,
        tiles_float=tiles_float,
        public=public,
        private=private,
    )


def make_obs(
    envs: int = 1,
    *,
    own_actors: int | Sequence[int] = 2,
    rival_actors: int | Sequence[int] = 1,
    shops: int | Sequence[int] = 3,
    order_limit: int = 3,
) -> kt.KaggricultureObsBatch:
    """A contract-valid batch of ``envs`` games, each seen from both seats.

    Player 0 owns ``own_actors`` actors and player 1 ``rival_actors`` (an int, or
    one count per env); seat ``s`` is player ``s``'s view, so seat 1 sees the
    counts swapped. ``shops`` shops are unlocked per env.
    """
    own_counts = _per_env(own_actors, envs, "own_actors")
    rival_counts = _per_env(rival_actors, envs, "rival_actors")
    shop_counts = _per_env(shops, envs, "shops")
    # The fixture writes one value as both the per-turn limit and the configured
    # maximum, whose supported envelope is 1..MAX_ORDER_LIMIT.
    if not 1 <= order_limit <= kt.MAX_ORDER_LIMIT:
        raise ValueError(f"order_limit must be in [1, {kt.MAX_ORDER_LIMIT}]")
    generator = torch.Generator().manual_seed(7)
    lead = (envs, kt.PLAYERS)

    def zeros(*shape: int, dtype: torch.dtype = torch.int64) -> torch.Tensor:
        return torch.zeros((*lead, *shape), dtype=dtype)

    out = {
        "tile_kind": zeros(kt.TILES),
        "tile_crop": zeros(kt.TILES),
        "tile_animal": zeros(kt.TILES),
        "tiles_int": zeros(kt.TILES, kt.TILE_INT_CHANNELS),
        "tiles_float": zeros(kt.TILES, kt.TILE_FLOAT_CHANNELS, dtype=torch.float32),
        "actor_slot": zeros(kt.ACTOR_SLOTS),
        "actor_cell": zeros(kt.ACTOR_SLOTS),
        "actor_role": zeros(kt.ACTOR_SLOTS),
        "actor_mask": zeros(kt.ACTOR_SLOTS, dtype=torch.bool),
        "actor_inventory": zeros(kt.MAX_ACTORS, kt.ITEM_COUNT),
        "actor_inventory_rank": zeros(kt.MAX_ACTORS, kt.ITEM_COUNT),
        "actors_float": zeros(
            kt.ACTOR_SLOTS, kt.ACTOR_FLOAT_CHANNELS, dtype=torch.float32
        ),
        "player_features": zeros(
            kt.PLAYERS, kt.PLAYER_FEATURE_CHANNELS, dtype=torch.float32
        ),
        "storage_counts": zeros(kt.STORAGE_COUNTS),
        "storage_rank": zeros(kt.ITEM_COUNT),
        "banks": zeros(kt.PLAYERS, dtype=torch.float64),
        "shop_type": zeros(kt.SHOP_SLOTS),
        "shop_slot": zeros(kt.SHOP_SLOTS),
        "shop_mask": zeros(kt.SHOP_SLOTS, dtype=torch.bool),
        "market_float": zeros(
            kt.PRODUCT_COUNT, kt.MARKET_FLOAT_CHANNELS, dtype=torch.float32
        ),
        "market_int": zeros(kt.PRODUCT_COUNT, kt.MARKET_FLOAT_CHANNELS),
        "global_features": zeros(kt.GLOBAL_FEATURE_CHANNELS, dtype=torch.float32),
        "globals_int": zeros(kt.GLOBAL_INT_CHANNELS),
    }
    can_act = zeros(kt.MAX_FRAMES, dtype=torch.bool)
    frames = torch.arange(kt.MAX_FRAMES)
    for env in range(envs):
        step = int(torch.randint(0, _EPISODE_STEPS - 1, (), generator=generator))
        day, hour = divmod(step, _TURNS_PER_DAY)
        players = [
            _player(generator, actors=count, step=step, day=day)
            for count in (own_counts[env], rival_counts[env])
        ]
        shop_count = shop_counts[env]
        shop_types = torch.randint(
            0, len(kt.SHOP_TYPES), (shop_count,), generator=generator
        )
        market_int = torch.stack(
            (
                torch.randint(-100, 300, (kt.PRODUCT_COUNT,), generator=generator),
                torch.randint(1, 300, (kt.PRODUCT_COUNT,), generator=generator),
            ),
            dim=-1,
        )
        config_ints = [
            _EPISODE_STEPS, _TURNS_PER_DAY, order_limit, _SHED_CAPACITY,
            100, 48, 24, 72, 500,
        ]  # fmt: skip
        global_features = torch.tensor(
            [
                step / _EPISODE_STEPS,
                hour / _TURNS_PER_DAY,
                day / (_EPISODE_STEPS / _TURNS_PER_DAY),
                max(0, _EPISODE_STEPS - 1 - step) / _EPISODE_STEPS,
                _TURNS_PER_DAY / 24,
                order_limit / 10,
                _SHED_CAPACITY / 1000,
                100 / 100,
                _EPISODE_STEPS / 1000,
                0.05,
                48 / _TURNS_PER_DAY,
                24 / _TURNS_PER_DAY,
                72 / _TURNS_PER_DAY,
                500 / 200_000,
                shop_count / 8,
            ]
        )
        for seat in range(kt.PLAYERS):
            me, rival = players[seat], players[1 - seat]
            own_n, rival_n = me.actor_cells.numel(), rival.actor_cells.numel()
            view = (env, seat)
            out["tile_kind"][view] = torch.cat((me.tile_kind, rival.tile_kind))
            out["tile_crop"][view] = torch.cat((me.tile_crop, rival.tile_crop))
            out["tile_animal"][view] = torch.cat((me.tile_animal, rival.tile_animal))
            out["tiles_int"][view] = torch.cat((me.tiles_int, rival.tiles_int))
            out["tiles_float"][view] = torch.cat((me.tiles_float, rival.tiles_float))
            for offset, owner, count, roles in (
                (0, me, own_n, (0, 1)),
                (kt.MAX_ACTORS, rival, rival_n, (2, 3)),
            ):
                rows = slice(offset, offset + count)
                out["actor_slot"][(*view, rows)] = torch.arange(count)
                out["actor_cell"][(*view, rows)] = owner.actor_cells
                out["actor_role"][(*view, rows)] = roles[1]
                out["actor_role"][(*view, offset)] = roles[0]
                out["actor_mask"][(*view, rows)] = True
                xy = torch.stack(
                    (owner.actor_cells % 10, owner.actor_cells // 10), dim=-1
                )
                out["actors_float"][(*view, rows, slice(0, 2))] = xy / 9
            own_rows = slice(0, own_n)
            out["actor_inventory"][(*view, own_rows)] = me.inventory
            out["actor_inventory_rank"][(*view, own_rows)] = me.inventory_rank
            out["actors_float"][(*view, own_rows, slice(2, 14))] = me.inventory / 32
            out["actors_float"][(*view, own_rows, slice(14, 26))] = (
                me.inventory_rank / 12
            )
            out["player_features"][(*view, 0, slice(0, 11))] = me.public
            out["player_features"][(*view, 0, _OWN_PRIVATE_PLAYER_CHANNELS)] = (
                me.private
            )
            out["player_features"][(*view, 1, slice(0, 11))] = rival.public
            out["storage_counts"][view] = me.storage
            out["storage_rank"][view] = me.storage_rank
            out["banks"][view] = torch.tensor(
                [me.bank, rival.bank], dtype=torch.float64
            )
            out["shop_type"][(*view, slice(0, shop_count))] = shop_types
            out["shop_slot"][(*view, slice(0, shop_count))] = torch.arange(shop_count)
            out["shop_mask"][(*view, slice(0, shop_count))] = True
            out["market_int"][view] = market_int
            out["market_float"][view] = market_int / torch.tensor([10_000.0, 250.0])
            out["global_features"][view] = global_features
            out["globals_int"][view] = torch.tensor(
                [
                    step, day, hour, *config_ints,
                    me.hires_today, rival.hires_today, own_n, rival_n,
                ]
            )  # fmt: skip
            can_act[view] = frames < own_n + order_limit + 1
    return kt.KaggricultureObsBatch(
        **out,
        tile_cell=torch.arange(kt.CELLS).repeat(2).expand(*lead, kt.TILES).clone(),
        tile_role=torch.arange(kt.PLAYERS)
        .repeat_interleave(kt.CELLS)
        .expand(*lead, kt.TILES)
        .clone(),
        market_product=torch.arange(kt.PRODUCT_COUNT)
        .expand(*lead, kt.PRODUCT_COUNT)
        .clone(),
        still_playing=torch.ones(lead, dtype=torch.bool),
        order_limits=torch.full(lead, order_limit, dtype=torch.int64),
        action_mask=kt.KaggricultureActionMask(can_act=can_act),
    )


@pytest.fixture
def obs() -> kt.KaggricultureObsBatch:
    return make_obs()
