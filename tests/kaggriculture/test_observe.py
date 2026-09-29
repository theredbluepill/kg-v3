"""Real Rust → NumPy → torch → Task 2.1 observation acceptance.

The real schema import is deliberately unconditional. An unmerged schema or a
missing final oracle is an integration failure, not an optional test dependency.
"""

from __future__ import annotations

import copy
import gzip
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from owl import rs
from owl.kaggriculture.types import (
    _SCHEMA,
    MAX_ACTORS,
    MAX_FRAMES,
    PLAYERS,
    KaggricultureActionMask,
    KaggricultureObsBatch,
)


def _header() -> dict[str, Any]:
    products = [
        "WHEAT",
        "CARROT",
        "TOMATO",
        "STRAWBERRY",
        "MELON",
        "EGG",
        "MILK",
        "WOOL",
        "FERTILIZER",
    ]
    farms = []
    for seat in range(2):
        tiles = [[None for _ in range(10)] for _ in range(10)]
        if seat == 0:
            tiles[2][7] = {
                "kind": "PLANT",
                "crop": "TOMATO",
                "yield_units": 3,
                "planted_day": 1,
                "max_lifespan_step": 200,
                "fertilized_until_day": -1,
                "watered_today": True,
                "consecutive_unwatered": 2,
            }
        farms.append(
            {
                "money": 3000.00001 if seat == 0 else -200.0,
                "tiles": tiles,
                "farmer": [7, 2] if seat == 0 else [2, 7],
                "hands": [],
                "unlocked_quadrants": ["SE", "NW"] if seat == 0 else ["NW"],
                "hires_today": 4 + seat,
            }
        )
    return {
        "format": "kaggriculture-re-parity-v1",
        "seed": 42,
        "configuration": {
            "episodeSteps": 720,
            "boardSize": 10,
            "startingMoney": 3000,
            "maxMarketOrdersPerTurn": 3,
            "turnsPerDay": 24,
            "shedCapacity": 100,
            "weedSpawnChance": 0.005,
            "townShopUnlockInterval": 3,
            "townShopSellInterval": 4,
            "townCenterSellInterval": 24,
            "farmHandCostMult": 7,
            "marketParams": {},
        },
        "initial": {
            "public": {
                "step": 239,
                "day": 9,
                "hour": 23,
                "farms": farms,
                "market": {
                    "inventory": {name: -12345 + i for i, name in enumerate(products)},
                    "prices": {name: 7777 + i for i, name in enumerate(products)},
                },
                "town": {"unlocked_shops": ["PIZZA_SHOP", "BAKERY", "PIZZA_SHOP"]},
            },
            "privates": [
                {
                    "shed": {"WOOL": 0, "WHEAT": 7},
                    "seeds": {"TOMATO": 3},
                    "inventories": [{"WOOL": 0, "WHEAT": 7}],
                },
                {"shed": {"MILK": 4}, "seeds": {}, "inventories": [{}]},
            ],
        },
        "shop_schedule": [],
        "rng_schedule": [],
        "terminal_banks": [],
        "transitions": 0,
    }


def _write(headers: str, arrays: dict[str, Any]) -> None:
    rs.encode_kaggriculture_headers_into(
        headers,
        tile_kind=arrays["tile_kind"],
        tile_crop=arrays["tile_crop"],
        tile_animal=arrays["tile_animal"],
        tile_cell=arrays["tile_cell"],
        tile_role=arrays["tile_role"],
        tiles_int=arrays["tiles_int"],
        tiles_float=arrays["tiles_float"],
        actor_slot=arrays["actor_slot"],
        actor_cell=arrays["actor_cell"],
        actor_role=arrays["actor_role"],
        actor_mask=arrays["actor_mask"],
        actor_inventory=arrays["actor_inventory"],
        actor_inventory_rank=arrays["actor_inventory_rank"],
        actors_float=arrays["actors_float"],
        player_features=arrays["player_features"],
        storage_counts=arrays["storage_counts"],
        storage_rank=arrays["storage_rank"],
        banks=arrays["banks"],
        shop_type=arrays["shop_type"],
        shop_slot=arrays["shop_slot"],
        shop_mask=arrays["shop_mask"],
        market_product=arrays["market_product"],
        market_float=arrays["market_float"],
        market_int=arrays["market_int"],
        global_features=arrays["global_features"],
        globals_int=arrays["globals_int"],
        still_playing=arrays["still_playing"],
        order_limits=arrays["order_limits"],
        can_act=arrays["can_act"],
    )


def _allocate(n_envs: int, *, pin_memory: bool = False) -> KaggricultureObsBatch:
    tensors = {
        name: torch.empty((n_envs, PLAYERS, *shape), dtype=dtype, pin_memory=pin_memory)
        for name, (dtype, shape, _, _) in _SCHEMA.items()
    }
    mask = KaggricultureActionMask(
        can_act=torch.empty(
            (n_envs, PLAYERS, MAX_FRAMES), dtype=torch.bool, pin_memory=pin_memory
        )
    )
    batch = KaggricultureObsBatch(**tensors, action_mask=mask)
    for tensor in _tensors(batch).values():
        tensor.fill_(True if tensor.dtype == torch.bool else -17)
    return batch


def _tensors(batch: KaggricultureObsBatch) -> dict[str, torch.Tensor]:
    # Only the actual class's declared fields are iterated for bookkeeping.
    tensors = {
        name: getattr(batch, name)
        for name in KaggricultureObsBatch.model_fields
        if name != "action_mask"
    }
    tensors["can_act"] = batch.action_mask.can_act
    return tensors


def _arrays(batch: KaggricultureObsBatch) -> dict[str, Any]:
    return {
        "tile_kind": batch.tile_kind.numpy(),
        "tile_crop": batch.tile_crop.numpy(),
        "tile_animal": batch.tile_animal.numpy(),
        "tile_cell": batch.tile_cell.numpy(),
        "tile_role": batch.tile_role.numpy(),
        "tiles_int": batch.tiles_int.numpy(),
        "tiles_float": batch.tiles_float.numpy(),
        "actor_slot": batch.actor_slot.numpy(),
        "actor_cell": batch.actor_cell.numpy(),
        "actor_role": batch.actor_role.numpy(),
        "actor_mask": batch.actor_mask.numpy(),
        "actor_inventory": batch.actor_inventory.numpy(),
        "actor_inventory_rank": batch.actor_inventory_rank.numpy(),
        "actors_float": batch.actors_float.numpy(),
        "player_features": batch.player_features.numpy(),
        "storage_counts": batch.storage_counts.numpy(),
        "storage_rank": batch.storage_rank.numpy(),
        "banks": batch.banks.numpy(),
        "shop_type": batch.shop_type.numpy(),
        "shop_slot": batch.shop_slot.numpy(),
        "shop_mask": batch.shop_mask.numpy(),
        "market_product": batch.market_product.numpy(),
        "market_float": batch.market_float.numpy(),
        "market_int": batch.market_int.numpy(),
        "global_features": batch.global_features.numpy(),
        "globals_int": batch.globals_int.numpy(),
        "still_playing": batch.still_playing.numpy(),
        "order_limits": batch.order_limits.numpy(),
        "can_act": batch.action_mask.can_act.numpy(),
    }


def _assert_semantics(
    batch: KaggricultureObsBatch, headers: list[dict[str, Any]]
) -> None:
    batch.check_contract()
    assert batch.banks.dtype == torch.float64
    assert batch.actor_inventory.dtype == torch.int64
    assert batch.action_mask.can_act.dtype == torch.bool
    assert torch.equal(
        batch.actor_mask[..., :MAX_ACTORS].sum(-1), batch.globals_int[..., 14]
    )
    assert torch.equal(
        batch.actor_mask[..., MAX_ACTORS:].sum(-1), batch.globals_int[..., 15]
    )
    assert torch.equal(
        batch.action_mask.can_act,
        torch.arange(MAX_FRAMES)[None, None, :]
        < (batch.globals_int[..., 14] + batch.order_limits + 1)[..., None],
    )
    assert bool(batch.still_playing.all())
    items = [
        "WHEAT",
        "CARROT",
        "TOMATO",
        "STRAWBERRY",
        "MELON",
        "EGG",
        "MILK",
        "WOOL",
        "FERTILIZER",
        "GOOSE",
        "COW",
        "SHEEP",
    ]
    for env, header in enumerate(headers):
        public = header["initial"]["public"]
        for seat in range(2):
            for role, farm_index in enumerate((seat, 1 - seat)):
                farm = public["farms"][farm_index]
                assert batch.banks[env, seat, role].item() == farm["money"]
                for local, position in enumerate([farm["farmer"], *farm["hands"]]):
                    actor = role * MAX_ACTORS + local
                    assert (
                        batch.actor_cell[env, seat, actor].item()
                        == 10 * position[1] + position[0]
                    )
                    assert batch.actor_slot[env, seat, actor].item() == local
            private = header["initial"]["privates"][seat]
            for item, name in enumerate(items):
                assert batch.storage_counts[env, seat, item].item() == private[
                    "shed"
                ].get(name, 0)
                rank = (
                    list(private["shed"]).index(name) + 1
                    if name in private["shed"]
                    else 0
                )
                assert batch.storage_rank[env, seat, item].item() == rank
            for actor, inventory in enumerate(private["inventories"]):
                for rank, (name, count) in enumerate(inventory.items(), 1):
                    item = items.index(name)
                    assert batch.actor_inventory[env, seat, actor, item].item() == count
                    assert (
                        batch.actor_inventory_rank[env, seat, actor, item].item()
                        == rank
                    )
            assert not bool(batch.actors_float[env, seat, MAX_ACTORS:, 2:].any())
            assert not bool(batch.player_features[env, seat, 1, 11:].any())


@pytest.mark.parametrize("n_envs", [1, 2])
@pytest.mark.parametrize("pin_memory", [False, True])
def test_native_writer_uses_real_schema_and_keeps_all_pointers(
    n_envs: int, pin_memory: bool
) -> None:
    if pin_memory:
        try:
            torch.empty(1, pin_memory=True)
        except RuntimeError as error:
            pytest.skip(f"host pinned allocator unavailable: {error}")
    batch = _allocate(n_envs, pin_memory=pin_memory)
    pointers = {name: tensor.data_ptr() for name, tensor in _tensors(batch).items()}
    headers = [_header() for _ in range(n_envs)]
    for env, header in enumerate(headers):
        header["initial"]["public"]["farms"][0]["money"] += env
    arrays = _arrays(batch)
    assert _write(json.dumps(headers), arrays) is None
    _assert_semantics(batch, headers)
    assert batch.tile_cell[0, 0, 27].item() == 27
    assert batch.tile_crop[0, 0, 27].item() == 3
    assert batch.tiles_int[0, 0, 27].tolist() == [3, 1, 200, -1, 0, 2, 0]
    assert batch.shop_type[0, 0].tolist() == [5, 0, 5, 0, 0, 0, 0, 0]
    assert batch.market_int[0, 0, 0].tolist() == [-12345, 7777]
    assert batch.player_features[0, 0, 0, 10].item() == np.float32(35 / 200000)
    assert pointers == {
        name: tensor.data_ptr() for name, tensor in _tensors(batch).items()
    }
    # Reusing the same arrays clears actor/storage/shop padding.
    for header in headers:
        header["initial"]["public"]["town"]["unlocked_shops"] = []
        header["initial"]["privates"][0]["inventories"][0] = {}
        header["initial"]["privates"][0]["shed"] = {}
    _write(json.dumps(headers), arrays)
    _assert_semantics(batch, headers)
    assert not bool(batch.shop_mask.any())
    assert not bool(batch.actor_inventory_rank[:, 0].any())
    assert not bool(batch.storage_rank[:, 0].any())
    assert pointers == {
        name: tensor.data_ptr() for name, tensor in _tensors(batch).items()
    }


def _rejected(
    arrays: dict[str, Any],
    payload: str,
    field: str,
    extras: tuple[np.ndarray, ...] = (),
) -> None:
    before = [value.tobytes() for value in arrays.values()] + [
        value.tobytes() for value in extras
    ]
    pointers = {name: value.ctypes.data for name, value in arrays.items()}
    with pytest.raises(ValueError, match=field):
        _write(payload, arrays)
    assert before == [value.tobytes() for value in arrays.values()] + [
        value.tobytes() for value in extras
    ]
    assert pointers == {name: value.ctypes.data for name, value in arrays.items()}


@pytest.mark.parametrize("name", [*_SCHEMA, "can_act"])
def test_every_field_rejects_right_numel_wrong_shape(name: str) -> None:
    arrays = _arrays(_allocate(1))
    arrays[name] = arrays[name].reshape(-1)
    _rejected(arrays, json.dumps([_header()]), name)


@pytest.mark.parametrize(
    ("name", "dtype"),
    [
        ("tile_kind", np.int32),
        ("tiles_float", np.float64),
        ("banks", np.float32),
        ("can_act", np.uint8),
    ],
)
def test_wrong_dtype_groups_preserve_every_buffer(name: str, dtype: Any) -> None:
    arrays = _arrays(_allocate(1))
    arrays[name] = arrays[name].astype(dtype)
    _rejected(arrays, json.dumps([_header()]), name)


@pytest.mark.parametrize("name", ["tile_kind", "tiles_float", "banks"])
def test_foreign_endian_numeric_arrays_are_rejected(name: str) -> None:
    arrays = _arrays(_allocate(1))
    arrays[name] = arrays[name].astype(arrays[name].dtype.newbyteorder("S"))
    _rejected(arrays, json.dumps([_header()]), name)


@pytest.mark.parametrize("mode", ["fortran", "stride", "readonly", "unaligned"])
def test_invalid_array_layout_and_writability_preserve_all_bytes(mode: str) -> None:
    arrays = _arrays(_allocate(1))
    original = arrays["tile_kind"]
    extras = ()
    if mode == "fortran":
        arrays["tile_kind"] = np.asfortranarray(original)
    elif mode == "stride":
        root = np.full((1, 2, 400), -17, dtype=np.int64)
        arrays["tile_kind"] = root[..., ::2]
        extras = (root,)
    elif mode == "readonly":
        arrays["tile_kind"].flags.writeable = False
    else:
        root = np.zeros(original.nbytes + 1, dtype=np.uint8)
        arrays["tile_kind"] = np.ndarray(
            original.shape, dtype=np.int64, buffer=root, offset=1
        )
        extras = (root,)
    _rejected(arrays, json.dumps([_header()]), "tile_kind", extras)


@pytest.mark.parametrize("mixed_dtype", [False, True])
@pytest.mark.parametrize("separate_torch_base", [False, True])
def test_overlapping_views_are_rejected_even_with_distinct_base_objects(
    mixed_dtype: bool, separate_torch_base: bool
) -> None:
    batch = _allocate(1)
    arrays = _arrays(batch)
    if mixed_dtype:
        name = "banks"
        arrays[name] = (
            batch.tile_kind.reshape(-1)[:4].view(torch.float64).reshape(1, 2, 2).numpy()
            if separate_torch_base
            else arrays["tile_kind"].reshape(-1)[:4].view(np.float64).reshape(1, 2, 2)
        )
    else:
        name = "tile_crop"
        arrays[name] = (
            batch.tile_kind.numpy()
            if separate_torch_base
            else arrays["tile_kind"].view()
        )
    if separate_torch_base:
        assert arrays[name].base is not arrays["tile_kind"].base
    assert np.shares_memory(arrays[name], arrays["tile_kind"])
    _rejected(arrays, json.dumps([_header()]), name)


def test_adjacent_nonoverlapping_views_are_accepted() -> None:
    batch = _allocate(1)
    root = torch.zeros((2, 1, 2, 200), dtype=torch.int64)
    batch.tile_kind, batch.tile_crop = root[0], root[1]
    arrays = _arrays(batch)
    _write(json.dumps([_header()]), arrays)
    _assert_semantics(batch, [_header()])


def test_full_actor_capacity_then_sparse_reuse_clears_every_field() -> None:
    dense = _header()
    dense["configuration"]["maxMarketOrdersPerTurn"] = 10
    for seat in range(2):
        dense["initial"]["public"]["farms"][seat]["hands"] = [
            [index % 10, (index // 10) % 10] for index in range(240)
        ]
        dense["initial"]["privates"][seat]["inventories"] = [
            {"WOOL": 0, "WHEAT": 16777217} for _ in range(241)
        ]
    batch = _allocate(1)
    arrays = _arrays(batch)
    pointers = {name: tensor.data_ptr() for name, tensor in _tensors(batch).items()}
    _write(json.dumps([dense]), arrays)
    _assert_semantics(batch, [dense])
    assert bool(batch.actor_mask.all())
    assert bool(batch.action_mask.can_act.all())
    assert batch.actor_inventory[0, 0, 240, 0].item() == 16777217
    sparse = _header()
    fresh = _allocate(1)
    _write(json.dumps([sparse]), _arrays(fresh))
    _write(json.dumps([sparse]), arrays)
    _assert_semantics(batch, [sparse])
    assert all(
        torch.equal(tensor, _tensors(fresh)[name])
        for name, tensor in _tensors(batch).items()
    )
    assert pointers == {
        name: tensor.data_ptr() for name, tensor in _tensors(batch).items()
    }


@pytest.mark.parametrize(
    "case",
    [
        "json",
        "object",
        "empty",
        "missing_second_header_fields",
        "late_state",
        "wrong_e",
    ],
)
def test_header_and_late_environment_errors_preserve_the_whole_batch(case: str) -> None:
    arrays = _arrays(_allocate(2))
    payload, field = json.dumps([_header()]), "tile_kind"
    if case == "json":
        payload, field = "not json", "headers"
    elif case == "object":
        payload, field = "{}", "headers"
    elif case == "empty":
        payload, field = "[]", "headers"
    elif case == "missing_second_header_fields":
        payload, field = json.dumps([_header(), {}]), "env=1"
    elif case == "late_state":
        bad = copy.deepcopy(_header())
        bad["initial"]["public"]["market"]["prices"]["WHEAT"] = 1.5
        payload, field = json.dumps([_header(), bad]), "env=1"
    _rejected(arrays, payload, field)
    _write(json.dumps([_header(), _header()]), arrays)


def test_every_frozen_oracle_record_passes_the_actual_schema() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "fixtures/kaggriculture/observation-v3/states.jsonl.gz"
    )
    spec = importlib.util.spec_from_file_location(
        "observation_oracle",
        Path(__file__).resolve().parents[2]
        / "scripts/kaggriculture_observation_oracle/regenerate.py",
    )
    assert spec is not None
    assert spec.loader is not None
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    oracle.validate_corpus(path.parent)
    batch = _allocate(1)
    arrays = _arrays(batch)
    pointers = {name: tensor.data_ptr() for name, tensor in _tensors(batch).items()}
    count = 0
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            header = record["header"]
            _write(json.dumps([header]), arrays)
            _assert_semantics(batch, [header])
            count += 1
    assert count == 512
    assert pointers == {
        name: tensor.data_ptr() for name, tensor in _tensors(batch).items()
    }
