"""Native boundary smoke only; no Python observation schema or substitute class."""

from __future__ import annotations

import copy
import json
from typing import Any

import numpy as np
from owl import rs


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


def _arrays(n_envs: int) -> dict[str, Any]:
    return {
        "tile_kind": np.full((n_envs, 2, 200), -17, dtype=np.int64),
        "tile_crop": np.full((n_envs, 2, 200), -17, dtype=np.int64),
        "tile_animal": np.full((n_envs, 2, 200), -17, dtype=np.int64),
        "tile_cell": np.full((n_envs, 2, 200), -17, dtype=np.int64),
        "tile_role": np.full((n_envs, 2, 200), -17, dtype=np.int64),
        "tiles_int": np.full((n_envs, 2, 200, 7), -17, dtype=np.int64),
        "tiles_float": np.full((n_envs, 2, 200, 15), -17, dtype=np.float32),
        "actor_slot": np.full((n_envs, 2, 482), -17, dtype=np.int64),
        "actor_cell": np.full((n_envs, 2, 482), -17, dtype=np.int64),
        "actor_role": np.full((n_envs, 2, 482), -17, dtype=np.int64),
        "actor_mask": np.full((n_envs, 2, 482), True, dtype=np.bool_),
        "actor_inventory": np.full((n_envs, 2, 241, 12), -17, dtype=np.int64),
        "actor_inventory_rank": np.full((n_envs, 2, 241, 12), -17, dtype=np.int64),
        "actors_float": np.full((n_envs, 2, 482, 26), -17, dtype=np.float32),
        "player_features": np.full((n_envs, 2, 2, 44), -17, dtype=np.float32),
        "storage_counts": np.full((n_envs, 2, 17), -17, dtype=np.int64),
        "storage_rank": np.full((n_envs, 2, 12), -17, dtype=np.int64),
        "banks": np.full((n_envs, 2, 2), -17, dtype=np.float64),
        "shop_type": np.full((n_envs, 2, 8), -17, dtype=np.int64),
        "shop_slot": np.full((n_envs, 2, 8), -17, dtype=np.int64),
        "shop_mask": np.full((n_envs, 2, 8), True, dtype=np.bool_),
        "market_product": np.full((n_envs, 2, 9), -17, dtype=np.int64),
        "market_float": np.full((n_envs, 2, 9, 2), -17, dtype=np.float32),
        "market_int": np.full((n_envs, 2, 9, 2), -17, dtype=np.int64),
        "global_features": np.full((n_envs, 2, 15), -17, dtype=np.float32),
        "globals_int": np.full((n_envs, 2, 16), -17, dtype=np.int64),
        "still_playing": np.full((n_envs, 2), True, dtype=np.bool_),
        "order_limits": np.full((n_envs, 2), -17, dtype=np.int64),
        "can_act": np.full((n_envs, 2, 252), True, dtype=np.bool_),
    }


def main() -> None:
    checks = 0
    for n_envs in (1, 2):
        arrays = _arrays(n_envs)
        pointers = {name: value.ctypes.data for name, value in arrays.items()}
        _write(json.dumps([_header() for _ in range(n_envs)]), arrays)
        assert pointers == {name: value.ctypes.data for name, value in arrays.items()}
        assert arrays["tile_cell"][0, 0, 27] == 27
        assert arrays["tile_crop"][0, 0, 27] == 3
        assert arrays["actor_inventory_rank"][0, 0, 0, 7] == 1
        assert arrays["actor_inventory_rank"][0, 0, 0, 0] == 2
        assert arrays["storage_rank"][0, 0, 7] == 1
        assert arrays["banks"][0, 0, 0] == 3000.00001
        assert arrays["can_act"][0, 0].sum() == 5
        assert arrays["still_playing"].all()
        checks += 1

    dense = _header()
    dense["configuration"]["maxMarketOrdersPerTurn"] = 10
    for seat in range(2):
        dense["initial"]["public"]["farms"][seat]["hands"] = [
            [index % 10, (index // 10) % 10] for index in range(240)
        ]
        dense["initial"]["privates"][seat]["inventories"] = [
            {"WOOL": 0, "WHEAT": 16777217} for _ in range(241)
        ]
    arrays = _arrays(1)
    pointers = {name: value.ctypes.data for name, value in arrays.items()}
    _write(json.dumps([dense]), arrays)
    assert arrays["actor_mask"].all()
    assert arrays["can_act"].all()
    assert arrays["actor_inventory"][0, 0, 240, 0] == 16777217
    fresh = _arrays(1)
    _write(json.dumps([_header()]), fresh)
    _write(json.dumps([_header()]), arrays)
    assert all(
        value.tobytes() == fresh[name].tobytes() for name, value in arrays.items()
    )
    assert pointers == {name: value.ctypes.data for name, value in arrays.items()}
    checks += 1

    def rejected(
        arrays: dict[str, Any],
        payload: str,
        field: str,
        extras: tuple[np.ndarray, ...] = (),
    ) -> None:
        nonlocal checks
        before = [value.tobytes() for value in arrays.values()] + [
            value.tobytes() for value in extras
        ]
        try:
            _write(payload, arrays)
        except ValueError as error:
            if field not in str(error):
                raise AssertionError((field, str(error))) from error
        else:
            raise AssertionError(f"accepted invalid {field}")
        after = [value.tobytes() for value in arrays.values()] + [
            value.tobytes() for value in extras
        ]
        assert before == after, field
        checks += 1

    payload = json.dumps([_header()])
    for name in _arrays(1):
        arrays = _arrays(1)
        arrays[name] = arrays[name].reshape(-1)
        rejected(arrays, payload, "action_mask.can_act" if name == "can_act" else name)
    for name, wrong in [
        ("tile_kind", np.int32),
        ("tiles_float", np.float64),
        ("banks", np.float32),
        ("can_act", np.uint8),
    ]:
        arrays = _arrays(1)
        arrays[name] = arrays[name].astype(wrong)
        rejected(arrays, payload, "action_mask.can_act" if name == "can_act" else name)
    for name in ("tile_kind", "tiles_float", "banks"):
        arrays = _arrays(1)
        arrays[name] = arrays[name].astype(arrays[name].dtype.newbyteorder("S"))
        rejected(arrays, payload, name)
    for mode in ("fortran", "stride", "readonly", "unaligned"):
        arrays = _arrays(1)
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
        rejected(arrays, payload, "tile_kind", extras)
    for mixed in (False, True):
        for separate_base in (False, True):
            arrays = _arrays(1)
            if separate_base:
                root = np.zeros(3200, dtype=np.uint8)
                arrays["tile_kind"] = np.frombuffer(
                    memoryview(root), dtype=np.int64
                ).reshape(1, 2, 200)
                name = "banks" if mixed else "tile_crop"
                dtype = np.float64 if mixed else np.int64
                shape = (1, 2, 2) if mixed else (1, 2, 200)
                arrays[name] = np.frombuffer(
                    memoryview(root), dtype=dtype, count=int(np.prod(shape))
                ).reshape(shape)
            elif mixed:
                name = "banks"
                arrays[name] = (
                    arrays["tile_kind"]
                    .reshape(-1)[:4]
                    .view(np.float64)
                    .reshape(1, 2, 2)
                )
            else:
                name = "tile_crop"
                arrays[name] = arrays["tile_kind"].view()
            rejected(arrays, payload, name)
    arrays = _arrays(1)
    root = np.zeros(800, dtype=np.int64)
    arrays["tile_kind"] = root[:400].reshape(1, 2, 200)
    arrays["tile_crop"] = root[400:].reshape(1, 2, 200)
    _write(payload, arrays)
    checks += 1
    rejected(_arrays(2), payload, "tile_kind")
    for payload, field in [
        ("not json", "headers"),
        ("{}", "headers"),
        ("[]", "headers"),
        (json.dumps([_header(), {}]), "env=1"),
    ]:
        rejected(_arrays(2), payload, field)
    late = copy.deepcopy(_header())
    late["initial"]["public"]["market"]["prices"]["WHEAT"] = 1.5
    retry_arrays = _arrays(2)
    rejected(retry_arrays, json.dumps([_header(), late]), "env=1")
    # Previously failed calls have released every native borrow.
    _write(json.dumps([_header(), _header()]), retry_arrays)
    checks += 1
    print(json.dumps({"native_boundary_checks": checks, "schema_checked": False}))


if __name__ == "__main__":
    main()
