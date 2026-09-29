import pytest
import torch
from owl.kaggriculture import types as kt

from tests.kaggriculture.conftest import make_obs


def test_pinned_enums_match_contract_v4() -> None:
    assert kt.TILE_KINDS == ("EMPTY", "LOCKED", "WEED", "PLANT", "COOP", "PASTURE")
    assert kt.CROPS == ("NONE", "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
    assert kt.ANIMALS == ("NONE", "GOOSE", "COW", "SHEEP")
    assert kt.ITEMS[0] == "WHEAT"
    assert kt.ITEMS[-1] == "SHEEP"
    assert len(kt.ITEMS) == 12
    assert kt.ITEMS[:9] == kt.PRODUCTS
    assert kt.PRODUCTS[-1] == "FERTILIZER"
    assert kt.SHOP_TYPES[0] == "BAKERY"
    assert kt.SHOP_TYPES[-1] == "YARN_STORE"
    assert ("NONE", *kt.ITEMS) == kt.ACTION_ITEMS
    assert len(kt.UNIT_KINDS) == 19
    assert kt.UNIT_KINDS[-1] == "DIG"
    assert kt.MARKET_KINDS[-1] == "EMPTY"
    assert kt.SLOT_WIDTHS == (241, 20, 128, 16, 2, 32, 32, 8, 16, 32, 32, 2)


def test_valid_batch_passes_including_sentinels_and_large_scaled_values() -> None:
    obs = make_obs(envs=2)
    obs.tiles_int[..., 2] = -1  # max_lifespan_step sentinel
    obs.tiles_float[..., 5] = -3.0  # signed overdue deadline
    obs.player_features[..., 10] = 4.16  # next-hire cost above 4
    obs.market_int[..., 0] = -50  # signed market inventory index
    obs.check_contract()


def test_twenty_nine_top_level_fields() -> None:
    assert len(kt.KaggricultureObsBatch.model_fields) == 29


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("tile_kind", lambda t: t.to(torch.int32)),
        ("tiles_float", lambda t: t[..., :-1]),
        ("banks", lambda t: t.float()),
        ("actor_mask", lambda t: t.long()),
    ],
)
def test_wrong_dtype_or_shape_names_the_field(field, bad) -> None:  # type: ignore[no-untyped-def]
    obs = make_obs()
    values = dict(obs)
    values[field] = bad(values[field])
    with pytest.raises(ValueError, match=field):
        kt.KaggricultureObsBatch(**values).check_contract()


@pytest.mark.parametrize(
    ("field", "limit"),
    [
        ("tile_kind", 6),
        ("tile_crop", 6),
        ("tile_animal", 4),
        ("tile_cell", 100),
        ("tile_role", 2),
        ("actor_slot", 241),
        ("actor_cell", 100),
        ("actor_role", 4),
        ("shop_type", 8),
        ("shop_slot", 8),
        ("market_product", 9),
    ],
)
def test_categorical_bounds_are_inclusive_below_and_exclusive_above(
    field, limit
) -> None:  # type: ignore[no-untyped-def]
    obs = make_obs()
    getattr(obs, field)[0, 0, 0] = limit - 1
    obs.check_contract()
    getattr(obs, field)[0, 0, 0] = limit
    with pytest.raises(ValueError, match=field):
        obs.check_contract()
    getattr(obs, field)[0, 0, 0] = -1
    with pytest.raises(ValueError, match=field):
        obs.check_contract()


def test_nonfinite_float_channel_is_rejected() -> None:
    obs = make_obs()
    obs.global_features[0, 0, 3] = float("nan")
    with pytest.raises(ValueError, match="global_features"):
        obs.check_contract()


def test_leading_dims_must_agree() -> None:
    obs = make_obs(envs=2)
    values = dict(obs)
    values["shop_mask"] = values["shop_mask"][:1]
    with pytest.raises(ValueError, match="shop_mask"):
        kt.KaggricultureObsBatch(**values).check_contract()
