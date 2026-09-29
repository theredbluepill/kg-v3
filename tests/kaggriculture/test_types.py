import pytest
import torch
from owl.kaggriculture import types as kt
from pydantic import ValidationError

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


# --- lead dims and per-field bounds (Codex verify-2.1 finding 1) ---------------


@pytest.mark.parametrize(
    "reshape",
    [lambda t: t[0], lambda t: t.unsqueeze(0)],
    ids=["lead_(2,)", "lead_(1,1,2)"],
)
def test_lead_dims_must_be_exactly_env_by_seat(reshape) -> None:  # type: ignore[no-untyped-def]
    obs = make_obs()
    values = {
        name: reshape(getattr(obs, name))
        for name in kt.KaggricultureObsBatch.model_fields
        if name != "action_mask"
    }
    batch = kt.KaggricultureObsBatch(
        **values,
        action_mask=kt.KaggricultureActionMask(
            can_act=reshape(obs.action_mask.can_act)
        ),
    )
    with pytest.raises(ValueError, match=r"still_playing must have shape \[env, 2\]"):
        batch.check_contract()


@pytest.mark.parametrize(
    "field",
    [
        "actor_inventory",
        "actor_inventory_rank",
        "storage_counts",
        "storage_rank",
        "globals_int",
        "order_limits",
    ],
)
def test_negative_counts_ranks_globals_and_order_limits_are_rejected(
    field: str,
) -> None:
    obs = make_obs()
    tensor: torch.Tensor = getattr(obs, field)
    tensor.view(-1)[0] = -1
    with pytest.raises(ValueError, match=rf"{field} values must be >= 0"):
        obs.check_contract()


@pytest.mark.parametrize(
    ("field", "limit"),
    [
        ("actor_inventory_rank", kt.ITEM_COUNT + 1),
        ("storage_rank", kt.ITEM_COUNT + 1),
        ("order_limits", kt.MAX_ORDER_LIMIT + 1),
    ],
)
def test_ranks_and_order_limits_have_exclusive_upper_bounds(
    field: str, limit: int
) -> None:
    obs = make_obs()
    tensor: torch.Tensor = getattr(obs, field)
    tensor.view(-1)[0] = limit - 1
    obs.check_contract()
    tensor.view(-1)[0] = limit
    with pytest.raises(ValueError, match=rf"{field} values must be < {limit}"):
        obs.check_contract()


def test_tiles_int_sentinels_and_signed_market_int_pass() -> None:
    obs = make_obs()
    obs.tiles_int[0, 0, 0, 1:5] = -1
    obs.market_int[0, 0, :, 0] = -1_000
    obs.check_contract()
    obs.tiles_int[0, 0, 0, 1] = -2
    with pytest.raises(ValueError, match=r"tiles_int values must be >= -1"):
        obs.check_contract()


# --- complete dtype / shape / missing-field checks (finding 6) -----------------

_TENSOR_FIELDS = tuple(
    name for name in kt.KaggricultureObsBatch.model_fields if name != "action_mask"
)
_WRONG_DTYPE = {
    torch.int64: torch.int32,
    torch.float32: torch.float64,
    torch.float64: torch.float32,
    torch.bool: torch.int64,
}


@pytest.mark.parametrize("field", _TENSOR_FIELDS)
def test_every_field_rejects_a_wrong_dtype(field: str) -> None:
    values = dict(make_obs())
    tensor: torch.Tensor = values[field]
    values[field] = tensor.to(_WRONG_DTYPE[tensor.dtype])
    with pytest.raises(ValueError, match=field):
        kt.KaggricultureObsBatch(**values).check_contract()


@pytest.mark.parametrize("field", _TENSOR_FIELDS)
def test_every_field_rejects_a_wrong_shape(field: str) -> None:
    values = dict(make_obs(envs=2))
    tensor: torch.Tensor = values[field]
    values[field] = tensor[..., :-1]
    with pytest.raises(ValueError, match=field):
        kt.KaggricultureObsBatch(**values).check_contract()


@pytest.mark.parametrize("field", kt.KaggricultureObsBatch.model_fields)
def test_every_missing_field_is_a_validation_error_naming_it(field: str) -> None:
    values = dict(make_obs())
    del values[field]
    with pytest.raises(ValidationError, match=field):
        kt.KaggricultureObsBatch(**values)


@pytest.mark.parametrize(
    "can_act",
    [
        torch.ones((1, 2, kt.MAX_FRAMES), dtype=torch.int64),
        torch.ones((1, 2, kt.MAX_FRAMES - 1), dtype=torch.bool),
    ],
    ids=["dtype", "shape"],
)
def test_can_act_dtype_and_shape_are_checked(can_act: torch.Tensor) -> None:
    obs = make_obs()
    values = dict(obs)
    values["action_mask"] = kt.KaggricultureActionMask(can_act=can_act)
    with pytest.raises(ValueError, match=r"action_mask\.can_act"):
        kt.KaggricultureObsBatch(**values).check_contract()


# --- the fixture itself is contract-valid (finding 7) --------------------------


@pytest.mark.parametrize(
    ("envs", "own", "rival", "shops"),
    [(1, 2, 1, 3), (3, (1, 2, 3), (4, 5, 6), (1, 4, 8))],
)
def test_make_obs_is_contract_valid(envs, own, rival, shops) -> None:  # type: ignore[no-untyped-def]
    obs = make_obs(envs, own_actors=own, rival_actors=rival, shops=shops)
    obs.check_contract()
    own_counts = obs.actor_mask[..., : kt.MAX_ACTORS].sum(-1)
    rival_counts = obs.actor_mask[..., kt.MAX_ACTORS :].sum(-1)
    # seat 1 sees seat 0's game with the players swapped
    torch.testing.assert_close(own_counts[:, 0], rival_counts[:, 1])
    torch.testing.assert_close(own_counts, obs.globals_int[..., 14])
    torch.testing.assert_close(rival_counts, obs.globals_int[..., 15])
    slots = torch.arange(kt.MAX_ACTORS)
    for env in range(envs):
        for seat in range(kt.PLAYERS):
            row = (env, seat)
            n_own, n_rival = int(own_counts[row]), int(rival_counts[row])
            expected_mask = torch.cat((slots < n_own, slots < n_rival))
            torch.testing.assert_close(obs.actor_mask[row], expected_mask)
            absent = ~expected_mask
            for field in ("actor_slot", "actor_cell", "actor_role"):
                assert int(getattr(obs, field)[row][absent].abs().sum()) == 0, field
            assert int(torch.count_nonzero(obs.actors_float[row][absent])) == 0
            rival_rows = obs.actors_float[row][kt.MAX_ACTORS :]
            assert int(torch.count_nonzero(rival_rows[:, 2:])) == 0
            assert int(torch.count_nonzero(obs.actor_inventory[row][n_own:])) == 0
            assert int(torch.count_nonzero(obs.actor_inventory_rank[row][n_own:])) == 0
            opponent = obs.player_features[row][1]
            assert int(torch.count_nonzero(opponent[11:42])) == 0
            assert int(torch.count_nonzero(obs.player_features[row][:, 42:])) == 0
            shop_absent = ~obs.shop_mask[row]
            assert int(obs.shop_type[row][shop_absent].abs().sum()) == 0
            assert int(obs.shop_slot[row][shop_absent].abs().sum()) == 0
            n_shops = int(obs.shop_mask[row].sum())
            torch.testing.assert_close(
                obs.shop_slot[row][:n_shops], torch.arange(n_shops)
            )
            frames = torch.arange(kt.MAX_FRAMES)
            limit = n_own + int(obs.order_limits[row]) + 1
            torch.testing.assert_close(obs.action_mask.can_act[row], frames < limit)
            torch.testing.assert_close(
                obs.banks[row].float(), obs.player_features[row][:, 0] * 200_000
            )
            not_plant = obs.tile_kind[row] != kt.TILE_KINDS.index("PLANT")
            no_animal = obs.tile_animal[row] == 0
            inapplicable = not_plant & no_animal
            assert int(torch.count_nonzero(obs.tiles_int[row][inapplicable])) == 0
            assert int(torch.count_nonzero(obs.tiles_float[row][inapplicable])) == 0
            assert int((obs.tiles_int[row][:, 1:4][not_plant] != 0).sum()) == 0
            assert int((obs.tiles_int[row][:, 4][no_animal] != 0).sum()) == 0
