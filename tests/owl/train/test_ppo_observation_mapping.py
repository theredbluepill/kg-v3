"""Schema-generic observation mapping in the PPO trainer (rebuild Task 3.1).

The ``_isaiah_*`` functions below are extracted successful-path copies of the
Orbit-specific helpers from Isaiah's ``python/owl/train/ppo.py`` at upstream
commit 32b3ec9, not a literal freeze of that file. The tensor field list is
derived from the current ``ObsBatch`` schema, and the copy helper omits the
original's validation and error paths. They are the oracle for successful Orbit
calls only: the schema-generic helpers must produce identical tensors for every
Orbit ``ObsBatch`` shape, with optional fields set and unset and every
action-mask type. Failure ordering and error wording are not compared.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import pytest
import torch
from owl.rl import (
    ACTION_ENTITY_SLOTS,
    OUTER_PLAYER_SLOTS,
    ActionMask,
    DiscreteTargetActionMask,
    DiscreteTargetBinActionMask,
    ObsBatch,
    PureActionMask,
)
from owl.train import ppo
from pydantic import BaseModel

MaskKind = Literal["pure", "discrete_target", "discrete_target_bin"]
_MASK_KINDS: tuple[MaskKind, ...] = ("pure", "discrete_target", "discrete_target_bin")
_SEEDS = (0, 1, 2)

# --- Oracle: successful-path extracts of Isaiah's 32b3ec9 helpers ------------

_ISAIAH_OBS_TENSOR_FIELDS = tuple(
    field
    for field in ObsBatch.model_fields
    if field
    not in {
        "action_mask",
        "player_features",
        "fleet_target",
        "target_incoming_features",
    }
)
_ISAIAH_OBS_OPTIONAL_TENSOR_FIELDS = (
    "player_features",
    "fleet_target",
    "target_incoming_features",
)


def _isaiah_map_action_mask(
    action_mask: ActionMask,
    fn: Callable[[torch.Tensor], torch.Tensor],
) -> ActionMask:
    if isinstance(action_mask, PureActionMask):
        return PureActionMask(
            can_act=fn(action_mask.can_act),
            max_launch=fn(action_mask.max_launch),
        )
    if isinstance(action_mask, DiscreteTargetActionMask):
        return DiscreteTargetActionMask(
            can_act=fn(action_mask.can_act),
            max_launch=fn(action_mask.max_launch),
        )
    return DiscreteTargetBinActionMask(can_act=fn(action_mask.can_act))


def _isaiah_map_optional_obs_tensors(
    obs: ObsBatch,
    fn: Callable[[torch.Tensor], torch.Tensor],
) -> dict[str, torch.Tensor | None]:
    return {
        field: None if (tensor := getattr(obs, field)) is None else fn(tensor)
        for field in _ISAIAH_OBS_OPTIONAL_TENSOR_FIELDS
    }


def _isaiah_map_obs(
    obs: ObsBatch,
    fn: Callable[[torch.Tensor], torch.Tensor],
) -> ObsBatch:
    return ObsBatch(
        **{field: fn(getattr(obs, field)) for field in _ISAIAH_OBS_TENSOR_FIELDS},
        **_isaiah_map_optional_obs_tensors(obs, fn),
        action_mask=_isaiah_map_action_mask(obs.action_mask, fn),
    )


def _isaiah_obs_segment_major(obs: ObsBatch) -> ObsBatch:
    return _isaiah_map_obs(obs, lambda tensor: tensor.transpose(0, 1).contiguous())


def _isaiah_obs_index(obs: ObsBatch, idx: torch.Tensor) -> ObsBatch:
    return _isaiah_map_obs(obs, lambda tensor: tensor[idx])


def _isaiah_flatten_obs_time(obs: ObsBatch) -> ObsBatch:
    return _isaiah_map_obs(
        obs,
        lambda tensor: tensor.reshape(
            tensor.shape[0] * tensor.shape[1], *tensor.shape[2:]
        ),
    )


def _isaiah_obs_to_device(
    obs: ObsBatch,
    device: torch.device,
    *,
    non_blocking: bool = False,
) -> ObsBatch:
    if device.type == "cpu":
        return _isaiah_map_obs(
            obs,
            lambda tensor: tensor.to(device, non_blocking=non_blocking).clone(),
        )
    return _isaiah_map_obs(
        obs,
        lambda tensor: tensor.to(device, non_blocking=non_blocking),
    )


def _isaiah_copy_obs(
    dst: ObsBatch,
    src: ObsBatch,
    copy: Callable[[torch.Tensor, torch.Tensor], None],
) -> None:
    for field in _ISAIAH_OBS_TENSOR_FIELDS:
        copy(getattr(dst, field), getattr(src, field))
    for field in _ISAIAH_OBS_OPTIONAL_TENSOR_FIELDS:
        dst_tensor = getattr(dst, field)
        src_tensor = getattr(src, field)
        if dst_tensor is not None:
            copy(dst_tensor, src_tensor)
    copy(dst.action_mask.can_act, src.action_mask.can_act)
    if isinstance(dst.action_mask, PureActionMask | DiscreteTargetActionMask):
        assert isinstance(src.action_mask, PureActionMask | DiscreteTargetActionMask)
        copy(dst.action_mask.max_launch, src.action_mask.max_launch)


# --- Random Orbit observations -----------------------------------------------


def _random_tensor(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    if dtype == torch.bool:
        return torch.rand(shape, generator=generator) > 0.5
    if dtype == torch.int64:
        return torch.randint(-1, 7, shape, generator=generator)
    return torch.randn(shape, generator=generator)


def _random_action_mask(
    kind: MaskKind,
    prefix: tuple[int, ...],
    generator: torch.Generator,
) -> ActionMask:
    slots = (OUTER_PLAYER_SLOTS, ACTION_ENTITY_SLOTS)
    max_launch = _random_tensor((*prefix, *slots), torch.int64, generator)
    if kind == "pure":
        return PureActionMask(
            can_act=_random_tensor((*prefix, *slots), torch.bool, generator),
            max_launch=max_launch,
        )
    if kind == "discrete_target":
        return DiscreteTargetActionMask(
            can_act=_random_tensor(
                (*prefix, *slots, ACTION_ENTITY_SLOTS), torch.bool, generator
            ),
            max_launch=max_launch,
        )
    return DiscreteTargetBinActionMask(
        can_act=_random_tensor(
            (*prefix, *slots, ACTION_ENTITY_SLOTS, 3), torch.bool, generator
        ),
    )


def _random_obs(
    *,
    seed: int,
    prefix: tuple[int, ...],
    with_optional: bool,
    mask_kind: MaskKind,
) -> ObsBatch:
    generator = torch.Generator().manual_seed(seed)
    planets, fleets, comets = 3, 4, 2

    def rand(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        return _random_tensor((*prefix, *shape), dtype, generator)

    return ObsBatch(
        planets=rand((planets, 5), torch.float32),
        orbiting_planets=rand((planets,), torch.bool),
        fleets=rand((fleets, 6), torch.float32),
        fleet_target=rand((fleets,), torch.int64) if with_optional else None,
        target_incoming_features=(
            rand((ACTION_ENTITY_SLOTS, 2), torch.float32) if with_optional else None
        ),
        comets=rand((comets, 4), torch.float32),
        entity_mask=rand((planets + fleets + comets,), torch.bool),
        still_playing=rand((OUTER_PLAYER_SLOTS,), torch.bool),
        global_features=rand((3,), torch.float32),
        action_mask=_random_action_mask(mask_kind, prefix, generator),
        player_features=(
            rand((OUTER_PLAYER_SLOTS, 2), torch.float32) if with_optional else None
        ),
    )


def _obs_tensors(obs: BaseModel) -> dict[str, torch.Tensor | None]:
    tensors: dict[str, torch.Tensor | None] = {}
    for field in type(obs).model_fields:
        value = getattr(obs, field)
        if isinstance(
            value,
            PureActionMask | DiscreteTargetActionMask | DiscreteTargetBinActionMask,
        ):
            tensors[f"{field}.can_act"] = value.can_act
            if isinstance(value, PureActionMask | DiscreteTargetActionMask):
                tensors[f"{field}.max_launch"] = value.max_launch
        else:
            tensors[field] = value
    return tensors


def _assert_identical(actual: BaseModel, expected: BaseModel) -> None:
    assert type(actual) is type(expected)
    for field in type(expected).model_fields:
        assert type(getattr(actual, field)) is type(getattr(expected, field)), field
    actual_tensors = _obs_tensors(actual)
    expected_tensors = _obs_tensors(expected)
    assert actual_tensors.keys() == expected_tensors.keys()
    for name, expected_tensor in expected_tensors.items():
        actual_tensor = actual_tensors[name]
        if expected_tensor is None:
            assert actual_tensor is None, name
            continue
        assert actual_tensor is not None, name
        assert actual_tensor.dtype == expected_tensor.dtype, name
        assert actual_tensor.shape == expected_tensor.shape, name
        assert actual_tensor.device == expected_tensor.device, name
        assert actual_tensor.is_contiguous() == expected_tensor.is_contiguous(), name
        assert torch.equal(actual_tensor, expected_tensor), name


def _data_ptrs(obs: BaseModel) -> dict[str, int]:
    return {
        name: tensor.data_ptr()
        for name, tensor in _obs_tensors(obs).items()
        if tensor is not None
    }


_ORBIT_CASES = [
    pytest.param(
        seed, with_optional, kind, id=f"s{seed}-opt{int(with_optional)}-{kind}"
    )
    for seed in _SEEDS
    for with_optional in (False, True)
    for kind in _MASK_KINDS
]


# --- Orbit equivalence against the extracted oracle --------------------------


@pytest.mark.parametrize(("seed", "with_optional", "mask_kind"), _ORBIT_CASES)
def test_orbit_segment_major_matches_isaiah(
    seed: int, with_optional: bool, mask_kind: MaskKind
) -> None:
    obs = _random_obs(
        seed=seed, prefix=(3, 2), with_optional=with_optional, mask_kind=mask_kind
    )

    _assert_identical(ppo._obs_segment_major(obs), _isaiah_obs_segment_major(obs))


@pytest.mark.parametrize(("seed", "with_optional", "mask_kind"), _ORBIT_CASES)
def test_orbit_index_matches_isaiah(
    seed: int, with_optional: bool, mask_kind: MaskKind
) -> None:
    obs = _random_obs(
        seed=seed, prefix=(4, 3), with_optional=with_optional, mask_kind=mask_kind
    )
    idx = torch.tensor([2, 0, 2])

    _assert_identical(ppo._obs_index(obs, idx), _isaiah_obs_index(obs, idx))


@pytest.mark.parametrize(("seed", "with_optional", "mask_kind"), _ORBIT_CASES)
def test_orbit_flatten_time_matches_isaiah(
    seed: int, with_optional: bool, mask_kind: MaskKind
) -> None:
    obs = _random_obs(
        seed=seed, prefix=(2, 3), with_optional=with_optional, mask_kind=mask_kind
    )

    _assert_identical(ppo._flatten_obs_time(obs), _isaiah_flatten_obs_time(obs))


@pytest.mark.parametrize(("seed", "with_optional", "mask_kind"), _ORBIT_CASES)
def test_orbit_cpu_to_device_matches_isaiah_and_clones(
    seed: int, with_optional: bool, mask_kind: MaskKind
) -> None:
    obs = _random_obs(
        seed=seed, prefix=(2,), with_optional=with_optional, mask_kind=mask_kind
    )
    source_ptrs = _data_ptrs(obs)

    moved = ppo._obs_to_device(obs, torch.device("cpu"))

    _assert_identical(moved, _isaiah_obs_to_device(obs, torch.device("cpu")))
    moved_ptrs = _data_ptrs(moved)
    assert moved_ptrs.keys() == source_ptrs.keys()
    for name, ptr in moved_ptrs.items():
        assert ptr != source_ptrs[name], name


@pytest.mark.parametrize("non_blocking", [False, True])
@pytest.mark.parametrize(("seed", "with_optional", "mask_kind"), _ORBIT_CASES)
def test_orbit_accelerator_to_device_does_not_clone(
    monkeypatch: pytest.MonkeyPatch,
    seed: int,
    with_optional: bool,
    mask_kind: MaskKind,
    non_blocking: bool,
) -> None:
    obs = _random_obs(
        seed=seed, prefix=(2,), with_optional=with_optional, mask_kind=mask_kind
    )
    source_ptrs = _data_ptrs(obs)
    calls: list[tuple[torch.device, bool]] = []

    def fake_to(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        calls.append((args[0], kwargs["non_blocking"]))
        return self

    def forbid_clone(
        self: torch.Tensor,  # noqa: ARG001
        *args: Any,  # noqa: ARG001
        **kwargs: Any,  # noqa: ARG001
    ) -> torch.Tensor:
        raise AssertionError("accelerator transfer must not clone")

    monkeypatch.setattr(torch.Tensor, "to", fake_to)
    monkeypatch.setattr(torch.Tensor, "clone", forbid_clone)

    device = torch.device("cuda")
    moved = ppo._obs_to_device(obs, device, non_blocking=non_blocking)

    assert _data_ptrs(moved) == source_ptrs
    assert calls == [(device, non_blocking)] * len(source_ptrs)


@pytest.mark.parametrize(("seed", "with_optional", "mask_kind"), _ORBIT_CASES)
def test_orbit_copy_time_step_matches_isaiah(
    seed: int, with_optional: bool, mask_kind: MaskKind
) -> None:
    step = 1
    src = _random_obs(
        seed=seed, prefix=(2,), with_optional=with_optional, mask_kind=mask_kind
    )
    dst = _random_obs(
        seed=seed + 100,
        prefix=(3, 2),
        with_optional=with_optional,
        mask_kind=mask_kind,
    )
    expected = _isaiah_obs_to_device(dst, torch.device("cpu"))
    _isaiah_copy_obs(expected, src, lambda d, s: d[step].copy_(s))
    dst_ptrs = _data_ptrs(dst)

    ppo._copy_obs_time_step(dst, step, src)

    _assert_identical(dst, expected)
    assert _data_ptrs(dst) == dst_ptrs


@pytest.mark.parametrize(("seed", "with_optional", "mask_kind"), _ORBIT_CASES)
def test_orbit_copy_to_device_matches_isaiah(
    seed: int, with_optional: bool, mask_kind: MaskKind
) -> None:
    src = _random_obs(
        seed=seed, prefix=(2,), with_optional=with_optional, mask_kind=mask_kind
    )
    dst = _random_obs(
        seed=seed + 100, prefix=(2,), with_optional=with_optional, mask_kind=mask_kind
    )
    expected = _isaiah_obs_to_device(dst, torch.device("cpu"))
    _isaiah_copy_obs(expected, src, lambda d, s: d.copy_(s))
    dst_ptrs = _data_ptrs(dst)

    ppo._copy_obs_to_device_(dst, src)

    _assert_identical(dst, expected)
    assert _data_ptrs(dst) == dst_ptrs


@pytest.mark.parametrize(
    "field", ["player_features", "fleet_target", "target_incoming_features"]
)
@pytest.mark.parametrize("dst_has_field", [False, True])
@pytest.mark.parametrize("time_step", [False, True])
def test_orbit_copy_rejects_optional_field_mismatch(
    field: str, dst_has_field: bool, time_step: bool
) -> None:
    src = _random_obs(seed=0, prefix=(2,), with_optional=True, mask_kind="pure")
    dst = _random_obs(
        seed=1,
        prefix=(3, 2) if time_step else (2,),
        with_optional=True,
        mask_kind="pure",
    )
    if dst_has_field:
        src = src.model_copy(update={field: None})
        message = f"source obs is missing {field}"
    else:
        dst = dst.model_copy(update={field: None})
        context = "rollout" if time_step else "destination"
        message = f"{context} obs has no {field} buffer"

    def copy() -> None:
        if time_step:
            ppo._copy_obs_time_step(dst, 0, src)
        else:
            ppo._copy_obs_to_device_(dst, src)

    with pytest.raises(ValueError, match=message):
        copy()


@pytest.mark.parametrize("time_step", [False, True])
def test_orbit_copy_rejects_action_mask_type_mismatch(time_step: bool) -> None:
    src = _random_obs(seed=0, prefix=(2,), with_optional=False, mask_kind="pure")
    dst = _random_obs(
        seed=1,
        prefix=(3, 2) if time_step else (2,),
        with_optional=False,
        mask_kind="discrete_target",
    )

    def copy() -> None:
        if time_step:
            ppo._copy_obs_time_step(dst, 0, src)
        else:
            ppo._copy_obs_to_device_(dst, src)

    with pytest.raises(ValueError, match="action-mask type mismatch"):
        copy()


# --- A second observation schema with different field names -------------------


class _OtherGameBatch(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    tiles: torch.Tensor
    market_prices: torch.Tensor | None = None
    seat_alive: torch.Tensor
    legal_moves: DiscreteTargetBinActionMask


class _UnsupportedFieldBatch(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    tiles: torch.Tensor
    turn: int


def _other_batch(
    *, seed: int, prefix: tuple[int, ...], with_market: bool
) -> _OtherGameBatch:
    generator = torch.Generator().manual_seed(seed)
    return _OtherGameBatch(
        tiles=_random_tensor((*prefix, 5, 3), torch.float32, generator),
        market_prices=(
            _random_tensor((*prefix, 4), torch.int64, generator)
            if with_market
            else None
        ),
        seat_alive=_random_tensor((*prefix, 2), torch.bool, generator),
        legal_moves=DiscreteTargetBinActionMask(
            can_act=_random_tensor((*prefix, 2, 6), torch.bool, generator)
        ),
    )


def _expected_other(
    batch: _OtherGameBatch, fn: Callable[[torch.Tensor], torch.Tensor]
) -> _OtherGameBatch:
    return _OtherGameBatch(
        tiles=fn(batch.tiles),
        market_prices=(
            None if batch.market_prices is None else fn(batch.market_prices)
        ),
        seat_alive=fn(batch.seat_alive),
        legal_moves=DiscreteTargetBinActionMask(can_act=fn(batch.legal_moves.can_act)),
    )


@pytest.mark.parametrize("with_market", [False, True])
def test_other_schema_round_trips_through_mapping_helpers(with_market: bool) -> None:
    batch = _other_batch(seed=3, prefix=(3, 2), with_market=with_market)
    idx = torch.tensor([1, 1, 0])

    mapped = ppo._map_observation(batch, lambda tensor: tensor.clone())
    segment_major = ppo._obs_segment_major(batch)
    indexed = ppo._obs_index(batch, idx)
    flat = ppo._flatten_obs_time(batch)
    moved = ppo._obs_to_device(batch, torch.device("cpu"))

    _assert_identical(mapped, batch)
    _assert_identical(
        segment_major,
        _expected_other(batch, lambda tensor: tensor.transpose(0, 1).contiguous()),
    )
    _assert_identical(indexed, _expected_other(batch, lambda tensor: tensor[idx]))
    _assert_identical(
        flat,
        _expected_other(batch, lambda tensor: tensor.reshape(6, *tensor.shape[2:])),
    )
    _assert_identical(moved, batch)
    assert not (set(_data_ptrs(moved).values()) & set(_data_ptrs(batch).values()))
    # Segment-major then back to time-major returns the original batch.
    _assert_identical(ppo._obs_segment_major(segment_major), batch)


@pytest.mark.parametrize("with_market", [False, True])
def test_other_schema_copies_through_copy_helpers(with_market: bool) -> None:
    src = _other_batch(seed=4, prefix=(2,), with_market=with_market)
    rollout = _other_batch(seed=5, prefix=(3, 2), with_market=with_market)
    device_buffer = _other_batch(seed=6, prefix=(2,), with_market=with_market)

    ppo._copy_obs_time_step(rollout, 2, src)
    ppo._copy_obs_to_device_(device_buffer, src)

    _assert_identical(ppo._obs_index(rollout, torch.tensor(2)), src)
    _assert_identical(device_buffer, src)


def test_copy_rejects_mismatched_observation_types() -> None:
    orbit = _random_obs(seed=0, prefix=(2,), with_optional=False, mask_kind="pure")
    other = _other_batch(seed=0, prefix=(2,), with_market=False)

    with pytest.raises(ValueError, match="observation type mismatch"):
        ppo._copy_obs_to_device_(other, orbit)


def test_map_observation_rejects_unsupported_field_types() -> None:
    batch = _UnsupportedFieldBatch(tiles=torch.zeros(2, 3), turn=4)

    with pytest.raises(TypeError, match=r"'turn'.*int"):
        ppo._map_observation(batch, lambda tensor: tensor)


# --- Replay-alarm diagnostics --------------------------------------------------


def _expected_shapes(obs: BaseModel) -> str:
    return ", ".join(
        f"{name}={tuple(tensor.shape)}"
        for name, tensor in _obs_tensors(obs).items()
        if tensor is not None
    )


@pytest.mark.parametrize("with_optional", [False, True])
@pytest.mark.parametrize("mask_kind", _MASK_KINDS)
def test_observation_tensor_shapes_include_action_mask_tensors(
    with_optional: bool, mask_kind: MaskKind
) -> None:
    obs = _random_obs(
        seed=0, prefix=(4, 2), with_optional=with_optional, mask_kind=mask_kind
    )

    shapes = ppo._observation_tensor_shapes(obs)

    assert shapes == _expected_shapes(obs)
    assert f"action_mask.can_act={tuple(obs.action_mask.can_act.shape)}" in shapes


def test_observation_tensor_shapes_follow_other_schema_field_names() -> None:
    batch = _other_batch(seed=7, prefix=(3, 2), with_market=True)

    assert ppo._observation_tensor_shapes(batch) == (
        "tiles=(3, 2, 5, 3), market_prices=(3, 2, 4), seat_alive=(3, 2, 2), "
        "legal_moves.can_act=(3, 2, 2, 6)"
    )
