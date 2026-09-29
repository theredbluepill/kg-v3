"""Caller-owned Kaggriculture buffers and the synchronous reuse boundary.

Native code owns transitions and rewards. Returned tensors alias one retained
buffer set; copy before the next mutating call when retaining a CPU value.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import torch
from numpy.typing import NDArray

from owl import rs
from owl.kaggriculture.rewards import KaggricultureRewardConfig
from owl.kaggriculture.types import (
    JsonValue,
    KaggricultureActionConfig,
    KaggricultureActionMask,
    KaggricultureActions,
    KaggricultureGameConfig,
    KaggricultureObsBatch,
    KaggricultureObsConfig,
)

if TYPE_CHECKING:
    from owl.rs import KaggricultureTerminalMetrics


def _allocate(
    shape: tuple[int, ...], dtype: torch.dtype, *, pin_memory: bool
) -> torch.Tensor:
    if pin_memory and not torch.cuda.is_available():
        raise RuntimeError(
            "requested pinned memory requires an available CUDA allocator"
        )
    tensor = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=pin_memory)
    if pin_memory and not tensor.is_pinned():
        raise RuntimeError("requested pinned allocation returned unpinned memory")
    return tensor


def allocate_observation_buffers(
    n_envs: int, *, pin_memory: bool
) -> KaggricultureObsBatch:
    """Allocate the contract's 29 CPU buffers; native observe fills all padding."""
    if type(n_envs) is not int or n_envs < 1:
        raise ValueError("n_envs must be a positive integer")
    return KaggricultureObsBatch(
        tile_kind=_allocate((n_envs, 2, 200), torch.int64, pin_memory=pin_memory),
        tile_crop=_allocate((n_envs, 2, 200), torch.int64, pin_memory=pin_memory),
        tile_animal=_allocate((n_envs, 2, 200), torch.int64, pin_memory=pin_memory),
        tile_cell=_allocate((n_envs, 2, 200), torch.int64, pin_memory=pin_memory),
        tile_role=_allocate((n_envs, 2, 200), torch.int64, pin_memory=pin_memory),
        tiles_int=_allocate((n_envs, 2, 200, 7), torch.int64, pin_memory=pin_memory),
        tiles_float=_allocate(
            (n_envs, 2, 200, 15), torch.float32, pin_memory=pin_memory
        ),
        actor_slot=_allocate((n_envs, 2, 482), torch.int64, pin_memory=pin_memory),
        actor_cell=_allocate((n_envs, 2, 482), torch.int64, pin_memory=pin_memory),
        actor_role=_allocate((n_envs, 2, 482), torch.int64, pin_memory=pin_memory),
        actor_mask=_allocate((n_envs, 2, 482), torch.bool, pin_memory=pin_memory),
        actor_inventory=_allocate(
            (n_envs, 2, 241, 12), torch.int64, pin_memory=pin_memory
        ),
        actor_inventory_rank=_allocate(
            (n_envs, 2, 241, 12), torch.int64, pin_memory=pin_memory
        ),
        actors_float=_allocate(
            (n_envs, 2, 482, 26), torch.float32, pin_memory=pin_memory
        ),
        player_features=_allocate(
            (n_envs, 2, 2, 44), torch.float32, pin_memory=pin_memory
        ),
        storage_counts=_allocate((n_envs, 2, 17), torch.int64, pin_memory=pin_memory),
        storage_rank=_allocate((n_envs, 2, 12), torch.int64, pin_memory=pin_memory),
        banks=_allocate((n_envs, 2, 2), torch.float64, pin_memory=pin_memory),
        shop_type=_allocate((n_envs, 2, 8), torch.int64, pin_memory=pin_memory),
        shop_slot=_allocate((n_envs, 2, 8), torch.int64, pin_memory=pin_memory),
        shop_mask=_allocate((n_envs, 2, 8), torch.bool, pin_memory=pin_memory),
        market_product=_allocate((n_envs, 2, 9), torch.int64, pin_memory=pin_memory),
        market_float=_allocate((n_envs, 2, 9, 2), torch.float32, pin_memory=pin_memory),
        market_int=_allocate((n_envs, 2, 9, 2), torch.int64, pin_memory=pin_memory),
        global_features=_allocate(
            (n_envs, 2, 15), torch.float32, pin_memory=pin_memory
        ),
        globals_int=_allocate((n_envs, 2, 16), torch.int64, pin_memory=pin_memory),
        still_playing=_allocate((n_envs, 2), torch.bool, pin_memory=pin_memory),
        order_limits=_allocate((n_envs, 2), torch.int64, pin_memory=pin_memory),
        action_mask=KaggricultureActionMask(
            can_act=_allocate((n_envs, 2, 252), torch.bool, pin_memory=pin_memory)
        ),
    )


@dataclass(frozen=True)
class NativeArrays:
    """One persistent NumPy view for each ABI output, never rebuilt on step."""

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
    rewards: NDArray[np.float32]
    dones: NDArray[np.bool_]
    transition_banks_before: NDArray[np.float64]
    transition_banks_after: NDArray[np.float64]
    transition_econ_before: NDArray[np.int64]
    transition_econ_after: NDArray[np.int64]


def _check_input(
    value: torch.Tensor, name: str, dtype: torch.dtype, shape: tuple[int, ...]
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.device.type != "cpu" or value.dtype != dtype:
        raise ValueError(f"{name} must be a CPU {dtype} tensor")
    if tuple(value.shape) != shape or not value.is_contiguous():
        raise ValueError(f"{name} must be C-contiguous with shape {shape}")


class KaggricultureVectorizedEnv:
    """Thin adapter over Task 1.4's native environment; no Python gameplay."""

    def __init__(
        self,
        *,
        n_envs: int,
        seed: int,
        seed_stride: int,
        config: KaggricultureGameConfig,
        reward_config: KaggricultureRewardConfig,
        reward_mode: Literal["win_loss"],
        native_threads: int,
        pin_memory: bool,
        transfer_device: torch.device,
        obs_spec: KaggricultureObsConfig,
        action_spec: KaggricultureActionConfig,
    ) -> None:
        for name, value, low in (
            ("n_envs", n_envs, 1),
            ("seed", seed, 0),
            ("seed_stride", seed_stride, 1),
            ("native_threads", native_threads, 1),
        ):
            if type(value) is not int or not low <= value <= 2**63 - 1:
                raise ValueError(f"{name} must be an integer in {low}..2**63-1")
        if reward_mode != "win_loss":
            raise ValueError("Kaggriculture requires reward_mode='win_loss'")
        self._n_envs = n_envs
        self._obs_spec = obs_spec
        self._action_spec = action_spec
        self._reward_mode = reward_mode
        self._pin_memory_enabled = pin_memory
        self.transfer_device = transfer_device
        self._observations = allocate_observation_buffers(n_envs, pin_memory=pin_memory)
        self._rewards = _allocate((n_envs, 2), torch.float32, pin_memory=pin_memory)
        self._dones = _allocate((n_envs, 2), torch.bool, pin_memory=pin_memory)
        self._transition_banks_before = _allocate(
            (n_envs, 2), torch.float64, pin_memory=pin_memory
        )
        self._transition_banks_after = _allocate(
            (n_envs, 2), torch.float64, pin_memory=pin_memory
        )
        self._transition_econ_before = _allocate(
            (n_envs, 2, 32), torch.int64, pin_memory=pin_memory
        )
        self._transition_econ_after = _allocate(
            (n_envs, 2, 32), torch.int64, pin_memory=pin_memory
        )
        obs = self._observations
        self._arrays = NativeArrays(
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
            rewards=self._rewards.numpy(),
            dones=self._dones.numpy(),
            transition_banks_before=self._transition_banks_before.numpy(),
            transition_banks_after=self._transition_banks_after.numpy(),
            transition_econ_before=self._transition_econ_before.numpy(),
            transition_econ_after=self._transition_econ_after.numpy(),
        )
        self._native = rs.KaggricultureEnv(
            n_envs,
            seed,
            seed_stride,
            config.to_native_json(),
            reward_config.to_native_dict(reward_mode),
            native_threads,
            hire_limit=action_spec.hire_limit,
        )
        arrays = self._arrays
        self._native.observe(
            tile_kind=arrays.tile_kind,
            tile_crop=arrays.tile_crop,
            tile_animal=arrays.tile_animal,
            tile_cell=arrays.tile_cell,
            tile_role=arrays.tile_role,
            tiles_int=arrays.tiles_int,
            tiles_float=arrays.tiles_float,
            actor_slot=arrays.actor_slot,
            actor_cell=arrays.actor_cell,
            actor_role=arrays.actor_role,
            actor_mask=arrays.actor_mask,
            actor_inventory=arrays.actor_inventory,
            actor_inventory_rank=arrays.actor_inventory_rank,
            actors_float=arrays.actors_float,
            player_features=arrays.player_features,
            storage_counts=arrays.storage_counts,
            storage_rank=arrays.storage_rank,
            banks=arrays.banks,
            shop_type=arrays.shop_type,
            shop_slot=arrays.shop_slot,
            shop_mask=arrays.shop_mask,
            market_product=arrays.market_product,
            market_float=arrays.market_float,
            market_int=arrays.market_int,
            global_features=arrays.global_features,
            globals_int=arrays.globals_int,
            still_playing=arrays.still_playing,
            order_limits=arrays.order_limits,
            can_act=arrays.can_act,
            rewards=arrays.rewards,
            dones=arrays.dones,
            transition_banks_before=arrays.transition_banks_before,
            transition_banks_after=arrays.transition_banks_after,
            transition_econ_before=arrays.transition_econ_before,
            transition_econ_after=arrays.transition_econ_after,
        )

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def obs_spec(self) -> KaggricultureObsConfig:
        return self._obs_spec

    @property
    def action_spec(self) -> KaggricultureActionConfig:
        return self._action_spec

    @property
    def reward_mode(self) -> Literal["win_loss"]:
        return self._reward_mode

    @property
    def pin_memory_enabled(self) -> bool:
        return self._pin_memory_enabled

    @property
    def observations(self) -> KaggricultureObsBatch:
        return self._observations

    @property
    def rewards(self) -> torch.Tensor:
        return self._rewards

    @property
    def dones(self) -> torch.Tensor:
        return self._dones

    @property
    def transition_banks_before(self) -> torch.Tensor:
        return self._transition_banks_before

    @property
    def transition_banks_after(self) -> torch.Tensor:
        return self._transition_banks_after

    @property
    def transition_econ_before(self) -> torch.Tensor:
        return self._transition_econ_before

    @property
    def transition_econ_after(self) -> torch.Tensor:
        return self._transition_econ_after

    def _fence(self) -> None:
        if self.transfer_device.type == "cuda" and self.pin_memory_enabled:
            torch.cuda.current_stream(self.transfer_device).synchronize()

    def reset(self) -> KaggricultureObsBatch:
        self._fence()
        arrays = self._arrays
        self._native.reset(
            tile_kind=arrays.tile_kind,
            tile_crop=arrays.tile_crop,
            tile_animal=arrays.tile_animal,
            tile_cell=arrays.tile_cell,
            tile_role=arrays.tile_role,
            tiles_int=arrays.tiles_int,
            tiles_float=arrays.tiles_float,
            actor_slot=arrays.actor_slot,
            actor_cell=arrays.actor_cell,
            actor_role=arrays.actor_role,
            actor_mask=arrays.actor_mask,
            actor_inventory=arrays.actor_inventory,
            actor_inventory_rank=arrays.actor_inventory_rank,
            actors_float=arrays.actors_float,
            player_features=arrays.player_features,
            storage_counts=arrays.storage_counts,
            storage_rank=arrays.storage_rank,
            banks=arrays.banks,
            shop_type=arrays.shop_type,
            shop_slot=arrays.shop_slot,
            shop_mask=arrays.shop_mask,
            market_product=arrays.market_product,
            market_float=arrays.market_float,
            market_int=arrays.market_int,
            global_features=arrays.global_features,
            globals_int=arrays.globals_int,
            still_playing=arrays.still_playing,
            order_limits=arrays.order_limits,
            can_act=arrays.can_act,
            rewards=arrays.rewards,
            dones=arrays.dones,
            transition_banks_before=arrays.transition_banks_before,
            transition_banks_after=arrays.transition_banks_after,
            transition_econ_before=arrays.transition_econ_before,
            transition_econ_after=arrays.transition_econ_after,
        )
        return self.observations

    def step(
        self, actions: KaggricultureActions
    ) -> tuple[
        KaggricultureObsBatch, torch.Tensor, torch.Tensor, dict[str, list[float]]
    ]:
        self._fence()
        if not isinstance(actions, KaggricultureActions):
            raise TypeError("actions must be KaggricultureActions")
        _check_input(actions.tokens, "tokens", torch.int64, (self.n_envs, 2, 252, 12))
        _check_input(actions.lengths, "lengths", torch.int64, (self.n_envs, 2))
        tokens = actions.tokens.numpy()
        lengths = actions.lengths.numpy()
        arrays = self._arrays
        metrics = self._native.step(
            tokens,
            lengths,
            tile_kind=arrays.tile_kind,
            tile_crop=arrays.tile_crop,
            tile_animal=arrays.tile_animal,
            tile_cell=arrays.tile_cell,
            tile_role=arrays.tile_role,
            tiles_int=arrays.tiles_int,
            tiles_float=arrays.tiles_float,
            actor_slot=arrays.actor_slot,
            actor_cell=arrays.actor_cell,
            actor_role=arrays.actor_role,
            actor_mask=arrays.actor_mask,
            actor_inventory=arrays.actor_inventory,
            actor_inventory_rank=arrays.actor_inventory_rank,
            actors_float=arrays.actors_float,
            player_features=arrays.player_features,
            storage_counts=arrays.storage_counts,
            storage_rank=arrays.storage_rank,
            banks=arrays.banks,
            shop_type=arrays.shop_type,
            shop_slot=arrays.shop_slot,
            shop_mask=arrays.shop_mask,
            market_product=arrays.market_product,
            market_float=arrays.market_float,
            market_int=arrays.market_int,
            global_features=arrays.global_features,
            globals_int=arrays.globals_int,
            still_playing=arrays.still_playing,
            order_limits=arrays.order_limits,
            can_act=arrays.can_act,
            rewards=arrays.rewards,
            dones=arrays.dones,
            transition_banks_before=arrays.transition_banks_before,
            transition_banks_after=arrays.transition_banks_after,
            transition_econ_before=arrays.transition_econ_before,
            transition_econ_after=arrays.transition_econ_after,
        )
        return self.observations, self.rewards, self.dones, metrics

    def truncate_envs(self, mask: torch.Tensor) -> KaggricultureObsBatch:
        self._fence()
        _check_input(mask, "mask", torch.bool, (self.n_envs,))
        mask_view = mask.numpy()
        arrays = self._arrays
        self._native.truncate_envs(
            mask_view,
            tile_kind=arrays.tile_kind,
            tile_crop=arrays.tile_crop,
            tile_animal=arrays.tile_animal,
            tile_cell=arrays.tile_cell,
            tile_role=arrays.tile_role,
            tiles_int=arrays.tiles_int,
            tiles_float=arrays.tiles_float,
            actor_slot=arrays.actor_slot,
            actor_cell=arrays.actor_cell,
            actor_role=arrays.actor_role,
            actor_mask=arrays.actor_mask,
            actor_inventory=arrays.actor_inventory,
            actor_inventory_rank=arrays.actor_inventory_rank,
            actors_float=arrays.actors_float,
            player_features=arrays.player_features,
            storage_counts=arrays.storage_counts,
            storage_rank=arrays.storage_rank,
            banks=arrays.banks,
            shop_type=arrays.shop_type,
            shop_slot=arrays.shop_slot,
            shop_mask=arrays.shop_mask,
            market_product=arrays.market_product,
            market_float=arrays.market_float,
            market_int=arrays.market_int,
            global_features=arrays.global_features,
            globals_int=arrays.globals_int,
            still_playing=arrays.still_playing,
            order_limits=arrays.order_limits,
            can_act=arrays.can_act,
            rewards=arrays.rewards,
            dones=arrays.dones,
            transition_banks_before=arrays.transition_banks_before,
            transition_banks_after=arrays.transition_banks_after,
            transition_econ_before=arrays.transition_econ_before,
            transition_econ_after=arrays.transition_econ_after,
        )
        return self.observations

    def terminal_metrics(self, i: int) -> KaggricultureTerminalMetrics | None:
        return self._native.terminal_metrics(i)

    def state_snapshot(self, i: int) -> dict[str, JsonValue]:
        snapshot = json.loads(self._native.state_snapshot(i))
        if not isinstance(snapshot, dict):
            raise ValueError("native state_snapshot must return a JSON object")
        return cast("dict[str, JsonValue]", snapshot)

    def seed_state(self) -> tuple[int, tuple[int, ...]]:
        return self._native.seed_state()
