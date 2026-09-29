"""Exact native-signature fake; tests do not substitute gameplay semantics."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from tests.kaggriculture.test_observe import _header, _write

if TYPE_CHECKING:
    from owl.rs import KaggricultureRewardDict


class FakeKaggricultureEnv:
    def __init__(
        self,
        n_envs: int,
        seed: int,
        seed_stride: int,
        config: str,
        reward_config: KaggricultureRewardDict,
        native_threads: int,
        *,
        hire_limit: int,
    ) -> None:
        self.n_envs = n_envs
        self.seed = seed
        self.seed_stride = seed_stride
        self.config = config
        self.reward_config = reward_config
        self.native_threads = native_threads
        self.hire_limit = hire_limit
        self.current_seeds = tuple(seed + i * seed_stride for i in range(n_envs))
        self.next_seed = seed + n_envs * seed_stride
        self.calls = []
        self.inputs = []
        self.outputs = []
        self.fail = False
        self.events = []

    def _call(self, method, arrays, inputs=()):
        self.events.append(f"native.{method}")
        self.calls.append(method)
        self.inputs.append(inputs)
        self.outputs.append(arrays)
        if self.fail:
            raise ValueError("injected native failure before publication")
        if method == "truncate_envs":
            arrays["globals_int"][inputs[0], :, 0] = 0
            arrays["global_features"][inputs[0], :, 0] = 0
            return
        headers = [_header() for _ in range(self.n_envs)]
        if method == "step":
            for header in headers:
                header["initial"]["public"]["step"] += 1
                header["initial"]["public"]["day"] = 10
                header["initial"]["public"]["hour"] = 0
        _write(json.dumps(headers), arrays)
        for name in (
            "rewards",
            "dones",
            "transition_banks_before",
            "transition_banks_after",
            "transition_econ_before",
            "transition_econ_after",
        ):
            arrays[name].fill(0)
        if method == "reset":
            self.current_seeds = tuple(
                self.next_seed + i * self.seed_stride for i in range(self.n_envs)
            )
            self.next_seed += self.n_envs * self.seed_stride

    def observe(
        self,
        *,
        tile_kind: NDArray[np.int64],
        tile_crop: NDArray[np.int64],
        tile_animal: NDArray[np.int64],
        tile_cell: NDArray[np.int64],
        tile_role: NDArray[np.int64],
        tiles_int: NDArray[np.int64],
        tiles_float: NDArray[np.float32],
        actor_slot: NDArray[np.int64],
        actor_cell: NDArray[np.int64],
        actor_role: NDArray[np.int64],
        actor_mask: NDArray[np.bool_],
        actor_inventory: NDArray[np.int64],
        actor_inventory_rank: NDArray[np.int64],
        actors_float: NDArray[np.float32],
        player_features: NDArray[np.float32],
        storage_counts: NDArray[np.int64],
        storage_rank: NDArray[np.int64],
        banks: NDArray[np.float64],
        shop_type: NDArray[np.int64],
        shop_slot: NDArray[np.int64],
        shop_mask: NDArray[np.bool_],
        market_product: NDArray[np.int64],
        market_float: NDArray[np.float32],
        market_int: NDArray[np.int64],
        global_features: NDArray[np.float32],
        globals_int: NDArray[np.int64],
        still_playing: NDArray[np.bool_],
        order_limits: NDArray[np.int64],
        can_act: NDArray[np.bool_],
        rewards: NDArray[np.float32],
        dones: NDArray[np.bool_],
        transition_banks_before: NDArray[np.float64],
        transition_banks_after: NDArray[np.float64],
        transition_econ_before: NDArray[np.int64],
        transition_econ_after: NDArray[np.int64],
    ) -> None:
        arrays = {
            "tile_kind": tile_kind,
            "tile_crop": tile_crop,
            "tile_animal": tile_animal,
            "tile_cell": tile_cell,
            "tile_role": tile_role,
            "tiles_int": tiles_int,
            "tiles_float": tiles_float,
            "actor_slot": actor_slot,
            "actor_cell": actor_cell,
            "actor_role": actor_role,
            "actor_mask": actor_mask,
            "actor_inventory": actor_inventory,
            "actor_inventory_rank": actor_inventory_rank,
            "actors_float": actors_float,
            "player_features": player_features,
            "storage_counts": storage_counts,
            "storage_rank": storage_rank,
            "banks": banks,
            "shop_type": shop_type,
            "shop_slot": shop_slot,
            "shop_mask": shop_mask,
            "market_product": market_product,
            "market_float": market_float,
            "market_int": market_int,
            "global_features": global_features,
            "globals_int": globals_int,
            "still_playing": still_playing,
            "order_limits": order_limits,
            "can_act": can_act,
            "rewards": rewards,
            "dones": dones,
            "transition_banks_before": transition_banks_before,
            "transition_banks_after": transition_banks_after,
            "transition_econ_before": transition_econ_before,
            "transition_econ_after": transition_econ_after,
        }
        self._call("observe", arrays)

    def reset(
        self,
        *,
        tile_kind: NDArray[np.int64],
        tile_crop: NDArray[np.int64],
        tile_animal: NDArray[np.int64],
        tile_cell: NDArray[np.int64],
        tile_role: NDArray[np.int64],
        tiles_int: NDArray[np.int64],
        tiles_float: NDArray[np.float32],
        actor_slot: NDArray[np.int64],
        actor_cell: NDArray[np.int64],
        actor_role: NDArray[np.int64],
        actor_mask: NDArray[np.bool_],
        actor_inventory: NDArray[np.int64],
        actor_inventory_rank: NDArray[np.int64],
        actors_float: NDArray[np.float32],
        player_features: NDArray[np.float32],
        storage_counts: NDArray[np.int64],
        storage_rank: NDArray[np.int64],
        banks: NDArray[np.float64],
        shop_type: NDArray[np.int64],
        shop_slot: NDArray[np.int64],
        shop_mask: NDArray[np.bool_],
        market_product: NDArray[np.int64],
        market_float: NDArray[np.float32],
        market_int: NDArray[np.int64],
        global_features: NDArray[np.float32],
        globals_int: NDArray[np.int64],
        still_playing: NDArray[np.bool_],
        order_limits: NDArray[np.int64],
        can_act: NDArray[np.bool_],
        rewards: NDArray[np.float32],
        dones: NDArray[np.bool_],
        transition_banks_before: NDArray[np.float64],
        transition_banks_after: NDArray[np.float64],
        transition_econ_before: NDArray[np.int64],
        transition_econ_after: NDArray[np.int64],
    ) -> None:
        arrays = {
            "tile_kind": tile_kind,
            "tile_crop": tile_crop,
            "tile_animal": tile_animal,
            "tile_cell": tile_cell,
            "tile_role": tile_role,
            "tiles_int": tiles_int,
            "tiles_float": tiles_float,
            "actor_slot": actor_slot,
            "actor_cell": actor_cell,
            "actor_role": actor_role,
            "actor_mask": actor_mask,
            "actor_inventory": actor_inventory,
            "actor_inventory_rank": actor_inventory_rank,
            "actors_float": actors_float,
            "player_features": player_features,
            "storage_counts": storage_counts,
            "storage_rank": storage_rank,
            "banks": banks,
            "shop_type": shop_type,
            "shop_slot": shop_slot,
            "shop_mask": shop_mask,
            "market_product": market_product,
            "market_float": market_float,
            "market_int": market_int,
            "global_features": global_features,
            "globals_int": globals_int,
            "still_playing": still_playing,
            "order_limits": order_limits,
            "can_act": can_act,
            "rewards": rewards,
            "dones": dones,
            "transition_banks_before": transition_banks_before,
            "transition_banks_after": transition_banks_after,
            "transition_econ_before": transition_econ_before,
            "transition_econ_after": transition_econ_after,
        }
        self._call("reset", arrays)

    def step(
        self,
        tokens: NDArray[np.int64],
        lengths: NDArray[np.int64],
        *,
        tile_kind: NDArray[np.int64],
        tile_crop: NDArray[np.int64],
        tile_animal: NDArray[np.int64],
        tile_cell: NDArray[np.int64],
        tile_role: NDArray[np.int64],
        tiles_int: NDArray[np.int64],
        tiles_float: NDArray[np.float32],
        actor_slot: NDArray[np.int64],
        actor_cell: NDArray[np.int64],
        actor_role: NDArray[np.int64],
        actor_mask: NDArray[np.bool_],
        actor_inventory: NDArray[np.int64],
        actor_inventory_rank: NDArray[np.int64],
        actors_float: NDArray[np.float32],
        player_features: NDArray[np.float32],
        storage_counts: NDArray[np.int64],
        storage_rank: NDArray[np.int64],
        banks: NDArray[np.float64],
        shop_type: NDArray[np.int64],
        shop_slot: NDArray[np.int64],
        shop_mask: NDArray[np.bool_],
        market_product: NDArray[np.int64],
        market_float: NDArray[np.float32],
        market_int: NDArray[np.int64],
        global_features: NDArray[np.float32],
        globals_int: NDArray[np.int64],
        still_playing: NDArray[np.bool_],
        order_limits: NDArray[np.int64],
        can_act: NDArray[np.bool_],
        rewards: NDArray[np.float32],
        dones: NDArray[np.bool_],
        transition_banks_before: NDArray[np.float64],
        transition_banks_after: NDArray[np.float64],
        transition_econ_before: NDArray[np.int64],
        transition_econ_after: NDArray[np.int64],
    ) -> dict[str, list[float]]:
        arrays = {
            "tile_kind": tile_kind,
            "tile_crop": tile_crop,
            "tile_animal": tile_animal,
            "tile_cell": tile_cell,
            "tile_role": tile_role,
            "tiles_int": tiles_int,
            "tiles_float": tiles_float,
            "actor_slot": actor_slot,
            "actor_cell": actor_cell,
            "actor_role": actor_role,
            "actor_mask": actor_mask,
            "actor_inventory": actor_inventory,
            "actor_inventory_rank": actor_inventory_rank,
            "actors_float": actors_float,
            "player_features": player_features,
            "storage_counts": storage_counts,
            "storage_rank": storage_rank,
            "banks": banks,
            "shop_type": shop_type,
            "shop_slot": shop_slot,
            "shop_mask": shop_mask,
            "market_product": market_product,
            "market_float": market_float,
            "market_int": market_int,
            "global_features": global_features,
            "globals_int": globals_int,
            "still_playing": still_playing,
            "order_limits": order_limits,
            "can_act": can_act,
            "rewards": rewards,
            "dones": dones,
            "transition_banks_before": transition_banks_before,
            "transition_banks_after": transition_banks_after,
            "transition_econ_before": transition_econ_before,
            "transition_econ_after": transition_econ_after,
        }
        self._call("step", arrays, (tokens, lengths))
        return {"fake_metric": [1.0]}

    def truncate_envs(
        self,
        mask: NDArray[np.bool_],
        *,
        tile_kind: NDArray[np.int64],
        tile_crop: NDArray[np.int64],
        tile_animal: NDArray[np.int64],
        tile_cell: NDArray[np.int64],
        tile_role: NDArray[np.int64],
        tiles_int: NDArray[np.int64],
        tiles_float: NDArray[np.float32],
        actor_slot: NDArray[np.int64],
        actor_cell: NDArray[np.int64],
        actor_role: NDArray[np.int64],
        actor_mask: NDArray[np.bool_],
        actor_inventory: NDArray[np.int64],
        actor_inventory_rank: NDArray[np.int64],
        actors_float: NDArray[np.float32],
        player_features: NDArray[np.float32],
        storage_counts: NDArray[np.int64],
        storage_rank: NDArray[np.int64],
        banks: NDArray[np.float64],
        shop_type: NDArray[np.int64],
        shop_slot: NDArray[np.int64],
        shop_mask: NDArray[np.bool_],
        market_product: NDArray[np.int64],
        market_float: NDArray[np.float32],
        market_int: NDArray[np.int64],
        global_features: NDArray[np.float32],
        globals_int: NDArray[np.int64],
        still_playing: NDArray[np.bool_],
        order_limits: NDArray[np.int64],
        can_act: NDArray[np.bool_],
        rewards: NDArray[np.float32],
        dones: NDArray[np.bool_],
        transition_banks_before: NDArray[np.float64],
        transition_banks_after: NDArray[np.float64],
        transition_econ_before: NDArray[np.int64],
        transition_econ_after: NDArray[np.int64],
    ) -> None:
        arrays = {
            "tile_kind": tile_kind,
            "tile_crop": tile_crop,
            "tile_animal": tile_animal,
            "tile_cell": tile_cell,
            "tile_role": tile_role,
            "tiles_int": tiles_int,
            "tiles_float": tiles_float,
            "actor_slot": actor_slot,
            "actor_cell": actor_cell,
            "actor_role": actor_role,
            "actor_mask": actor_mask,
            "actor_inventory": actor_inventory,
            "actor_inventory_rank": actor_inventory_rank,
            "actors_float": actors_float,
            "player_features": player_features,
            "storage_counts": storage_counts,
            "storage_rank": storage_rank,
            "banks": banks,
            "shop_type": shop_type,
            "shop_slot": shop_slot,
            "shop_mask": shop_mask,
            "market_product": market_product,
            "market_float": market_float,
            "market_int": market_int,
            "global_features": global_features,
            "globals_int": globals_int,
            "still_playing": still_playing,
            "order_limits": order_limits,
            "can_act": can_act,
            "rewards": rewards,
            "dones": dones,
            "transition_banks_before": transition_banks_before,
            "transition_banks_after": transition_banks_after,
            "transition_econ_before": transition_econ_before,
            "transition_econ_after": transition_econ_after,
        }
        self._call("truncate_envs", arrays, (mask,))

    def terminal_metrics(self, env_index: int):
        self.events.append(f"terminal_metrics:{env_index}")

    def state_snapshot(self, env_index: int) -> str:
        self.events.append(f"state_snapshot:{env_index}")
        return '{"public":{"step":0}}'

    def seed_state(self) -> tuple[int, tuple[int, ...]]:
        return self.next_seed, self.current_seeds
