"""Pinned native Kaggriculture environments for the existing PPO trainer.

Rust owns reset, transitions, feature encoding and action decoding. Python owns
buffer lifetime, terminal reward conversion and synchronous auto-reset. Public
tensor objects are reused: retain rollout observations by copying before step.
"""

from __future__ import annotations

import json
import warnings
from typing import Any

import numpy as np
import torch

from owl.kaggriculture.native_bridge import (
    MyolieBuffers,
    RustBatch,
)
from owl.kaggriculture.rewards import (
    ECON_FIELDS,
    KaggricultureRewardConfig,
    RewardMode,
    economic_rewards,
    terminal_rewards,
)
from owl.kaggriculture.types import (
    ACTION_SLOTS,
    MAX_ACTORS,
    MAX_FRAMES,
    KaggricultureActionConfig,
    KaggricultureActionMask,
    KaggricultureActions,
    KaggricultureObsBatch,
    KaggricultureObsConfig,
)
from owl.rs import kaggriculture_seed_header


def native_seed_header(seed: int, configuration: dict[str, Any]) -> dict[str, Any]:
    """Create a genuine native initial state for the legacy batch reset envelope."""
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    return json.loads(
        kaggriculture_seed_header(
            json.dumps({"seed": seed, "configuration": configuration})
        )
    )


class KaggricultureVectorizedEnv:
    """Two-seat native games, with terminal rewards followed by immediate reset."""

    def __init__(
        self,
        *,
        n_envs: int,
        obs_spec: KaggricultureObsConfig | None = None,
        action_spec: KaggricultureActionConfig | None = None,
        reward_mode: RewardMode = "win_loss",
        reward_shaping: KaggricultureRewardConfig | None = None,
        pin_memory: bool = True,
        seed: int = 0,
        seed_stride: int = 1,
        configuration: dict[str, Any] | None = None,
        threads: int = 1,
    ) -> None:
        if type(n_envs) is not int or n_envs < 1:
            raise ValueError("n_envs must be a positive integer")
        if reward_mode not in ("win_loss", "win_only", "win_share", "margin"):
            raise ValueError("unsupported Kaggriculture reward mode")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        if type(seed_stride) is not int or seed_stride < 1:
            raise ValueError("seed_stride must be a positive integer")
        self.obs_spec = obs_spec or KaggricultureObsConfig()
        self.action_spec = action_spec or KaggricultureActionConfig()
        self.reward_mode = reward_mode
        self.reward_shaping = reward_shaping or KaggricultureRewardConfig()
        if self.reward_shaping.enabled and reward_mode not in ("win_loss", "win_share"):
            raise ValueError("economic shaping requires win_loss or win_share rewards")
        self.n_envs, self.n_players = n_envs, 2
        self.configuration = dict(configuration or {})
        if pin_memory and not torch.cuda.is_available():
            warnings.warn(
                "pin_memory=True requires CUDA; proceeding without pinned memory",
                RuntimeWarning,
                stacklevel=2,
            )
            pin_memory = False
        self.pin_memory_enabled = pin_memory
        self._next_seed = seed
        self._seed_stride = seed_stride
        self._terminal_snapshots: list[dict[str, Any] | None] = [None] * n_envs
        self._terminal_metrics: list[dict[str, float] | None] = [None] * n_envs
        self._batch = RustBatch(self._headers(n_envs), threads=threads)

        def allocate(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
            return torch.zeros(shape, dtype=dtype, pin_memory=pin_memory)

        self.observations = KaggricultureObsBatch(
            features=allocate((n_envs, 2, self.obs_spec.feature_count), torch.float32),
            context=allocate((n_envs, 4), torch.int64),
            entity_mask=allocate((n_envs, 2, MAX_ACTORS), torch.bool),
            still_playing=allocate((n_envs, 2), torch.bool),
            action_mask=KaggricultureActionMask(
                can_act=allocate((n_envs, 2, MAX_FRAMES), torch.bool)
            ),
        )
        self._banks = allocate((n_envs, 2), torch.float64)
        self._done = allocate((n_envs,), torch.uint8)
        self._econ = allocate((n_envs, 2, ECON_FIELDS), torch.uint64)
        self.previous_banks = allocate((n_envs, 2), torch.float64)
        self.current_banks = allocate((n_envs, 2), torch.float64)
        self.previous_econ = allocate((n_envs, 2, ECON_FIELDS), torch.uint64)
        self.current_econ = allocate((n_envs, 2, ECON_FIELDS), torch.uint64)
        self.rewards = allocate((n_envs, 2), torch.float32)
        self.dones = allocate((n_envs, 2), torch.bool)
        # Narrow once into reusable native input storage; rollout actions remain
        # int64 and are still checked before any narrowing can wrap a token.
        self._frames = np.empty((n_envs, 2, MAX_FRAMES, ACTION_SLOTS), dtype=np.int16)
        self._lengths = np.empty((n_envs, 2), dtype=np.int32)
        self._actor_indices = torch.arange(MAX_ACTORS)[None, None, :]
        self._frame_indices = torch.arange(MAX_FRAMES)[None, None, :]
        self._econ_array = self._econ.numpy()
        self._buffers = MyolieBuffers(
            n_envs,
            features=self.observations.features.numpy(),
            context=self.observations.context.numpy(),
            banks=self._banks.numpy(),
            done=self._done.numpy(),
            owner=self,
            observation_version=self.obs_spec.observation_version,
        )
        self._refresh_observations()
        self.previous_banks.copy_(self._banks)
        self.current_banks.copy_(self._banks)
        self.previous_econ.copy_(self._econ)
        self.current_econ.copy_(self._econ)

    def _headers(self, count: int) -> list[dict[str, Any]]:
        headers = [
            native_seed_header(
                self._next_seed + i * self._seed_stride, self.configuration
            )
            for i in range(count)
        ]
        self._next_seed += count * self._seed_stride
        return headers

    def _refresh_observations(self) -> KaggricultureObsBatch:
        self._batch.myolie_observe(self._buffers)
        self._batch.econ(self._econ_array)
        self._update_masks()
        return self.observations

    def _update_masks(self) -> None:
        obs = self.observations
        counts = obs.context[:, 1:3]
        obs.entity_mask.copy_(self._actor_indices < counts[..., None])
        obs.still_playing.copy_(~self._done.bool()[:, None].expand(-1, 2))
        # The decoder determines early market STOP; this is the envelope of
        # possible frames, not a claim that every envelope position was sampled.
        envelope = counts + obs.context[:, 3:4] + 1
        obs.action_mask.can_act.copy_(
            (self._frame_indices < envelope[..., None]) & obs.still_playing[..., None]
        )

    def reset(self) -> KaggricultureObsBatch:
        self._batch.reset(self._headers(self.n_envs))
        self.rewards.zero_()
        self.dones.zero_()
        self._terminal_snapshots = [None] * self.n_envs
        self._terminal_metrics = [None] * self.n_envs
        obs = self._refresh_observations()
        self.previous_banks.copy_(self._banks)
        self.current_banks.copy_(self._banks)
        self.previous_econ.copy_(self._econ)
        self.current_econ.copy_(self._econ)
        return obs

    def step(
        self, actions: KaggricultureActions
    ) -> tuple[
        KaggricultureObsBatch, torch.Tensor, torch.Tensor, dict[str, list[float]]
    ]:
        if not isinstance(actions, KaggricultureActions):
            raise TypeError("expected KaggricultureActions")
        expected = (self.n_envs, 2, MAX_FRAMES, ACTION_SLOTS)
        if actions.tokens.shape != expected or actions.lengths.shape != (
            self.n_envs,
            2,
        ):
            raise ValueError(
                "Kaggriculture action tensor shapes differ from the native ABI"
            )
        if actions.tokens.dtype != torch.int64 or actions.lengths.dtype != torch.int64:
            raise TypeError("Kaggriculture action tokens and lengths must be int64")
        if actions.tokens.device.type != "cpu" or actions.lengths.device.type != "cpu":
            raise ValueError("environment actions must be on CPU")
        if bool(((actions.lengths < 1) | (actions.lengths > MAX_FRAMES)).any()):
            raise ValueError("action lengths must be in 1..252")
        # Check before narrowing, preventing wraparound from admitting invalid
        # int64 tokens. Rust validates complete grammar and active frame bounds.
        if bool(((actions.tokens < 0) | (actions.tokens > 32767)).any()):
            raise ValueError("action tokens exceed the native signed-int16 domain")
        np.copyto(self._frames, actions.tokens.numpy(), casting="unsafe")
        np.copyto(self._lengths, actions.lengths.numpy(), casting="unsafe")
        self.previous_banks.copy_(self._banks)
        self.previous_econ.copy_(self._econ)
        self._batch.myolie_step_frames(self._frames, self._lengths, self._buffers)
        self._batch.econ(self._econ_array)
        self.current_banks.copy_(self._banks)
        self.current_econ.copy_(self._econ)
        if self.reward_shaping.enabled:
            self.rewards.copy_(
                economic_rewards(
                    self.previous_econ, self.current_econ, self.reward_shaping
                )
            )
        else:
            self.rewards.zero_()
        self.dones.copy_(self._done.bool()[:, None].expand(-1, 2))
        self._terminal_snapshots = [None] * self.n_envs
        self._terminal_metrics = [None] * self.n_envs
        finished = self._done.nonzero().flatten().tolist()
        metrics: dict[str, list[float]] = {}
        if finished:
            snapshots = self._batch.snapshots()
            metrics = {"total_games_played": [], "bank": [], "margin": []}
            for i in finished:
                bank = self._banks[i]
                reward = terminal_rewards(bank, self.reward_mode, self.reward_shaping)
                self.rewards[i].add_(reward * self.reward_shaping.terminal_scale)
                self._terminal_snapshots[i] = snapshots[i]
                bank0, bank1 = bank.tolist()
                self._terminal_metrics[i] = {
                    "bank_0": bank0,
                    "bank_1": bank1,
                    "margin_0": bank0 - bank1,
                    "episode_steps": float(self.observations.context[i, 0]),
                }
                metrics["total_games_played"].append(1.0)
                metrics["bank"].extend((bank0, bank1))
                metrics["margin"].extend((bank0 - bank1, bank1 - bank0))
            self._batch.myolie_reset_indices(
                np.asarray(finished, dtype=np.uint32),
                self._headers(len(finished)),
                observation_version=self.obs_spec.observation_version,
            )
            self._refresh_observations()
        else:
            self._update_masks()
        return self.observations, self.rewards, self.dones, metrics

    def truncate_envs(
        self, truncate_mask: np.ndarray[Any, Any] | torch.Tensor
    ) -> KaggricultureObsBatch:
        if isinstance(truncate_mask, torch.Tensor):
            truncate_mask = truncate_mask.cpu().numpy()
        if truncate_mask.dtype != np.bool_ or truncate_mask.shape != (self.n_envs,):
            raise ValueError("truncate mask must be bool[n_envs]")
        indices = np.flatnonzero(truncate_mask).tolist()
        if indices:
            self._batch.myolie_reset_indices(
                np.asarray(indices, dtype=np.uint32),
                self._headers(len(indices)),
                observation_version=self.obs_spec.observation_version,
            )
            for i in indices:
                self._terminal_snapshots[i] = None
                self._terminal_metrics[i] = None
            self._refresh_observations()
        return self.observations

    def state_snapshot(self, env_index: int) -> dict[str, Any]:
        return self._batch.snapshots()[env_index]

    def terminal_snapshot(self, env_index: int) -> dict[str, Any] | None:
        return self._terminal_snapshots[env_index]

    def terminal_metrics(self, env_index: int) -> dict[str, float] | None:
        return self._terminal_metrics[env_index]

    def close(self) -> None:
        self._batch.close()

    def __enter__(self) -> KaggricultureVectorizedEnv:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
