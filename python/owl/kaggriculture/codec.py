"""Cold JSON/replay wrappers around the native Kaggriculture grammar.

Task 1.4 supplies the runtime bindings. There is no Python grammar or fallback;
the live environment transports action tensors directly instead of using JSON.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np
import torch
from numpy.typing import NDArray

from owl import rs
from owl.kaggriculture.types import (
    ACTION_SLOTS,
    MAX_ACTORS,
    MAX_FRAMES,
    PLAYERS,
    JsonValue,
    KaggricultureActionConfig,
    KaggricultureActions,
    KaggricultureObsBatch,
)


def encode_action_into(
    action: dict[str, JsonValue],
    *,
    actors: int,
    order_limit: int,
    hire_limit: int,
    out: NDArray[np.int64],
) -> int:
    """Encode one action, letting native validation publish atomically."""
    return rs.kaggriculture_encode(
        json.dumps(action, allow_nan=False, separators=(",", ":")),
        actors,
        order_limit,
        hire_limit,
        out,
    )


def decode_action(
    tokens: NDArray[np.int64],
    length: int,
    *,
    actors: int,
    order_limit: int,
    hire_limit: int,
) -> dict[str, JsonValue]:
    """Decode one canonical native program, preserving command list order."""
    result: JsonValue = json.loads(
        rs.kaggriculture_decode(tokens, length, actors, order_limit, hire_limit)
    )
    if not isinstance(result, dict):
        raise ValueError("native Kaggriculture decoder must return a JSON object")
    return result


def encode_actions(
    actions: Sequence[tuple[dict[str, JsonValue], dict[str, JsonValue]]],
    obs: KaggricultureObsBatch,
    *,
    action_spec: KaggricultureActionConfig,
) -> KaggricultureActions:
    """Encode a cold batch into private CPU tensors, returning only on success."""
    obs.check_contract()
    n_envs = obs.still_playing.shape[0]
    if len(actions) != n_envs:
        raise ValueError(
            "actions must contain exactly one pair per observation's environments"
        )
    if any(len(pair) != PLAYERS for pair in actions):
        raise ValueError("actions must contain exactly two seats per environment")
    tokens = torch.empty((n_envs, PLAYERS, MAX_FRAMES, ACTION_SLOTS), dtype=torch.int64)
    lengths = torch.empty((n_envs, PLAYERS), dtype=torch.int64)
    rows = tokens.numpy()
    for env, pair in enumerate(actions):
        for seat, action in enumerate(pair):
            lengths[env, seat] = encode_action_into(
                action,
                actors=int(obs.actor_mask[env, seat, :MAX_ACTORS].sum().item()),
                order_limit=int(obs.order_limits[env, seat].item()),
                hire_limit=action_spec.hire_limit,
                out=rows[env, seat],
            )
    return KaggricultureActions(tokens=tokens, lengths=lengths)


def decode_actions(
    actions: KaggricultureActions,
    obs: KaggricultureObsBatch,
    *,
    action_spec: KaggricultureActionConfig,
) -> list[tuple[dict[str, JsonValue], dict[str, JsonValue]]]:
    """Decode a cold CPU batch using each seat's current legal observation."""
    obs.check_contract()
    n_envs = obs.still_playing.shape[0]
    for name, tensor, shape in (
        ("tokens", actions.tokens, (n_envs, PLAYERS, MAX_FRAMES, ACTION_SLOTS)),
        ("lengths", actions.lengths, (n_envs, PLAYERS)),
    ):
        if tensor.device.type != "cpu" or tensor.dtype != torch.int64:
            raise ValueError(f"{name} must be a CPU int64 tensor")
        if tuple(tensor.shape) != shape or not tensor.is_contiguous():
            raise ValueError(f"{name} must be C-contiguous with shape {shape}")
    rows = actions.tokens.numpy()
    result = []
    for env in range(n_envs):
        seats = [
            decode_action(
                rows[env, seat],
                int(actions.lengths[env, seat].item()),
                actors=int(obs.actor_mask[env, seat, :MAX_ACTORS].sum().item()),
                order_limit=int(obs.order_limits[env, seat].item()),
                hire_limit=action_spec.hire_limit,
            )
            for seat in range(PLAYERS)
        ]
        result.append((seats[0], seats[1]))
    return result
