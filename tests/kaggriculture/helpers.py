"""Shared Kaggriculture model-test helpers (heads and teacher tests).

Tiny CPU models, hand-built grammar programs and replay cases on the synthetic
``expected_grammar_tables``; moved from ``test_model_heads.py`` (Phase 4.1).
"""

from __future__ import annotations

from typing import Any

import torch
from owl.kaggriculture import gpu_grammar as gg
from owl.kaggriculture import types as kt
from owl.model import kaggriculture as km

from tests.kaggriculture.conftest import make_obs

F, K = kt.MAX_FRAMES, kt.ACTION_SLOTS
PASS, NORTH = kt.UNIT_KINDS.index("PASS"), kt.UNIT_KINDS.index("NORTH")
PICKUP, PLACE = kt.UNIT_KINDS.index("PICKUP"), kt.UNIT_KINDS.index("PLACE")
PLANT = kt.UNIT_KINDS.index("PLANT")
NONE, HIRE = gg.MARKET_NONE, gg.MARKET_HIRE
BUY_SEED, SELL, EMPTY = (
    kt.MARKET_KINDS.index("BUY_SEED"),
    kt.MARKET_KINDS.index("SELL"),
    gg.MARKET_EMPTY,
)


def _tiny(*, hire_limit: int = 241, seed: int = 5, **overrides: Any) -> Any:
    torch.manual_seed(seed)
    fields: dict[str, Any] = {
        "embed_dim": 16,
        "depth": 1,
        "n_heads": 2,
        "mlp_ratio": 2.0,
        "n_scratch_tokens": 1,
    }
    return km.KaggricultureTransformer(
        km.KaggricultureTransformerConfig(**(fields | overrides)),
        obs_spec=kt.KaggricultureObsConfig(),
        action_spec=kt.KaggricultureActionConfig(hire_limit=hire_limit),
    )


def _obs_fields() -> list[str]:
    return [
        name for name in kt.KaggricultureObsBatch.model_fields if name != "action_mask"
    ]


def _map_obs(obs: kt.KaggricultureObsBatch, fn: Any) -> kt.KaggricultureObsBatch:
    return kt.KaggricultureObsBatch(
        **{name: fn(getattr(obs, name)) for name in _obs_fields()},
        action_mask=kt.KaggricultureActionMask(can_act=fn(obs.action_mask.can_act)),
    )


def _cat_obs(batches: list[kt.KaggricultureObsBatch]) -> kt.KaggricultureObsBatch:
    return kt.KaggricultureObsBatch(
        **{
            name: torch.cat([getattr(b, name) for b in batches])
            for name in _obs_fields()
        },
        action_mask=kt.KaggricultureActionMask(
            can_act=torch.cat([b.action_mask.can_act for b in batches])
        ),
    )


def _take_obs(obs: kt.KaggricultureObsBatch, index: torch.Tensor) -> Any:
    return _map_obs(obs, lambda t: t.index_select(0, index))


def _take_actions(actions: kt.KaggricultureActions, index: torch.Tensor) -> Any:
    return kt.KaggricultureActions(
        tokens=actions.tokens.index_select(0, index),
        lengths=actions.lengths.index_select(0, index),
    )


def _set_order_limits(obs: kt.KaggricultureObsBatch, limits: list[int]) -> None:
    """Per-env order limits (both seats), with the matching ``can_act``."""
    own = obs.actor_mask[..., : kt.MAX_ACTORS].sum(-1)
    obs.order_limits[:] = torch.tensor(limits)[:, None]
    frames = torch.arange(F)
    obs.action_mask.can_act[:] = frames < (own + obs.order_limits + 1)[..., None]


def _obs_double(obs: kt.KaggricultureObsBatch) -> kt.KaggricultureObsBatch:
    return _map_obs(
        obs, lambda t: t.double() if t.dtype == torch.float32 else t.clone()
    )


Unit = tuple[int, int, int, int, int]  # kind, item, present, high, low
Order = tuple[int, int, int, int]  # kind, item, high, low


def _seat_program(units: list[Unit], orders: list[Order]) -> tuple[torch.Tensor, int]:
    tokens = torch.zeros(F, K, dtype=torch.int64)
    for actor, (kind, item, present, high, low) in enumerate(units):
        tokens[actor, :7] = torch.tensor([actor, kind, 0, item, present, high, low])
    base = len(units)
    for position, (kind, item, high, low) in enumerate(orders):
        tokens[base + position, 7:11] = torch.tensor([kind, item, high, low])
    tokens[base + len(orders), 11] = 1  # STOP: market NONE with the stop bit
    return tokens, base + len(orders) + 1


def _actions(programs: list[list[tuple[torch.Tensor, int]]]) -> kt.KaggricultureActions:
    return kt.KaggricultureActions(
        tokens=torch.stack([torch.stack([t for t, _ in env]) for env in programs]),
        lengths=torch.tensor([[n for _, n in env] for env in programs]),
    )


def _base_case() -> tuple[kt.KaggricultureObsBatch, kt.KaggricultureActions]:
    """Hand-built programs for one env.

    Seat 0: 2 actors, STOP at queue position 2 of 3; seat 1: 3 actors, STOP at
    the forced sentinel (position 3).
    """
    obs = make_obs(envs=1, own_actors=2, rival_actors=3, order_limit=3)
    seat0 = _seat_program(
        [(PICKUP, 2, 1, 1, 5), (PASS, 0, 0, 0, 0)],
        [(SELL, 3, 0, 7), (EMPTY, 0, 0, 0)],
    )
    seat1 = _seat_program(
        [(NORTH, 0, 0, 0, 0)] * 3,
        [(HIRE, 0, 0, 0), (BUY_SEED, 4, 31, 31), (EMPTY, 0, 0, 0)],
    )
    return obs, _actions([[seat0, seat1]])


def _replay_case(name: str) -> tuple[Any, kt.KaggricultureObsBatch]:
    if name == "dense":
        return _tiny(), make_obs(
            envs=2, own_actors=241, rival_actors=241, order_limit=10
        )
    if name == "hire_capacity":
        model = _tiny()
        with torch.no_grad():
            bias = model.actor.heads["market_kind"].out.bias
            bias[HIRE] = 8.0
            bias[NONE] = -8.0
        obs = make_obs(
            envs=3,
            own_actors=(238, 240, 241),
            rival_actors=(239, 1, 236),
            order_limit=10,
        )
        return model, obs
    if name == "stop_every_position":
        model = _tiny()
        with torch.no_grad():
            model.actor.heads["market_kind"].out.bias[NONE] = -30.0
        obs = _cat_obs(
            [
                make_obs(own_actors=1 + p % 3, rival_actors=2, order_limit=max(p, 1))
                for p in range(11)
            ]
        )
        _set_order_limits(obs, list(range(11)))
        return model, obs
    assert name == "mixed"
    obs = _cat_obs(
        [
            make_obs(envs=2, own_actors=(1, 241), rival_actors=(7, 3), order_limit=10),
            make_obs(envs=2, own_actors=(5, 2), rival_actors=(1, 60), order_limit=4),
            make_obs(envs=1, own_actors=9, rival_actors=4, order_limit=1),
        ]
    )
    _set_order_limits(obs, [10, 3, 4, 0, 1])
    obs.still_playing[1, 0] = False
    obs.still_playing[3, 1] = False
    return _tiny(), obs
