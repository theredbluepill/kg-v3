"""The Kaggle Kaggriculture agent: one seat, current observation only.

Each call encodes the seat's own Kaggle observation with the native
``write_seat`` (``kaggle_view``), runs one eager CPU fp32 forward of the
checkpoint's ``KaggricultureTransformer``, decodes the program with the native
grammar and validates the resulting action by a native encode round trip.

Nothing carries between turns except the loaded weights, the reusable buffers
(fully overwritten each call) and telemetry counters that never feed the model.
With ``final_turn_liquidation`` (off by default) the validated action of the
last resolved turn is rewritten by the stateless rule in ``final_turn`` and
validated again.

``act`` is the Kaggle process boundary. An exception there loses the episode,
so it returns the default PASS program and counts the failure, unless the agent
is strict (every test, benchmark and qualifying episode), which re-raises.
Loading failures are never caught.

With ``block_late_investments`` (default off) the validated action passes
through ``late_invest.filter_late_investments``: each market purchase that
cannot be sold back before the game ends becomes the empty market command, and
the filtered action is validated again. The model is untouched. When both
rules are on, rule 2 runs first and rule 1 then rewrites the final turn.
"""

from __future__ import annotations

import json
import traceback
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
import yaml
from pydantic import ConfigDict

from owl.config import BaseConfig
from owl.kaggriculture.codec import decode_action, encode_action_into
from owl.kaggriculture.config import KaggricultureEnvConfig
from owl.kaggriculture.env import allocate_single_seat_buffers
from owl.kaggriculture.final_turn import liquidate_final_turn
from owl.kaggriculture.kaggle_view import SeatArrays, encode_seat_into, seat_view_json
from owl.kaggriculture.late_invest import (
    GameClock,
    filter_late_investments,
    observation_step,
)
from owl.kaggriculture.types import (
    ACTION_SLOTS,
    MAX_ACTORS,
    MAX_FRAMES,
    JsonValue,
    KaggricultureObsBatch,
)
from owl.model.factory import create_model
from owl.model.kaggriculture import (
    KaggricultureTransformer,
    KaggricultureTransformerConfig,
)

ACTION_KEYS = ("farmer", "hands", "market")
WARMUP_PATH = Path(__file__).resolve().parent / "kaggle_warmup.json"
# Checkpoint state names that would carry between-turn memory or opponent
# identity; the stateless policy Decision prohibits both.
PROHIBITED_STATE_MARKERS = ("hidden", "recurrent", "memory", "opponent")


def pass_action() -> dict[str, JsonValue]:
    """Kaggriculture's schema default action: a legal PASS program."""
    return {"farmer": ["PASS"], "hands": [], "market": []}


class ActionValidationError(ValueError):
    """A returned action is not a canonical, natively encodable program."""


class KaggricultureAgentCheckpointConfig(BaseConfig):
    """The two sections of a training ``config.yaml`` the agent needs."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    env: KaggricultureEnvConfig
    model: KaggricultureTransformerConfig


def load_checkpoint_config(path: Path) -> KaggricultureAgentCheckpointConfig:
    """Load the packaged config; ``env.action_spec`` must be explicit."""
    with path.open() as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or not isinstance(raw.get("env"), dict):
        raise ValueError(f"{path} must contain an env mapping")
    if "action_spec" not in raw["env"]:
        raise ValueError(f"{path} must set env.action_spec (hire_limit is not implied)")
    return KaggricultureAgentCheckpointConfig.model_validate(raw)


def load_model(
    config: KaggricultureAgentCheckpointConfig, checkpoint_path: Path
) -> KaggricultureTransformer:
    """Build the CPU fp32 model and load its weights strictly.

    Only the runtime dispatch flag ``force_flash_attn`` is overridden: it is not
    a weight, and CUDA FlashAttention cannot run on CPU. The model is built on
    CPU, not the meta device, because its native grammar tables are
    non-persistent buffers that ``load_state_dict`` never fills.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"model"}:
        keys = sorted(checkpoint) if isinstance(checkpoint, dict) else type(checkpoint)
        raise ValueError(
            f"{checkpoint_path} must be a slim checkpoint with only 'model', got {keys}"
        )
    state = checkpoint["model"]
    if not isinstance(state, dict):
        raise ValueError(f"{checkpoint_path} 'model' must be a state dict")
    prohibited = [
        name
        for name in state
        if any(marker in name.lower() for marker in PROHIBITED_STATE_MARKERS)
    ]
    if prohibited:
        raise ValueError(f"checkpoint carries prohibited state: {prohibited[:5]}")
    not_fp32 = [
        name
        for name, tensor in state.items()
        if tensor.is_floating_point() and tensor.dtype != torch.float32
    ]
    if not_fp32:
        raise ValueError(f"checkpoint tensors must be fp32: {not_fp32[:5]}")
    model_config = config.model.model_copy(update={"force_flash_attn": False})
    model = create_model(
        model_config,
        obs_spec=config.env.obs_spec,
        action_spec=config.env.action_spec,
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _check_command(command: object, *, where: str, allow_empty: bool) -> None:
    if type(command) is not list:
        raise ActionValidationError(f"{where} must be a list command")
    if not command:
        if allow_empty:
            return
        raise ActionValidationError(f"{where} must not be empty")
    if type(command[0]) is not str:
        raise ActionValidationError(f"{where} op must be a string")
    for argument in command[1:]:
        if type(argument) not in (str, int):
            raise ActionValidationError(
                f"{where} arguments must be strings or integers, got {argument!r}"
            )


def validate_action(
    action: object,
    *,
    actors: int,
    order_limit: int,
    hire_limit: int,
    scratch: np.ndarray,
) -> dict[str, JsonValue]:
    """Require a canonical program that the native grammar re-encodes exactly.

    Kaggle silently replaces a non-dict return with PASS, so the structure is
    checked here, then the dict round-trips through native encode and decode
    with the same actor, order and hire context used to decode it.
    """
    if type(action) is not dict or tuple(action) != ACTION_KEYS:
        raise ActionValidationError(
            f"action must be a dict with keys {ACTION_KEYS}, got {action!r}"
        )
    _check_command(action["farmer"], where="farmer", allow_empty=False)
    for key in ("hands", "market"):
        commands = action[key]
        if type(commands) is not list:
            raise ActionValidationError(f"{key} must be a list of commands")
        for index, command in enumerate(commands):
            _check_command(
                command, where=f"{key}[{index}]", allow_empty=key == "market"
            )
    try:
        length = encode_action_into(
            action,
            actors=actors,
            order_limit=order_limit,
            hire_limit=hire_limit,
            out=scratch,
        )
        canonical = decode_action(
            scratch,
            length,
            actors=actors,
            order_limit=order_limit,
            hire_limit=hire_limit,
        )
    except ValueError as error:
        raise ActionValidationError(f"native round trip failed: {error}") from error
    if canonical != action:
        raise ActionValidationError(f"action is not canonical: {action!r}")
    return action


class KaggricultureAgent:
    """One Kaggle seat's policy; see the module docstring for the contract."""

    def __init__(
        self,
        model_root: Path,
        *,
        deterministic: bool,
        strict: bool,
        min_overage_time: float,
        final_turn_liquidation: bool = False,
        block_late_investments: bool = False,
    ) -> None:
        start = perf_counter()
        self.config = load_checkpoint_config(model_root / "config.yaml")
        self.model = load_model(self.config, model_root / "checkpoint.pt")
        self.hire_limit = self.config.env.action_spec.hire_limit
        self.deterministic = deterministic
        self.strict = strict
        self.min_overage_time = min_overage_time
        self.final_turn_liquidation = final_turn_liquidation
        self.liquidations = 0
        self.block_late_investments = block_late_investments
        self.obs: KaggricultureObsBatch = allocate_single_seat_buffers()
        self.arrays = SeatArrays.of(self.obs)
        self._scratch = np.empty((MAX_FRAMES, ACTION_SLOTS), dtype=np.int64)
        self.calls = 0
        self.caught_errors = 0
        self.budget_passes = 0
        self.blocked_orders = 0
        self._logged_dropped: set[str] = set()
        print(
            f"kg load_s={perf_counter() - start:.2f} "
            f"final_turn_liquidation={int(final_turn_liquidation)} "
            f"block_late_investments={int(block_late_investments)}",
            flush=True,
        )

    def warm_up(self) -> None:
        """Run one unguarded turn on the bundled step-0 observation.

        Called at load so the first forward's one-time cost lands in turn 0's
        overage bank, and so a model that cannot run fails the load loudly
        instead of passing every turn behind the guard.
        """
        with WARMUP_PATH.open() as handle:
            fixture = json.load(handle)
        self.policy_action(fixture["observation"], fixture["configuration"])

    def act(
        self, observation: Mapping[str, Any], configuration: Mapping[str, Any]
    ) -> dict[str, JsonValue]:
        """The guarded Kaggle boundary: a validated action, or PASS on failure."""
        self.calls += 1
        try:
            remaining = observation["remainingOverageTime"]
            if remaining < self.min_overage_time:
                self.budget_passes += 1
                print(
                    f"kg step={observation['step']} overage={remaining:.2f} "
                    f"below {self.min_overage_time:.2f}: PASS",
                    flush=True,
                )
                return pass_action()
            return self.policy_action(observation, configuration)
        except Exception:
            if self.strict:
                raise
            self.caught_errors += 1
            print(
                f"kg call={self.calls} caught_errors={self.caught_errors}: PASS",
                flush=True,
            )
            traceback.print_exc()
            return pass_action()

    def policy_action(
        self, observation: Mapping[str, Any], configuration: Mapping[str, Any]
    ) -> dict[str, JsonValue]:
        """Encode, forward, decode and validate one turn; raises on any failure."""
        start = perf_counter()
        view, seat, dropped = seat_view_json(observation, configuration)
        for key in dropped:
            if key not in self._logged_dropped:
                self._logged_dropped.add(key)
                print(f"kg dropped configuration key {key!r}", flush=True)
        encode_seat_into(view, seat, self.arrays)
        self.obs.check_single_seat_contract()
        encoded = perf_counter()
        with torch.inference_mode():
            output = self.model(self.obs, deterministic=self.deterministic)
        forwarded = perf_counter()
        tokens = output.actions.tokens[0, 0].numpy()
        length = int(output.actions.lengths[0, 0].item())
        actors = int(self.arrays.actor_mask[0, 0, :MAX_ACTORS].sum())
        order_limit = int(self.arrays.order_limits[0, 0])
        action = decode_action(
            tokens,
            length,
            actors=actors,
            order_limit=order_limit,
            hire_limit=self.hire_limit,
        )
        validate_action(
            action,
            actors=actors,
            order_limit=order_limit,
            hire_limit=self.hire_limit,
            scratch=self._scratch,
        )
        if self.block_late_investments:
            clock = GameClock.of(configuration)
            step = observation_step(observation, clock)
            action, reasons = filter_late_investments(action, step, clock)
            if reasons:
                action = validate_action(
                    action,
                    actors=actors,
                    order_limit=order_limit,
                    hire_limit=self.hire_limit,
                    scratch=self._scratch,
                )
                self.blocked_orders += len(reasons)
                print(f"kg step={step} late-invest blocked {list(reasons)}", flush=True)
        if self.final_turn_liquidation:
            liquidated = liquidate_final_turn(
                observation, configuration, action, order_limit=order_limit
            )
            if liquidated is not action:
                action = validate_action(
                    liquidated,
                    actors=actors,
                    order_limit=order_limit,
                    hire_limit=self.hire_limit,
                    scratch=self._scratch,
                )
                self.liquidations += 1
                print(
                    f"kg step={observation['step']} final_turn_liquidation "
                    f"farmer={action['farmer']} hands={action['hands']} "
                    f"market={action['market']}",
                    flush=True,
                )
        done = perf_counter()
        print(
            f"kg step={observation['step']} seat={seat} "
            f"enc_ms={1e3 * (encoded - start):.1f} "
            f"fwd_ms={1e3 * (forwarded - encoded):.1f} "
            f"dec_ms={1e3 * (done - forwarded):.1f} "
            f"overage={observation['remainingOverageTime']:.2f} len={length}",
            flush=True,
        )
        return action
