# mypy: allow-untyped-defs
"""Versioned actor-domain extension of the frozen demonstration grammar.

Copied from the artifact-owned demo_codec.py; the frozen source stays unchanged.
The default engine actor envelope is 241. The historical oracle defaults to a
HIRE limit of 16; v3 callers always supply the explicit v3 action-spec limit.
Only action syntax and masks are retained; all model migration code is removed.
Private oracle helpers retain their upstream dynamic typing; public codec
boundaries below declare their mapping contracts.
All previous quantity/order semantics are retained.

Masks constrain syntax, phase and bounded representation, not profitability,
state effects, affordability or simultaneous market fills. Empty market slots
and explicit zero market quantities survive exactly. Explicit unit transfer
quantities must be positive to pass the frozen evaluator's outer validator;
omission remains distinct from an explicit quantity. No legacy code is modified.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

SCHEMA = "myolie-demo-action-base32-actor241-v3"
MASK_SCHEMA = SCHEMA + "/syntax-masks"
QUANTITY_BASE = 32
MAX_QUANTITY = 1023
# Default engine hands reset daily: at most 24 turns * 10 HIRE orders + farmer.
# This is a conservative representational envelope, not an affordability claim.
MAX_ACTORS = 241
MAX_FRAMES = MAX_ACTORS + 10 + 1
LEGACY_HIRE_LIMIT = 16
UNIT_KIND_VALUES = (
    "NONE",
    "PASS",
    "NORTH",
    "SOUTH",
    "EAST",
    "WEST",
    "PICKUP",
    "PLACE",
    "PLANT",
    "WATER",
    "HARVEST",
    "DROP",
    "BUILD_COOP",
    "BUILD_PASTURE",
    "FEED",
    "FERTILIZE",
    "COLLECT_FERTILIZER",
    "CARE",
    "DIG",
    "RESERVED",
)
MARKET_KIND_VALUES = (
    "NONE",
    "HIRE",
    "BUY_LAND",
    "BUY_SEED",
    "BUY_PRODUCT",
    "BUY_ANIMAL",
    "SELL",
    "EMPTY",
)
ITEM_VALUES = (
    "NONE",
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
    "GOOSE",
    "COW",
    "SHEEP",
    "RESERVED_1",
    "RESERVED_2",
    "RESERVED_3",
)
SLOT_NAMES = (
    "unit_actor",
    "unit_kind",
    "unit_target",
    "unit_item",
    "unit_quantity_present",
    "unit_quantity_high",
    "unit_quantity",
    "market_kind",
    "market_item",
    "market_quantity_high",
    "market_quantity",
    "stop",
)
SLOT_WIDTHS = dict(
    zip(SLOT_NAMES, (MAX_ACTORS, 20, 128, 16, 2, 32, 32, 8, 16, 32, 32, 2), strict=True)
)
ACTION_SLOT_SCHEMA = tuple(
    {"name": name, "width": SLOT_WIDTHS[name]} for name in SLOT_NAMES
)
_FIELDS = ("farmer", "hands", "market")
_CROPS = tuple(range(1, 6))
_PRODUCTS = tuple(range(1, 10))
_ANIMALS = (10, 11, 12)
_TRANSFERS = {"PICKUP", "PLACE"}
_QUANTITY_ORDERS = {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"}


class CodecError(ValueError):
    """Unsupported raw syntax or malformed typed program; never repaired silently."""


def _require(condition, reason):
    if not condition:
        raise CodecError(reason)


def _legal_counts(legal_input):
    if legal_input is None:
        return None, 10
    try:
        observation = legal_input["observation"]
        player = observation["player"]
        _require(type(player) is int and player in (0, 1), "invalid public seat")
        actors = 1 + len(observation["farms"][player]["hands"])
        limit = max(
            1, int(legal_input["configuration"].get("maxMarketOrdersPerTurn", 10))
        )
    except (KeyError, TypeError, IndexError, ValueError) as exc:
        raise CodecError("incomplete legal input") from exc
    _require(actors <= MAX_ACTORS, f"at most {MAX_ACTORS} actors are represented")
    return actors, limit


def canonical_layout(legal_input):
    actors, _ = _legal_counts(legal_input)
    _require(actors is not None, "canonical layout requires legal input")
    return {"keys": list(_FIELDS), "hands_count": actors - 1}


def _layout(layout, legal_input=None):
    _require(
        isinstance(layout, Mapping) and set(layout) == {"keys", "hands_count"},
        "invalid layout",
    )
    keys, count = layout["keys"], layout["hands_count"]
    _require(
        isinstance(keys, list)
        and all(type(k) is str and k in _FIELDS for k in keys)
        and len(set(keys)) == len(keys),
        "unsupported action keys",
    )
    _require(type(count) is int and 0 <= count < MAX_ACTORS, "invalid hand count")
    _require("hands" in keys or count == 0, "absent hands cannot contain commands")
    actual, limit = _legal_counts(legal_input)
    _require(actual is None or count < actual, "commands exceed observed hand count")
    actors = ([0] if "farmer" in keys else []) + list(range(1, count + 1))
    return actors, limit


def _indices(kind, *, market=False):
    if market:
        if kind == "BUY_SEED":
            return _CROPS
        if kind == "BUY_PRODUCT":
            return (1, 9)
        if kind == "BUY_ANIMAL":
            return _ANIMALS
        if kind == "SELL":
            return _PRODUCTS
    elif kind == "PLANT":
        return _CROPS
    elif kind in _TRANSFERS:
        return tuple(range(1, 13))
    return (0,)


def _frame(value):
    # Accept a bare choice mapping or an explicitly wrapped training frame.
    if isinstance(value, Mapping) and set(value) == {"choices"}:
        value = value["choices"]
    _require(
        isinstance(value, Mapping) and tuple(value) == SLOT_NAMES,
        "frame slot order/schema differs",
    )
    _require(
        all(
            type(value[k]) is int and 0 <= value[k] < SLOT_WIDTHS[k] for k in SLOT_NAMES
        ),
        "frame token out of vocabulary",
    )
    return value


@lru_cache(maxsize=512)
def _boolean_mask(width, allowed):
    """Reuse immutable vocabulary masks; cache no observation or mutable prefix state.

    All input/layout/program validation stays in mask_for_slot on every call.
    The bounded cache only replaces repeated expansion of the same allowed set.
    """
    return tuple(i in allowed for i in range(width))


def mask_for_slot(
    slot,
    legal_input,
    previous_frames,
    prefix,
    *,
    layout=None,
    hire_limit=LEGACY_HIRE_LIMIT,
):
    """Conditional syntax mask; previous frames must be an admitted prefix.

    Default layout emits all observed actors and all three action fields.
    Layout metadata preserves optional fields/short hands lists for raw replay;
    it is not an unobserved model input or a learned output in this corpus.
    """
    _require(
        type(hire_limit) is int and 1 <= hire_limit <= MAX_ACTORS, "invalid hire limit"
    )
    _require(slot in SLOT_WIDTHS, "unknown slot")
    layout = canonical_layout(legal_input) if layout is None else layout
    actors, limit = _layout(layout, legal_input)
    unit_count = sum(int(frame["unit_kind"] != 0) for frame in previous_frames)
    orders = sum(int(frame["market_kind"] != 0) for frame in previous_frames)
    _require(not any(frame["stop"] for frame in previous_frames), "frame after STOP")
    _require(unit_count <= len(actors) and orders <= limit, "program exceeds shape")
    ready = unit_count == len(actors)
    unit = UNIT_KIND_VALUES[prefix.get("unit_kind", 0)]
    market = MARKET_KIND_VALUES[prefix.get("market_kind", 0)]
    can_hire = True
    if slot == "market_kind":
        actual_actors, _ = _legal_counts(legal_input)
        actor_bound = (
            actual_actors if actual_actors is not None else 1 + layout["hands_count"]
        )
        hire_requests = sum(frame["market_kind"] == 1 for frame in previous_frames)
        can_hire = actor_bound + hire_requests < hire_limit
    allowed = _allowed_for_slot(
        slot,
        0 if ready else actors[unit_count],
        ready,
        orders < limit and "market" in layout["keys"],
        can_hire,
        unit,
        market,
        prefix,
    )
    return _boolean_mask(SLOT_WIDTHS[slot], allowed)


def _allowed_for_slot(slot, actor, ready, can_market, can_hire, unit, market, prefix):
    """Syntax choices for an already checked shape/prefix; no neural or RNG work."""
    if slot == "unit_actor":
        allowed = (actor,)
    elif slot == "unit_kind":
        allowed = (0,) if ready else tuple(range(1, 19))
    elif slot == "unit_target":
        allowed = (0,)
    elif slot == "unit_item":
        allowed = _indices(unit)
    elif slot == "unit_quantity_present":
        allowed = (0, 1) if unit in _TRANSFERS else (0,)
    elif slot in {"unit_quantity_high", "unit_quantity"}:
        allowed = (
            tuple(range(32))
            if unit in _TRANSFERS and prefix.get("unit_quantity_present")
            else (0,)
        )
        # The official interpreter permits unit quantity zero, but the frozen
        # evaluator's unit validator rejects it. The saved corpus has no such
        # targets. Keep market zero/empty slots exact; restrict only emission
        # of explicit transfer quantities at this declared support boundary.
        if (
            slot == "unit_quantity"
            and unit in _TRANSFERS
            and prefix.get("unit_quantity_present")
            and prefix.get("unit_quantity_high") == 0
        ):
            allowed = tuple(range(1, 32))
    elif slot == "market_kind":
        allowed = tuple(range(8)) if ready and unit == "NONE" and can_market else (0,)
        # Checkpoint-owned policy support is separate from actor representation.
        # Migrated models retain the historical 16-actor HIRE mask. Explicit
        # activation can widen it; failed HIRE requests still consume capacity.
        if not can_hire:
            allowed = tuple(index for index in allowed if index != 1)
    elif slot == "market_item":
        allowed = _indices(market, market=True)
    elif slot in {"market_quantity_high", "market_quantity"}:
        allowed = tuple(range(32)) if market in _QUANTITY_ORDERS else (0,)
    else:
        # Always append a distinct sentinel, including after a full queue.
        allowed = (int(ready and unit == "NONE" and market == "NONE"),)
    return allowed


class _SamplingGrammar:
    """Private cursor for FastRuntime's canonical, mask-admitted generation.

    Shape is checked once. Counters advance only after a complete sampled frame;
    every token must come from options(). Public mask/decode entry points keep
    their full validation for arbitrary caller-supplied programs. Do not use this
    cursor to admit replay data or externally constructed frames.
    """

    def __init__(self, legal, layout, hire_limit):
        _require(
            type(hire_limit) is int and 1 <= hire_limit <= MAX_ACTORS,
            "invalid hire limit",
        )
        self.actors, self.limit = _layout(layout, legal)
        actual, _ = _legal_counts(legal)
        self.actor_bound = actual if actual is not None else 1 + layout["hands_count"]
        self.has_market = "market" in layout["keys"]
        self.hire_limit = hire_limit
        self.units = self.orders = self.hires = 0
        self.stopped = False

    def options(self, slot, prefix):
        _require(not self.stopped, "frame after STOP")
        ready = self.units == len(self.actors)
        allowed = _allowed_for_slot(
            slot,
            0 if ready else self.actors[self.units],
            ready,
            self.orders < self.limit and self.has_market,
            self.actor_bound + self.hires < self.hire_limit,
            UNIT_KIND_VALUES[prefix.get("unit_kind", 0)],
            MARKET_KIND_VALUES[prefix.get("market_kind", 0)],
            prefix,
        )
        return _boolean_mask(SLOT_WIDTHS[slot], allowed), allowed

    def append(self, frame):
        self.units += frame["unit_kind"] != 0
        self.orders += frame["market_kind"] != 0
        self.hires += frame["market_kind"] == 1
        self.stopped = bool(frame["stop"])


def frame_masks(
    legal_input, previous_frames, choices, *, layout=None, hire_limit=LEGACY_HIRE_LIMIT
):
    choices = _frame(choices)
    masks, prefix = {}, {}
    for slot in SLOT_NAMES:
        mask = mask_for_slot(
            slot,
            legal_input,
            previous_frames,
            prefix,
            layout=layout,
            hire_limit=hire_limit,
        )
        _require(mask[choices[slot]], f"target violates syntax mask: {slot}")
        masks[slot] = mask
        prefix[slot] = choices[slot]
    return masks


def _quantity(value):
    _require(
        type(value) is int and 0 <= value <= MAX_QUANTITY,
        "quantity must be an integer in 0..1023",
    )
    return divmod(value, QUANTITY_BASE)


def _token():
    return dict.fromkeys(SLOT_NAMES, 0)


def _unit_frame(command, actor):
    _require(
        isinstance(command, list) and bool(command) and type(command[0]) is str,
        "unit command must be a nonempty list",
    )
    kind = command[0]
    _require(kind in UNIT_KIND_VALUES[1:-1], "unknown unit command")
    expected = (2, 3) if kind in _TRANSFERS else ((2,) if kind == "PLANT" else (1,))
    _require(len(command) in expected, "unsupported unit command arguments")
    frame = _token()
    frame.update(unit_actor=actor, unit_kind=UNIT_KIND_VALUES.index(kind))
    if kind in _TRANSFERS or kind == "PLANT":
        _require(
            type(command[1]) is str and command[1] in ITEM_VALUES,
            "unsupported unit item",
        )
        frame["unit_item"] = ITEM_VALUES.index(command[1])
    if kind in _TRANSFERS and len(command) == 3:
        high, low = _quantity(command[2])
        frame.update(
            unit_quantity_present=1, unit_quantity_high=high, unit_quantity=low
        )
    return frame


def _market_frame(command):
    _require(isinstance(command, list), "market command must be a list")
    frame = _token()
    if not command:
        frame["market_kind"] = MARKET_KIND_VALUES.index("EMPTY")
        return frame
    kind = command[0]
    _require(
        type(kind) is str and kind in MARKET_KIND_VALUES[1:-1], "unknown market command"
    )
    _require(
        len(command) == (3 if kind in _QUANTITY_ORDERS else 1),
        "unsupported market arguments",
    )
    frame["market_kind"] = MARKET_KIND_VALUES.index(kind)
    if kind in _QUANTITY_ORDERS:
        _require(
            type(command[1]) is str and command[1] in ITEM_VALUES,
            "unsupported market item",
        )
        high, low = _quantity(command[2])
        frame.update(
            market_item=ITEM_VALUES.index(command[1]),
            market_quantity_high=high,
            market_quantity=low,
        )
    return frame


def encode_action(
    action: Mapping[str, Any],
    legal_input: Mapping[str, Any] | None = None,
    *,
    hire_limit: int = LEGACY_HIRE_LIMIT,
) -> dict[str, Any]:
    _require(
        isinstance(action, Mapping)
        and all(type(k) is str and k in _FIELDS for k in action),
        "unsupported action mapping",
    )
    hands = action.get("hands", [])
    markets = action.get("market", [])
    _require(
        isinstance(hands, list) and isinstance(markets, list),
        "hands and market must be lists",
    )
    layout = {"keys": list(action), "hands_count": len(hands)}
    _, limit = _layout(layout, legal_input)
    _require(len(markets) <= limit, "market queue exceeds configured limit")
    frames = []
    if "farmer" in action:
        frames.append(_unit_frame(action["farmer"], 0))
    frames.extend(
        _unit_frame(command, actor + 1) for actor, command in enumerate(hands)
    )
    frames.extend(_market_frame(command) for command in markets)
    final = _token()
    final["stop"] = 1
    frames.append(final)
    for index, frame in enumerate(frames):
        frame_masks(
            legal_input, frames[:index], frame, layout=layout, hire_limit=hire_limit
        )
    return {"schema": SCHEMA, "layout": layout, "frames": frames}


def decode_action(
    encoded: Mapping[str, Any],
    legal_input: Mapping[str, Any] | None = None,
    *,
    hire_limit: int = LEGACY_HIRE_LIMIT,
) -> dict[str, Any]:
    _require(
        isinstance(encoded, Mapping)
        and set(encoded) == {"schema", "layout", "frames"}
        and encoded["schema"] == SCHEMA,
        "encoded action schema differs",
    )
    layout, frames = encoded["layout"], encoded["frames"]
    _layout(layout, legal_input)
    _require(
        isinstance(frames, Sequence)
        and not isinstance(frames, (str, bytes))
        and bool(frames),
        "empty program",
    )
    previous: list[dict[str, int]] = []
    for raw in frames:
        frame = _frame(raw)
        frame_masks(legal_input, previous, frame, layout=layout, hire_limit=hire_limit)
        previous.append(frame)
    return _action_from_admitted_frames(layout, previous)


def _action_from_admitted_frames(layout, frames):
    """Render mask-admitted frames. Only the validator and private sampler call this."""
    _require(bool(frames), "empty program")
    _require(frames[-1]["stop"] == 1, "program lacks final STOP")
    units, market = {}, []
    for frame in frames:
        kind = UNIT_KIND_VALUES[frame["unit_kind"]]
        if kind != "NONE":
            command = [kind]
            if kind in _TRANSFERS or kind == "PLANT":
                command.append(ITEM_VALUES[frame["unit_item"]])
            if kind in _TRANSFERS and frame["unit_quantity_present"]:
                command.append(
                    32 * frame["unit_quantity_high"] + frame["unit_quantity"]
                )
            units[frame["unit_actor"]] = command
        kind = MARKET_KIND_VALUES[frame["market_kind"]]
        if kind != "NONE":
            command = [] if kind == "EMPTY" else [kind]
            if kind in _QUANTITY_ORDERS:
                command.extend(
                    (
                        ITEM_VALUES[frame["market_item"]],
                        32 * frame["market_quantity_high"] + frame["market_quantity"],
                    )
                )
            market.append(command)
    result = {}
    for key in layout["keys"]:
        if key == "farmer":
            result[key] = units[0]
        elif key == "hands":
            result[key] = [units[i + 1] for i in range(layout["hands_count"])]
        else:
            result[key] = market
    return result


def validate_action(
    action: Mapping[str, Any],
    legal_input: Mapping[str, Any] | None,
    *,
    hire_limit: int = LEGACY_HIRE_LIMIT,
) -> None:
    """Validate versioned representability and exact raw syntax; no mutation.

    Legal no-effects/partial fills are deliberately admitted. This does not
    certify an outcome, claim all official syntax is representable, or call an
    engine. HIRE support is explicitly bounded by the checkpoint-owned hire_limit.
    Explicit zero unit transfer quantities are outside frozen-validator support.
    The caller records the official engine's real result separately.
    """
    encoded = encode_action(action, legal_input, hire_limit=hire_limit)
    restored = decode_action(encoded, legal_input, hire_limit=hire_limit)
    _require(
        list(restored) == list(action) and restored == action, "codec altered action"
    )
