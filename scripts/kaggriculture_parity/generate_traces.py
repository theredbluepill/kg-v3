r"""Generate Kaggriculture parity traces from Kaggle's own Python engine.

The traces use the replay format of the four official fixtures in
``engine_rs/fixtures`` (``kaggriculture-re-parity-v1``): one header line, then one
``transition`` record per accepted step with the complete public state, both
private states, statuses and rewards. Generated traces add one optional record
type, ``rejected``: the Python interpreter raised on those actions, so Kaggle's
``env.step`` failed and left the state unchanged. The Rust replay must reject the
same actions without mutating state, then continue with the next record.

The generator must run against ``kaggle-environments==1.32.7`` whose
``envs/kaggriculture/kaggriculture.py`` matches the SHA-256 pinned in
``engine_rs/Cargo.toml``. The project lock pins an older package without this
environment, so run it in an isolated environment::

    UV_OFFLINE=1 uv run --isolated --no-project \
        --with kaggle-environments==1.32.7 \
        python scripts/kaggriculture_parity/generate_traces.py --preset committed \
        --out engine_rs/fixtures/generated
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import importlib.metadata
import io
import json
import random
import sys
import time
import tomllib
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
CARGO_TOML = REPO_ROOT / "engine_rs/Cargo.toml"
TRACE_FORMAT = "kaggriculture-re-parity-v1"
GENERATOR = "scripts/kaggriculture_parity/generate_traces.py"
ROLLS_PER_DAY = 200

MOVES = ["NORTH", "SOUTH", "EAST", "WEST"]
CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
ANIMALS = ["GOOSE", "COW", "SHEEP"]
PRODUCTS = [*CROPS, "EGG", "MILK", "WOOL", "FERTILIZER"]
SHED_ITEMS = [*PRODUCTS, *ANIMALS]
TILE_OPS = [
    "WATER",
    "HARVEST",
    "FERTILIZE",
    "BUILD_COOP",
    "BUILD_PASTURE",
    "DIG",
    "FEED",
    "COLLECT_FERTILIZER",
    "CARE",
]
UNIT_VERBS = [*MOVES, "PASS", "PICKUP", "PLACE", "DROP", "PLANT", *TILE_OPS]
MARKET_VERBS = ["BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL", "HIRE", "BUY_LAND"]
PASS_ACTION: dict[str, Any] = {"farmer": ["PASS"], "hands": [], "market": []}

Json = Any
Action = Json


class ParityGeneratorError(RuntimeError):
    """The generator cannot produce a trace that is valid parity evidence."""


# --------------------------------------------------------------------------- pins


@dataclass(frozen=True)
class EnginePin:
    version: str
    sha256: str


def pinned_engine(cargo_toml: Path = CARGO_TOML) -> EnginePin:
    """Read the compatibility target from the vendored crate's Cargo metadata."""
    metadata = tomllib.loads(cargo_toml.read_text(encoding="utf-8"))
    target = metadata["package"]["metadata"]["kaggriculture"]
    version = target["kaggle-environments-version"]
    digest = target["python-engine-sha256"]
    if not isinstance(version, str) or not isinstance(digest, str):
        raise ParityGeneratorError(f"{cargo_toml}: malformed kaggriculture metadata")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ParityGeneratorError(f"{cargo_toml}: python-engine-sha256 is not hex")
    return EnginePin(version=version, sha256=digest)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_engine(engine_path: Path, installed_version: str, pin: EnginePin) -> str:
    """Refuse to generate unless the installed engine is the pinned bytes."""
    if installed_version != pin.version:
        raise ParityGeneratorError(
            f"kaggle-environments {installed_version} is installed; the Rust kernel "
            f"targets {pin.version}. Run in an isolated env with "
            f"--with kaggle-environments=={pin.version}."
        )
    digest = sha256_file(engine_path)
    if digest != pin.sha256:
        raise ParityGeneratorError(
            f"{engine_path} has SHA-256 {digest}; engine_rs/Cargo.toml pins "
            f"{pin.sha256}. Refusing to generate parity traces from another engine."
        )
    return digest


def load_pinned_kaggle(pin: EnginePin) -> tuple[ModuleType, ModuleType, str]:
    """Import kaggle_environments and its kaggriculture module after the hash guard."""
    try:
        version = importlib.metadata.version("kaggle-environments")
    except importlib.metadata.PackageNotFoundError as error:
        raise ParityGeneratorError("kaggle-environments is not installed") from error
    if version != pin.version:
        raise ParityGeneratorError(
            f"kaggle-environments {version} is installed; the Rust kernel targets "
            f"{pin.version}. Run in an isolated env with "
            f"--with kaggle-environments=={pin.version}."
        )
    # Importing the package loads every bundled environment and prints noise.
    with _silenced():
        kaggle_environments = importlib.import_module("kaggle_environments")
    if kaggle_environments.__file__ is None:
        raise ParityGeneratorError("kaggle_environments has no module file")
    root = Path(kaggle_environments.__file__).resolve().parent
    engine_path = root / "envs/kaggriculture/kaggriculture.py"
    if not engine_path.is_file():
        raise ParityGeneratorError(
            f"kaggle-environments {version} has no kaggriculture environment"
        )
    digest = verify_engine(engine_path, version, pin)
    module = importlib.import_module(
        "kaggle_environments.envs.kaggriculture.kaggriculture"
    )
    if Path(module.__file__ or "").resolve() != engine_path:
        raise ParityGeneratorError(f"imported {module.__file__}, hashed {engine_path}")
    return kaggle_environments, module, digest


@contextmanager
def _silenced() -> Iterator[None]:
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        yield


# ----------------------------------------------------------------------- configs

CONFIG_VARIANTS: dict[str, dict[str, Any]] = {
    # Kaggle's defaults: the competition configuration.
    "default": {},
    # Free hires and 12 orders per turn: up to 288 hands per day, past the
    # model's 241-actor capacity (1 farmer + 240 hands).
    "free-hire": {
        "episodeSteps": 72,
        "farmHandCostMult": 0,
        "maxMarketOrdersPerTurn": 12,
    },
    # Enough money (at least 1,000,000) for the edge policy's 10**12-unit seed
    # orders, which hit Python's 100,000-iteration market-loop escape.
    "rich": {"episodeSteps": 96, "startingMoney": 10_000_000},
    # Minimal probes: a two-step preamble, one probe step, then passes.
    "probe": {"episodeSteps": 6},
    # Known-divergence repros: one transition, the divergent action.
    "divergence": {"episodeSteps": 2},
    # Non-default intervals, capacity, weeds and sparse market-curve overrides.
    "custom": {
        "episodeSteps": 200,
        "turnsPerDay": 7,
        "shedCapacity": 30,
        "weedSpawnChance": 0.2,
        "townShopUnlockInterval": 1,
        "townShopSellInterval": 3,
        "townCenterSellInterval": 5,
        "farmHandCostMult": 3,
        "maxMarketOrdersPerTurn": 4,
        "startingMoney": 5000,
        "marketParams": {
            "WHEAT": {"base": 30, "below_func": "log10", "T": 50},
            "MELON": {"above_func": "hinge", "above_target": 0.5},
            "FERTILIZER": {"I0": 40, "T": 10},
        },
    },
}


# ---------------------------------------------------------------------- policies


@dataclass
class Choice:
    """A policy decision: the submitted action plus a replacement if Python raises."""

    action: Action
    fallback: Action | None = None


Policy = Callable[[dict[str, Any], dict[str, Any], random.Random], Choice]


def _shed_adjacent(pos: list[int], board_size: int) -> bool:
    half = board_size // 2
    return (pos[0], pos[1]) in {
        (half - 1, half - 1),
        (half, half - 1),
        (half - 1, half),
        (half, half),
    }


def _qty(rng: random.Random) -> int:
    return rng.choice([1, 1, 1, 2, 2, 3, 4, 5, 8, 12, 30])


def _schema_unit(rng: random.Random) -> list[Any]:
    """Any unit verb from the action schema with a plausible argument."""
    verb = rng.choice(UNIT_VERBS)
    if verb in ("PICKUP", "PLACE"):
        cmd: list[Any] = [verb, rng.choice(SHED_ITEMS)]
        if rng.random() < 0.7:
            cmd.append(_qty(rng))
        return cmd
    if verb == "PLANT":
        return [verb, rng.choice(CROPS)]
    return [verb]


def _context_unit(
    farm: dict[str, Any],
    private: dict[str, Any],
    inventory: dict[str, Any],
    pos: list[int],
    board_size: int,
    rng: random.Random,
) -> list[Any]:
    """A unit command that is usually meaningful on the actor's current tile."""
    tile = farm["tiles"][pos[1]][pos[0]]
    options: list[list[Any]] = [[rng.choice(MOVES)] for _ in range(3)]
    if _shed_adjacent(pos, board_size):
        shed = private["shed"]
        stocked = [item for item in SHED_ITEMS if shed.get(item, 0) > 0]
        if stocked:
            options += [["PICKUP", rng.choice(stocked), _qty(rng)]] * 2
        if inventory:
            options += [["PLACE", rng.choice(sorted(inventory)), _qty(rng)], ["DROP"]]
    if tile is None:
        seeds = [crop for crop in CROPS if private["seeds"].get(crop, 0) > 0]
        if seeds:
            options += [["PLANT", rng.choice(seeds)]] * 3
        if rng.random() < 0.15:
            options.append([rng.choice(["BUILD_COOP", "BUILD_PASTURE"])])
    elif isinstance(tile, dict):
        kind = tile.get("kind")
        if kind == "PLANT":
            options += [["WATER"]] * 3 + [["HARVEST"]] * 2 + [["FERTILIZE"], ["DIG"]]
        elif kind == "WEED":
            options += [["DIG"]] * 2
        elif "animal" in tile:
            options += [["FEED"]] * 2 + [
                ["CARE"],
                ["HARVEST"],
                ["COLLECT_FERTILIZER"],
                ["DIG"],
            ]
        elif kind in ("COOP", "PASTURE"):
            animals = [a for a in ANIMALS if inventory.get(a, 0) > 0]
            if animals:
                options += [["PLACE", rng.choice(animals)]] * 3
            options.append(["DIG"])
    return rng.choice(options)


def _random_market(
    farm: dict[str, Any], private: dict[str, Any], max_orders: int, rng: random.Random
) -> list[list[Any]]:
    orders: list[list[Any]] = []
    count = rng.choice([0, 0, 1, 1, 2, 3, 4])
    if rng.random() < 0.03:
        count = max_orders + rng.randint(1, 3)
    shed = private["shed"]
    money = float(farm["money"])
    for _ in range(count):
        roll = rng.random()
        if roll < 0.25:
            orders.append(["BUY_SEED", rng.choice(CROPS), _qty(rng)])
        elif roll < 0.6:
            stocked = [p for p in PRODUCTS if shed.get(p, 0) > 0]
            item = rng.choice(stocked) if stocked else rng.choice(PRODUCTS)
            orders.append(["SELL", item, rng.choice([1, 2, 5, 20, 100])])
        elif roll < 0.68:
            orders.append(
                ["BUY_PRODUCT", rng.choice(["WHEAT", "FERTILIZER"]), _qty(rng)]
            )
        elif roll < 0.73 and money > 1500:
            orders.append(["BUY_ANIMAL", rng.choice(ANIMALS), rng.choice([1, 1, 2])])
        elif roll < 0.83 and money > 300:
            orders.append(["HIRE"])
        elif roll < 0.86 and money > 2000:
            orders.append(["BUY_LAND"])
        elif roll > 0.9:
            verb = rng.choice(MARKET_VERBS)
            if verb in ("HIRE", "BUY_LAND"):
                orders.append([verb])
            else:
                orders.append([verb, rng.choice(SHED_ITEMS), _qty(rng)])
    return orders


def random_policy(
    obs: dict[str, Any], config: dict[str, Any], rng: random.Random
) -> Choice:
    """Seeded legal-looking commands for every actor and several market slots."""
    player = obs["player"]
    farm = obs["farms"][player]
    private = obs["private"]
    board_size = int(config["boardSize"])
    actors = [farm["farmer"], *farm["hands"]]
    commands = []
    for index, pos in enumerate(actors):
        inventories = private["inventories"]
        inventory = inventories[index] if index < len(inventories) else {}
        if rng.random() < 0.25:
            commands.append(_schema_unit(rng))
        else:
            commands.append(
                _context_unit(farm, private, inventory, pos, board_size, rng)
            )
    market = _random_market(farm, private, int(config["maxMarketOrdersPerTurn"]), rng)
    return Choice({"farmer": commands[0], "hands": commands[1:], "market": market})


# Quantities whose Python ``int(...)`` behavior is part of the input contract.
ODD_QUANTITIES: list[Any] = [
    0,
    1023,
    -1,
    2.7,
    -0.5,
    "3",
    " 4 ",
    "1_0",
    "2.0",
    "x",
    True,
    False,
    None,
    [],
    {},
    1e3,
    10**12,
]
UNKNOWN_ORDERS: list[Any] = [
    ["FLY"],
    ["hire"],
    ["SELL_ALL", "WHEAT", 1],
    ["BUY_SEED", "GOLD", 1],
    ["BUY_PRODUCT", "MELON", 2],
    ["BUY_ANIMAL", "DRAGON", 1],
    ["SELL", "GOOSE", 1],
    ["SELL", "COW", 1],
    ["BUY_SEED", "EGG", 1],
]
MALFORMED_ORDERS: list[Any] = [
    [],
    "HIRE",
    5,
    None,
    {"op": "HIRE"},
    ["SELL"],
    ["SELL", "WHEAT"],
    [["SELL"], "WHEAT", 1],
    ["BUY_LAND", 3],
    ["HIRE", "HIRE"],
    [None, "WHEAT", 1],
    [7, 8, 9],
    ["SELL", 5, 1],
]
MALFORMED_UNITS: list[Any] = [
    "NORTH",
    [],
    [None],
    ["PLANT"],
    ["PLANT", "GOLD"],
    ["PLANT", 3],
    ["PICKUP"],
    ["PICKUP", "WHEAT", 0],
    ["PICKUP", "WHEAT", -3],
    ["PICKUP", "WHEAT", 1023],
    ["PICKUP", 5, 1],
    ["PICKUP", "WHEAT", 2.9],
    ["PICKUP", "WHEAT", "2"],
    ["PICKUP", "WHEAT", True],
    ["PLACE"],
    ["PLACE", "WHEAT", 1023],
    ["PLACE", "WHEAT", 0],
    ["PLACE", "COW"],
    ["PLACE", 7],
    ["TELEPORT"],
    ["plant", "WHEAT"],
    ["DROP", "WHEAT"],
    ["WATER", "extra", "args"],
    {"op": "NORTH"},
    3.5,
    None,
]
MALFORMED_ACTIONS: list[Any] = [
    [],
    "PASS",
    None,
    {},
    7,
    {"farmer": None, "hands": "x", "market": {"a": 1}},
    {"farmer": ["NORTH"], "hands": None, "market": None},
    {"farmer": "NORTH", "market": "HIRE"},
    {"hands": [["NORTH"]], "items": [1], "extra": True},
]
# Inputs on which Python's interpreter raises an uncaught int() error. Kaggle's
# env.step fails and keeps the old state; Rust returns an error (they agree).
CRASH_UNITS: list[Any] = [
    ["PICKUP", "WHEAT", "abc"],
    ["PICKUP", "WHEAT", None],
    ["PICKUP", "WHEAT", [1]],
    ["PLACE", "WHEAT", "abc"],
]

# Known divergences (docs/rules-parity-coverage.md, "Kaggriculture Live
# Differential Parity"). Full-game policies exclude these inputs by default so a
# game can test everything after them; --include-known-divergences restores them.
# D1: Python int() accepts any Unicode decimal digit string; Rust rejects it.
DIVERGENT_QUANTITIES: list[Any] = ["\u0663", "\uff13"]
# D2: Python raises TypeError on unhashable (array/object) verbs and items that
# reach a dict lookup; Rust treats them as no-ops and accepts the step.
DIVERGENT_CRASH_UNITS: list[Any] = [
    ["PLANT", ["WHEAT"]],
    ["PLANT", {"crop": "WHEAT"}],
    [["NORTH"]],
    [{"op": "NORTH"}],
    ["PICKUP", ["WHEAT"]],
    ["PLACE", {"crop": "WHEAT"}, 1],
]
DIVERGENT_CRASH_ORDERS: list[Any] = [
    ["BUY_SEED", ["WHEAT"], 1],
    ["BUY_ANIMAL", {"x": 1}, 1],
]
D1_REASON = (
    "D1: Python int() parses non-ASCII Unicode decimal digit strings "
    "(e.g. U+0663, U+FF13); the Rust kernel rejects them. Unit counts: Rust "
    "errors where Python acts; market quantities: Rust drops the order where "
    "Python executes it. The vendored kernel bytes are pinned, so this is "
    "recorded, not repaired."
)
D2_REASON = (
    "D2: Python raises TypeError when an unhashable JSON array/object reaches a "
    "dict lookup (unit verb, PLANT crop incl. missing hands, PICKUP/PLACE item, "
    "BUY_SEED/BUY_ANIMAL item), so Kaggle's env.step fails and keeps its state; "
    "the Rust kernel treats the command as a no-op and accepts the step. The "
    "vendored kernel bytes are pinned, so this is recorded, not repaired."
)
# (name, the first action of a one-step game, Rust divergence kind, field, reason).
# Minimal repros: no preamble, no trailing turns and only the divergent field.
KNOWN_DIVERGENCES: list[tuple[str, Any, str, str, str]] = [
    (
        "divergence-d1-unicode-digit-unit-count",
        {"farmer": ["PICKUP", "WHEAT", "\u0663"]},
        "rust_error",
        "step",
        D1_REASON,
    ),
    (
        "divergence-d1-unicode-digit-market-quantity",
        {"market": [["BUY_PRODUCT", "FERTILIZER", "\uff13"]]},
        "public state",
        "public.farms[0].money",
        D1_REASON,
    ),
    (
        "divergence-d2-unhashable-plant-crop",
        {"farmer": ["PLANT", ["WHEAT"]]},
        "rust_accepted",
        "step",
        D2_REASON,
    ),
    (
        "divergence-d2-unhashable-unit-verb",
        {"farmer": [["NORTH"]]},
        "rust_accepted",
        "step",
        D2_REASON,
    ),
    (
        "divergence-d2-unhashable-shed-item",
        {"farmer": ["PICKUP", {"crop": "WHEAT"}]},
        "rust_accepted",
        "step",
        D2_REASON,
    ),
    (
        "divergence-d2-unhashable-plant-in-missing-hand",
        {"hands": [["PLANT", ["WHEAT"]]]},
        "rust_accepted",
        "step",
        D2_REASON,
    ),
    (
        "divergence-d2-unhashable-market-item",
        {"market": [["BUY_SEED", ["WHEAT"], 1]]},
        "rust_accepted",
        "step",
        D2_REASON,
    ),
]


def make_edge_policy(include_known_divergences: bool = False) -> Policy:
    """Build the edge-case policy, optionally with the known-divergent inputs."""
    quantities = ODD_QUANTITIES + (
        DIVERGENT_QUANTITIES if include_known_divergences else []
    )
    crash_units = CRASH_UNITS + (
        DIVERGENT_CRASH_UNITS if include_known_divergences else []
    )
    crash_orders = DIVERGENT_CRASH_ORDERS if include_known_divergences else []

    def policy(
        obs: dict[str, Any], config: dict[str, Any], rng: random.Random
    ) -> Choice:
        return _edge_choice(obs, config, rng, quantities, crash_units, crash_orders)

    return policy


def _edge_choice(
    obs: dict[str, Any],
    config: dict[str, Any],
    rng: random.Random,
    quantities: list[Any],
    crash_units: list[Any],
    crash_orders: list[Any],
) -> Choice:
    """Deliberately awkward inputs layered over the random policy."""
    base = random_policy(obs, config, rng).action
    action: dict[str, Any] = copy.deepcopy(base)
    player = obs["player"]
    farm = obs["farms"][player]
    max_orders = int(config["maxMarketOrdersPerTurn"])
    market: list[Any] = action["market"]
    hands: list[Any] = action["hands"]

    if rng.random() < 0.1:
        action["market"] = market = []
    free_hires = int(config["farmHandCostMult"]) == 0
    if rng.random() < (0.95 if free_hires else 0.3):
        # HIRE until and past the actor cap / order limit.
        market[:0] = [["HIRE"]] * (max_orders + rng.choice([0, 0, 2]))
    if rng.random() < 0.35:
        for _ in range(rng.randint(1, 3)):
            verb = rng.choice(["BUY_SEED", "SELL", "BUY_PRODUCT", "BUY_ANIMAL"])
            item = {
                "BUY_SEED": CROPS,
                "SELL": PRODUCTS,
                "BUY_PRODUCT": ["WHEAT", "FERTILIZER"],
                "BUY_ANIMAL": ANIMALS,
            }[verb]
            market.insert(
                rng.randrange(len(market) + 1),
                [verb, rng.choice(item), rng.choice(quantities)],
            )
    if rng.random() < 0.15:
        market.insert(rng.randrange(len(market) + 1), [])
    if float(farm["money"]) >= 1_000_000 and rng.random() < 0.1:
        # 10**12-unit orders exhaust Python's 100,000-iteration market loop.
        market.insert(0, ["BUY_SEED", rng.choice(CROPS), 10**12])
    if rng.random() < 0.2 and market:
        # Duplicates and reversed (out-of-order) queues.
        market.append(copy.deepcopy(rng.choice(market)))
        if rng.random() < 0.5:
            market.reverse()
    if rng.random() < 0.1:
        market += [["BUY_LAND"]] * 4
    if rng.random() < 0.2:
        market.insert(rng.randrange(len(market) + 1), rng.choice(UNKNOWN_ORDERS))
    if rng.random() < 0.2:
        market.insert(rng.randrange(len(market) + 1), rng.choice(MALFORMED_ORDERS))
    if rng.random() < 0.2:
        slot = rng.randrange(len(hands) + 1)
        malformed = copy.deepcopy(rng.choice(MALFORMED_UNITS))
        if slot == 0:
            action["farmer"] = malformed
        else:
            hands[slot - 1] = malformed
    if rng.random() < 0.15:
        # Commands for actors that do not exist.
        hands += [_schema_unit(rng) for _ in range(rng.randint(1, 4))]
    if rng.random() < 0.1:
        # Oversubscribed PLANT: every actor plants one crop (atomic seed check).
        crop = rng.choice(CROPS)
        action["farmer"] = ["PLANT", crop]
        for index in range(len(hands)):
            hands[index] = ["PLANT", crop]
    if rng.random() < 0.08:
        action["hands"] = [] if rng.random() < 0.5 else hands[: len(farm["hands"]) // 2]

    if rng.random() < 0.03:
        return Choice(copy.deepcopy(rng.choice(MALFORMED_ACTIONS)))
    if rng.random() < 0.025:
        crashing = copy.deepcopy(action)
        if crash_orders and rng.random() < 0.25:
            crashing["market"] = [copy.deepcopy(rng.choice(crash_orders))]
        else:
            crashing["farmer"] = copy.deepcopy(rng.choice(crash_units))
        return Choice(crashing, fallback=action)
    return Choice(action)


def _builtin(name: str, module: ModuleType) -> Policy:
    agent = module.agents[name]

    def policy(
        obs: dict[str, Any], config: dict[str, Any], rng: random.Random
    ) -> Choice:
        del config
        if name == "random":
            # random_agent seeds itself from OS entropy; lend it the policy RNG
            # (only during the agent call; the interpreter's RNG is untouched).
            shim = SimpleNamespace(Random=lambda *_: rng)
            with mock.patch.object(module, "random", shim):
                return Choice(agent(obs))
        return Choice(agent(obs))

    return policy


def resolve_policy(
    name: str, module: ModuleType, include_known_divergences: bool = False
) -> Policy:
    if name == "random":
        return random_policy
    if name == "edge":
        return make_edge_policy(include_known_divergences)
    if name.startswith("builtin:"):
        builtin = name.removeprefix("builtin:")
        if builtin not in module.agents:
            raise ParityGeneratorError(
                f"unknown built-in agent {builtin!r}; have {sorted(module.agents)}"
            )
        return _builtin(builtin, module)
    raise ParityGeneratorError(f"unknown policy {name!r}")


# ------------------------------------------------------------------------ traces


@dataclass(frozen=True)
class GameSpec:
    name: str
    seed: int
    policies: tuple[str, str]
    variant: str
    policy_seed: int
    # Scripted probes: explicit per-step action pairs instead of policies.
    script: tuple[Any, ...] = ()
    probe: str = ""

    @property
    def configuration(self) -> dict[str, Any]:
        return {**copy.deepcopy(CONFIG_VARIANTS[self.variant]), "seed": self.seed}


@dataclass
class GameResult:
    spec: GameSpec
    records: list[dict[str, Any]]
    transitions: int
    rejected: int
    seconds: float
    rejected_errors: list[str] = field(default_factory=list)


def _plain(value: Any) -> Json:
    """Kaggle Struct trees to plain JSON data, preserving key order and types."""
    return json.loads(json.dumps(value, allow_nan=False))


def _public(env: Any) -> dict[str, Any]:
    obs = env.state[0].observation
    return {
        "step": obs.step,
        "day": obs.day,
        "hour": obs.hour,
        "farms": _plain(obs.farms),
        "market": _plain(obs.market),
        "town": _plain(obs.town),
    }


def _privates(env: Any) -> list[Any]:
    return [_plain(agent.observation.private) for agent in env.state]


def _observations(env: Any) -> list[dict[str, Any]]:
    public = env.state[0].observation
    return [
        {
            "player": index,
            "step": public.step,
            "day": public.day,
            "hour": public.hour,
            "farms": _plain(public.farms),
            "market": _plain(public.market),
            "town": _plain(public.town),
            "private": _plain(agent.observation.private),
        }
        for index, agent in enumerate(env.state)
    ]


def scripted_choices(spec: GameSpec, step: int) -> list[Choice]:
    """A scripted game's submitted pair at ``step``; both seats pass afterwards.

    Scripted actions are submitted exactly, including a null whole action.
    """
    pair = spec.script[step] if step < len(spec.script) else (PASS_ACTION,) * 2
    return [Choice(action, fallback=PASS_ACTION) for action in pair]


def play(
    spec: GameSpec,
    kaggle: ModuleType,
    module: ModuleType,
    pin: EnginePin,
    engine_sha256: str,
    include_known_divergences: bool = False,
) -> GameResult:
    """Run one seeded game in Kaggle's engine and record a parity trace."""
    started = time.perf_counter()
    policies = [
        resolve_policy(name, module, include_known_divergences)
        for name in ([] if spec.script else spec.policies)
    ]
    rngs = [random.Random(spec.policy_seed * 2 + seat) for seat in range(2)]
    with _silenced():
        env = kaggle.make("kaggriculture", configuration=spec.configuration)
    if env.info.get("seed") != spec.seed:
        raise ParityGeneratorError(f"{spec.name}: env did not adopt seed {spec.seed}")
    config = _plain(dict(env.configuration))
    turns_per_day = max(1, int(config["turnsPerDay"]))
    initial = {"public": _public(env), "privates": _privates(env)}
    records: list[dict[str, Any]] = []
    rejected_errors: list[str] = []
    end_days: list[int] = []
    transitions = 0
    while not env.done:
        step = env.state[0].observation.step
        observations = _observations(env)
        if spec.script:
            choices = scripted_choices(spec, step)
        else:
            choices = [
                policy(obs, config, rng)
                for policy, obs, rng in zip(policies, observations, rngs, strict=True)
            ]
        # Both engines see exactly the JSON bytes recorded in the trace.
        actions = _plain([choice.action for choice in choices])
        try:
            with _silenced():
                env.step(copy.deepcopy(actions))
        except Exception as error:
            if env.state[0].observation.step != step:
                raise ParityGeneratorError(
                    f"{spec.name}: failed step mutated the step counter"
                ) from error
            if all(choice.fallback is None for choice in choices):
                raise ParityGeneratorError(
                    f"{spec.name}: step {step} raised without a probe: {error!r}"
                ) from error
            message = f"{type(error).__name__}: {error}"
            rejected_errors.append(message)
            records.append(
                {
                    "type": "rejected",
                    "from_step": step,
                    "actions": actions,
                    "python_error": message,
                }
            )
            actions = _plain(
                [
                    choice.action if choice.fallback is None else choice.fallback
                    for choice in choices
                ]
            )
            with _silenced():
                env.step(copy.deepcopy(actions))
        if (step + 1) % turns_per_day == 0:
            end_days.append(step // turns_per_day)
        records.append(
            {
                "type": "transition",
                "from_step": step,
                "actions": actions,
                "expected": _public(env),
                "privates": _privates(env),
                "statuses": [agent.status for agent in env.state],
                "rewards": [agent.reward for agent in env.state],
            }
        )
        transitions += 1
    seed = env.info["seed"]
    header = {
        "type": "header",
        "format": TRACE_FORMAT,
        "source": {
            "generator": GENERATOR,
            "module_version": pin.version,
            "engine_sha256": engine_sha256,
            "name": spec.name,
            "policies": list(spec.policies),
            "policy_seed": spec.policy_seed,
            "config_variant": spec.variant,
            **({"probe": spec.probe} if spec.probe else {}),
        },
        "seed": seed,
        "configuration": dict(sorted(config.items())),
        "shop_schedule": list(env.state[0].observation.town["unlocked_shops"]),
        "rng_schedule": [
            {"day": day, "weed_rolls": _weed_rolls(seed, day)} for day in end_days
        ],
        "initial": initial,
        "terminal_banks": [float(agent.reward) for agent in env.state],
        "transitions": transitions,
    }
    return GameResult(
        spec=spec,
        records=[header, *records],
        transitions=transitions,
        rejected=len(rejected_errors),
        seconds=time.perf_counter() - started,
        rejected_errors=rejected_errors,
    )


def _weed_rolls(seed: int, day: int) -> list[float]:
    """The first rolls of Python's end-of-day RNG, as in the official traces."""
    rng = random.Random((seed * 1_000_003) ^ day)
    return [rng.random() for _ in range(ROLLS_PER_DAY)]


def encode_trace(records: list[dict[str, Any]]) -> bytes:
    """Deterministic gzip JSONL (fixed mtime, no filename) for stable hashes."""
    text = "".join(
        json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n"
        for record in records
    )
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as handle:
        handle.write(text.encode("utf-8"))
    return buffer.getvalue()


# ------------------------------------------------------------------------ presets

COMMITTED: list[tuple[str, int, tuple[str, str], str]] = [
    ("random-vs-random", 11, ("random", "random"), "default"),
    ("edge-vs-edge", 22, ("edge", "edge"), "default"),
    ("starter-vs-random", 33, ("builtin:starter", "builtin:random"), "default"),
    ("random-vs-edge", 44, ("random", "edge"), "default"),
    ("edge-free-hire", 55, ("edge", "edge"), "free-hire"),
    ("edge-vs-random-rich", 66, ("edge", "random"), "rich"),
    ("random-vs-starter-custom", 2**64 + 7, ("random", "builtin:starter"), "custom"),
    ("pass-vs-edge-negative-seed", -123_456, ("builtin:pass", "edge"), "default"),
]

SWEEP_ROTATION: list[tuple[tuple[str, str], str]] = [
    (("random", "random"), "default"),
    (("edge", "edge"), "default"),
    (("random", "edge"), "default"),
    (("edge", "random"), "default"),
    (("builtin:starter", "random"), "default"),
    (("edge", "builtin:starter"), "default"),
    (("builtin:random", "edge"), "default"),
    (("builtin:starter", "builtin:random"), "default"),
    (("edge", "edge"), "free-hire"),
    (("random", "edge"), "rich"),
    (("edge", "random"), "custom"),
    (("random", "builtin:pass"), "custom"),
]


def committed_specs() -> list[GameSpec]:
    games = [
        GameSpec(
            name=f"gen-{name}",
            seed=seed,
            policies=policies,
            variant=variant,
            policy_seed=index + 1,
        )
        for index, (name, seed, policies, variant) in enumerate(COMMITTED)
    ]
    return games + [
        _probe_spec(name, action, preamble=(), variant="divergence")
        for name, action, *_ in KNOWN_DIVERGENCES
    ]


def _probe_spec(
    name: str,
    action: Any,
    seed: int = 7,
    preamble: Sequence[Any] | None = None,
    variant: str = "probe",
) -> GameSpec:
    """Seat 0 plays ``preamble`` (default PROBE_PREAMBLE), then ``action``.

    Seat 1 passes throughout.
    """
    steps = [*(PROBE_PREAMBLE if preamble is None else preamble), action]
    return GameSpec(
        name=name,
        seed=seed,
        policies=("script", "builtin:pass"),
        variant=variant,
        policy_seed=0,
        script=tuple((act, PASS_ACTION) for act in steps),
        probe=json.dumps(action, separators=(",", ":")),
    )


def sweep_specs(games: int, base_seed: int) -> list[GameSpec]:
    specs = []
    for index in range(games):
        policies, variant = SWEEP_ROTATION[index % len(SWEEP_ROTATION)]
        seed = base_seed + index * 7919
        label = "-vs-".join(p.removeprefix("builtin:") for p in policies)
        specs.append(
            GameSpec(
                name=f"sweep-{index:03d}-{label}-{variant}",
                seed=seed,
                policies=policies,
                variant=variant,
                policy_seed=base_seed * 1_000 + index,
            )
        )
    return specs


# Quantity spellings whose Python int() result (or exception) is the contract.
QUANTITY_FORMS: list[Any] = [
    0,
    1,
    2,
    1023,
    -1,
    2.7,
    -0.5,
    1e3,
    1e30,
    10**12,
    10**30,
    -(10**30),
    "3",
    " 4 ",
    "\t2\n",
    "+2",
    "-2",
    "0002",
    "1_0",
    "1__0",
    "_1",
    "\u0663",
    "\uff13",  # FULLWIDTH DIGIT THREE
    "2.0",
    "1e3",
    "x",
    "",
    True,
    False,
    None,
    [],
    {},
    [2],
]
UNHASHABLE: list[Any] = [["WHEAT"], {"crop": "WHEAT"}]
PROBE_PREAMBLE: list[Any] = [
    {
        "farmer": ["PASS"],
        "hands": [],
        "market": [
            ["BUY_PRODUCT", "WHEAT", 5],
            ["BUY_ANIMAL", "GOOSE", 1],
            ["BUY_SEED", "CARROT", 2],
            ["HIRE"],
        ],
    },
    {"farmer": ["PICKUP", "WHEAT", 2], "hands": [["PICKUP", "GOOSE", 1]], "market": []},
]


def probe_cases() -> list[tuple[str, Any]]:
    """(label, seat-0 action at step 2) for every minimal input probe."""
    units: list[Any] = [*MALFORMED_UNITS, *CRASH_UNITS, *DIVERGENT_CRASH_UNITS]
    units += [["PICKUP", "WHEAT", q] for q in QUANTITY_FORMS]
    units += [["PLACE", "WHEAT", q] for q in QUANTITY_FORMS]
    for bad in UNHASHABLE:
        units += [["PICKUP", bad], ["PLACE", bad], ["PLACE", bad, 1], ["PLANT", bad]]
        units += [[bad], [bad, "WHEAT"]]
    units += [["PLANT", 3], ["PLANT", None], ["PLANT", True], ["PLANT", "CARROT"]]
    cases: list[tuple[str, Any]] = [
        ("unit", {"farmer": cmd, "hands": [["PASS"]], "market": []}) for cmd in units
    ]
    for cmd in [["PLANT", ["WHEAT"]], [["NORTH"]], ["PLACE", ["WHEAT"]], ["PASS"]]:
        cases.append(
            ("hand", {"farmer": ["PASS"], "hands": [["PASS"], cmd], "market": []})
        )
    orders: list[Any] = [*MALFORMED_ORDERS, *UNKNOWN_ORDERS, *DIVERGENT_CRASH_ORDERS]
    for verb, item in [
        ("BUY_SEED", "WHEAT"),
        ("SELL", "WHEAT"),
        ("BUY_PRODUCT", "FERTILIZER"),
        ("BUY_ANIMAL", "GOOSE"),
    ]:
        orders += [[verb, item, q] for q in QUANTITY_FORMS]
        orders += [[verb, bad, 1] for bad in UNHASHABLE]
        orders += [[verb, ["WHEAT"], 0], [verb, item]]
    cases += [
        ("market", {"farmer": ["PASS"], "hands": [], "market": [order]})
        for order in orders
    ]
    cases += [("action", action) for action in MALFORMED_ACTIONS]
    return cases


def probe_specs() -> list[GameSpec]:
    return [
        _probe_spec(f"probe-{index:03d}-{kind}", action)
        for index, (kind, action) in enumerate(probe_cases())
    ]


def known_divergence(name: str) -> dict[str, Any] | None:
    for known, _, kind, field_path, reason in KNOWN_DIVERGENCES:
        if known == name:
            # Line 1 is step 0: the repro's divergent action is its first step.
            return {
                "line": 1,
                "from_step": 0,
                "kind": kind,
                "field": field_path,
                "reason": reason,
            }
    return None


def write_manifest(out: Path, results: list[GameResult], pin: EnginePin) -> Path:
    entries = []
    for result in results:
        path = out / f"{result.spec.name}.jsonl.gz"
        entries.append(
            {
                "path": path.name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "seed": result.spec.seed,
                "policies": list(result.spec.policies),
                "policy_seed": result.spec.policy_seed,
                "config_variant": result.spec.variant,
                "transitions": result.transitions,
                "rejected": result.rejected,
                **(
                    {"probe": result.spec.probe, "expected_divergence": expected}
                    if (expected := known_divergence(result.spec.name))
                    else {}
                ),
            }
        )
    manifest = {
        "schema_version": 1,
        "format": TRACE_FORMAT,
        "generator": GENERATOR,
        "kaggle_environments_version": pin.version,
        "python_engine_sha256": pin.sha256,
        "traces": entries,
    }
    path = out / "MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--preset", choices=["committed", "sweep", "probes"], required=True
    )
    parser.add_argument("--games", type=int, default=40, help="sweep games")
    parser.add_argument("--base-seed", type=int, default=20260929, help="sweep seed")
    parser.add_argument("--manifest", action="store_true", help="write MANIFEST.json")
    parser.add_argument("--summary", type=Path, help="per-game JSON summary")
    parser.add_argument(
        "--include-known-divergences",
        action="store_true",
        help="let the edge policy emit the documented divergent inputs (D1/D2)",
    )
    args = parser.parse_args(argv)
    try:
        pin = pinned_engine()
        kaggle, module, digest = load_pinned_kaggle(pin)
    except ParityGeneratorError as error:
        print(f"refusing to generate: {error}", file=sys.stderr)
        return 2
    if args.preset == "committed":
        specs = committed_specs()
    elif args.preset == "sweep":
        specs = sweep_specs(args.games, args.base_seed)
    else:
        specs = probe_specs()
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for spec in specs:
        result = play(spec, kaggle, module, pin, digest, args.include_known_divergences)
        path = args.out / f"{spec.name}.jsonl.gz"
        path.write_bytes(encode_trace(result.records))
        results.append(result)
        print(
            f"{path.name}: {result.transitions} transitions, {result.rejected} "
            f"rejected, {path.stat().st_size:,} B, {result.seconds:.1f}s",
            flush=True,
        )
    if args.manifest:
        write_manifest(args.out, results, pin)
    if args.summary:
        summary = [
            {
                "path": f"{r.spec.name}.jsonl.gz",
                "seed": r.spec.seed,
                "policies": list(r.spec.policies),
                "policy_seed": r.spec.policy_seed,
                "config_variant": r.spec.variant,
                **({"probe": r.spec.probe} if r.spec.probe else {}),
                "transitions": r.transitions,
                "rejected": r.rejected,
                "rejected_errors": sorted(set(r.rejected_errors)),
                "seconds": round(r.seconds, 3),
            }
            for r in results
        ]
        args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
