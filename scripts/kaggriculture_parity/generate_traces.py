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
``engine_rs/Cargo.toml``. The project lock now pins that version too, so
``uv run python`` works; the sweep and the committed receipts use an isolated
environment, which does not depend on the project lock::

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
import importlib.util
import io
import json
import os
import random
import resource
import signal
import subprocess
import sys
import tempfile
import time
import tomllib
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
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


SIBLING_REPO = Path("/Users/poonszesen/kaggriculture")
SIBLING_COMMIT = "e8884aae82eddeb7a1aeae99ecceeca7c830d67e"
SIBLING_ENTRY_SHA256 = {
    "r04": "22d074391822206872448a6114a37ba2a4a39eda2ddbbcdc0bc8aa8cc6a64188",
    "ecobot": "0dc02e03c94ef60c06b5093efc2e2fd0530aa6eea20df507a90b90d6651bd067",
    "e776": "0cc2a88594f82b6c8d3cb15fcd4aa2df2bdd2a0a7fb0133d95a3204dd2d6ba38",
}
E776_MANIFEST_SHA256 = (
    "55dcc45b4c56324599d7832dfbc53ce6f1ce86aa44524fcb5a59a3c694857d58"
)
ORACLE_BYTE_BUDGET = 4_000_000
# Kaggle's simulation image (gcr.io/kaggle-images/python:v163, read directly)
# runs CPython 3.11.13. CPython 3.12 made float sum() compensated, which changes
# R04's anchor thresholds (ops/rebuild-2026-09-29/7.1/run2/r04-mismatch.md), so
# original-submission oracles are only generated on the competition's 3.11.
ORACLE_PYTHON = (3, 11)


def oracle_python_runtime(version_info: Sequence[int] | None = None) -> str:
    """The running interpreter's version, if it can produce opponent oracles."""
    version = tuple(sys.version_info if version_info is None else version_info)
    if version[:2] != ORACLE_PYTHON:
        raise ParityGeneratorError(
            "original opponent oracles require CPython 3.11 (the Kaggle "
            f"competition runtime); this is {'.'.join(map(str, version[:3]))}"
        )
    return ".".join(str(part) for part in version[:3])


def _sibling_blob(path: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(SIBLING_REPO), "show", f"{SIBLING_COMMIT}:{path}"],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise ParityGeneratorError(
            f"cannot read pinned sibling blob {path}: {result.stderr.decode()}"
        )
    return result.stdout


def _verified_sibling_blob(path: str, expected: str) -> bytes:
    blob = _sibling_blob(path)
    actual = hashlib.sha256(blob).hexdigest()
    if actual != expected:
        raise ParityGeneratorError(f"{path}: SHA-256 {actual}, expected {expected}")
    return blob


def sibling_oracle_files(bot: str) -> dict[str, bytes]:
    """Read verified original submission blobs; never consult the sibling worktree."""
    if bot not in SIBLING_ENTRY_SHA256:
        raise ParityGeneratorError(f"unknown sibling oracle {bot!r}")
    prefix = f"agents/{bot}/"
    files = {
        prefix + "main.py": _verified_sibling_blob(
            prefix + "main.py", SIBLING_ENTRY_SHA256[bot]
        )
    }
    if bot == "e776":
        manifest = _verified_sibling_blob(
            prefix + "MANIFEST.sha256", E776_MANIFEST_SHA256
        )
        files[prefix + "MANIFEST.sha256"] = manifest
        seen = set()
        for line in manifest.decode().splitlines():
            digest, name = line.split("  ", 1)
            path = Path(name)
            if (
                path.is_absolute()
                or any(part in {".", ".."} for part in name.split("/"))
                or name in seen
            ):
                raise ParityGeneratorError(
                    f"unsafe or duplicate E776 manifest path {name!r}"
                )
            seen.add(name)
            files[prefix + name] = _verified_sibling_blob(prefix + name, digest)
    return files


def sibling_oracle_metadata(bot: str) -> dict[str, Any]:
    files = sibling_oracle_files(bot)
    return {
        "bot": bot,
        "source_repo": str(SIBLING_REPO),
        "source_commit": SIBLING_COMMIT,
        "files": [
            {"path": path, "sha256": hashlib.sha256(blob).hexdigest()}
            for path, blob in sorted(files.items())
        ],
        "provenance": (
            "Read-only pinned original submission. EcoBot and E776 declare no "
            "software license; local research oracle only, do not redistribute. "
            "R04 has no agent PROVENANCE.md."
        ),
    }


class SiblingPolicy:
    """One isolated original Python submission per seat and episode.

    E776 imports siblings by fixed names and resolves data relative to __file__.
    Its verified closure therefore lives temporarily outside the repository. Each
    invocation swaps only this closure's module names, then restores sys.path and
    the KG_* environment variables used by R04. This synchronous generator never
    invokes two policies concurrently. No Python submission is redistributed.
    """

    def __init__(self, bot: str):
        self.bot = bot
        files = sibling_oracle_files(bot)
        self._directory = tempfile.TemporaryDirectory(
            prefix="kagg-oracle-", dir="/private/tmp"
        )
        self.root = Path(self._directory.name)
        self.name = "_kagg_oracle_" + uuid.uuid4().hex
        self.modules: dict[str, ModuleType] = {}
        self.module_names = {
            self.name,
            "e776_pkg",
            "e776_pkg.entry",
            "e776_packaged_policy",
            "e749a_attributed_niklita_trace",
            "e766a_kenjo_medoid_source",
        }
        prefix = f"agents/{bot}/"
        try:
            for path, blob in files.items():
                relative = Path(path.removeprefix(prefix))
                destination = self.root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(blob)
                if relative.parent == Path("agents") and relative.suffix == ".py":
                    self.module_names.add(relative.stem)
            with self._scope():
                spec = importlib.util.spec_from_file_location(
                    self.name, self.root / "main.py"
                )
                if spec is None or spec.loader is None:
                    raise ParityGeneratorError(f"cannot load {bot} entry")
                module = importlib.util.module_from_spec(spec)
                sys.modules[self.name] = module
                spec.loader.exec_module(module)
                self.agent = (
                    module.kaggriculture_e776_agent if bot == "e776" else module.agent
                )
        except BaseException:
            self.close()
            raise

    @contextmanager
    def _scope(self) -> Iterator[None]:
        previous = {
            name: sys.modules[name] for name in self.module_names if name in sys.modules
        }
        previous_path = list(sys.path)
        previous_kg = {
            key: value for key, value in os.environ.items() if key.startswith("KG_")
        }
        previous_bytecode = sys.dont_write_bytecode
        for name in self.module_names:
            sys.modules.pop(name, None)
        sys.modules.update(self.modules)
        sys.path.insert(0, str(self.root))
        sys.dont_write_bytecode = True
        try:
            yield
        finally:
            self.modules = {
                name: sys.modules[name]
                for name in self.module_names
                if name in sys.modules
            }
            for name in self.module_names:
                sys.modules.pop(name, None)
            sys.modules.update(previous)
            sys.path[:] = previous_path
            for key in list(os.environ):
                if key.startswith("KG_"):
                    del os.environ[key]
            os.environ.update(previous_kg)
            sys.dont_write_bytecode = previous_bytecode

    def __call__(
        self, obs: dict[str, Any], config: dict[str, Any], rng: random.Random
    ) -> Choice:
        del rng
        with self._scope():
            action = (
                self.agent(obs) if self.bot == "ecobot" else self.agent(obs, config)
            )
        return Choice(action)

    def close(self) -> None:
        self.modules.clear()
        self._directory.cleanup()


def resolve_policy(
    name: str, module: ModuleType, include_known_divergences: bool = False
) -> Policy:
    if name.startswith("sibling:"):
        return SiblingPolicy(name.removeprefix("sibling:"))
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
    peak_rss_bytes: int = 0


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
    """Run one seeded game; original oracle modules never outlive the game."""
    with ExitStack() as stack:
        policies = []
        for name in [] if spec.script else spec.policies:
            policy = resolve_policy(name, module, include_known_divergences)
            if isinstance(policy, SiblingPolicy):
                stack.callback(policy.close)
            policies.append(policy)
        return _play(spec, kaggle, pin, engine_sha256, policies)


def _play(
    spec: GameSpec,
    kaggle: ModuleType,
    pin: EnginePin,
    engine_sha256: str,
    policies: list[Policy],
) -> GameResult:
    started = time.perf_counter()
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
        if spec.name.startswith("oracle-") and peak_rss_bytes() >= 1_000_000_000:
            raise ParityGeneratorError("opponent game reached 1 GB Mac memory bound")
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
        peak_rss_bytes=peak_rss_bytes(),
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


def _manifest_data(
    out: Path, results: list[GameResult], pin: EnginePin
) -> dict[str, Any]:
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
    return {
        "schema_version": 1,
        "format": TRACE_FORMAT,
        "generator": GENERATOR,
        "kaggle_environments_version": pin.version,
        "python_engine_sha256": pin.sha256,
        "traces": entries,
    }


def write_manifest(out: Path, results: list[GameResult], pin: EnginePin) -> Path:
    manifest = _manifest_data(out, results, pin)
    path = out / "MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


OPPONENT_PAIRS = (
    ("builtin:starter", "sibling:r04"),
    ("sibling:r04", "sibling:ecobot"),
    ("sibling:ecobot", "sibling:e776"),
    ("sibling:e776", "builtin:starter"),
)


def opponent_specs(start: int, games: int, base_seed: int) -> list[GameSpec]:
    """A bounded slice; four consecutive indices cover every bot in both seats."""
    if games not in (1, 2) or start < 0:
        raise ParityGeneratorError(
            "opponents requires one or two games and a nonnegative start"
        )
    return [
        GameSpec(
            name=f"oracle-{index:02d}-"
            + "-vs-".join(p.split(":")[1] for p in OPPONENT_PAIRS[index % 4]),
            seed=base_seed + index,
            policies=OPPONENT_PAIRS[index % 4],
            variant="default",
            policy_seed=base_seed + index,
        )
        for index in range(start, start + games)
    ]


def opponent_coverage(records: list[dict[str, Any]]) -> list[dict[str, int]]:
    """Count observable events; rejected_steps means whole Python-step rejection.

    BUY_PRODUCT quantities above the signed inventory index are a diagnostic
    proxy only: this index is not physical stock, so neither actual shortages
    nor individual-order rejection is measured. Hires count positive changes
    in hands. A day reset is an action at hour zero after step zero. Final-day
    sell orders count intent, not proceeds. Contiguous traces never replay;
    mid-episode replay is frozen separately (``opponent_replay``).
    """
    counts = [
        dict.fromkeys(
            (
                "openings",
                "day_resets",
                "weed_presence",
                "buy_quantity_above_inventory_index",
                "rejected_steps",
                "hires",
                "final_day_sell_orders",
                "mid_episode_replay",
            ),
            0,
        )
        for _ in range(2)
    ]
    public = records[0]["initial"]["public"]
    for record in records[1:]:
        for seat, action in enumerate(record["actions"]):
            count = counts[seat]
            farm = public["farms"][seat]
            if record["type"] == "rejected":
                count["rejected_steps"] += 1
                continue
            count["openings"] += int(record["from_step"] == 0)
            count["day_resets"] += int(record["from_step"] > 0 and public["hour"] == 0)
            count["weed_presence"] += int(
                any(
                    isinstance(tile, dict) and tile["kind"] == "WEED"
                    for row in farm["tiles"]
                    for tile in row
                )
            )
            for order in action.get("market", []):
                if order[0] == "BUY_PRODUCT" and int(order[2]) > public["market"][
                    "inventory"
                ].get(order[1], 0):
                    count["buy_quantity_above_inventory_index"] += 1
                count["final_day_sell_orders"] += int(
                    public["day"] == 29 and order[0] == "SELL"
                )
            count["hires"] += max(
                0, len(record["expected"]["farms"][seat]["hands"]) - len(farm["hands"])
            )
        if record["type"] == "transition":
            public = record["expected"]
    return counts


def peak_rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)


class _OpponentDeadline(BaseException):
    """Original agents may swallow Exception; the Mac bound must still interrupt."""


@contextmanager
def opponent_deadline(seconds: int = 120) -> Iterator[None]:
    """Interrupt even a stuck original policy; this CLI runs on the main thread."""

    def expired(signum: int, frame: Any) -> None:
        del signum, frame
        raise _OpponentDeadline

    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(seconds)
    try:
        yield
    except _OpponentDeadline as error:
        raise ParityGeneratorError(
            f"opponent game exceeded {seconds}s Mac bound"
        ) from error
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def write_opponent_manifest(
    out: Path, results: list[GameResult], pin: EnginePin
) -> Path:
    """Merge bounded batches while rechecking previous fixture custody."""
    manifest_path = out / "MANIFEST.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    existing = (
        {}
        if previous is None
        else {entry["path"]: entry for entry in previous["traces"]}
    )
    for entry in existing.values():
        path = out / entry["path"]
        if (
            path.parent != out
            or sha256_file(path) != entry["sha256"]
            or path.stat().st_size != entry["bytes"]
        ):
            raise ParityGeneratorError(f"existing oracle custody mismatch: {path}")
    # Build the existing trace schema, then add explicit availability and events.
    manifest = _manifest_data(out, results, pin)
    runtime = oracle_python_runtime()
    for entry, result in zip(manifest["traces"], results, strict=True):
        entry["python_runtime"] = runtime
        entry["available_actions"] = [result.transitions, result.transitions]
        entry["coverage"] = opponent_coverage(result.records)
        existing[entry["path"]] = entry
    manifest["traces"] = [existing[name] for name in sorted(existing)]
    manifest["byte_budget"] = ORACLE_BYTE_BUDGET
    if sum(entry["bytes"] for entry in existing.values()) > ORACLE_BYTE_BUDGET:
        raise ParityGeneratorError("opponent trace byte budget exceeded")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


# Mid-episode replay: fresh controllers rebuilt from a recorded prefix, then
# resumed. Points are mid-day (day 1 hour 13), a day reset (day 15 hour 0) and
# the last pre-liquidation hour, whose window covers the whole final day.
REPLAY_FORMAT = "kaggriculture-opponent-replay-v1"
REPLAY_POINTS = (37, 360, 695)
REPLAY_RESUME_STEPS = 24
REPLAY_FILE = "REPLAY.json.gz"
REPLAY_BYTE_BUDGET = 1_000_000
ORACLE_DIR = REPO_ROOT / "opponents_rs/fixtures/oracle"


def replay_window(point: int, resume: int, transitions: int) -> None:
    """A replay must rebuild a nonempty prefix and resume inside the episode."""
    if point <= 0 or resume <= 0 or point + resume > transitions:
        raise ParityGeneratorError(
            f"replay window {point}+{resume} lies outside {transitions} transitions"
        )


def _canonical(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


def replay_case(
    records: list[dict[str, Any]],
    point: int,
    resume: int,
    kaggle: ModuleType,
    module: ModuleType,
) -> dict[str, Any]:
    """Rebuild fresh controllers from a recorded prefix, then resume them.

    This mirrors the native lifecycle, which admits a fresh controller only at
    step zero: a new controller per seat (a fresh module for original
    submissions) observes every prefix step and must choose the recorded action,
    which drives Kaggle's engine; the rebuilt state must equal the recorded state.
    From ``point`` the controllers act on their own for ``resume`` steps. Their
    actions and the final state are the frozen expectations for native replay.
    """
    header, rows = records[0], records[1:]
    source = header["source"]
    replay_window(point, resume, int(header["transitions"]))
    if source["config_variant"] != "default":
        raise ParityGeneratorError("opponent replay supports the default config only")
    spec = GameSpec(
        name=str(source["name"]),
        seed=int(header["seed"]),
        policies=(source["policies"][0], source["policies"][1]),
        variant="default",
        policy_seed=int(source["policy_seed"]),
    )
    resumed: list[Json] = []
    with ExitStack() as stack:
        policies = []
        for name in spec.policies:
            policy = resolve_policy(name, module)
            if isinstance(policy, SiblingPolicy):
                stack.callback(policy.close)
            policies.append(policy)
        rngs = [random.Random(spec.policy_seed * 2 + seat) for seat in range(2)]
        with _silenced():
            env = kaggle.make("kaggriculture", configuration=spec.configuration)
        if env.info.get("seed") != spec.seed:
            raise ParityGeneratorError(f"{spec.name}: env did not adopt seed")
        config = _plain(dict(env.configuration))
        if _public(env) != header["initial"]["public"]:
            raise ParityGeneratorError(f"{spec.name}: reconstruction initial state")
        for row in rows[: point + resume]:
            step = env.state[0].observation.step
            if row["type"] != "transition" or row["from_step"] != step:
                raise ParityGeneratorError(f"{spec.name}: record for step {step}")
            actions = _plain(
                [
                    policy(obs, config, rng).action
                    for policy, obs, rng in zip(
                        policies, _observations(env), rngs, strict=True
                    )
                ]
            )
            if step < point:
                for seat in range(2):
                    if _canonical(actions[seat]) != _canonical(row["actions"][seat]):
                        raise ParityGeneratorError(
                            f"{spec.name}: prefix step {step} seat {seat}: fresh "
                            "controller diverged from the recorded action"
                        )
                submitted = row["actions"]
            else:
                resumed.append(actions)
                submitted = actions
            with _silenced():
                env.step(copy.deepcopy(submitted))
            if step < point and (
                _public(env) != row["expected"] or _privates(env) != row["privates"]
            ):
                raise ParityGeneratorError(
                    f"{spec.name}: reconstruction step {step} diverged from the "
                    "recorded state"
                )
        final = {
            "public": _public(env),
            "privates": _privates(env),
            "statuses": [agent.status for agent in env.state],
        }
    return {
        "source": f"{spec.name}.jsonl.gz",
        "seed": spec.seed,
        "policies": list(spec.policies),
        "reconstruct_step": point,
        "resume_steps": resume,
        "resumed_actions": resumed,
        # Descriptive only: equal actions show deterministic reconstruction; the
        # native test compares against resumed_actions, never these counts.
        "continuous_equal_actions": [
            sum(
                _canonical(actions[seat])
                == _canonical(rows[point + i]["actions"][seat])
                for i, actions in enumerate(resumed)
            )
            for seat in range(2)
        ],
        "final": final,
    }


def oracle_sources(oracle_dir: Path) -> tuple[bytes, list[list[dict[str, Any]]]]:
    """Read the frozen opponent oracles, rechecking their manifest custody."""
    manifest_bytes = (oracle_dir / "MANIFEST.json").read_bytes()
    traces = []
    for entry in json.loads(manifest_bytes)["traces"]:
        path = oracle_dir / entry["path"]
        data = path.read_bytes()
        if (
            path.parent != oracle_dir
            or hashlib.sha256(data).hexdigest() != entry["sha256"]
            or len(data) != entry["bytes"]
        ):
            raise ParityGeneratorError(f"oracle custody mismatch: {path}")
        traces.append(
            [json.loads(line) for line in gzip.decompress(data).decode().splitlines()]
        )
    return manifest_bytes, traces


def write_replay_fixture(out: Path, document: dict[str, Any]) -> Path:
    """Write deterministic gzip JSON; a frozen fixture is never overwritten."""
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as handle:
        handle.write((_canonical(document) + "\n").encode("utf-8"))
    encoded = buffer.getvalue()
    path = out / REPLAY_FILE
    if path.exists() and path.read_bytes() != encoded:
        raise ParityGeneratorError(f"refusing to overwrite frozen replay {path}")
    if len(encoded) > REPLAY_BYTE_BUDGET:
        raise ParityGeneratorError(
            f"replay fixture {len(encoded):,} B exceeds {REPLAY_BYTE_BUDGET:,} B"
        )
    out.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return path


def opponent_replay(
    oracle_dir: Path,
    kaggle: ModuleType,
    module: ModuleType,
    pin: EnginePin,
    engine_sha256: str,
) -> dict[str, Any]:
    """Replay every frozen oracle at every point; each bot resumes in both seats."""
    runtime = oracle_python_runtime()
    manifest_bytes, traces = oracle_sources(oracle_dir)
    cases = []
    for records in traces:
        for point in REPLAY_POINTS:
            with opponent_deadline():
                cases.append(
                    replay_case(records, point, REPLAY_RESUME_STEPS, kaggle, module)
                )
            if peak_rss_bytes() >= 1_000_000_000:
                raise ParityGeneratorError("opponent replay reached 1 GB Mac bound")
    return {
        "type": "header",
        "format": REPLAY_FORMAT,
        "generator": GENERATOR,
        "module_version": pin.version,
        "engine_sha256": engine_sha256,
        "python_runtime": runtime,
        "oracle_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "points": list(REPLAY_POINTS),
        "resume_steps": REPLAY_RESUME_STEPS,
        "cases": cases,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--preset",
        choices=["committed", "sweep", "probes", "opponents", "opponent-replay"],
        required=True,
    )
    parser.add_argument(
        "--games", type=int, help="sweep games (40) or opponent games (1; maximum 2)"
    )
    parser.add_argument(
        "--opponent-start", type=int, default=0, help="first opponent-pair index"
    )
    parser.add_argument(
        "--oracle-dir",
        type=Path,
        default=ORACLE_DIR,
        help="frozen opponent oracles that opponent-replay rebuilds prefixes from",
    )
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
        if args.preset in ("opponents", "opponent-replay"):
            oracle_python_runtime()
        pin = pinned_engine()
        kaggle, module, digest = load_pinned_kaggle(pin)
    except ParityGeneratorError as error:
        print(f"refusing to generate: {error}", file=sys.stderr)
        return 2
    if args.preset == "opponent-replay":
        started = time.perf_counter()
        document = opponent_replay(args.oracle_dir, kaggle, module, pin, digest)
        path = write_replay_fixture(args.out, document)
        replay_summary = {
            "path": path.name,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "cases": len(document["cases"]),
            "seconds": round(time.perf_counter() - started, 3),
            "peak_rss_bytes": peak_rss_bytes(),
            "python_version": sys.version,
        }
        print(json.dumps(replay_summary), flush=True)
        if args.summary:
            args.summary.write_text(json.dumps(replay_summary, indent=2) + "\n")
        return 0
    if args.preset == "committed":
        specs = committed_specs()
    elif args.preset == "sweep":
        specs = sweep_specs(
            args.games if args.games is not None else 40, args.base_seed
        )
    elif args.preset == "opponents":
        try:
            specs = opponent_specs(
                args.opponent_start,
                args.games if args.games is not None else 1,
                args.base_seed,
            )
        except ParityGeneratorError as error:
            print(f"refusing to generate: {error}", file=sys.stderr)
            return 2
    else:
        specs = probe_specs()
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for spec in specs:
        with ExitStack() as stack:
            if args.preset == "opponents":
                stack.enter_context(opponent_deadline())
            result = play(
                spec, kaggle, module, pin, digest, args.include_known_divergences
            )
        path = args.out / f"{spec.name}.jsonl.gz"
        encoded = encode_trace(result.records)
        if args.preset == "opponents":
            total = sum(
                p.stat().st_size for p in args.out.glob("*.jsonl.gz") if p != path
            )
            if total + len(encoded) > ORACLE_BYTE_BUDGET:
                raise ParityGeneratorError(
                    "opponent trace byte budget exceeded before writing"
                )
        if (
            args.preset == "opponents"
            and path.exists()
            and path.read_bytes() != encoded
        ):
            raise ParityGeneratorError(f"refusing to overwrite frozen oracle {path}")
        path.write_bytes(encoded)
        results.append(result)
        print(
            f"{path.name}: {result.transitions} transitions, {result.rejected} "
            f"rejected, {path.stat().st_size:,} B, {result.seconds:.1f}s",
            flush=True,
        )
    if args.preset == "opponents":
        write_opponent_manifest(args.out, results, pin)
    elif args.manifest:
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
                "peak_rss_bytes": r.peak_rss_bytes,
                "python_version": sys.version,
            }
            for r in results
        ]
        args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
