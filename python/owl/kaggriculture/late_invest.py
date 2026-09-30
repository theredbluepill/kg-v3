"""Rule 2: drop market purchases that physically cannot pay back before the end.

The final reward is cash (``kaggriculture.py:963``), and cash only rises through
``SELL`` of a shed item (``kaggriculture.py:653-661``). A purchase is dropped
only when no play at all could turn it into a sale by the last processed
action, so the filter can remove waste but never a purchase that could still
earn. Every threshold is a lower bound on the earliest sale step, so any
configuration can only under-block. At the default configuration each last
allowed purchase is still sold by a scripted play in the Kaggle engine
(``tests/kaggriculture/test_late_invest.py``), so there the thresholds are
exact; at ``episodeSteps`` 699 the seed and animal plays sell on the last
action itself. The filter is a pure function of the current observation's step
and the Kaggle configuration; it keeps no state.

Kaggle source is ``kaggle_environments/envs/kaggriculture/kaggriculture.py``
from ``kaggle-environments==1.32.7`` (``pyproject.toml``); the Rust mirror is
``engine_rs/src/lib.rs``.

Step structure (``T = turnsPerDay``, ``E = episodeSteps``):

==========================  ==============================  =====================
Fact                        kaggriculture.py                engine_rs/src/lib.rs
==========================  ==============================  =====================
Last processed action is    960 (``step >= E - 2``: DONE)   1487
step ``S = E - 2``
Unit actions run before     935-941                         1458, 1464
the market in one step
End of day runs after the   945-946, 878-882                1476, 4498, 4505
market when
``(step + 1) % T == 0``;
it drops inventories to
the shed and clears hands
Sales come only from the    596, 653-661                    3873, 4048
shed; FERTILIZER is a
product                     25                              25-35
Items reach the shed by     343-410, 843-857                3160-3251
DROP/PLACE (one action per
unit per step) or the end
of day, so a sale follows
its harvest by >= 1 step
==========================  ==============================  =====================

Blocked purchases, with ``day(s) = s // T`` and ``ready(d) = d * T + 1 <= S``
("a product first collectible at hour 0 of day ``d`` can still be sold"):

=========================  =============================  ==========================
Order                      Dropped when                   Why (Kaggle / lib.rs)
=========================  =============================  ==========================
``BUY_SEED c n``           not ready(day(t + 1)           Seeds credit in the market
                           + first_yield_day[c])          phase (673-678 / 3908);
                                                          PLANT is a later unit
                                                          action (417-429 / 3257);
                                                          HARVEST needs age >=
                                                          first_yield_day (453 /
                                                          3320); ongoing crops first
                                                          yield at that day's start
                                                          (789-800 / 4233).
``BUY_ANIMAL a n``         not ready(day(t + 2) + 1)      The animal lands in the
                                                          shed (679-686 / 3916);
                                                          PICKUP then PLACE by one
                                                          unit take two later steps
                                                          (358-392 / 3183-3221). The
                                                          first sellable product is
                                                          FERTILIZER, set at every
                                                          end of day (831 / 4309);
                                                          EGG/MILK/WOOL come later
                                                          (822-828 / 4292).
``BUY_LAND``               not ready(day(t + 1) + 1)      Tiles unlock in the market
                                                          phase (712-725 / 3838), so
                                                          the first use is step
                                                          ``t + 1``; the fastest use
                                                          is an animal placed that
                                                          day, whose FERTILIZER comes
                                                          after that end of day (a
                                                          crop needs >= 2 days).
``HIRE``                   ``t % T == T - 1`` or          The hand first acts at
                           ``t + 2 > S``                  ``t + 1`` (702-709 /
                                                          3813) and is removed at
                                                          the end of day ``day(t)``
                                                          (880 / 4505), so an
                                                          end-of-day hire never
                                                          acts; a first action at
                                                          ``S`` cannot reach the
                                                          shed before the last
                                                          market. The engine's own
                                                          wasted-hire counter uses
                                                          the end-of-day half
                                                          (lib.rs 4015-4018).
=========================  =============================  ==========================

Never dropped: ``SELL``; ``BUY_PRODUCT`` (WHEAT is animal feed, FERTILIZER is
an input for existing plants); every unit action (feeding, watering, harvesting
and planting held seeds cost nothing); orders the engine ignores anyway. A
dropped order becomes the canonical empty market command ``[]``, which the
engine skips (``_parse_order``, 631-633 / 3442-3448), so the other orders keep
their lockstep slots.

At the default ``E = 720, T = 24`` (``S = 718``, day 29 hour 22) the first
dropped step is: WHEAT/CARROT 671 (day 27 hour 23), TOMATO 527 (day 21 hour
23), STRAWBERRY/MELON 479 (day 19 hour 23), animals 694 (day 28 hour 22), land
695 (day 28 hour 23), hires 717 (day 29 hour 21) plus hour 23 of every day.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from owl.kaggriculture.types import JsonValue

# kaggriculture.py:11-17 / engine_rs/src/lib.rs:68-112 (first_yield_day).
CROP_FIRST_YIELD_DAY: Mapping[str, int] = {
    "WHEAT": 2,
    "CARROT": 2,
    "TOMATO": 8,
    "STRAWBERRY": 10,
    "MELON": 10,
}
# kaggriculture.py:19-23 / engine_rs/src/lib.rs:124-152.
ANIMALS: frozenset[str] = frozenset({"GOOSE", "COW", "SHEEP"})
# An animal alive at the end of its placement day sets fertilizer_available
# (kaggriculture.py:831 / engine_rs/src/lib.rs:4309): one day after placement.
FERTILIZER_READY_DAYS = 1
# kaggriculture.py:960 / engine_rs/src/lib.rs:1487: the interpreter marks DONE
# after processing step episodeSteps - 2, so that is the last processed action.
LAST_ACTION_OFFSET = 2


@dataclass(frozen=True)
class GameClock:
    """Turns per day and the last processed action step of one game."""

    turns_per_day: int
    last_action_step: int

    @classmethod
    def of(cls, configuration: Mapping[str, Any]) -> GameClock:
        turns_per_day = configuration["turnsPerDay"]
        episode_steps = configuration["episodeSteps"]
        for name, value in (
            ("turnsPerDay", turns_per_day),
            ("episodeSteps", episode_steps),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        return cls(
            turns_per_day=turns_per_day,
            last_action_step=episode_steps - LAST_ACTION_OFFSET,
        )

    def day(self, step: int) -> int:
        return step // self.turns_per_day

    def sellable_from_day(self, day: int) -> bool:
        """A product first collectible at hour 0 of ``day`` can still be sold.

        Collection is a unit action; the item reaches the shed no earlier than
        the next step (DROP, or the end-of-day drop), where SELL can run.
        """
        return day * self.turns_per_day + 1 <= self.last_action_step


def seed_can_pay(clock: GameClock, step: int, crop: str) -> bool:
    return clock.sellable_from_day(clock.day(step + 1) + CROP_FIRST_YIELD_DAY[crop])


def animal_can_pay(clock: GameClock, step: int) -> bool:
    return clock.sellable_from_day(clock.day(step + 2) + FERTILIZER_READY_DAYS)


def land_can_pay(clock: GameClock, step: int) -> bool:
    return clock.sellable_from_day(clock.day(step + 1) + FERTILIZER_READY_DAYS)


def hire_can_pay(clock: GameClock, step: int) -> bool:
    last_hour = step % clock.turns_per_day == clock.turns_per_day - 1
    return not last_hour and step + 2 <= clock.last_action_step


def blocked_reason(order: JsonValue, step: int, clock: GameClock) -> str | None:
    """Why ``order`` at ``step`` cannot pay back, or None to keep it."""
    if type(order) is not list or not order:
        return None
    op = order[0]
    item = order[1] if len(order) > 1 else None
    last = clock.last_action_step
    if op == "BUY_SEED" and type(item) is str and item in CROP_FIRST_YIELD_DAY:
        can_pay = seed_can_pay(clock, step, item)
        what = f"BUY_SEED {item}: no harvest"
    elif op == "BUY_ANIMAL" and type(item) is str and item in ANIMALS:
        can_pay = animal_can_pay(clock, step)
        what = f"BUY_ANIMAL {item}: no product"
    elif op == "BUY_LAND":
        can_pay = land_can_pay(clock, step)
        what = "BUY_LAND: no use"
    elif op == "HIRE":
        can_pay = hire_can_pay(clock, step)
        what = "HIRE: no hand action"
    else:
        return None
    return None if can_pay else f"{what} can sell by step {last}"


def observation_step(observation: Mapping[str, Any], clock: GameClock) -> int:
    """The observation's step, checked against its day and hour."""
    step = observation["step"]
    if type(step) is not int or step < 0:
        raise ValueError(
            f"observation step must be a non-negative integer, got {step!r}"
        )
    expected = (clock.day(step), step % clock.turns_per_day)
    if (observation["day"], observation["hour"]) != expected:
        raise ValueError(
            f"observation day/hour {(observation['day'], observation['hour'])} "
            f"disagree with step {step} at turnsPerDay {clock.turns_per_day}"
        )
    return step


def filter_late_investments(
    action: dict[str, JsonValue], step: int, clock: GameClock
) -> tuple[dict[str, JsonValue], tuple[str, ...]]:
    """Replace each unproductive purchase with ``[]``; return the reasons.

    With nothing to drop, the input object itself is returned unchanged.
    """
    market = action["market"]
    if type(market) is not list:
        raise ValueError("action market must be a list of commands")
    reasons: list[str] = []
    filtered: list[JsonValue] = []
    for order in market:
        reason = blocked_reason(order, step, clock)
        if reason is None:
            filtered.append(order)
        else:
            reasons.append(reason)
            filtered.append([])
    if not reasons:
        return action, ()
    return {**action, "market": filtered}, tuple(reasons)
