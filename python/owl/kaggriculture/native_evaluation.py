"""Drive complete evaluation games on the native Kaggriculture environment.

This is the native evaluation seam for the replay recorder: it owns game
ordinals, consumed-seed custody and before-auto-reset terminal capture over
``owl.rs.KaggricultureEnv`` and its caller-owned buffers. Model inference stays
behind ``policy``; the Task 1.5 torch adapter and ``run_ppo._evaluate_games``
can call it without changing custody. Seeds, seat schedules and checkpoint
identity are host metadata and never reach the policy's observation buffers.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from owl import rs
from owl.kaggriculture.replay_export import ReplayRecorder

TokenPolicy = Callable[
    [Mapping[str, NDArray[Any]], NDArray[np.int64]],
    tuple[NDArray[np.int64], NDArray[np.int64]],
]
"""``policy(buffers, candidate_seats) -> (tokens [E,2,252,12], lengths [E,2])``.

``candidate_seats[e]`` is the candidate model's seat in env ``e``'s counted game,
or ``-1`` once that env only plays uncounted filler games.
"""


@dataclass(frozen=True)
class NativeGameResult:
    """Raw-bank outcome of one counted evaluation game."""

    game_ordinal: int
    env_index: int
    seed: int
    candidate_seat: int
    banks: tuple[float, float]
    winner: int
    episode_steps: int


def evaluate_native_games(
    env: rs.KaggricultureEnv,
    buffers: dict[str, NDArray[Any]],
    policy: TokenPolicy,
    *,
    n_games: int,
    seat_assignments: Sequence[int],
    configuration: Mapping[str, Any],
    hire_limit: int,
    recorder: ReplayRecorder | None,
    checkpoint_hashes: Mapping[str, str],
    versions: Mapping[str, Any],
    start: Literal["observe", "reset"] = "observe",
) -> list[NativeGameResult]:
    """Play ``n_games`` complete games and record the recorder's selections.

    Ordinals follow ``_evaluate_games``: envs start ordinals in env-index order,
    and each terminal env starts the next ordinal in env-index order. ``start``
    chooses whether the first games are the constructor's (``observe``) or an
    explicit ``reset``'s; either way each game's seed is read from the env after
    it was consumed. ``configuration`` must be the overrides the env was built
    with. Returns results ordered by game ordinal.

    Any exception (policy, decoder, native step, evidence check) first writes
    error custody for every active selected game, holding the transitions
    recorded before the abort, then re-raises the original exception. A failed
    custody write is attached to it as a note, never raised in its place. An
    abort while recording a committed step's transitions leaves that step out
    of the tapes not yet recorded; their error names the cause.
    """
    n_envs = int(buffers["dones"].shape[0])
    if type(n_games) is not int or n_games < 1:
        raise ValueError("n_games must be a positive integer")
    if len(seat_assignments) != n_games:
        raise ValueError("seat_assignments length must equal n_games")
    if any(seat not in (0, 1) for seat in seat_assignments):
        raise ValueError("seat_assignments entries must be 0 or 1")
    if start == "observe":
        env.observe(**buffers)
    elif start == "reset":
        env.reset(**buffers)
    else:
        raise ValueError(f"start must be 'observe' or 'reset', got {start!r}")

    selected = recorder.selected_games if recorder is not None else frozenset()
    ordinals: list[int | None] = [None] * n_envs
    seeds: list[int] = [0] * n_envs
    candidate_seats = np.full(n_envs, -1, dtype=np.int64)
    started = 0

    def begin(env_index: int) -> None:
        nonlocal started
        if started >= n_games:
            ordinals[env_index] = None
            candidate_seats[env_index] = -1
            return
        ordinal = started
        started += 1
        ordinals[env_index] = ordinal
        seeds[env_index] = env.seed_state()[1][env_index]
        candidate_seats[env_index] = seat_assignments[ordinal]
        if recorder is not None and ordinal in selected:
            recorder.start_game(
                ordinal,
                env_index,
                seeds[env_index],
                configuration,
                seat_assignments[ordinal],
                checkpoint_hashes,
                versions,
                captured_initial=json.loads(env.state_snapshot(env_index)),
            )

    try:
        for env_index in range(n_envs):
            begin(env_index)

        results: dict[int, NativeGameResult] = {}
        while len(results) < n_games:
            tokens, lengths = policy(buffers, candidate_seats.copy())
            recorded = [
                env_index
                for env_index, ordinal in enumerate(ordinals)
                if ordinal is not None and ordinal in selected
            ]
            # Decode against the pre-step observation; the step overwrites it.
            actions = {
                env_index: [
                    json.loads(
                        rs.kaggriculture_decode(
                            np.ascontiguousarray(tokens[env_index, seat]),
                            int(lengths[env_index, seat]),
                            int(buffers["globals_int"][env_index, seat, 14]),
                            int(buffers["order_limits"][env_index, seat]),
                            hire_limit,
                        )
                    )
                    for seat in range(2)
                ]
                for env_index in recorded
            }
            env.step(tokens, lengths, **buffers)
            dones = buffers["dones"]
            terminal: NDArray[np.bool_] = np.logical_and(dones[:, 0], dones[:, 1])
            for env_index in recorded:
                ordinal = ordinals[env_index]
                if ordinal is None or recorder is None:
                    raise RuntimeError(f"env {env_index} lost its recorded game")
                snapshot_json = (
                    env.terminal_snapshot(env_index)
                    if terminal[env_index]
                    else env.state_snapshot(env_index)
                )
                if snapshot_json is None:
                    raise RuntimeError(f"missing terminal snapshot for env {env_index}")
                snapshot = json.loads(snapshot_json)
                banks = [farm["money"] for farm in snapshot["public"]["farms"]]
                if banks != buffers["transition_banks_after"][env_index].tolist():
                    # The abort handler writes every active game's error custody.
                    raise RuntimeError(
                        f"env {env_index} snapshot banks {banks} differ from the "
                        "published transition banks"
                    )
                recorder.record_transition(
                    ordinal,
                    actions[env_index],
                    tokens=[
                        {
                            "tokens": tokens[env_index, seat].reshape(-1).tolist(),
                            "length": int(lengths[env_index, seat]),
                        }
                        for seat in range(2)
                    ],
                    captured=snapshot,
                )
                if terminal[env_index]:
                    recorder.finish_game(ordinal, snapshot)
            for env_index in np.flatnonzero(terminal).tolist():
                ordinal = ordinals[env_index]
                if ordinal is not None:
                    metrics = env.terminal_metrics(env_index)
                    if metrics is None:
                        raise RuntimeError(
                            f"missing terminal metrics for env {env_index}"
                        )
                    results[ordinal] = NativeGameResult(
                        game_ordinal=ordinal,
                        env_index=env_index,
                        seed=seeds[env_index],
                        candidate_seat=seat_assignments[ordinal],
                        banks=(metrics["bank_0"], metrics["bank_1"]),
                        winner=metrics["winner"],
                        episode_steps=metrics["episode_steps"],
                    )
                begin(env_index)
    except BaseException as error:
        if recorder is not None:
            _publish_error_custody(recorder, error)
        raise
    return [results[ordinal] for ordinal in range(n_games)]


def _publish_error_custody(recorder: ReplayRecorder, error: BaseException) -> None:
    """Persist every active selected game as error custody for an aborted run.

    Each tape keeps only transitions the native env committed before the abort.
    Publication failures are attached to ``error`` as notes so the caller still
    sees the original decoder, native-step or evidence failure.
    """
    reason = f"evaluation aborted: {type(error).__name__}: {error}"
    for game_ordinal, publication_error in recorder.fail_active_games(error=reason):
        error.add_note(
            f"error custody for game {game_ordinal} was not published: "
            f"{type(publication_error).__name__}: {publication_error}"
        )
