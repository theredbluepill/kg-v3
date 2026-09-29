"""Tiny live probes for failed-evaluation custody, without modifying sources.

Question: after one committed native transition, does a decoder rejection or
native batch transaction rejection write explicit error custody for each active
selected game? Expected: non-success sidecars containing exactly the committed
one-turn action tape, no successful episode, then the original failure raised.
Stop after one injected rejection in each case. No model or training is run.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from owl import rs
from owl.kaggriculture.native_evaluation import evaluate_native_games
from owl.kaggriculture.replay_export import ReplayRecorder
from tests.kaggriculture.test_native_env import REWARD, buffers
from tests.kaggriculture.test_replay_export_integration import (
    HASHES, VERSIONS, market_policy,
)


def probe(kind: str) -> dict:
    with TemporaryDirectory(prefix=f"t73-{kind}-") as directory:
        output = Path(directory)
        count = 2 if kind == "selected_decoder_failure" else 1
        recorder = ReplayRecorder(
            output_dir=output,
            evaluation_identity="verify-rejected-transaction",
            total_games=2,
            config=SimpleNamespace(eval_replay_games=count),
            seat_assignments=[0, 0],
        )
        selected = sorted(recorder.selected_games)
        rejected_env = (
            selected[0]
            if kind == "selected_decoder_failure"
            else next(index for index in range(2) if index not in selected)
        )
        configuration = {"episodeSteps": 4}
        env = rs.KaggricultureEnv(
            2, 91, 3, json.dumps(configuration), REWARD, 1, hire_limit=241
        )
        arrays = buffers(2)
        calls = 0

        def policy(observation, seats):
            nonlocal calls
            calls += 1
            tokens, lengths = market_policy(observation, seats)
            if calls == 2:
                lengths[rejected_env, 0] = 0
            return tokens, lengths

        error_type, error_text = None, None
        try:
            evaluate_native_games(
                env, arrays, policy,
                n_games=2, seat_assignments=[0, 0],
                configuration=configuration, hire_limit=241,
                recorder=recorder, checkpoint_hashes=HASHES, versions=VERSIONS,
            )
        except Exception as error:
            error_type, error_text = type(error).__name__, str(error)
        files = sorted(path.name for path in output.iterdir())
        custody = [json.loads(path.read_text()) for path in output.glob("*.custody.json")]
        return {
            "case": kind,
            "selected_ordinals": selected,
            "rejected_env": rejected_env,
            "policy_calls": calls,
            "error_type": error_type,
            "error_text": error_text,
            "native_committed_steps": arrays["globals_int"][:, 0, 0].tolist(),
            "expected_error_custody_count": len(selected),
            "actual_error_custody_count": len(custody),
            "actual_files": files,
            "active_ordinals_after_failure": sorted(recorder._active),
            "committed_turns_retained_only_in_memory": {
                str(ordinal): len(record.transitions)
                for ordinal, record in recorder._active.items()
            },
        }


if __name__ == "__main__":
    results = [probe(kind) for kind in (
        "selected_decoder_failure", "unselected_native_transaction_failure"
    )]
    receipt = {
        "question": __doc__,
        "source": "python/owl/kaggriculture/native_evaluation.py:119-141",
        "expected": "error custody for every active selected game, then rethrow",
        "results": results,
    }
    print(json.dumps(receipt, indent=2))
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(receipt, indent=2) + "\n")
