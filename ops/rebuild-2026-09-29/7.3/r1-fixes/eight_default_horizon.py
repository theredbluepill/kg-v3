"""Receipt: eight default-horizon native evaluation games, exported and verified.

Question: does the native evaluation seam export 8/8 selected complete games at
the default 720-step horizon that round-trip byte-exactly with full captured
evidence at every transition? Bounded Mac check: 2 envs, 1 native thread, debug
extension. Bulk episodes go to argv[1] (scratch); this writes a compact summary.
"""

from __future__ import annotations

import hashlib
import json
import resource
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from owl import rs
from owl.kaggriculture import replay_export
from owl.kaggriculture.native_evaluation import evaluate_native_games

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from tests.kaggriculture.test_native_env import REWARD, buffers  # noqa: E402
from tests.kaggriculture.test_replay_export_integration import (  # noqa: E402
    market_policy,
)

out = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
n_games = 8
seats = [ordinal % 2 for ordinal in range(n_games)]
identity = "task-7.3-r1-default-horizon"
recorder = replay_export.ReplayRecorder(
    output_dir=out,
    evaluation_identity=identity,
    total_games=n_games,
    config=SimpleNamespace(eval_replay_games=8),
    seat_assignments=seats,
)
env = rs.KaggricultureEnv(2, 2**62 + 1, 1, "{}", REWARD, 1, hire_limit=241)
source = subprocess.run(
    ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
).stdout.strip()
started = time.perf_counter()
results = evaluate_native_games(
    env,
    buffers(2),
    market_policy,
    n_games=n_games,
    seat_assignments=seats,
    configuration={},
    hire_limit=241,
    recorder=recorder,
    checkpoint_hashes={"candidate": "a" * 64, "incumbent": "b" * 64},
    versions={"source": source, "engine": "owl.rs debug", "schema_version": 1},
)
elapsed = time.perf_counter() - started
games = []
for result in results:
    custody = json.loads(
        (out / f"game_{result.game_ordinal:06d}.custody.json").read_text()
    )
    episode_bytes = (out / f"game_{result.game_ordinal:06d}.json").read_bytes()
    report = json.loads(rs.verify_kaggriculture_episode(episode_bytes.decode(), None))
    games.append(
        {
            "ordinal": result.game_ordinal,
            "env_index": result.env_index,
            "seed": result.seed,
            "candidate_seat": result.candidate_seat,
            "banks": result.banks,
            "winner": result.winner,
            "episode_steps": result.episode_steps,
            "status": custody["status"],
            "captured": custody["verification"]["captured"],
            "transitions": custody["verification"]["transitions"],
            "episode_sha256": custody["episode_sha256"],
            "episode_sha256_recomputed": hashlib.sha256(episode_bytes).hexdigest(),
            "reverify_mode": report["mode"],
        }
    )
summary = {
    "question": __doc__.split("\n\n")[1].replace("\n", " "),
    "source_head": source,
    "evaluation_identity": identity,
    "n_envs": 2,
    "n_games": n_games,
    "complete": sum(g["status"] == "complete" for g in games),
    "elapsed_seconds": round(elapsed, 2),
    "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    "games": games,
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({k: v for k, v in summary.items() if k != "games"}))
assert summary["complete"] == 8
assert all(g["episode_sha256"] == g["episode_sha256_recomputed"] for g in games)
