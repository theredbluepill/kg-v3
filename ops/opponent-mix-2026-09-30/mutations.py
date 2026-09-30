"""Source mutations for the opponent-mix change: each must turn a named test red.

Each mutation replaces one exact snippet, runs the named test command, records
whether it failed (killed) and restores the file byte for byte.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV = {**os.environ, "CARGO_BUILD_JOBS": "2", "OMP_NUM_THREADS": "2"}
RUST = ["cargo", "test", "--offline", "-q"]
PY = ["uv", "run", "--offline", "pytest", "-q", "-p", "no:cacheprovider", "-x"]

MUTATIONS = [
    (
        "M1 bot seat executes PASS instead of the bot's action",
        "src/kaggriculture/env.rs",
        "pair[seat] = bot\n",
        'pair[seat] = serde_json::json!({"farmer":["PASS"],"hands":[],"market":[]}); let _ = bot\n',
        RUST + ["opponent_env_tests::scripted_seat_plays_the_bot"],
    ),
    (
        "M2 auto-reset keeps the finished game's controller",
        "src/kaggriculture/env.rs",
        "                                if seeds[i].is_some() {\n                                    bot = new_bot(",
        "                                if false {\n                                    bot = new_bot(",
        RUST + ["opponent_env_tests::seats_alternate"],
    ),
    (
        "M3 learned seat ignores the episode",
        "src/kaggriculture/env.rs",
        "((env % 2) + (episode % 2) as usize) % 2",
        "(env % 2) + 0 * (episode as usize)",
        RUST + ["opponent_env_tests::seats_alternate"],
    ),
    (
        "M4 scripted-seat transport is not checked",
        "src/kaggriculture/env.rs",
        "if *length != 0 || tokens.iter().any(|token| *token != 0) {",
        "if false {",
        RUST + ["opponent_env_tests::scripted_rows_must_submit"],
    ),
    (
        "M5 terminal record drops the learned seat",
        "src/kaggriculture/env.rs",
        "learner_seat: bot.as_ref().map(|bot| 1 - bot.seat()),",
        "learner_seat: bot.as_ref().map(|bot| bot.seat()),",
        RUST + ["opponent_env_tests::seats_alternate"],
    ),
    (
        "M6 learner mask not applied to the loss masks",
        "python/owl/train/ppo.py",
        "            value_mask, policy_mask, policy_entity_mask = _apply_learner_mask(\n                segments.learner, value_mask, policy_mask, policy_entity_mask\n            )",
        "            pass",
        PY + ["tests/kaggriculture/test_opponent_mix.py::test_learner_mask_excludes_scripted_rows_from_every_loss_term"],
    ),
    (
        "M7 advantage/value masks keep scripted rows (value mask only)",
        "python/owl/train/ppo.py",
        "        value_mask & learner,\n",
        "        value_mask,\n",
        PY + ["tests/kaggriculture/test_opponent_mix.py::test_learner_mask_excludes_scripted_rows_from_every_loss_term"],
    ),
    (
        "M8 rollout forward runs the full batch",
        "python/owl/train/ppo.py",
        "                if self._learner_host is None:\n                    with _autocast_context",
        "                if True:\n                    with _autocast_context",
        PY + ["tests/kaggriculture/test_opponent_mix.py::test_rollout_forward_runs_on_learner_rows_only"],
    ),
    (
        "M9 learner mask not refreshed after a step",
        "python/owl/train/ppo.py",
        "                    self._learner_host.copy_(learner_mask)\n",
        "                    pass\n",
        PY + ["tests/kaggriculture/test_opponent_mix.py::test_training_telemetry_keys_against_the_bot"],
    ),
    (
        "M10 self-play bank keys include scripted games",
        "python/owl/train/ppo.py",
        '        if "_terminal_learner_seat" not in env_metrics:',
        "        if True:",
        PY + ["tests/kaggriculture/test_opponent_mix.py::test_training_telemetry_keys_against_the_bot"],
    ),
    (
        "M11 self-play config dumps opponent_mix: null",
        "python/owl/kaggriculture/config.py",
        "        if self.opponent_mix is None:\n            del data[\"opponent_mix\"]",
        "        if False:\n            del data[\"opponent_mix\"]",
        PY + ["tests/kaggriculture/test_opponent_mix.py::test_self_play_config_dumps_exactly_as_before_the_mix"],
    ),
    (
        "M12 last-best evaluation env keeps the training mix",
        "scripts/run_ppo.py",
        '            cfg.env.model_copy(update={"opponent_mix": opponent_mix}),',
        "            cfg.env,",
        PY + ["tests/scripts/test_run_ppo.py::test_last_best_evaluation_env_never_hosts_the_training_opponent"],
    ),
    (
        "M13 replay input keeps scripted rows live",
        "python/owl/train/ppo.py",
        '        obs.model_copy(update={"still_playing": obs.still_playing & learner}),',
        "        obs,",
        PY + ["tests/kaggriculture/test_opponent_mix.py::test_learner_mask_excludes_scripted_rows_from_every_loss_term"],
    ),
]


def main() -> int:
    only = set(sys.argv[1:])
    killed = 0
    total = 0
    for name, rel, old, new, command in MUTATIONS:
        if only and name.split()[0] not in only:
            continue
        total += 1
        path = ROOT / rel
        original = path.read_bytes()
        text = original.decode()
        if text.count(old) != 1:
            print(f"{name}: SNIPPET NOT UNIQUE ({text.count(old)})", flush=True)
            continue
        path.write_text(text.replace(old, new))
        try:
            result = subprocess.run(
                command, cwd=ROOT, env=ENV, capture_output=True, text=True
            )
        finally:
            path.write_bytes(original)
        status = "KILLED" if result.returncode != 0 else "SURVIVED"
        killed += result.returncode != 0
        tail = (result.stdout + result.stderr).strip().splitlines()[-2:]
        print(f"{name}: {status} (exit {result.returncode}) :: {' | '.join(tail)}", flush=True)
    print(f"killed {killed}/{total}")
    return 0 if killed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
