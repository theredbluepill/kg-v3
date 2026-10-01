"""run_game.py plus a per-game log of every rule-2 block and rule-1 rewrite.

Runs ``../run_game.py`` unchanged (same argv), with two pass-through taps:

* ``owl.kaggriculture.kaggle_agent.filter_late_investments`` is wrapped once the
  packaged agent module is loaded (the packaged ``main.py`` is exec'd inside the
  first ``act``; rule 2 cannot block before step 479 at 720/24, so no block is
  missed). Each call that returns reasons is recorded with the step, the
  original market, the filtered market and the reasons. The return value is the
  wrapped function's own, unchanged.
* ``kaggle_environments.agent.Agent.act`` is wrapped outside run_game's own
  recorder to keep the agent's captured stdout lines that mention
  ``late-invest blocked`` or ``final_turn_liquidation`` (Kaggle captures agent
  stdout with ``debug=False``), as a cross-check of the tap.

Writes ``--blocks OUT.json`` (after run_game finishes, pass or fail).
usage: python run_game_logged.py --blocks OUT.json <run_game.py args...>
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] != "--blocks":
        raise SystemExit(__doc__)
    blocks_path = Path(sys.argv[2])
    sys.argv = [str(HERE.parent / "run_game.py"), *sys.argv[3:]]
    spec = importlib.util.spec_from_file_location("run_game", HERE.parent / "run_game.py")
    assert spec is not None and spec.loader is not None
    run_game = importlib.util.module_from_spec(spec)
    sys.modules["run_game"] = run_game
    spec.loader.exec_module(run_game)

    kaggle_agent_mod = importlib.import_module("kaggle_environments.agent")
    blocks: list[dict[str, Any]] = []
    stdout_lines: list[dict[str, Any]] = []
    state = {"tapped": False}

    def tap() -> None:
        agent_mod = sys.modules.get("owl.kaggriculture.kaggle_agent")
        if state["tapped"] or agent_mod is None:
            return
        original = agent_mod.filter_late_investments

        def logged(action: dict, step: int, clock: Any) -> Any:
            before = json.loads(json.dumps(action["market"]))
            filtered, reasons = original(action, step, clock)
            if reasons:
                blocks.append({
                    "step": step,
                    "market_before": before,
                    "market_after": filtered["market"],
                    "reasons": list(reasons),
                })
            return filtered, reasons

        agent_mod.filter_late_investments = logged
        state["tapped"] = True

    real_act = kaggle_agent_mod.Agent.act

    def act(self: Any, observation: Any) -> Any:
        tap()
        action, log = real_act(self, observation)
        tap()
        out = log.get("stdout") or ""
        for line in out.splitlines():
            if "late-invest blocked" in line or "final_turn_liquidation" in line or "load_s=" in line:
                stdout_lines.append({
                    "obs_step": int(observation["step"]),
                    "seat": int(observation["player"]),
                    "line": line,
                })
        return action, log

    kaggle_agent_mod.Agent.act = act
    try:
        run_game.main()
    finally:
        blocks_path.parent.mkdir(parents=True, exist_ok=True)
        blocks_path.write_text(json.dumps({
            "tapped": state["tapped"],
            "blocks": blocks,
            "stdout_lines": stdout_lines,
        }, indent=1) + "\n")


if __name__ == "__main__":
    main()
