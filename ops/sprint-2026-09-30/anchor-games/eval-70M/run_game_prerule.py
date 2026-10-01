"""run_game.py plus a record of the policy's own (pre-rule) final-turn action.

Runs ``../run_game.py`` unchanged (same argv) with one pass-through tap: once the
packaged agent module is loaded (the packaged ``main.py`` is exec'd inside the
first ``act``), ``owl.kaggriculture.kaggle_agent.liquidate_final_turn`` is
wrapped. On the observation where the rule applies (obs step == episodeSteps-2,
i.e. 718) the wrapper records the step, the validated policy action handed to
the rule (the model's own action, what rule-1-OFF would return) and the rule's
output. The return value is the wrapped function's own, unchanged. The tap only
observes; with the rule switch at 0 the function is never called.

Writes ``--prerule OUT.json`` after run_game finishes (pass or fail).
usage: python run_game_prerule.py --prerule OUT.json <run_game.py args...>
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

W = Path(__file__).resolve().parents[1]


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] != "--prerule":
        raise SystemExit(__doc__)
    out_path = Path(sys.argv[2])
    sys.argv = [str(W / "run_game.py"), *sys.argv[3:]]
    spec = importlib.util.spec_from_file_location("run_game", W / "run_game.py")
    assert spec is not None and spec.loader is not None
    run_game = importlib.util.module_from_spec(spec)
    sys.modules["run_game"] = run_game
    spec.loader.exec_module(run_game)

    kaggle_agent_mod = importlib.import_module("kaggle_environments.agent")
    records: list[dict[str, Any]] = []
    state = {"tapped_after_call": None, "calls": 0}

    def tap() -> None:
        agent_mod = sys.modules.get("owl.kaggriculture.kaggle_agent")
        if state["tapped_after_call"] is not None or agent_mod is None:
            return
        original = agent_mod.liquidate_final_turn

        def logged(observation: Any, configuration: Any, action: dict, *, order_limit: int) -> Any:
            before = json.loads(json.dumps(action))
            result = original(observation, configuration, action, order_limit=order_limit)
            if result is not action:
                records.append({
                    "obs_step": int(observation["step"]),
                    "seat": int(observation["player"]),
                    "policy_action": before,
                    "rule_action": json.loads(json.dumps(result)),
                })
            return result

        agent_mod.liquidate_final_turn = logged
        state["tapped_after_call"] = state["calls"]

    real_act = kaggle_agent_mod.Agent.act

    def act(self: Any, observation: Any) -> Any:
        tap()
        action, log = real_act(self, observation)
        state["calls"] += 1
        tap()
        return action, log

    kaggle_agent_mod.Agent.act = act
    try:
        run_game.main()
    finally:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "tapped_after_call": state["tapped_after_call"],
            "records": records,
        }, indent=1) + "\n")


if __name__ == "__main__":
    main()
