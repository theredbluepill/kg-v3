"""Check the liquidation rule against the packaged 08bc agent on recorded final observations.

For each replay: rebuild our seat's obs 718 exactly as Kaggle passes it (shared keys
from seat 0 when the replay stores them only there), run the packaged agent's
deterministic policy_action (must reproduce the recorded action), apply
liquidate_final_turn, and require the result to pass the agent's native
validate_action round trip. Also checks the rule is the identity on obs 717.
Run with the local Kaggle venv and PYTHONPATH=<pkg-p4>.
"""
import gzip, json, re, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from endgame_rule import liquidate_final_turn  # noqa: E402
from owl.kaggriculture.kaggle_agent import KaggricultureAgent, validate_action  # noqa: E402
from owl.kaggriculture.kaggle_view import SeatArrays, encode_seat_into, seat_view_json  # noqa: E402
from owl.kaggriculture.types import MAX_ACTORS  # noqa: E402

PKG = os.environ["PKG"]
agent = KaggricultureAgent(__import__("pathlib").Path(PKG) / "models" / "primary",
                           deterministic=True, strict=True, min_overage_time=0.0)


def seat_obs(steps, t, me):
    o = dict(steps[t][me]["observation"])
    for k, v in steps[t][0]["observation"].items():
        o.setdefault(k, v)
    o.setdefault("step", t)
    o["remainingOverageTime"] = 60.0
    return o


ok = bad = same = 0
for path in sys.argv[1:]:
    me = int(re.search(r"seat(\d)", path.split("/")[-1]).group(1))
    d = json.load(gzip.open(path)); s = d["steps"]; cfg = d["configuration"]; T = len(s) - 2
    for t in (T - 1, T):
        obs = seat_obs(s, t, me)
        act = agent.policy_action(obs, cfg)
        recorded = s[t + 1][me]["action"]
        same += act == recorded
        new = liquidate_final_turn(obs, cfg, act)
        if t < T:
            assert new is act, "rule must be identity before the final turn"
            continue
        view, seat, _ = seat_view_json(obs, cfg)
        encode_seat_into(view, seat, agent.arrays)
        actors = int(agent.arrays.actor_mask[0, 0, :MAX_ACTORS].sum())
        order_limit = int(agent.arrays.order_limits[0, 0])
        try:
            validate_action(new, actors=actors, order_limit=order_limit,
                            hire_limit=agent.hire_limit, scratch=agent._scratch)
            ok += 1
        except Exception as e:  # noqa: BLE001
            bad += 1; print("INVALID", path, e, new["market"])
        print(path.split("/")[-1], "order_limit", order_limit, "model==recorded", act == recorded,
              "market", new["market"], flush=True)
print(f"valid {ok}, invalid {bad}, model reproduced recorded action {same}/{2*len(sys.argv[1:])}")
