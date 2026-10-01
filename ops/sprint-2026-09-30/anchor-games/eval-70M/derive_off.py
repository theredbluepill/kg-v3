"""Exact rule-1-OFF arm from rule-1-ON games (final-turn counterfactual).

Rule 1 rewrites only our action on obs step 718 (episodeSteps-2); before that the
two arms are the same trajectory (proven on 96 c50/60M game pairs). So the OFF game
= the ON game's recorded actions for both seats on obs steps 0..717, the opponent's
recorded final action, and our model's own (pre-rule) obs-718 action.

Primary derivation: replay those actions through the fixed-shop kaggle-environments
engine (same make("kaggriculture", {"seed": seed}) as run_game.py; agents are
replay callables, no model inference). Cross-check: endgame/cf.py's market model
(step_ctx + run_market) on the derived final transition must reproduce the engine's
money change for our seat.

Modes:
  validate60  ON=60M-ft-on, OFF action = 60M-off's recorded obs-718 action; the
              derived banks must equal the played 60M-off banks, and the no-override
              replay must equal the played ON banks.
  derive70    ON=70M-ft-on, OFF action = prerule/<label>.json policy_action.
  replay-check LABELS...  (70M) replay ON with no override and compare to the played ON game.

Run: venv-kaggle python, PYTHONPATH=<pkg-70M>:<kenv-fixedshop>
usage: python derive_off.py MODE OUT.jsonl [--jobs N] [--only anchor:seed:seat ...]
"""

from __future__ import annotations

import gzip
import importlib
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any

W = Path(__file__).resolve().parents[1]
G = W / "games-fixedshop"
sys.path.insert(0, str(W / "endgame"))
ANCHORS = ("smaller_market_shock", "cha22", "v56")
SEEDS = tuple(range(93001, 93009))


def load_replay(root: Path, label: str) -> dict:
    return json.load(gzip.open(root / "replays" / f"replay-{label}.json.gz"))


def load_receipt(root: Path, label: str) -> dict:
    return json.loads((root / "receipts" / f"{label}.json").read_text())


def replay_engine(steps: list, seed: int, override: tuple[int, int, dict] | None) -> list:
    """Re-run the engine from the recorded actions; override = (seat, obs_step, action)."""
    make = importlib.import_module("kaggle_environments").make

    def agent_for(seat: int) -> Any:
        def agent(obs: Any, config: Any) -> Any:
            t = int(obs["step"])
            if override is not None and override[0] == seat and override[1] == t:
                return override[2]
            return steps[t + 1][seat]["action"]
        return agent

    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.run([agent_for(0), agent_for(1)])
    return env.steps


def banks_of(env_steps: list) -> list[float]:
    return [float(f["money"]) for f in env_steps[-1][0]["observation"]["farms"]]


def cf_money_delta(steps_json: list, me: int) -> dict:
    from cf import run_market, step_ctx
    T = len(steps_json) - 2
    o_prev, o_cur, acts, sheds, money, queues = step_ctx(steps_json, T)
    inv = dict(o_prev[0]["market"]["inventory"])
    units, cash, _ = run_market(dict(inv), [dict(s) for s in sheds], list(money), queues, me)
    observed = o_cur[me]["farms"][me]["money"] - o_prev[me]["farms"][me]["money"]
    market = acts[me].get("market") or []
    only_sells = all(isinstance(o, list) and o and o[0] == "SELL" for o in market)
    return {"cf_cash": cash, "cf_units": units, "engine_money_delta": observed,
            "only_sells": only_sells, "cf_matches_engine": cash == observed}


def job(args: tuple) -> dict:
    mode, anchor, seed, me = args
    if mode == "validate60":
        on_root, off_root, prefix = G / "60M-ft-on", G / "60M", "60M"
    else:
        on_root, off_root, prefix = G / "70M-ft-on", None, "70M"
    label = f"{prefix}-{anchor}-s{seed}-seat{me}"
    on = load_replay(on_root, label)
    on_steps = on["steps"]
    final = len(on_steps) - 1          # replay index holding the action for obs final-1
    obs_final = final - 1              # 718
    on_banks = load_receipt(on_root, label)["summary"]["final_banks"]
    row: dict[str, Any] = {"mode": mode, "label": label, "anchor": anchor, "seed": seed,
                           "seat": me, "obs_step": obs_final, "on_banks": on_banks}
    if mode == "replay-check":
        env_steps = replay_engine(on_steps, seed, None)
        js = json.loads(json.dumps([[s for s in st] for st in env_steps], default=str))
        row["replay_banks"] = banks_of(env_steps)
        row["replay_equals_on_banks"] = row["replay_banks"] == on_banks
        row["replay_actions_equal"] = all(
            js[t][p]["action"] == on_steps[t][p]["action"] for t in range(len(on_steps)) for p in (0, 1)
        ) and len(js) == len(on_steps)
        row["replay_obs_equal"] = all(
            {k: v for k, v in js[t][p]["observation"].items() if k != "remainingOverageTime"}
            == {k: v for k, v in on_steps[t][p]["observation"].items() if k != "remainingOverageTime"}
            for t in range(len(on_steps)) for p in (0, 1))
        return row
    if mode == "validate60":
        off = load_replay(off_root, label)
        own_action = off["steps"][final][me]["action"]
        row["off_played_banks"] = load_receipt(off_root, label)["summary"]["final_banks"]
    else:
        pre = json.loads((on_root / "prerule" / f"{label}.json").read_text())
        recs = [r for r in pre["records"] if r["seat"] == me]
        if len(recs) != 1 or recs[0]["obs_step"] != obs_final:
            raise RuntimeError(f"{label}: expected one obs-{obs_final} prerule record, got {recs}")
        own_action = recs[0]["policy_action"]
        row["rule_action_equals_on_replay"] = recs[0]["rule_action"] == on_steps[final][me]["action"]
        row["tapped_after_call"] = pre["tapped_after_call"]
    row["own_action"] = own_action
    row["rule_action"] = on_steps[final][me]["action"]
    row["rule_changed_action"] = own_action != on_steps[final][me]["action"]
    env_steps = replay_engine(on_steps, seed, (me, obs_final, own_action))
    js = json.loads(json.dumps(env_steps, default=str))
    row["derived_banks"] = banks_of(env_steps)
    row["derived_n_steps"] = len(js)
    row["derived_prefix_actions_equal_on"] = all(
        js[t][p]["action"] == on_steps[t][p]["action"]
        for t in range(final) for p in (0, 1))
    row["derived_final_opp_action_equal_on"] = js[final][1 - me]["action"] == on_steps[final][1 - me]["action"]
    row["derived_final_own_action_is_policy"] = js[final][me]["action"] == own_action
    row["derived_statuses_final"] = [s["status"] for s in js[-1]]
    row["derived_bad_statuses"] = sum(
        s["status"] in ("ERROR", "TIMEOUT", "INVALID") for st in js for s in st)
    row["cf"] = cf_money_delta(js, me)
    if mode == "validate60":
        row["derived_equals_off_played"] = row["derived_banks"] == row["off_played_banks"]
    # keep the derived replay (compact) for audit
    outdir = on_root.parent / (f"{prefix}-off-derived" if mode == "derive70" else "60M-off-derived-validate") / "replays"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"replay-{label}.json.gz").write_bytes(gzip.compress(
        json.dumps({"derived_from": str(on_root / 'replays' / f'replay-{label}.json.gz'),
                    "override": {"seat": me, "obs_step": obs_final, "action": own_action},
                    "seed": seed, "steps": js}, separators=(",", ":")).encode(), mtime=0))
    return row


def main() -> None:
    mode, out = sys.argv[1], Path(sys.argv[2])
    rest = sys.argv[3:]
    jobs_n = 1
    only: list[tuple] | None = None
    if "--jobs" in rest:
        jobs_n = int(rest[rest.index("--jobs") + 1])
    if "--only" in rest:
        only = [(a, int(s), int(p)) for a, s, p in (x.split(":") for x in rest[rest.index("--only") + 1:])]
    keys = only or [(a, s, p) for a in ANCHORS for s in SEEDS for p in (0, 1)]
    args = [(mode, *k) for k in keys]
    with Pool(jobs_n) as pool, out.open("w") as fh:
        for row in pool.imap(job, args):
            fh.write(json.dumps(row) + "\n"); fh.flush()
            brief = {k: row[k] for k in row if k in (
                "label", "derived_banks", "off_played_banks", "derived_equals_off_played",
                "replay_equals_on_banks", "replay_obs_equal", "rule_changed_action")}
            brief["cf_ok"] = row.get("cf", {}).get("cf_matches_engine")
            print(json.dumps(brief), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
