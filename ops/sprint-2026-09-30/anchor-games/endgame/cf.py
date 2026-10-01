"""Counterfactual liquidation at the end of a recorded game (read-only analysis).

Uses the market-analysis slot/unit-interleaved market model (validated here
against the observed money change of our last step). Our seat = label's -seatN.

Variants (opponent actions held fixed as recorded):
  final_sell    : at the last actionable observation (obs step 718) our market
                  queue is replaced by SELL <all shed units> per product, highest
                  current price first; our unit commands unchanged.
  final_sell_drop: final_sell, and every actor of ours that stands on a shed
                  access tile and carries products DROPs first (before market).
  window_K      : approximate; for obs steps 719-K..718 our queue keeps its
                  non-SELL orders and its SELLs are replaced by SELL <all shed>
                  per product; extra units sold shift our later shed down and the
                  pool up (opponent fills not re-simulated). Plus the final DROP.
Exact for final_* up to the market model; window_K is an approximation.

Usage: python3 cf.py <out.jsonl> <replay.json.gz>...
"""
import gzip, json, os, re, sys
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "market-analysis"))
from extract import P, PRODUCTS, SHOPS, price, parse, carried  # noqa: E402

ACCESS = {(4, 4), (5, 4), (4, 5), (5, 5)}  # shed_access_tiles(10)
CAP = 100


def pre_market_shed(o_prev, o_cur, action, day_end):
    """extract.py's inference of the pre-market shed from the recorded transition."""
    s = dict(o_prev["private"]["shed"])
    ci0 = carried(o_prev["private"]); ci1 = carried(o_cur["private"])
    a = action or {}
    cmds = [a.get("farmer")] + list(a.get("hands") or []) if isinstance(a, dict) else []
    for ai, inv0 in enumerate(ci0):
        c = cmds[ai] if ai < len(cmds) else None
        op = c[0] if isinstance(c, list) and c else None
        if day_end:
            if op in ("DROP", "PLACE"):
                for it, q in inv0.items():
                    if it in P: s[it] = s.get(it, 0) + q
            continue
        if op in ("FEED", "PLANT"): continue
        inv1 = ci1[ai] if ai < len(ci1) else {}
        if op == "PICKUP":
            for it, q in inv1.items():
                dq = q - inv0.get(it, 0)
                if dq > 0 and it in P: s[it] = s.get(it, 0) - dq
            continue
        for it, q in inv0.items():
            dq = q - inv1.get(it, 0)
            if dq > 0 and it in P: s[it] = s.get(it, 0) + dq
    return s


def run_market(inv, sheds, money, queues, me):
    """Mutates inv/sheds/money; returns (our units, our cash, per-item our units)."""
    units = 0; cash = 0; per = {}
    for slot in range(max([len(q) for q in queues] + [0])):
        st = [list(queues[p][slot]) if slot < len(queues[p]) and queues[p][slot] else None for p in range(2)]
        while True:
            quotes = [None, None]
            for p in range(2):
                s_ = st[p]
                if not s_ or s_[2] <= 0: continue
                k, it = s_[0], s_[1]
                if k == "SELL" and it in P: quotes[p] = price(it, inv[it])
                elif k == "BUY_PRODUCT" and it in ("WHEAT", "FERTILIZER"): quotes[p] = price(it, inv[it] - 1)
                else: st[p] = None
            if all(q is None for q in quotes): break
            any_c = False
            for p in range(2):
                q = quotes[p]
                if q is None: continue
                k, it = st[p][0], st[p][1]
                if k == "SELL":
                    if sheds[p].get(it, 0) <= 0: st[p] = None; continue
                    sheds[p][it] -= 1; money[p] += q
                    if q > 1: inv[it] += 1
                    if p == me:
                        units += 1; cash += q; per[it] = per.get(it, 0) + 1
                else:
                    if money[p] < q: st[p] = None; continue
                    money[p] -= q; inv[it] -= 1; sheds[p][it] = sheds[p].get(it, 0) + 1
                st[p][2] -= 1; any_c = True
            if not any_c: break
    return units, cash, per


def liquidation_queue(shed, inv, keep=()):
    items = [it for it in PRODUCTS if shed.get(it, 0) > 0]
    items.sort(key=lambda it: -price(it, inv[it]))
    q = list(keep) + [["SELL", it, shed[it]] for it in items]
    return q[:10]


def drop_extra(o, me, action):
    """Products our access-tile actors carry at obs `o` that a forced DROP would add
    beyond what the recorded commands already moved (recorded DROP/PLACE excluded)."""
    f = o["farms"][me]; inv = o["private"]["inventories"]
    pos = [f["farmer"]] + list(f["hands"])
    cmds = [action.get("farmer")] + list(action.get("hands") or [])
    add = {}; actors = 0
    for i, p in enumerate(pos):
        if i >= len(inv) or tuple(p) not in ACCESS: continue
        c = cmds[i] if i < len(cmds) else None
        if isinstance(c, list) and c and c[0] in ("DROP", "PLACE", "PICKUP"): continue
        got = {k: v for k, v in inv[i].items() if k in P and v > 0}
        if got: actors += 1
        for k, v in got.items(): add[k] = add.get(k, 0) + v
    return add, actors


def carried_off_access(o, me):
    f = o["farms"][me]; inv = o["private"]["inventories"]
    pos = [f["farmer"]] + list(f["hands"])
    n = 0
    for i, p in enumerate(pos):
        if i < len(inv) and tuple(p) not in ACCESS:
            n += sum(v for k, v in inv[i].items() if k in P and v > 0)
    return n


def town(inv, ps, shops):
    if ps % 4 == 0:
        for shop in shops:
            prods = SHOPS[shop]; mlt = 2 if len(prods) == 1 else 1
            for it in prods: inv[it] -= mlt
    if ps % 24 == 0:
        for it in PRODUCTS:
            if it != "FERTILIZER": inv[it] -= 1


def step_ctx(steps, t):
    """Context for the action chosen on obs t (recorded at steps[t+1])."""
    prev, cur = steps[t], steps[t + 1]
    o_prev = [prev[p]["observation"] for p in range(2)]
    o_cur = [cur[p]["observation"] for p in range(2)]
    acts = [cur[p].get("action") or {} for p in range(2)]
    day_end = ((t + 1) % 24 == 0)
    sheds = [pre_market_shed(o_prev[p], o_cur[p], acts[p], day_end) for p in range(2)]
    money = [o_prev[p]["farms"][p]["money"] for p in range(2)]
    queues = [[parse(o) for o in ((acts[p].get("market") if isinstance(acts[p], dict) else None) or [])[:10]]
              for p in range(2)]
    return o_prev, o_cur, acts, sheds, money, queues


def only_sells(q):
    raw = q or []
    return all(isinstance(o, list) and o and o[0] == "SELL" for o in raw)


def process(path):
    label = os.path.basename(path).replace("replay-", "").replace(".json.gz", "")
    me = int(re.search(r"-seat(\d)$", label).group(1)); opp = 1 - me
    d = json.load(gzip.open(path)); steps = d["steps"]
    T = len(steps) - 2  # last obs step acted on (718)
    rew = d["rewards"]; gap = rew[me] - rew[opp]
    res = {"label": label, "gap": gap, "bank": [rew[me], rew[opp]]}
    # --- actual final step, validation
    o_prev, o_cur, acts, sheds, money, queues = step_ctx(steps, T)
    inv = dict(o_prev[0]["market"]["inventory"])
    a_units, a_cash, _ = run_market(dict(inv), [dict(s) for s in sheds], list(money), queues, me)
    obs_delta = o_cur[me]["farms"][me]["money"] - o_prev[me]["farms"][me]["money"]
    res["actual_final"] = {"units": a_units, "cash": a_cash, "observed_money_delta": obs_delta,
                           "validatable": only_sells(acts[me].get("market")),
                           "orders": acts[me].get("market")}
    res["shed_pre_final"] = {k: v for k, v in sheds[me].items() if k in P and v > 0}
    # non-SELL spend on the final step (wasted: nothing bought can pay back)
    res["final_nonsell_orders"] = [o for o in (acts[me].get("market") or []) if isinstance(o, list) and o and o[0] != "SELL"]
    # --- final_sell
    q = [list(queues[0]), list(queues[1])]
    q[me] = liquidation_queue(sheds[me], inv)
    u, c, per = run_market(dict(inv), [dict(s) for s in sheds], list(money), q, me)
    res["final_sell"] = {"units": u, "cash": c, "gain": c - a_cash, "per_item": per, "flip": gap < 0 and gap + c - a_cash > 0}
    # --- final_sell_drop
    add, actors = drop_extra(o_prev[me], me, acts[me])
    s2 = [dict(s) for s in sheds]
    room = CAP - sum(v for v in s2[me].values() if v > 0)
    for k, v in add.items():
        take = min(v, max(room, 0)); s2[me][k] = s2[me].get(k, 0) + take; room -= take
    q[me] = liquidation_queue(s2[me], inv)
    u, c, per = run_market(dict(inv), s2, list(money), q, me)
    res["final_sell_drop"] = {"units": u, "cash": c, "gain": c - a_cash, "drop_units": sum(add.values()),
                              "drop_actors": actors, "flip": gap < 0 and gap + c - a_cash > 0}
    res["carried_off_access_final"] = carried_off_access(o_prev[me], me)
    # --- window_K approximations
    for K in (1, 5, 12, 23):
        extra_shed = {}; extra_pool = {}; gain = 0
        for t in range(T - K + 1, T + 1):
            o_prev, o_cur, acts, sheds, money, queues = step_ctx(steps, t)
            inv = dict(o_prev[0]["market"]["inventory"])
            base_u, base_c, _ = run_market(dict(inv), [dict(s) for s in sheds], list(money), queues, me)
            inv2 = {k: inv[k] + extra_pool.get(k, 0) for k in inv}
            s2 = [dict(s) for s in sheds]
            for k, v in extra_shed.items(): s2[me][k] = s2[me].get(k, 0) - v
            if t == T:
                add, _ = drop_extra(o_prev[me], me, acts[me])
                room = CAP - sum(v for v in s2[me].values() if v > 0)
                for k, v in add.items():
                    take = min(v, max(room, 0)); s2[me][k] = s2[me].get(k, 0) + take; room -= take
            keep = [o for o in queues[me] if o and o[0] != "SELL"] if t < T else []
            q = [list(queues[0]), list(queues[1])]
            q[me] = liquidation_queue(s2[me], inv2, keep)
            inv_run = dict(inv2)
            u, c, per = run_market(inv_run, s2, list(money), q, me)
            _, _, base_per = run_market(dict(inv), [dict(s) for s in sheds], list(money), queues, me)
            gain += c - base_c
            for k in set(per) | set(base_per):
                dx = per.get(k, 0) - base_per.get(k, 0)
                extra_shed[k] = extra_shed.get(k, 0) + dx
            # pool shift relative to the recorded path: our extra pool additions this step
            inv_rec = dict(inv); run_market(inv_rec, [dict(s) for s in sheds], list(money), queues, me)
            for k in inv:
                extra_pool[k] = extra_pool.get(k, 0) + (inv_run[k] - inv2[k]) - (inv_rec[k] - inv[k])
        res[f"window_{K}"] = {"gain": gain, "flip": gap < 0 and gap + gain > 0}
    return res


if __name__ == "__main__":
    out = sys.argv[1]; files = sys.argv[2:]
    with Pool(int(os.environ.get("NPROC", "6"))) as pool, open(out, "w") as f:
        for r in pool.imap_unordered(process, files):
            f.write(json.dumps(r, separators=(",", ":")) + "\n"); f.flush()
            print(r["label"], r["gap"], r["actual_final"]["cash"], r["actual_final"]["observed_money_delta"],
                  r["final_sell"]["gain"], r["final_sell_drop"]["gain"],
                  [r[f"window_{K}"]["gain"] for K in (1, 5, 12, 23)], flush=True)
