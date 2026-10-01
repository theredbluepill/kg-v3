"""End-of-game unsold goods and last-sale timing from Kaggle replays (read-only).

Per replay (our seat from the label's ``-seatN``): final holdings of both seats by
location (shed / carried / tile yield), their value at the final observed prices
and a price-impact-aware value (sell unit by unit into the final pool, pool +1
per sale above the price-1 floor, same default market curves as extract.py),
whether that value would flip a loss, and when our committed SELLs happened
(re-simulated with the market-analysis slot/unit-interleaved model).

Usage: python3 endgame.py <out.jsonl> <replay.json.gz>...
"""
import gzip, json, os, re, sys
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "market-analysis"))
from extract import P, PRODUCTS, SHOPS, price, parse, carried  # noqa: E402

ANIMAL_PRODUCT = {"COW": "MILK", "SHEEP": "WOOL", "GOOSE": "EGG"}


def holdings(obs, p):
    priv = obs["private"]
    shed = {k: v for k, v in priv["shed"].items() if k in P and v > 0}
    animals_in_shed = {k: v for k, v in priv["shed"].items() if k not in P and v > 0}
    car = {}
    for inv in priv["inventories"]:
        for k, v in inv.items():
            if k in P and v > 0:
                car[k] = car.get(k, 0) + v
    tile = {}
    for row in obs["farms"][p]["tiles"]:
        for t in row:
            if not isinstance(t, dict) or not t.get("yield_units"):
                continue
            it = t.get("crop") if t.get("kind") == "PLANT" else ANIMAL_PRODUCT.get(t.get("animal"))
            if it:
                tile[it] = tile.get(it, 0) + t["yield_units"]
    return {"shed": shed, "carried": car, "tile": tile, "animals_in_shed": animals_in_shed,
            "seeds": sum(v for v in priv["seeds"].values() if v > 0)}


def naive_value(units, prices):
    return sum(q * prices[it] for it, q in units.items())


def impact_value(units, inv0):
    """Sell each item's units one by one into its own pool from inventory inv0."""
    total = 0
    for it, q in units.items():
        inv = inv0[it]
        for _ in range(q):
            pr = price(it, inv)
            total += pr
            if pr > 1:
                inv += 1
    return total


def merge(*ds):
    out = {}
    for d in ds:
        for k, v in d.items():
            out[k] = out.get(k, 0) + v
    return out


def sim_sells(steps, me):
    """Committed SELL units/cash of seat `me` per recorded step (action taken on obs t-1)."""
    per_step = {}
    for t in range(1, len(steps)):
        prev, cur = steps[t - 1], steps[t]
        o_prev = [prev[p]["observation"] for p in range(2)]
        o_cur = [cur[p]["observation"] for p in range(2)]
        inv = dict(o_prev[0]["market"]["inventory"])
        shed = []
        day_end = (t % 24 == 0)
        for p in range(2):
            s = dict(o_prev[p]["private"]["shed"])
            ci0 = carried(o_prev[p]["private"]); ci1 = carried(o_cur[p]["private"])
            a = cur[p].get("action") or {}
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
            shed.append(s)
        money = [o_prev[p]["farms"][p]["money"] for p in range(2)]
        queues = []
        for p in range(2):
            a = cur[p].get("action") or {}
            m = a.get("market") if isinstance(a, dict) else None
            queues.append([parse(o) for o in (m or [])[:10]])
        units = [0, 0]; cash = [0, 0]
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
                        if shed[p].get(it, 0) <= 0: st[p] = None; continue
                        shed[p][it] -= 1; money[p] += q
                        if q > 1: inv[it] += 1
                        if p == me: units[p] += 1; cash[p] += q
                    else:
                        if money[p] < q: st[p] = None; continue
                        money[p] -= q; inv[it] -= 1; shed[p][it] = shed[p].get(it, 0) + 1
                    st[p][2] -= 1; any_c = True
                if not any_c: break
        if units[me]:
            per_step[t - 1] = (units[me], cash[me])  # key = obs step the action was chosen on
    return per_step


def process(path):
    label = os.path.basename(path).replace("replay-", "").replace(".json.gz", "")
    m = re.search(r"-seat(\d)$", label)
    me = int(m.group(1)); opp = 1 - me
    d = json.load(gzip.open(path))
    steps = d["steps"]
    final = steps[-1]
    obs = [final[p]["observation"] for p in range(2)]
    prices = obs[0]["market"]["prices"]; inv0 = obs[0]["market"]["inventory"]
    price_check = all(price(it, inv0[it]) == prices[it] for it in PRODUCTS)
    rew = d["rewards"]
    res = {"label": label, "me": me, "bank": [rew[me], rew[opp]], "gap": rew[me] - rew[opp],
           "final_step": obs[0]["step"], "prices": prices, "price_check": price_check,
           "last_obs_step_acted": len(steps) - 2}
    for who, p in (("me", me), ("opp", opp)):
        h = holdings(obs[p], p)
        sc = merge(h["shed"], h["carried"])
        h["naive"] = {"shed": naive_value(h["shed"], prices), "carried": naive_value(h["carried"], prices),
                      "tile": naive_value(h["tile"], prices)}
        h["impact"] = {"shed": impact_value(h["shed"], inv0), "shed_carried": impact_value(sc, inv0),
                       "all": impact_value(merge(sc, h["tile"]), inv0)}
        h["units"] = {k: sum(h[k].values()) for k in ("shed", "carried", "tile")}
        res[who] = h
    sells = sim_sells(steps, me)
    res["sells_by_step"] = {str(k): v for k, v in sells.items() if k >= 600}
    res["last_sell_step"] = max(sells) if sells else None
    res["sells_total_units"] = sum(v[0] for v in sells.values())
    res["sells_day29_units"] = sum(v[0] for k, v in sells.items() if k >= 696)
    # our last action's market orders
    res["last_market_orders"] = (final[me].get("action") or {}).get("market")
    return res


if __name__ == "__main__":
    out = sys.argv[1]; files = sys.argv[2:]
    with Pool(int(os.environ.get("NPROC", "6"))) as pool, open(out, "w") as f:
        for r in pool.imap_unordered(process, files):
            f.write(json.dumps(r, separators=(",", ":")) + "\n"); f.flush()
            print(r["label"], r["gap"], r["me"]["units"], r["me"]["impact"], r["last_sell_step"], flush=True)
