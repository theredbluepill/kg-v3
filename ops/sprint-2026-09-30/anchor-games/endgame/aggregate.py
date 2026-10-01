"""Aggregate endgame.py jsonl files into per policy x opponent tables (markdown to stdout)."""
import glob, json, os, re, statistics as st
from collections import defaultdict

D = os.path.dirname(os.path.abspath(__file__))
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]


def opp_of(label):
    return re.sub(r"-s\d+-seat\d$", "", label).split("-", 1)[1] if True else None


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(D, "holdings", "*.jsonl"))):
        src = os.path.basename(f)[:-6]
        env, pol = src.split("-", 1)
        for line in open(f):
            r = json.loads(line)
            lab = r["label"]
            assert lab.startswith(pol + "-"), (lab, pol)
            opp = re.sub(r"-s\d+-seat\d$", "", lab[len(pol) + 1:])
            rows.append((env, pol, opp, r))
    bad_price = [r["label"] for *_, r in rows if not r["price_check"]]
    print(f"games: {len(rows)}; final-price formula mismatches: {len(bad_price)} {bad_price[:5]}\n")
    groups = defaultdict(list)
    for env, pol, opp, r in rows:
        groups[(env, pol, opp)].append(r)
    hdr = ("| env | policy | opponent | n | W-L | mean gap | our unsold units shed/carried/tile | our value shed+carried naive / impact | "
           "flips (shed / shed+carried / +tile) | opp unsold units shed/carried/tile | opp value shed+carried impact | "
           "our last SELL obs step (median, min) | our SELL units day 30 (obs 696-718) |")
    print(hdr); print("|" + "---|" * hdr.count("|")[:-1] if False else "|---" * (hdr.count("|") - 1) + "|")
    tot = defaultdict(lambda: defaultdict(float))
    for (env, pol, opp), rs in sorted(groups.items()):
        n = len(rs)
        w = sum(r["gap"] > 0 for r in rs)
        loss = [r for r in rs if r["gap"] < 0]
        f_shed = sum(r["me"]["impact"]["shed"] > -r["gap"] for r in loss)
        f_sc = sum(r["me"]["impact"]["shed_carried"] > -r["gap"] for r in loss)
        f_all = sum(r["me"]["impact"]["all"] > -r["gap"] for r in loss)
        mu = lambda k, who="me": st.mean(r[who]["units"][k] for r in rs)
        nv = st.mean(r["me"]["naive"]["shed"] + r["me"]["naive"]["carried"] for r in rs)
        iv = st.mean(r["me"]["impact"]["shed_carried"] for r in rs)
        oiv = st.mean(r["opp"]["impact"]["shed_carried"] for r in rs)
        ls = [r["last_sell_step"] for r in rs if r["last_sell_step"] is not None]
        d29 = st.mean(r["sells_day29_units"] for r in rs)
        print(f"| {env} | {pol} | {opp} | {n} | {w}-{len(loss)} | {st.mean(r['gap'] for r in rs):+,.0f} | "
              f"{mu('shed'):.1f} / {mu('carried'):.1f} / {mu('tile'):.1f} | {nv:,.0f} / {iv:,.0f} | "
              f"{f_shed} / {f_sc} / {f_all} of {len(loss)} | "
              f"{mu('shed','opp'):.1f} / {mu('carried','opp'):.1f} / {mu('tile','opp'):.1f} | {oiv:,.0f} | "
              f"{st.median(ls) if ls else '-'}, {min(ls) if ls else '-'} | {d29:.1f} |")
    # per-item breakdown of our shed+carried unsold, per policy (all envs pooled per env)
    print("\n## Our unsold shed+carried units by item (mean per game)\n")
    print("| env | policy | n | " + " | ".join(PRODUCTS) + " | animals in shed |")
    print("|---" * (len(PRODUCTS) + 4) + "|")
    byp = defaultdict(list)
    for env, pol, opp, r in rows:
        byp[(env, pol)].append(r)
    for (env, pol), rs in sorted(byp.items()):
        cells = []
        for it in PRODUCTS:
            cells.append(f"{st.mean(r['me']['shed'].get(it,0)+r['me']['carried'].get(it,0) for r in rs):.1f}")
        an = st.mean(sum(r["me"]["animals_in_shed"].values()) for r in rs)
        print(f"| {env} | {pol} | {len(rs)} | " + " | ".join(cells) + f" | {an:.1f} |")
    # timing: our SELL units per hour of the last day and the day before
    print("\n## Our committed SELL units per obs step, last 30 steps (mean per game)\n")
    steps = list(range(690, 719))
    print("| env | policy | " + " | ".join(str(s) for s in steps) + " |")
    print("|---" * (len(steps) + 2) + "|")
    for (env, pol), rs in sorted(byp.items()):
        cells = [f"{st.mean(r['sells_by_step'].get(str(s), [0,0])[0] for r in rs):.1f}" for s in steps]
        print(f"| {env} | {pol} | " + " | ".join(cells) + " |")
    # last-step orders of ours: share of games whose final action has any SELL
    print("\n## Final action (obs 718) market orders\n")
    for (env, pol), rs in sorted(byp.items()):
        has = sum(any(o and o[0] == "SELL" for o in (r["last_market_orders"] or [])) for r in rs)
        print(f"- {env} {pol}: {has}/{len(rs)} final actions contain a SELL")


if __name__ == "__main__":
    main()
