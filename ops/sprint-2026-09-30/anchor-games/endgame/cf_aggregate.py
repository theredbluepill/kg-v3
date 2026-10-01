"""Aggregate cf/*.jsonl (counterfactual liquidation) into a markdown table."""
import glob, json, os, re, statistics as st
from collections import defaultdict
D = os.path.dirname(os.path.abspath(__file__))
g = defaultdict(list); nval = nok = 0
for f in sorted(glob.glob(os.path.join(D, "cf", "*.jsonl"))):
    env, pol = os.path.basename(f)[3:-6].split("-", 1)
    for line in open(f):
        r = json.loads(line)
        opp = re.sub(r"-s\d+-seat\d$", "", r["label"][len(pol) + 1:])
        if env == "kaggle":
            opp = "self-play validation" if "115836977" in r["label"] else "ladder"
        g[(env, pol, opp)].append(r)
        a = r["actual_final"]
        if a["validatable"]:
            nval += 1; nok += a["cash"] == a["observed_money_delta"]
print(f"market-model check on final steps with SELL-only orders: {nok}/{nval} exact (sim cash == observed money change)\n")
print("| env | policy | opponent | n | W-L | mean gap | final-step SELL-all gain mean (min/max) | + DROP on access tile gain | carried off access at obs 718 (units) | flips final SELL-all | window 5 / 12 / 23 gain (approx) | final-step non-SELL orders (games) |")
print("|---" * 12 + "|")
for (env, pol, opp), rs in sorted(g.items()):
    n = len(rs); w = sum(r["gap"] > 0 for r in rs); l = sum(r["gap"] < 0 for r in rs)
    fs = [r["final_sell"]["gain"] for r in rs]; fd = [r["final_sell_drop"]["gain"] for r in rs]
    flips = sum(r["final_sell_drop"]["flip"] for r in rs)
    co = st.mean(r["carried_off_access_final"] for r in rs)
    wins = " / ".join(f"{st.mean(r[f'window_{K}']['gain'] for r in rs):,.0f}" for K in (5, 12, 23))
    ns = sum(bool(r["final_nonsell_orders"]) for r in rs)
    print(f"| {env} | {pol} | {opp} | {n} | {w}-{l} | {st.mean(r['gap'] for r in rs):+,.0f} | {st.mean(fs):,.0f} ({min(fs):,}/{max(fs):,}) | {st.mean(fd):,.0f} | {co:.1f} | {flips} of {l} | {wins} | {ns} |")
