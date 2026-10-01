"""Carried products that a shed-routing rule could still deposit: units carried at obs
718-k by our actors whose Manhattan distance to the nearest shed access tile is <= k
(movement is never blocked, only bounded; DROP needs the actor on an access tile).
Valued at the final observed prices (naive). Usage: python3 carried_reach.py <glob>..."""
import gzip, json, glob, re, sys, statistics as st
from collections import defaultdict
ACCESS = [(4, 4), (5, 4), (4, 5), (5, 5)]
PR = {"WHEAT","CARROT","TOMATO","STRAWBERRY","MELON","EGG","MILK","WOOL","FERTILIZER"}
out = defaultdict(lambda: defaultdict(list))
for pat in sys.argv[1:]:
    for f in sorted(glob.glob(pat)):
        lab = f.split("/")[-1]; me = int(re.search(r"seat(\d)", lab).group(1))
        grp = f.split("/")[-3] if "kaggle" not in f else "kaggle"
        d = json.load(gzip.open(f)); s = d["steps"]; T = len(s) - 2
        prices = s[-1][me]["observation"]["market"]["prices"]
        for k in range(0, 6):
            o = s[T - k][me]["observation"]; fm = o["farms"][me]; inv = o["private"]["inventories"]
            pos = [fm["farmer"]] + list(fm["hands"])
            u = v = tot = 0
            for i, p in enumerate(pos):
                if i >= len(inv): continue
                goods = {a: b for a, b in inv[i].items() if a in PR and b > 0}
                n = sum(goods.values()); tot += n
                dist = min(abs(p[0] - x) + abs(p[1] - y) for x, y in ACCESS)
                if dist <= k:
                    u += n; v += sum(b * prices[a] for a, b in goods.items())
            out[grp][k].append((u, v, tot))
for grp, ks in out.items():
    print(grp, " | ".join(f"k={k}: reach {st.mean(x[0] for x in ks[k]):.1f}u/{st.mean(x[1] for x in ks[k]):,.0f} of {st.mean(x[2] for x in ks[k]):.1f}u" for k in sorted(ks)))
