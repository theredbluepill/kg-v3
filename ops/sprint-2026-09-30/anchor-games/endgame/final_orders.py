"""Decision diagnostics for our last-day SELL orders (obs 696-718): per order, compare the
requested quantity with the pre-market shed of that item (market-analysis inference)."""
import gzip, json, glob, re, sys, os, statistics as st
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cf import step_ctx, P, PRODUCTS
for pat in sys.argv[1:]:
    qty = Counter(); zero = full = partial = total = 0; slots = []; unordered_final = Counter(); final_units = 0
    ordered_final_units = 0
    files = sorted(glob.glob(pat))
    for f in files:
        me = int(re.search(r"seat(\d)", f.split('/')[-1]).group(1))
        d = json.load(gzip.open(f)); s = d["steps"]; T = len(s) - 2
        for t in range(696, T + 1):
            o_prev, o_cur, acts, sheds, money, queues = step_ctx(s, t)
            sh = dict(sheds[me]); orders = [o for o in queues[me] if o]
            slots.append(len(acts[me].get("market") or []))
            for o in orders:
                if o[0] != "SELL": continue
                total += 1; qty[o[2]] += 1
                have = sh.get(o[1], 0)
                if have <= 0: zero += 1
                elif o[2] >= have: full += 1
                else: partial += 1
                sh[o[1]] = max(0, have - o[2])
            if t == T:
                for it in PRODUCTS:
                    if sh.get(it, 0) > 0: unordered_final[it] += sh[it]
                final_units += sum(v for k, v in sheds[me].items() if k in P and v > 0)
    n = len(files)
    print(f"{pat}: games {n}; last-day SELL orders {total}: shed empty {zero} ({zero/total:.0%}), qty>=shed {full} ({full/total:.0%}), partial {partial} ({partial/total:.0%})")
    print("  top requested quantities:", qty.most_common(8))
    print(f"  market slots used per last-day step: mean {st.mean(slots):.2f}, max {max(slots)} (limit 10)")
    print(f"  final step: shed products pre-market {final_units/n:.1f}/game; left unordered by item (per game):",
          {k: round(v / n, 1) for k, v in unordered_final.most_common()})
