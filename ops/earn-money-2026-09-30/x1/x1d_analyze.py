"""X1d analysis: per-game (mean of both self-play seats) counters, each checkpoint vs BC.
Usage: x1d_analyze.py DIR  (reads games_{BC,J2_final,hz4_10M,M_10M,M_20M}.json)
Screening rule (declared for this diagnostic, not a calibrated threshold): a counter 'moved' at a
checkpoint if |Welch t| >= 2.5 over n=8 game means vs BC AND |relative change| >= 20%
(for counters with a BC mean of 0, only the t rule applies)."""
import json, math, statistics as st, sys
D = sys.argv[1]; ORDER = sys.argv[2].split(',') if len(sys.argv) > 2 else ['BC', 'J2_final', 'hz4_10M', 'M_10M', 'M_20M']
E = ['starv','drought','ineff','cmds','pass','harvest','water','feed','sell_units','sell_cash','expiry_units','overflow','care_lost','fert_wasted','weeds','unsold_end','death_held','clipped','missed_growth','dug','redundant_fert','care_wasted','term_flags','seeds_unused','unused_land_cash','hire_wasted_cash','idle_hand_steps','mkt_unfilled','malformed','sale_shortfall_cash','floor_sale_units','buy_premium_cash']
def derived(s):
    td = s['tile_days']; g = lambda k: td.get(k, 0)
    q = s['quads_by_day']; first = next((d for d, v in enumerate(q) if v > q[0]), None)
    d = {k: s[k] for k in E}
    d.update(bank=s['bank'], deaths=s['starv'] + s['drought'], land_purchases=q[-1] - q[0],
             first_land_snapshot=first if first is not None else 30,
             strawberry_td=g('STRAWBERRY'), melon_td=g('MELON'), wheat_td=g('WHEAT'), carrot_td=g('CARROT'),
             empty_td=g('empty'), weed_td=g('WEED'), animal_days=g('COW') + g('SHEEP') + g('GOOSE'),
             cow_days=g('COW'), sheep_days=g('SHEEP'), goose_days=g('GOOSE'), pasture_td=g('PASTURE'), coop_td=g('COOP'),
             hands_mean=st.mean(s['hands_by_day']), money_d5=s['money_by_day'][5], money_d10=s['money_by_day'][10],
             money_d15=s['money_by_day'][15], money_d20=s['money_by_day'][20])
    return d
data = {}; meta = {}
for L in ORDER:
    j = json.load(open(f'{D}/games_{L}.json')); meta[L] = {k: j[k] for k in ('ckpt','n_envs','seed_state','eval_base_seed','torch_seed','wall_s','steps')}
    per_env = {}
    for s in j['seats']: per_env.setdefault(s['env'], []).append(derived(s))
    data[L] = {'games': [{k: (a[k] + b[k]) / 2 for k in a} for a, b in (per_env[e] for e in sorted(per_env))],
               'seats': [x for e in sorted(per_env) for x in per_env[e]],
               'seats_no_land': sum(1 for e in per_env.values() for x in e if x['land_purchases'] == 0)}
KEYS = ['bank','sell_cash','sell_units','harvest','deaths','land_purchases','first_land_snapshot','strawberry_td','animal_days','sheep_days','cow_days','goose_days','melon_td','wheat_td','carrot_td','empty_td','pasture_td','hands_mean','money_d5','money_d10','money_d15','money_d20'] + [k for k in E if k not in ('sell_cash','sell_units','harvest')]
def welch(a, b):
    va, vb = st.variance(a) / len(a), st.variance(b) / len(b)
    return (st.mean(b) - st.mean(a)) / math.sqrt(va + vb) if va + vb > 0 else (0.0 if st.mean(a) == st.mean(b) else math.inf)
res = {}
for k in KEYS:
    base = [g[k] for g in data['BC']['games']]; row = {'BC': st.mean(base)}
    for L in ORDER[1:]:
        x = [g[k] for g in data[L]['games']]; t = welch(base, x); m0 = st.mean(base); m1 = st.mean(x)
        rel = (m1 - m0) / abs(m0) if m0 else None
        moved = abs(t) >= 2.5 and (rel is None or abs(rel) >= 0.20)
        row[L] = (m1, t, rel, moved)
    res[k] = row
print('meta', json.dumps(meta))
print('seats never buying land:', {L: f"{data[L]['seats_no_land']}/16" for L in ORDER})
print('per-game bank sums (joint):', {L: [round(2 * g['bank']) for g in data[L]['games']] for L in ORDER})
hdr = f"{'counter':22s}{'BC':>10s}" + ''.join(f"{L:>26s}" for L in ORDER[1:]); print(hdr)
for k, row in res.items():
    s = f"{k:22s}{row['BC']:10.1f}"
    for L in ORDER[1:]:
        m1, t, rel, mv = row[L]; s += f"{m1:10.1f} t{t:+6.1f} {'' if rel is None else f'{rel*100:+5.0f}%':>6s}{'*' if mv else ' '}"
    print(s)
print('\nfirst checkpoint at which each counter moved (screen rule):')
first = {}
for k, row in res.items():
    f = next((L for L in ORDER[1:] if row[L][3]), None); first.setdefault(f, []).append((k, row[f][1] if f else 0))
for L in ORDER[1:] + [None]:
    if L in first: print(' ', L, sorted(first[L], key=lambda z: -abs(z[1])))
