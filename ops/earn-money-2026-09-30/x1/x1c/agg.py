"""Aggregate x1c replay classification (classify.jsonl lines) per label."""
import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1])]
out = {}
for label in sorted({r["label"] for r in rows}):
    R = [r for r in rows if r["label"] == label]
    seats = 2 * len(R)
    parity = all(r["banks_replay"] == r["banks_driver"] for r in R)
    banks = [b for r in R for b in r["banks_replay"]]
    cls = collections.Counter(); clsverb = collections.defaultdict(collections.Counter)
    for r in R:
        for c, vs in r["classes"].items():
            for v, n in vs.items():
                cls[c] += n; clsverb[c][v] += n
    pairs = collections.Counter()
    for r in R: pairs.update(r["collision_pairs"])
    lost = collections.Counter()
    for r in R: lost.update(r["plants_lost"])
    eff_pl = collections.Counter(); prod = collections.Counter(); su = collections.Counter(); sc = collections.Counter()
    eff_cmd = 0; vi = collections.Counter(); ve = collections.Counter()
    for r in R:
        for p in (0, 1):
            eff_pl.update(r["eff_plants"][p]); s = r[f"seat{p}"]
            prod.update(s["produced"]); su.update(s["sell_units"]); sc.update(s["sell_cash"])
            eff_cmd += s["slot_effective_total"]; vi.update(s["verb_ineffective"]); ve.update(s["verb_effective"])
    total = sum(r["total_ineffective"] for r in R)
    classified = sum(cls.values())
    bank_mean = sum(banks) / seats
    val_per_eff_cmd = sum(banks) / eff_cmd
    plant_val = {}
    for crop, n in lost.items():
        ypp = prod[crop] / eff_pl[crop] if eff_pl[crop] else float("nan")
        price = sc[crop] / su[crop] if su[crop] else float("nan")
        sold_frac = su[crop] / prod[crop] if prod[crop] else float("nan")
        plant_val[crop] = {"plants_lost": n, "plants_lost_per_seat": n / seats, "eff_plants": eff_pl[crop],
                           "yield_units_per_eff_plant": ypp, "harvest_units_at_stake": n * ypp,
                           "sold_fraction": sold_frac, "mean_sale_price": price,
                           "sales_cash_upper_bound_per_seat": n * ypp * sold_frac * price / seats}
    out[label] = {
        "games": len(R), "seats": seats, "replay_bank_parity": parity, "seeds": [r["seed"] for r in R],
        "bank_mean_per_seat": bank_mean, "joint_bank_mean": 2 * bank_mean,
        "ineffective_total": total, "ineffective_per_seat": total / seats, "classified": classified,
        "slot15_unidentified": sum(r["slot15_unidentified"] for r in R),
        "class_per_seat": {c: n / seats for c, n in cls.most_common()},
        "class_share": {c: n / classified for c, n in cls.most_common()},
        "class_by_verb_per_seat": {c: {v: n / seats for v, n in vs.most_common()} for c, vs in clsverb.items()},
        "verb_ineffective_per_seat": {v: n / seats for v, n in vi.most_common()},
        "verb_effective_per_seat": {v: n / seats for v, n in ve.most_common()},
        "top_collision_pairs": pairs.most_common(12),
        "plant_block_events_with_seeds": sum(r["plant_block_events_with_seeds"] for r in R),
        "plant_loss": plant_val,
        "alone_harvest_units_of_interacting_harvests": sum(r["alone_harvest_units_of_interacting_harvests"] for r in R),
        "effective_cmds_per_seat": eff_cmd / seats, "bank_per_effective_cmd": val_per_eff_cmd,
        "labor_value_avg_product_per_seat": {c: n / seats * val_per_eff_cmd for c, n in cls.most_common()},
        "produced_per_seat": {k: v / seats for k, v in prod.items() if v}, "sell_units_per_seat": {k: v / seats for k, v in su.items() if v},
        "sell_cash_per_seat": {k: v / seats for k, v in sc.items() if v},
        "harvest_eff_per_seat": ve["HARVEST"] / seats,
        "per_day_ineffective_mean_seat": [sum(r["per_day"][d][0] + r["per_day"][d][1] for r in R) / seats for d in range(31)],
    }
print(json.dumps(out, indent=1))
