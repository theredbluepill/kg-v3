//! X1c: replay dumped self-play games through the engine and classify every ineffective unit command by
//! counterfactual clones of the start-of-step Game (units act before the market, in farmer-then-hand order).
use kaggriculture_engine::{ATTRIB_FIELDS, ATTRIB_VERBS, Config, Game, PRODUCTS};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader};

const A_PROD: usize = 56;
const A_VE: usize = 65;
const A_VI: usize = 83;
const A_SE: usize = 101;
const A_SI: usize = 117;
type Row = [[u64; ATTRIB_FIELDS]; 2];

fn attrib(g: &Game) -> Row { *g.attrib_counters().expect("two seats") }
fn pass() -> Value { json!(["PASS"]) }
fn op(v: &Value) -> &str { v.as_array().and_then(|a| a.first()).and_then(Value::as_str).unwrap_or("PASS") }
fn unit_cmd(prog: &Value, u: usize) -> Value {
    if u == 0 { prog.get("farmer").cloned().unwrap_or_else(pass) }
    else { prog.get("hands").and_then(|h| h.get(u - 1)).cloned().unwrap_or_else(pass) }
}
/// Seat program with only the listed units keeping their commands; market keys kept.
fn only(prog: &Value, keep: &[usize], n_units: usize) -> Value {
    let mut p = prog.clone();
    let obj = p.as_object_mut().unwrap();
    obj.insert("farmer".into(), if keep.contains(&0) { unit_cmd(prog, 0) } else { pass() });
    let n_h = prog.get("hands").and_then(Value::as_array).map(|a| a.len()).unwrap_or(0);
    let hands: Vec<Value> = (1..=n_h).map(|u| if keep.contains(&u) { unit_cmd(prog, u) } else { pass() }).collect();
    let _ = n_units;
    obj.insert("hands".into(), Value::Array(hands));
    p
}
fn run(pre: &Game, joint: &[Value]) -> Game { let mut g = pre.clone(); g.step_with_market_metrics(joint).expect("step"); g }
fn slot(u: usize) -> usize { u.min(15) }
fn ineff(a0: &Row, a1: &Row, p: usize, u: usize) -> u64 { a1[p][A_SI + slot(u)] - a0[p][A_SI + slot(u)] }
fn pos(snap: &Value, p: usize, u: usize) -> Value {
    let f = &snap["public"]["farms"][p];
    if u == 0 { f["farmer"].clone() } else { f["hands"][u - 1].clone() }
}
fn is_move(o: &str) -> bool { matches!(o, "NORTH" | "SOUTH" | "EAST" | "WEST") }

fn main() {
    let path = std::env::args().nth(1).expect("jsonl");
    let reader = BufReader::new(std::fs::File::open(&path).unwrap());
    for line in reader.lines() {
        let rec: Value = serde_json::from_str(&line.unwrap()).unwrap();
        let config: Config = serde_json::from_value(rec["config"].clone()).expect("config");
        let mut game = Game::new_with_seed_decimal(config, rec["seed"].as_str().unwrap(), 2).unwrap();
        // counters: class -> verb -> count ; plus plants lost per crop; eff plants per crop
        let mut cls: BTreeMap<String, BTreeMap<String, u64>> = BTreeMap::new();
        let mut plants_lost: BTreeMap<String, u64> = BTreeMap::new();
        let mut plant_block_events = 0u64;
        let mut eff_plants: [BTreeMap<String, u64>; 2] = [BTreeMap::new(), BTreeMap::new()];
        let mut pairs: BTreeMap<String, u64> = BTreeMap::new();
        let mut alone_harvest_units = 0u64;
        let mut unidentified = 0u64;
        let mut total_ineff = 0u64;
        let mut per_seat_ineff = [0u64; 2];
        let mut per_day: Vec<[u64; 2]> = vec![[0; 2]; 31];
        let steps = rec["steps"].as_array().unwrap();
        for (t, joint) in steps.iter().enumerate() {
            let joint: Vec<Value> = joint.as_array().unwrap().clone();
            let pre = game.clone();
            let snap = serde_json::to_value(pre.snapshot()).unwrap();
            let a0 = attrib(&pre);
            game.step_with_market_metrics(&joint).expect("replay step");
            let a1 = attrib(&game);
            for p in 0..2 {
                let prog = &joint[p];
                let n_hands = snap["public"]["farms"][p]["hands"].as_array().map(|a| a.len()).unwrap_or(0);
                let n_units = 1 + n_hands;
                let seat_ineff: u64 = (0..16).map(|s| a1[p][A_SI + s] - a0[p][A_SI + s]).sum();
                total_ineff += seat_ineff;
                per_seat_ineff[p] += seat_ineff;
                per_day[(t / 24).min(30)][p] += seat_ineff;
                let slot15 = a1[p][A_SI + 15] - a0[p][A_SI + 15];
                if slot15 > 0 { unidentified += slot15; }
                // effective PLANTs per crop (units < 15 exactly)
                for u in 0..n_units.min(15) {
                    let c = unit_cmd(prog, u);
                    if op(&c) == "PLANT" && ineff(&a0, &a1, p, u) == 0 {
                        let crop = c[1].as_str().unwrap_or("?").to_string();
                        *eff_plants[p].entry(crop).or_default() += 1;
                    }
                }
                if seat_ineff == 0 { continue; }
                // PLANT demand vs seeds at the start of the step
                let mut demand: BTreeMap<String, Vec<usize>> = BTreeMap::new();
                for u in 0..n_units {
                    let c = unit_cmd(prog, u);
                    if op(&c) == "PLANT" { if let Some(cr) = c.get(1).and_then(Value::as_str) { demand.entry(cr.into()).or_default().push(u); } }
                }
                let seeds = |cr: &str| snap["privates"][p]["seeds"][cr].as_i64().unwrap_or(0);
                let blocked: Vec<String> = demand.iter().filter(|(cr, us)| us.len() as i64 > seeds(cr)).map(|(c, _)| c.clone()).collect();
                for cr in &blocked {
                    let s = seeds(cr).max(0) as usize;
                    if s == 0 { continue; }
                    plant_block_events += 1;
                    // trimmed: keep the first s PLANT cr commands, PASS the other PLANT cr commands
                    let drop: Vec<usize> = demand[cr][s..].to_vec();
                    let keep: Vec<usize> = (0..n_units).filter(|u| !drop.contains(u)).collect();
                    let mut j2 = joint.clone(); j2[p] = only(prog, &keep, n_units);
                    let g2 = run(&pre, &j2);
                    let a2 = attrib(&g2);
                    let vi = ATTRIB_VERBS.iter().position(|v| *v == "PLANT").unwrap();
                    let eff_trim = a2[p][A_VE + vi] - a0[p][A_VE + vi];
                    let eff_act = a1[p][A_VE + vi] - a0[p][A_VE + vi];
                    *plants_lost.entry(cr.clone()).or_default() += eff_trim.saturating_sub(eff_act);
                }
                for u in 0..n_units {
                    let c = unit_cmd(prog, u);
                    let o = op(&c).to_string();
                    if o == "PASS" { continue; }
                    let bad = if u < 15 { ineff(&a0, &a1, p, u) > 0 } else { false };
                    if !bad { continue; }
                    let class: String;
                    if o == "PLANT" && c.get(1).and_then(Value::as_str).is_some_and(|cr| blocked.iter().any(|b| b == cr)) {
                        let cr = c[1].as_str().unwrap();
                        class = if seeds(cr) > 0 { "plant_overdemand_pass".into() } else { "plant_no_seeds_pass".into() };
                    } else {
                        let mut ja = joint.clone(); ja[p] = only(prog, &[u], n_units);
                        let ga = run(&pre, &ja);
                        let aa = attrib(&ga);
                        if ineff(&a0, &aa, p, u) > 0 {
                            class = "predictable_alone".into();
                        } else {
                            if o == "HARVEST" { alone_harvest_units += (0..9).map(|k| aa[p][A_PROD + k] - a0[p][A_PROD + k]).sum::<u64>(); }
                            let snap_a = serde_json::to_value(ga.snapshot()).unwrap();
                            let tile_u = if is_move(&o) { pos(&snap_a, p, u) } else { pos(&snap, p, u) };
                            let mut found = None;
                            for j in 0..u {
                                let cj = unit_cmd(prog, j);
                                if op(&cj) == "PASS" { continue; }
                                let mut jp = joint.clone(); jp[p] = only(prog, &[j, u], n_units);
                                let gp = run(&pre, &jp);
                                let ap = attrib(&gp);
                                if ineff(&a0, &ap, p, u) > 0 && ineff(&a0, &ap, p, j) == 0 {
                                    let snap_p = serde_json::to_value(gp.snapshot()).unwrap();
                                    let tile_j = if is_move(op(&cj)) { pos(&snap_p, p, j) } else { pos(&snap, p, j) };
                                    found = Some((op(&cj).to_string(), tile_j == tile_u));
                                    break;
                                }
                            }
                            class = match found {
                                Some((vj, same)) => {
                                    *pairs.entry(format!("{vj}->{o} same_tile={same}")).or_default() += 1;
                                    if same { "collision_same_target".into() } else { "collision_other_resource".into() }
                                }
                                None => "interaction_multi_unit".into(),
                            };
                        }
                    }
                    *cls.entry(class).or_default().entry(o).or_default() += 1;
                }
            }
        }
        let fin = attrib(&game);
        let banks = game.terminal_banks().map(|b| b.to_vec());
        let seat = |p: usize| json!({
            "verb_ineffective": ATTRIB_VERBS.iter().enumerate().map(|(i, v)| (v.to_string(), fin[p][A_VI + i])).filter(|(_, n)| *n > 0).collect::<BTreeMap<_, _>>(),
            "verb_effective": ATTRIB_VERBS.iter().enumerate().map(|(i, v)| (v.to_string(), fin[p][A_VE + i])).filter(|(_, n)| *n > 0).collect::<BTreeMap<_, _>>(),
            "slot_effective_total": (0..16).map(|s| fin[p][A_SE + s]).sum::<u64>(),
            "produced": PRODUCTS.iter().enumerate().map(|(i, v)| (v.to_string(), fin[p][A_PROD + i])).collect::<BTreeMap<_, _>>(),
            "sell_units": PRODUCTS.iter().enumerate().map(|(i, v)| (v.to_string(), fin[p][i])).collect::<BTreeMap<_, _>>(),
            "sell_cash": PRODUCTS.iter().enumerate().map(|(i, v)| (v.to_string(), fin[p][9 + i])).collect::<BTreeMap<_, _>>(),
        });
        println!("{}", json!({
            "label": rec["label"], "env": rec["env"], "seed": rec["seed"], "steps": steps.len(),
            "banks_replay": banks, "banks_driver": rec["banks"],
            "total_ineffective": total_ineff, "per_seat_ineffective": per_seat_ineff, "slot15_unidentified": unidentified,
            "classes": cls, "collision_pairs": pairs, "plants_lost": plants_lost, "plant_block_events_with_seeds": plant_block_events,
            "alone_harvest_units_of_interacting_harvests": alone_harvest_units,
            "eff_plants": eff_plants, "per_day": per_day, "seat0": seat(0), "seat1": seat(1),
        }));
    }
}
