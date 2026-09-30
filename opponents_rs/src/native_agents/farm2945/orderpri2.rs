//! 2945 Farm ORDERPRI2 layer: sales ordered by the rival's estimated sellable stock.
//! Source: agents/farm2945/main.py (`_or2_*` and the ORDERPRI2 `agent`).
//! The source's sell-now block is disabled (`_OR2_SN_K = 0`) and is not ported.
use super::race::shop_items;
use super::util::*;
use crate::native_agents::v43::common::*;
use crate::native_agents::v43::core::Core;
use crate::native_agents::v43::production::{market_price, quote_priority};
use serde_json::{Value, json};
use std::collections::BTreeMap;

const CAP: i64 = 30;
const SLOT_H: i64 = 6;
const SLOT_MARGIN: i64 = 50;
const SN_ITEMS: [&str; 5] = ["MILK", "STRAWBERRY", "WOOL", "MELON", "EGG"];
const ITEMS: [&str; 8] = [
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
];

/// A candidate slot sale: exposure, item, units and the (due step, units) debts it books.
type Slot = (i64, &'static str, i64, Vec<(i64, i64)>);

/// One rival tile: plant or animal, its product, planted/placed day and yield.
#[derive(Clone, Debug, PartialEq)]
struct Standing {
    plant: bool,
    item: String,
    born: i64,
    units: i64,
}

#[derive(Clone, Debug, Default)]
struct Previous {
    step: i64,
    tiles: BTreeMap<Pos, Standing>,
    inventory: Value,
    own: Value,
    prices: Value,
    shops: Value,
}

#[derive(Clone, Debug, Default)]
struct SeatState {
    step: i64,
    stock: Value,
    prev: Option<Previous>,
}

#[derive(Clone, Debug)]
pub struct OrderPri2 {
    states: [Option<SeatState>; 2],
    pub report: Value,
}
impl Default for OrderPri2 {
    fn default() -> Self {
        Self {
            states: [None, None],
            report: json!({"or2_reordered":0,"or2_changed_vs_v39":0,"or2_rival_harvest":0,
                "or2_rival_sold":0,"or2_errors":0}),
        }
    }
}

fn standing(farm: &Value) -> BTreeMap<Pos, Standing> {
    let mut out = BTreeMap::new();
    for (y, row) in array(&farm["tiles"]).iter().enumerate() {
        for (x, t) in array(row).iter().enumerate() {
            if !t.is_object() {
                continue;
            }
            let entry = if t["kind"] == "PLANT" && truth(&t["crop"]) {
                Standing {
                    plant: true,
                    item: text(&t["crop"]).to_owned(),
                    born: t.get("planted_day").map(int).unwrap_or(-1),
                    units: int(&t["yield_units"]),
                }
            } else {
                let item = match text(&t["animal"]) {
                    "GOOSE" => "EGG",
                    "COW" => "MILK",
                    "SHEEP" => "WOOL",
                    _ => continue,
                };
                Standing {
                    plant: false,
                    item: item.to_owned(),
                    born: t.get("placed_day").map(int).unwrap_or(-1),
                    units: int(&t["yield_units"]),
                }
            };
            out.insert((x as i64, y as i64), entry);
        }
    }
    out
}

/// `_or2_draw`: every shop product plus the daily town-centre unit.
fn draw(shops: &Value, step: i64, item: &str) -> i64 {
    let mut n = 0;
    if step % 4 == 0 {
        for shop in array(shops) {
            let products = shop_items(text(shop));
            if products.contains(&item) {
                n += if products.len() == 1 { 2 } else { 1 };
            }
        }
    }
    if step % 24 == 0 && ITEMS.contains(&item) {
        n += 1;
    }
    n
}

/// `_or2_exposure`: quote lost on `qty` units if a rival batch lands first.
fn exposure(obs: &Value, item: &str, qty: i64, batch: i64) -> i64 {
    if qty <= 0 || batch <= 0 || !PRODUCTS.contains(&item) {
        return 0;
    }
    let params = obs["market"].get("params");
    let inventory = int(&obs["market"]["inventory"][item]);
    (0..qty)
        .map(|j| {
            market_price(item, (inventory + j) as f64, params)
                - market_price(item, (inventory + batch + j) as f64, params)
        })
        .sum()
}

impl OrderPri2 {
    pub fn layer(&mut self, obs: &Value, action: Value, core: &mut Core) -> Value {
        let step = int(&obs["step"]);
        let who = seat(obs).min(1);
        if step == 0 || self.states[who].as_ref().is_none_or(|st| step <= st.step) {
            let mut stock = json!({});
            for item in ITEMS {
                stock[item] = json!(0);
            }
            self.states[who] = Some(SeatState {
                step: -1,
                stock,
                prev: None,
            });
            if step == 0
                && let Some(report) = self.report.as_object_mut()
            {
                for value in report.values_mut() {
                    *value = json!(0);
                }
            }
        }
        let st = self.states[who].as_mut().unwrap();
        let rival = &obs["farms"][1 - who];
        let tiles = standing(rival);
        let mut inventory = json!({});
        for item in ITEMS {
            inventory[item] = json!(int(&obs["market"]["inventory"][item]));
        }
        if let Some(prev) = st.prev.as_ref().filter(|p| p.step == step - 1) {
            for (pos, old) in &prev.tiles {
                if old.units <= 0 {
                    continue;
                }
                let new = tiles.get(pos);
                let ongoing = old.item == "TOMATO" || old.item == "STRAWBERRY";
                let got = if old.plant && !ongoing {
                    let cleared = match new {
                        None => tile(rival, *pos)["kind"] != "WEED",
                        Some(new) => new.born != old.born,
                    };
                    if cleared { old.units } else { 0 }
                } else if let Some(new) = new.filter(|n| {
                    n.plant == old.plant
                        && n.item == old.item
                        && n.born == old.born
                        && n.units < old.units
                }) {
                    if step % 24 != 0 {
                        old.units - new.units
                    } else if new.units == 0 {
                        old.units
                    } else {
                        0
                    }
                } else {
                    0
                };
                if got > 0 {
                    increment(&mut st.stock, &old.item, got);
                    inc(&mut self.report, "or2_rival_harvest", got);
                }
            }
            for item in ITEMS {
                if int(&prev.prices[item]) <= 1 {
                    continue;
                }
                let moved = int(&inventory[item]) - int(&prev.inventory[item])
                    + draw(&prev.shops, prev.step, item)
                    - int(&prev.own[item]);
                if moved > 0 {
                    st.stock[item] = json!((int(&st.stock[item]) - moved).max(0));
                    inc(&mut self.report, "or2_rival_sold", moved);
                }
            }
        }
        let mut result = action;
        let mut market = orders(&result);
        let proj = Core::projected_shed(&result, &View::new(obs));
        let batch = |item: &str| (if core.cha22 { 24 } else { CAP }).min(int(&st.stock[item]));
        let key = who.to_string();
        if (288..694).contains(&step) && market.len() >= MAX_ORDERS && truth(&core.players[&key]) {
            let native = &mut core.players[&key];
            if native["sell_state"]["r36_debts"].is_null() {
                native["sell_state"]["r36_debts"] = json!({});
            }
            let route = int(&native["route"]);
            let debts = &mut native["sell_state"]["r36_debts"];
            let sells: Vec<Value> = market
                .iter()
                .filter(|o| array(o).len() >= 3 && o[0] == "SELL")
                .cloned()
                .collect();
            let bought = |item: &str| {
                market.iter().any(|o| {
                    array(o).len() >= 2
                        && (o[0] == "BUY_PRODUCT" || o[0] == "BUY_ANIMAL")
                        && o[1] == item
                })
            };
            let mut best: Option<Slot> = None;
            for item in SN_ITEMS {
                if sells.iter().any(|o| o[1] == item) || bought(item) {
                    continue;
                }
                let avail = int(&proj[item]);
                if avail <= 0 || int(&obs["market"]["prices"][item]) < 2 {
                    continue;
                }
                let (mut plan, mut take) = (vec![], 0);
                for t in step + 1..=694.min(step + SLOT_H) {
                    if take >= avail {
                        break;
                    }
                    let planned: i64 = orders(&tape(if t >= 648 { 2 } else { route }, t))
                        .iter()
                        .filter(|o| is_order(o, "SELL", item))
                        .map(|o| int(&o[2]).max(0))
                        .sum();
                    let q = (planned - int(&debts[t.to_string()][item])).min(avail - take);
                    if q > 0 {
                        plan.push((t, q));
                        take += q;
                    }
                }
                if take <= 0 {
                    continue;
                }
                let value = exposure(obs, item, take, batch(item).max(1));
                if best.as_ref().is_none_or(|b| value > b.0) {
                    best = Some((value, item, take, plan));
                }
            }
            let sale_value = |o: &Value| {
                let item = text(&o[1]);
                let q = int(&o[2]).max(0).min(int(&proj[item]).max(0));
                exposure(obs, item, q, batch(item).max(1))
            };
            if let Some((value, item, take, plan)) = best
                && let Some(weakest) = sells.iter().min_by_key(|o| sale_value(o))
                && value
                    > sale_value(weakest)
                        + if core.cha22 {
                            12
                        } else if core.metav4 {
                            20
                        } else {
                            SLOT_MARGIN
                        }
                && (!matches!(text(&weakest[1]), "WHEAT" | "FERTILIZER") || int(&weakest[2]) <= 2)
            {
                // Drop the weakest sale; give its booked debts back, nearest due first.
                let dropped = text(&weakest[1]);
                let mut refund = int(&weakest[2]).max(0);
                for t in step + 1..step + 49 {
                    if refund <= 0 {
                        break;
                    }
                    let owed = int(&debts[t.to_string()][dropped]);
                    let back = owed.min(refund);
                    if back > 0 {
                        debts[t.to_string()][dropped] = json!(owed - back);
                        refund -= back;
                    }
                }
                if let Some(at) = market.iter().position(|o| o == weakest) {
                    market.remove(at);
                }
                market.push(json!(["SELL", item, take]));
                for (t, q) in plan {
                    if debts[t.to_string()].is_null() {
                        debts[t.to_string()] = json!({});
                    }
                    increment(&mut debts[t.to_string()], item, q);
                }
                inc(&mut self.report, "or2_slot_swaps", 1);
                set_orders(&mut result, market.clone());
            }
        }
        let mut bought: Vec<String> = vec![];
        let (mut movable, mut fixed) = (vec![], vec![]);
        for (index, o) in market.iter().enumerate() {
            if array(o).len() >= 2 && (o[0] == "BUY_PRODUCT" || o[0] == "BUY_ANIMAL") {
                bought.push(text(&o[1]).to_owned());
            }
            let sale = array(o).len() >= 3
                && o[0] == "SELL"
                && int(&o[2]) > 0
                && !bought.iter().any(|b| o[1] == b.as_str());
            if sale {
                movable.push((index, o.clone()));
            } else {
                fixed.push(o.clone());
            }
        }
        if !movable.is_empty() && step >= 1 {
            let score = |entry: &(usize, Value)| {
                let item = text(&entry.1[1]);
                let qty = int(&entry.1[2]).min(int(&proj[item]).max(0));
                (
                    -exposure(obs, item, qty, batch(item)),
                    -quote_priority(obs, &entry.1, &proj),
                    entry.0,
                )
            };
            let mut scored = movable.clone();
            scored.sort_by_key(score);
            let mut new: Vec<Value> = scored.iter().map(|e| e.1.clone()).collect();
            new.extend(fixed);
            if new != market {
                let mut legacy = movable;
                legacy.sort_by_key(|e| (-quote_priority(obs, &e.1, &proj), e.0));
                if legacy.iter().map(|e| &e.1).ne(scored.iter().map(|e| &e.1)) {
                    inc(&mut self.report, "or2_changed_vs_v39", 1);
                }
                inc(&mut self.report, "or2_reordered", 1);
                set_orders(&mut result, new.clone());
                market = new;
            }
        }
        let mut left = proj.clone();
        let mut own = json!({});
        for o in market.iter().take(MAX_ORDERS) {
            let item = text(&o[1]);
            if array(o).len() >= 3 && o[0] == "SELL" && ITEMS.contains(&item) {
                let got = int(&o[2]).max(0).min(int(&left[item]).max(0));
                increment(&mut left, item, -got);
                increment(&mut own, item, got);
            }
        }
        st.prev = Some(Previous {
            step,
            tiles,
            inventory,
            own,
            prices: obs["market"]["prices"].clone(),
            shops: obs["town"]["unlocked_shops"].clone(),
        });
        st.step = step;
        result
    }

    /// Oracle view: the rival stock estimate and last recorded own sales per seat.
    pub fn states(&self) -> Value {
        let mut out = json!({});
        for (who, st) in self.states.iter().enumerate() {
            if let Some(st) = st {
                out[who.to_string()] = json!({
                    "step": st.step,
                    "stock": st.stock,
                    "own": st.prev.as_ref().map(|p| p.own.clone()),
                });
            }
        }
        out
    }
}
