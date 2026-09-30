//! 2945 Farm v9 RACE horizon and v9/2 PREDICT sale forecasting.
//! Source: agents/farm2945/main.py (`_v9_race_update`, RACE `agent`,
//! `_v92_p_*`, `_v92_predict`). PREDICT2 reads its stream library from the
//! `V92_SELL_LIB` environment file, which the published agent never sets: it has
//! no streams, changes no action and is not ported.
use super::util::*;
use crate::native_agents::v43::common::*;
use crate::native_agents::v43::core::Core;
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap};
use std::sync::OnceLock;

pub const RACE_DEFAULT: i64 = 40;
pub const RACE_MAX: i64 = 48;
pub const RACE_MARGIN: i64 = 12;
const RACE_GAP: i64 = 3;
const RACE_WINDOW: i64 = 30;
pub const RACE_ITEMS: [&str; 7] = [
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
];

pub fn shop_items(shop: &str) -> &'static [&'static str] {
    match shop {
        "BAKERY" => &["EGG", "WHEAT"],
        "PIZZA_SHOP" => &["MILK", "TOMATO", "WHEAT"],
        "BRUNCH_SPOT" => &["EGG", "WHEAT", "STRAWBERRY"],
        "YARN_STORE" => &["WOOL"],
        "ICE_CREAM_SHOP" => &["STRAWBERRY", "MILK", "WHEAT"],
        "PET_CAFE" => &["CARROT"],
        "SMOOTHIE_SHOP" => &["STRAWBERRY", "MILK"],
        "FARMERS_MARKET" => &["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
        _ => &[],
    }
}

/// `_v9_town_draw`: units each shop and the town centre remove after `step`'s market.
pub fn town_draw(shops: &Value, step: i64, item: &str) -> i64 {
    let mut draw = 0;
    if step % 4 == 0 {
        for shop in array(shops) {
            let items = shop_items(text(shop));
            if items.contains(&item) {
                draw += if items.len() == 1 { 2 } else { 1 };
            }
        }
    }
    if step % 24 == 0 {
        draw += 1;
    }
    draw
}

#[derive(Clone, Debug)]
pub struct Race {
    pub states: Value,
    pub report: Value,
}
impl Default for Race {
    fn default() -> Self {
        Self {
            states: json!({}),
            report: json!({"rival_sales":0,"leads":0,"race_errors":0}),
        }
    }
}

impl Race {
    /// RACE before its parent: recover rival sales and publish `_V9_ITEM_HZ`.
    pub fn before(&mut self, obs: &Value, core: &Core, item_hz: &mut Value) {
        let step = int(&obs["step"]);
        let key = seat(obs).to_string();
        if self.states[&key].is_null() || step <= int(&self.states[&key]["step"]) {
            self.states[&key] = json!({"step":-1,"lead":-RACE_MARGIN,"prev":null});
            if step == 0 {
                for k in ["rival_sales", "leads", "race_errors"] {
                    self.report[k] = json!(0);
                }
            }
        }
        self.states[&key]["step"] = json!(step);
        self.update(obs, core);
        let lead = int(&self.states[&key]["lead"]);
        let default_horizon = if core.cha22 {
            44
        } else if core.v56 {
            41
        } else {
            RACE_DEFAULT
        };
        item_hz[&key] = json!(RACE_MAX.min(default_horizon.max(lead + RACE_MARGIN)));
    }

    fn update(&mut self, obs: &Value, core: &Core) {
        let key = seat(obs).to_string();
        let prev = self.states[&key]["prev"].clone();
        let step = int(&obs["step"]);
        if !truth(&prev) || int(&prev["step"]) != step - 1 {
            return;
        }
        let Some(route) = native_route(core, obs) else {
            return;
        };
        let t = int(&prev["step"]);
        let inventory = &obs["market"]["inventory"];
        for item in RACE_ITEMS {
            if int(&prev["prices"][item]) <= 3 {
                continue;
            }
            let sold = int(&inventory[item]) - int(&prev["inventory"][item])
                + town_draw(&prev["shops"], t, item)
                - int(&prev["own"][item]);
            if sold < 2 {
                continue;
            }
            inc(&mut self.report, "rival_sales", 1);
            if int(&prev["left"][item]) <= 0 {
                continue;
            }
            let planned: Vec<i64> = ((t - RACE_WINDOW).max(0)
                ..tape_len(route).min(t + RACE_WINDOW + 1))
                .filter(|s| {
                    orders(&tape(route, *s))
                        .iter()
                        .any(|o| is_order(o, "SELL", item) && int(&o[2]) > 0)
                })
                .collect();
            let after = planned.iter().find(|s| **s >= t);
            let before = planned.iter().rfind(|s| **s < t);
            let Some(after) = after else {
                continue;
            };
            if before.is_some_and(|b| t - b < RACE_GAP) {
                continue;
            }
            let lead = int(&self.states[&key]["lead"]).max(after - t);
            self.states[&key]["lead"] = json!(lead);
            inc(&mut self.report, "leads", 1);
        }
    }

    /// RACE after its parent: remember this turn's book and our executed sales.
    pub fn after(&mut self, obs: &Value, action: &Value) {
        let key = seat(obs).to_string();
        let stock = Core::projected_shed(action, &View::new(obs));
        let mut own = json!({});
        for o in orders(action) {
            if array(&o).len() >= 3 && o[0] == "SELL" && RACE_ITEMS.contains(&text(&o[1])) {
                let item = text(&o[1]);
                let n = int(&o[2])
                    .max(0)
                    .min((int(&stock[item]) - int(&own[item])).max(0));
                inc(&mut own, item, n);
            }
        }
        let mut left = json!({});
        for item in RACE_ITEMS {
            left[item] = json!(int(&stock[item]) - int(&own[item]));
        }
        self.states[&key]["prev"] = json!({
            "step": int(&obs["step"]),
            "inventory": obs["market"]["inventory"],
            "prices": obs["market"]["prices"],
            "own": own,
            "left": left,
            "shops": obs["town"]["unlocked_shops"],
        });
    }
}

const SHOP_NAMES: [&str; 8] = [
    "BAKERY",
    "BRUNCH_SPOT",
    "FARMERS_MARKET",
    "ICE_CREAM_SHOP",
    "PET_CAFE",
    "PIZZA_SHOP",
    "SMOOTHIE_SHOP",
    "YARN_STORE",
];
/// Byte lengths of the 64 shop-pair blocks (`_V92_P_INDEX`), in `SHOP_NAMES` order.
const LENGTHS: [usize; 64] = [
    25399, 14352, 21611, 35153, 35952, 11363, 25921, 9203, 19344, 13054, 25941, 27205, 20309,
    40503, 19384, 21949, 19598, 13148, 17134, 18046, 16318, 25000, 31290, 29200, 31807, 18968,
    16748, 24580, 14426, 30363, 28672, 15474, 17452, 17531, 16214, 13566, 15563, 21994, 12636,
    25314, 19514, 22772, 15989, 15367, 28435, 19570, 21771, 27566, 13436, 17469, 18125, 33503,
    25979, 23209, 17240, 12137, 32087, 12673, 21374, 13958, 23222, 21525, 13235, 24473,
];
const PREDICT_ITEMS: [&str; 5] = ["MILK", "WOOL", "STRAWBERRY", "EGG", "MELON"];
const PREDICT_USE: usize = 3; // MILK, WOOL, STRAWBERRY
const PREDICT_H: i64 = 48;
const PREDICT_K: i64 = 4;
const PREDICT_EVERY: i64 = 3;

type Stream = HashMap<(i64, u8), i64>;

/// Decompressed `_V92_P_BLOB`: recorded rival sale streams per first-two-shop pair.
fn library(metav4: bool) -> &'static Vec<Vec<Stream>> {
    static LIBRARY: OnceLock<Vec<Vec<Stream>>> = OnceLock::new();
    static METAV4_LIBRARY: OnceLock<Vec<Vec<Stream>>> = OnceLock::new();
    let cache = if metav4 { &METAV4_LIBRARY } else { &LIBRARY };
    cache.get_or_init(|| {
        let raw: &[u8] = if metav4 {
            include_bytes!("../../../fixtures/metav4-sell-library.bin")
        } else {
            include_bytes!("../../../fixtures/farm2945-sell-library.bin")
        };
        let lengths = if metav4 {
            &crate::native_agents::metav4::library_lengths::LENGTHS
        } else {
            &LENGTHS
        };
        let mut start = 0;
        lengths
            .iter()
            .map(|length| {
                let mut pos = start;
                start += length;
                let word = |pos: usize| raw[pos] as usize | (raw[pos + 1] as usize) << 8;
                let streams = word(pos);
                pos += 2;
                (0..streams)
                    .map(|_| {
                        let events = word(pos);
                        pos += 2;
                        let mut stream = Stream::new();
                        let mut last = 0;
                        for _ in 0..events {
                            let delta = raw[pos];
                            pos += 1;
                            let t = if delta == 255 {
                                pos += 2;
                                word(pos - 2) as i64
                            } else {
                                last + delta as i64
                            };
                            stream.insert((t, raw[pos]), raw[pos + 1] as i64);
                            pos += 2;
                            last = t;
                        }
                        stream
                    })
                    .collect()
            })
            .collect()
    })
}

/// Exact decoded stream rows for source-to-fixture semantic custody checks.
pub fn library_pair(metav4: bool, pair: usize) -> Value {
    let Some(streams) = library(metav4).get(pair) else {
        return Value::Null;
    };
    json!(
        streams
            .iter()
            .map(|stream| {
                let mut rows: Vec<_> = stream.iter().map(|((t, i), q)| (*t, *i, *q)).collect();
                rows.sort();
                rows
            })
            .collect::<Vec<_>>()
    )
}

fn pair_index(shops: &[Value]) -> Option<usize> {
    if shops.len() != 2 {
        return None;
    }
    let a = SHOP_NAMES.iter().position(|n| shops[0] == *n)?;
    let b = SHOP_NAMES.iter().position(|n| shops[1] == *n)?;
    Some(a * 8 + b)
}

#[derive(Clone, Debug, Default)]
struct PredictState {
    step: i64,
    seen: BTreeMap<(i64, u8), i64>,
    /// `st["best"]`: the chosen stream, `Some(None)` when the pair has no streams.
    best: Option<Option<(usize, usize)>>,
}

#[derive(Clone, Debug)]
pub struct Predict {
    states: [Option<PredictState>; 2],
    pub report: Value,
}
impl Default for Predict {
    fn default() -> Self {
        Self {
            states: [None, None],
            report: json!({"pred_units":0,"pred_fires":0,"pred_errors":0}),
        }
    }
}

impl Predict {
    pub fn inject(&mut self, states: &Value) {
        for seat in 0..2 {
            let value = &states[seat.to_string()];
            if value.is_null() {
                self.states[seat] = None;
                continue;
            }
            let mut seen = BTreeMap::new();
            if let Some(rows) = value["obs"].as_object() {
                for (key, quantity) in rows {
                    let pair: Vec<_> = key.split(',').collect();
                    if pair.len() == 2
                        && let (Ok(t), Ok(i)) =
                            (pair[0].trim().parse::<i64>(), pair[1].trim().parse::<u8>())
                    {
                        seen.insert((t, i), int(quantity));
                    }
                }
            }
            let best = if value["best"].is_null() {
                None
            } else {
                Some(Some((
                    int(&value["best"][0]) as usize,
                    int(&value["best"][1]) as usize,
                )))
            };
            self.states[seat] = Some(PredictState {
                step: int(&value["step"]),
                seen,
                best,
            });
        }
    }

    /// `_v92_p_forecast`: the library stream that best explains the last 240 turns.
    fn forecast(obs: &Value, st: &PredictState, metav4: bool) -> Option<(usize, usize)> {
        let step = int(&obs["step"]);
        let all = array(&obs["town"]["unlocked_shops"]);
        let pair = pair_index(&all[..all.len().min(2)])?;
        let lo = step - 240;
        let around = |t: i64| [t, t - 1, t + 1];
        let recent: Vec<(i64, u8)> = st.seen.keys().filter(|k| k.0 >= lo).copied().collect();
        let mut best: Option<(i64, usize)> = None;
        for (index, stream) in library(metav4)[pair].iter().enumerate() {
            let (mut m, mut f) = (0, 0);
            for (t, i) in stream.keys() {
                if lo <= *t && *t < step - 1 {
                    if around(*t).iter().any(|x| st.seen.contains_key(&(*x, *i))) {
                        m += 1;
                    } else {
                        f += 1;
                    }
                }
            }
            let miss = recent
                .iter()
                .filter(|(t, i)| !around(*t).iter().any(|x| stream.contains_key(&(*x, *i))))
                .count() as i64;
            // Twice the source score `m - 0.5 f - 0.5 miss`; the first maximum wins.
            let score = 2 * m - f - miss;
            if best.is_none_or(|b| score > b.0) {
                best = Some((score, index));
            }
        }
        best.map(|b| (pair, b.1))
    }

    /// PREDICT: sell planned lots just before the forecast rival stream does.
    pub fn layer(&mut self, obs: &Value, action: Value, core: &Core, race: &Race) -> Value {
        let step = int(&obs["step"]);
        let who = seat(obs).min(1);
        if self.states[who].as_ref().is_none_or(|st| step <= st.step) {
            self.states[who] = Some(PredictState {
                step: -1,
                ..Default::default()
            });
        }
        let st = self.states[who].as_mut().unwrap();
        st.step = step;
        // `_v92_p_update`: the rival's recovered sales of the previous turn.
        let prev = &race.states[who.to_string()]["prev"];
        if truth(prev) && int(&prev["step"]) == step - 1 {
            for (i, item) in PREDICT_ITEMS.iter().enumerate() {
                if int(&prev["prices"][*item]) <= 3 {
                    continue;
                }
                let sold = int(&obs["market"]["inventory"][*item]) - int(&prev["inventory"][*item])
                    + town_draw(&prev["shops"], int(&prev["step"]), item)
                    - int(&prev["own"][*item]);
                if sold >= 2 {
                    st.seen.insert((step - 1, i as u8), sold);
                }
            }
        }
        if !(150..700).contains(&step) {
            return action;
        }
        if step % PREDICT_EVERY == 0 || st.best.is_none() {
            st.best = Some(Self::forecast(obs, st, core.metav4));
        }
        let Some(Some((pair, index))) = st.best else {
            return action;
        };
        let stream = &library(core.metav4)[pair][index];
        let Some(route) = native_route(core, obs) else {
            return action;
        };
        let mut market = orders(&action);
        let already: Vec<String> = market
            .iter()
            .filter(|o| array(o).len() > 1 && (o[0] == "SELL" || o[0] == "BUY_PRODUCT"))
            .map(|o| text(&o[1]).to_owned())
            .collect();
        let stock = Core::projected_shed(&action, &View::new(obs));
        let mut changed = false;
        for (i, item) in PREDICT_ITEMS.iter().enumerate().take(PREDICT_USE) {
            if already.iter().any(|a| a == item) || market.len() >= MAX_ORDERS {
                continue;
            }
            let at = |t: i64| stream.get(&(t, i as u8)).copied().unwrap_or(0);
            if at(step + 1) + at(step + 2) < PREDICT_K {
                continue;
            }
            let ours: i64 = (step + 1..tape_len(route).min(step + PREDICT_H + 1))
                .flat_map(|t| orders(&tape(route, t)))
                .filter(|o| is_order(o, "SELL", item))
                .map(|o| int(&o[2]).min(100))
                .sum();
            let qty = int(&stock[*item]).min(ours);
            if qty > 0 {
                market.insert(0, json!(["SELL", item, qty]));
                inc(&mut self.report, "pred_units", qty);
                inc(&mut self.report, "pred_fires", 1);
                changed = true;
            }
        }
        if !changed {
            return action;
        }
        market.truncate(MAX_ORDERS);
        let mut result = action;
        set_orders(&mut result, market);
        result
    }

    /// Oracle view: recovered rival sale ticks and the chosen stream per seat.
    pub fn states(&self) -> Value {
        let mut out = json!({});
        for (who, st) in self.states.iter().enumerate() {
            if let Some(st) = st {
                let seen: serde_json::Map<String, Value> = st
                    .seen
                    .iter()
                    .map(|((t, i), q)| (format!("{t},{i}"), json!(q)))
                    .collect();
                out[who.to_string()] = json!({
                    "step": st.step,
                    "obs": seen,
                    "best": st.best.map(|b| b.map(|(pair, index)| json!([pair, index]))),
                });
            }
        }
        out
    }
}
