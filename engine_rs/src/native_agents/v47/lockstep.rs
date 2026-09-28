//! V47 v44y exact lockstep best-response SELL ordering against a detected clone.
//! Source: agents/v47/main.py (`_v44y_*`, `v44y_lockstep_agent`), after Seyit Kaan Gunes;
//! EXP298 per-item score caching preserves the donor's search order and ties.
use super::race::Race;
use crate::native_agents::v43::common::*;
use crate::native_agents::v43::core::Core;
use crate::native_agents::v43::production::market_price;
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Clone, Debug)]
pub struct Lockstep {
    pub report: Value,
}
impl Default for Lockstep {
    fn default() -> Self {
        Self {
            report: json!({"v44y_reorder_turns":0,"v44y_reorder_gain":0.0,"v44y_errors":0}),
        }
    }
}

type Rem = Option<(String, String, i64)>;
type Schedule = Vec<(usize, String, i64)>;

fn valid(o: &Value) -> bool {
    truth(o)
        && array(o).len() >= 3
        && (o[0] == "SELL" || o[0] == "BUY_PRODUCT")
        && PRODUCTS.contains(&text(&o[1]))
}

/// `_v44y_lockstep`: per-slot, per-unit engine lockstep for SELL/BUY_PRODUCT
/// orders with unbounded money. Returns (revenue_me, revenue_opp).
pub fn lockstep(
    mine: &[Value],
    theirs: &[Value],
    inv: &mut BTreeMap<String, i64>,
    stock: &mut [BTreeMap<String, i64>; 2],
    params: Option<&Value>,
) -> Result<(f64, f64), &'static str> {
    let queues = [mine, theirs];
    let mut rev = [0.0_f64, 0.0_f64];
    for i in 0..mine.len().max(theirs.len()) {
        let mut rem: [Rem; 2] = [None, None];
        for p in 0..2 {
            if let Some(o) = queues[p].get(i)
                && valid(o)
            {
                let n = int(&o[2]);
                if n > 0 {
                    rem[p] = Some((text(&o[0]).to_owned(), text(&o[1]).to_owned(), n));
                }
            }
        }
        let mut guard = 0;
        loop {
            guard += 1;
            if guard > 5000 {
                break;
            }
            let mut quoted: [Option<(bool, String, i64)>; 2] = [None, None];
            for p in 0..2 {
                let Some((op, item, n)) = rem[p].clone() else {
                    continue;
                };
                if n <= 0 {
                    continue;
                }
                let current = *inv.get(&item).ok_or("lockstep: inventory")?;
                if op == "SELL" {
                    quoted[p] = Some((
                        true,
                        item.clone(),
                        market_price(&item, current as f64, params),
                    ));
                } else if item == "WHEAT" || item == "FERTILIZER" {
                    quoted[p] = Some((
                        false,
                        item.clone(),
                        market_price(&item, (current - 1) as f64, params),
                    ));
                } else {
                    rem[p] = None;
                }
            }
            if quoted[0].is_none() && quoted[1].is_none() {
                break;
            }
            let mut committed = false;
            for p in 0..2 {
                let Some((sell, item, price)) = quoted[p].clone() else {
                    continue;
                };
                if sell {
                    if *stock[p].get(&item).unwrap_or(&0) <= 0 {
                        rem[p] = None;
                        continue;
                    }
                    *stock[p].entry(item.clone()).or_insert(0) -= 1;
                    rev[p] += price as f64;
                    if price > 1 {
                        *inv.entry(item.clone()).or_insert(0) += 1;
                    }
                } else {
                    *stock[p].entry(item.clone()).or_insert(0) += 1;
                    rev[p] -= price as f64;
                    *inv.entry(item.clone()).or_insert(0) -= 1;
                }
                if let Some(r) = rem[p].as_mut() {
                    r.2 -= 1;
                }
                committed = true;
            }
            if !committed {
                break;
            }
        }
    }
    Ok((rev[0], rev[1]))
}

/// `_v44y_factor_margin`: independent per-item schedules with exact caching.
struct Margin<'a> {
    opp_len: usize,
    opp_items: Vec<String>,
    opp_schedules: HashMap<String, Vec<Value>>,
    inv0: BTreeMap<String, i64>,
    stock: BTreeMap<String, i64>,
    params: Option<&'a Value>,
    cache: HashMap<(String, Schedule), f64>,
}

impl<'a> Margin<'a> {
    fn new(
        opp: &[Value],
        inv0: BTreeMap<String, i64>,
        stock: BTreeMap<String, i64>,
        params: Option<&'a Value>,
    ) -> Self {
        let mut opp_items = vec![];
        let mut opp_schedules: HashMap<String, Vec<Value>> = HashMap::new();
        for (i, order) in opp.iter().enumerate() {
            if valid(order) {
                let item = text(&order[1]).to_owned();
                if !opp_items.contains(&item) {
                    opp_items.push(item.clone());
                }
                let padded = opp_schedules
                    .entry(item)
                    .or_insert_with(|| vec![json!([]); opp.len()]);
                padded[i] = order.clone();
            }
        }
        Self {
            opp_len: opp.len(),
            opp_items,
            opp_schedules,
            inv0,
            stock,
            params,
            cache: HashMap::new(),
        }
    }

    fn of(&mut self, cand: &[Value]) -> Result<f64, &'static str> {
        let mut order: Vec<String> = self.opp_items.clone();
        let mut schedules: HashMap<String, Schedule> =
            order.iter().map(|item| (item.clone(), vec![])).collect();
        for (i, o) in cand.iter().enumerate() {
            if valid(o) {
                let item = text(&o[1]).to_owned();
                if !schedules.contains_key(&item) {
                    order.push(item.clone());
                }
                schedules
                    .entry(item)
                    .or_default()
                    .push((i, text(&o[0]).to_owned(), int(&o[2])));
            }
        }
        let mut total = 0.0;
        for item in order {
            let schedule = schedules.remove(&item).unwrap_or_default();
            let key = (item.clone(), schedule.clone());
            let value = match self.cache.get(&key) {
                Some(v) => *v,
                None => {
                    let mut mine = vec![json!([]); cand.len()];
                    for (i, op, n) in &schedule {
                        mine[*i] = json!([op, item, n]);
                    }
                    let theirs = self
                        .opp_schedules
                        .get(&item)
                        .cloned()
                        .unwrap_or_else(|| vec![json!([]); self.opp_len]);
                    let start = *self.inv0.get(&item).ok_or("lockstep: inv0")?;
                    let held = *self.stock.get(&item).unwrap_or(&0);
                    let mut inv = BTreeMap::from([(item.clone(), start)]);
                    let mut stock = [
                        BTreeMap::from([(item.clone(), held)]),
                        BTreeMap::from([(item.clone(), held)]),
                    ];
                    let (a, b) = lockstep(&mine, &theirs, &mut inv, &mut stock, self.params)?;
                    let v = a - b;
                    self.cache.insert(key, v);
                    v
                }
            };
            total += value;
        }
        Ok(total)
    }
}

/// Lexicographic successor, matching `itertools.permutations(range(n))` order.
fn next_permutation(perm: &mut [usize]) -> bool {
    if perm.len() < 2 {
        return false;
    }
    let mut i = perm.len() - 1;
    while i > 0 && perm[i - 1] >= perm[i] {
        i -= 1;
    }
    if i == 0 {
        return false;
    }
    let mut j = perm.len() - 1;
    while perm[j] <= perm[i - 1] {
        j -= 1;
    }
    perm.swap(i - 1, j);
    perm[i..].reverse();
    true
}

impl Lockstep {
    /// `_v44y_reorder`: permute each 2..6-long SELL block against a clone of the
    /// same list, keeping the first strictly better (+0.5) ordering.
    pub fn reorder(&mut self, obs: &Value, action: Value) -> Result<Value, &'static str> {
        let market = orders(&action);
        if market.len() < 2 {
            return Ok(action);
        }
        let mut ords = market;
        let mut blocks = vec![];
        let mut i = 0;
        while i < ords.len() {
            if truth(&ords[i]) && ords[i][0] == "SELL" {
                let mut j = i;
                while j < ords.len() && truth(&ords[j]) && ords[j][0] == "SELL" {
                    j += 1;
                }
                if (2..=6).contains(&(j - i)) {
                    blocks.push((i, j));
                }
                i = j;
            } else {
                i += 1;
            }
        }
        if blocks.is_empty() {
            return Ok(action);
        }
        let view = View::new(obs);
        let stock: BTreeMap<String, i64> = Core::projected_shed(&action, &view)
            .as_object()
            .map(|m| m.iter().map(|(k, v)| (k.clone(), int(v).max(0))).collect())
            .unwrap_or_default();
        let params = obs["market"].get("params");
        let inv0: BTreeMap<String, i64> = obs["market"]["inventory"]
            .as_object()
            .ok_or("lockstep: market inventory")?
            .iter()
            .map(|(k, v)| (k.clone(), int(v)))
            .collect();
        let opp = ords.clone();
        let mut margin = Margin::new(&opp, inv0, stock, params);
        let base = margin.of(&ords)?;
        let mut best = base;
        let mut best_orders: Option<Vec<Value>> = None;
        for (i, j) in blocks {
            let blk: Vec<Value> = ords[i..j].to_vec();
            let n = j - i;
            let mut seen: HashSet<Vec<(String, i64)>> = HashSet::new();
            let mut perm: Vec<usize> = (0..n).collect();
            loop {
                let mut key = Vec::with_capacity(n);
                for &p in &perm {
                    if array(&blk[p]).len() < 3 {
                        return Err("lockstep: short order");
                    }
                    key.push((text(&blk[p][1]).to_owned(), int(&blk[p][2])));
                }
                if seen.insert(key) {
                    let mut cand: Vec<Value> = ords[..i].to_vec();
                    cand.extend(perm.iter().map(|&p| blk[p].clone()));
                    cand.extend_from_slice(&ords[j..]);
                    let v = margin.of(&cand)?;
                    if v > best + 0.5 {
                        best = v;
                        best_orders = Some(cand);
                    }
                }
                if !next_permutation(&mut perm) {
                    break;
                }
            }
            if let Some(chosen) = best_orders.take() {
                ords = chosen;
            }
        }
        if best <= base + 0.5 {
            return Ok(action);
        }
        increment(&mut self.report, "v44y_reorder_turns", 1);
        let gain = num(&self.report["v44y_reorder_gain"]) + (best - base);
        self.report["v44y_reorder_gain"] = json!(gain);
        let mut result = action;
        set_orders(&mut result, ords);
        Ok(result)
    }

    pub fn layer(&mut self, obs: &Value, action: Value, race: &Race) -> Value {
        let step = int(&obs["step"]);
        if step < 216 || !race.clone_gate(obs) {
            return action;
        }
        match self.reorder(obs, action.clone()) {
            Ok(result) => result,
            Err(_) => {
                increment(&mut self.report, "v44y_errors", 1);
                action
            }
        }
    }
}
