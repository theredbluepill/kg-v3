//! V48 EXP334/335 market compaction: drop sale slots that cannot sell, merge
//! repeated cash-product sales, and move executable sales into freed slots.
//! Source: agents/v48/main.py (`_E334_*`, `_e334_compact`, `_E335_ORIGINAL_COMPACT`).
use crate::native_agents::v43::common::*;
use crate::native_agents::v43::contracts::fields;
use serde_json::{Value, json};

pub const ITEMS: [&str; 7] = [
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
];

fn cash_sale(o: &Value) -> bool {
    array(o).len() >= 3 && o[0] == "SELL" && ITEMS.contains(&text(&o[1]))
}

/// EXP334 original `_e334_compact`: merge consecutive cash-product sale runs.
pub fn original(obs: &Value, action: &Value, report: &mut Value) -> Result<Value, &'static str> {
    let market = orders(action);
    let step = int(obs.get("step").ok_or("compact: step")?);
    if step < 144 || market.len() < 2 {
        return Ok(action.clone());
    }
    let mut segments = vec![];
    let mut i = 0;
    while i < market.len() {
        if cash_sale(&market[i]) {
            let mut j = i + 1;
            while j < market.len() && cash_sale(&market[j]) {
                j += 1;
            }
            if j - i >= 2 {
                segments.push((i, j));
            }
            i = j;
        } else {
            i += 1;
        }
    }
    if segments.is_empty() {
        return Ok(action.clone());
    }
    let (_, private) = fields(obs, action)?;
    let mut remaining = private["shed"].clone();
    if !remaining.is_object() {
        remaining = json!({});
    }
    let mut new = market.clone();
    let mut removed = 0;
    for (start, end) in segments {
        let mut order: Vec<String> = vec![];
        let mut quantities = json!({});
        for o in &market[start..end] {
            let p = text(&o[1]).to_owned();
            if !order.contains(&p) {
                order.push(p.clone());
            }
            increment(&mut quantities, &p, int(&o[2]).max(0));
        }
        let mut kept: Vec<Value> = vec![];
        for p in order {
            let q = int(&quantities[&p]).min(int(&remaining[&p]).max(0));
            if q != 0 {
                kept.push(json!(["SELL", p, q]));
                increment(&mut remaining, &p, -q);
            }
        }
        let pad = end - start - kept.len();
        let mut replacement = kept;
        replacement.extend(std::iter::repeat_n(json!([]), pad));
        if replacement[..] != market[start..end] {
            removed += pad as i64;
            drop(new.splice(start..end, replacement));
        }
    }
    if new == market {
        return Ok(action.clone());
    }
    increment(report, "changed", 1);
    increment(report, "removed", removed);
    let mut result = action.clone();
    set_orders(&mut result, new);
    Ok(result)
}

/// EXP335 `_e334_compact` (active): on sale-only turns every empty slot is a
/// hole, including wheat/fertilizer sales that cannot sell.
pub fn compact(obs: &Value, action: &Value, report: &mut Value) -> Result<Value, &'static str> {
    let market = orders(action);
    let step = int(obs.get("step").ok_or("compact: step")?);
    if step < 144 || market.len() < 2 {
        return Ok(action.clone());
    }
    if !market.iter().all(|o| array(o).len() >= 3 && o[0] == "SELL") {
        return original(obs, action, report);
    }
    let (_, private) = fields(obs, action)?;
    let mut remaining = private["shed"].clone();
    if !remaining.is_object() {
        remaining = json!({});
    }
    let mut effective: Vec<Value> = vec![];
    for o in &market {
        let item = text(&o[1]).to_owned();
        let q = int(&o[2]).max(0).min(int(&remaining[&item]).max(0));
        remaining[&item] = json!((int(&remaining[&item]) - q).max(0));
        effective.push(if q != 0 {
            json!(["SELL", item, q])
        } else {
            json!([])
        });
    }
    let mut new = effective.clone();
    let mut i = 0;
    let mut removed = 0;
    while i < effective.len() {
        if truth(&effective[i]) && !ITEMS.contains(&text(&effective[i][1])) {
            i += 1;
            continue;
        }
        let mut j = i + 1;
        while j < effective.len()
            && (!truth(&effective[j]) || ITEMS.contains(&text(&effective[j][1])))
        {
            j += 1;
        }
        let mut order: Vec<String> = vec![];
        let mut qty = json!({});
        for o in &effective[i..j] {
            if !truth(o) {
                continue;
            }
            let item = text(&o[1]).to_owned();
            if !order.contains(&item) {
                order.push(item.clone());
            }
            increment(&mut qty, &item, int(&o[2]));
        }
        let mut kept: Vec<Value> = order
            .iter()
            .map(|item| json!(["SELL", item, int(&qty[item])]))
            .collect();
        let pad = j - i - kept.len();
        kept.extend(std::iter::repeat_n(json!([]), pad));
        drop(new.splice(i..j, kept));
        removed += pad as i64;
        i = j;
    }
    if new == market {
        return Ok(action.clone());
    }
    increment(report, "changed", 1);
    increment(report, "removed", removed);
    let mut result = action.clone();
    set_orders(&mut result, new);
    Ok(result)
}
