//! Native conditional grammar for Myolie's canonical actor241 programs.
//!
//! This module computes the masks, finite-state transitions and canonical action
//! decoding. Neural heads, floating-point sampling and RNG remain in the caller.
//! States discard token values once they no longer affect any future mask; plan
//! construction is iterative, so the maximum actor program cannot overflow the
//! call stack. HIRE support is checkpoint-owned and distinct from actor capacity.

use serde_json::{Map, Value};
use std::collections::HashMap;

pub const MAX_ACTORS: u32 = 241;
pub const MAX_ORDERS: u32 = 10;
pub const SLOTS: usize = 12;
pub const WIDTHS: [usize; SLOTS] = [241, 20, 128, 16, 2, 32, 32, 8, 16, 32, 32, 2];
pub const MAGIC: u32 = 0x4d53_4731;
pub const ABI: u32 = 1;

const UNIT_NAMES: [&str; 19] = [
    "NONE",
    "PASS",
    "NORTH",
    "SOUTH",
    "EAST",
    "WEST",
    "PICKUP",
    "PLACE",
    "PLANT",
    "WATER",
    "HARVEST",
    "DROP",
    "BUILD_COOP",
    "BUILD_PASTURE",
    "FEED",
    "FERTILIZE",
    "COLLECT_FERTILIZER",
    "CARE",
    "DIG",
];
const MARKET_NAMES: [&str; 8] = [
    "NONE",
    "HIRE",
    "BUY_LAND",
    "BUY_SEED",
    "BUY_PRODUCT",
    "BUY_ANIMAL",
    "SELL",
    "EMPTY",
];
const ITEMS: [&str; 13] = [
    "NONE",
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
    "GOOSE",
    "COW",
    "SHEEP",
];

#[derive(Clone, Copy, Debug)]
struct Shape {
    actors: u16,
    orders: u8,
    hire_limit: u16,
}

impl Shape {
    fn new(actors: u32, orders: u32, hire_limit: u32) -> Result<Self, String> {
        if !(1..=MAX_ACTORS).contains(&actors) {
            return Err("Myolie sampler actors must be in 1..241".into());
        }
        if !(1..=MAX_ORDERS).contains(&orders) {
            return Err("Myolie sampler market limit must be in 1..10".into());
        }
        if !(1..=MAX_ACTORS).contains(&hire_limit) {
            return Err("Myolie sampler hire limit must be in 1..241".into());
        }
        Ok(Self {
            actors: actors as u16,
            orders: orders as u8,
            hire_limit: hire_limit as u16,
        })
    }

    fn validate_length(self, tokens: usize) -> Result<(), String> {
        let frames = tokens / SLOTS;
        if !tokens.is_multiple_of(SLOTS)
            || frames < usize::from(self.actors) + 1
            || frames > usize::from(self.actors) + usize::from(self.orders) + 1
        {
            return Err(format!("invalid Myolie sampler token length {tokens}"));
        }
        Ok(())
    }
}

/// Validate scalar shape/length before an FFI caller constructs a raw slice.
pub fn validate_decode_length(
    actors: u32,
    limit: u32,
    hire_limit: u32,
    tokens: usize,
) -> Result<(), String> {
    Shape::new(actors, limit, hire_limit)?.validate_length(tokens)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
enum Unit {
    #[default]
    None,
    Ordinary,
    Transfer,
    Plant,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
enum Market {
    #[default]
    None,
    Hire,
    Other,
    Seed,
    Product,
    Animal,
    Sell,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
struct State {
    units: u16,
    orders: u8,
    hires: u8,
    slot: u8,
    unit: Unit,
    market: Market,
    quantity_present: bool,
    quantity_high_zero: bool,
}

impl State {
    fn allows(self, token: usize, shape: Shape) -> bool {
        if token >= WIDTHS[usize::from(self.slot)] {
            return false;
        }
        let ready = self.units == shape.actors;
        match self.slot {
            0 => token == if ready { 0 } else { usize::from(self.units) },
            1 => {
                if ready {
                    token == 0
                } else {
                    (1..=18).contains(&token)
                }
            }
            2 => token == 0,
            3 => match self.unit {
                Unit::Transfer => (1..=12).contains(&token),
                Unit::Plant => (1..=5).contains(&token),
                _ => token == 0,
            },
            4 => self.unit == Unit::Transfer || token == 0,
            5 => self.quantity_present || token == 0,
            6 => {
                if self.quantity_present {
                    !self.quantity_high_zero || token != 0
                } else {
                    token == 0
                }
            }
            7 => {
                if ready && self.unit == Unit::None && self.orders < shape.orders {
                    token != 1 || shape.actors + u16::from(self.hires) < shape.hire_limit
                } else {
                    token == 0
                }
            }
            8 => match self.market {
                Market::Seed => (1..=5).contains(&token),
                Market::Product => token == 1 || token == 9,
                Market::Animal => (10..=12).contains(&token),
                Market::Sell => (1..=9).contains(&token),
                _ => token == 0,
            },
            9 | 10 => self.market == Market::Seed || token == 0,
            11 => {
                token
                    == usize::from(ready && self.unit == Unit::None && self.market == Market::None)
            }
            _ => unreachable!("private state has a fixed slot range"),
        }
    }

    /// Advance an admitted token. None means the unique terminal STOP.
    fn advance(mut self, token: usize) -> Option<Self> {
        match self.slot {
            1 => {
                self.unit = match token {
                    0 => Unit::None,
                    6 | 7 => Unit::Transfer,
                    8 => Unit::Plant,
                    _ => Unit::Ordinary,
                };
            }
            3 => {
                if self.unit == Unit::Plant {
                    self.unit = Unit::Ordinary;
                }
            }
            4 => {
                self.quantity_present = token == 1;
                if self.unit == Unit::Transfer && !self.quantity_present {
                    self.unit = Unit::Ordinary;
                }
            }
            5 => self.quantity_high_zero = self.quantity_present && token == 0,
            6 => {
                self.quantity_present = false;
                self.quantity_high_zero = false;
                if self.unit == Unit::Transfer {
                    self.unit = Unit::Ordinary;
                }
            }
            7 => {
                self.market = match token {
                    0 => Market::None,
                    1 => Market::Hire,
                    3 => Market::Seed,
                    4 => Market::Product,
                    5 => Market::Animal,
                    6 => Market::Sell,
                    _ => Market::Other,
                };
            }
            8 => {
                if matches!(self.market, Market::Product | Market::Animal | Market::Sell) {
                    self.market = Market::Seed;
                }
            }
            10 => {
                if self.market == Market::Seed {
                    self.market = Market::Other;
                }
            }
            11 => {
                if token == 1 {
                    return None;
                }
                return Some(Self {
                    units: self.units + u16::from(self.unit != Unit::None),
                    orders: self.orders + u8::from(self.market != Market::None),
                    hires: self.hires + u8::from(self.market == Market::Hire),
                    ..Self::default()
                });
            }
            _ => {}
        }
        self.slot += 1;
        Some(self)
    }
}

/// Compute all reachable masks/transitions. Node numbers are deterministic BFS
/// discovery order, and each vocabulary is enumerated in ascending token order.
pub fn plan(actors: u32, limit: u32, hire_limit: u32) -> Result<Vec<u8>, String> {
    let shape = Shape::new(actors, limit, hire_limit)?;
    let start = State::default();
    let mut states = vec![start];
    let mut interned = HashMap::from([(start, 0_i32)]);
    let mut rows = Vec::<[u32; 4]>::new();
    let mut masks = Vec::<u8>::new();
    let mut next = Vec::<i32>::new();
    let mut index = 0;
    while index < states.len() {
        let state = states[index];
        let width = WIDTHS[usize::from(state.slot)];
        rows.push([
            u32::from(state.slot),
            u32::from(state.hires),
            masks.len() as u32,
            width as u32,
        ]);
        for token in 0..width {
            let allowed = state.allows(token, shape);
            masks.push(u8::from(allowed));
            next.push(if !allowed {
                -2
            } else if let Some(successor) = state.advance(token) {
                if let Some(&number) = interned.get(&successor) {
                    number
                } else {
                    let number = states.len() as i32;
                    interned.insert(successor, number);
                    states.push(successor);
                    number
                }
            } else {
                -1
            });
        }
        index += 1;
    }
    // At most 24 states per unit and 32 per (orders, hires) pair. There are
    // (10+1)*(10+2)/2 = 66 such pairs. This also bounds all integer conversions.
    debug_assert!(states.len() <= MAX_ACTORS as usize * 24 + 66 * 32);
    let mut bytes = Vec::with_capacity(20 + rows.len() * 16 + masks.len() * 5);
    for value in [MAGIC, ABI, rows.len() as u32, 0, masks.len() as u32] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    for row in rows {
        for value in row {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    bytes.extend_from_slice(&masks);
    for value in next {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(bytes)
}

/// Validate the complete canonical program and decode exact raw action syntax.
/// Quantity omission, explicit market zero, EMPTY and a separate STOP survive.
pub fn decode(actors: u32, limit: u32, hire_limit: u32, frames: &[i16]) -> Result<Value, String> {
    let shape = Shape::new(actors, limit, hire_limit)?;
    shape.validate_length(frames.len())?;
    let mut state = Some(State::default());
    for (index, &value) in frames.iter().enumerate() {
        let current = state.ok_or("Myolie sampler frame after STOP")?;
        if value < 0 || !current.allows(value as usize, shape) {
            return Err(format!(
                "Myolie sampler frame {} slot {} token {value} violates syntax mask",
                index / SLOTS,
                current.slot
            ));
        }
        state = current.advance(value as usize);
    }
    if state.is_some() {
        return Err("Myolie sampler program lacks distinct final STOP".into());
    }
    let mut units = Vec::with_capacity(actors as usize);
    let mut market = Vec::new();
    for frame in frames.chunks_exact(SLOTS) {
        let kind = frame[1] as usize;
        if kind != 0 {
            let mut command = vec![Value::String(UNIT_NAMES[kind].into())];
            if matches!(kind, 6..=8) {
                command.push(Value::String(ITEMS[frame[3] as usize].into()));
            }
            if frame[4] == 1 {
                command.push(Value::from(i64::from(frame[5]) * 32 + i64::from(frame[6])));
            }
            units.push(Value::Array(command));
        } else {
            let kind = frame[7] as usize;
            if kind == 0 {
                continue;
            }
            let mut command = if kind == 7 {
                Vec::new()
            } else {
                vec![Value::String(MARKET_NAMES[kind].into())]
            };
            if matches!(kind, 3..=6) {
                command.push(Value::String(ITEMS[frame[8] as usize].into()));
                command.push(Value::from(i64::from(frame[9]) * 32 + i64::from(frame[10])));
            }
            market.push(Value::Array(command));
        }
    }
    let mut raw = Map::new();
    raw.insert("farmer".into(), units.remove(0));
    raw.insert("hands".into(), Value::Array(units));
    raw.insert("market".into(), Value::Array(market));
    Ok(Value::Object(raw))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn frame(kind: i16, item: i16, quantity: Option<i16>) -> [i16; SLOTS] {
        let mut frame = [0; SLOTS];
        frame[1] = kind;
        frame[3] = item;
        if let Some(quantity) = quantity {
            frame[4] = 1;
            frame[5] = quantity / 32;
            frame[6] = quantity % 32;
        }
        frame
    }

    fn stop() -> [i16; SLOTS] {
        let mut frame = [0; SLOTS];
        frame[11] = 1;
        frame
    }

    fn flatten(frames: &[[i16; SLOTS]]) -> Vec<i16> {
        frames.iter().flatten().copied().collect()
    }

    #[test]
    fn exact_quantities_empty_and_canonical_fields() {
        let mut buy = [0; SLOTS];
        buy[7] = 3;
        buy[8] = 1;
        let mut empty = [0; SLOTS];
        empty[7] = 7;
        for quantity in [None, Some(1), Some(32), Some(1023)] {
            let program = flatten(&[frame(6, 12, quantity), buy, empty, stop()]);
            let mut unit = vec![json!("PICKUP"), json!("SHEEP")];
            if let Some(quantity) = quantity {
                unit.push(json!(quantity));
            }
            assert_eq!(
                decode(1, 2, 16, &program).unwrap(),
                json!({"farmer": unit, "hands": [], "market": [["BUY_SEED", "WHEAT", 0], []]})
            );
        }
        assert!(decode(1, 2, 16, &flatten(&[frame(6, 1, Some(0)), stop()])).is_err());
    }

    #[test]
    fn hire_support_tracks_pre_frame_requests() {
        let mut hire = [0; SLOTS];
        hire[7] = 1;
        let once = flatten(&[frame(1, 0, None), hire, stop()]);
        let twice = flatten(&[frame(1, 0, None), hire, hire, stop()]);
        assert!(decode(1, 10, 1, &once).is_err());
        assert!(decode(1, 10, 2, &once).is_ok());
        assert!(decode(1, 10, 2, &twice).is_err());
        assert!(decode(1, 10, 3, &twice).is_ok());
        let shape = Shape::new(1, 10, 2).unwrap();
        let mut state = State::default();
        for value in flatten(&[frame(1, 0, None)]) {
            assert!(state.allows(value as usize, shape));
            state = state.advance(value as usize).unwrap();
        }
        for &value in &hire[..11] {
            assert_eq!(state.hires, 0);
            state = state.advance(value as usize).unwrap();
        }
        assert_eq!(state.hires, 0);
        assert_eq!(state.advance(0).unwrap().hires, 1);
    }

    #[test]
    fn malformed_and_noncanonical_programs_rejected() {
        let base = flatten(&[frame(1, 0, None), stop()]);
        for (index, token) in [
            (0, 1),
            (1, 0),
            (1, 19),
            (1, -1),
            (2, 1),
            (3, 1),
            (4, 1),
            (11, 1),
            (23, 0),
        ] {
            let mut bad = base.clone();
            bad[index] = token;
            assert!(
                decode(1, 10, 16, &bad).is_err(),
                "accepted {index}: {token}"
            );
        }
        assert!(decode(1, 10, 16, &base[..23]).is_err());
        let after_stop = flatten(&[frame(1, 0, None), stop(), stop()]);
        assert!(decode(1, 10, 16, &after_stop).is_err());
        for config in [
            (0, 10, 16),
            (242, 10, 16),
            (1, 0, 16),
            (1, 11, 16),
            (1, 10, 0),
            (1, 10, 242),
        ] {
            assert!(plan(config.0, config.1, config.2).is_err());
        }
    }

    #[test]
    fn plans_are_bounded_deterministic_and_well_formed() {
        for (actors, limit, hire) in [
            (1, 1, 1),
            (1, 10, 241),
            (16, 10, 16),
            (231, 10, 241),
            (241, 10, 241),
        ] {
            let bytes = plan(actors, limit, hire).unwrap();
            assert_eq!(bytes, plan(actors, limit, hire).unwrap());
            let number = |offset| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
            let nodes = number(8) as usize;
            let edges = number(16) as usize;
            assert!(nodes <= 24 * MAX_ACTORS as usize + 66 * 32);
            assert!(edges < 1_000_000);
            assert_eq!(number(0), MAGIC);
            assert_eq!(number(4), ABI);
            assert_eq!(number(12), 0);
            assert_eq!(bytes.len(), 20 + nodes * 16 + edges * 5);
            for id in 0..nodes {
                let row = 20 + id * 16;
                let slot = number(row) as usize;
                let offset = number(row + 8) as usize;
                let width = number(row + 12) as usize;
                assert_eq!(width, WIDTHS[slot]);
                let mask = &bytes[20 + nodes * 16 + offset..20 + nodes * 16 + offset + width];
                assert!(mask.contains(&1));
                for (token, &admitted) in mask.iter().enumerate() {
                    let edge = 20 + nodes * 16 + edges + (offset + token) * 4;
                    let target = i32::from_le_bytes(bytes[edge..edge + 4].try_into().unwrap());
                    assert_eq!(target == -2, admitted == 0);
                    assert!(target < nodes as i32);
                    if target == -1 {
                        assert_eq!((slot, token), (11, 1));
                    }
                }
            }
            println!(
                "sampler_plan actors={actors} limit={limit} hire={hire}: nodes={nodes} edges={edges} bytes={}",
                bytes.len()
            );
        }
    }
}
