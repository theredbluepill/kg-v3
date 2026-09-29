//! Economic attribution ledger (blind-spot audit rank 3, E08/E32/E35/E37-E39/E62; owner 2026-09-28 «馬上修2，3，4»).
//!
//! Telemetry only: gameplay, observations, rewards and the econ shaping counters never read it. Per seat and
//! cumulative since the `Game` was constructed (cloned with the state, like `econ_counters`). It separates what
//! the econ v4 counters merge: sales and purchases per item (units and exact integer cash), hires and land,
//! produced units per product, effective / ineffective commands per verb and per actor slot, idle slots, market
//! partial fills per order kind, shed-transfer partial fills, and the day-end state of service not done
//! (unwatered plants, unfed / uncared animals, mature yield and fertilizer still on the tiles). These are exact
//! engine events and states; none is an estimate of recoverable money (a counterfactual continuation would be).
use super::*;

pub const ATTRIB_VERSION: u32 = 1;
/// UNIT_KIND_VALUES[1..19] of the Myolie codec (actor_codec.py), engine command strings.
pub const ATTRIB_VERBS: [&str; 18] = [
    "PASS", "NORTH", "SOUTH", "EAST", "WEST", "PICKUP", "PLACE", "PLANT", "WATER", "HARVEST", "DROP",
    "BUILD_COOP", "BUILD_PASTURE", "FEED", "FERTILIZE", "COLLECT_FERTILIZER", "CARE", "DIG",
];
/// Actor slots: farmer 0, hands 1..14, and every hand index >= 15 in slot 15.
pub const ATTRIB_SLOTS: usize = 16;
/// Per-product blocks follow `PRODUCTS` (WHEAT..FERTILIZER); crops are its first five, animals `ANIMAL_NAMES`.
pub const A_SELL_UNITS: usize = 0;
pub const A_SELL_CASH: usize = 9;
pub const A_BUY_PRODUCT_UNITS: usize = 18;
pub const A_BUY_PRODUCT_CASH: usize = 27;
pub const A_BUY_SEED_UNITS: usize = 36;
pub const A_BUY_SEED_CASH: usize = 41;
pub const A_BUY_ANIMAL_UNITS: usize = 46;
pub const A_BUY_ANIMAL_CASH: usize = 49;
pub const A_HIRES: usize = 52;
pub const A_HIRE_CASH: usize = 53;
pub const A_LANDS: usize = 54;
pub const A_LAND_CASH: usize = 55;
/// Units collected by committed HARVEST (crop or animal product) and COLLECT_FERTILIZER, per product.
pub const A_PRODUCED_UNITS: usize = 56;
pub const A_VERB_EFFECTIVE: usize = 65;
pub const A_VERB_INEFFECTIVE: usize = 83;
pub const A_SLOT_EFFECTIVE: usize = 101;
pub const A_SLOT_INEFFECTIVE: usize = 117;
/// Existing actors whose command was PASS or omitted (farmer included, unlike ECON_IDLE_HAND_STEPS).
pub const A_SLOT_IDLE: usize = 133;
/// Order kinds BUY_SEED, BUY_PRODUCT, BUY_ANIMAL, SELL: parsed requested and committed units.
pub const A_MARKET_REQUESTED: usize = 149;
pub const A_MARKET_COMMITTED: usize = 153;
/// PICKUP, PLACE: requested units (explicit count or 1) and units actually moved from/to the actor.
pub const A_TRANSFER_REQUESTED: usize = 157;
pub const A_TRANSFER_MOVED: usize = 159;
/// Day-end states before the daily refresh (plant-days, animal-days, units, flags).
pub const A_EOD_UNWATERED_PLANTS: usize = 161;
pub const A_EOD_UNFED_ANIMALS: usize = 162;
pub const A_EOD_UNCARED_ANIMALS: usize = 163;
pub const A_EOD_READY_CROP_UNITS: usize = 164;
pub const A_EOD_ANIMAL_PRODUCT_UNITS: usize = 165;
pub const A_EOD_FERTILIZER_UNCOLLECTED: usize = 166;
pub const ATTRIB_FIELDS: usize = 167;

pub(crate) fn product_index(item: &str) -> Option<usize> {
    PRODUCTS.iter().position(|p| *p == item)
}

fn verb_index(op: &str) -> Option<usize> {
    ATTRIB_VERBS.iter().position(|v| *v == op)
}

fn op_of(action: &Value) -> Option<&str> {
    action.as_array().and_then(|parts| parts.first()).and_then(Value::as_str)
}

fn integer_cash(price: &Value) -> u64 {
    price.as_u64().or_else(|| price.as_f64().map(|value| value.round().max(0.0) as u64)).unwrap_or(0)
}

/// The actor's carried count of a PICKUP/PLACE item before the command (None for other commands).
pub(crate) fn transfer_before(private: &PrivateState, index: usize, action: &Value) -> Option<(String, i64)> {
    let parts = action.as_array()?;
    if !matches!(parts.first()?.as_str()?, "PICKUP" | "PLACE") {
        return None;
    }
    let item = parts.get(1)?.as_str()?;
    let held = private.inventories.get(index).and_then(|inv| inv.get(item)).copied().unwrap_or(0);
    Some((item.to_string(), held))
}

impl Game {
    pub fn attrib_counters(&self) -> Option<&[[u64; ATTRIB_FIELDS]; 2]> {
        (self.farms.len() == 2).then_some(&self.attrib)
    }

    fn attrib_row(&mut self, player: usize) -> Option<&mut [u64; ATTRIB_FIELDS]> {
        self.attrib.get_mut(player).filter(|_| {
            #[cfg(test)] { !self.econ_counting_disabled }
            #[cfg(not(test))] { true }
        })
    }

    /// Idle actor slots of one seat's submitted program (existing actors whose command is PASS or omitted).
    pub(crate) fn count_attrib_idle(&mut self, player: usize, farmer_action: &Value, hands_actions: &[Value]) {
        let farmer = farmer_position(&self.farms[player], 0).is_some() && op_of(farmer_action) == Some("PASS");
        let hands = self.farms[player].hands.len();
        let idle: Vec<usize> = (0..hands)
            .filter(|&i| hands_actions.get(i).is_none_or(|a| op_of(a) == Some("PASS")))
            .map(|i| (i + 1).min(ATTRIB_SLOTS - 1))
            .collect();
        let Some(c) = self.attrib_row(player) else { return; };
        c[A_SLOT_IDLE] += farmer as u64;
        for slot in idle {
            c[A_SLOT_IDLE + slot] += 1;
        }
    }

    /// One existing unit's submitted command after it was applied: effective / ineffective per verb and slot,
    /// produced units (from the tile before a committed HARVEST), shed-transfer request and movement.
    pub(crate) fn count_attrib_unit(&mut self, player: usize, index: usize, action: &Value, exists: bool,
                                    effect: UnitActionEffect, before: Option<&(usize, usize, Value)>,
                                    transfer: Option<(String, i64)>) {
        if !exists {
            return;
        }
        let Some(op) = op_of(action) else { return; };
        let Some(verb) = verb_index(op) else { return; };
        let slot = index.min(ATTRIB_SLOTS - 1);
        let produced = if op == "HARVEST" && effect != UnitActionEffect::None {
            before.and_then(|(_, _, tile)| tile.as_object()).and_then(|t| {
                let units = map_i64(t, "yield_units").max(0) as u64;
                let item = if map_str(t, "kind") == Some("PLANT") { map_str(t, "crop") }
                    else { map_str(t, "animal").and_then(animal).map(|a| a.product) };
                item.and_then(product_index).map(|p| (p, units))
            })
        } else if op == "COLLECT_FERTILIZER" && effect != UnitActionEffect::None {
            product_index("FERTILIZER").map(|p| (p, 1))
        } else {
            None
        };
        let moved = transfer.map(|(item, held)| {
            let after = self.privates[player].inventories.get(index).and_then(|inv| inv.get(&item)).copied()
                .unwrap_or(0);
            let requested = match action.as_array().and_then(|parts| parts.get(2)) {
                Some(value) => value.as_i64().filter(|&n| n > 0).unwrap_or(0) as u64,
                None => 1,
            };
            (usize::from(op == "PLACE"), requested, (after - held).unsigned_abs())
        });
        let Some(c) = self.attrib_row(player) else { return; };
        if effect != UnitActionEffect::None {
            c[A_VERB_EFFECTIVE + verb] += 1;
            c[A_SLOT_EFFECTIVE + slot] += 1;
        } else if op != "PASS" {
            c[A_VERB_INEFFECTIVE + verb] += 1;
            c[A_SLOT_INEFFECTIVE + slot] += 1;
        }
        if let Some((p, units)) = produced {
            c[A_PRODUCED_UNITS + p] += units;
        }
        if let Some((k, requested, moved)) = moved {
            c[A_TRANSFER_REQUESTED + k] += requested;
            c[A_TRANSFER_MOVED + k] += moved;
        }
    }

    /// One committed market unit (a quote accepted by commit_unit).
    pub(crate) fn count_attrib_commit(&mut self, player: usize, kind: OrderKind, item: &str, price: &Value) {
        let cash = integer_cash(price);
        let slot = match kind {
            OrderKind::Sell => product_index(item).map(|p| (A_SELL_UNITS + p, A_SELL_CASH + p)),
            OrderKind::BuyProduct => product_index(item).map(|p| (A_BUY_PRODUCT_UNITS + p, A_BUY_PRODUCT_CASH + p)),
            OrderKind::BuySeed => product_index(item).filter(|&p| p < 5)
                .map(|p| (A_BUY_SEED_UNITS + p, A_BUY_SEED_CASH + p)),
            OrderKind::BuyAnimal => ANIMAL_NAMES.iter().position(|a| *a == item)
                .map(|a| (A_BUY_ANIMAL_UNITS + a, A_BUY_ANIMAL_CASH + a)),
            OrderKind::Hire | OrderKind::BuyLand => None,
        };
        if let (Some((units, money)), Some(c)) = (slot, self.attrib_row(player)) {
            c[units] += 1;
            c[money] += cash;
        }
    }

    /// A committed HIRE or BUY_LAND and the cash it cost.
    pub(crate) fn count_attrib_investment(&mut self, player: usize, kind: OrderKind, cash_before: f64) {
        let paid = (cash_before - self.farms[player].money).max(0.0).round() as u64;
        let (count, cash) = if kind == OrderKind::Hire { (A_HIRES, A_HIRE_CASH) } else { (A_LANDS, A_LAND_CASH) };
        if let Some(c) = self.attrib_row(player) {
            c[count] += 1;
            c[cash] += paid;
        }
    }

    /// One market queue slot's parsed requested and committed units, for quantity orders.
    pub(crate) fn count_attrib_fill(&mut self, player: usize, kind: Option<OrderKind>, requested: u64, committed: u64) {
        let k = match kind {
            Some(OrderKind::BuySeed) => 0,
            Some(OrderKind::BuyProduct) => 1,
            Some(OrderKind::BuyAnimal) => 2,
            Some(OrderKind::Sell) => 3,
            _ => return,
        };
        if let Some(c) = self.attrib_row(player) {
            c[A_MARKET_REQUESTED + k] += requested;
            c[A_MARKET_COMMITTED + k] += committed;
        }
    }

    /// The seat's day-end service state, before the daily refresh.
    pub(crate) fn count_attrib_day_end(&mut self, player: usize, day: i64) {
        let mut add = [0u64; 6];
        for t in self.farms[player].tiles.iter().flatten().filter_map(Value::as_object) {
            let held = map_i64(t, "yield_units").max(0) as u64;
            if map_str(t, "kind") == Some("PLANT") {
                add[0] += !map_bool(t, "watered_today") as u64;
                if let Some(data) = map_str(t, "crop").and_then(crop)
                    && day - map_i64(t, "planted_day") >= data.first_yield_day {
                    add[3] += held;
                }
            } else if map_str(t, "animal").and_then(animal).is_some() {
                add[1] += !map_bool(t, "fed_today") as u64;
                add[2] += !map_bool(t, "cared_today") as u64;
                add[4] += held;
                add[5] += map_bool(t, "fertilizer_available") as u64;
            }
        }
        const FIELDS: [usize; 6] = [A_EOD_UNWATERED_PLANTS, A_EOD_UNFED_ANIMALS, A_EOD_UNCARED_ANIMALS,
                                    A_EOD_READY_CROP_UNITS, A_EOD_ANIMAL_PRODUCT_UNITS, A_EOD_FERTILIZER_UNCOLLECTED];
        if let Some(c) = self.attrib_row(player) {
            for (field, value) in FIELDS.into_iter().zip(add) {
                c[field] += value;
            }
        }
    }
}
