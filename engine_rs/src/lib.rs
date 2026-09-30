//! Replay-verified Kaggriculture transition kernel derived from the Apache-2.0
//! public Python engine.
//!
//! RE0 established exact transition parity while consuming exported exogenous
//! schedules. RE1 owns reset and CPython-compatible per-day randomness so rollout
//! callers need only a configuration, resolved seed, and actions.

use indexmap::IndexMap;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{FromPrimitive, ToPrimitive};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Number, Value, json};
use std::cmp::min;
use std::collections::BTreeMap;
use std::str::FromStr;

mod econ_attrib;
pub use econ_attrib::{ATTRIB_FIELDS, ATTRIB_VERBS, ATTRIB_VERSION};
pub mod py_random;

use py_random::PyRandom;

pub const TRACE_FORMAT: &str = "kaggriculture-re-parity-v1";
pub const PRODUCTS: [&str; 9] = [
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
];
pub const ANIMAL_NAMES: [&str; 3] = ["GOOSE", "COW", "SHEEP"];
pub const CROP_NAMES: [&str; 5] = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"];
const LAND_ORDER: [&str; 3] = ["NE", "SW", "SE"];
const LAND_PRICES: [i64; 3] = [1_000, 2_000, 4_000];
const PRICE_FLOOR: i64 = 1;
const HINGE_GAIN: f64 = 8.0;
const MAX_SHOP_INSTANCES: usize = 8;
// CPython 3.11.15's default `sys.get_int_max_str_digits()` value. Numeric JSON
// integers do not pass through this text conversion limit; permissive action
// strings do, and `_parse_order` catches the resulting ValueError.
const CPYTHON_INT_MAX_STR_DIGITS: usize = 4_300;
const SORTED_SHOPS: [&str; 8] = [
    "BAKERY",
    "BRUNCH_SPOT",
    "FARMERS_MARKET",
    "ICE_CREAM_SHOP",
    "PET_CAFE",
    "PIZZA_SHOP",
    "SMOOTHIE_SHOP",
    "YARN_STORE",
];

#[derive(Clone, Copy, Debug)]
struct CropData {
    seed: i64,
    first_yield_day: i64,
    max_yield_day: i64,
    interval: i64,
    max_yield: i64,
    ongoing: bool,
}

fn crop(name: &str) -> Option<CropData> {
    Some(match name {
        "WHEAT" => CropData {
            seed: 10,
            first_yield_day: 2,
            max_yield_day: 4,
            interval: 0,
            max_yield: 6,
            ongoing: false,
        },
        "CARROT" => CropData {
            seed: 20,
            first_yield_day: 2,
            max_yield_day: 3,
            interval: 0,
            max_yield: 4,
            ongoing: false,
        },
        "TOMATO" => CropData {
            seed: 50,
            first_yield_day: 8,
            max_yield_day: 8,
            interval: 1,
            max_yield: 4,
            ongoing: true,
        },
        "STRAWBERRY" => CropData {
            seed: 100,
            first_yield_day: 10,
            max_yield_day: 10,
            interval: 2,
            max_yield: 4,
            ongoing: true,
        },
        "MELON" => CropData {
            seed: 80,
            first_yield_day: 10,
            max_yield_day: 12,
            interval: 0,
            max_yield: 6,
            ongoing: false,
        },
        _ => return None,
    })
}

#[derive(Clone, Copy, Debug)]
struct AnimalData {
    cost: i64,
    structure: &'static str,
    first_yield_day: i64,
    interval: i64,
    max_held: i64,
    product: &'static str,
}

fn animal(name: &str) -> Option<AnimalData> {
    Some(match name {
        "GOOSE" => AnimalData {
            cost: 300,
            structure: "COOP",
            first_yield_day: 4,
            interval: 1,
            max_held: 4,
            product: "EGG",
        },
        "COW" => AnimalData {
            cost: 400,
            structure: "PASTURE",
            first_yield_day: 8,
            interval: 2,
            max_held: 6,
            product: "MILK",
        },
        "SHEEP" => AnimalData {
            cost: 500,
            structure: "PASTURE",
            first_yield_day: 6,
            interval: 3,
            max_held: 6,
            product: "WOOL",
        },
        _ => return None,
    })
}

/// The official engine deliberately treats nested `marketParams` entries as
/// permissive dictionaries. Keeping their JSON values avoids narrowing float-valued
/// overrides, dropping unknown metadata keys, or rejecting non-dict patches that
/// Python simply ignores.
pub type MarketParam = IndexMap<String, Value>;
pub type MarketParams = IndexMap<String, MarketParam>;

fn market_param(
    base: i64,
    t: i64,
    below_func: &str,
    below_target: f64,
    above_func: &str,
    above_target: f64,
) -> MarketParam {
    IndexMap::from([
        ("base".to_string(), Value::from(base)),
        ("I0".to_string(), Value::from(10_000)),
        ("T".to_string(), Value::from(t)),
        ("below_func".to_string(), Value::from(below_func)),
        ("below_target".to_string(), Value::from(below_target)),
        ("above_func".to_string(), Value::from(above_func)),
        ("above_target".to_string(), Value::from(above_target)),
    ])
}

fn default_market_params() -> MarketParams {
    let rows = [
        ("WHEAT", 25, 400, "sqrt", 0.80, "log", 0.20),
        ("CARROT", 35, 450, "hinge", 1.00, "sqrt", 0.70),
        ("TOMATO", 60, 200, "hinge", 0.40, "sqrt", 0.60),
        ("STRAWBERRY", 120, 100, "sqrt", 0.70, "linear", 1.60),
        ("MELON", 250, 300, "log", 0.20, "sq", 3.60),
        ("EGG", 50, 332, "hinge", 0.40, "log", 0.20),
        ("MILK", 160, 122, "sqrt", 0.60, "linear", 1.60),
        ("WOOL", 200, 105, "log", 0.20, "sq", 3.20),
        ("FERTILIZER", 100, 200, "linear", 0.40, "linear", 0.40),
    ];
    rows.into_iter()
        .map(
            |(name, base, t, below_func, below_target, above_func, above_target)| {
                (
                    name.to_string(),
                    market_param(base, t, below_func, below_target, above_func, above_target),
                )
            },
        )
        .collect()
}

fn resolve_market_params(patches: &IndexMap<String, Value>) -> MarketParams {
    let mut params = default_market_params();
    for (item, patch) in patches {
        let Some(p) = params.get_mut(item) else {
            continue;
        };
        let Some(patch) = patch.as_object() else {
            // `_resolve_market_params` ignores non-dict entries.
            continue;
        };
        for (key, value) in patch {
            p.insert(key.clone(), value.clone());
        }
    }
    params
}

fn numeric(value: &Value, label: &str) -> Result<f64, String> {
    if let Some(number) = value.as_f64() {
        return Ok(number);
    }
    if let Some(boolean) = value.as_bool() {
        // Python bool is an int subclass. The JSON schema does not advertise bools,
        // but permissive nested market patches can still contain them.
        return Ok(if boolean { 1.0 } else { 0.0 });
    }
    Err(format!("{label} must be numeric, got {value}"))
}

fn integer_value(value: &Value) -> Option<Result<BigInt, String>> {
    let Value::Number(number) = value else {
        return value
            .as_bool()
            .map(|boolean| Ok(BigInt::from(i64::from(boolean))));
    };
    let rendered = number.to_string();
    if rendered.contains(['.', 'e', 'E']) {
        return None;
    }
    Some(
        BigInt::from_str(&rendered)
            .map_err(|error| format!("invalid JSON integer {rendered}: {error}")),
    )
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PyInt(BigInt);

#[derive(Clone, Debug)]
enum PyIntError {
    Invalid(String),
    Overflow(String),
}

impl std::fmt::Display for PyIntError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(message) | Self::Overflow(message) => formatter.write_str(message),
        }
    }
}

fn python_int(value: &Value, label: &str) -> Result<BigInt, PyIntError> {
    if let Some(integer) = integer_value(value) {
        return integer.map_err(PyIntError::Invalid);
    }
    if let Some(number) = value.as_f64() {
        if !number.is_finite() {
            return Err(PyIntError::Overflow(format!(
                "{label} cannot convert non-finite float {number} to Python int"
            )));
        }
        return BigInt::from_f64(number.trunc())
            .ok_or_else(|| PyIntError::Overflow(format!("{label} does not fit a Python integer")));
    }
    if value.is_number() {
        let rendered = value.to_string();
        let number = rendered.parse::<f64>().map_err(|_| {
            PyIntError::Invalid(format!("{label} cannot parse {value} as a Python number"))
        })?;
        if !number.is_finite() {
            return Err(PyIntError::Overflow(format!(
                "{label} cannot convert non-finite float {number} to Python int"
            )));
        }
        return BigInt::from_f64(number.trunc())
            .ok_or_else(|| PyIntError::Overflow(format!("{label} does not fit a Python integer")));
    }
    if let Some(text) = value.as_str() {
        let text = text.trim();
        let (negative, digits) = if let Some(digits) = text.strip_prefix('-') {
            (true, digits)
        } else {
            (false, text.strip_prefix('+').unwrap_or(text))
        };
        // Python permits a single underscore only between decimal digits.  Do
        // not blindly strip separators: malformed action counts are ignored by
        // `_parse_order` and raise from PICKUP/PLACE rather than being executed.
        // The action protocol intentionally supports ASCII decimal strings; the
        // framework's JSON agents do not rely on PyLong's wider Unicode digits.
        let bytes = digits.as_bytes();
        let valid = !bytes.is_empty()
            && bytes.iter().enumerate().all(|(index, byte)| {
                byte.is_ascii_digit()
                    || (*byte == b'_'
                        && index > 0
                        && index + 1 < bytes.len()
                        && bytes[index - 1].is_ascii_digit()
                        && bytes[index + 1].is_ascii_digit())
            });
        if !valid {
            return Err(PyIntError::Invalid(format!(
                "{label} cannot parse {value} as a Python integer"
            )));
        }
        if bytes.iter().filter(|byte| byte.is_ascii_digit()).count() > CPYTHON_INT_MAX_STR_DIGITS {
            return Err(PyIntError::Invalid(format!(
                "{label} exceeds CPython's {CPYTHON_INT_MAX_STR_DIGITS}-digit string conversion limit"
            )));
        }
        let mut normalized = digits.replace('_', "");
        if negative {
            normalized.insert(0, '-');
        }
        return BigInt::from_str(&normalized).map_err(|_| {
            PyIntError::Invalid(format!("{label} cannot parse {value} as a Python integer"))
        });
    }
    Err(PyIntError::Invalid(format!(
        "{label} cannot convert {value} to a Python integer"
    )))
}

fn schema_integer(value: &Value, label: &str) -> Result<BigInt, String> {
    if !value.is_number() {
        return Err(format!(
            "{label} must be an integral JSON number, got {value}"
        ));
    }
    if let Some(integer) = integer_value(value) {
        return integer;
    }
    let number = value
        .as_f64()
        .filter(|number| number.is_finite() && number.fract() == 0.0)
        .ok_or_else(|| format!("{label} must be a finite integral number, got {value}"))?;
    BigInt::from_f64(number).ok_or_else(|| format!("{label} cannot convert {value} to an integer"))
}

fn python_int_i64(value: &Value, label: &str) -> Result<i64, PyIntError> {
    let integer = python_int(value, label)?;
    Ok(integer.to_i64().unwrap_or_else(|| {
        if integer < BigInt::from(0) {
            i64::MIN
        } else {
            i64::MAX
        }
    }))
}

impl PyInt {
    fn is_negative(&self) -> bool {
        self.0 < BigInt::from(0)
    }

    fn is_positive(&self) -> bool {
        self.0 > BigInt::from(0)
    }

    fn to_f64(&self, label: &str) -> Result<f64, String> {
        self.0
            .to_f64()
            .filter(|number| number.is_finite())
            .ok_or_else(|| format!("{label} is too large to convert to Python float"))
    }

    fn capped_usize(&self) -> usize {
        self.0.to_usize().unwrap_or(usize::MAX)
    }

    fn div_rem_usize(&self, numerator: usize) -> (usize, usize) {
        if let Some(divisor) = self.0.to_usize() {
            (numerator / divisor, numerator % divisor)
        } else {
            (0, numerator)
        }
    }

    fn divides_usize(&self, numerator: usize) -> bool {
        if let Some(divisor) = self.0.to_usize() {
            numerator.is_multiple_of(divisor)
        } else {
            numerator == 0
        }
    }

    fn room_from_i64_total(&self, total: i64) -> i64 {
        let room = &self.0 - BigInt::from(total);
        if room <= BigInt::from(0) {
            0
        } else {
            room.to_i64().unwrap_or(i64::MAX)
        }
    }

    fn contains_i64_total(&self, total: i64) -> bool {
        BigInt::from(total) < self.0
    }
}

impl From<i64> for PyInt {
    fn from(value: i64) -> Self {
        Self(BigInt::from(value))
    }
}

impl From<usize> for PyInt {
    fn from(value: usize) -> Self {
        Self(BigInt::from(value))
    }
}

impl Serialize for PyInt {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let number = Number::from_str(&self.0.to_string()).map_err(serde::ser::Error::custom)?;
        number.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PyInt {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        schema_integer(&value, "configuration integer")
            .map(Self)
            .map_err(serde::de::Error::custom)
    }
}

fn python_float(value: &Value, label: &str) -> Result<f64, String> {
    if let Some(integer) = integer_value(value) {
        return integer?
            .to_f64()
            .filter(|number| number.is_finite())
            .ok_or_else(|| format!("{label} integer is too large to convert to Python float"));
    }
    if let Some(number) = value.as_f64() {
        return Ok(number);
    }
    if value.is_number() {
        return value
            .to_string()
            .parse::<f64>()
            .map_err(|error| format!("{label} is not a Python float: {error}"));
    }
    if let Some(text) = value.as_str() {
        return text
            .trim()
            .parse::<f64>()
            .map_err(|error| format!("{label} is not a Python float: {error}"));
    }
    Err(format!("{label} cannot convert {value} to Python float"))
}

fn nonnegative_python_number(value: &Value) -> bool {
    if !value.is_number() {
        return false;
    }
    if let Some(integer) = integer_value(value) {
        return integer.is_ok_and(|integer| integer >= BigInt::from(0));
    }
    value.is_number() && python_float(value, "configuration number").is_ok_and(|n| n >= 0.0)
}

fn small_integer(value: &Value) -> Option<i128> {
    if let Some(value) = value.as_i64() {
        return Some(i128::from(value));
    }
    if let Some(value) = value.as_u64() {
        return Some(i128::from(value));
    }
    value
        .as_bool()
        .map(|boolean| i128::from(i64::from(boolean)))
}

fn numeric_less(left: &Value, right: &Value, label: &str) -> Result<bool, String> {
    if let (Some(left), Some(right)) = (small_integer(left), small_integer(right)) {
        return Ok(left < right);
    }
    if let (Some(left), Some(right)) = (integer_value(left), integer_value(right)) {
        return Ok(left? < right?);
    }
    if let Some(left) = integer_value(left) {
        let left = BigRational::from_integer(left?);
        let right = numeric(right, label)?;
        if right.is_nan() {
            return Ok(false);
        }
        if right == f64::INFINITY {
            return Ok(true);
        }
        if right == f64::NEG_INFINITY {
            return Ok(false);
        }
        return Ok(left < BigRational::from_float(right).expect("finite f64 is rational"));
    }
    if let Some(right) = integer_value(right) {
        let left = numeric(left, label)?;
        if left.is_nan() || left == f64::INFINITY {
            return Ok(false);
        }
        if left == f64::NEG_INFINITY {
            return Ok(true);
        }
        return Ok(
            BigRational::from_float(left).expect("finite f64 is rational")
                < BigRational::from_integer(right?),
        );
    }
    Ok(numeric(left, label)? < numeric(right, label)?)
}

fn numeric_distance(left: &Value, right: &Value, label: &str) -> Result<f64, String> {
    if let (Some(left), Some(right)) = (small_integer(left), small_integer(right)) {
        return Ok((left - right) as f64);
    }
    if let (Some(left), Some(right)) = (integer_value(left), integer_value(right)) {
        return (left? - right?)
            .to_f64()
            .ok_or_else(|| format!("{label} difference does not fit a finite float"));
    }
    Ok(numeric(left, label)? - numeric(right, label)?)
}

fn integer_json(value: &BigInt) -> Result<Value, String> {
    if let Some(value) = value.to_i64() {
        return Ok(Value::from(value));
    }
    if let Some(value) = value.to_u64() {
        return Ok(Value::from(value));
    }
    serde_json::from_str(&value.to_string())
        .map_err(|error| format!("cannot encode Python integer {value}: {error}"))
}

fn param_number(params: &MarketParams, item: &str, key: &str) -> Result<f64, String> {
    let value = params
        .get(item)
        .and_then(|param| param.get(key))
        .ok_or_else(|| format!("marketParams[{item:?}][{key:?}] is missing"))?;
    numeric(value, &format!("marketParams[{item:?}][{key:?}]"))
}

fn param_string<'a>(params: &'a MarketParams, item: &str, key: &str) -> &'a str {
    params
        .get(item)
        .and_then(|param| param.get(key))
        .and_then(Value::as_str)
        .unwrap_or("")
}

/// Add an integer market flow while retaining Python's numeric representation:
/// int +/- int remains int, float +/- int remains float, and bool arithmetic becomes int.
fn add_market_units(value: &mut Value, delta: i64) -> Result<(), String> {
    *value = match value {
        Value::Number(number) if number.is_i64() => {
            let result = i128::from(number.as_i64().expect("checked i64")) + i128::from(delta);
            if let Ok(result) = i64::try_from(result) {
                Value::from(result)
            } else if let Ok(result) = u64::try_from(result) {
                Value::from(result)
            } else {
                integer_json(&BigInt::from(result))?
            }
        }
        Value::Number(number) if number.is_u64() => {
            let result = i128::from(number.as_u64().expect("checked u64")) + i128::from(delta);
            if let Ok(result) = u64::try_from(result) {
                Value::from(result)
            } else if let Ok(result) = i64::try_from(result) {
                Value::from(result)
            } else {
                integer_json(&BigInt::from(result))?
            }
        }
        Value::Number(number) => {
            let rendered = number.to_string();
            if !rendered.contains(['.', 'e', 'E']) {
                let result = BigInt::from_str(&rendered)
                    .map_err(|error| format!("invalid integer inventory: {error}"))?
                    + BigInt::from(delta);
                serde_json::from_str(&result.to_string())
                    .map_err(|error| format!("cannot encode market inventory: {error}"))?
            } else {
                let number = number
                    .as_f64()
                    .ok_or_else(|| format!("market inventory {rendered} is not finite"))?;
                Value::from(number + delta as f64)
            }
        }
        Value::Bool(boolean) => Value::from(i64::from(*boolean) + delta),
        other => {
            return Err(format!(
                "official engine rejects non-numeric market inventory {other}"
            ));
        }
    };
    Ok(())
}

fn make_market(params: &MarketParams, expose_params: bool) -> Result<Market, String> {
    let mut inventory = IndexMap::new();
    let mut prices = IndexMap::new();
    for item in PRODUCTS {
        let param = params
            .get(item)
            .ok_or_else(|| format!("market params have no {item}"))?;
        let i0 = param
            .get("I0")
            .ok_or_else(|| format!("marketParams[{item:?}][\"I0\"] is missing"))?;
        let base = param
            .get("base")
            .ok_or_else(|| format!("marketParams[{item:?}][\"base\"] is missing"))?;
        inventory.insert(item.to_string(), i0.clone());
        prices.insert(item.to_string(), base.clone());
    }
    Ok(Market {
        inventory,
        prices,
        params: expose_params.then(|| params.clone()),
    })
}

fn deserialize_integral_usize<'de, D>(deserializer: D) -> Result<usize, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;
    if let Some(value) = value.as_u64() {
        return usize::try_from(value).map_err(serde::de::Error::custom);
    }
    if let Some(value) = value.as_i64() {
        return usize::try_from(value).map_err(serde::de::Error::custom);
    }
    if let Some(value) = value.as_f64()
        && value.is_finite()
        && value.fract() == 0.0
        && value >= 0.0
        && value <= usize::MAX as f64
    {
        return Ok(value as usize);
    }
    Err(serde::de::Error::custom(format!(
        "expected a non-negative integral number, got {value}"
    )))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    #[serde(rename = "episodeSteps")]
    pub episode_steps: PyInt,
    #[serde(rename = "boardSize")]
    #[serde(deserialize_with = "deserialize_integral_usize")]
    pub board_size: usize,
    #[serde(rename = "startingMoney")]
    pub starting_money: PyInt,
    #[serde(rename = "maxMarketOrdersPerTurn")]
    pub max_market_orders_per_turn: PyInt,
    #[serde(rename = "turnsPerDay")]
    pub turns_per_day: PyInt,
    #[serde(rename = "shedCapacity")]
    pub shed_capacity: PyInt,
    #[serde(rename = "weedSpawnChance")]
    pub weed_spawn_chance: Value,
    #[serde(rename = "townShopUnlockInterval")]
    pub town_shop_unlock_interval: PyInt,
    #[serde(rename = "townShopSellInterval")]
    pub town_shop_sell_interval: PyInt,
    #[serde(rename = "townCenterSellInterval")]
    pub town_center_sell_interval: PyInt,
    #[serde(rename = "farmHandCostMult")]
    pub farm_hand_cost_mult: PyInt,
    #[serde(rename = "marketParams")]
    pub market_params: IndexMap<String, Value>,
    /// Framework/forward-compatible configuration fields do not affect the current
    /// kernel but are retained rather than silently erased.
    #[serde(flatten)]
    pub extra: IndexMap<String, Value>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            episode_steps: 720_i64.into(),
            board_size: 10,
            starting_money: 3_000_i64.into(),
            max_market_orders_per_turn: 10_i64.into(),
            turns_per_day: 24_i64.into(),
            shed_capacity: 100_i64.into(),
            weed_spawn_chance: Value::from(0.005),
            town_shop_unlock_interval: 3_i64.into(),
            town_shop_sell_interval: 4_i64.into(),
            town_center_sell_interval: 24_i64.into(),
            farm_hand_cost_mult: 1_i64.into(),
            market_params: IndexMap::new(),
            extra: IndexMap::new(),
        }
    }
}

impl Config {
    fn validate(&self) -> Result<(), String> {
        if !self.episode_steps.is_positive() {
            return Err("episodeSteps must be at least 1".to_string());
        }
        if self.board_size < 4 {
            return Err("boardSize must be at least 4".to_string());
        }
        if self.starting_money.is_negative() {
            return Err("startingMoney must be non-negative".to_string());
        }
        if !self.max_market_orders_per_turn.is_positive() {
            return Err("maxMarketOrdersPerTurn must be at least 1".to_string());
        }
        if !self.turns_per_day.is_positive() {
            return Err("turnsPerDay must be at least 1".to_string());
        }
        if !self.shed_capacity.is_positive() {
            return Err("shedCapacity must be at least 1".to_string());
        }
        if !nonnegative_python_number(&self.weed_spawn_chance) {
            return Err("weedSpawnChance must be a non-negative number".to_string());
        }
        if !self.town_shop_unlock_interval.is_positive() {
            return Err("townShopUnlockInterval must be at least 1".to_string());
        }
        if !self.town_shop_sell_interval.is_positive() {
            return Err("townShopSellInterval must be at least 1".to_string());
        }
        if !self.town_center_sell_interval.is_positive() {
            return Err("townCenterSellInterval must be at least 1".to_string());
        }
        if self.farm_hand_cost_mult.is_negative() {
            return Err("farmHandCostMult must be non-negative".to_string());
        }
        Ok(())
    }
}

pub type Counts = IndexMap<String, i64>;
pub type Inventory = IndexMap<String, i64>;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Farm {
    pub money: f64,
    pub tiles: Vec<Vec<Value>>,
    pub farmer: Vec<i64>,
    pub hands: Vec<Vec<i64>>,
    pub unlocked_quadrants: Vec<String>,
    pub hires_today: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrivateState {
    pub shed: Counts,
    pub seeds: Counts,
    pub inventories: Vec<Inventory>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Market {
    /// These are normally integers. Custom `I0`/`base` overrides may be floats,
    /// and Python preserves that numeric representation until arithmetic/refresh.
    pub inventory: IndexMap<String, Value>,
    pub prices: IndexMap<String, Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<MarketParams>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Town {
    pub unlocked_shops: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PublicState {
    pub step: usize,
    pub day: usize,
    pub hour: usize,
    pub farms: Vec<Farm>,
    pub market: Market,
    pub town: Town,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InitialState {
    pub public: PublicState,
    pub privates: Vec<PrivateState>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RngDay {
    pub day: usize,
    pub weed_rolls: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceHeader {
    pub format: String,
    pub seed: Number,
    pub configuration: Config,
    pub shop_schedule: Vec<String>,
    pub rng_schedule: Vec<RngDay>,
    pub initial: InitialState,
    pub terminal_banks: Vec<f64>,
    pub transitions: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Transition {
    pub actions: Vec<Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplayInput {
    pub header: TraceHeader,
    pub transitions: Vec<Transition>,
}

#[derive(Clone, Debug, Serialize)]
pub struct StepSnapshot {
    pub public: PublicState,
    pub privates: Vec<PrivateState>,
    pub done: bool,
    pub statuses: Vec<String>,
    pub rewards: Vec<f64>,
}

/// Aggregate market intent/execution telemetry for one batched environment step.
/// Counts observe official engine effects only; they never alter action handling.
#[derive(Clone, Debug, Default)]
pub struct MarketSeatMetrics {
    pub submitted_orders: u64,
    pub zero_commit_orders: u64,
    pub committed_orders: u64,
    pub committed_units: u64,
    pub sell_orders: u64,
    pub buy_orders: u64,
    pub hire_orders: u64,
    pub land_orders: u64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct MarketStepMetrics {
    pub submitted_orders: u64,
    pub zero_commit_orders: u64,
    pub committed_orders: u64,
    pub committed_units: u64,
    pub sell_orders: u64,
    pub buy_orders: u64,
    pub hire_orders: u64,
    pub land_orders: u64,
    #[serde(skip)]
    pub(crate) by_seat: [MarketSeatMetrics; 2],
}

/// Numeric-only upstream counters; all keys are present even when zero.
const UPSTREAM_VERBS: [(&str, &str, &str); 17] = [
    (
        "NORTH",
        "upstream_submitted_NORTH",
        "upstream_committed_NORTH",
    ),
    (
        "SOUTH",
        "upstream_submitted_SOUTH",
        "upstream_committed_SOUTH",
    ),
    ("EAST", "upstream_submitted_EAST", "upstream_committed_EAST"),
    ("WEST", "upstream_submitted_WEST", "upstream_committed_WEST"),
    ("DROP", "upstream_submitted_DROP", "upstream_committed_DROP"),
    (
        "PICKUP",
        "upstream_submitted_PICKUP",
        "upstream_committed_PICKUP",
    ),
    (
        "PLACE",
        "upstream_submitted_PLACE",
        "upstream_committed_PLACE",
    ),
    (
        "PLANT",
        "upstream_submitted_PLANT",
        "upstream_committed_PLANT",
    ),
    (
        "WATER",
        "upstream_submitted_WATER",
        "upstream_committed_WATER",
    ),
    (
        "HARVEST",
        "upstream_submitted_HARVEST",
        "upstream_committed_HARVEST",
    ),
    (
        "FERTILIZE",
        "upstream_submitted_FERTILIZE",
        "upstream_committed_FERTILIZE",
    ),
    ("DIG", "upstream_submitted_DIG", "upstream_committed_DIG"),
    (
        "BUILD_COOP",
        "upstream_submitted_BUILD_COOP",
        "upstream_committed_BUILD_COOP",
    ),
    (
        "BUILD_PASTURE",
        "upstream_submitted_BUILD_PASTURE",
        "upstream_committed_BUILD_PASTURE",
    ),
    ("FEED", "upstream_submitted_FEED", "upstream_committed_FEED"),
    (
        "COLLECT_FERTILIZER",
        "upstream_submitted_COLLECT_FERTILIZER",
        "upstream_committed_COLLECT_FERTILIZER",
    ),
    ("CARE", "upstream_submitted_CARE", "upstream_committed_CARE"),
];

fn upstream_counts() -> BTreeMap<&'static str, u64> {
    UPSTREAM_VERBS
        .iter()
        .flat_map(|(_, submitted, committed)| [(*submitted, 0), (*committed, 0)])
        .chain([
            ("upstream_observed_seat_steps", 0),
            ("upstream_crop_drought_deaths", 0),
            ("upstream_crop_expiry_deaths", 0),
            ("upstream_animal_starvation_deaths", 0),
            ("upstream_crop_tile_steps", 0),
            ("upstream_animal_tile_steps", 0),
            ("upstream_ready_crop_tile_steps", 0),
            ("upstream_ready_animal_tile_steps", 0),
            ("upstream_ready_product_unit_steps", 0),
        ])
        .collect()
}

fn accumulate_upstream(
    into: &mut BTreeMap<&'static str, u64>,
    other: &BTreeMap<&'static str, u64>,
) {
    for (&key, &value) in other {
        *into.entry(key).or_default() += value;
    }
}

/// Aggregate unit intent/effect telemetry for one seat.  "Production" is the
/// farm-working action family (plant/water/harvest/fertilize/dig/build/feed/care/
/// collect-fertilizer); shed and placement actions are reported as logistics.
#[derive(Clone, Debug)]
pub struct UnitSeatMetrics {
    pub submitted_move_actions: u64,
    pub committed_move_actions: u64,
    pub submitted_production_actions: u64,
    pub committed_production_actions: u64,
    pub submitted_logistics_actions: u64,
    pub committed_logistics_actions: u64,
    pub upstream: BTreeMap<&'static str, u64>,
}
impl Default for UnitSeatMetrics {
    fn default() -> Self {
        Self {
            submitted_move_actions: 0,
            committed_move_actions: 0,
            submitted_production_actions: 0,
            committed_production_actions: 0,
            submitted_logistics_actions: 0,
            committed_logistics_actions: 0,
            upstream: upstream_counts(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct UnitStepMetrics {
    pub submitted_move_actions: u64,
    pub committed_move_actions: u64,
    pub submitted_production_actions: u64,
    pub committed_production_actions: u64,
    pub submitted_logistics_actions: u64,
    pub committed_logistics_actions: u64,
    #[serde(skip)]
    pub(crate) by_seat: [UnitSeatMetrics; 2],
    #[serde(flatten)]
    pub upstream: BTreeMap<&'static str, u64>,
}
impl Default for UnitStepMetrics {
    fn default() -> Self {
        Self {
            submitted_move_actions: 0,
            committed_move_actions: 0,
            submitted_production_actions: 0,
            committed_production_actions: 0,
            submitted_logistics_actions: 0,
            committed_logistics_actions: 0,
            upstream: upstream_counts(),
            by_seat: Default::default(),
        }
    }
}

impl UnitStepMetrics {
    pub fn accumulate(&mut self, other: &Self) {
        self.submitted_move_actions += other.submitted_move_actions;
        self.committed_move_actions += other.committed_move_actions;
        self.submitted_production_actions += other.submitted_production_actions;
        self.committed_production_actions += other.committed_production_actions;
        self.submitted_logistics_actions += other.submitted_logistics_actions;
        self.committed_logistics_actions += other.committed_logistics_actions;
        accumulate_upstream(&mut self.upstream, &other.upstream);
        for seat in 0..self.by_seat.len() {
            self.by_seat[seat].accumulate(&other.by_seat[seat]);
        }
    }

    pub fn accumulate_seat(&mut self, other: &UnitSeatMetrics) {
        self.submitted_move_actions += other.submitted_move_actions;
        self.committed_move_actions += other.committed_move_actions;
        self.submitted_production_actions += other.submitted_production_actions;
        self.committed_production_actions += other.committed_production_actions;
        self.submitted_logistics_actions += other.submitted_logistics_actions;
        self.committed_logistics_actions += other.committed_logistics_actions;
        accumulate_upstream(&mut self.upstream, &other.upstream);
    }
}

impl UnitSeatMetrics {
    fn accumulate(&mut self, other: &Self) {
        self.submitted_move_actions += other.submitted_move_actions;
        self.committed_move_actions += other.committed_move_actions;
        self.submitted_production_actions += other.submitted_production_actions;
        self.committed_production_actions += other.committed_production_actions;
        self.submitted_logistics_actions += other.submitted_logistics_actions;
        self.committed_logistics_actions += other.committed_logistics_actions;
        accumulate_upstream(&mut self.upstream, &other.upstream);
    }
}

#[derive(Clone, Debug, Default)]
pub struct StepMetrics {
    pub market: MarketStepMetrics,
    pub unit: UnitStepMetrics,
}

impl MarketStepMetrics {
    pub fn accumulate(&mut self, other: &Self) {
        self.submitted_orders += other.submitted_orders;
        self.zero_commit_orders += other.zero_commit_orders;
        self.committed_orders += other.committed_orders;
        self.committed_units += other.committed_units;
        self.sell_orders += other.sell_orders;
        self.buy_orders += other.buy_orders;
        self.hire_orders += other.hire_orders;
        self.land_orders += other.land_orders;
        for seat in 0..self.by_seat.len() {
            self.by_seat[seat].submitted_orders += other.by_seat[seat].submitted_orders;
            self.by_seat[seat].zero_commit_orders += other.by_seat[seat].zero_commit_orders;
            self.by_seat[seat].committed_orders += other.by_seat[seat].committed_orders;
            self.by_seat[seat].committed_units += other.by_seat[seat].committed_units;
            self.by_seat[seat].sell_orders += other.by_seat[seat].sell_orders;
            self.by_seat[seat].buy_orders += other.by_seat[seat].buy_orders;
            self.by_seat[seat].hire_orders += other.by_seat[seat].hire_orders;
            self.by_seat[seat].land_orders += other.by_seat[seat].land_orders;
        }
    }

    pub fn accumulate_seat(&mut self, other: &MarketSeatMetrics) {
        self.submitted_orders += other.submitted_orders;
        self.zero_commit_orders += other.zero_commit_orders;
        self.committed_orders += other.committed_orders;
        self.committed_units += other.committed_units;
        self.sell_orders += other.sell_orders;
        self.buy_orders += other.buy_orders;
        self.hire_orders += other.hire_orders;
        self.land_orders += other.land_orders;
    }
}

#[derive(Clone, Debug)]
enum ResolvedSeed {
    Integer(BigInt),
    NonInteger(Number),
}

fn resolved_seed(number: &Number) -> Result<ResolvedSeed, String> {
    let rendered = number.to_string();
    if rendered.contains(['.', 'e', 'E']) {
        Ok(ResolvedSeed::NonInteger(number.clone()))
    } else {
        BigInt::from_str(&rendered)
            .map(ResolvedSeed::Integer)
            .map_err(|error| format!("invalid integer seed: {error}"))
    }
}

/// `Game::econ_counters` layout (per seat). 0-2 drive the opt-in econ shaping; the rest
/// are its telemetry (commands and PASS shares, productive commits, sales).
pub const ECON_STARVATION: usize = 0;
pub const ECON_DROUGHT: usize = 1;
pub const ECON_INEFFECTIVE: usize = 2;
pub const ECON_COMMANDS: usize = 3;
pub const ECON_PASS: usize = 4;
pub const ECON_HARVEST: usize = 5;
pub const ECON_WATER: usize = 6;
pub const ECON_FEED: usize = 7;
pub const ECON_SELL_UNITS: usize = 8;
pub const ECON_SELL_CASH: usize = 9;
/// Version 4 loss counters (owner 2026-09-27: every loss the engine can cause, for econ shaping):
/// plant units lost to decay past `max_lifespan_step` (a decrement, or the units still held when the
/// plant turns WEED); goods discarded because the shed was full (DROP or the end-of-day deposit);
/// animal production lost at the day end (a pending care bonus cleared by an unfed production day,
/// plus any increment cut by `max_held`); fertilizer flags already set at a day end (an uncollected
/// unit that cannot accumulate); weeds spawned on the seat's empty tiles; and, set once when the game
/// ends, the goods left unsold (shed and carried units plus yield still held on crop and animal tiles).
pub const ECON_EXPIRY_UNITS: usize = 10;
pub const ECON_OVERFLOW_UNITS: usize = 11;
pub const ECON_CARE_LOST: usize = 12;
pub const ECON_FERT_WASTED: usize = 13;
pub const ECON_WEEDS: usize = 14;
pub const ECON_UNSOLD_END: usize = 15;
/// Destroyed held crop/animal yield + pending care units + fertilizer flags on drought/starvation death (units).
pub const ECON_DEATH_HELD_UNITS: usize = 16;
/// Crop increments rejected by max_yield on WATER or ongoing production (units).
pub const ECON_CLIPPED_UNITS: usize = 17;
/// Surviving unwatered growth/bonus and nonongoing early-harvest max_yield minus held (potential units).
pub const ECON_MISSED_GROWTH_UNITS: usize = 18;
/// Live PLANT DIG destroys held yield plus one plant (units).
pub const ECON_DUG_UNITS: usize = 19;
/// Committed FERTILIZE whose expiry does not extend (fertilizer units).
pub const ECON_REDUNDANT_FERT: usize = 20;
/// cared_today without fed_today at day end, including deaths (animal-days).
pub const ECON_CARE_WASTED: usize = 21;
/// Terminal animal fertilizer flags + pending care units + fed_today flags (units).
pub const ECON_TERMINAL_FLAGS: usize = 22;
/// Terminal remaining seeds (units).
pub const ECON_SEEDS_UNUSED_END: usize = 23;
/// Terminal bought quadrants never used by a PLANT or placed animal (land-price cash).
pub const ECON_UNUSED_LAND_CASH: usize = 24;
/// Paid HIRE with no next work slot before EOD or episode termination (cash).
pub const ECON_HIRE_WASTED_CASH: usize = 25;
/// Existing hired-hand PASS or omitted slots, farmer excluded (worker-steps).
pub const ECON_IDLE_HAND_STEPS: usize = 26;
/// Parsed requested minus committed market units; malformed/truncated slots each add one (mixed intent units).
pub const ECON_MARKET_UNFILLED_UNITS: usize = 27;
/// Malformed existing-unit command slots, excluding PASS and valid but ineffective syntax (slots).
pub const ECON_MALFORMED_UNIT_CMDS: usize = 28;
/// Committed SELL sum max(official product base minus actual price, 0) (reference cash gap).
pub const ECON_SALE_SHORTFALL_CASH: usize = 29;
/// Committed SELL at price <= 1 (units).
pub const ECON_FLOOR_SALE_UNITS: usize = 30;
/// Committed product/seed purchase sum max(actual minus official base/crop seed cost, 0) (reference cash gap).
pub const ECON_BUY_PREMIUM_CASH: usize = 31;
pub const ECON_FIELDS: usize = 32;

// A test-only sink lets the identical transition kernel run without storing any economic
// counters. The predicate compiles to true in production; no runtime feature/ABI change.
macro_rules! econ_mut {
    ($game:expr, $player:expr) => {
        $game.econ_counters.get_mut($player).filter(|_| {
            #[cfg(test)] { !$game.econ_counting_disabled }
            #[cfg(not(test))] { true }
        })
    };
}

#[derive(Clone, Debug)]
pub struct Game {
    config: Config,
    seed: ResolvedSeed,
    params: MarketParams,
    farms: Vec<Farm>,
    privates: Vec<PrivateState>,
    market: Market,
    town: Town,
    step: usize,
    done: bool,
    rewards: Option<Vec<f64>>,
    /// Cumulative economic counters per seat since this `Game` was constructed, indexed
    /// by the `ECON_*` constants: deaths (the step metrics' `upstream_*_deaths`),
    /// ineffective unit commands (submitted minus committed over every upstream verb;
    /// PASS is not a verb; a seed-blocked PLANT is submitted, not committed), and the
    /// telemetry counters (unit commands of existing units, PASS commands, committed
    /// HARVEST/WATER/FEED, committed SELL units and their cash). Not part of any
    /// snapshot or observation; cloned with the state (fork/clone). A fixed two-seat
    /// array (no per-step allocation).
    econ_counters: [[u64; ECON_FIELDS]; 2],
    /// Observational lifetime land-use provenance; never serialized into game observations.
    econ_land_used: [[bool; 3]; 2],
    /// Economic attribution ledger per seat (`econ_attrib`, `A_*` layout); telemetry only, like `econ_counters`.
    attrib: [[u64; ATTRIB_FIELDS]; 2],
    #[cfg(test)]
    econ_counting_disabled: bool,
}

impl Game {
    /// Construct the official initial state from configuration and a resolved seed.
    /// The seed is already hidden/persisted by `resolve_episode_seed` on Python's side;
    /// Rust accepts it explicitly so batched rollout reset has no framework dependency.
    pub fn new(config: Config, seed: i64, num_agents: usize) -> Result<Self, String> {
        Self::new_with_seed(config, BigInt::from(seed), num_agents)
    }

    /// Arbitrary-width integer-seed reset matching Python's unbounded `int`.
    pub fn new_with_seed_decimal(
        config: Config,
        seed: &str,
        num_agents: usize,
    ) -> Result<Self, String> {
        let seed =
            BigInt::from_str(seed).map_err(|error| format!("invalid integer seed: {error}"))?;
        Self::new_with_seed(config, seed, num_agents)
    }

    fn new_with_seed(config: Config, seed: BigInt, num_agents: usize) -> Result<Self, String> {
        if num_agents != 2 {
            return Err(format!(
                "Kaggriculture requires exactly 2 agents, got {num_agents}"
            ));
        }
        config.validate()?;

        let params = resolve_market_params(&config.market_params);
        let starting_money = config.starting_money.to_f64("startingMoney")?;
        let farms = (0..num_agents)
            .map(|_| Farm {
                money: starting_money,
                tiles: (0..config.board_size)
                    .map(|y| {
                        (0..config.board_size)
                            .map(|x| {
                                if quadrant_of(x, y, config.board_size) == "NW" {
                                    Value::Null
                                } else {
                                    Value::String("LOCKED".to_string())
                                }
                            })
                            .collect()
                    })
                    .collect(),
                farmer: default_spawn(config.board_size),
                hands: vec![],
                unlocked_quadrants: vec!["NW".to_string()],
                hires_today: 0,
            })
            .collect();
        let privates = (0..num_agents)
            .map(|_| PrivateState {
                shed: PRODUCTS
                    .into_iter()
                    .chain(ANIMAL_NAMES)
                    .map(|item| (item.to_string(), 0))
                    .collect(),
                seeds: CROP_NAMES
                    .into_iter()
                    .map(|item| (item.to_string(), 0))
                    .collect(),
                inventories: vec![IndexMap::new()],
            })
            .collect();
        let market = make_market(&params, !config.market_params.is_empty())?;

        Ok(Self {
            config,
            seed: ResolvedSeed::Integer(seed),
            params,
            farms,
            privates,
            market,
            town: Town {
                unlocked_shops: vec![],
            },
            step: 0,
            done: false,
            rewards: None,
            econ_counters: [[0; ECON_FIELDS]; 2],
            #[cfg(test)]
            econ_counting_disabled: false,
            econ_land_used: [[false; 3]; 2],
            attrib: [[0; ATTRIB_FIELDS]; 2],
        })
    }

    /// Native reset for a trace payload. The header's initial state and exported
    /// schedules remain independent parity oracles; they do not initialize Rust.
    pub fn from_seed_header(header: &TraceHeader) -> Result<Self, String> {
        Self::validate_header(header)?;
        let mut game = Self::new_with_seed(
            header.configuration.clone(),
            BigInt::from(0),
            header.initial.privates.len(),
        )?;
        game.seed = resolved_seed(&header.seed)?;
        Ok(game)
    }

    fn validate_header(header: &TraceHeader) -> Result<(), String> {
        if header.format != TRACE_FORMAT {
            return Err(format!(
                "unsupported trace format {:?}; expected {TRACE_FORMAT:?}",
                header.format
            ));
        }
        if header.initial.public.farms.len() != header.initial.privates.len() {
            return Err("initial farms/private counts differ".to_string());
        }
        header.configuration.validate()?;
        Ok(())
    }

    /// Load an explicit state oracle. This is useful for tiny first-divergence
    /// regression cases; rollout and the default replay CLI use native reset above.
    pub fn from_header(header: &TraceHeader) -> Result<Self, String> {
        Self::validate_header(header)?;
        let public = &header.initial.public;
        let seed = resolved_seed(&header.seed)?;
        // Runtime Python pricing reads `market.get("params")` and then applies
        // `(params or MARKET_PARAMS)`: missing *or empty* explicit-state params use
        // defaults, even if the unrelated configuration still contains overrides.
        let params = public
            .market
            .params
            .clone()
            .filter(|params| !params.is_empty())
            .unwrap_or_else(default_market_params);
        Ok(Self {
            config: header.configuration.clone(),
            seed,
            params,
            farms: public.farms.clone(),
            privates: header.initial.privates.clone(),
            market: public.market.clone(),
            town: public.town.clone(),
            step: public.step,
            done: false,
            rewards: None,
            econ_counters: [[0; ECON_FIELDS]; 2],
            #[cfg(test)]
            econ_counting_disabled: false,
            // Imported states have no past provenance: initialize from currently occupied tiles.
            econ_land_used: std::array::from_fn(|p| public.farms.get(p)
                .map(|f| land_use(f, header.configuration.board_size)).unwrap_or([false; 3])),
            attrib: [[0; ATTRIB_FIELDS]; 2],
        })
    }

    /// Per seat `[animal starvation, crop drought, ineffective unit commands]` since
    /// construction; `None` for a (non-competition) state without exactly two seats.
    pub fn econ_counters(&self) -> Option<&[[u64; ECON_FIELDS]; 2]> {
        (self.farms.len() == 2).then_some(&self.econ_counters)
    }

    pub fn public_state(&self) -> PublicState {
        let (day, hour) = self.config.turns_per_day.div_rem_usize(self.step);
        PublicState {
            step: self.step,
            day,
            hour,
            farms: self.farms.clone(),
            market: self.market.clone(),
            town: self.town.clone(),
        }
    }

    pub fn snapshot(&self) -> StepSnapshot {
        StepSnapshot {
            public: self.public_state(),
            privates: self.privates.clone(),
            done: self.done,
            statuses: vec![if self.done { "DONE" } else { "ACTIVE" }.to_string(); self.farms.len()],
            rewards: self
                .rewards
                .clone()
                .unwrap_or_else(|| vec![0.0; self.farms.len()]),
        }
    }

    pub fn terminal_banks(&self) -> Option<&[f64]> {
        self.rewards.as_deref()
    }

    /// Current cash per seat, read without touching the transition.
    pub fn seat_money(&self) -> Vec<f64> {
        self.farms.iter().map(|farm| farm.money).collect()
    }

    /// Conservative mark-to-market wealth per seat for JA26 potential-based credit.
    ///
    /// Cash plus liquid holdings (shed, carried units and harvestable yield on tiles)
    /// at the current market price, plus productive assets at book value: seeds in
    /// hand and living plants at seed cost, animals at purchase cost, extra quadrants
    /// at their land price.  Yield that `HARVEST` would refuse before a plant's first
    /// yield day is not counted.  Structures and weeds are worth nothing.  This is a
    /// read-only function of state: it never enters the transition or the official
    /// terminal reward.
    pub fn seat_wealth(&self) -> Vec<f64> {
        let (day, _) = self.config.turns_per_day.div_rem_usize(self.step);
        let day = i64::try_from(day).unwrap_or(i64::MAX);
        self.farms
            .iter()
            .zip(&self.privates)
            .map(|(farm, private)| seat_wealth(farm, private, &self.market, day))
            .collect()
    }

    pub fn step(&mut self, actions: &[Value]) -> Result<(), String> {
        self.step_with_market_metrics(actions).map(|_| ())
    }

    pub fn step_with_market_metrics(&mut self, actions: &[Value]) -> Result<StepMetrics, String> {
        let (candidate, metrics) = self.stepped_with_market_metrics(actions)?;
        *self = candidate;
        Ok(metrics)
    }

    /// Return a successfully stepped copy while leaving the source game untouched.
    /// Batch callers can stage this candidate without first cloning the source.
    pub fn stepped_with_market_metrics(&self, actions: &[Value]) -> Result<(Self, StepMetrics), String> {
        if self.done {
            return Err("cannot step a completed game".to_string());
        }
        if actions.len() != self.farms.len() {
            return Err(format!(
                "got {} actions for {} farms",
                actions.len(),
                self.farms.len()
            ));
        }

        // The Kaggle framework interprets a copied action state and commits it only
        // when the interpreter returns successfully. Mirror that transaction so a
        // deferred config/custom-market/malformed-count error cannot leak mutations.
        let mut candidate = self.clone();
        let metrics = candidate.step_in_place(actions)?;
        Ok((candidate, metrics))
    }

    fn step_in_place(&mut self, actions: &[Value]) -> Result<StepMetrics, String> {
        let previous_step = self.step;
        let (day, _) = self.config.turns_per_day.div_rem_usize(previous_step);
        let day = i64::try_from(day).map_err(|_| "day index exceeds Rust i64 resources")?;
        let mut unit_metrics = UnitStepMetrics::default();
        for (player, action) in actions.iter().enumerate() {
            let seat_metrics = self.apply_player_actions(player, action, day)?;
            if let Some(counters) = econ_mut!(self, player) {
                counters[ECON_INEFFECTIVE] += seat_metrics.ineffective_unit_commands();
            }
            unit_metrics.by_seat[player] = seat_metrics.clone();
        }
        let market_metrics = self.process_market(actions)?;
        self.town_consume(previous_step)?;
        for (player, farm) in self.farms.iter_mut().enumerate() {
            let (deaths, units_lost) = decay_plants(farm, previous_step);
            *unit_metrics.by_seat[player]
                .upstream
                .get_mut("upstream_crop_expiry_deaths")
                .unwrap() += deaths;
            if let Some(counters) = econ_mut!(self, player) {
                counters[ECON_EXPIRY_UNITS] += units_lost;
            }
        }
        if self.config.turns_per_day.divides_usize(previous_step + 1) {
            self.end_of_day(day as usize, &mut unit_metrics)?;
        }

        self.step = previous_step + 1;
        let (observation_day, _) = self.config.turns_per_day.div_rem_usize(self.step);
        for (player, farm) in self.farms.iter().enumerate() {
            record_upstream_stock(&mut unit_metrics.by_seat[player], farm, observation_day);
            let seat_metrics = unit_metrics.by_seat[player].clone();
            unit_metrics.accumulate_seat(&seat_metrics);
        }
        if BigInt::from(previous_step) >= &self.config.episode_steps.0 - BigInt::from(2) {
            self.done = true;
            self.rewards = Some(self.farms.iter().map(|farm| farm.money).collect());
            for player in 0..self.farms.len() {
                // econ v4: goods left unsold have no terminal value (reward = money only)
                let unsold = held_goods(&self.privates[player]) + held_tile_yield(&self.farms[player]);
                if let Some(counters) = econ_mut!(self, player) {
                    counters[ECON_UNSOLD_END] = unsold;
                    counters[ECON_SEEDS_UNUSED_END] = self.privates[player].seeds.values()
                        .map(|&n| n.max(0) as u64).sum();
                    counters[ECON_TERMINAL_FLAGS] = self.farms[player].tiles.iter().flatten()
                        .filter_map(Value::as_object).filter(|t| t.contains_key("animal"))
                        .map(|t| map_bool(t, "fertilizer_available") as u64
                            + map_i64(t, "pending_care_bonus").max(0) as u64
                            + map_bool(t, "fed_today") as u64).sum();
                    counters[ECON_UNUSED_LAND_CASH] = LAND_ORDER.iter().enumerate()
                        .filter(|(i, q)| !self.econ_land_used[player][*i]
                            && self.farms[player].unlocked_quadrants.iter().any(|v| v == **q))
                        .map(|(i, _)| LAND_PRICES[i] as u64).sum();
                }
            }
        }
        Ok(StepMetrics {
            market: market_metrics,
            unit: unit_metrics,
        })
    }

    fn apply_player_actions(
        &mut self,
        player: usize,
        action: &Value,
        day: i64,
    ) -> Result<UnitSeatMetrics, String> {
        let action_obj = action.as_object();
        let farmer_action = action_obj
            .and_then(|o| o.get("farmer"))
            .cloned()
            .unwrap_or_else(pass_action);
        let hands_actions = action_obj
            .and_then(|o| o.get("hands"))
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();

        self.count_attrib_idle(player, &farmer_action, &hands_actions);
        if let Some(counters) = econ_mut!(self, player) {
            counters[ECON_IDLE_HAND_STEPS] += (0..self.farms[player].hands.len())
                .filter(|&i| hands_actions.get(i).is_none_or(|a| a.as_array()
                    .and_then(|p| p.first()).and_then(Value::as_str) == Some("PASS")))
                .count() as u64;
        }
        let mut plant_demand: IndexMap<String, i64> = IndexMap::new();
        for unit_action in std::iter::once(&farmer_action).chain(hands_actions.iter()) {
            let Some(parts) = unit_action.as_array() else {
                continue;
            };
            if parts.first().and_then(Value::as_str) == Some("PLANT")
                && let Some(name) = parts.get(1).and_then(Value::as_str)
            {
                *plant_demand.entry(name.to_string()).or_default() += 1;
            }
        }
        let blocked: Vec<String> = plant_demand
            .into_iter()
            .filter_map(|(name, demand)| {
                (demand > *self.privates[player].seeds.get(&name).unwrap_or(&0)).then_some(name)
            })
            .collect();

        let allowed = |value: &Value| {
            let crop_name = value
                .as_array()
                .filter(|a| a.first().and_then(Value::as_str) == Some("PLANT"))
                .and_then(|a| a.get(1))
                .and_then(Value::as_str);
            if crop_name.is_some_and(|name| blocked.iter().any(|b| b == name)) {
                pass_action()
            } else {
                value.clone()
            }
        };

        let mut metrics = UnitSeatMetrics::default();
        *metrics
            .upstream
            .get_mut("upstream_observed_seat_steps")
            .unwrap() = 1;
        let farmer_exists = farmer_position(&self.farms[player], 0).is_some();
        record_unit_submission(&mut metrics, &farmer_action, farmer_exists);
        let allowed_farmer_action = allowed(&farmer_action);
        let drop_before = is_drop(&allowed_farmer_action).then(|| held_goods(&self.privates[player]));
        let econ_before = self.econ_unit_before(player, 0, &allowed_farmer_action);
        let transfer = econ_attrib::transfer_before(&self.privates[player], 0, &allowed_farmer_action);
        let malformed = self.econ_malformed_unit(player, 0, &farmer_action);
        let effect = apply_unit_action(
            &mut self.farms[player],
            &mut self.privates[player],
            0,
            &allowed_farmer_action,
            self.config.board_size,
            day,
            &self.config.turns_per_day,
            &self.config.shed_capacity,
        )?;
        self.count_attrib_unit(player, 0, &farmer_action, farmer_exists, effect, econ_before.as_ref(), transfer);
        self.count_econ_tile(player, &allowed_farmer_action, day, econ_before, effect);
        self.count_drop_overflow(player, drop_before);
        record_unit_effect(&mut metrics, &farmer_action, effect);
        self.count_econ_unit(player, &farmer_action, farmer_exists, effect, malformed);
        for (hand_index, hand_action) in hands_actions.iter().enumerate() {
            let hand_exists = farmer_position(&self.farms[player], hand_index + 1).is_some();
            record_unit_submission(&mut metrics, hand_action, hand_exists);
            let allowed_hand_action = allowed(hand_action);
            let drop_before = is_drop(&allowed_hand_action).then(|| held_goods(&self.privates[player]));
            let econ_before = self.econ_unit_before(player, hand_index + 1, &allowed_hand_action);
            let transfer = econ_attrib::transfer_before(&self.privates[player], hand_index + 1, &allowed_hand_action);
            let malformed = self.econ_malformed_unit(player, hand_index + 1, hand_action);
            let effect = apply_unit_action(
                &mut self.farms[player],
                &mut self.privates[player],
                hand_index + 1,
                &allowed_hand_action,
                self.config.board_size,
                day,
                &self.config.turns_per_day,
                &self.config.shed_capacity,
            )?;
            self.count_attrib_unit(player, hand_index + 1, hand_action, hand_exists, effect, econ_before.as_ref(),
                                   transfer);
            self.count_econ_tile(player, &allowed_hand_action, day, econ_before, effect);
            self.count_drop_overflow(player, drop_before);
            record_unit_effect(&mut metrics, hand_action, effect);
            self.count_econ_unit(player, hand_action, hand_exists, effect, malformed);
        }
        Ok(metrics)
    }

    fn econ_malformed_unit(&self, player: usize, unit: usize, action: &Value) -> bool {
        if !malformed_unit(action) { return false; }
        // Animal placement does not inspect its optional quantity, even when inventory is empty.
        if let Some(parts) = action.as_array()
            && parts.first().and_then(Value::as_str) == Some("PLACE")
            && let Some(data) = parts.get(1).and_then(Value::as_str).and_then(animal)
            && let Some((x, y)) = farmer_position(&self.farms[player], unit)
            && x >= 0 && y >= 0
            && let Some(tile) = self.farms[player].tiles.get(y as usize).and_then(|r| r.get(x as usize)).and_then(Value::as_object)
            && map_str(tile, "kind") == Some(data.structure) && !tile.contains_key("animal") {
            return false;
        }
        true
    }

    // Capture only relevant tile commands; gameplay never reads this copy or these counters.
    fn econ_unit_before(&self, player: usize, unit: usize, action: &Value) -> Option<(usize, usize, Value)> {
        #[cfg(test)]
        if self.econ_counting_disabled { return None; }
        let op = action.as_array()?.first()?.as_str()?;
        if !matches!(op, "WATER" | "HARVEST" | "FERTILIZE" | "DIG" | "PLANT" | "PLACE") { return None; }
        let (x, y) = farmer_position(&self.farms[player], unit)?;
        if x < 0 || y < 0 { return None; }
        self.farms[player].tiles.get(y as usize)?.get(x as usize)
            .map(|tile| (x as usize, y as usize, tile.clone()))
    }

    fn count_econ_tile(&mut self, player: usize, action: &Value, day: i64,
        before: Option<(usize, usize, Value)>, effect: UnitActionEffect) {
        if effect == UnitActionEffect::None { return; }
        let Some((x, y, value)) = before else { return; };
        let op = action.as_array().and_then(|a| a.first()).and_then(Value::as_str);
        let after = &self.farms[player].tiles[y][x];
        if matches!(op, Some("PLANT" | "PLACE")) && after.as_object().is_some_and(|t|
            map_str(t, "kind") == Some("PLANT") || t.contains_key("animal")) {
            if let Some(i) = LAND_ORDER.iter().position(|q| *q == quadrant_of(x, y, self.config.board_size)) {
                if let Some(used) = self.econ_land_used.get_mut(player) { used[i] = true; }
            }
        }
        let Some(t) = value.as_object().filter(|t| map_str(t, "kind") == Some("PLANT")) else { return; };
        let Some(data) = map_str(t, "crop").and_then(crop) else { return; };
        let Some(c) = econ_mut!(self, player) else { return; };
        let held = map_i64(t, "yield_units");
        let age = day - map_i64(t, "planted_day");
        let in_window = (data.max_yield_day + 1) / 2 <= age && age <= data.max_yield_day;
        match op {
            Some("WATER") if !data.ongoing && in_window => {
                let increment = if map_i64(t, "fertilized_until_day") >= day { 2 } else { 1 };
                c[ECON_CLIPPED_UNITS] += (held + increment - data.max_yield).max(0) as u64;
            }
            Some("HARVEST") if !data.ongoing && in_window =>
                c[ECON_MISSED_GROWTH_UNITS] += (data.max_yield - held).max(0) as u64,
            Some("DIG") => c[ECON_DUG_UNITS] += held.max(0) as u64 + 1,
            Some("FERTILIZE") if map_i64(t, "fertilized_until_day") >= day + 2 =>
                c[ECON_REDUNDANT_FERT] += 1,
            _ => {}
        }
    }

    /// econ v4: goods a DROP discarded because the shed was full (DROP only moves carried goods into the
    /// shed, so any fall in held goods across it is overflow). `before` is None for other commands.
    fn count_drop_overflow(&mut self, player: usize, before: Option<u64>) {
        if let Some(before) = before {
            let discarded = before.saturating_sub(held_goods(&self.privates[player]));
            if let Some(counters) = econ_mut!(self, player) {
                counters[ECON_OVERFLOW_UNITS] += discarded;
            }
        }
    }

    /// Econ telemetry of one unit's command (the submitted command, as the unit metrics
    /// record it): commands of existing units, PASS commands, and committed HARVEST,
    /// WATER and FEED, plus malformed syntax on noncommitting existing-unit slots.
    fn count_econ_unit(&mut self, player: usize, action: &Value, exists: bool, effect: UnitActionEffect, malformed: bool) {
        let Some(counters) = econ_mut!(self, player).filter(|_| exists) else {
            return;
        };
        counters[ECON_COMMANDS] += 1;
        counters[ECON_MALFORMED_UNIT_CMDS] += (effect == UnitActionEffect::None && malformed) as u64;
        let op = action.as_array().and_then(|parts| parts.first()).and_then(Value::as_str);
        let field = match op {
            Some("PASS") => ECON_PASS,
            Some("HARVEST") if effect != UnitActionEffect::None => ECON_HARVEST,
            Some("WATER") if effect != UnitActionEffect::None => ECON_WATER,
            Some("FEED") if effect != UnitActionEffect::None => ECON_FEED,
            _ => return,
        };
        counters[field] += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_header(episode_steps: usize) -> TraceHeader {
        let config = Config {
            episode_steps: episode_steps.into(),
            weed_spawn_chance: Value::from(0.0),
            ..Config::default()
        };
        let params = default_market_params();
        let farms = (0..2)
            .map(|_| Farm {
                money: config.starting_money.to_f64("startingMoney").unwrap(),
                tiles: (0..config.board_size)
                    .map(|y| {
                        (0..config.board_size)
                            .map(|x| {
                                if quadrant_of(x, y, config.board_size) == "NW" {
                                    Value::Null
                                } else {
                                    Value::String("LOCKED".to_string())
                                }
                            })
                            .collect()
                    })
                    .collect(),
                farmer: default_spawn(config.board_size),
                hands: vec![],
                unlocked_quadrants: vec!["NW".to_string()],
                hires_today: 0,
            })
            .collect();
        let privates = (0..2)
            .map(|_| PrivateState {
                shed: PRODUCTS
                    .into_iter()
                    .chain(ANIMAL_NAMES)
                    .map(|item| (item.to_string(), 0))
                    .collect(),
                seeds: CROP_NAMES
                    .into_iter()
                    .map(|item| (item.to_string(), 0))
                    .collect(),
                inventories: vec![IndexMap::new()],
            })
            .collect();
        let market = make_market(&params, false).unwrap();
        TraceHeader {
            format: TRACE_FORMAT.to_string(),
            seed: Number::from(7),
            configuration: config,
            shop_schedule: vec![],
            rng_schedule: vec![],
            initial: InitialState {
                public: PublicState {
                    step: 0,
                    day: 0,
                    hour: 0,
                    farms,
                    market,
                    town: Town {
                        unlocked_shops: vec![],
                    },
                },
                privates,
            },
            terminal_banks: vec![3_000.0, 3_000.0],
            transitions: 1,
        }
    }

    fn passes() -> Vec<Value> {
        vec![
            json!({"farmer": ["PASS"], "hands": [], "market": []}),
            json!({"farmer": ["PASS"], "hands": [], "market": []}),
        ]
    }

    #[test]
    fn upstream_schema_is_numeric_complete_and_accumulates_all_paths() {
        let zero = UnitStepMetrics::default();
        let encoded = serde_json::to_value(&zero).unwrap();
        assert_eq!(zero.upstream.len(), 43);
        assert!(
            encoded
                .as_object()
                .unwrap()
                .values()
                .all(|v| v.as_u64() == Some(0))
        );
        assert!(!encoded.as_object().unwrap().contains_key("upstream"));
        let mut seat = UnitSeatMetrics::default();
        for value in seat.upstream.values_mut() {
            *value = 3;
        }
        let mut first = UnitStepMetrics::default();
        first.by_seat[1] = seat.clone();
        first.accumulate_seat(&seat);
        let mut second = UnitStepMetrics::default();
        second.accumulate(&first);
        second.accumulate(&first);
        assert!(second.upstream.values().all(|&v| v == 6));
        assert!(second.by_seat[1].upstream.values().all(|&v| v == 6));
        assert!(second.by_seat[0].upstream.values().all(|&v| v == 0));
    }

    #[test]
    fn upstream_verbs_preserve_coarse_family_accounting() {
        let mut metrics = UnitSeatMetrics::default();
        for (verb, submitted, committed) in UPSTREAM_VERBS {
            let action = json!([verb]);
            record_unit_submission(&mut metrics, &action, true);
            let effect = match unit_action_kind(&action) {
                UnitActionKind::Movement => UnitActionEffect::Movement,
                UnitActionKind::Production => UnitActionEffect::Production,
                UnitActionKind::Logistics => UnitActionEffect::Logistics,
                UnitActionKind::None => panic!("unrecognized supported verb"),
            };
            record_unit_effect(&mut metrics, &action, effect);
            record_unit_effect(&mut metrics, &action, UnitActionEffect::None);
            record_unit_submission(&mut metrics, &action, false);
            assert_eq!(metrics.upstream[submitted], 1);
            assert_eq!(metrics.upstream[committed], 1);
        }
        for action in [json!(["PASS"]), json!(["UNKNOWN"]), Value::Null] {
            record_unit_submission(&mut metrics, &action, true);
            record_unit_effect(&mut metrics, &action, UnitActionEffect::None);
        }
        assert_eq!(metrics.submitted_move_actions, 4);
        assert_eq!(metrics.committed_move_actions, 4);
        assert_eq!(metrics.submitted_production_actions, 10);
        assert_eq!(metrics.committed_production_actions, 10);
        assert_eq!(metrics.submitted_logistics_actions, 3);
        assert_eq!(metrics.committed_logistics_actions, 3);
    }

    #[test]
    fn ineffective_commands_equal_the_per_verb_upstream_sums() {
        let mut game = Game::from_header(&fresh_header(40)).unwrap();
        game.privates[0].seeds.insert("WHEAT".into(), 1);
        let commands = [
            json!({"farmer": ["PLANT", "WHEAT"], "hands": [["PLANT", "WHEAT"]]}), // blocked: 2 > 1 seed
            json!({"farmer": ["NORTH"]}),
            json!({"farmer": ["NORTH"]}),
            json!({"farmer": ["NORTH"]}),
            json!({"farmer": ["NORTH"]}),
            json!({"farmer": ["NORTH"]}), // off the board eventually
            json!({"farmer": ["FEED"]}),  // nothing to feed
            json!({"farmer": ["HARVEST"]}),
            json!({"farmer": ["WATER"]}),
            json!({"farmer": ["DIG"]}),
            json!({"farmer": ["PASS"]}),
            json!({"farmer": ["PLANT", "WHEAT"]}),
            json!({"farmer": ["WATER"]}),
            json!({"farmer": ["WATER"]}),
        ];
        let mut expected = [0_u64; 2];
        for (step, own) in commands.iter().cycle().take(30).enumerate() {
            let actions = [own.clone(), commands[(step + 3) % commands.len()].clone()];
            let metrics = game.step_with_market_metrics(&actions).unwrap().unit;
            for seat in 0..2 {
                let by_verb: u64 = UPSTREAM_VERBS
                    .iter()
                    .map(|(_, submitted, committed)| {
                        metrics.by_seat[seat].upstream[submitted] - metrics.by_seat[seat].upstream[committed]
                    })
                    .sum();
                assert_eq!(metrics.by_seat[seat].ineffective_unit_commands(), by_verb, "step {step}");
                expected[seat] += by_verb;
            }
            let counters = game.econ_counters().unwrap();
            assert_eq!([counters[0][ECON_INEFFECTIVE], counters[1][ECON_INEFFECTIVE]], expected, "step {step}");
        }
        assert!(expected[0] > 0 && expected[1] > 0);
    }

    #[test]
    fn attrib_ledger_reconciles_with_econ_counters() {
        use crate::econ_attrib::*;
        let mut game = Game::from_header(&fresh_header(200)).unwrap();
        game.config.weed_spawn_chance = json!(0.2);
        for p in 0..2 {
            game.farms[p].money = 20_000.0;
            game.farms[p].tiles[0][0] = new_plant("TOMATO", -7, &game.config.turns_per_day);
            game.farms[p].tiles[1][1] = new_animal("GOOSE", -3);
            game.privates[p].shed.insert("WHEAT".into(), 20);
            game.privates[p].seeds.insert("CARROT".into(), 3);
        }
        let commands = [
            json!({"farmer":["PLANT","CARROT"],"hands":[["PLANT","CARROT"],["WATER"]],"market":[["HIRE"],["BUY_LAND"]]}),
            json!({"farmer":["WATER"],"hands":[["PASS"],["HARVEST"]],"market":[["SELL","WHEAT",3],["SELL","EGG",2]]}),
            json!({"farmer":["FERTILIZE"],"hands":[null,["FEED"]],"market":[["BUY_PRODUCT","WHEAT",1],["BUY_ANIMAL","GOOSE",1]]}),
            json!({"farmer":["HARVEST"],"hands":[["PICKUP","WHEAT",4],["CARE"]],"market":[["BUY_SEED","WHEAT",2]]}),
            json!({"farmer":["DIG"],"hands":[["NORTH"],["PLACE","WHEAT",2]],"market":[null,["SELL","MELON",3]]}),
            json!({"farmer":["PASS"],"hands":[["COLLECT_FERTILIZER"],["EAST"]]}),
            json!({"farmer":["SOUTH"],"hands":[["WEST"],["DROP"]],"market":[["HIRE"]]}),
            json!({"farmer":["EAST"],"hands":[["FEED"],["HARVEST"]]}),
        ];
        let mut money_spent = [0.0f64; 2];
        for step in 0..199 {
            let actions = [commands[step % commands.len()].clone(), commands[(step + 3) % commands.len()].clone()];
            let before = [game.farms[0].money, game.farms[1].money];
            game.step(&actions).unwrap();
            for p in 0..2 { money_spent[p] += (before[p] - game.farms[p].money).max(0.0); }
        }
        let econ = *game.econ_counters().unwrap();
        let a = *game.attrib_counters().unwrap();
        let sum = |row: &[u64; ATTRIB_FIELDS], at: usize, n: usize| row[at..at + n].iter().sum::<u64>();
        let verb = |name: &str| ATTRIB_VERBS.iter().position(|v| *v == name).unwrap();
        for p in 0..2 {
            let (c, r) = (&econ[p], &a[p]);
            assert_eq!(sum(r, A_VERB_INEFFECTIVE, 18), c[ECON_INEFFECTIVE], "seat {p} ineffective");
            assert_eq!(r[A_VERB_INEFFECTIVE + verb("PASS")], 0);
            assert_eq!(r[A_VERB_EFFECTIVE + verb("HARVEST")], c[ECON_HARVEST]);
            assert_eq!(r[A_VERB_EFFECTIVE + verb("WATER")], c[ECON_WATER]);
            assert_eq!(r[A_VERB_EFFECTIVE + verb("FEED")], c[ECON_FEED]);
            assert_eq!(sum(r, A_SLOT_EFFECTIVE, 16), sum(r, A_VERB_EFFECTIVE, 18));
            assert_eq!(sum(r, A_SLOT_INEFFECTIVE, 16), sum(r, A_VERB_INEFFECTIVE, 18));
            assert_eq!(sum(r, A_SLOT_IDLE + 1, 15), c[ECON_IDLE_HAND_STEPS], "seat {p} idle hands");
            // an existing unit's recognized command is effective, ineffective or PASS; a null/unknown one is neither
            assert!(sum(r, A_SLOT_EFFECTIVE, 16) + sum(r, A_SLOT_INEFFECTIVE, 16) + c[ECON_PASS] <= c[ECON_COMMANDS]);
            assert!(sum(r, A_SLOT_EFFECTIVE, 16) + sum(r, A_SLOT_INEFFECTIVE, 16) + c[ECON_PASS] > 0);
            assert_eq!(sum(r, A_SELL_UNITS, 9), c[ECON_SELL_UNITS]);
            assert_eq!(sum(r, A_SELL_CASH, 9), c[ECON_SELL_CASH]);
            assert_eq!(r[A_MARKET_COMMITTED + 3], c[ECON_SELL_UNITS]);
            let bought = sum(r, A_BUY_PRODUCT_CASH, 9) + sum(r, A_BUY_SEED_CASH, 5) + sum(r, A_BUY_ANIMAL_CASH, 3)
                + r[A_HIRE_CASH] + r[A_LAND_CASH];
            // cash conservation: every money change is a sale or a purchase the ledger itemizes
            assert_eq!(game.farms[p].money, 20_000.0 + c[ECON_SELL_CASH] as f64 - bought as f64, "seat {p} cash");
            assert!(money_spent[p] > 0.0);
            assert!(r[A_HIRES] > 0 && r[A_LANDS] > 0 && r[A_LAND_CASH] >= 1000);
            assert!(r[A_TRANSFER_REQUESTED] >= r[A_TRANSFER_MOVED] && r[A_TRANSFER_REQUESTED] > 0);
            // each committed HARVEST collects at least one unit; each COLLECT_FERTILIZER exactly one
            assert!(sum(r, A_PRODUCED_UNITS, 8) >= r[A_VERB_EFFECTIVE + verb("HARVEST")]);
            assert_eq!(r[A_PRODUCED_UNITS + 8], r[A_VERB_EFFECTIVE + verb("COLLECT_FERTILIZER")]);
            assert!(r[A_EOD_UNWATERED_PLANTS] + r[A_EOD_UNFED_ANIMALS] > 0);
        }
    }

    #[test]
    fn attrib_ledger_exact_small_cases() {
        use crate::econ_attrib::*;
        let mut game = Game::from_header(&fresh_header(200)).unwrap();
        let mut tomato = new_plant("TOMATO", -20, &game.config.turns_per_day);
        tomato.as_object_mut().unwrap().insert("yield_units".into(), json!(3));
        let pos = game.farms[0].farmer.clone();
        let (x, y) = (pos[0] as usize, pos[1] as usize);
        game.farms[0].tiles[y][x] = tomato;
        game.step(&[json!({"farmer":["HARVEST"]}), json!({})]).unwrap();
        assert_eq!(game.attrib[0][A_PRODUCED_UNITS + 2], 3);
        // shed transfer partial fill: request 5, shed has 2 (the farmer spawns next to the shed)
        game.privates[0].shed.insert("WHEAT".into(), 2);
        game.step(&[json!({"farmer":["PICKUP","WHEAT",5]}), json!({})]).unwrap();
        assert_eq!((game.attrib[0][A_TRANSFER_REQUESTED], game.attrib[0][A_TRANSFER_MOVED]), (5, 2));
        // market partial fill and per-item cash: SELL 5 TOMATO holding 3 (carried goods do not sell; shed only)
        game.privates[0].shed.insert("TOMATO".into(), 3);
        let money = game.farms[0].money;
        game.step(&[json!({"market":[["SELL","TOMATO",5]]}), json!({})]).unwrap();
        assert_eq!(game.attrib[0][A_MARKET_REQUESTED + 3], 5);
        assert_eq!(game.attrib[0][A_MARKET_COMMITTED + 3], game.attrib[0][A_SELL_UNITS + 2]);
        assert_eq!(game.attrib[0][A_SELL_CASH + 2] as f64, game.farms[0].money - money);
        let mut idle_only = [0; ATTRIB_FIELDS];
        idle_only[A_SLOT_IDLE] = 3; // the rival's omitted farmer command is a PASS on every step
        assert_eq!(game.attrib[1], idle_only);
    }

    #[test]
    fn econ_v4_counting_is_state_neutral() {
        let mut counted = Game::from_header(&fresh_header(82)).unwrap();
        counted.config.weed_spawn_chance = json!(0.3); // compare stochastic day-end outcomes too
        counted.farms[0].money = 20_000.0;
        counted.farms[0].tiles[0][0] = new_plant("TOMATO", -7, &counted.config.turns_per_day);
        counted.farms[1].tiles[0][0] = new_animal("GOOSE", -3);
        counted.privates[0].shed.insert("WHEAT".into(), 20);
        counted.privates[0].seeds.insert("CARROT".into(), 10);
        let mut uncounted = counted.clone();
        uncounted.econ_counting_disabled = true;
        let commands = [
            json!({"farmer":["PLANT","CARROT"],"market":[["HIRE"],["BUY_LAND"]]}),
            json!({"farmer":["WATER"],"hands":[["PASS"]],"market":[["SELL","WHEAT",3]]}),
            json!({"farmer":["FERTILIZE"],"hands":[null],"market":[["BUY_PRODUCT","WHEAT",1]]}),
            json!({"farmer":["HARVEST"],"market":[["BUY_SEED","WHEAT",2]]}),
            json!({"farmer":["DIG"],"market":[null,["SELL","MELON",3]]}),
            json!({"farmer":["PASS"]}),
        ];
        for step in 0..81 {
            let actions = [commands[step % commands.len()].clone(), commands[(step + 3) % commands.len()].clone()];
            let a = counted.step_with_market_metrics(&actions).unwrap();
            let b = uncounted.step_with_market_metrics(&actions).unwrap();
            assert_eq!(serde_json::to_value(counted.snapshot()).unwrap(),
                serde_json::to_value(uncounted.snapshot()).unwrap(), "snapshot step {step}");
            assert_eq!(format!("{a:?}"), format!("{b:?}"), "metrics step {step}");
            assert_eq!(counted.done, uncounted.done);
            assert_eq!(counted.rewards, uncounted.rewards);
            assert_eq!(uncounted.econ_counters, [[0; ECON_FIELDS]; 2]);
            assert_eq!(uncounted.attrib, [[0; ATTRIB_FIELDS]; 2]);
        }
        assert!(counted.done);
        assert!(counted.econ_counters[0][ECON_MARKET_UNFILLED_UNITS] > 0);
        assert!(counted.econ_counters[0][ECON_UNUSED_LAND_CASH] > 0);
    }

    #[test]
    fn econ_v4_death_clipping_and_missed_growth() {
        let mut game = Game::from_header(&fresh_header(720)).unwrap();
        game.step = 71; // day 2 end
        let mut drought = new_plant("WHEAT", 0, &game.config.turns_per_day);
        drought["consecutive_unwatered"] = json!(1); drought["yield_units"] = json!(3);
        game.farms[0].tiles[0][0] = drought;
        let mut dying = new_animal("GOOSE", 0);
        dying["consecutive_unfed"] = json!(1); dying["yield_units"] = json!(2);
        dying["pending_care_bonus"] = json!(3); dying["fertilizer_available"] = json!(true);
        dying["cared_today"] = json!(true); game.farms[0].tiles[0][1] = dying;
        let mut survivor = new_animal("GOOSE", 0); survivor["cared_today"] = json!(true);
        game.farms[0].tiles[0][2] = survivor;
        let mut wheat = new_plant("WHEAT", 0, &game.config.turns_per_day);
        wheat["fertilized_until_day"] = json!(4); wheat["consecutive_unwatered"] = json!(0); game.farms[0].tiles[0][3] = wheat;
        let mut tomato = new_plant("TOMATO", -5, &game.config.turns_per_day);
        tomato["yield_units"] = json!(4); tomato["watered_today"] = json!(true);
        tomato["fertilized_until_day"] = json!(4); game.farms[0].tiles[1][0] = tomato;
        game.step(&[json!({}), json!({})]).unwrap();
        let c = game.econ_counters[0];
        assert_eq!(c[ECON_DEATH_HELD_UNITS], 9);
        assert_eq!(c[ECON_CARE_WASTED], 2); // death and survivor, once each
        assert_eq!(c[ECON_CLIPPED_UNITS], 2);
        assert_eq!(c[ECON_MISSED_GROWTH_UNITS], 2);
        assert_eq!(game.econ_counters[1][ECON_DEATH_HELD_UNITS], 0);
    }

    #[test]
    fn econ_v4_unit_losses() {
        let mut game = Game::from_header(&fresh_header(720)).unwrap();
        game.step = 72; // day3, inside wheat growth window
        game.farms[0].farmer = vec![0, 0];
        let mut wheat = new_plant("WHEAT", 0, &game.config.turns_per_day);
        wheat["yield_units"] = json!(5); wheat["fertilized_until_day"] = json!(5);
        game.farms[0].tiles[0][0] = wheat;
        game.privates[0].inventories = vec![IndexMap::from([("FERTILIZER".into(), 2)])];
        game.step(&[json!({"farmer":["WATER"]}),json!({})]).unwrap();
        assert_eq!(game.econ_counters[0][ECON_CLIPPED_UNITS], 1);
        game.step(&[json!({"farmer":["FERTILIZE"]}),json!({})]).unwrap();
        assert_eq!(game.econ_counters[0][ECON_REDUNDANT_FERT], 1);
        game.farms[0].tiles[0][0]["yield_units"] = json!(2);
        game.step(&[json!({"farmer":["HARVEST"]}),json!({})]).unwrap();
        assert_eq!(game.econ_counters[0][ECON_MISSED_GROWTH_UNITS], 4);
        game.farms[0].tiles[0][0] = new_plant("WHEAT", 3, &game.config.turns_per_day);
        game.farms[0].tiles[0][0]["yield_units"] = json!(2);
        game.step(&[json!({"farmer":["DIG"]}),json!({})]).unwrap();
        assert_eq!(game.econ_counters[0][ECON_DUG_UNITS], 3);
    }

    #[test]
    fn econ_v4_terminal_provenance_and_flags() {
        let mut game = Game::from_header(&fresh_header(30)).unwrap();
        game.farms[0].money = 20_000.0;
        game.step(&[json!({"market":[["BUY_LAND"],["BUY_LAND"],["BUY_LAND"]]}),json!({})]).unwrap();
        game.farms[0].farmer = vec![5,0];
        game.privates[0].seeds.insert("WHEAT".into(), 4);
        game.step(&[json!({"farmer":["PLANT","WHEAT"]}),json!({})]).unwrap();
        game.step(&[json!({"farmer":["DIG"]}),json!({})]).unwrap();
        assert!(game.econ_land_used[0][0]); // remembers used NE after destruction
        let mut goose = new_animal("GOOSE", 0);
        goose["fertilizer_available"] = json!(true); goose["pending_care_bonus"] = json!(2);
        goose["fed_today"] = json!(true); game.farms[0].tiles[0][0] = goose;
        game.step = 28;
        game.step(&[json!({}),json!({})]).unwrap();
        assert_eq!(game.econ_counters[0][ECON_TERMINAL_FLAGS], 4);
        assert_eq!(game.econ_counters[0][ECON_SEEDS_UNUSED_END], 3);
        assert_eq!(game.econ_counters[0][ECON_UNUSED_LAND_CASH], 6000);
        let clone = game.clone(); assert_eq!(clone.econ_land_used, game.econ_land_used);
        let before = game.econ_counters;
        assert!(game.step(&[json!({}),json!({})]).is_err()); assert_eq!(game.econ_counters, before);
    }

    #[test]
    fn econ_v4_labor_and_malformed_commands() {
        let mut game = Game::from_header(&fresh_header(30)).unwrap();
        game.step = 22;
        game.step(&[json!({"market":[["HIRE"],["HIRE"],["HIRE"]]}),json!({})]).unwrap();
        assert_eq!(game.econ_counters[0][ECON_HIRE_WASTED_CASH], 0);
        game.step(&[json!({"farmer":[], "hands":[["PASS"],null], "market":[["HIRE"]]}),json!({})]).unwrap();
        assert_eq!(game.econ_counters[0][ECON_IDLE_HAND_STEPS], 2); // PASS plus omitted third
        assert_eq!(game.econ_counters[0][ECON_MALFORMED_UNIT_CMDS], 2);
        assert_eq!(game.econ_counters[0][ECON_HIRE_WASTED_CASH], 3);
        game.step = 28;
        game.step(&[json!({"market":[["HIRE"]]}),json!({})]).unwrap();
        assert_eq!(game.econ_counters[0][ECON_HIRE_WASTED_CASH], 4); // final partial day
        assert!(!malformed_unit(&json!(["PASS"])));
        assert!(!malformed_unit(&json!(["PLANT","UNKNOWN"]))); // parsed but ineffective
        assert!(malformed_unit(&json!(["PLANT"])));
        assert!(malformed_unit(&json!(["UNKNOWN"])));
        let mut placement = Game::from_header(&fresh_header(30)).unwrap();
        placement.farms[0].farmer = vec![0,0];
        placement.farms[0].tiles[0][0] = json!({"kind":"COOP"});
        placement.step(&[json!({"farmer":["PLACE","GOOSE","ignored"]}),json!({})]).unwrap();
        assert_eq!(placement.econ_counters[0][ECON_MALFORMED_UNIT_CMDS], 0);
        placement.privates[0].inventories[0].insert("GOOSE".into(),1);
        placement.step(&[json!({"farmer":["PLACE","GOOSE","ignored"]}),json!({})]).unwrap();
        assert_eq!(placement.econ_counters[0][ECON_MALFORMED_UNIT_CMDS], 0);
    }

    #[test]
    fn econ_v4_market_failures_and_reference_prices() {
        let mut game = Game::from_header(&fresh_header(720)).unwrap();
        game.config.max_market_orders_per_turn = 3usize.into();
        game.privates[0].shed.insert("WHEAT".into(), 2);
        game.step(&[json!({"market":[["SELL","WHEAT",5],null,["BUY_PRODUCT","MELON",4],["HIRE"],["HIRE"]]}),json!({})]).unwrap();
        assert_eq!(game.econ_counters[0][ECON_MARKET_UNFILLED_UNITS], 10); // 3 + 1 + 4 + 2 truncated
        assert_eq!(game.econ_counters[0][ECON_SALE_SHORTFALL_CASH], 1); // 25 then24
        game.market.inventory.insert("WHEAT".into(), json!(1_000_000_000_000_000i64));
        game.privates[0].shed.insert("WHEAT".into(), 2);
        game.step(&[json!({"market":[["SELL","WHEAT",2]]}),json!({})]).unwrap();
        assert_eq!(game.econ_counters[0][ECON_FLOOR_SALE_UNITS], 2);
        assert_eq!(game.econ_counters[0][ECON_SALE_SHORTFALL_CASH], 49);
        game.market.inventory.insert("WHEAT".into(), json!(0));
        let before = game.farms[0].money;
        game.step(&[json!({"market":[["BUY_PRODUCT","WHEAT",1],["BUY_SEED","WHEAT",1]]}),json!({})]).unwrap();
        assert_eq!(game.econ_counters[0][ECON_BUY_PREMIUM_CASH], (before - game.farms[0].money - 25.0 - 10.0) as u64);
        assert!(game.econ_counters[0][ECON_BUY_PREMIUM_CASH] > 0);
        let before_unfilled = game.econ_counters[0][ECON_MARKET_UNFILLED_UNITS];
        game.step(&[json!({"market":[["BUY_SEED",17,5],["SELL",null,3],["BUY_PRODUCT","WHEAT",0]]}),json!({})]).unwrap();
        assert_eq!(game.econ_counters[0][ECON_MARKET_UNFILLED_UNITS] - before_unfilled, 9);
        assert_eq!(econ_requested_market_units(&json!(["HIRE",999])), 1);
    }

    #[test]
    fn econ_v4_loss_counters() {
        let pass = || json!({"farmer": ["PASS"]});
        // decay: a 3-unit plant loses one unit; a 1-unit plant dies holding it (step == max_lifespan_step)
        let mut game = Game::from_header(&fresh_header(200)).unwrap();
        game.step = 10;
        let mut old3 = new_plant("WHEAT", 0, &game.config.turns_per_day);
        old3["max_lifespan_step"] = json!(10);
        old3["yield_units"] = json!(3);
        let mut old1 = old3.clone();
        old1["yield_units"] = json!(1);
        game.farms[0].tiles[0][0] = old3;
        game.farms[0].tiles[0][1] = old1;
        game.step(&[pass(), pass()]).unwrap();
        assert_eq!(game.econ_counters().unwrap()[0][ECON_EXPIRY_UNITS], 2);
        assert_eq!(game.econ_counters().unwrap()[1][ECON_EXPIRY_UNITS], 0);

        // DROP into a full shed discards the carried goods (the farmer spawns at a shed access tile)
        let mut game = Game::from_header(&fresh_header(200)).unwrap();
        game.step = 5;
        game.privates[0].shed.insert("MELON".into(), 100);
        if game.privates[0].inventories.is_empty() {
            game.privates[0].inventories.push(Inventory::new());
        }
        game.privates[0].inventories[0].insert("WHEAT".into(), 4);
        game.step(&[json!({"farmer": ["DROP"]}), pass()]).unwrap();
        assert_eq!(game.econ_counters().unwrap()[0][ECON_OVERFLOW_UNITS], 4);

        // day end: overflow of the automatic deposit, a forfeited care bonus, an uncollected fertilizer, weeds
        let mut game = Game::from_header(&fresh_header(200)).unwrap();
        game.step = 23;
        game.config.weed_spawn_chance = Value::from(1.0);
        game.privates[0].shed.insert("MELON".into(), 98);
        if game.privates[0].inventories.is_empty() {
            game.privates[0].inventories.push(Inventory::new());
        }
        game.privates[0].inventories[0].insert("WHEAT".into(), 5);
        let mut goose = new_animal("GOOSE", -3); // production day at this day end (first yield day 4)
        goose["pending_care_bonus"] = json!(2);
        goose["fertilizer_available"] = json!(true);
        game.farms[0].tiles[0][0] = goose;
        let empty: usize = game.farms[0].tiles.iter().flatten().filter(|t| t.is_null()).count();
        game.step(&[pass(), pass()]).unwrap();
        let own = game.econ_counters().unwrap()[0];
        assert_eq!(own[ECON_OVERFLOW_UNITS], 3); // 98 + 5 into 100
        assert_eq!(own[ECON_CARE_LOST], 2); // unfed: the pending bonus is forfeited
        assert_eq!(own[ECON_FERT_WASTED], 1);
        assert_eq!(own[ECON_WEEDS] as usize, empty); // chance 1: every empty tile
        assert_eq!(own[ECON_STARVATION], 0); // first unfed day end only

        // game end: unsold goods (shed + carried + yield still on tiles), set once
        let mut game = Game::from_header(&fresh_header(30)).unwrap();
        game.step = 28;
        game.privates[0].shed.insert("WHEAT".into(), 7);
        let mut ripe = new_plant("WHEAT", 0, &game.config.turns_per_day);
        ripe["yield_units"] = json!(2);
        game.farms[0].tiles[0][0] = ripe;
        assert_eq!(game.econ_counters().unwrap()[0][ECON_UNSOLD_END], 0);
        game.step(&[pass(), pass()]).unwrap();
        assert!(game.done);
        assert_eq!(game.econ_counters().unwrap()[0][ECON_UNSOLD_END], 9);
    }

    #[test]
    fn econ_telemetry_counts_commands_pass_commits_and_sales() {
        let mut game = Game::from_header(&fresh_header(40)).unwrap();
        game.step = 5;
        let mut wet = new_plant("WHEAT", 0, &game.config.turns_per_day);
        wet["watered_today"] = json!(false);
        game.farms[0].tiles[4][4] = wet; // under the farmer's spawn
        game.privates[0].shed.insert("WHEAT".into(), 3);
        let before = game.econ_counters().unwrap()[0];
        let actions = [json!({"farmer": ["WATER"], "hands": [["PASS"]],
                              "market": [["SELL", "WHEAT", 2]]}),
                       json!({"hands": []})]; // a missing farmer command is PASS
        let money = game.farms[0].money;
        let metrics = game.step_with_market_metrics(&actions).unwrap();
        let after = game.econ_counters().unwrap();
        let committed_water = metrics.unit.by_seat[0].upstream["upstream_committed_WATER"];
        assert_eq!(after[0][ECON_COMMANDS] - before[ECON_COMMANDS], 1); // the hand does not exist
        assert_eq!(after[0][ECON_PASS], 0);
        assert_eq!((after[0][ECON_WATER], committed_water), (1, 1));
        assert_eq!(after[0][ECON_SELL_UNITS], 2);
        assert_eq!(after[0][ECON_SELL_CASH] as f64, game.farms[0].money - money);
        assert_eq!((after[1][ECON_COMMANDS], after[1][ECON_PASS]), (1, 1));
        assert_eq!(after[1][ECON_SELL_UNITS], 0);
    }

    #[test]
    fn upstream_plant_overcommit_and_missing_hand_are_not_commits() {
        let mut game = Game::from_header(&fresh_header(10)).unwrap();
        game.privates[0].seeds.insert("WHEAT".into(), 1);
        let mut actions = passes();
        actions[0] = json!({"farmer": ["PLANT", "WHEAT"], "hands": [["PLANT", "WHEAT"]]});
        // Even a nonexistent hand enters the official demand check, but it is
        // excluded from the established submitted-unit telemetry definition.
        let before_farm = serde_json::to_value(&game.farms[0]).unwrap();
        let result = game.step_with_market_metrics(&actions).unwrap();
        assert_eq!(result.unit.upstream["upstream_submitted_PLANT"], 1);
        assert_eq!(result.unit.upstream["upstream_committed_PLANT"], 0);
        assert_eq!(game.privates[0].seeds["WHEAT"], 1);
        assert_eq!(serde_json::to_value(&game.farms[0]).unwrap(), before_farm);
        assert_eq!(result.unit.upstream["upstream_observed_seat_steps"], 2);
        assert_eq!(
            result.unit.by_seat[0].upstream["upstream_observed_seat_steps"],
            1
        );
        assert_eq!(
            result.unit.by_seat[1].upstream["upstream_submitted_PLANT"],
            0
        );
        let planted = game
            .step_with_market_metrics(&[json!({"farmer": ["PLANT", "WHEAT"]}), json!({})])
            .unwrap();
        assert_eq!(planted.unit.upstream["upstream_committed_PLANT"], 1);
        assert_eq!(game.privates[0].seeds["WHEAT"], 0);
    }

    #[test]
    fn upstream_deaths_are_exact_disjoint_removal_events() {
        let mut game = Game::from_header(&fresh_header(30)).unwrap();
        game.step = 23;
        let drought = new_plant("WHEAT", 0, &game.config.turns_per_day);
        let mut expiry = drought.clone();
        expiry["max_lifespan_step"] = json!(23);
        expiry["yield_units"] = json!(1);
        let mut decaying = expiry.clone();
        decaying["yield_units"] = json!(3);
        decaying["watered_today"] = json!(true);
        let mut starved = new_animal("GOOSE", 0);
        starved["consecutive_unfed"] = json!(1);
        let mut fed = starved.clone();
        fed["fed_today"] = json!(true);
        game.farms[0].tiles[0][0] = drought;
        game.farms[0].tiles[0][1] = expiry;
        game.farms[0].tiles[0][2] = decaying;
        game.farms[0].tiles[0][3] = starved;
        game.farms[0].tiles[0][4] = fed;
        let metrics = game.step_with_market_metrics(&passes()).unwrap().unit;
        assert_eq!(metrics.upstream["upstream_crop_drought_deaths"], 1);
        assert_eq!(metrics.upstream["upstream_crop_expiry_deaths"], 1);
        assert_eq!(metrics.upstream["upstream_animal_starvation_deaths"], 1);
        assert_eq!(metrics.upstream["upstream_crop_tile_steps"], 1);
        assert_eq!(metrics.upstream["upstream_animal_tile_steps"], 1);
        assert_eq!(game.farms[0].tiles[0][0], json!({"kind": "WEED"}));
        assert_eq!(game.farms[0].tiles[0][1], json!({"kind": "WEED"}));
        assert_eq!(game.farms[0].tiles[0][2]["yield_units"], json!(2));
        assert_eq!(game.farms[0].tiles[0][3], json!({"kind": "COOP"}));
        // Cumulative per-seat [starvation, drought] counters (econ shaping) follow the same
        // events; expiry is not a drought death.
        assert_eq!(game.econ_counters().map(|c| [c[0][..3].to_vec(), c[1][..3].to_vec()]),
                   Some([vec![1, 1, 0], vec![0, 0, 0]]));
        let again = game.step_with_market_metrics(&passes()).unwrap().unit;
        assert_eq!(game.econ_counters().map(|c| [c[0][..3].to_vec(), c[1][..3].to_vec()]),
                   Some([vec![1, 1, 0], vec![0, 0, 0]]));
        for key in [
            "upstream_crop_drought_deaths",
            "upstream_crop_expiry_deaths",
            "upstream_animal_starvation_deaths",
        ] {
            assert_eq!(again.upstream[key], 0);
            assert_eq!(metrics.by_seat[1].upstream[key], 0);
        }
    }

    #[test]
    fn upstream_ready_stock_uses_harvest_age_post_close_and_includes_terminal() {
        let mut game = Game::from_header(&fresh_header(49)).unwrap();
        game.step = 47; // This one transition both closes day 1 and terminates.
        let mut unripe = new_plant("WHEAT", 1, &game.config.turns_per_day);
        unripe["watered_today"] = json!(true);
        let mut ripe = new_plant("WHEAT", 0, &game.config.turns_per_day);
        ripe["watered_today"] = json!(true);
        ripe["yield_units"] = json!(3);
        let mut empty = ripe.clone();
        empty["yield_units"] = json!(0);
        let mut goose = new_animal("GOOSE", 0);
        goose["fed_today"] = json!(true);
        goose["yield_units"] = json!(2);
        game.farms[0].tiles[0][0] = unripe;
        game.farms[0].tiles[0][1] = ripe;
        game.farms[0].tiles[0][2] = empty;
        game.farms[0].tiles[0][3] = goose;
        let metrics = game.step_with_market_metrics(&passes()).unwrap().unit;
        assert!(game.done);
        assert_eq!(game.step, 48);
        assert_eq!(metrics.upstream["upstream_crop_tile_steps"], 3);
        assert_eq!(metrics.upstream["upstream_ready_crop_tile_steps"], 1);
        assert_eq!(metrics.upstream["upstream_ready_animal_tile_steps"], 1);
        assert_eq!(metrics.upstream["upstream_ready_product_unit_steps"], 5);
        assert!(game.step_with_market_metrics(&passes()).is_err());
    }

    #[test]
    fn upstream_harvest_counts_success_not_unripe_request() {
        let mut game = Game::from_header(&fresh_header(80)).unwrap();
        game.step = 48;
        game.farms[0].farmer = vec![0, 0];
        game.farms[1].farmer = vec![0, 0];
        game.farms[0].tiles[0][0] = new_plant("WHEAT", 0, &game.config.turns_per_day);
        game.farms[1].tiles[0][0] = new_plant("WHEAT", 1, &game.config.turns_per_day);
        let harvests = vec![json!({"farmer": ["HARVEST"]}); 2];
        let metrics = game.step_with_market_metrics(&harvests).unwrap().unit;
        assert_eq!(metrics.upstream["upstream_submitted_HARVEST"], 2);
        assert_eq!(metrics.upstream["upstream_committed_HARVEST"], 1);
        assert_eq!(metrics.by_seat[0].upstream["upstream_committed_HARVEST"], 1);
        assert_eq!(metrics.by_seat[1].upstream["upstream_committed_HARVEST"], 0);
        assert_eq!(game.privates[0].inventories[0]["WHEAT"], 1);
        assert!(game.farms[0].tiles[0][0].is_null());
        assert_eq!(metrics.upstream["upstream_ready_crop_tile_steps"], 0);
        let before = serde_json::to_value(game.snapshot()).unwrap();
        let mut observed = UnitSeatMetrics::default();
        record_upstream_stock(&mut observed, &game.farms[1], 2);
        assert_eq!(serde_json::to_value(game.snapshot()).unwrap(), before);
        let mut live = UnitStepMetrics::default();
        let mut opponent = UnitStepMetrics::default();
        live.accumulate_seat(&metrics.by_seat[0]);
        opponent.accumulate_seat(&metrics.by_seat[1]);
        println!(
            "UPSTREAM_FIXTURE={}",
            json!({"unit": metrics, "unit_live": live, "unit_opponent": opponent})
        );
    }

    #[test]
    fn python_ties_to_even_price_boundary_is_preserved() {
        let params = default_market_params();
        assert_eq!(market_price("CARROT", 10_000, &params), 35);
        // Python computes 10.5 here and round() chooses the even integer 10.
        assert_eq!(market_price("CARROT", 10_450, &params), 10);
    }

    #[test]
    fn step_zero_town_tick_and_terminal_lifecycle_match_python() {
        let header = fresh_header(2);
        let mut game = Game::from_header(&header).unwrap();
        let initial = game.snapshot();
        assert!(!initial.done);
        assert_eq!(initial.statuses, ["ACTIVE", "ACTIVE"]);
        assert_eq!(initial.rewards, [0.0, 0.0]);
        game.step(&passes()).unwrap();
        let state = game.public_state();
        assert_eq!(state.step, 1);
        assert_eq!(state.market.inventory["WHEAT"], json!(9_999));
        assert_eq!(state.market.inventory["FERTILIZER"], json!(10_000));
        assert_eq!(game.terminal_banks(), Some(&[3_000.0, 3_000.0][..]));
        let terminal = game.snapshot();
        assert!(terminal.done);
        assert_eq!(terminal.statuses, ["DONE", "DONE"]);
        assert_eq!(terminal.rewards, [3_000.0, 3_000.0]);
        assert_eq!(
            game.step(&passes()).unwrap_err(),
            "cannot step a completed game"
        );
    }

    #[test]
    fn concurrent_sellers_receive_the_same_precommit_quote() {
        let mut header = fresh_header(2);
        header.initial.privates[0].shed.insert("WHEAT".into(), 1);
        header.initial.privates[1].shed.insert("WHEAT".into(), 1);
        let actions = vec![
            json!({"farmer": ["PASS"], "market": [["SELL", "WHEAT", 1]]}),
            json!({"farmer": ["PASS"], "market": [["SELL", "WHEAT", 1]]}),
        ];
        let mut game = Game::from_header(&header).unwrap();
        game.step(&actions).unwrap();
        assert_eq!(game.farms[0].money, 3_025.0);
        assert_eq!(game.farms[1].money, 3_025.0);
        // two sells, then the source-step-zero town-center drain
        assert_eq!(game.market.inventory["WHEAT"], json!(10_001));
    }

    #[test]
    fn shed_overflow_keeps_inventory_in_insertion_order() {
        let mut private = fresh_header(2).initial.privates.remove(0);
        private.shed.insert("WHEAT".into(), 99);
        let mut carried = IndexMap::new();
        carried.insert("MILK".to_string(), 2);
        carried.insert("WOOL".to_string(), 2);
        private.inventories = vec![carried];
        drop_inventories_to_shed(&mut private, &PyInt::from(100_i64));
        assert_eq!(private.shed["MILK"], 1);
        assert_eq!(private.shed["WOOL"], 0);
        assert!(private.inventories[0].is_empty());
    }

    #[test]
    fn market_escape_guard_matches_pythons_99_999_unit_limit() {
        let mut header = fresh_header(2);
        for farm in &mut header.initial.public.farms {
            farm.money = 2_000_000.0;
        }
        let actions = vec![
            json!({"farmer": ["PASS"], "market": [["BUY_SEED", "WHEAT", 100_000]]}),
            json!({"farmer": ["PASS"], "market": []}),
        ];
        let mut game = Game::from_header(&header).unwrap();
        game.step(&actions).unwrap();
        assert_eq!(game.privates[0].seeds["WHEAT"], 99_999);
        assert_eq!(game.farms[0].money, 1_000_010.0);
    }

    #[test]
    fn normalized_integral_float_configuration_and_unknown_keys_are_retained() {
        let config: Config = serde_json::from_value(json!({
            "episodeSteps": 2.0,
            "boardSize": 6.0,
            "startingMoney": 1234.0,
            "maxMarketOrdersPerTurn": 4.0,
            "turnsPerDay": 2.0,
            "shedCapacity": 7.0,
            "townShopUnlockInterval": 1.0,
            "townShopSellInterval": 1.0,
            "townCenterSellInterval": 2.0,
            "farmHandCostMult": 3.0,
            "futureConfigKey": {"retained": true}
        }))
        .unwrap();
        assert_eq!(config.episode_steps.0, BigInt::from(2));
        assert_eq!(config.board_size, 6);
        assert_eq!(config.starting_money.0, BigInt::from(1_234));
        assert_eq!(config.extra["futureConfigKey"], json!({"retained": true}));
        assert!(Game::new(config, -7, 2).is_ok());
    }

    #[test]
    fn schema_integer_configuration_does_not_apply_python_action_coercions() {
        let wide: Config = serde_json::from_str(
            r#"{"startingMoney":100000000000000000000,"turnsPerDay":100000000000000000000}"#,
        )
        .unwrap();
        assert_eq!(
            wide.starting_money.0,
            BigInt::from_str("100000000000000000000").unwrap()
        );
        assert_eq!(wide.turns_per_day.0, wide.starting_money.0);

        for invalid in [json!(3000.5), json!("3000"), json!(true)] {
            let result = serde_json::from_value::<Config>(json!({"startingMoney": invalid}));
            assert!(result.is_err());
        }
    }

    #[test]
    fn python_action_integer_requires_valid_decimal_underscore_grammar() {
        assert_eq!(
            python_int(&json!(" +1_0 "), "count").unwrap(),
            BigInt::from(10)
        );
        for invalid in ["_1", "1_", "1__0", "+_1", "-"] {
            assert!(matches!(
                python_int(&json!(invalid), "count"),
                Err(PyIntError::Invalid(_))
            ));
        }
        assert!(python_int(&json!("9".repeat(4_300)), "count").is_ok());
        assert!(matches!(
            python_int(&json!("9".repeat(4_301)), "count"),
            Err(PyIntError::Invalid(_))
        ));
        let infinite_float = serde_json::from_str::<Value>("1e400").unwrap();
        assert!(matches!(
            python_int(&infinite_float, "count"),
            Err(PyIntError::Overflow(_))
        ));
    }

    #[test]
    fn failed_action_is_transactional() {
        let mut header = fresh_header(3);
        header.initial.privates[0].shed.insert("WHEAT".into(), 1);
        let actions = vec![
            json!({"farmer": ["PICKUP", "WHEAT", "_1"], "hands": [], "market": []}),
            json!({"farmer": ["PASS"], "hands": [], "market": []}),
        ];
        let mut game = Game::from_header(&header).unwrap();
        let before = serde_json::to_value(game.snapshot()).unwrap();
        let error = game.step(&actions).unwrap_err();
        assert!(error.contains("cannot parse"));
        assert_eq!(serde_json::to_value(game.snapshot()).unwrap(), before);
    }

    #[test]
    fn zero_count_pickup_is_a_pass() {
        // Anchor labels (a1, 2026-09-26): cha22 can emit ["PICKUP", item, 0]; the Myolie codec has no zero quantity,
        // so the label canonicalizes it to PASS. That is exact only if the engine state after both is identical.
        let mut header = fresh_header(3);
        header.initial.privates[0].shed.insert("WHEAT".into(), 1);
        let after = |farmer: Value| {
            let mut game = Game::from_header(&header).unwrap();
            game.step(&[
                json!({"farmer": farmer, "hands": [], "market": []}),
                json!({"farmer": ["PASS"], "hands": [], "market": []}),
            ])
            .unwrap();
            serde_json::to_value(game.snapshot()).unwrap()
        };
        let pass = after(json!(["PASS"]));
        // Positive control: a count of 1 changes the state here, so the count check is the path under test.
        assert_ne!(after(json!(["PICKUP", "WHEAT", 1])), pass);
        assert_eq!(after(json!(["PICKUP", "WHEAT", 0])), pass);
        assert_eq!(after(json!(["PICKUP", "WHEAT", -3])), pass);
    }

    #[test]
    fn native_reset_rejects_non_two_player_cardinality() {
        let error = Game::new(Config::default(), 7, 3).unwrap_err();
        assert_eq!(error, "Kaggriculture requires exactly 2 agents, got 3");
    }

    #[test]
    fn boolean_weed_probability_is_not_a_schema_number() {
        let config: Config = serde_json::from_value(json!({"weedSpawnChance": true})).unwrap();
        assert_eq!(
            Game::new(config, 7, 2).unwrap_err(),
            "weedSpawnChance must be a non-negative number"
        );
    }

    #[test]
    fn float_seed_is_accepted_at_reset_but_fails_at_the_first_daily_mix() {
        let mut header = fresh_header(3);
        header.configuration.turns_per_day = 2_i64.into();
        header.seed = Number::from_f64(1.5).unwrap();
        let mut game = Game::from_seed_header(&header).unwrap();
        game.step(&passes()).unwrap();
        let error = game.step(&passes()).unwrap_err();
        assert!(error.contains("daily seed mixing requires an integer seed"));
        assert_eq!(game.public_state().step, 1);
    }

    #[test]
    fn zero_market_horizon_fails_when_python_first_prices_the_market() {
        let mut config = Config {
            episode_steps: 2_i64.into(),
            weed_spawn_chance: Value::from(0.0),
            ..Config::default()
        };
        config
            .market_params
            .insert("WHEAT".to_string(), json!({"T": 0}));
        let mut game = Game::new(config, 7, 2).unwrap();
        let error = game.step(&passes()).unwrap_err();
        assert!(error.contains("divides by zero at T=0"));
    }

    #[test]
    fn nested_market_horizon_does_not_coerce_strings() {
        let mut config = Config {
            episode_steps: 2_i64.into(),
            weed_spawn_chance: Value::from(0),
            ..Config::default()
        };
        config
            .market_params
            .insert("WHEAT".to_string(), json!({"T": "10"}));
        let mut game = Game::new(config, 7, 2).unwrap();
        let before = serde_json::to_value(game.snapshot()).unwrap();
        let error = game.step(&passes()).unwrap_err();
        assert!(error.contains("must be numeric"));
        assert_eq!(serde_json::to_value(game.snapshot()).unwrap(), before);
    }

    #[test]
    fn exact_linear_amplitude_does_not_eagerly_float_wide_horizon() {
        let wide = serde_json::from_str::<Value>(&format!("1{}", "0".repeat(400))).unwrap();
        let mut config = Config {
            episode_steps: 2_i64.into(),
            board_size: 4,
            weed_spawn_chance: Value::from(0),
            ..Config::default()
        };
        config.market_params.insert(
            "WHEAT".to_string(),
            json!({
                "base": 1,
                "I0": 0,
                "T": wide,
                "below_func": "linear",
                "below_target": wide,
                "above_func": "linear",
                "above_target": wide,
            }),
        );
        let mut game = Game::new(config, 7, 2).unwrap();
        game.step(&passes()).unwrap();
        assert_eq!(game.market.inventory["WHEAT"], json!(-1));
        assert_eq!(game.market.prices["WHEAT"], json!(2));
    }

    #[test]
    fn wide_integer_hinge_uses_exact_t_over_t_and_distance_ratio() {
        let wide = serde_json::from_str::<Value>(&format!("1{}", "0".repeat(400))).unwrap();
        let params = resolve_market_params(&IndexMap::from([(
            "WHEAT".to_string(),
            json!({
                "base": 1,
                "I0": 0,
                "T": wide,
                "below_func": "hinge",
                "below_target": 1,
                "above_func": "hinge",
                "above_target": 1,
            }),
        )]));
        assert_eq!(
            market_price_numeric("WHEAT", &json!(-1), &params).unwrap(),
            json!(1)
        );
    }

    #[test]
    fn squared_integer_market_distance_is_shaped_before_float_conversion() {
        let distance = serde_json::from_str::<Value>("9007199254740993").unwrap();
        let base = serde_json::from_str::<Value>("9999999999999980").unwrap();
        let params = resolve_market_params(&IndexMap::from([(
            "WHEAT".to_string(),
            json!({
                "base": base,
                "I0": 0,
                "T": distance,
                "below_func": "linear",
                "below_target": 1,
                "above_func": "sq",
                "above_target": 1,
            }),
        )]));
        assert_eq!(
            market_price_numeric("WHEAT", &distance, &params).unwrap(),
            json!(1)
        );
    }

    #[test]
    fn mixed_integer_float_market_comparison_is_exact_before_arithmetic() {
        let inventory = serde_json::from_str::<Value>("9007199254740995").unwrap();
        let params = resolve_market_params(&IndexMap::from([(
            "WHEAT".to_string(),
            json!({
                "base": 100,
                "I0": 9007199254740996.0,
                "T": 10,
                "below_func": "linear",
                "below_target": 0,
                "above_func": "linear",
                "above_target": "must not be evaluated",
            }),
        )]));
        assert!(numeric_less(&inventory, &json!(9007199254740996.0), "mixed comparison").unwrap());
        assert_eq!(
            market_price_numeric("WHEAT", &inventory, &params).unwrap(),
            json!(100)
        );
    }

    #[test]
    fn mixed_float_numerator_squares_integer_t_before_float_conversion() {
        let params = resolve_market_params(&IndexMap::from([(
            "WHEAT".to_string(),
            json!({
                "base": 1.5,
                "I0": 10000,
                "T": 9007199254740993_u64,
                "below_func": "sq",
                "below_target": 162259276829213399420375029252086_u128,
            }),
        )]));
        // Official Python computes raw 4.5 and ties-to-even rounds to four. If T
        // is converted to f64 before squaring, it lands just above 4.5 and rounds
        // incorrectly to five.
        assert_eq!(
            market_price_numeric("WHEAT", &json!(9999), &params).unwrap(),
            json!(4)
        );
    }

    #[test]
    fn mixed_float_division_rejects_an_exact_square_beyond_f64() {
        let wide = serde_json::from_str::<Value>(&format!("1{}", "0".repeat(200))).unwrap();
        let params = resolve_market_params(&IndexMap::from([(
            "WHEAT".to_string(),
            json!({
                "base": 1.5,
                "I0": 10000,
                "T": wide,
                "below_func": "sq",
                "below_target": 1.0,
            }),
        )]));
        let error = market_price_numeric("WHEAT", &json!(9999), &params).unwrap_err();
        assert!(error.contains("market amplitude denominator"));
        assert!(error.contains("does not fit a finite float"));
    }

    #[test]
    fn mixed_float_numerator_does_not_eagerly_convert_a_wide_hinge_t() {
        let wide = serde_json::from_str::<Value>(&format!("1{}", "0".repeat(400))).unwrap();
        let params = resolve_market_params(&IndexMap::from([(
            "WHEAT".to_string(),
            json!({
                "base": 1.5,
                "I0": 10000,
                "T": wide,
                "below_func": "hinge",
                "below_target": 1.0,
            }),
        )]));
        assert_eq!(
            market_price_numeric("WHEAT", &json!(9999), &params).unwrap(),
            json!(2)
        );
    }

    #[test]
    fn integral_float_and_boolean_horizons_keep_their_python_numeric_kinds() {
        assert!(exact_integer_shape_at_t("sq", &json!(9_007_199_254_740_992.0)).is_none());
        assert_eq!(
            exact_integer_shape_at_t("sq", &json!(true))
                .unwrap()
                .unwrap(),
            BigInt::from(1)
        );
        assert!(
            exact_integer_shape_at_t("sq", &json!(false))
                .unwrap()
                .is_err()
        );
    }

    #[test]
    fn seat_wealth_marks_liquid_holdings_to_market_and_productive_assets_at_book() {
        let mut header = fresh_header(8);
        header.initial.privates[0].shed.insert("WHEAT".into(), 2);
        header.initial.privates[0].seeds.insert("CARROT".into(), 1);
        let mut carried = IndexMap::new();
        carried.insert("EGG".to_string(), 1);
        header.initial.privates[0].inventories = vec![carried];
        let mut game = Game::from_header(&header).unwrap();
        // Freshly planted wheat carries one nominal unit that HARVEST refuses before
        // day 2, so it is book value only; a wheat plant at age 2 with three units and a
        // goose holding two eggs are marked to market; the weed is worth nothing; the
        // extra quadrant is carried at its purchase price.
        game.farms[0].tiles[0][0] = new_plant("WHEAT", 0, &game.config.turns_per_day);
        game.farms[0].tiles[0][1] = new_plant("WHEAT", -2, &game.config.turns_per_day);
        set_i64(
            game.farms[0].tiles[0][1].as_object_mut().unwrap(),
            "yield_units",
            3,
        );
        game.farms[0].tiles[0][2] = new_animal("GOOSE", 0);
        set_i64(
            game.farms[0].tiles[0][2].as_object_mut().unwrap(),
            "yield_units",
            2,
        );
        game.farms[0].tiles[0][3] = json!({"kind": "WEED"});
        game.farms[0].unlocked_quadrants.push("NE".to_string());
        let wheat = game.market.prices["WHEAT"].as_f64().unwrap();
        let egg = game.market.prices["EGG"].as_f64().unwrap();
        assert_eq!(wheat, 25.0);
        let expected = 3_000.0
            + 2.0 * wheat // shed
            + 20.0 // carrot seed in hand
            + egg // carried egg
            + 10.0 // planted wheat, age 0, book only
            + 10.0 + 3.0 * wheat // planted wheat, age 2, book plus ready units
            + 300.0 + 2.0 * egg // goose at cost plus held eggs
            + 1_000.0; // NE quadrant
        assert_eq!(game.seat_wealth(), vec![expected, 3_000.0]);
        assert_eq!(game.seat_money(), vec![3_000.0, 3_000.0]);
    }
}

fn pass_action() -> Value {
    json!(["PASS"])
}

fn market_price_number(market: &Market, item: &str) -> f64 {
    market.prices.get(item).and_then(Value::as_f64).unwrap_or(0.0)
}

fn seat_wealth(farm: &Farm, private: &PrivateState, market: &Market, day: i64) -> f64 {
    let mut wealth = farm.money;
    for (item, count) in &private.shed {
        wealth += market_price_number(market, item) * *count as f64;
    }
    for inventory in &private.inventories {
        for (item, count) in inventory {
            wealth += market_price_number(market, item) * *count as f64;
        }
    }
    for (name, count) in &private.seeds {
        if let Some(data) = crop(name) {
            wealth += data.seed as f64 * *count as f64;
        }
    }
    for row in &farm.tiles {
        for tile in row {
            let Value::Object(map) = tile else {
                continue;
            };
            if let Some(data) = map_str(map, "crop").and_then(crop) {
                wealth += data.seed as f64;
                if day - map_i64(map, "planted_day") >= data.first_yield_day {
                    let name = map_str(map, "crop").unwrap_or_default();
                    wealth += market_price_number(market, name) * map_i64(map, "yield_units") as f64;
                }
            } else if let Some(data) = map_str(map, "animal").and_then(animal) {
                wealth += data.cost as f64;
                wealth += market_price_number(market, data.product) * map_i64(map, "yield_units") as f64;
            }
        }
    }
    for extra in 0..farm.unlocked_quadrants.len().saturating_sub(1) {
        wealth += LAND_PRICES[min(extra, LAND_PRICES.len() - 1)] as f64;
    }
    wealth
}

fn value_i64(value: &Value) -> Option<i64> {
    python_int_i64(value, "integer value").ok()
}

fn map_i64(map: &Map<String, Value>, key: &str) -> i64 {
    map.get(key).and_then(value_i64).unwrap_or(0)
}

fn map_bool(map: &Map<String, Value>, key: &str) -> bool {
    map.get(key).and_then(Value::as_bool).unwrap_or(false)
}

fn map_str<'a>(map: &'a Map<String, Value>, key: &str) -> Option<&'a str> {
    map.get(key).and_then(Value::as_str)
}

fn set_i64(map: &mut Map<String, Value>, key: &str, value: i64) {
    map.insert(key.to_string(), Value::from(value));
}

fn set_bool(map: &mut Map<String, Value>, key: &str, value: bool) {
    map.insert(key.to_string(), Value::from(value));
}

fn quadrant_of(x: usize, y: usize, board_size: usize) -> String {
    let half = board_size / 2;
    format!(
        "{}{}",
        if y < half { "N" } else { "S" },
        if x < half { "W" } else { "E" }
    )
}

fn shed_access_tiles(board_size: usize) -> Vec<(i64, i64)> {
    let half = (board_size / 2) as i64;
    vec![
        (half - 1, half - 1),
        (half, half - 1),
        (half - 1, half),
        (half, half),
    ]
}

fn default_spawn(board_size: usize) -> Vec<i64> {
    shed_access_tiles(board_size)
        .into_iter()
        .find(|(x, y)| quadrant_of(*x as usize, *y as usize, board_size) == "NW")
        .map(|(x, y)| vec![x, y])
        .unwrap_or_else(|| vec![0, 0])
}

fn is_shed_adjacent(pos: (i64, i64), board_size: usize) -> bool {
    shed_access_tiles(board_size).contains(&pos)
}

fn farmer_position(farm: &Farm, index: usize) -> Option<(i64, i64)> {
    let pos = if index == 0 {
        &farm.farmer
    } else {
        farm.hands.get(index - 1)?
    };
    (pos.len() >= 2).then(|| (pos[0], pos[1]))
}

fn set_farmer_position(farm: &mut Farm, index: usize, pos: (i64, i64)) {
    let target = vec![pos.0, pos.1];
    if index == 0 {
        farm.farmer = target;
    } else if let Some(hand) = farm.hands.get_mut(index - 1) {
        *hand = target;
    }
}

fn inventory(private: &mut PrivateState, index: usize) -> &mut Inventory {
    while private.inventories.len() <= index {
        private.inventories.push(IndexMap::new());
    }
    &mut private.inventories[index]
}

fn inv_add(inv: &mut Inventory, item: &str, count: i64) {
    *inv.entry(item.to_string()).or_default() += count;
}

fn inv_take(inv: &mut Inventory, item: &str, count: i64) -> bool {
    let available = inv.get(item).copied().unwrap_or(0);
    if available < count {
        return false;
    }
    let left = available - count;
    if left == 0 {
        inv.shift_remove(item);
    } else {
        inv.insert(item.to_string(), left);
    }
    true
}

fn new_plant(name: &str, day: i64, turns_per_day: &PyInt) -> Value {
    let data = crop(name).expect("validated crop");
    let max_lifespan_step = if data.ongoing {
        Value::from(-1)
    } else {
        let lifespan = BigInt::from(day + data.max_yield_day + 1) * &turns_per_day.0;
        integer_json(&lifespan).expect("BigInt is always a valid arbitrary-precision JSON integer")
    };
    json!({
        "kind": "PLANT",
        "crop": name,
        "planted_day": day,
        "watered_today": false,
        "consecutive_unwatered": 1,
        "yield_units": if data.ongoing { 0 } else { 1 },
        "max_lifespan_step": max_lifespan_step,
        "fertilized_until_day": -1,
    })
}

fn new_animal(name: &str, day: i64) -> Value {
    let data = animal(name).expect("validated animal");
    json!({
        "kind": data.structure,
        "animal": name,
        "placed_day": day,
        "yield_units": 0,
        "consecutive_unfed": 0,
        "fed_today": false,
        "cared_today": false,
        "fertilizer_available": false,
        "pending_care_bonus": 0,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UnitActionKind {
    None,
    Movement,
    Production,
    Logistics,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UnitActionEffect {
    None,
    Movement,
    Production,
    Logistics,
}

fn unit_action_kind(action: &Value) -> UnitActionKind {
    let Some(op) = action
        .as_array()
        .and_then(|parts| parts.first())
        .and_then(Value::as_str)
    else {
        return UnitActionKind::None;
    };
    match op {
        "NORTH" | "SOUTH" | "EAST" | "WEST" => UnitActionKind::Movement,
        "DROP" | "PICKUP" | "PLACE" => UnitActionKind::Logistics,
        "PLANT" | "WATER" | "HARVEST" | "FERTILIZE" | "DIG" | "BUILD_COOP" | "BUILD_PASTURE"
        | "FEED" | "COLLECT_FERTILIZER" | "CARE" => UnitActionKind::Production,
        _ => UnitActionKind::None,
    }
}

/// Post-transition biological readiness, after decay and any day-close. This
/// does not imply an available worker, reachable tile, or time to sell at terminal.
fn record_upstream_stock(metrics: &mut UnitSeatMetrics, farm: &Farm, day: usize) {
    for tile in farm.tiles.iter().flatten().filter_map(Value::as_object) {
        let units = map_i64(tile, "yield_units");
        let ready_key = if map_str(tile, "kind") == Some("PLANT") {
            *metrics
                .upstream
                .get_mut("upstream_crop_tile_steps")
                .unwrap() += 1;
            let ready = map_str(tile, "crop").and_then(crop).is_some_and(|data| {
                BigInt::from(day) - BigInt::from(map_i64(tile, "planted_day"))
                    >= BigInt::from(data.first_yield_day)
            });
            ready.then_some("upstream_ready_crop_tile_steps")
        } else if map_str(tile, "animal").and_then(animal).is_some() {
            *metrics
                .upstream
                .get_mut("upstream_animal_tile_steps")
                .unwrap() += 1;
            Some("upstream_ready_animal_tile_steps")
        } else {
            None
        };
        if units > 0
            && let Some(key) = ready_key
        {
            *metrics.upstream.get_mut(key).unwrap() += 1;
            *metrics
                .upstream
                .get_mut("upstream_ready_product_unit_steps")
                .unwrap() += units as u64;
        }
    }
}

impl UnitSeatMetrics {
    /// Submitted minus committed unit commands over every upstream verb: each verb has
    /// exactly one movement/production/logistics kind, and a command commits exactly
    /// when its effect is not `None`, so the kind totals equal the per-verb sums
    /// (test `ineffective_commands_equal_the_per_verb_upstream_sums`).
    pub fn ineffective_unit_commands(&self) -> u64 {
        let submitted = self.submitted_move_actions
            + self.submitted_production_actions
            + self.submitted_logistics_actions;
        let committed = self.committed_move_actions
            + self.committed_production_actions
            + self.committed_logistics_actions;
        submitted.saturating_sub(committed)
    }
}

fn record_unit_submission(metrics: &mut UnitSeatMetrics, action: &Value, exists: bool) {
    if !exists {
        return;
    }
    if let Some((_, submitted, _)) = upstream_verb(action) {
        *metrics.upstream.get_mut(submitted).unwrap() += 1;
    }
    match unit_action_kind(action) {
        UnitActionKind::Movement => metrics.submitted_move_actions += 1,
        UnitActionKind::Production => metrics.submitted_production_actions += 1,
        UnitActionKind::Logistics => metrics.submitted_logistics_actions += 1,
        UnitActionKind::None => {}
    }
}

fn upstream_verb(action: &Value) -> Option<&'static (&'static str, &'static str, &'static str)> {
    let op = action.as_array()?.first()?.as_str()?;
    UPSTREAM_VERBS.iter().find(|(verb, _, _)| *verb == op)
}

fn record_unit_effect(metrics: &mut UnitSeatMetrics, action: &Value, effect: UnitActionEffect) {
    if effect != UnitActionEffect::None
        && let Some((_, _, committed)) = upstream_verb(action)
    {
        *metrics.upstream.get_mut(committed).unwrap() += 1;
    }
    match effect {
        UnitActionEffect::Movement => metrics.committed_move_actions += 1,
        UnitActionEffect::Production => metrics.committed_production_actions += 1,
        UnitActionEffect::Logistics => metrics.committed_logistics_actions += 1,
        UnitActionEffect::None => {}
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_unit_action(
    farm: &mut Farm,
    private: &mut PrivateState,
    index: usize,
    action: &Value,
    board_size: usize,
    day: i64,
    turns_per_day: &PyInt,
    shed_capacity: &PyInt,
) -> Result<UnitActionEffect, String> {
    let Some(parts) = action.as_array() else {
        return Ok(UnitActionEffect::None);
    };
    let Some(op) = parts.first().and_then(Value::as_str) else {
        return Ok(UnitActionEffect::None);
    };
    let Some((fx, fy)) = farmer_position(farm, index) else {
        return Ok(UnitActionEffect::None);
    };

    let movement = match op {
        "NORTH" => Some((0, -1)),
        "SOUTH" => Some((0, 1)),
        "EAST" => Some((1, 0)),
        "WEST" => Some((-1, 0)),
        _ => None,
    };
    if let Some((dx, dy)) = movement {
        let (nx, ny) = (fx + dx, fy + dy);
        if nx >= 0 && ny >= 0 && nx < board_size as i64 && ny < board_size as i64 {
            set_farmer_position(farm, index, (nx, ny));
            return Ok(UnitActionEffect::Movement);
        }
        return Ok(UnitActionEffect::None);
    }
    if op == "PASS" {
        return Ok(UnitActionEffect::None);
    }
    if fx < 0 || fy < 0 || fx >= board_size as i64 || fy >= board_size as i64 {
        return Ok(UnitActionEffect::None);
    }
    let (x, y) = (fx as usize, fy as usize);
    let tile_snapshot = farm.tiles[y][x].clone();

    // Shed operations intentionally run before the LOCKED guard.
    if op == "DROP" {
        if !is_shed_adjacent((fx, fy), board_size) {
            return Ok(UnitActionEffect::None);
        }
        let carried = std::mem::take(inventory(private, index));
        let changed = carried.values().any(|count| *count > 0);
        for (item, count) in carried {
            if count <= 0 {
                continue;
            }
            let room = shed_capacity.room_from_i64_total(private.shed.values().sum::<i64>());
            let take = min(count, room);
            if take > 0 {
                *private.shed.entry(item).or_default() += take;
            }
        }
        return Ok(if changed {
            UnitActionEffect::Logistics
        } else {
            UnitActionEffect::None
        });
    }

    if op == "PICKUP" {
        if !is_shed_adjacent((fx, fy), board_size) {
            return Ok(UnitActionEffect::None);
        }
        let Some(item) = parts.get(1).and_then(Value::as_str) else {
            return Ok(UnitActionEffect::None);
        };
        let requested = match parts.get(2) {
            Some(value) => {
                python_int_i64(value, "PICKUP count").map_err(|error| error.to_string())?
            }
            None => 1,
        };
        if requested <= 0 {
            return Ok(UnitActionEffect::None);
        }
        let count = min(requested, private.shed.get(item).copied().unwrap_or(0));
        if count <= 0 {
            return Ok(UnitActionEffect::None);
        }
        if let Some(held) = private.shed.get_mut(item) {
            *held -= count;
        }
        inv_add(inventory(private, index), item, count);
        return Ok(UnitActionEffect::Logistics);
    }

    if op == "PLACE" {
        let Some(item) = parts.get(1).and_then(Value::as_str) else {
            return Ok(UnitActionEffect::None);
        };
        if let (Some(animal_data), Some(tile)) = (animal(item), tile_snapshot.as_object())
            && map_str(tile, "kind") == Some(animal_data.structure)
            && !tile.contains_key("animal")
        {
            if inv_take(inventory(private, index), item, 1) {
                farm.tiles[y][x] = new_animal(item, day);
                return Ok(UnitActionEffect::Logistics);
            }
            return Ok(UnitActionEffect::None);
        }
        if is_shed_adjacent((fx, fy), board_size) {
            let requested = match parts.get(2) {
                Some(value) => {
                    python_int_i64(value, "PLACE count").map_err(|error| error.to_string())?
                }
                None => 1,
            };
            if requested <= 0 {
                return Ok(UnitActionEffect::None);
            }
            let available = inventory(private, index).get(item).copied().unwrap_or(0);
            let room = shed_capacity.room_from_i64_total(private.shed.values().sum::<i64>());
            let count = min(min(requested, available), room);
            if count <= 0 {
                return Ok(UnitActionEffect::None);
            }
            let inv = inventory(private, index);
            let left = available - count;
            if left == 0 {
                inv.shift_remove(item);
            } else {
                inv.insert(item.to_string(), left);
            }
            *private.shed.entry(item.to_string()).or_default() += count;
            return Ok(UnitActionEffect::Logistics);
        }
        return Ok(UnitActionEffect::None);
    }

    if tile_snapshot.as_str() == Some("LOCKED") {
        return Ok(UnitActionEffect::None);
    }

    if op == "PLANT" {
        let Some(name) = parts.get(1).and_then(Value::as_str) else {
            return Ok(UnitActionEffect::None);
        };
        if crop(name).is_none() || !tile_snapshot.is_null() {
            return Ok(UnitActionEffect::None);
        }
        let available = private.seeds.get(name).copied().unwrap_or(0);
        if available <= 0 {
            return Ok(UnitActionEffect::None);
        }
        private.seeds.insert(name.to_string(), available - 1);
        farm.tiles[y][x] = new_plant(name, day, turns_per_day);
        return Ok(UnitActionEffect::Production);
    }

    if op == "WATER" {
        let Some(tile) = farm.tiles[y][x].as_object_mut() else {
            return Ok(UnitActionEffect::None);
        };
        if map_str(tile, "kind") != Some("PLANT") || map_bool(tile, "watered_today") {
            return Ok(UnitActionEffect::None);
        }
        let Some(name) = map_str(tile, "crop").map(str::to_string) else {
            return Ok(UnitActionEffect::None);
        };
        set_bool(tile, "watered_today", true);
        let data = crop(&name).expect("plant crop is valid");
        if !data.ongoing {
            let age_days = day - map_i64(tile, "planted_day");
            let window_start = (data.max_yield_day + 1) / 2;
            if window_start <= age_days && age_days <= data.max_yield_day {
                let bonus = if map_i64(tile, "fertilized_until_day") >= day {
                    2
                } else {
                    1
                };
                set_i64(
                    tile,
                    "yield_units",
                    min(data.max_yield, map_i64(tile, "yield_units") + bonus),
                );
            }
        }
        return Ok(UnitActionEffect::Production);
    }

    if op == "HARVEST" {
        let Some(tile) = tile_snapshot.as_object() else {
            return Ok(UnitActionEffect::None);
        };
        let units = map_i64(tile, "yield_units");
        if units <= 0 {
            return Ok(UnitActionEffect::None);
        }
        let mut harvested = false;
        if map_str(tile, "kind") == Some("PLANT") {
            let Some(name) = map_str(tile, "crop") else {
                return Ok(UnitActionEffect::None);
            };
            let Some(data) = crop(name) else {
                return Ok(UnitActionEffect::None);
            };
            if day - map_i64(tile, "planted_day") < data.first_yield_day {
                return Ok(UnitActionEffect::None);
            }
            inv_add(inventory(private, index), name, units);
            if data.ongoing {
                if let Some(live) = farm.tiles[y][x].as_object_mut() {
                    set_i64(live, "yield_units", 0);
                }
            } else {
                farm.tiles[y][x] = Value::Null;
            }
            harvested = true;
        } else if let Some(name) = map_str(tile, "animal") {
            let Some(data) = animal(name) else {
                return Ok(UnitActionEffect::None);
            };
            inv_add(inventory(private, index), data.product, units);
            if let Some(live) = farm.tiles[y][x].as_object_mut() {
                set_i64(live, "yield_units", 0);
            }
            harvested = true;
        }
        return Ok(if harvested {
            UnitActionEffect::Production
        } else {
            UnitActionEffect::None
        });
    }

    if op == "FERTILIZE" {
        let is_plant = farm.tiles[y][x]
            .as_object()
            .is_some_and(|tile| map_str(tile, "kind") == Some("PLANT"));
        if !is_plant || !inv_take(inventory(private, index), "FERTILIZER", 1) {
            return Ok(UnitActionEffect::None);
        }
        if let Some(tile) = farm.tiles[y][x].as_object_mut() {
            let until = map_i64(tile, "fertilized_until_day").max(day + 2);
            set_i64(tile, "fertilized_until_day", until);
        }
        return Ok(UnitActionEffect::Production);
    }

    if op == "DIG" {
        if tile_snapshot.is_null() {
            return Ok(UnitActionEffect::None);
        }
        if tile_snapshot
            .as_object()
            .is_some_and(|tile| tile.contains_key("animal"))
        {
            return Ok(UnitActionEffect::None);
        }
        farm.tiles[y][x] = Value::Null;
        return Ok(UnitActionEffect::Production);
    }

    if op == "BUILD_COOP" || op == "BUILD_PASTURE" {
        if tile_snapshot.is_null() {
            farm.tiles[y][x] = json!({"kind": if op == "BUILD_COOP" { "COOP" } else { "PASTURE" }});
            return Ok(UnitActionEffect::Production);
        }
        return Ok(UnitActionEffect::None);
    }

    if op == "FEED" {
        let eligible = farm.tiles[y][x]
            .as_object()
            .is_some_and(|tile| tile.contains_key("animal") && !map_bool(tile, "fed_today"));
        if !eligible || !inv_take(inventory(private, index), "WHEAT", 1) {
            return Ok(UnitActionEffect::None);
        }
        if let Some(tile) = farm.tiles[y][x].as_object_mut() {
            set_bool(tile, "fed_today", true);
        }
        return Ok(UnitActionEffect::Production);
    }

    if op == "COLLECT_FERTILIZER" {
        let eligible = farm.tiles[y][x].as_object().is_some_and(|tile| {
            tile.contains_key("animal") && map_bool(tile, "fertilizer_available")
        });
        if !eligible {
            return Ok(UnitActionEffect::None);
        }
        if let Some(tile) = farm.tiles[y][x].as_object_mut() {
            set_bool(tile, "fertilizer_available", false);
        }
        inv_add(inventory(private, index), "FERTILIZER", 1);
        return Ok(UnitActionEffect::Production);
    }

    if op == "CARE" {
        let Some(tile) = farm.tiles[y][x].as_object_mut() else {
            return Ok(UnitActionEffect::None);
        };
        if !tile.contains_key("animal") || map_bool(tile, "cared_today") {
            return Ok(UnitActionEffect::None);
        }
        set_bool(tile, "cared_today", true);
        return Ok(UnitActionEffect::Production);
    }
    Ok(UnitActionEffect::None)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OrderKind {
    Hire,
    BuyLand,
    BuySeed,
    BuyProduct,
    BuyAnimal,
    Sell,
}

#[derive(Clone, Debug)]
struct OrderState {
    kind: OrderKind,
    item: String,
    remaining: i64,
}

fn parse_order(value: &Value) -> Result<Option<OrderState>, String> {
    let Some(parts) = value.as_array() else {
        return Ok(None);
    };
    let Some(op) = parts.first().and_then(Value::as_str) else {
        return Ok(None);
    };
    if op == "HIRE" {
        return Ok(Some(OrderState {
            kind: OrderKind::Hire,
            item: String::new(),
            remaining: 0,
        }));
    }
    if op == "BUY_LAND" {
        return Ok(Some(OrderState {
            kind: OrderKind::BuyLand,
            item: String::new(),
            remaining: 0,
        }));
    }
    let kind = match op {
        "BUY_SEED" => OrderKind::BuySeed,
        "BUY_PRODUCT" => OrderKind::BuyProduct,
        "BUY_ANIMAL" => OrderKind::BuyAnimal,
        "SELL" => OrderKind::Sell,
        _ => return Ok(None),
    };
    let Some(item) = parts.get(1).and_then(Value::as_str) else {
        return Ok(None);
    };
    let Some(count) = parts.get(2) else {
        return Ok(None);
    };
    let remaining = match python_int_i64(count, "market order count") {
        Ok(remaining) => remaining,
        Err(PyIntError::Invalid(_)) => return Ok(None),
        Err(PyIntError::Overflow(error)) => return Err(error),
    };
    Ok((remaining > 0).then_some(OrderState {
        kind,
        item: item.to_string(),
        remaining,
    }))
}

fn shape(kind: &str, x: f64, t: Option<f64>) -> f64 {
    let x = x.max(0.0);
    match kind {
        "linear" => x,
        "sq" => x * x,
        "sqrt" => x.sqrt(),
        "log" => (1.0 + x).ln(),
        "log10" => (1.0 + x).log10(),
        "hinge" => {
            let Some(t) = t.filter(|t| *t > 0.0) else {
                return x;
            };
            let u = x / t;
            u + HINGE_GAIN * (u - 1.0).max(0.0).powi(2)
        }
        _ => x,
    }
}

fn exact_integer_shape_at_t(kind: &str, t: &Value) -> Option<Result<BigInt, String>> {
    let value = integer_value(t)?;
    Some(value.and_then(|value| {
        // `_shape` begins with max(0.0, x). At zero or below, Python retains
        // the float 0.0 branch, so the integer fast path must not apply.
        if value <= BigInt::from(0) {
            return Err("shape denominator is zero".to_string());
        }
        match kind {
            "linear" => Ok(value),
            "sq" => Ok(&value * &value),
            // For positive T, Python's hinge denominator is exactly
            // `_shape("hinge", T, T) == 1.0`, even when T is too wide to
            // convert to float on its own: the integer true-division is T/T.
            "hinge" => Ok(BigInt::from(1)),
            "sqrt" | "log" | "log10" => {
                Err("shape denominator uses floating arithmetic".to_string())
            }
            // Unknown functions deliberately degenerate to linear.
            _ => Ok(value),
        }
    }))
}

#[derive(Debug)]
enum MarketArithmetic {
    Integer(BigInt),
    Float(f64),
}

fn market_float(value: &Value, label: &str) -> Result<f64, String> {
    if let Some(integer) = integer_value(value) {
        return integer?
            .to_f64()
            .filter(|number| number.is_finite())
            .ok_or_else(|| format!("{label} integer is too large to convert to Python float"));
    }
    numeric(value, label)
}

fn market_amplitude_product(
    target: &Value,
    base: &Value,
    item: &str,
    target_key: &str,
) -> Result<MarketArithmetic, String> {
    if let (Some(target), Some(base)) = (integer_value(target), integer_value(base)) {
        return Ok(MarketArithmetic::Integer(target? * base?));
    }

    let target = market_float(target, &format!("marketParams[{item:?}][{target_key:?}]"))?;
    let base = market_float(base, &format!("marketParams[{item:?}][\"base\"]"))?;
    Ok(MarketArithmetic::Float(target * base))
}

fn market_shape_at_t(
    function: &str,
    t_value: &Value,
    item: &str,
) -> Result<MarketArithmetic, String> {
    if let Some(Ok(denominator)) = exact_integer_shape_at_t(function, t_value) {
        return Ok(MarketArithmetic::Integer(denominator));
    }

    let t = market_float(t_value, &format!("marketParams[{item:?}][\"T\"]"))?;
    Ok(MarketArithmetic::Float(shape(function, t, Some(t))))
}

fn market_true_divide(
    numerator: MarketArithmetic,
    denominator: MarketArithmetic,
    item: &str,
    t_value: &Value,
) -> Result<f64, String> {
    let amplitude = match (numerator, denominator) {
        (MarketArithmetic::Integer(numerator), MarketArithmetic::Integer(denominator)) => {
            if denominator == BigInt::from(0) {
                return Err(format!(
                    "marketParams[{item:?}] divides by zero at T={t_value}"
                ));
            }
            BigRational::new(numerator, denominator)
                .to_f64()
                .filter(|value| value.is_finite())
                .ok_or_else(|| format!("market amplitude for {item} does not fit a finite float"))?
        }
        (MarketArithmetic::Integer(numerator), MarketArithmetic::Float(denominator)) => {
            if denominator == 0.0 {
                return Err(format!(
                    "marketParams[{item:?}] divides by zero at T={t_value}"
                ));
            }
            let numerator = numerator
                .to_f64()
                .filter(|value| value.is_finite())
                .ok_or_else(|| {
                    format!("market amplitude numerator for {item} does not fit a finite float")
                })?;
            numerator / denominator
        }
        (MarketArithmetic::Float(numerator), MarketArithmetic::Integer(denominator)) => {
            if denominator == BigInt::from(0) {
                return Err(format!(
                    "marketParams[{item:?}] divides by zero at T={t_value}"
                ));
            }
            // Python shapes an integer T before true division converts that shaped
            // denominator to float. In particular, sq(T) must not pre-coerce T.
            let denominator = denominator
                .to_f64()
                .filter(|value| value.is_finite())
                .ok_or_else(|| {
                    format!("market amplitude denominator for {item} does not fit a finite float")
                })?;
            numerator / denominator
        }
        (MarketArithmetic::Float(numerator), MarketArithmetic::Float(denominator)) => {
            if denominator == 0.0 {
                return Err(format!(
                    "marketParams[{item:?}] divides by zero at T={t_value}"
                ));
            }
            numerator / denominator
        }
    };

    if !amplitude.is_finite() {
        return Err(format!(
            "market amplitude for {item} is non-finite: {amplitude}"
        ));
    }
    Ok(amplitude)
}

fn market_distance_shape(
    params: &MarketParams,
    item: &str,
    function: &str,
    left: &Value,
    right: &Value,
    label: &str,
) -> Result<f64, String> {
    if let (Some(left), Some(right)) = (integer_value(left), integer_value(right)) {
        let distance = (left? - right?).max(BigInt::from(0));
        match function {
            "linear" => {
                return distance
                    .to_f64()
                    .ok_or_else(|| format!("{label} does not fit a finite float"));
            }
            "sq" => {
                return (&distance * &distance)
                    .to_f64()
                    .ok_or_else(|| format!("squared {label} does not fit a finite float"));
            }
            "hinge" => {
                let t_value = params
                    .get(item)
                    .and_then(|param| param.get("T"))
                    .ok_or_else(|| format!("marketParams[{item:?}][\"T\"] is missing"))?;
                if let Some(t) = integer_value(t_value) {
                    let t = t?;
                    if t <= BigInt::from(0) {
                        return distance
                            .to_f64()
                            .ok_or_else(|| format!("{label} does not fit a finite float"));
                    }
                    let u = BigRational::new(distance, t)
                        .to_f64()
                        .ok_or_else(|| format!("{label}/T does not fit a finite float"))?;
                    return Ok(u + HINGE_GAIN * (u - 1.0).max(0.0).powi(2));
                }
            }
            "sqrt" | "log" | "log10" => {}
            // `_shape` deliberately treats unknown functions as linear.
            _ => {
                return distance
                    .to_f64()
                    .ok_or_else(|| format!("{label} does not fit a finite float"));
            }
        }
    }

    let distance = numeric_distance(left, right, label)?;
    let t = (function == "hinge")
        .then(|| param_number(params, item, "T"))
        .transpose()?;
    Ok(shape(function, distance, t))
}

fn market_amplitude(
    params: &MarketParams,
    item: &str,
    target_key: &str,
    function: &str,
) -> Result<f64, String> {
    let param = params
        .get(item)
        .ok_or_else(|| format!("marketParams has no {item:?}"))?;
    let target = param
        .get(target_key)
        .ok_or_else(|| format!("marketParams[{item:?}][{target_key:?}] is missing"))?;
    let base_value = param
        .get("base")
        .ok_or_else(|| format!("marketParams[{item:?}][\"base\"] is missing"))?;
    let t_value = param
        .get("T")
        .ok_or_else(|| format!("marketParams[{item:?}][\"T\"] is missing"))?;

    // Preserve Python's left-to-right evaluation: multiply first, shape T with
    // its original int/float representation second, and only then true-divide.
    let numerator = market_amplitude_product(target, base_value, item, target_key)?;
    let denominator = market_shape_at_t(function, t_value, item)?;
    market_true_divide(numerator, denominator, item, t_value)
}

fn market_price_numeric(
    item: &str,
    inventory: &Value,
    params: &MarketParams,
) -> Result<Value, String> {
    let base = param_number(params, item, "base")?;
    let i0 = params
        .get(item)
        .and_then(|param| param.get("I0"))
        .ok_or_else(|| format!("marketParams[{item:?}][\"I0\"] is missing"))?;
    let price = if numeric_less(inventory, i0, "market inventory/I0")? {
        let function = param_string(params, item, "below_func");
        let amplitude = market_amplitude(params, item, "below_target", function)?;
        let distance_shape =
            market_distance_shape(params, item, function, i0, inventory, "market scarcity")?;
        base + amplitude * distance_shape
    } else {
        let function = param_string(params, item, "above_func");
        let amplitude = market_amplitude(params, item, "above_target", function)?;
        let distance_shape =
            market_distance_shape(params, item, function, inventory, i0, "market glut")?;
        base - amplitude * distance_shape
    };
    if !price.is_finite() {
        return Err(format!("market price for {item} is non-finite: {price}"));
    }
    // Python's round() is bankers/ties-to-even; f64::round() is not.
    let rounded = price.round_ties_even();
    if let Some(rounded) = rounded.to_i64()
        && rounded >= PRICE_FLOOR
    {
        return Ok(Value::from(rounded));
    }
    if rounded < PRICE_FLOOR as f64 {
        return Ok(Value::from(PRICE_FLOOR));
    }
    let rounded = BigInt::from_f64(rounded)
        .ok_or_else(|| format!("cannot convert rounded market price {rounded} to integer"))?;
    integer_json(&rounded.max(BigInt::from(PRICE_FLOOR)))
}

pub fn market_price(item: &str, inventory: i64, params: &MarketParams) -> i64 {
    market_price_numeric(item, &Value::from(inventory), params)
        .expect("valid market parameters")
        .as_i64()
        .expect("default market price fits i64")
}

fn refresh_prices(market: &mut Market, params: &MarketParams) -> Result<(), String> {
    for item in PRODUCTS {
        let inventory = market
            .inventory
            .get(item)
            .ok_or_else(|| format!("market inventory has no {item}"))?;
        market.prices.insert(
            item.to_string(),
            market_price_numeric(item, inventory, params)?,
        );
    }
    Ok(())
}

fn fib(index: usize) -> BigInt {
    let (mut a, mut b) = (BigInt::from(1), BigInt::from(1));
    for _ in 0..index {
        (a, b) = (b.clone(), a + b);
    }
    a
}

fn spawn_hand(farm: &Farm, board_size: usize) -> Vec<i64> {
    let access = shed_access_tiles(board_size);
    let mut occupancy = vec![0_usize; access.len()];
    for position in std::iter::once(&farm.farmer).chain(farm.hands.iter()) {
        if position.len() < 2 {
            continue;
        }
        if let Some(index) = access
            .iter()
            .position(|tile| *tile == (position[0], position[1]))
        {
            occupancy[index] += 1;
        }
    }
    let best = (0..access.len())
        .min_by_key(|index| (occupancy[*index], *index))
        .unwrap_or(0);
    vec![access[best].0, access[best].1]
}

fn do_hire(farm: &mut Farm, private: &mut PrivateState, board_size: usize, mult: &PyInt) -> bool {
    let cost = &mult.0 * fib(farm.hires_today);
    let cannot_afford = if farm.money.is_nan() {
        false
    } else if farm.money == f64::NEG_INFINITY {
        true
    } else if farm.money == f64::INFINITY {
        false
    } else {
        BigInt::from_f64(farm.money.floor()).is_none_or(|money| money < cost)
    };
    if cannot_afford {
        return false;
    }
    let Some(cost) = cost.to_f64() else {
        // A finite float bank cannot afford an integer outside the float domain.
        return false;
    };
    farm.money -= cost;
    farm.hires_today += 1;
    farm.hands.push(spawn_hand(farm, board_size));
    private.inventories.push(IndexMap::new());
    true
}

fn do_buy_land(farm: &mut Farm, board_size: usize) -> bool {
    let unlocked_extra = farm.unlocked_quadrants.len().saturating_sub(1);
    if unlocked_extra >= LAND_ORDER.len() {
        return false;
    }
    let cost = LAND_PRICES[unlocked_extra];
    if farm.money < cost as f64 {
        return false;
    }
    farm.money -= cost as f64;
    let quadrant = LAND_ORDER[unlocked_extra];
    farm.unlocked_quadrants.push(quadrant.to_string());
    for y in 0..board_size {
        for x in 0..board_size {
            if quadrant_of(x, y, board_size) == quadrant
                && farm.tiles[y][x].as_str() == Some("LOCKED")
            {
                farm.tiles[y][x] = Value::Null;
            }
        }
    }
    true
}

fn commit_unit(
    kind: OrderKind,
    item: &str,
    price: &Value,
    farm: &mut Farm,
    private: &mut PrivateState,
    market: &mut Market,
    shed_capacity: &PyInt,
) -> Result<bool, String> {
    let price_number = numeric(price, &format!("quoted price for {item}"))?;
    match kind {
        OrderKind::Sell => {
            let held = private.shed.get(item).copied().unwrap_or(0);
            if held <= 0 {
                return Ok(false);
            }
            private.shed.insert(item.to_string(), held - 1);
            farm.money += price_number;
            if numeric_less(&Value::from(1), price, "quoted price")? {
                add_market_units(
                    market
                        .inventory
                        .entry(item.to_string())
                        .or_insert_with(|| Value::from(0)),
                    1,
                )?;
            }
            Ok(true)
        }
        OrderKind::BuyProduct => {
            if farm.money < price_number
                || !shed_capacity.contains_i64_total(private.shed.values().sum::<i64>())
            {
                return Ok(false);
            }
            farm.money -= price_number;
            *private.shed.entry(item.to_string()).or_default() += 1;
            add_market_units(
                market
                    .inventory
                    .entry(item.to_string())
                    .or_insert_with(|| Value::from(0)),
                -1,
            )?;
            Ok(true)
        }
        OrderKind::BuySeed => {
            if farm.money < price_number {
                return Ok(false);
            }
            farm.money -= price_number;
            *private.seeds.entry(item.to_string()).or_default() += 1;
            Ok(true)
        }
        OrderKind::BuyAnimal => {
            if farm.money < price_number
                || !shed_capacity.contains_i64_total(private.shed.values().sum::<i64>())
            {
                return Ok(false);
            }
            farm.money -= price_number;
            *private.shed.entry(item.to_string()).or_default() += 1;
            Ok(true)
        }
        OrderKind::Hire | OrderKind::BuyLand => Ok(false),
    }
}

impl Game {
    fn process_market(&mut self, actions: &[Value]) -> Result<MarketStepMetrics, String> {
        let mut metrics = MarketStepMetrics::default();
        let max_orders = self.config.max_market_orders_per_turn.capped_usize();
        for (player, action) in actions.iter().enumerate() {
            if let Some(market) = action.as_object().and_then(|a| a.get("market")) {
                let ignored = market.as_array().map(|a| a.len().saturating_sub(max_orders) as u64).unwrap_or(1);
                if let Some(c) = econ_mut!(self, player) { c[ECON_MARKET_UNFILLED_UNITS] += ignored; }
            }
        }
        let queues: Vec<Vec<Value>> = actions
            .iter()
            .map(|action| {
                action
                    .as_object()
                    .and_then(|obj| obj.get("market"))
                    .and_then(Value::as_array)
                    .map(|orders| orders.iter().take(max_orders).cloned().collect())
                    .unwrap_or_default()
            })
            .collect();
        let max_len = queues.iter().map(Vec::len).max().unwrap_or(0);
        for slot in 0..max_len {
            // Kaggriculture is exactly two-player; keep telemetry counters on the
            // stack rather than allocating two auxiliary vectors per market slot.
            let mut committed_units = [0_u64; 2];
            let mut states: Vec<Option<OrderState>> = queues
                .iter()
                .map(|queue| queue.get(slot).map(parse_order).transpose())
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .map(Option::flatten)
                .collect();
            let requested: [u64; 2] = std::array::from_fn(|player| {
                if let Some(state) = states.get(player).and_then(Option::as_ref) {
                    if matches!(state.kind, OrderKind::Hire | OrderKind::BuyLand) { 1 }
                    else { state.remaining.max(0) as u64 }
                } else {
                    queues.get(player).and_then(|queue| queue.get(slot))
                        .map(econ_requested_market_units).unwrap_or(0)
                }
            });
            let slot_kinds: [Option<OrderKind>; 2] =
                std::array::from_fn(|player| states.get(player).and_then(Option::as_ref).map(|state| state.kind));
            let mut submitted = [false; 2];
            for (player, state) in states.iter().enumerate() {
                let Some(state) = state else {
                    continue;
                };
                submitted[player] = true;
                metrics.submitted_orders += 1;
                metrics.by_seat[player].submitted_orders += 1;
                match state.kind {
                    OrderKind::Sell => {
                        metrics.sell_orders += 1;
                        metrics.by_seat[player].sell_orders += 1;
                    }
                    OrderKind::BuySeed | OrderKind::BuyProduct | OrderKind::BuyAnimal => {
                        metrics.buy_orders += 1;
                        metrics.by_seat[player].buy_orders += 1;
                    }
                    OrderKind::Hire => {
                        metrics.hire_orders += 1;
                        metrics.by_seat[player].hire_orders += 1;
                    }
                    OrderKind::BuyLand => {
                        metrics.land_orders += 1;
                        metrics.by_seat[player].land_orders += 1;
                    }
                }
            }

            for (player, state) in states.iter_mut().enumerate() {
                let kind = state.as_ref().map(|state| state.kind);
                match kind {
                    Some(OrderKind::Hire) => {
                        let cash_before = self.farms[player].money;
                        if do_hire(
                            &mut self.farms[player],
                            &mut self.privates[player],
                            self.config.board_size,
                            &self.config.farm_hand_cost_mult,
                        ) {
                            committed_units[player] += 1;
                            self.count_attrib_investment(player, OrderKind::Hire, cash_before);
                            if self.config.turns_per_day.divides_usize(self.step + 1)
                                || BigInt::from(self.step) >= &self.config.episode_steps.0 - BigInt::from(2) {
                                if let Some(c) = econ_mut!(self, player) {
                                    c[ECON_HIRE_WASTED_CASH] += (cash_before - self.farms[player].money).max(0.0) as u64;
                                }
                            }
                        }
                        *state = None;
                    }
                    Some(OrderKind::BuyLand) => {
                        let cash_before = self.farms[player].money;
                        if do_buy_land(&mut self.farms[player], self.config.board_size) {
                            committed_units[player] += 1;
                            self.count_attrib_investment(player, OrderKind::BuyLand, cash_before);
                        }
                        *state = None;
                    }
                    _ => {}
                }
            }

            // Python increments its guard and breaks before quoting at 100_000,
            // permitting exactly 99_999 unit commits in one queue slot.
            for _ in 1..100_000 {
                let mut quotes: Vec<Option<(OrderKind, String, Value)>> = vec![None; states.len()];
                for player in 0..states.len() {
                    let Some(state) = states[player].as_ref() else {
                        continue;
                    };
                    if state.remaining <= 0 {
                        continue;
                    }
                    let price = match state.kind {
                        OrderKind::Sell if PRODUCTS.contains(&state.item.as_str()) => {
                            let inventory =
                                self.market.inventory.get(&state.item).ok_or_else(|| {
                                    format!("market inventory has no {}", state.item)
                                })?;
                            Some(market_price_numeric(&state.item, inventory, &self.params)?)
                        }
                        OrderKind::BuyProduct
                            if state.item == "WHEAT" || state.item == "FERTILIZER" =>
                        {
                            let mut inventory = self
                                .market
                                .inventory
                                .get(&state.item)
                                .ok_or_else(|| format!("market inventory has no {}", state.item))?
                                .clone();
                            add_market_units(&mut inventory, -1)?;
                            Some(market_price_numeric(&state.item, &inventory, &self.params)?)
                        }
                        OrderKind::BuySeed => crop(&state.item).map(|data| Value::from(data.seed)),
                        OrderKind::BuyAnimal => {
                            animal(&state.item).map(|data| Value::from(data.cost))
                        }
                        _ => None,
                    };
                    if let Some(price) = price {
                        quotes[player] = Some((state.kind, state.item.clone(), price));
                    } else {
                        states[player] = None;
                    }
                }
                if quotes.iter().all(Option::is_none) {
                    break;
                }

                let mut committed_any = false;
                for player in 0..quotes.len() {
                    let Some((kind, item, price)) = quotes[player].as_ref() else {
                        continue;
                    };
                    let ok = commit_unit(
                        *kind,
                        item,
                        price,
                        &mut self.farms[player],
                        &mut self.privates[player],
                        &mut self.market,
                        &self.config.shed_capacity,
                    )?;
                    if ok {
                        if let Some(state) = states[player].as_mut() {
                            state.remaining -= 1;
                        }
                        committed_units[player] += 1;
                        committed_any = true;
                        self.count_attrib_commit(player, *kind, item, price);
                        if let Some(c) = econ_mut!(self, player) {
                            let actual = price.as_f64().unwrap_or(0.0);
                            if *kind == OrderKind::Sell {
                                c[ECON_SALE_SHORTFALL_CASH] += (official_base_price(item) as f64 - actual).max(0.0) as u64;
                                c[ECON_FLOOR_SALE_UNITS] += (actual <= 1.0) as u64;
                            } else if matches!(kind, OrderKind::BuyProduct | OrderKind::BuySeed) {
                                let base = if *kind == OrderKind::BuySeed { crop(item).map(|d| d.seed).unwrap_or(0) }
                                    else { official_base_price(item) };
                                c[ECON_BUY_PREMIUM_CASH] += (actual - base as f64).max(0.0) as u64;
                            }
                        }
                        if *kind == OrderKind::Sell
                            && let Some(counters) = econ_mut!(self, player)
                        {
                            counters[ECON_SELL_UNITS] += 1;
                            // prices are integers (Python rounds them); cash is kept exact
                            counters[ECON_SELL_CASH] += price
                                .as_u64()
                                .or_else(|| price.as_f64().map(|value| value.round().max(0.0) as u64))
                                .unwrap_or(0);
                        }
                    } else {
                        states[player] = None;
                    }
                }
                if !committed_any {
                    break;
                }
            }
            refresh_prices(&mut self.market, &self.params)?;
            for (player, was_submitted) in submitted.into_iter().enumerate() {
                if let Some(c) = econ_mut!(self, player) {
                    c[ECON_MARKET_UNFILLED_UNITS] += requested[player].saturating_sub(committed_units[player]);
                }
                self.count_attrib_fill(player, slot_kinds[player], requested[player], committed_units[player]);
                if !was_submitted {
                    continue;
                }
                let units = committed_units[player];
                if units == 0 {
                    metrics.zero_commit_orders += 1;
                    metrics.by_seat[player].zero_commit_orders += 1;
                } else {
                    metrics.committed_orders += 1;
                    metrics.committed_units += units;
                    metrics.by_seat[player].committed_orders += 1;
                    metrics.by_seat[player].committed_units += units;
                }
            }
        }
        Ok(metrics)
    }
}

fn decay_plants(farm: &mut Farm, step: usize) -> (u64, u64) {
    let mut deaths = 0;
    let mut units_lost = 0;
    let board_size = farm.tiles.len();
    for y in 0..board_size {
        for x in 0..board_size {
            let tile = &mut farm.tiles[y][x];
            let Some(object) = tile.as_object_mut() else {
                continue;
            };
            if map_str(object, "kind") != Some("PLANT") {
                continue;
            }
            let Some(Ok(max_lifespan)) = object.get("max_lifespan_step").and_then(integer_value)
            else {
                continue;
            };
            let step = BigInt::from(step);
            if max_lifespan < BigInt::from(0)
                || step < max_lifespan
                || (&step - &max_lifespan) % BigInt::from(2) != BigInt::from(0)
            {
                continue;
            }
            let held = map_i64(object, "yield_units");
            let remaining = held - 1;
            if remaining <= 0 {
                units_lost += held.max(0) as u64;
                *tile = json!({"kind": "WEED"});
                deaths += 1;
            } else {
                units_lost += 1;
                set_i64(object, "yield_units", remaining);
            }
        }
    }
    (deaths, units_lost)
}

fn daily_refresh_plants(farm: &mut Farm, current_day: i64, turns_per_day: &PyInt) -> u64 {
    let mut deaths = 0;
    let next_day = current_day + 1;
    let board_size = farm.tiles.len();
    for y in 0..board_size {
        for x in 0..board_size {
            let tile_value = &mut farm.tiles[y][x];
            let Some(tile) = tile_value.as_object_mut() else {
                continue;
            };
            if map_str(tile, "kind") != Some("PLANT") {
                continue;
            }
            let was_watered = map_bool(tile, "watered_today");
            let unwatered = if was_watered {
                0
            } else {
                map_i64(tile, "consecutive_unwatered") + 1
            };
            set_i64(tile, "consecutive_unwatered", unwatered);
            set_bool(tile, "watered_today", false);
            if unwatered >= 2 {
                *tile_value = json!({"kind": "WEED"});
                deaths += 1;
                continue;
            }

            let Some(name) = map_str(tile, "crop").map(str::to_string) else {
                continue;
            };
            let Some(data) = crop(&name) else {
                continue;
            };
            if !data.ongoing {
                continue;
            }
            let days_since_first = next_day - map_i64(tile, "planted_day") - data.first_yield_day;
            if days_since_first < 0 || days_since_first % data.interval != 0 {
                continue;
            }
            let production_count = days_since_first / data.interval + 1;
            if production_count > data.max_yield {
                continue;
            }
            let fertilized = was_watered && map_i64(tile, "fertilized_until_day") >= current_day;
            let produced = if fertilized { 2 } else { 1 };
            set_i64(
                tile,
                "yield_units",
                min(data.max_yield, map_i64(tile, "yield_units") + produced),
            );
            if production_count == data.max_yield {
                let lifespan = BigInt::from(next_day + 1) * &turns_per_day.0;
                tile.insert(
                    "max_lifespan_step".to_string(),
                    integer_json(&lifespan)
                        .expect("BigInt is always a valid arbitrary-precision JSON integer"),
                );
            }
        }
    }
    deaths
}

fn daily_refresh_animals(farm: &mut Farm, current_day: i64) -> (u64, u64, u64) {
    let mut deaths = 0;
    let mut care_lost = 0u64;
    let mut fert_wasted = 0u64;
    let next_day = current_day + 1;
    let board_size = farm.tiles.len();
    for y in 0..board_size {
        for x in 0..board_size {
            let tile_value = &mut farm.tiles[y][x];
            let Some(tile) = tile_value.as_object_mut() else {
                continue;
            };
            let Some(name) = map_str(tile, "animal").map(str::to_string) else {
                continue;
            };
            let Some(data) = animal(&name) else {
                continue;
            };
            let fed = map_bool(tile, "fed_today");
            let consecutive = if fed {
                0
            } else {
                map_i64(tile, "consecutive_unfed") + 1
            };
            set_i64(tile, "consecutive_unfed", consecutive);
            if consecutive >= 2 {
                *tile_value = json!({"kind": data.structure});
                deaths += 1;
                continue;
            }

            let days_since_first = next_day - map_i64(tile, "placed_day") - data.first_yield_day;
            if days_since_first >= 0 && days_since_first % data.interval == 0 {
                let pending = map_i64(tile, "pending_care_bonus");
                let bonus = if fed { pending } else { 0 };
                let raw = map_i64(tile, "yield_units") + 1 + bonus;
                // econ v4: an unfed production day forfeits the pending bonus; max_held cuts the rest
                care_lost += (pending - bonus).max(0) as u64 + (raw - data.max_held).max(0) as u64;
                set_i64(tile, "yield_units", min(data.max_held, raw));
                set_i64(tile, "pending_care_bonus", 0);
            }
            if map_bool(tile, "cared_today") && fed {
                let pending = map_i64(tile, "pending_care_bonus") + 1;
                set_i64(tile, "pending_care_bonus", pending);
            }
            if map_bool(tile, "fertilizer_available") {
                fert_wasted += 1; // econ v4: yesterday's unit was never collected and cannot accumulate
            }
            set_bool(tile, "fertilizer_available", true);
            set_bool(tile, "fed_today", false);
            set_bool(tile, "cared_today", false);
        }
    }
    (deaths, care_lost, fert_wasted)
}

/// No extra state transition or RNG call: inspect the state immediately before daily refresh.
fn daily_econ_losses(farm: &Farm, day: i64) -> [u64; ECON_FIELDS] {
    let mut c = [0; ECON_FIELDS];
    for t in farm.tiles.iter().flatten().filter_map(Value::as_object) {
        let held = map_i64(t, "yield_units");
        if map_str(t, "kind") == Some("PLANT") {
            let watered = map_bool(t, "watered_today");
            if !watered && map_i64(t, "consecutive_unwatered") + 1 >= 2 {
                c[ECON_DEATH_HELD_UNITS] += held.max(0) as u64;
                continue;
            }
            let Some(data) = map_str(t, "crop").and_then(crop) else { continue; };
            let fertilized = map_i64(t, "fertilized_until_day") >= day;
            if data.ongoing {
                let since = day + 1 - map_i64(t, "planted_day") - data.first_yield_day;
                if since < 0 || since % data.interval != 0 || since / data.interval + 1 > data.max_yield { continue; }
                let increment = if watered && fertilized { 2 } else { 1 };
                c[ECON_CLIPPED_UNITS] += (held + increment - data.max_yield).max(0) as u64;
                if !watered && fertilized {
                    c[ECON_MISSED_GROWTH_UNITS] += (min(data.max_yield, held + 2) - min(data.max_yield, held + 1)).max(0) as u64;
                }
            } else {
                let age = day - map_i64(t, "planted_day");
                if !watered && (data.max_yield_day + 1) / 2 <= age && age <= data.max_yield_day {
                    c[ECON_MISSED_GROWTH_UNITS] += min((data.max_yield - held).max(0), if fertilized { 2 } else { 1 }) as u64;
                }
            }
        } else if map_str(t, "animal").and_then(animal).is_some() {
            let fed = map_bool(t, "fed_today");
            c[ECON_CARE_WASTED] += (map_bool(t, "cared_today") && !fed) as u64;
            if !fed && map_i64(t, "consecutive_unfed") + 1 >= 2 {
                c[ECON_DEATH_HELD_UNITS] += held.max(0) as u64
                    + map_i64(t, "pending_care_bonus").max(0) as u64
                    + map_bool(t, "fertilizer_available") as u64;
            }
        }
    }
    c
}

fn land_use(farm: &Farm, board_size: usize) -> [bool; 3] {
    let mut used = [false; 3];
    for (y, row) in farm.tiles.iter().enumerate() {
        for (x, tile) in row.iter().enumerate() {
            if tile.as_object().is_some_and(|t| map_str(t, "kind") == Some("PLANT") || t.contains_key("animal")) {
                if let Some(i) = LAND_ORDER.iter().position(|q| *q == quadrant_of(x, y, board_size)) { used[i] = true; }
            }
        }
    }
    used
}

/// Syntax only. Well-formed but ineffective actions are counted independently in v3.
/// Extra arguments are accepted by the official engine and therefore are not malformed.
fn malformed_unit(action: &Value) -> bool {
    let Some(p) = action.as_array() else { return true; };
    let Some(op) = p.first().and_then(Value::as_str) else { return true; };
    if op != "PASS" && !UPSTREAM_VERBS.iter().any(|(verb, _, _)| *verb == op) { return true; }
    if matches!(op, "PLANT" | "PICKUP" | "PLACE") && p.get(1).and_then(Value::as_str).is_none() { return true; }
    if matches!(op, "PICKUP" | "PLACE") && p.get(2).is_some_and(|v| python_int_i64(v, "econ command syntax").is_err()) { return true; }
    false
}

/// Official _parse_order accepts any item value; invalid commodity membership is a later
/// rejection. Keep its positive requested count even when the Rust execution parser rejects
/// a nonstring item. This observer neither changes parsing/commit behavior nor raises errors.
fn econ_requested_market_units(order: &Value) -> u64 {
    let Some(p) = order.as_array() else { return 1; };
    match p.first().and_then(Value::as_str) {
        Some("HIRE" | "BUY_LAND") => 1,
        Some("BUY_SEED" | "BUY_PRODUCT" | "BUY_ANIMAL" | "SELL") if p.len() >= 3 =>
            python_int_i64(&p[2], "econ market request").ok().filter(|&n| n > 0)
                .map(|n| n as u64).unwrap_or(1),
        _ => 1,
    }
}

fn official_base_price(item: &str) -> i64 {
    match item {
        "WHEAT" => 25, "CARROT" => 35, "TOMATO" => 60, "STRAWBERRY" => 120,
        "MELON" => 250, "EGG" => 50, "MILK" => 160, "WOOL" => 200,
        "FERTILIZER" => 100, _ => 0,
    }
}

fn is_drop(action: &Value) -> bool {
    action.as_array().and_then(|parts| parts.first()).and_then(Value::as_str) == Some("DROP")
}

/// Units held in the shed and every carried inventory (econ v4 overflow and unsold accounting).
fn held_goods(private: &PrivateState) -> u64 {
    let shed: i64 = private.shed.values().sum();
    let carried: i64 = private.inventories.iter().flat_map(|inv| inv.values()).sum();
    (shed.max(0) + carried.max(0)) as u64
}

/// Yield still held on crop and animal tiles (econ v4 unsold accounting at the game end).
fn held_tile_yield(farm: &Farm) -> u64 {
    farm.tiles
        .iter()
        .flatten()
        .filter_map(Value::as_object)
        .filter(|tile| map_str(tile, "kind") == Some("PLANT") || tile.get("animal").is_some())
        .map(|tile| map_i64(tile, "yield_units").max(0) as u64)
        .sum()
}

fn drop_inventories_to_shed(private: &mut PrivateState, capacity: &PyInt) {
    for index in 0..private.inventories.len() {
        let carried = std::mem::take(&mut private.inventories[index]);
        for (item, count) in carried {
            if count <= 0 {
                continue;
            }
            let room = capacity.room_from_i64_total(private.shed.values().sum::<i64>());
            let take = min(count, room);
            if take > 0 {
                *private.shed.entry(item).or_default() += take;
            }
        }
    }
}

impl Game {
    fn end_of_day(&mut self, day: usize, metrics: &mut UnitStepMetrics) -> Result<(), String> {
        // Python defers `float(weedSpawnChance)` until the first EOD. Keeping the
        // raw JSON number in Config preserves that failure phase for huge integers.
        let weed_spawn_chance = python_float(&self.config.weed_spawn_chance, "weedSpawnChance")?;
        // Python creates a fresh `random.Random` from this integer for every day.
        // `BigInt` preserves Python's unbounded signed multiply/XOR semantics even
        // for explicitly configured seeds wider than the competition's usual 31 bits.
        let seed = match &self.seed {
            ResolvedSeed::Integer(seed) => seed,
            ResolvedSeed::NonInteger(seed) => {
                return Err(format!(
                    "daily seed mixing requires an integer seed; got Python float {seed}"
                ));
            }
        };
        let mixed_seed = (seed * BigInt::from(1_000_003_u32)) ^ BigInt::from(day);
        let mut rng = PyRandom::from_seed_decimal(&mixed_seed.to_string())
            .map_err(|error| format!("daily RNG seed failed: {error}"))?;

        for player in 0..self.farms.len() {
            self.count_attrib_day_end(player, day as i64);
            let losses = daily_econ_losses(&self.farms[player], day as i64);
            if let Some(counters) = econ_mut!(self, player) {
                for (dst, delta) in counters.iter_mut().zip(losses) { *dst += delta; }
            }
            let drought_deaths = daily_refresh_plants(
                &mut self.farms[player],
                day as i64,
                &self.config.turns_per_day,
            );
            let (starvation_deaths, care_lost, fert_wasted) =
                daily_refresh_animals(&mut self.farms[player], day as i64);
            *metrics.by_seat[player]
                .upstream
                .get_mut("upstream_crop_drought_deaths")
                .unwrap() += drought_deaths;
            *metrics.by_seat[player]
                .upstream
                .get_mut("upstream_animal_starvation_deaths")
                .unwrap() += starvation_deaths;
            if let Some(counters) = econ_mut!(self, player) {
                counters[ECON_STARVATION] += starvation_deaths;
                counters[ECON_DROUGHT] += drought_deaths;
                counters[ECON_CARE_LOST] += care_lost;
                counters[ECON_FERT_WASTED] += fert_wasted;
            }
            let mut weeds_spawned = 0u64;
            for y in 0..self.config.board_size {
                for x in 0..self.config.board_size {
                    if self.farms[player].tiles[y][x].is_null() && rng.random() < weed_spawn_chance
                    {
                        self.farms[player].tiles[y][x] = json!({"kind": "WEED"});
                        weeds_spawned += 1;
                    }
                }
            }
            let held_before = held_goods(&self.privates[player]);
            drop_inventories_to_shed(&mut self.privates[player], &self.config.shed_capacity);
            let discarded = held_before.saturating_sub(held_goods(&self.privates[player]));
            if let Some(counters) = econ_mut!(self, player) {
                counters[ECON_WEEDS] += weeds_spawned;
                counters[ECON_OVERFLOW_UNITS] += discarded; // econ v4: the day-end deposit into a full shed
            }
            self.farms[player].farmer = default_spawn(self.config.board_size);
            self.farms[player].hands.clear();
            self.farms[player].hires_today = 0;
            self.privates[player].inventories = vec![IndexMap::new()];
        }

        let next_day = day + 1;
        if next_day > 0
            && self
                .config
                .town_shop_unlock_interval
                .divides_usize(next_day)
            && self.town.unlocked_shops.len() < MAX_SHOP_INSTANCES
        {
            let shop = rng
                .choice(&SORTED_SHOPS)
                .map_err(|error| format!("shop choice failed: {error}"))?;
            self.town.unlocked_shops.push((*shop).to_string());
        }
        Ok(())
    }
}

fn shop_products(name: &str) -> Option<&'static [&'static str]> {
    Some(match name {
        "BAKERY" => &["EGG", "WHEAT"],
        "PIZZA_SHOP" => &["MILK", "TOMATO", "WHEAT"],
        "BRUNCH_SPOT" => &["EGG", "WHEAT", "STRAWBERRY"],
        "YARN_STORE" => &["WOOL"],
        "ICE_CREAM_SHOP" => &["STRAWBERRY", "MILK", "WHEAT"],
        "PET_CAFE" => &["CARROT"],
        "SMOOTHIE_SHOP" => &["STRAWBERRY", "MILK"],
        "FARMERS_MARKET" => &["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
        _ => return None,
    })
}

impl Game {
    fn town_consume(&mut self, step: usize) -> Result<(), String> {
        if self.config.town_shop_sell_interval.divides_usize(step) {
            for shop in &self.town.unlocked_shops {
                let products =
                    shop_products(shop).ok_or_else(|| format!("unknown unlocked shop {shop:?}"))?;
                let multiplier = if products.len() == 1 { 2 } else { 1 };
                for item in products {
                    let inventory = self
                        .market
                        .inventory
                        .get_mut(*item)
                        .ok_or_else(|| format!("market inventory has no {item}"))?;
                    add_market_units(inventory, -multiplier)?;
                }
            }
        }
        if self.config.town_center_sell_interval.divides_usize(step) {
            for item in PRODUCTS {
                if item != "FERTILIZER" {
                    let inventory = self
                        .market
                        .inventory
                        .get_mut(item)
                        .ok_or_else(|| format!("market inventory has no {item}"))?;
                    add_market_units(inventory, -1)?;
                }
            }
        }
        refresh_prices(&mut self.market, &self.params)?;
        Ok(())
    }
}
