//! Checked public rule context and exact next-hire costs.
use kaggriculture_engine::{Config, PyInt};
use num_bigint::BigInt;
use num_traits::ToPrimitive;

use super::observe::checked_f32;
use super::{ObserveError, ObserveErrorKind, Seat};

#[derive(Debug)]
enum HireCosts {
    Free,
    FinitePrefix {
        costs: Vec<f32>,
        last_supported_hire: usize,
    },
}

/// Immutable rule context bound to the same configuration used by the engine.
#[derive(Debug)]
pub struct ObservationConfig {
    pub(super) episode_steps: i64,
    pub(super) starting_money: i64,
    pub(super) orders: i64,
    pub(super) turns_per_day: i64,
    pub(super) shed_capacity: i64,
    pub(super) weed_spawn_chance: f64,
    pub(super) town_shop_unlock_interval: i64,
    pub(super) town_shop_sell_interval: i64,
    pub(super) town_center_sell_interval: i64,
    pub(super) farm_hand_cost_mult: i64,
    hire_costs: HireCosts,
}

fn integer(value: &PyInt, field: &str, positive: bool) -> Result<i64, ObserveError> {
    // Pinned PyInt has Serialize, but no Display or public integer getter.
    // Serialize its exact decimal only at config admission, never live state.
    let decimal = serde_json::to_string(value)
        .map_err(|error| ObserveError::new(ObserveErrorKind::Config, field, error.to_string()))?;
    let value = decimal.parse::<i64>().map_err(|_| {
        ObserveError::new(
            ObserveErrorKind::IntegerRange,
            field,
            "integer outside int64",
        )
    })?;
    if value < i64::from(positive) {
        return Err(ObserveError::new(
            ObserveErrorKind::Config,
            field,
            if positive {
                "must be positive"
            } else {
                "must be nonnegative"
            },
        ));
    }
    Ok(value)
}

impl ObservationConfig {
    pub fn new(config: &Config) -> Result<Self, ObserveError> {
        if config.board_size != 10 {
            return Err(ObserveError::new(
                ObserveErrorKind::Config,
                "boardSize",
                "must equal 10",
            ));
        }
        if !config.market_params.is_empty() {
            return Err(ObserveError::new(
                ObserveErrorKind::Config,
                "marketParams",
                "must be empty",
            ));
        }
        for name in config.extra.keys() {
            if !matches!(name.as_str(), "actTimeout" | "runTimeout" | "seed") {
                return Err(ObserveError::new(
                    ObserveErrorKind::Config,
                    name,
                    "unknown configuration key",
                ));
            }
        }
        let episode_steps = integer(&config.episode_steps, "episodeSteps", true)?;
        let starting_money = integer(&config.starting_money, "startingMoney", false)?;
        let orders = integer(
            &config.max_market_orders_per_turn,
            "maxMarketOrdersPerTurn",
            true,
        )?;
        let turns_per_day = integer(&config.turns_per_day, "turnsPerDay", true)?;
        let shed_capacity = integer(&config.shed_capacity, "shedCapacity", true)?;
        let town_shop_unlock_interval = integer(
            &config.town_shop_unlock_interval,
            "townShopUnlockInterval",
            true,
        )?;
        let town_shop_sell_interval = integer(
            &config.town_shop_sell_interval,
            "townShopSellInterval",
            true,
        )?;
        let town_center_sell_interval = integer(
            &config.town_center_sell_interval,
            "townCenterSellInterval",
            true,
        )?;
        let farm_hand_cost_mult = integer(&config.farm_hand_cost_mult, "farmHandCostMult", false)?;
        if orders > 10 {
            return Err(ObserveError::new(
                ObserveErrorKind::Config,
                "maxMarketOrdersPerTurn",
                "must be in 1..=10",
            ));
        }
        if turns_per_day
            .checked_mul(orders)
            .is_none_or(|product| product > 240)
        {
            return Err(ObserveError::new(
                ObserveErrorKind::Config,
                "turnsPerDay * maxMarketOrdersPerTurn",
                "checked product must be at most 240",
            ));
        }
        if !config.weed_spawn_chance.is_number() {
            return Err(ObserveError::new(
                ObserveErrorKind::Config,
                "weedSpawnChance",
                "must be numeric",
            ));
        }
        let weed_spawn_chance = config.weed_spawn_chance.as_f64().ok_or_else(|| {
            ObserveError::new(
                ObserveErrorKind::NonFinite,
                "weedSpawnChance",
                "must be finite in float64 and float32",
            )
        })?;
        if weed_spawn_chance < 0.0 {
            return Err(ObserveError::new(
                ObserveErrorKind::Config,
                "weedSpawnChance",
                "must be nonnegative",
            ));
        }
        for (field, value) in [
            ("episodeSteps", episode_steps as f64 / 1000.0),
            ("startingMoney", starting_money as f64 / 200000.0),
            ("maxMarketOrdersPerTurn", orders as f64 / 10.0),
            ("turnsPerDay", turns_per_day as f64 / 24.0),
            ("shedCapacity", shed_capacity as f64 / 1000.0),
            ("weedSpawnChance", weed_spawn_chance),
            (
                "townShopUnlockInterval",
                town_shop_unlock_interval as f64 / turns_per_day as f64,
            ),
            (
                "townShopSellInterval",
                town_shop_sell_interval as f64 / turns_per_day as f64,
            ),
            (
                "townCenterSellInterval",
                town_center_sell_interval as f64 / turns_per_day as f64,
            ),
            ("farmHandCostMult", farm_hand_cost_mult as f64 / 100.0),
        ] {
            checked_f32(value, field, None)?;
        }
        let hire_costs = if farm_hand_cost_mult == 0 {
            HireCosts::Free
        } else {
            let multiplier = BigInt::from(farm_hand_cost_mult);
            let (mut a, mut b) = (BigInt::from(1), BigInt::from(1));
            let mut costs = Vec::new();
            // Multiplication and recurrence are exact before the one f64 conversion.
            while let Some(cost) = (&multiplier * &a).to_f64() {
                let scaled = cost / 200000.0;
                let narrowed = scaled as f32;
                if !scaled.is_finite() || !narrowed.is_finite() {
                    break;
                }
                costs.push(narrowed);
                (a, b) = (b.clone(), a + b);
            }
            let last_supported_hire = costs.len() - 1;
            HireCosts::FinitePrefix {
                costs,
                last_supported_hire,
            }
        };
        Ok(Self {
            episode_steps,
            starting_money,
            orders,
            turns_per_day,
            shed_capacity,
            weed_spawn_chance,
            town_shop_unlock_interval,
            town_shop_sell_interval,
            town_center_sell_interval,
            farm_hand_cost_mult,
            hire_costs,
        })
    }

    pub(super) fn validate_hire_cost(
        &self,
        hires: usize,
        seat: Seat,
        role: usize,
    ) -> Result<(), ObserveError> {
        if let HireCosts::FinitePrefix {
            last_supported_hire,
            ..
        } = &self.hire_costs
        {
            if hires > *last_supported_hire {
                return Err(ObserveError::new(
                    ObserveErrorKind::NonFinite,
                    format!("player_features[{role},10]"),
                    format!("hire count {hires} exceeds last finite scaled-cost index {last_supported_hire}"),
                ).at_seat(seat));
            }
        }
        Ok(())
    }

    pub(super) fn prepared_hire_cost(&self, hires: usize) -> f32 {
        match &self.hire_costs {
            HireCosts::Free => 0.0,
            HireCosts::FinitePrefix { costs, .. } => costs[hires],
        }
    }
}
