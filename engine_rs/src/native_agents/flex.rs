//! Complete native port of Flexonafft's adaptive Fieldbook controller.
//!
//! The public source is a stateful decision tree over six frozen 719-step
//! action programmes.  This port keeps those programmes as an expanded JSON
//! fixture and implements every observation-dependent operation in Rust:
//! route selection at steps 144 and 216, per-seat reset/state, hand alignment,
//! one-step sale leading and suppression, projected-shed accounting, terminal
//! liquidation, and the final fertilizer sweep.  It performs no Python calls.

use crate::{Farm, Game, PrivateState};
use indexmap::IndexMap;
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::HashSet;
use std::sync::OnceLock;

const SOURCE_SHA256: &str = "71bb7c8fcd2470dbbbf956df18061e5f8024f2924c96f11060e72c968ab638f1";
const LAST_PROGRAM_STEP: usize = 718;
const PROGRAM_STEPS: usize = 719;

const ITEMS: [&str; 12] = [
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
const PRODUCTS: [&str; 9] = [
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

#[derive(Debug, Deserialize)]
struct FieldbookFixture {
    source_sha256: String,
    routes: IndexMap<String, Vec<Value>>,
}

fn fixture() -> &'static FieldbookFixture {
    static FIXTURE: OnceLock<FieldbookFixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let decoded: FieldbookFixture =
            serde_json::from_str(include_str!("../../fixtures/flex-fieldbook-routes.json"))
                .expect("embedded Flex Fieldbook fixture must be valid JSON");
        assert_eq!(
            decoded.source_sha256, SOURCE_SHA256,
            "Flex Fieldbook fixture source digest changed"
        );
        for route in [0, 2, 3, 14, 16, 18] {
            let rows = decoded
                .routes
                .get(&route.to_string())
                .unwrap_or_else(|| panic!("Flex Fieldbook route {route} is missing"));
            assert_eq!(
                rows.len(),
                PROGRAM_STEPS,
                "Flex Fieldbook route {route} must contain 719 actions"
            );
        }
        decoded
    })
}

#[derive(Clone, Debug, PartialEq)]
struct Action {
    farmer: Vec<Value>,
    hands: Vec<Vec<Value>>,
    market: Vec<Vec<Value>>,
}

impl Action {
    fn pass() -> Vec<Value> {
        vec![Value::String("PASS".into())]
    }

    fn from_value(value: &Value) -> Result<Self, String> {
        let object = value
            .as_object()
            .ok_or_else(|| "Flex route action is not an object".to_string())?;
        let farmer = object
            .get("farmer")
            .and_then(Value::as_array)
            .cloned()
            .ok_or_else(|| "Flex route action has no farmer command".to_string())?;
        let hands = object
            .get("hands")
            .and_then(Value::as_array)
            .ok_or_else(|| "Flex route action has no hands list".to_string())?
            .iter()
            .map(|row| {
                row.as_array()
                    .cloned()
                    .ok_or_else(|| "Flex hand command is not an array".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let market = object
            .get("market")
            .and_then(Value::as_array)
            .ok_or_else(|| "Flex route action has no market list".to_string())?
            .iter()
            .map(|row| {
                row.as_array()
                    .cloned()
                    .ok_or_else(|| "Flex market order is not an array".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            farmer,
            hands,
            market,
        })
    }

    fn into_value(self) -> Value {
        json!({"farmer": self.farmer, "hands": self.hands, "market": self.market})
    }

    fn align_hands(&mut self, farm: &Farm) {
        self.hands.resize_with(farm.hands.len(), Self::pass);
        self.hands.truncate(farm.hands.len());
    }
}

fn route_action(route: i64, step: usize) -> Result<Action, String> {
    let rows = fixture()
        .routes
        .get(&route.to_string())
        .ok_or_else(|| format!("unknown Flex Fieldbook route {route}"))?;
    Action::from_value(
        rows.get(step.min(LAST_PROGRAM_STEP))
            .ok_or_else(|| format!("Flex Fieldbook route {route} has no step {step}"))?,
    )
}

fn text(value: &Value) -> Option<&str> {
    value.as_str()
}

fn integer(value: Option<&Value>, default: i64) -> i64 {
    value
        .and_then(|value| value.as_i64().or_else(|| value.as_f64().map(|x| x as i64)))
        .unwrap_or(default)
}

fn number(value: Option<&Value>) -> f64 {
    value.and_then(Value::as_f64).unwrap_or(0.0)
}

fn count(map: &IndexMap<String, i64>, item: &str) -> i64 {
    map.get(item).copied().unwrap_or(0)
}

fn order_limit(game: &Game) -> usize {
    game.config
        .max_market_orders_per_turn
        .capped_usize()
        .min(16)
}

fn episode_steps(game: &Game) -> usize {
    game.config.episode_steps.capped_usize()
}

#[derive(Clone, Debug)]
pub struct FlexSettings {
    pub branch_depth: i64,
    pub sell_lead: bool,
    pub terminal_liquidation: bool,
    pub final_fertilizer_sweep: bool,
    pub severe_cash_gap: f64,
    pub very_severe_cash_gap: f64,
    pub wheat_cheap: f64,
    pub low_cash: f64,
}

impl Default for FlexSettings {
    fn default() -> Self {
        Self {
            branch_depth: 2,
            sell_lead: true,
            terminal_liquidation: true,
            final_fertilizer_sweep: true,
            severe_cash_gap: -199.0,
            very_severe_cash_gap: -721.5,
            wheat_cheap: 29.5,
            low_cash: 242.5,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RouteGroup {
    Balanced,
    Yarn,
    LandRecovery,
}

#[derive(Clone, Debug)]
struct DecisionCache {
    shops: Vec<String>,
    cash: f64,
    cash_gap: f64,
    land_gap: i64,
    wheat_price: f64,
}

#[derive(Clone, Debug)]
struct RouteState {
    stage: u8,
    cache: DecisionCache,
    group: RouteGroup,
    route: i64,
}

#[derive(Clone, Debug, Default)]
struct SellState {
    due_step: i64,
    suppress: IndexMap<String, i64>,
}

impl SellState {
    fn fresh() -> Self {
        Self {
            due_step: -1,
            suppress: IndexMap::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct FlexController {
    settings: FlexSettings,
    route_states: [Option<RouteState>; 2],
    sell_states: [SellState; 2],
    last_debug: Value,
}

impl Default for FlexController {
    fn default() -> Self {
        Self::with_settings(FlexSettings::default())
    }
}

impl FlexController {
    pub fn with_settings(settings: FlexSettings) -> Self {
        Self {
            settings,
            route_states: std::array::from_fn(|_| None),
            sell_states: std::array::from_fn(|_| SellState::fresh()),
            last_debug: Value::Null,
        }
    }

    fn reset_seat(&mut self, seat: usize) {
        self.route_states[seat] = None;
        self.sell_states[seat] = SellState::fresh();
    }

    fn route(&mut self, game: &Game, step: usize, seat: usize) -> Result<i64, String> {
        if self.settings.branch_depth <= 0 || step < 144 {
            return Ok(0);
        }
        if self.route_states[seat]
            .as_ref()
            .is_none_or(|state| state.stage < 2)
        {
            let cache = capture(game, seat)?;
            let (group, route) = root_choice(&cache);
            self.route_states[seat] = Some(RouteState {
                stage: 2,
                cache,
                group,
                route,
            });
        }
        let needs_terminal_choice = self.settings.branch_depth >= 2
            && step >= 216
            && self.route_states[seat]
                .as_ref()
                .is_some_and(|state| state.stage < 3);
        if needs_terminal_choice {
            let terminal = capture(game, seat)?;
            let third_shop = game.town().unlocked_shops.get(2).map(String::as_str);
            let state = self.route_states[seat]
                .as_mut()
                .expect("Flex route state was initialized above");
            state.route = terminal_choice(
                state.group,
                &state.cache,
                &terminal,
                third_shop,
                &self.settings,
            );
            state.stage = 3;
        }
        Ok(self.route_states[seat]
            .as_ref()
            .expect("Flex route state was initialized above")
            .route)
    }

    pub fn action(&mut self, game: &Game, seat: usize) -> Result<Value, String> {
        if seat >= 2 {
            return Err(format!("Flex seat {seat} is outside the two-player game"));
        }
        let step = game.step_index().min(LAST_PROGRAM_STEP);
        if step == 0 {
            self.reset_seat(seat);
        }
        let route = self.route(game, step, seat)?;
        let farm = game
            .farms()
            .get(seat)
            .ok_or_else(|| format!("Flex seat {seat} is missing its farm"))?;
        let private = game
            .privates()
            .get(seat)
            .ok_or_else(|| format!("Flex seat {seat} is missing its private state"))?;
        let mut action = route_action(route, step)?;
        action.align_hands(farm);

        suppress_advanced_sale(&mut action, &self.sell_states[seat], step);
        if self.settings.sell_lead {
            let future = if step + 1 < PROGRAM_STEPS {
                route_action(route, step + 1)?
            } else {
                Action {
                    farmer: Action::pass(),
                    hands: vec![],
                    market: vec![],
                }
            };
            lead_sale(
                game,
                farm,
                private,
                &mut action,
                &future,
                &mut self.sell_states[seat],
                step,
            );
        } else {
            self.sell_states[seat] = SellState::fresh();
        }
        if self.settings.terminal_liquidation {
            terminal_sale(game, farm, private, &mut action, step);
            if self.settings.final_fertilizer_sweep {
                final_fertilizer_sweep(game, &mut action, step);
            }
        }

        let route_state = self.route_states[seat].as_ref();
        self.last_debug = json!({
            "step": step,
            "seat": seat,
            "route": route,
            "stage": route_state.map(|state| state.stage).unwrap_or(0),
            "group": route_state.map(|state| match state.group {
                RouteGroup::Balanced => "BALANCED",
                RouteGroup::Yarn => "YARN",
                RouteGroup::LandRecovery => "LAND_RECOVERY",
            }),
            "sell_due_step": self.sell_states[seat].due_step,
            "sell_suppress": self.sell_states[seat].suppress,
        });
        Ok(action.into_value())
    }

    pub fn debug(&self) -> &Value {
        &self.last_debug
    }
}

fn capture(game: &Game, seat: usize) -> Result<DecisionCache, String> {
    let own = game
        .farms()
        .get(seat)
        .ok_or_else(|| format!("Flex seat {seat} is missing its farm"))?;
    let rival_seat = 1usize
        .checked_sub(seat)
        .ok_or_else(|| format!("Flex seat {seat} has no rival"))?;
    let rival = game
        .farms()
        .get(rival_seat)
        .ok_or_else(|| format!("Flex seat {seat} is missing its rival farm"))?;
    Ok(DecisionCache {
        shops: game.town().unlocked_shops.iter().take(2).cloned().collect(),
        cash: own.money,
        cash_gap: own.money - rival.money,
        land_gap: own.unlocked_quadrants.len() as i64 - rival.unlocked_quadrants.len() as i64,
        wheat_price: number(game.market().prices.get("WHEAT")),
    })
}

fn root_choice(cache: &DecisionCache) -> (RouteGroup, i64) {
    let first = cache.shops.first().map(String::as_str);
    let second = cache.shops.get(1).map(String::as_str);
    if first == Some("YARN_STORE") {
        return (RouteGroup::Yarn, 3);
    }
    if second == Some("YARN_STORE") {
        if cache.land_gap < 0 {
            return (RouteGroup::LandRecovery, 16);
        }
        return (RouteGroup::Yarn, 3);
    }
    (RouteGroup::Balanced, 0)
}

fn terminal_choice(
    group: RouteGroup,
    cache: &DecisionCache,
    terminal: &DecisionCache,
    third_shop: Option<&str>,
    settings: &FlexSettings,
) -> i64 {
    match group {
        RouteGroup::Yarn => 3,
        RouteGroup::LandRecovery => {
            if terminal.cash_gap <= settings.severe_cash_gap {
                if terminal.wheat_price <= settings.wheat_cheap {
                    return if terminal.cash <= settings.low_cash {
                        18
                    } else {
                        16
                    };
                }
                return if terminal.cash_gap <= settings.very_severe_cash_gap {
                    16
                } else {
                    18
                };
            }
            let second = cache.shops.get(1).map(String::as_str);
            if second != Some("BRUNCH_SPOT") && third_shop != Some("FARMERS_MARKET") {
                16
            } else {
                18
            }
        }
        RouteGroup::Balanced => {
            let second = cache.shops.get(1).map(String::as_str);
            if third_shop != Some("YARN_STORE") && second != Some("YARN_STORE") {
                if terminal.cash_gap < 0.0 { 2 } else { 0 }
            } else {
                14
            }
        }
    }
}

fn shed_adjacent(position: &[i64], board_size: usize) -> bool {
    if position.len() < 2 {
        return false;
    }
    let half = board_size as i64 / 2;
    [half - 1, half].contains(&position[0]) && [half - 1, half].contains(&position[1])
}

fn projected_shed(
    game: &Game,
    farm: &Farm,
    private: &PrivateState,
    action: &Action,
) -> IndexMap<String, i64> {
    let mut projected: IndexMap<String, i64> = PRODUCTS
        .iter()
        .map(|item| ((*item).to_string(), count(&private.shed, item).max(0)))
        .collect();
    let mut total: i64 = projected.values().sum();
    let positions =
        std::iter::once(farm.farmer.as_slice()).chain(farm.hands.iter().map(Vec::as_slice));
    let commands =
        std::iter::once(action.farmer.as_slice()).chain(action.hands.iter().map(Vec::as_slice));

    for (index, (position, command)) in positions.zip(commands).enumerate() {
        if !shed_adjacent(position, farm.tiles.len()) {
            continue;
        }
        match command.first().and_then(text) {
            Some("PICKUP") => {
                let Some(item) = command.get(1).and_then(text) else {
                    continue;
                };
                let requested = integer(command.get(2), 1).max(0);
                if let Some(quantity) = projected.get_mut(item) {
                    let removed = (*quantity).min(requested);
                    *quantity -= removed;
                    total -= removed;
                }
            }
            Some("DROP") => {
                let inventory = private.inventories.get(index);
                for item in ITEMS {
                    let held = inventory.map(|row| count(row, item)).unwrap_or(0).max(0);
                    let room = game.config.shed_capacity.room_from_i64_total(total);
                    let dropped = held.min(room);
                    if let Some(quantity) = projected.get_mut(item) {
                        *quantity += dropped;
                    }
                    // Python includes dropped animals in capacity while returning
                    // only product counts in the projected shed.
                    total += dropped;
                }
            }
            _ => {}
        }
    }
    projected
}

fn suppress_advanced_sale(action: &mut Action, sell_state: &SellState, step: usize) {
    if sell_state.due_step != step as i64 {
        return;
    }
    let mut remaining = sell_state.suppress.clone();
    let mut kept = Vec::with_capacity(action.market.len());
    for mut order in action.market.drain(..) {
        let is_sell = order.first().and_then(text) == Some("SELL");
        if is_sell && order.len() >= 3 {
            let item = order.get(1).and_then(text).unwrap_or_default().to_string();
            let remaining_quantity = remaining.get(&item).copied().unwrap_or(0);
            if remaining_quantity > 0 {
                let quantity = integer(order.get(2), 0);
                let removed = quantity.max(0).min(remaining_quantity);
                order[2] = Value::from(quantity - removed);
                if let Some(value) = remaining.get_mut(&item) {
                    *value -= removed;
                }
            }
        }
        if !is_sell || integer(order.get(2), 0) > 0 {
            kept.push(order);
        }
    }
    action.market = kept;
}

fn lead_sale(
    game: &Game,
    farm: &Farm,
    private: &PrivateState,
    action: &mut Action,
    future_action: &Action,
    sell_state: &mut SellState,
    step: usize,
) {
    let mut next_state = SellState::fresh();
    let future_step = step + 1;
    let unlock_period = game
        .config
        .town_shop_unlock_interval
        .capped_usize()
        .saturating_mul(game.config.turns_per_day.capped_usize());
    if future_step >= PROGRAM_STEPS
        || future_step.is_multiple_of(unlock_period.max(1))
        || game.config.town_shop_sell_interval.divides_usize(step)
    {
        *sell_state = next_state;
        return;
    }

    let projected = projected_shed(game, farm, private, action);
    let mut planned: IndexMap<&str, i64> = PRODUCTS.iter().map(|item| (*item, 0)).collect();
    for order in &future_action.market {
        if order.first().and_then(text) != Some("SELL") {
            continue;
        }
        let Some(item) = order.get(1).and_then(text) else {
            continue;
        };
        if let Some(quantity) = planned.get_mut(item) {
            *quantity += integer(order.get(2), 0).max(0);
        }
    }
    let already: HashSet<String> = action
        .market
        .iter()
        .filter(|order| order.first().and_then(text) == Some("SELL"))
        .filter_map(|order| order.get(1).and_then(text).map(str::to_string))
        .collect();
    for item in PRODUCTS {
        let planned_quantity = planned.get(item).copied().unwrap_or(0);
        if matches!(item, "WHEAT" | "FERTILIZER") || planned_quantity <= 0 || already.contains(item)
        {
            continue;
        }
        let quantity = count(&projected, item).min(planned_quantity);
        if quantity <= 0 || number(game.market().prices.get(item)) < 2.0 {
            continue;
        }
        if action.market.len() >= order_limit(game) {
            break;
        }
        action.market.push(vec![
            Value::String("SELL".into()),
            Value::String(item.into()),
            Value::from(quantity),
        ]);
        next_state.suppress.insert(item.into(), quantity);
    }
    if !next_state.suppress.is_empty() {
        next_state.due_step = future_step as i64;
    }
    *sell_state = next_state;
}

fn terminal_sale(
    game: &Game,
    farm: &Farm,
    private: &PrivateState,
    action: &mut Action,
    step: usize,
) {
    if step < episode_steps(game).saturating_sub(2) {
        return;
    }
    action.market = projected_shed(game, farm, private, action)
        .into_iter()
        .filter(|(_, quantity)| *quantity > 0)
        .take(order_limit(game))
        .map(|(item, quantity)| {
            vec![
                Value::String("SELL".into()),
                Value::String(item),
                Value::from(quantity),
            ]
        })
        .collect();
}

fn final_fertilizer_sweep(game: &Game, action: &mut Action, step: usize) {
    if step != episode_steps(game).saturating_sub(2) || action.market.len() >= order_limit(game) {
        return;
    }
    let quantity = serde_json::to_value(&game.config.shed_capacity)
        .expect("validated shedCapacity must serialize as an integer");
    action.market.push(vec![
        Value::String("SELL".into()),
        Value::String("FERTILIZER".into()),
        quantity,
    ]);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache(shops: &[&str], cash: f64, cash_gap: f64, land_gap: i64, wheat: f64) -> DecisionCache {
        DecisionCache {
            shops: shops.iter().map(|shop| (*shop).to_string()).collect(),
            cash,
            cash_gap,
            land_gap,
            wheat_price: wheat,
        }
    }

    #[test]
    fn fixture_contains_all_six_complete_routes() {
        let fixture = fixture();
        assert_eq!(fixture.source_sha256, SOURCE_SHA256);
        assert_eq!(fixture.routes.len(), 6);
        for route in [0, 2, 3, 14, 16, 18] {
            assert_eq!(fixture.routes[&route.to_string()].len(), PROGRAM_STEPS);
        }
    }

    #[test]
    fn route_tree_matches_the_python_boundaries() {
        assert_eq!(
            root_choice(&cache(&["YARN_STORE"], 0.0, 0.0, 0, 0.0)),
            (RouteGroup::Yarn, 3)
        );
        assert_eq!(
            root_choice(&cache(&["CAFE", "YARN_STORE"], 0.0, 0.0, -1, 0.0)),
            (RouteGroup::LandRecovery, 16)
        );
        let settings = FlexSettings::default();
        let initial = cache(&["CAFE", "YARN_STORE"], 0.0, 0.0, -1, 0.0);
        assert_eq!(
            terminal_choice(
                RouteGroup::LandRecovery,
                &initial,
                &cache(&[], 200.0, -200.0, 0, 29.0),
                None,
                &settings
            ),
            18
        );
        assert_eq!(
            terminal_choice(
                RouteGroup::LandRecovery,
                &initial,
                &cache(&[], 300.0, -200.0, 0, 29.0),
                None,
                &settings
            ),
            16
        );
        assert_eq!(
            terminal_choice(
                RouteGroup::LandRecovery,
                &initial,
                &cache(&[], 300.0, -800.0, 0, 40.0),
                None,
                &settings
            ),
            16
        );
        assert_eq!(
            terminal_choice(
                RouteGroup::LandRecovery,
                &initial,
                &cache(&[], 300.0, 0.0, 0, 40.0),
                None,
                &settings
            ),
            16
        );
        assert_eq!(
            terminal_choice(
                RouteGroup::Balanced,
                &cache(&["CAFE", "PIZZA_SHOP"], 0.0, 0.0, 0, 0.0),
                &cache(&[], 0.0, -1.0, 0, 0.0),
                Some("FARMERS_MARKET"),
                &settings
            ),
            2
        );
    }

    #[test]
    fn suppression_removes_only_the_led_quantity() {
        let mut action = Action {
            farmer: Action::pass(),
            hands: vec![],
            market: vec![
                json!(["SELL", "MILK", 5]).as_array().unwrap().clone(),
                json!(["SELL", "WOOL", 2]).as_array().unwrap().clone(),
            ],
        };
        let state = SellState {
            due_step: 10,
            suppress: IndexMap::from([("MILK".into(), 3)]),
        };
        suppress_advanced_sale(&mut action, &state, 10);
        assert_eq!(
            action.market,
            vec![
                json!(["SELL", "MILK", 2]).as_array().unwrap().clone(),
                json!(["SELL", "WOOL", 2]).as_array().unwrap().clone(),
            ]
        );
    }
}
