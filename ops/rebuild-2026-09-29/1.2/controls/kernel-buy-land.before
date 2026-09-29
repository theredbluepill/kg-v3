//! Observation-local canonical Kaggriculture action grammar.

pub const MAX_ACTORS: usize = 241;
pub const MAX_ORDERS: usize = 10;
pub const MAX_FRAMES: usize = MAX_ACTORS + MAX_ORDERS + 1;
pub const SLOTS: usize = 12;
pub const TOKENS_PER_SEAT: usize = MAX_FRAMES * SLOTS;
pub const DEFAULT_HIRE_LIMIT: i64 = 241;
pub const QUANTITY_BASE: i64 = 32;
pub const MAX_QUANTITY: i64 = 1023;
pub const GRAMMAR_TABLES_VERSION: u32 = 1;
pub const UNIT_ACTOR_WIDTH: usize = 241;
pub const UNIT_KIND_WIDTH: usize = 20;
pub const UNIT_TARGET_WIDTH: usize = 128;
pub const UNIT_ITEM_WIDTH: usize = 16;
pub const UNIT_QUANTITY_PRESENT_WIDTH: usize = 2;
pub const UNIT_QUANTITY_HIGH_WIDTH: usize = 32;
pub const UNIT_QUANTITY_WIDTH: usize = 32;
pub const MARKET_KIND_WIDTH: usize = 8;
pub const MARKET_ITEM_WIDTH: usize = 16;
pub const MARKET_QUANTITY_HIGH_WIDTH: usize = 32;
pub const MARKET_QUANTITY_WIDTH: usize = 32;
pub const STOP_WIDTH: usize = 2;
pub const SLOT_NAMES: [&str; SLOTS] = [
    "unit_actor",
    "unit_kind",
    "unit_target",
    "unit_item",
    "unit_quantity_present",
    "unit_quantity_high",
    "unit_quantity",
    "market_kind",
    "market_item",
    "market_quantity_high",
    "market_quantity",
    "stop",
];
pub const SLOT_WIDTHS: [usize; SLOTS] = [
    UNIT_ACTOR_WIDTH,
    UNIT_KIND_WIDTH,
    UNIT_TARGET_WIDTH,
    UNIT_ITEM_WIDTH,
    UNIT_QUANTITY_PRESENT_WIDTH,
    UNIT_QUANTITY_HIGH_WIDTH,
    UNIT_QUANTITY_WIDTH,
    MARKET_KIND_WIDTH,
    MARKET_ITEM_WIDTH,
    MARKET_QUANTITY_HIGH_WIDTH,
    MARKET_QUANTITY_WIDTH,
    STOP_WIDTH,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum Slot {
    UnitActor = 0,
    UnitKind = 1,
    UnitTarget = 2,
    UnitItem = 3,
    UnitQuantityPresent = 4,
    UnitQuantityHigh = 5,
    UnitQuantity = 6,
    MarketKind = 7,
    MarketItem = 8,
    MarketQuantityHigh = 9,
    MarketQuantity = 10,
    Stop = 11,
}
impl TryFrom<usize> for Slot {
    type Error = String;
    fn try_from(value: usize) -> Result<Self, String> {
        match value {
            0 => Ok(Self::UnitActor),
            1 => Ok(Self::UnitKind),
            2 => Ok(Self::UnitTarget),
            3 => Ok(Self::UnitItem),
            4 => Ok(Self::UnitQuantityPresent),
            5 => Ok(Self::UnitQuantityHigh),
            6 => Ok(Self::UnitQuantity),
            7 => Ok(Self::MarketKind),
            8 => Ok(Self::MarketItem),
            9 => Ok(Self::MarketQuantityHigh),
            10 => Ok(Self::MarketQuantity),
            11 => Ok(Self::Stop),
            _ => Err(format!("invalid Slot value {value}")),
        }
    }
}
impl Slot {
    pub fn name(self) -> &'static str {
        match self {
            Self::UnitActor => "unit_actor",
            Self::UnitKind => "unit_kind",
            Self::UnitTarget => "unit_target",
            Self::UnitItem => "unit_item",
            Self::UnitQuantityPresent => "unit_quantity_present",
            Self::UnitQuantityHigh => "unit_quantity_high",
            Self::UnitQuantity => "unit_quantity",
            Self::MarketKind => "market_kind",
            Self::MarketItem => "market_item",
            Self::MarketQuantityHigh => "market_quantity_high",
            Self::MarketQuantity => "market_quantity",
            Self::Stop => "stop",
        }
    }
    pub fn width(self) -> usize {
        SLOT_WIDTHS[self as usize]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum UnitKind {
    None = 0,
    Pass = 1,
    North = 2,
    South = 3,
    East = 4,
    West = 5,
    Pickup = 6,
    Place = 7,
    Plant = 8,
    Water = 9,
    Harvest = 10,
    Drop = 11,
    BuildCoop = 12,
    BuildPasture = 13,
    Feed = 14,
    Fertilize = 15,
    CollectFertilizer = 16,
    Care = 17,
    Dig = 18,
}
impl TryFrom<i64> for UnitKind {
    type Error = String;
    fn try_from(value: i64) -> Result<Self, String> {
        match value {
            0 => Ok(Self::None),
            1 => Ok(Self::Pass),
            2 => Ok(Self::North),
            3 => Ok(Self::South),
            4 => Ok(Self::East),
            5 => Ok(Self::West),
            6 => Ok(Self::Pickup),
            7 => Ok(Self::Place),
            8 => Ok(Self::Plant),
            9 => Ok(Self::Water),
            10 => Ok(Self::Harvest),
            11 => Ok(Self::Drop),
            12 => Ok(Self::BuildCoop),
            13 => Ok(Self::BuildPasture),
            14 => Ok(Self::Feed),
            15 => Ok(Self::Fertilize),
            16 => Ok(Self::CollectFertilizer),
            17 => Ok(Self::Care),
            18 => Ok(Self::Dig),
            _ => Err(format!("invalid UnitKind value {value}")),
        }
    }
}
impl UnitKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "NONE",
            Self::Pass => "PASS",
            Self::North => "NORTH",
            Self::South => "SOUTH",
            Self::East => "EAST",
            Self::West => "WEST",
            Self::Pickup => "PICKUP",
            Self::Place => "PLACE",
            Self::Plant => "PLANT",
            Self::Water => "WATER",
            Self::Harvest => "HARVEST",
            Self::Drop => "DROP",
            Self::BuildCoop => "BUILD_COOP",
            Self::BuildPasture => "BUILD_PASTURE",
            Self::Feed => "FEED",
            Self::Fertilize => "FERTILIZE",
            Self::CollectFertilizer => "COLLECT_FERTILIZER",
            Self::Care => "CARE",
            Self::Dig => "DIG",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MarketKind {
    None = 0,
    Hire = 1,
    BuyLand = 2,
    BuySeed = 3,
    BuyProduct = 4,
    BuyAnimal = 5,
    Sell = 6,
    Empty = 7,
}
impl TryFrom<i64> for MarketKind {
    type Error = String;
    fn try_from(value: i64) -> Result<Self, String> {
        match value {
            0 => Ok(Self::None),
            1 => Ok(Self::Hire),
            2 => Ok(Self::BuyLand),
            3 => Ok(Self::BuySeed),
            4 => Ok(Self::BuyProduct),
            5 => Ok(Self::BuyAnimal),
            6 => Ok(Self::Sell),
            7 => Ok(Self::Empty),
            _ => Err(format!("invalid MarketKind value {value}")),
        }
    }
}
impl MarketKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "NONE",
            Self::Hire => "HIRE",
            Self::BuyLand => "BUY_LAND",
            Self::BuySeed => "BUY_SEED",
            Self::BuyProduct => "BUY_PRODUCT",
            Self::BuyAnimal => "BUY_ANIMAL",
            Self::Sell => "SELL",
            Self::Empty => "EMPTY",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ActionItem {
    None = 0,
    Wheat = 1,
    Carrot = 2,
    Tomato = 3,
    Strawberry = 4,
    Melon = 5,
    Egg = 6,
    Milk = 7,
    Wool = 8,
    Fertilizer = 9,
    Goose = 10,
    Cow = 11,
    Sheep = 12,
}
impl TryFrom<i64> for ActionItem {
    type Error = String;
    fn try_from(value: i64) -> Result<Self, String> {
        match value {
            0 => Ok(Self::None),
            1 => Ok(Self::Wheat),
            2 => Ok(Self::Carrot),
            3 => Ok(Self::Tomato),
            4 => Ok(Self::Strawberry),
            5 => Ok(Self::Melon),
            6 => Ok(Self::Egg),
            7 => Ok(Self::Milk),
            8 => Ok(Self::Wool),
            9 => Ok(Self::Fertilizer),
            10 => Ok(Self::Goose),
            11 => Ok(Self::Cow),
            12 => Ok(Self::Sheep),
            _ => Err(format!("invalid ActionItem value {value}")),
        }
    }
}
impl ActionItem {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "NONE",
            Self::Wheat => "WHEAT",
            Self::Carrot => "CARROT",
            Self::Tomato => "TOMATO",
            Self::Strawberry => "STRAWBERRY",
            Self::Melon => "MELON",
            Self::Egg => "EGG",
            Self::Milk => "MILK",
            Self::Wool => "WOOL",
            Self::Fertilizer => "FERTILIZER",
            Self::Goose => "GOOSE",
            Self::Cow => "COW",
            Self::Sheep => "SHEEP",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Shape {
    actors: u16,
    orders: u8,
    hire_limit: u16,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GrammarPlan {
    shape: Shape,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum UnitPhase {
    None,
    Ordinary,
    Transfer,
    Plant,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum MarketPhase {
    None,
    Hire,
    Other,
    Seed,
    Product,
    Animal,
    Sell,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct State {
    shape: Shape,
    units: u16,
    orders: u8,
    hires: u8,
    slot: Slot,
    unit: UnitPhase,
    market: MarketPhase,
    quantity_present: bool,
    quantity_high_zero: bool,
}
impl Shape {
    pub fn actors(self) -> usize {
        usize::from(self.actors)
    }
    pub fn order_limit(self) -> usize {
        usize::from(self.orders)
    }
    pub fn hire_limit(self) -> usize {
        usize::from(self.hire_limit)
    }
}
impl GrammarPlan {
    pub fn shape(&self) -> Shape {
        self.shape
    }
    pub fn start(&self) -> State {
        State {
            shape: self.shape,
            units: 0,
            orders: 0,
            hires: 0,
            slot: Slot::UnitActor,
            unit: UnitPhase::None,
            market: MarketPhase::None,
            quantity_present: false,
            quantity_high_zero: false,
        }
    }
}
/// Validate observation-derived dimensions before any narrowing or indexing.
pub fn plan(actors: i64, order_limit: i64, hire_limit: i64) -> Result<GrammarPlan, String> {
    for (field, value, max) in [
        ("actors", actors, 241),
        ("order_limit", order_limit, 10),
        ("hire_limit", hire_limit, 241),
    ] {
        if !(1..=max).contains(&value) {
            return Err(format!(
                "grammar shape {field} value {value} must be in 1..={max}"
            ));
        }
    }
    Ok(GrammarPlan {
        shape: Shape {
            actors: u16::try_from(actors).map_err(|e| format!("grammar shape actors: {e}"))?,
            orders: u8::try_from(order_limit)
                .map_err(|e| format!("grammar shape order_limit: {e}"))?,
            hire_limit: u16::try_from(hire_limit)
                .map_err(|e| format!("grammar shape hire_limit: {e}"))?,
        },
    })
}
impl State {
    pub fn slot(self) -> Slot {
        self.slot
    }

    pub fn allows(self, token: i64, plan: &GrammarPlan) -> bool {
        if self.shape != plan.shape {
            return false;
        }
        let Ok(token) = usize::try_from(token) else {
            return false;
        };
        if token >= self.slot.width() {
            return false;
        }
        let ready = self.units == self.shape.actors;
        match self.slot {
            Slot::UnitActor => token == if ready { 0 } else { usize::from(self.units) },
            Slot::UnitKind => {
                if ready {
                    token == 0
                } else {
                    (1..=18).contains(&token)
                }
            },
            Slot::UnitTarget => token == 0,
            Slot::UnitItem => match self.unit {
                UnitPhase::Transfer => (1..=12).contains(&token),
                UnitPhase::Plant => (1..=5).contains(&token),
                UnitPhase::None | UnitPhase::Ordinary => token == 0,
            },
            Slot::UnitQuantityPresent => self.unit == UnitPhase::Transfer || token == 0,
            Slot::UnitQuantityHigh => self.quantity_present || token == 0,
            Slot::UnitQuantity => {
                if self.quantity_present {
                    !self.quantity_high_zero || token != 0
                } else {
                    token == 0
                }
            },
            Slot::MarketKind => {
                if ready && self.unit == UnitPhase::None && self.orders < self.shape.orders {
                    token != 1 || self.shape.actors + u16::from(self.hires) < self.shape.hire_limit
                } else {
                    token == 0
                }
            },
            Slot::MarketItem => match self.market {
                MarketPhase::Seed => (1..=5).contains(&token),
                MarketPhase::Product => token == 1 || token == 9,
                MarketPhase::Animal => (10..=12).contains(&token),
                MarketPhase::Sell => (1..=9).contains(&token),
                MarketPhase::None | MarketPhase::Hire | MarketPhase::Other => token == 0,
            },
            Slot::MarketQuantityHigh | Slot::MarketQuantity => {
                self.market == MarketPhase::Seed || token == 0
            },
            Slot::Stop => {
                token
                    == usize::from(
                        ready && self.unit == UnitPhase::None && self.market == MarketPhase::None,
                    )
            },
        }
    }

    /// Check support before advancing; STOP is the only terminal state.
    pub fn advance(mut self, token: i64, plan: &GrammarPlan) -> Result<Option<State>, String> {
        if self.shape != plan.shape {
            return Err("grammar plan/state shape mismatch".into());
        }
        if !self.allows(token, plan) {
            let rule = if self.slot == Slot::MarketKind
                && token == MarketKind::Hire as i64
                && self.units == self.shape.actors
                && self.orders < self.shape.orders
            {
                "hire capacity"
            } else {
                match self.slot {
                    Slot::UnitItem => "unit item syntax: grammar support",
                    Slot::MarketItem => "market item syntax: grammar support",
                    _ => "grammar support",
                }
            };
            return Err(format!(
                "{rule}: frame {} slot {} value {token}",
                usize::from(self.units) + usize::from(self.orders),
                self.slot.name()
            ));
        }
        match self.slot {
            Slot::UnitKind => {
                self.unit = match UnitKind::try_from(token)? {
                    UnitKind::None => UnitPhase::None,
                    UnitKind::Pickup | UnitKind::Place => UnitPhase::Transfer,
                    UnitKind::Plant => UnitPhase::Plant,
                    _ => UnitPhase::Ordinary,
                }
            },
            Slot::UnitItem if self.unit == UnitPhase::Plant => self.unit = UnitPhase::Ordinary,
            Slot::UnitQuantityPresent => {
                self.quantity_present = token == 1;
                if self.unit == UnitPhase::Transfer && !self.quantity_present {
                    self.unit = UnitPhase::Ordinary;
                }
            },
            Slot::UnitQuantityHigh => self.quantity_high_zero = self.quantity_present && token == 0,
            Slot::UnitQuantity => {
                self.quantity_present = false;
                self.quantity_high_zero = false;
                if self.unit == UnitPhase::Transfer {
                    self.unit = UnitPhase::Ordinary;
                }
            },
            Slot::MarketKind => {
                self.market = match MarketKind::try_from(token)? {
                    MarketKind::None => MarketPhase::None,
                    MarketKind::Hire => MarketPhase::Hire,
                    MarketKind::BuySeed => MarketPhase::Seed,
                    MarketKind::BuyProduct => MarketPhase::Product,
                    MarketKind::BuyAnimal => MarketPhase::Animal,
                    MarketKind::Sell => MarketPhase::Sell,
                    MarketKind::BuyLand | MarketKind::Empty => MarketPhase::Other,
                }
            },
            Slot::MarketItem
                if matches!(
                    self.market,
                    MarketPhase::Product | MarketPhase::Animal | MarketPhase::Sell
                ) =>
            {
                self.market = MarketPhase::Seed
            },
            Slot::MarketQuantity if self.market == MarketPhase::Seed => {
                self.market = MarketPhase::Other
            },
            Slot::Stop => {
                if token == 1 {
                    return Ok(None);
                }
                // Counts are bounded by the validated shape and admitted support.
                return Ok(Some(Self {
                    units: self.units + u16::from(self.unit != UnitPhase::None),
                    orders: self.orders + u8::from(self.market != MarketPhase::None),
                    hires: self.hires + u8::from(self.market == MarketPhase::Hire),
                    ..plan.start()
                }));
            },
            _ => {},
        }
        self.slot = Slot::try_from(self.slot as usize + 1)?;
        Ok(Some(self))
    }

    pub fn write_mask(self, plan: &GrammarPlan, out: &mut [bool]) -> Result<(), String> {
        if self.shape != plan.shape {
            return Err("grammar plan/state shape mismatch".into());
        }
        if out.len() != self.slot.width() {
            return Err(format!(
                "grammar mask slot {} shape: expected {}, got {}",
                self.slot.name(),
                self.slot.width(),
                out.len()
            ));
        }
        for (token, value) in out.iter_mut().enumerate() {
            // Enumeration is bounded by the checked slot width (at most 241).
            *value = self.allows(i64::try_from(token).map_err(|e| e.to_string())?, plan);
        }
        Ok(())
    }
}
pub fn write_can_act(plan: &GrammarPlan, out: &mut [bool]) -> Result<(), String> {
    if out.len() != MAX_FRAMES {
        return Err(format!(
            "can_act shape: expected {MAX_FRAMES}, got {}",
            out.len()
        ));
    }
    let active = plan.shape.actors() + plan.shape.order_limit() + 1;
    for (frame, value) in out.iter_mut().enumerate() {
        *value = frame < active;
    }
    Ok(())
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GrammarTables {
    pub unit_kind: [bool; 20],
    pub unit_item: [[bool; 16]; 20],
    pub unit_quantity_present: [[bool; 2]; 20],
    pub unit_quantity_high: [[bool; 32]; 2],
    pub unit_quantity_low: [[[bool; 32]; 2]; 2],
    pub market_kind: [bool; 8],
    pub market_item: [[bool; 16]; 8],
    pub market_quantity: [[bool; 32]; 8],
}
fn prefix_mask<const N: usize>(plan: &GrammarPlan, prefix: &[i64]) -> Result<[bool; N], String> {
    let mut state = plan.start();
    for &token in prefix {
        state = state
            .advance(token, plan)?
            .ok_or("grammar table prefix ended at STOP")?;
    }
    let mut out = [false; N];
    state.write_mask(plan, &mut out)?;
    Ok(out)
}

fn first_token(mask: &[bool]) -> Result<i64, String> {
    let index = mask
        .iter()
        .position(|&bit| bit)
        .ok_or("grammar table has empty support")?;
    i64::try_from(index).map_err(|e| e.to_string())
}

/// Extract the table ABI from admitted prefixes, never from duplicated masks.
pub fn grammar_tables() -> Result<GrammarTables, String> {
    let p = plan(1, 1, DEFAULT_HIRE_LIMIT)?;
    let unit_kind = prefix_mask::<20>(&p, &[0])?;
    let mut unit_item = [prefix_mask::<16>(&p, &[0, 1, 0])?; 20];
    let mut unit_quantity_present = [prefix_mask::<2>(&p, &[0, 1, 0, 0])?; 20];
    for (kind, &admitted) in unit_kind.iter().enumerate() {
        if admitted {
            let token = i64::try_from(kind).map_err(|e| e.to_string())?;
            unit_item[kind] = prefix_mask(&p, &[0, token, 0])?;
            unit_quantity_present[kind] =
                prefix_mask(&p, &[0, token, 0, first_token(&unit_item[kind])?])?;
        }
    }
    let mut unit_quantity_high = [[false; 32]; 2];
    let mut unit_quantity_low = [[[false; 32]; 2]; 2];
    for (present, high) in unit_quantity_high.iter_mut().enumerate() {
        let present_token = i64::try_from(present).map_err(|e| e.to_string())?;
        *high = prefix_mask(&p, &[0, 6, 0, 1, present_token])?;
        for (high_zero, low) in unit_quantity_low[present].iter_mut().enumerate() {
            let digit = i64::from(high_zero == 0 && present != 0);
            *low = prefix_mask(&p, &[0, 6, 0, 1, present_token, digit])?;
        }
    }
    let mut prefix = vec![0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    prefix.extend([0; 7]);
    let market_kind = prefix_mask::<8>(&p, &prefix)?;
    let mut market_item = [[false; 16]; 8];
    let mut market_quantity = [[false; 32]; 8];
    for (kind, items) in market_item.iter_mut().enumerate() {
        let mut kind_prefix = prefix.clone();
        kind_prefix.push(i64::try_from(kind).map_err(|e| e.to_string())?);
        *items = prefix_mask(&p, &kind_prefix)?;
        let mut first_prefix = kind_prefix.clone();
        first_prefix.push(first_token(items)?);
        market_quantity[kind] = prefix_mask(&p, &first_prefix)?;
        // Check every admitted item and high digit, also in release builds.
        for (item, &allowed) in items.iter().enumerate() {
            if !allowed {
                continue;
            }
            let mut item_prefix = kind_prefix.clone();
            item_prefix.push(i64::try_from(item).map_err(|e| e.to_string())?);
            let high = prefix_mask::<32>(&p, &item_prefix)?;
            if high != market_quantity[kind] {
                return Err("market quantity item supports differ".into());
            }
            for (digit, &allowed_high) in high.iter().enumerate() {
                if !allowed_high {
                    continue;
                }
                let mut low_prefix = item_prefix.clone();
                low_prefix.push(i64::try_from(digit).map_err(|e| e.to_string())?);
                if prefix_mask::<32>(&p, &low_prefix)? != high {
                    return Err("market quantity digits differ".into());
                }
            }
        }
    }
    Ok(GrammarTables {
        unit_kind,
        unit_item,
        unit_quantity_present,
        unit_quantity_high,
        unit_quantity_low,
        market_kind,
        market_item,
        market_quantity,
    })
}

fn validate_program(plan: &GrammarPlan, tokens: &[i64], length: i64) -> Result<usize, String> {
    if tokens.len() != TOKENS_PER_SEAT {
        return Err(format!(
            "tokens shape: expected {TOKENS_PER_SEAT}, got {}",
            tokens.len()
        ));
    }
    let min = i64::from(plan.shape.actors) + 1;
    let max = min + i64::from(plan.shape.orders);
    if !(min..=max).contains(&length) {
        return Err(format!("length value {length} must be in {min}..={max}"));
    }
    // Validate every external value before calculating the active slice or
    // using token-dependent data. Padding is part of the transport contract.
    for (index, &value) in tokens.iter().enumerate() {
        let slot = Slot::try_from(index % SLOTS)?;
        let width = i64::try_from(slot.width()).map_err(|e| e.to_string())?;
        if !(0..width).contains(&value) {
            return Err(format!(
                "vocabulary frame {} slot {} value {value} outside 0..{width}",
                index / SLOTS,
                slot.name()
            ));
        }
    }
    let frames = usize::try_from(length).map_err(|e| format!("length: {e}"))?;
    let active_end = frames * SLOTS;
    for (index, &value) in tokens.iter().enumerate().skip(active_end) {
        if value != 0 {
            return Err(format!(
                "padding frame {} slot {} value {value} must be zero",
                index / SLOTS,
                Slot::try_from(index % SLOTS)?.name()
            ));
        }
    }
    let mut state = Some(plan.start());
    for (index, &value) in tokens[..active_end].iter().enumerate() {
        let current = state.ok_or_else(|| {
            format!(
                "frame {} slot {} value {value} follows terminal STOP",
                index / SLOTS,
                SLOT_NAMES[index % SLOTS]
            )
        })?;
        state = current.advance(value, plan)?;
    }
    if state.is_some() {
        return Err("program lacks distinct final STOP".into());
    }
    Ok(frames)
}

fn render_unit(frame: &[i64]) -> Result<serde_json::Value, String> {
    use serde_json::Value;
    let kind = UnitKind::try_from(frame[Slot::UnitKind as usize])?;
    let mut command = vec![Value::from(kind.as_str())];
    if matches!(kind, UnitKind::Pickup | UnitKind::Place | UnitKind::Plant) {
        command.push(Value::from(
            ActionItem::try_from(frame[Slot::UnitItem as usize])?.as_str(),
        ));
    }
    if frame[Slot::UnitQuantityPresent as usize] == 1 {
        command.push(Value::from(
            QUANTITY_BASE * frame[Slot::UnitQuantityHigh as usize]
                + frame[Slot::UnitQuantity as usize],
        ));
    }
    Ok(Value::Array(command))
}

fn render_market(frame: &[i64]) -> Result<serde_json::Value, String> {
    use serde_json::Value;
    let kind = MarketKind::try_from(frame[Slot::MarketKind as usize])?;
    let mut command = if kind == MarketKind::Empty {
        vec![]
    } else {
        vec![Value::from(kind.as_str())]
    };
    if matches!(
        kind,
        MarketKind::BuySeed | MarketKind::BuyProduct | MarketKind::BuyAnimal | MarketKind::Sell
    ) {
        command.push(Value::from(
            ActionItem::try_from(frame[Slot::MarketItem as usize])?.as_str(),
        ));
        command.push(Value::from(
            QUANTITY_BASE * frame[Slot::MarketQuantityHigh as usize]
                + frame[Slot::MarketQuantity as usize],
        ));
    }
    Ok(Value::Array(command))
}

/// Validate the entire i64 transport before rendering exact canonical syntax.
/// Inputs are borrowed only for this call; no token is narrowed or repaired.
pub fn decode(
    plan: &GrammarPlan,
    tokens: &[i64],
    length: i64,
) -> Result<serde_json::Value, String> {
    use serde_json::{Map, Value};
    let frames = validate_program(plan, tokens, length)?;
    let actors = plan.shape.actors();
    let farmer = render_unit(&tokens[..SLOTS])?;
    let hands = tokens[SLOTS..actors * SLOTS]
        .chunks_exact(SLOTS)
        .map(render_unit)
        .collect::<Result<Vec<_>, _>>()?;
    let market = tokens[actors * SLOTS..(frames - 1) * SLOTS]
        .chunks_exact(SLOTS)
        .map(render_market)
        .collect::<Result<Vec<_>, _>>()?;
    let mut action = Map::new();
    action.insert("farmer".into(), farmer);
    action.insert("hands".into(), Value::Array(hands));
    action.insert("market".into(), Value::Array(market));
    Ok(Value::Object(action))
}

fn encode_item(value: &serde_json::Value, field: &str, category: &str) -> Result<i64, String> {
    for token in 1..=12 {
        if value.as_str() == Some(ActionItem::try_from(token)?.as_str()) {
            return Ok(token);
        }
    }
    Err(format!("{category}: {field} unsupported item {value}"))
}

fn encode_quantity(value: &serde_json::Value, min: i64, field: &str) -> Result<i64, String> {
    value.as_i64().filter(|q| (min..=MAX_QUANTITY).contains(q))
        .ok_or_else(|| format!("quantity range: {field} value {value} must be a JSON integer in {min}..={MAX_QUANTITY}"))
}

fn encode_unit(command: &serde_json::Value, actor: usize, frame: &mut [i64]) -> Result<(), String> {
    let field = if actor == 0 {
        "farmer".to_owned()
    } else {
        format!("hands[{}]", actor - 1)
    };
    let args = command
        .as_array()
        .filter(|args| !args.is_empty())
        .ok_or_else(|| {
            format!("unsupported unit command arguments: {field} must be a nonempty array")
        })?;
    let mut found = None;
    for token in 1..=18 {
        let kind = UnitKind::try_from(token)?;
        if args[0].as_str() == Some(kind.as_str()) {
            found = Some(kind);
            break;
        }
    }
    let kind =
        found.ok_or_else(|| format!("unknown unit command: {field}[0] value {}", args[0]))?;
    let transfer = matches!(kind, UnitKind::Pickup | UnitKind::Place);
    let valid_arity = if transfer {
        matches!(args.len(), 2 | 3)
    } else if kind == UnitKind::Plant {
        args.len() == 2
    } else {
        args.len() == 1
    };
    if !valid_arity {
        return Err(format!(
            "unsupported unit command arguments: {field} {} arity {}",
            kind.as_str(),
            args.len()
        ));
    }
    frame[Slot::UnitActor as usize] =
        i64::try_from(actor).map_err(|e| format!("unit actor: {e}"))?;
    frame[Slot::UnitKind as usize] = kind as i64;
    if transfer || kind == UnitKind::Plant {
        frame[Slot::UnitItem as usize] =
            encode_item(&args[1], &format!("{field}[1]"), "unit item syntax")?;
    }
    if transfer && args.len() == 3 {
        let quantity = encode_quantity(&args[2], 1, &format!("{field}[2]"))?;
        frame[Slot::UnitQuantityPresent as usize] = 1;
        frame[Slot::UnitQuantityHigh as usize] = quantity / QUANTITY_BASE;
        frame[Slot::UnitQuantity as usize] = quantity % QUANTITY_BASE;
    }
    Ok(())
}

fn encode_market(
    command: &serde_json::Value,
    index: usize,
    frame: &mut [i64],
) -> Result<(), String> {
    let field = format!("market[{index}]");
    let args = command
        .as_array()
        .ok_or_else(|| format!("market command syntax: {field} must be an array"))?;
    if args.is_empty() {
        frame[Slot::MarketKind as usize] = MarketKind::Empty as i64;
        return Ok(());
    }
    let mut found = None;
    for token in 1..=6 {
        let kind = MarketKind::try_from(token)?;
        if args[0].as_str() == Some(kind.as_str()) {
            found = Some(kind);
            break;
        }
    }
    let kind =
        found.ok_or_else(|| format!("unknown market command: {field}[0] value {}", args[0]))?;
    let has_quantity = matches!(
        kind,
        MarketKind::BuySeed | MarketKind::BuyProduct | MarketKind::BuyAnimal | MarketKind::Sell
    );
    let arity = if has_quantity { 3 } else { 1 };
    if args.len() != arity {
        return Err(format!(
            "unsupported market arguments: {field} {} requires {arity} arguments, got {}",
            kind.as_str(),
            args.len()
        ));
    }
    frame[Slot::MarketKind as usize] = kind as i64;
    if has_quantity {
        frame[Slot::MarketItem as usize] =
            encode_item(&args[1], &format!("{field}[1]"), "market item syntax")?;
        let quantity = encode_quantity(&args[2], 0, &format!("{field}[2]"))?;
        frame[Slot::MarketQuantityHigh as usize] = quantity / QUANTITY_BASE;
        frame[Slot::MarketQuantity as usize] = quantity % QUANTITY_BASE;
    }
    Ok(())
}

/// Strict inverse of decode. No normalization, defaults or partial writes.
pub fn encode(
    plan: &GrammarPlan,
    action: &serde_json::Value,
    out: &mut [i64],
) -> Result<i64, String> {
    if out.len() != TOKENS_PER_SEAT {
        return Err(format!(
            "encode output shape: expected {TOKENS_PER_SEAT}, got {}",
            out.len()
        ));
    }
    let raw = action
        .as_object()
        .filter(|raw| {
            raw.len() == 3
                && ["farmer", "hands", "market"]
                    .iter()
                    .all(|key| raw.contains_key(*key))
        })
        .ok_or("action keys: expected exactly farmer, hands, market")?;
    let hands = raw["hands"]
        .as_array()
        .ok_or("hands syntax: hands must be an array")?;
    let actors = plan.shape.actors();
    if hands.len() != actors - 1 {
        let category = if hands.len() > actors - 1 {
            "extra hand commands"
        } else {
            "hand count"
        };
        return Err(format!(
            "{category}: hands expected {}, got {}",
            actors - 1,
            hands.len()
        ));
    }
    let market = raw["market"]
        .as_array()
        .ok_or("market syntax: market must be an array")?;
    if market.len() > plan.shape.order_limit() {
        return Err(format!(
            "market queue limit: market has {}, limit {}",
            market.len(),
            plan.shape.order_limit()
        ));
    }
    let mut tokens = vec![0; TOKENS_PER_SEAT];
    for (actor, command) in std::iter::once(&raw["farmer"])
        .chain(hands.iter())
        .enumerate()
    {
        encode_unit(
            command,
            actor,
            &mut tokens[actor * SLOTS..(actor + 1) * SLOTS],
        )?;
    }
    for (index, command) in market.iter().enumerate() {
        let start = (actors + index) * SLOTS;
        encode_market(command, index, &mut tokens[start..start + SLOTS])?;
    }
    let stop = actors + market.len();
    tokens[stop * SLOTS + Slot::Stop as usize] = 1;
    let length = i64::try_from(stop + 1).map_err(|e| format!("encode length: {e}"))?;
    validate_program(plan, &tokens, length)?;
    out.copy_from_slice(&tokens);
    Ok(length)
}

#[cfg(test)]
#[path = "grammar_tests.rs"]
mod tests;
