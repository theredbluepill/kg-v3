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
pub const SLOT_NAMES: [&str; SLOTS] = ["unit_actor", "unit_kind", "unit_target", "unit_item", "unit_quantity_present", "unit_quantity_high", "unit_quantity", "market_kind", "market_item", "market_quantity_high", "market_quantity", "stop"];
pub const SLOT_WIDTHS: [usize; SLOTS] = [UNIT_ACTOR_WIDTH, UNIT_KIND_WIDTH, UNIT_TARGET_WIDTH, UNIT_ITEM_WIDTH, UNIT_QUANTITY_PRESENT_WIDTH, UNIT_QUANTITY_HIGH_WIDTH, UNIT_QUANTITY_WIDTH, MARKET_KIND_WIDTH, MARKET_ITEM_WIDTH, MARKET_QUANTITY_HIGH_WIDTH, MARKET_QUANTITY_WIDTH, STOP_WIDTH];

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
    pub fn width(self) -> usize { SLOT_WIDTHS[self as usize] }
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

#[cfg(test)]
#[path = "grammar_tests.rs"]
mod tests;
