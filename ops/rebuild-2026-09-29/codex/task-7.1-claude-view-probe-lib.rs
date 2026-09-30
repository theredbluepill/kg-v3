//! Disposable compile probe: v3-owned Game view for byte-exact controllers.

use num_bigint::BigInt;
use serde_json::Value;

pub use kaggriculture_engine::{Config, Farm, Inventory, PrivateState};
use kaggriculture_engine::{Market, StepSnapshot, Town};

pub mod native_agents;

#[derive(Clone, Debug)]
struct ControllerConfig {
    farm_hand_cost_mult: CostMult,
}
#[derive(Clone, Debug)]
struct CostMult(BigInt);

impl ControllerConfig {
    fn from_config(config: &Config) -> Result<Self, String> {
        let text = serde_json::to_value(&config.farm_hand_cost_mult).map_err(|e| e.to_string())?.to_string();
        let value: BigInt = text.parse().map_err(|e| format!("{e}"))?;
        Ok(Self { farm_hand_cost_mult: CostMult(value) })
    }
}

#[derive(Clone, Debug)]
pub struct Game {
    config: ControllerConfig,
    engine: kaggriculture_engine::Game,
    snapshot: StepSnapshot,
}

impl Game {
    pub fn new(config: Config, seed: i64, num_agents: usize) -> Result<Self, String> {
        let engine = kaggriculture_engine::Game::new(config.clone(), seed, num_agents)?;
        let snapshot = engine.snapshot();
        Ok(Self { config: ControllerConfig::from_config(&config)?, engine, snapshot })
    }
    pub fn step(&mut self, actions: &[Value]) -> Result<(), String> {
        self.engine.step(actions)?;
        self.snapshot = self.engine.snapshot();
        Ok(())
    }
    pub fn farms(&self) -> &[Farm] { &self.snapshot.public.farms }
    pub fn privates(&self) -> &[PrivateState] { &self.snapshot.privates }
    pub fn market(&self) -> &Market { &self.snapshot.public.market }
    pub fn town(&self) -> &Town { &self.snapshot.public.town }
    pub fn step_index(&self) -> usize { self.snapshot.public.step }
}

fn fib(index: usize) -> BigInt {
    let (mut a, mut b) = (BigInt::from(1), BigInt::from(1));
    for _ in 0..index {
        (a, b) = (b.clone(), a + b);
    }
    a
}
