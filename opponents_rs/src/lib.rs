//! Byte-exact evaluation controllers behind a v3-owned snapshot view.
//!
//! A refresh clones the full snapshot: this is an evaluation interface, not a
//! training hot path. Both private seats are present; perturbation tests audit
//! that imported controllers only consume the acting seat's private state.

use kaggriculture_engine::{Market, StepMetrics, StepSnapshot, Town};
use num_bigint::BigInt;
use serde_json::Value;
use std::sync::atomic::{AtomicU64, Ordering};

pub use kaggriculture_engine::{Config, Farm, Inventory, PrivateState};
pub use registry::{OpponentKind, SeatController};
pub use runner::{MatchResult, SeatStep, StepRecord, play_match};

mod native_agents;
mod registry;
mod runner;

// The extra module prevents child controller modules from reaching an engine
// reference through a parent-private field. It exposes no seed/RNG/counter API.
mod engine_owner {
    use kaggriculture_engine::{Game, StepMetrics, StepSnapshot};
    use serde_json::Value;

    #[derive(Clone)]
    pub(super) struct EngineOwner(Game);

    impl EngineOwner {
        pub(super) fn new(engine: Game) -> Self {
            Self(engine)
        }
        pub(super) fn snapshot(&self) -> StepSnapshot {
            self.0.snapshot()
        }
        pub(super) fn step(&mut self, actions: &[Value]) -> Result<StepMetrics, String> {
            self.0.step_with_market_metrics(actions)
        }
    }
}

// Required only by the pinned Starter hire helper and its inline tests.
#[allow(dead_code)]
#[derive(Clone, Debug)]
struct ControllerConfig {
    farm_hand_cost_mult: CostMult,
}

// Preserve Starter's .0 cost access; live Starter actions do not hire.
#[allow(dead_code)]
#[derive(Clone, Debug)]
struct CostMult(BigInt);

impl ControllerConfig {
    fn from_config(config: &Config) -> Result<Self, String> {
        let value = serde_json::to_value(&config.farm_hand_cost_mult)
            .map_err(|error| format!("serialize farmHandCostMult: {error}"))?;
        let value = value
            .to_string()
            .parse::<BigInt>()
            .map_err(|error| format!("farmHandCostMult must serialize as an integer: {error}"))?;
        Ok(Self {
            farm_hand_cost_mult: CostMult(value),
        })
    }
}

static NEXT_EPISODE: AtomicU64 = AtomicU64::new(1);

/// Controller input plus engine ownership. Clones preserve episode identity for
/// diagnostic forks; separately constructed views always have a new identity.
#[derive(Clone)]
pub struct Game {
    // Required by Starter's byte-exact inline tests/historical hire helper.
    #[allow(dead_code)]
    config: ControllerConfig,
    // The public engine configuration, which the v2-imported controllers receive
    // serialized as their `configuration` argument (Cha22's closure).
    configuration: Config,
    engine: engine_owner::EngineOwner,
    snapshot: StepSnapshot,
    episode: u64,
}

// Never delegate formatting to the owned engine: its derived Debug contains
// private seed and counter fields that controller modules must not inspect.
impl std::fmt::Debug for Game {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Game")
            .field("snapshot", &self.snapshot)
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl Game {
    pub fn new(config: Config, seed: i64, num_agents: usize) -> Result<Self, String> {
        let engine = kaggriculture_engine::Game::new(config.clone(), seed, num_agents)?;
        Self::from_engine(engine, &config)
    }

    /// Adopt an existing engine. The caller must supply the configuration used
    /// to construct it: the frozen engine exposes no configuration getter.
    pub fn from_engine(
        engine: kaggriculture_engine::Game,
        config: &Config,
    ) -> Result<Self, String> {
        let engine = engine_owner::EngineOwner::new(engine);
        Ok(Self {
            config: ControllerConfig::from_config(config)?,
            configuration: config.clone(),
            snapshot: engine.snapshot(),
            engine,
            episode: NEXT_EPISODE.fetch_add(1, Ordering::Relaxed),
        })
    }

    pub fn step(&mut self, actions: &[Value]) -> Result<(), String> {
        self.step_with_metrics(actions).map(|_| ())
    }

    fn step_with_metrics(&mut self, actions: &[Value]) -> Result<StepMetrics, String> {
        let metrics = self.engine.step(actions)?;
        self.snapshot = self.engine.snapshot();
        Ok(metrics)
    }

    pub fn snapshot(&self) -> &StepSnapshot {
        &self.snapshot
    }
    pub fn farms(&self) -> &[Farm] {
        &self.snapshot.public.farms
    }
    pub fn privates(&self) -> &[PrivateState] {
        &self.snapshot.privates
    }
    pub fn market(&self) -> &Market {
        &self.snapshot.public.market
    }
    pub fn town(&self) -> &Town {
        &self.snapshot.public.town
    }
    pub fn step_index(&self) -> usize {
        self.snapshot.public.step
    }
    /// The configuration supplied at construction (see `from_engine`).
    pub fn configuration(&self) -> &Config {
        &self.configuration
    }
}

/// Authored port of the engine's private Fibonacci hire-cost helper.
/// Index zero and one both return 1; BigInt preserves unbounded costs.
#[allow(dead_code)] // The byte-exact hire helper is tested but not called by the runner.
fn fib(index: usize) -> BigInt {
    let (mut a, mut b) = (BigInt::from(1), BigInt::from(1));
    for _ in 0..index {
        (a, b) = (b.clone(), a + b);
    }
    a
}

#[cfg(test)]
mod view_tests;
