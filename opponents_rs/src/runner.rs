use crate::{Config, Game, OpponentKind, SeatController};
use serde_json::Value;

#[derive(Clone, Debug)]
pub struct SeatStep {
    /// Exact official JSON supplied to the kernel; None means controller error.
    pub action: Option<Value>,
    /// None if no engine call; otherwise whole joint-action transaction acceptance.
    /// Accepted commands/orders can still have zero effect; metrics report this.
    pub engine_accepted: Option<bool>,
    pub controller_error: Option<String>,
}

impl SeatStep {
    fn new(action: Result<Value, String>) -> Self {
        let (action, controller_error) = match action {
            Ok(value) => (Some(value), None),
            Err(error) => (None, Some(error)),
        };
        Self {
            action,
            controller_error,
            engine_accepted: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct StepRecord {
    pub step: usize,
    pub seats: [SeatStep; 2],
    pub engine_error: Option<String>,
    /// Joint execution counters; the frozen engine keeps per-seat metrics private.
    pub market_metrics: Option<kaggriculture_engine::MarketStepMetrics>,
}

#[derive(Clone, Debug)]
pub struct MatchResult {
    pub seed: i64,
    pub kinds: [OpponentKind; 2],
    pub steps: Vec<StepRecord>,
    pub completed: bool,
    pub final_banks: [f64; 2],
    /// Only terminal raw banks decide the winner; None also covers incomplete games.
    pub winner: Option<usize>,
}

/// Default configuration only. At most 719 kernel transitions; no normalization,
/// grammar truncation, silent PASS substitution or per-order reinterpretation.
pub fn play_match(
    config: Config,
    seed: i64,
    kinds: [OpponentKind; 2],
) -> Result<MatchResult, String> {
    if serde_json::to_value(&config).map_err(|e| e.to_string())?
        != serde_json::to_value(Config::default()).map_err(|e| e.to_string())?
    {
        return Err("evaluation opponents support default configuration only".into());
    }
    let mut game = Game::new(config, seed, 2)?;
    let mut controllers = [
        SeatController::new(kinds[0], 0, &game)?,
        SeatController::new(kinds[1], 1, &game)?,
    ];
    let mut steps = Vec::with_capacity(719);
    while !game.snapshot().done && steps.len() < 719 {
        let mut record = StepRecord {
            step: game.step_index(),
            seats: std::array::from_fn(|seat| SeatStep::new(controllers[seat].action(&game, seat))),
            engine_error: None,
            market_metrics: None,
        };
        let [Some(a), Some(b)] = record.seats.each_ref().map(|seat| seat.action.as_ref()) else {
            steps.push(record);
            break;
        };
        match game.step_with_metrics(&[a.clone(), b.clone()]) {
            Ok(metrics) => {
                record.market_metrics = Some(metrics.market);
                for result in &mut record.seats {
                    result.engine_accepted = Some(true);
                }
            },
            Err(error) => {
                for seat in &mut record.seats {
                    seat.engine_accepted = Some(false);
                }
                record.engine_error = Some(error);
                steps.push(record);
                break;
            },
        }
        steps.push(record);
    }
    let completed = game.snapshot().done;
    let final_banks = [game.farms()[0].money, game.farms()[1].money];
    let winner = if !completed || final_banks[0] == final_banks[1] {
        None
    } else {
        Some(usize::from(final_banks[1] > final_banks[0]))
    };
    Ok(MatchResult {
        seed,
        kinds,
        steps,
        completed,
        final_banks,
        winner,
    })
}
