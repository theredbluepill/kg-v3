//! A scripted seat inside a host-owned stepping loop (the training environment).
//!
//! The host owns and steps the engine. Before each of the bot's turns it passes
//! the current full snapshot; the controller keeps its own scripted state across
//! turns, exactly as in `play_match`. One `HostedSeat` serves one environment,
//! seat and episode: construct a new one at every reset. The snapshot carries
//! both private states, as the evaluation view does; the rival-perturbation
//! tests bound what the imported controllers read. The bot's identity and state
//! stay in the host's bookkeeping and never reach learned inputs.
use crate::{Config, Game, OpponentKind, SeatController};
use kaggriculture_engine::StepSnapshot;
use serde_json::Value;

#[derive(Clone)]
pub struct HostedSeat {
    view: Game,
    controller: SeatController,
}

impl std::fmt::Debug for HostedSeat {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("HostedSeat")
            .field("kind", &self.controller.kind())
            .field("seat", &self.controller.seat())
            .field("step", &self.view.step_index())
            .finish_non_exhaustive()
    }
}

impl HostedSeat {
    /// Start a controller for `seat` at a fresh episode's step-zero snapshot.
    /// `config` must be the configuration the host's engine was built with.
    pub fn new(
        kind: OpponentKind,
        seat: usize,
        config: &Config,
        snapshot: StepSnapshot,
    ) -> Result<Self, String> {
        let view = Game::hosted(config, snapshot)?;
        let controller = SeatController::new(kind, seat, &view)?;
        Ok(Self { view, controller })
    }

    pub fn kind(&self) -> OpponentKind {
        self.controller.kind()
    }

    pub fn seat(&self) -> usize {
        self.controller.seat()
    }

    /// The bot's official JSON action for the host's current state. Call once
    /// per host transition with that transition's pre-step snapshot; the
    /// controller rejects a repeated or skipped step and a completed game. An
    /// error consumes the turn, as for `SeatController::action`.
    pub fn action(&mut self, snapshot: StepSnapshot) -> Result<Value, String> {
        self.view.refresh(snapshot);
        let seat = self.controller.seat();
        self.controller.action(&self.view, seat)
    }
}
