use crate::Game;
use crate::native_agents::{E776Controller, EcoBotController, R04Controller, StarterController};
use serde_json::Value;
use std::str::FromStr;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpponentKind {
    Starter,
    R04,
    Ecobot,
    E776,
}

impl OpponentKind {
    pub const ALL: [Self; 4] = [Self::Starter, Self::R04, Self::Ecobot, Self::E776];
    pub fn key(self) -> &'static str {
        match self {
            Self::Starter => "starter",
            Self::R04 => "r04",
            Self::Ecobot => "ecobot",
            Self::E776 => "e776",
        }
    }
}

impl FromStr for OpponentKind {
    type Err = String;
    fn from_str(key: &str) -> Result<Self, String> {
        Self::ALL
            .into_iter()
            .find(|kind| kind.key() == key)
            .ok_or_else(|| {
                format!("unknown opponent {key:?}; expected starter, r04, ecobot or e776")
            })
    }
}

#[derive(Clone)]
enum Controller {
    Starter(StarterController),
    R04(Box<R04Controller>),
    Ecobot(Box<EcoBotController>),
    E776(Box<E776Controller>),
}

impl Controller {
    fn new(kind: OpponentKind) -> Self {
        match kind {
            OpponentKind::Starter => Self::Starter(StarterController::default()),
            OpponentKind::R04 => Self::R04(Box::default()),
            OpponentKind::Ecobot => Self::Ecobot(Box::default()),
            OpponentKind::E776 => Self::E776(Box::default()),
        }
    }
    fn action(&mut self, game: &Game, seat: usize) -> Result<Value, String> {
        match self {
            Self::Starter(bot) => bot.action(game, seat),
            Self::R04(bot) => bot.action(game, seat),
            Self::Ecobot(bot) => bot.action(game, seat),
            Self::E776(bot) => bot.action(game, seat),
        }
    }
}

/// One controller per environment, seat and episode. Errors consume the turn:
/// calling a stateful controller twice is forbidden even if its first call failed.
#[derive(Clone)]
pub struct SeatController {
    kind: OpponentKind,
    seat: usize,
    episode: u64,
    next_step: usize,
    controller: Controller,
}

impl SeatController {
    pub fn new(kind: OpponentKind, seat: usize, game: &Game) -> Result<Self, String> {
        if seat >= game.farms().len() {
            return Err(format!("invalid seat {seat}"));
        }
        if game.step_index() != 0 {
            return Err("fresh controller requires step zero; replay the episode first".into());
        }
        Ok(Self {
            kind,
            seat,
            episode: game.episode,
            next_step: 0,
            controller: Controller::new(kind),
        })
    }
    pub fn reset(&mut self, game: &Game) -> Result<(), String> {
        if game.episode == self.episode {
            return Err("reset requires a new episode, not the current game".into());
        }
        *self = Self::new(self.kind, self.seat, game)?;
        Ok(())
    }
    pub fn action(&mut self, game: &Game, seat: usize) -> Result<Value, String> {
        if seat != self.seat {
            return Err(format!(
                "controller seat {} called for seat {seat}",
                self.seat
            ));
        }
        if game.episode != self.episode {
            return Err("controller belongs to another environment/episode; reset required".into());
        }
        if game.snapshot.done {
            return Err("controller cannot act after episode completion".into());
        }
        if game.step_index() != self.next_step {
            return Err(format!(
                "expected step {}, got {} (repeated or skipped step)",
                self.next_step,
                game.step_index()
            ));
        }
        self.next_step += 1;
        self.controller.action(game, seat)
    }
}
