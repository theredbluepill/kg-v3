//! Fully native, stateful external controllers used by JA opponent rollouts.
//!
//! These ports are held to observation-to-action parity with the frozen Python
//! submissions.  They are not policy approximations and do not read hidden RNG state.

pub mod cha22;
pub mod e776;
pub mod ecobot;
pub mod evgen;
pub mod farm2945;
pub mod flex;
pub mod jsonl;
pub mod library;
pub mod r04;
pub mod shoprouter;
pub mod smaller;
pub mod metav4;
pub mod pipe16;
pub mod v56;
pub mod starter;
pub mod tetsu65;
pub mod tetsu342;
pub mod tetsutani;
pub mod v39;
pub mod v43;
pub mod v47;
pub mod v48;

use crate::Game;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum NativeAgentKind {
    R04,
    E776,
    Flex,
    Evgen,
    Ecobot,
    Starter,
    Tetsutani,
    ShopRouter,
    V39,
    V43,
    V47,
    V48,
    Farm2945,
    Tetsu65,
    Smaller,
    Tetsu342,
    Metav4,
    Pipe16,
    V56,
    Cha22,
}

#[derive(Clone, Debug)]
pub enum NativeAgent {
    R04(r04::R04Controller),
    E776(e776::E776Controller),
    Flex(flex::FlexController),
    Evgen(evgen::EvgenController),
    EcoBot(ecobot::EcoBotController),
    Starter(starter::StarterController),
    Tetsutani(tetsutani::TetsutaniController),
    ShopRouter(shoprouter::ShopRouterController),
    V39(Box<v39::V39Controller>),
    V43(Box<v43::V43Controller>),
    V47(Box<v47::V47Controller>),
    V48(Box<v48::V48Controller>),
    Farm2945(Box<farm2945::Farm2945Controller>),
    Tetsu65(Box<tetsu65::Tetsu65Controller>),
    Smaller(Box<smaller::SmallerController>),
    Tetsu342(Box<tetsu342::Tetsu342Controller>),
    Metav4(Box<metav4::Metav4Controller>),
    Pipe16(Box<pipe16::Pipe16Controller>),
    V56(Box<v56::V56Controller>),
    Cha22(Box<cha22::Cha22Controller>),
}

impl NativeAgent {
    pub fn new(kind: NativeAgentKind) -> Self {
        match kind {
            NativeAgentKind::Tetsu65 => Self::Tetsu65(Box::default()),
            NativeAgentKind::Tetsu342 => Self::Tetsu342(Box::default()),
            NativeAgentKind::Farm2945 => Self::Farm2945(Box::default()),
            NativeAgentKind::V48 => Self::V48(Box::default()),
            NativeAgentKind::V47 => Self::V47(Box::default()),
            NativeAgentKind::V43 => Self::V43(Box::default()),
            NativeAgentKind::V39 => Self::V39(Box::default()),
            NativeAgentKind::ShopRouter => {
                Self::ShopRouter(shoprouter::ShopRouterController::default())
            }
            NativeAgentKind::R04 => Self::R04(r04::R04Controller::default()),
            NativeAgentKind::E776 => Self::E776(e776::E776Controller::default()),
            NativeAgentKind::Flex => Self::Flex(flex::FlexController::default()),
            NativeAgentKind::Evgen => Self::Evgen(evgen::EvgenController::default()),
            NativeAgentKind::Ecobot => Self::EcoBot(ecobot::EcoBotController::default()),
            NativeAgentKind::Starter => Self::Starter(starter::StarterController::default()),
            NativeAgentKind::Tetsutani => {
                Self::Tetsutani(tetsutani::TetsutaniController::default())
            }
            NativeAgentKind::Smaller => Self::Smaller(Box::default()),
            NativeAgentKind::Metav4 => Self::Metav4(Box::default()),
            NativeAgentKind::Pipe16 => Self::Pipe16(Box::default()),
            NativeAgentKind::V56 => Self::V56(Box::default()),
            NativeAgentKind::Cha22 => Self::Cha22(Box::default()),
        }
    }

    pub fn action(&mut self, game: &Game, seat: usize) -> Result<Value, String> {
        match self {
            Self::Tetsu65(agent) => agent.action(game, seat),
            Self::Tetsu342(agent) => agent.action(game, seat),
            Self::Farm2945(agent) => agent.action(game, seat),
            Self::Smaller(agent) => agent.action(game, seat),
            Self::Metav4(agent) => agent.action(game, seat),
            Self::Pipe16(agent) => agent.action(game, seat),
            Self::V56(agent) => agent.action(game, seat),
            Self::Cha22(agent) => agent.action(game, seat),
            Self::V48(agent) => agent.action(game, seat),
            Self::V47(agent) => agent.action(game, seat),
            Self::V43(agent) => agent.action(game, seat),
            Self::V39(agent) => agent.action(game, seat),
            Self::ShopRouter(agent) => agent.action(game, seat),
            Self::R04(agent) => agent.action(game, seat),
            Self::E776(agent) => agent.action(game, seat),
            Self::Flex(agent) => agent.action(game, seat),
            Self::Evgen(agent) => agent.action(game, seat),
            Self::EcoBot(agent) => agent.action(game, seat),
            Self::Starter(agent) => agent.action(game, seat),
            Self::Tetsutani(agent) => agent.action(game, seat),
        }
    }

    pub fn debug(&self) -> Value {
        match self {
            Self::Tetsu65(agent) => agent.debug(),
            Self::Tetsu342(agent) => agent.debug(),
            Self::Farm2945(agent) => agent.debug(),
            Self::Smaller(agent) => agent.debug(),
            Self::Metav4(agent) => agent.debug(),
            Self::Pipe16(agent) => agent.debug(),
            Self::V56(agent) => agent.debug(),
            Self::Cha22(agent) => agent.debug(),
            Self::V48(agent) => agent.debug(),
            Self::V47(agent) => agent.debug(),
            Self::V43(agent) => agent.debug().clone(),
            Self::V39(agent) => agent.debug().clone(),
            Self::ShopRouter(agent) => agent.debug().clone(),
            Self::R04(agent) => agent.debug().clone(),
            Self::E776(agent) => agent.debug().clone(),
            Self::Flex(agent) => agent.debug().clone(),
            Self::Evgen(agent) => agent.debug().clone(),
            Self::EcoBot(agent) => agent.debug().clone(),
            Self::Starter(agent) => agent.debug().clone(),
            Self::Tetsutani(agent) => agent.debug().clone(),
        }
    }
}
