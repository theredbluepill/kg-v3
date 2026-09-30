// Pinned source formatting is part of byte custody; never rustfmt these modules.
// Historical helper/debug APIs are retained but not exposed by the registry.
#[allow(dead_code)]
#[rustfmt::skip]
mod starter;
// R04 already carries its own dead-code allowance in the pinned file.
#[rustfmt::skip]
mod r04;
// Historical helper/debug APIs are retained but not exposed by the registry.
#[allow(dead_code)]
#[rustfmt::skip]
mod ecobot;
// Historical helper/debug APIs are retained but not exposed by the registry.
#[allow(dead_code)]
#[rustfmt::skip]
mod e776;

// Cha22's full ig_agent and its v2 dependency closure (V43/V47/V48, Farm2945,
// Metav4, Pipe16), imported from kaggriculture-v2 commit 30a3ac47. Only the
// Game-view accessor lines differ (OPPONENT_MANIFEST.json `adapted`). The
// dependency controllers are not registered opponents; only Cha22 is.
#[allow(dead_code, clippy::all)]
#[rustfmt::skip]
mod cha22;
#[allow(dead_code, clippy::all)]
#[rustfmt::skip]
mod farm2945;
#[allow(dead_code, clippy::all)]
#[rustfmt::skip]
mod metav4;
#[allow(dead_code, clippy::all)]
#[rustfmt::skip]
mod pipe16;
#[allow(dead_code, clippy::all)]
#[rustfmt::skip]
mod v43;
#[allow(dead_code, clippy::all)]
#[rustfmt::skip]
mod v47;
#[allow(dead_code, clippy::all)]
#[rustfmt::skip]
mod v48;

pub(crate) use cha22::Cha22Controller;
pub(crate) use e776::E776Controller;
pub(crate) use ecobot::EcoBotController;
pub(crate) use r04::R04Controller;
pub(crate) use starter::StarterController;
