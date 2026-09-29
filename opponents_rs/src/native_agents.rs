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

pub(crate) use e776::E776Controller;
pub(crate) use ecobot::EcoBotController;
pub(crate) use r04::R04Controller;
pub(crate) use starter::StarterController;
