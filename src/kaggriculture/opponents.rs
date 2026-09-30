//! The fixed-opponent controllers, or an uninhabited stand-in without them.
//!
//! With the default `fixed-opponents` feature this re-exports `opponents_rs`.
//! A `--no-default-features` build (the Kaggle submission, which must not ship
//! the imported controllers) gets types with no values: the registry is empty,
//! every opponent key is refused, and no hosted seat can exist, so the env's
//! fixed-opponent paths are statically unreachable.
#[cfg(feature = "fixed-opponents")]
pub use kaggriculture_opponents::{HostedSeat, OpponentKind};

#[cfg(not(feature = "fixed-opponents"))]
pub use disabled::{HostedSeat, OpponentKind};

#[cfg(not(feature = "fixed-opponents"))]
mod disabled {
    use kaggriculture_engine::{Config, StepSnapshot};
    use serde_json::Value;
    use std::str::FromStr;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum OpponentKind {}

    impl OpponentKind {
        pub const ALL: [Self; 0] = [];
        pub fn key(self) -> &'static str {
            match self {}
        }
    }

    impl FromStr for OpponentKind {
        type Err = String;
        fn from_str(key: &str) -> Result<Self, String> {
            Err(format!(
                "unknown opponent {key:?}: this build has no fixed opponents (feature fixed-opponents is off)"
            ))
        }
    }

    #[derive(Clone, Debug)]
    pub enum HostedSeat {}

    impl HostedSeat {
        pub fn new(
            kind: OpponentKind,
            _seat: usize,
            _config: &Config,
            _snapshot: StepSnapshot,
        ) -> Result<Self, String> {
            match kind {}
        }
        pub fn seat(&self) -> usize {
            match *self {}
        }
        pub fn action(&mut self, _snapshot: StepSnapshot) -> Result<Value, String> {
            match *self {}
        }
    }
}
