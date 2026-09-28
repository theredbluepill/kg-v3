//! Stateful JSONL entry point for the native R04 controller.
use kaggriculture_engine::native_agents::{NativeAgentKind, jsonl};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    jsonl::run(NativeAgentKind::R04)
}
