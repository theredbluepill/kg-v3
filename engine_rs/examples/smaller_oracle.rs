//! JSONL observation/direct-layer adapter for Smaller Market Shock.
use kaggriculture_engine::native_agents::smaller::SmallerController;
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};

fn main() {
    let mut controller = SmallerController::default();
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    for line in stdin.lock().lines() {
        let line = line.expect("stdin");
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(&line).expect("JSON request");
        let response = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            controller.oracle_request(&value)
        }));
        let response =
            response.unwrap_or_else(|_| json!({"id":value["id"],"error":"native oracle panic"}));
        serde_json::to_writer(&mut stdout, &response).expect("JSON response");
        writeln!(stdout).expect("newline");
        stdout.flush().expect("flush");
        if response.get("error").is_some() {
            break;
        }
    }
}
