//! JSONL V48 observation and direct-layer adapter. No engine transitions.
use std::io::{self, BufRead, Write};

use kaggriculture_engine::native_agents::v48::V48Controller;
use serde_json::{Value, json};

fn main() {
    let mut controller = V48Controller::default();
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
