//! Stateful JSONL entry point for tetsutani V104's native controller.
use kaggriculture_engine::native_agents::tetsu342::Tetsu342Controller;
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut agent = Tetsu342Controller::default();
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    for line in io::stdin().lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let input: Value = serde_json::from_str(&line)?;
        let observation = input.get("observation").unwrap_or(&input);
        let config = input.get("configuration").cloned().unwrap_or(json!({}));
        serde_json::to_writer(&mut stdout, &agent.act(observation, &config))?;
        writeln!(stdout)?;
        stdout.flush()?;
    }
    Ok(())
}
