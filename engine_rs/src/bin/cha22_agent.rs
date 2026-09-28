//! Stateful JSONL entry point for the cha22 native controller.
use kaggriculture_engine::native_agents::cha22::Cha22Controller;
use serde_json::{Value, json};
use std::io::{self, BufRead, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut agent = Cha22Controller::default();
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    for line in io::stdin().lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let input: Value = serde_json::from_str(&line)?;
        let observation = input.get("observation").unwrap_or(&input);
        let config = input.get("configuration").cloned().unwrap_or(json!({}));
        let action = agent.act(observation, &config);
        serde_json::to_writer(&mut stdout, &action)?;
        writeln!(stdout)?;
        stdout.flush()?;
    }
    Ok(())
}
