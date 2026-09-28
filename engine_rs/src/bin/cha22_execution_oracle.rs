//! Diagnostic reference: persistent Cha22 driven by explicit observation/action history.
//! Separate from batch teacher assignment, proposal caching and commit integration.
use kaggriculture_engine::native_agents::cha22::Cha22Controller;
use serde_json::Value;
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
        let response = agent.oracle_request(&input);
        serde_json::to_writer(&mut stdout, &response)?;
        writeln!(stdout)?;
        stdout.flush()?;
    }
    Ok(())
}
