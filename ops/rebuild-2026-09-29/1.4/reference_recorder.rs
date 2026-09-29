//! Runs only as an example in a byte-verified reference export, never the root kernel.
use kaggriculture_engine::training::TrainingBatch;
use serde::Deserialize;
use serde_json::{Value, json};
use std::error::Error;
use std::io::{self, BufRead, Write};

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum Request {
    Start { seed: i64, reward: Value },
    Step { tokens: Vec<i64>, lengths: [i64; 2] },
}

fn view(batch: &TrainingBatch) -> Result<Value, String> {
    let snapshot = batch.snapshot(0, false)?.ok_or("missing live snapshot")?;
    let farms = &snapshot.public.farms;
    let (next, seeds) = batch.seeds();
    let output = &batch.output;
    let terminal = match batch.terminal_metrics(0)? {
        None => Value::Null,
        Some(metrics) => json!({
            "banks_bits": [metrics["bank_0"].to_bits(), metrics["bank_1"].to_bits()],
            "margin_bits": metrics["margin_0"].to_bits(),
            "episode_steps": metrics["episode_steps"],
        }),
    };
    let goose = farms[0]
        .tiles
        .iter()
        .flatten()
        .any(|tile| tile.get("animal").and_then(Value::as_str) == Some("GOOSE"));
    Ok(json!({
        "public": {"step": snapshot.public.step,
                   "farms": [{"hands": farms[0].hands}, {"hands": farms[1].hands}]},
        "goose_placed": goose,
        "rewards_bits": output.rewards.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "dones": output.dones,
        "banks_before_bits": output.previous_banks.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "banks_after_bits": output.current_banks.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "econ_before": output.previous_econ,
        "econ_after": output.current_econ,
        "terminal": terminal,
        "next_seed": next,
        "seeds": seeds,
    }))
}

fn main() -> Result<(), Box<dyn Error>> {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    let mut current: Option<TrainingBatch> = None;
    for line in stdin.lock().lines() {
        let request: Request = serde_json::from_str(&line?)?;
        match request {
            Request::Start { seed, reward } => {
                // Drop the preceding live environment before creating the next one.
                drop(current.take());
                current = Some(TrainingBatch::from_json(
                    1,
                    "{}",
                    &seed.to_string(),
                    "1",
                    1,
                    2,
                    &reward.to_string(),
                )?);
            },
            Request::Step { tokens, lengths } => {
                current
                    .as_mut()
                    .ok_or("step before start")?
                    .step(&tokens, &lengths)?;
            },
        }
        let result = view(current.as_ref().ok_or("missing batch")?)?;
        serde_json::to_writer(&mut stdout, &result)?;
        stdout.write_all(b"\n")?;
        stdout.flush()?;
    }
    Ok(())
}
