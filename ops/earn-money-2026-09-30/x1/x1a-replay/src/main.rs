//! X1a: replay Kaggle episode JSONs through the v3 Rust engine and print per-seat
//! terminal bank + econ counters as one JSON line per episode.
use kaggriculture_engine::*;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

fn main() {
    let team = std::env::args().nth(1).expect("team sha256");
    for path in std::env::args().skip(2) {
        let text = std::fs::read_to_string(&path).expect("read");
        let d: Value = serde_json::from_str(&text).expect("json");
        let names: Vec<String> = d["info"]["TeamNames"].as_array().unwrap().iter()
            .map(|n| n.as_str().unwrap().to_string()).collect();
        let teacher: Vec<usize> = (0..2).filter(|&i| {
            format!("{:x}", Sha256::digest(names[i].as_bytes())) == team }).collect();
        let config: Config = serde_json::from_value(d["configuration"].clone()).expect("config");
        let seed = d["info"]["seed"].to_string();
        let mut game = Game::new_with_seed_decimal(config, &seed, 2).expect("game");
        let steps = d["steps"].as_array().unwrap();
        let mut errors = 0usize;
        let mut first_error = String::new();
        for t in 1..steps.len() {
            let acts: Vec<Value> = (0..2).map(|i| steps[t][i]["action"].clone()).collect();
            if let Err(e) = game.step(&acts) { errors += 1; if first_error.is_empty() { first_error = format!("t={t}: {e}"); } }
            if game.snapshot().done { break; }
        }
        let snap = game.snapshot();
        let recorded: Vec<f64> = d["rewards"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap_or(f64::NAN)).collect();
        let c = game.econ_counters().unwrap();
        let seats: Vec<Value> = (0..2).map(|p| json!({
            "bank": snap.rewards[p], "recorded": recorded[p],
            "starvation": c[p][ECON_STARVATION], "drought": c[p][ECON_DROUGHT],
            "ineffective": c[p][ECON_INEFFECTIVE], "commands": c[p][ECON_COMMANDS], "pass": c[p][ECON_PASS],
            "harvest": c[p][ECON_HARVEST], "water": c[p][ECON_WATER], "feed": c[p][ECON_FEED],
            "sell_units": c[p][ECON_SELL_UNITS], "sell_cash": c[p][ECON_SELL_CASH],
            "unsold_end": c[p][ECON_UNSOLD_END], "malformed": c[p][ECON_MALFORMED_UNIT_CMDS],
        })).collect();
        println!("{}", json!({"episode": d["info"]["EpisodeId"], "seed": seed, "teacher": teacher,
            "steps": snap.public.step, "done": snap.done, "step_errors": errors, "first_error": first_error,
            "seats": seats}));
    }
}
