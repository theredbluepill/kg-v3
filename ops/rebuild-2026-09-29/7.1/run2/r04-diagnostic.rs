// Read-only source diagnostic: compile this standalone against the built crate.
// It imports the same unmodified R04 file to access its existing debug method.
pub use kaggriculture_opponents::{Farm, Game, Inventory};
#[path = "/Users/poonszesen/kg-v3-t71/opponents_rs/src/native_agents/r04.rs"]
mod r04;
use serde_json::{Value, json};
use std::process::Command;

fn main() {
    let trace = std::env::args().nth(1).expect("trace path");
    let result = Command::new("gzip").args(["-dc", &trace]).output().unwrap();
    assert!(result.status.success());
    let rows: Vec<Value> = String::from_utf8(result.stdout).unwrap().lines().map(|s| serde_json::from_str(s).unwrap()).collect();
    let header = &rows[0];
    let mut game = Game::new(serde_json::from_value(header["configuration"].clone()).unwrap(), header["seed"].as_i64().unwrap(), 2).unwrap();
    let mut bot = r04::R04Controller::default();
    let mut report = Vec::new();
    for row in &rows[1..14] {
        let action = bot.action(&game, 1).unwrap();
        report.push(json!({"step":game.step_index(),"native_action":action,"python_action":row["actions"][1],"debug":bot.debug()}));
        game.step(row["actions"].as_array().unwrap()).unwrap();
    }
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}
