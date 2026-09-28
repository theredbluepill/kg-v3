//! JSONL V43 observation and direct-contract adapter. No engine transitions.
use std::io::{self, BufRead, Write};

use kaggriculture_engine::native_agents::v43::{V43Controller, common};
use serde_json::{Value, json};

fn states(controller: &V43Controller) -> Value {
    let mut debug = controller.debug().clone();
    if let Some(stages) = controller.contracts.stages.as_object() {
        for (key, value) in stages {
            debug["stages"][key] = value.clone();
        }
    }
    json!({
        "core": controller.core.players,
        "diagnostics": controller.core.diagnostics,
        "v219":controller.production.v219,
        "v231":controller.production.v231,
        "v233":controller.production.v233,
        "r37":controller.production.r37,
        "r44":controller.production.r44,
        "horizons":controller.production.horizons,
        "input":controller.late.input,
        "r124":controller.contracts.states124,
        "r127":controller.contracts.states127,
        "r128":controller.contracts.states128,
        "r148":controller.contracts.pending148,
        "contract_report":controller.contracts.report,
        "debug":debug,
    })
}

fn inject(controller: &mut V43Controller, state: &Value) {
    if let Some(value) = state.get("core") {
        controller.core.players = value.clone();
    }
    if let Some(value) = state.get("r124") {
        controller.contracts.states124 = value.clone();
    }
    if let Some(value) = state.get("r127") {
        controller.contracts.states127 = value.clone();
    }
    if let Some(value) = state.get("r128") {
        controller.contracts.states128 = value.clone();
    }
    if let Some(value) = state.get("r148") {
        controller.contracts.pending148 = value.clone();
    }
}

fn request(controller: &mut V43Controller, value: &Value) -> Value {
    if value["reset"] == true {
        *controller = V43Controller::default();
    }
    if let Some(state) = value.get("state") {
        inject(controller, state);
    }
    let obs = &value["observation"];
    let config = value.get("configuration").cloned().unwrap_or(json!({}));
    let action = value.get("action").cloned().unwrap_or(Value::Null);
    let mut state = value.get("contract_state").cloned().unwrap_or(json!({}));
    let mode = value["mode"].as_str().unwrap_or("act");
    if mode == "overflow" {
        for name in [
            "overflow_turns",
            "overflow_units_reclaimed",
            "overflow_quote_exposure",
        ] {
            controller.contracts.report[name] = json!(0);
        }
    }
    let result = match mode {
        "act" => Ok(controller.act(obs, &config)),
        "core" => Ok(controller.core.act(obs)),
        "tomato_request" => Ok(controller.production.tomato_request(
            obs,
            action,
            &mut state,
            &value["native_state"],
        )),
        "sheep_request" => {
            Ok(controller
                .production
                .sheep_request(obs, action, &mut state, &value["native_state"]))
        }
        "seed_budget" => {
            controller
                .contracts
                .seed_budget(obs, action, common::int(&value["reserve"]))
        }
        "atomic" => controller.contracts.atomic(obs, action, &mut state),
        "last_hour" => controller.contracts.last_hour(obs, action),
        "priority" => controller
            .contracts
            .priority(obs, action, &controller.core, &mut state),
        "service" => controller
            .contracts
            .service(obs, action, &controller.core, &mut state),
        "credit_supply" => {
            controller
                .contracts
                .credit_supply(obs, action, &controller.core, &mut state)
        }
        "overflow" => controller.contracts.overflow(obs, action),
        "clone_act" => {
            let mut fork = controller.clone();
            let actual = controller.act(obs, &config);
            let cloned = fork.act(obs, &config);
            assert_eq!(actual, cloned, "cloned controller action");
            assert_eq!(states(controller), states(&fork), "cloned controller state");
            Ok(actual)
        }
        _ => return json!({"id":value["id"],"error":format!("unknown mode {mode}")}),
    };
    match result {
        Ok(result) => {
            let mut snapshot = states(controller);
            snapshot["contract_state"] = state;
            json!({"id":value["id"],"result":result,"state":snapshot,"engine_transitions":0})
        }
        Err(error) => json!({"id":value["id"],"error":error}),
    }
}

fn main() {
    let mut controller = V43Controller::default();
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    for line in stdin.lock().lines() {
        let line = line.expect("stdin");
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(&line).expect("JSON request");
        let response = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            request(&mut controller, &value)
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
