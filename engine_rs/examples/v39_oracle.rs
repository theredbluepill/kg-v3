//! JSONL observation-only oracle adapter. No engine creation or transition.
use std::io::{self, BufRead, Write};

use kaggriculture_engine::native_agents::v39::{V39Controller, common, core, late, production};
use serde_json::{Value, json};

fn states(controller: &V39Controller) -> Value {
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
        "production_report":controller.production.report,
        "debug":controller.debug(),
    })
}

fn inject(controller: &mut V39Controller, state: &Value) {
    if let Some(value) = state.get("core") {
        controller.core.players = value.clone();
    }
    if let Some(value) = state.get("v219") {
        controller.production.v219 = value.clone();
    }
    if let Some(value) = state.get("v231") {
        controller.production.v231 = value.clone();
    }
    if let Some(value) = state.get("v233") {
        controller.production.v233 = value.clone();
    }
    if let Some(value) = state.get("r37") {
        controller.production.r37 = value.clone();
    }
    if let Some(value) = state.get("r44") {
        controller.production.r44 = value.clone();
    }
    if let Some(value) = state.get("horizons") {
        controller.production.horizons = value.clone();
    }
    if let Some(value) = state.get("input") {
        controller.late.input = value.clone();
    }
}

fn request(controller: &mut V39Controller, value: &Value) -> Value {
    if value["reset"] == true {
        *controller = V39Controller::default();
    }
    if let Some(state) = value.get("state") {
        inject(controller, state);
    }
    let obs = &value["observation"];
    let config = value.get("configuration").cloned().unwrap_or(json!({}));
    let mode = value["mode"].as_str().unwrap_or("act");
    let result = match mode {
        "reset" => {
            *controller = V39Controller::default();
            Value::Null
        }
        "act" => controller.act(obs, &config),
        "core" => controller.core.act(obs),
        "shop_terminal" => core::shop_terminal(obs, value["action"].clone()),
        "room_guard" => core::room_guard(obs, value["action"].clone()),
        "projected_shed" => core::Core::projected_shed(&value["action"], &common::View::new(obs)),
        "production_before" => {
            controller.production.before(obs);
            Value::Null
        }
        "production_after" => controller.production.after(
            obs,
            value["action"].clone(),
            &mut controller.core,
            value["valid_configuration"] != false,
        ),
        "late" => controller.late.after(
            obs,
            value["action"].clone(),
            &controller.core,
            &controller.production,
            &config,
        ),
        "market_price" => json!(production::market_price(
            value["item"].as_str().expect("item"),
            common::num(&value["inventory"]),
            value.get("params").filter(|v| !v.is_null())
        )),
        "parent_fert_qty" => json!(late::parent_fert_qty(
            obs,
            &value["action"],
            common::array(&value["planned"]),
            common::int(&value["offset"])
        )),
        "tomato_fertilizer_worthwhile" => {
            json!(late::tomato_fertilizer_worthwhile(obs, &value["action"]))
        }
        "labor_assignment" => {
            late::labor_assignment(obs, &value["action"], common::truth(&value["fertilizer"]))
                .unwrap_or(Value::Null)
        }
        "clone_act" => {
            let mut fork = controller.clone();
            let actual = controller.act(obs, &config);
            let cloned = fork.act(obs, &config);
            assert_eq!(actual, cloned, "cloned controller action");
            assert_eq!(states(controller), states(&fork), "cloned controller state");
            actual
        }
        "state" => Value::Null,
        _ => return json!({"id":value["id"],"error":format!("unknown mode {mode}")}),
    };
    json!({"id":value["id"],"result":result,"state":states(controller),"engine_transitions":0})
}

fn main() {
    let mut controller = V39Controller::default();
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
