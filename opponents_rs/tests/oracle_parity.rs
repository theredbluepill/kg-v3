use serde_json::json;

#[test]
fn comparator_preserves_numbers_array_order_and_state_key_order() {
    assert!(difference(&json!(1), &json!(1.0), "action", false).is_some());
    assert!(difference(&json!([1, 2]), &json!([2, 1]), "action", false).is_some());
    let a = json!({"a": 1, "b": 2});
    let b = json!({"b": 2, "a": 1});
    assert!(difference(&a, &b, "action", false).is_none());
    assert!(difference(&a, &b, "state", true).is_some());
    assert!(difference(&json!({"a":1}), &json!({"a":1,"b":2}), "action", false).is_some());
}

use kaggriculture_engine::TraceHeader;
use kaggriculture_opponents::{Game, OpponentKind, SeatController};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Command;

const ENGINE_SHA: &str = "bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e";

/// Actions compare canonical content (including integer/float representation and
/// array order). Engine states additionally compare map insertion order.
fn difference(actual: &Value, expected: &Value, path: &str, order: bool) -> Option<String> {
    match (actual, expected) {
        (Value::Object(a), Value::Object(b)) => {
            for (key, value) in b {
                match a.get(key) {
                    None => return Some(format!("{path}.{key}: missing in native")),
                    Some(other) => {
                        if let Some(diff) =
                            difference(other, value, &format!("{path}.{key}"), order)
                        {
                            return Some(diff);
                        }
                    },
                }
            }
            if let Some(key) = a.keys().find(|key| !b.contains_key(*key)) {
                return Some(format!("{path}.{key}: extra in native"));
            }
            (order && !a.keys().eq(b.keys())).then(|| format!("{path}: map key order"))
        },
        (Value::Array(a), Value::Array(b)) => {
            for (index, (a, b)) in a.iter().zip(b).enumerate() {
                if let Some(diff) = difference(a, b, &format!("{path}[{index}]"), order) {
                    return Some(diff);
                }
            }
            (a.len() != b.len())
                .then(|| format!("{path}: length native={} python={}", a.len(), b.len()))
        },
        _ => (actual != expected).then(|| format!("{path}: native={actual} python={expected}")),
    }
}

#[derive(Debug)]
struct ReplayReport {
    compared_actions: [usize; 2],
    matched_actions: [usize; 2],
    transitions: usize,
    mismatch: Option<String>,
}

impl ReplayReport {
    fn new() -> Self {
        Self {
            compared_actions: [0; 2],
            matched_actions: [0; 2],
            transitions: 0,
            mismatch: None,
        }
    }
    fn check(
        &mut self,
        actual: &Value,
        expected: &Value,
        label: &str,
        order: bool,
    ) -> Result<(), String> {
        difference(actual, expected, label, order).map_or(Ok(()), Err)
    }
}

fn replay(rows: &[Value]) -> ReplayReport {
    let mut report = ReplayReport::new();
    report.mismatch = replay_inner(rows, &mut report).err();
    report
}

fn replay_inner(rows: &[Value], report: &mut ReplayReport) -> Result<(), String> {
    let raw = &rows[0];
    if raw["type"] != "header"
        || raw["format"] != "kaggriculture-re-parity-v1"
        || raw["source"]["module_version"] != "1.32.7"
        || raw["source"]["engine_sha256"] != ENGINE_SHA
    {
        return Err("oracle header identity mismatch".into());
    }
    let header: TraceHeader = serde_json::from_value(raw.clone()).map_err(|e| e.to_string())?;
    if header.transitions != 719 || rows.len() != 720 {
        return Err("default oracle must contain 719 transitions".into());
    }
    let engine = kaggriculture_engine::Game::new_with_seed_decimal(
        header.configuration.clone(),
        &header.seed.to_string(),
        2,
    )?;
    let mut game = Game::from_engine(engine, &header.configuration)?;
    let policies = raw["source"]["policies"]
        .as_array()
        .ok_or("missing policies")?;
    if policies.len() != 2 {
        return Err("expected two policies".into());
    }
    let kinds: Vec<OpponentKind> = policies
        .iter()
        .map(|v| {
            let key = v.as_str().ok_or("policy must be a string")?;
            key.strip_prefix("builtin:")
                .or_else(|| key.strip_prefix("sibling:"))
                .ok_or("unknown policy namespace")?
                .parse()
        })
        .collect::<Result<_, String>>()?;
    let mut seats = [
        SeatController::new(kinds[0], 0, &game)?,
        SeatController::new(kinds[1], 1, &game)?,
    ];
    report.check(
        &serde_json::to_value(&game.snapshot().public).unwrap(),
        &raw["initial"]["public"],
        "initial.public",
        true,
    )?;
    report.check(
        &serde_json::to_value(&game.snapshot().privates).unwrap(),
        &raw["initial"]["privates"],
        "initial.privates",
        true,
    )?;
    for row in &rows[1..] {
        let step = game.step_index();
        if row["type"] != "transition" || row["from_step"].as_u64() != Some(step as u64) {
            return Err(format!("step {step}: unexpected record type or step"));
        }
        let recorded = row["actions"].as_array().ok_or("missing actions")?;
        if recorded.len() != 2 {
            return Err(format!("step {step}: expected two actions"));
        }
        for seat in 0..2 {
            let action = seats[seat].action(&game, seat)?;
            report.compared_actions[seat] += 1;
            report.check(
                &action,
                &recorded[seat],
                &format!("step {step} seat {seat} {} action", kinds[seat].key()),
                false,
            )?;
            report.matched_actions[seat] += 1;
        }
        // State reconstruction uses recorded Python actions, never native output.
        game.step(recorded)?;
        report.transitions += 1;
        let state = game.snapshot();
        for (actual, expected, label, order) in [
            (
                serde_json::to_value(&state.public).unwrap(),
                &row["expected"],
                "public",
                true,
            ),
            (
                serde_json::to_value(&state.privates).unwrap(),
                &row["privates"],
                "privates",
                true,
            ),
            (json!(state.statuses), &row["statuses"], "statuses", false),
        ] {
            report.check(&actual, expected, &format!("step {step} {label}"), order)?;
        }
        // Rewards are typed f64 in the engine, as in its existing replay comparator.
        let rewards: Vec<f64> =
            serde_json::from_value(row["rewards"].clone()).map_err(|e| e.to_string())?;
        if state.rewards != rewards {
            return Err(format!(
                "step {step}: rewards native={:?} python={rewards:?}",
                state.rewards
            ));
        }
        if state.done != (step == 718) {
            return Err(format!("step {step}: completion mismatch"));
        }
    }
    report.check(
        &json!(
            game.farms()
                .iter()
                .map(|farm| farm.money)
                .collect::<Vec<_>>()
        ),
        &json!(header.terminal_banks),
        "terminal_banks",
        false,
    )?;
    Ok(())
}

fn directory() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/oracle")
}

fn entries() -> Vec<Value> {
    let manifest: Value = serde_json::from_slice(
        &std::fs::read(directory().join("MANIFEST.json"))
            .expect("generate the original-Python opponent oracle first"),
    )
    .unwrap();
    let entries = manifest["traces"].as_array().unwrap().clone();
    assert!(!entries.is_empty(), "oracle corpus must not be empty");
    entries
}

fn read_rows(entry: &Value) -> Vec<Value> {
    let path = directory().join(entry["path"].as_str().unwrap());
    let output = Command::new("gzip")
        .args(["-dc"])
        .arg(&path)
        .output()
        .expect("gzip required");
    assert!(output.status.success(), "{}", path.display());
    String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[test]
fn original_python_oracles_match_native_actions_and_states() {
    let mut results = Vec::new();
    let mut failures = Vec::new();
    for entry in entries() {
        // Kaggle's simulation image runs CPython 3.11; 3.12's compensated float
        // sum() changes R04's decisions, so only 3.11 traces are oracles.
        let runtime = entry["python_runtime"].as_str().unwrap_or_default();
        assert!(
            runtime.starts_with("3.11."),
            "{}: oracle runtime {runtime:?} is not CPython 3.11",
            entry["path"]
        );
        let report = replay(&read_rows(&entry));
        eprintln!("{}: {report:?}", entry["path"]);
        if let Some(mismatch) = &report.mismatch {
            failures.push(format!("{}: {mismatch}", entry["path"]));
        }
        results.push(
            json!({"path":entry["path"], "seed":entry["seed"], "policies":entry["policies"],
            "compared_actions":report.compared_actions, "matched_actions":report.matched_actions,
            "transitions":report.transitions, "mismatch":report.mismatch}),
        );
    }
    if let Some(path) = std::env::var_os("OPPONENT_PARITY_REPORT") {
        std::fs::write(path, serde_json::to_string_pretty(&results).unwrap() + "\n").unwrap();
    }
    assert!(
        failures.is_empty(),
        "original Python parity failed:\n{}",
        failures.join("\n")
    );
}

#[test]
fn every_oracle_rejects_tampered_actions_for_both_seats() {
    for entry in entries() {
        let rows = read_rows(&entry);
        for seat in 0..2 {
            let mut altered = rows.clone();
            altered[1]["actions"][seat]["farmer"] = json!(["TASK_7_1_MUTATION"]);
            let report = replay(&altered);
            let mismatch = report.mismatch.expect("tampered action must fail");
            assert!(
                mismatch.contains(&format!("step 0 seat {seat}"))
                    && mismatch.contains("action.farmer"),
                "{}: {mismatch}",
                entry["path"]
            );
            eprintln!("mutation {} seat {seat}: {mismatch}", entry["path"]);
        }
    }
}
