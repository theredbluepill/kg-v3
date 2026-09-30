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

fn policy_kinds(header: &Value) -> Result<Vec<OpponentKind>, String> {
    let policies = header["source"]["policies"]
        .as_array()
        .ok_or("missing policies")?;
    if policies.len() != 2 {
        return Err("expected two policies".into());
    }
    policies
        .iter()
        .map(|v| {
            let key = v.as_str().ok_or("policy must be a string")?;
            key.strip_prefix("builtin:")
                .or_else(|| key.strip_prefix("sibling:"))
                .or_else(|| key.strip_prefix("upstream:"))
                .ok_or("unknown policy namespace")?
                .parse()
        })
        .collect()
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
    let kinds = policy_kinds(raw)?;
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

/// Cha22's three-game corpus lives apart from the 7.1 corpus, whose replay
/// fixture covers exactly the frozen four-bot traces.
fn cha22_directory() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/oracle-cha22")
}

fn entries() -> Vec<Value> {
    entries_in(&directory())
}

fn entries_in(directory: &Path) -> Vec<Value> {
    let manifest: Value = serde_json::from_slice(
        &std::fs::read(directory.join("MANIFEST.json"))
            .expect("generate the original-Python opponent oracle first"),
    )
    .unwrap();
    let entries = manifest["traces"].as_array().unwrap().clone();
    assert!(!entries.is_empty(), "oracle corpus must not be empty");
    entries
}

fn read_rows(entry: &Value) -> Vec<Value> {
    read_rows_in(&directory(), entry)
}

fn read_rows_in(directory: &Path, entry: &Value) -> Vec<Value> {
    let path = directory.join(entry["path"].as_str().unwrap());
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
    compare_corpus(&directory(), "OPPONENT_PARITY_REPORT");
}

#[test]
fn original_cha22_oracles_match_native_actions_and_states() {
    let entries = entries_in(&cha22_directory());
    assert_eq!(entries.len(), 3, "Cha22 corpus is three games");
    for seat in 0..2 {
        assert!(
            entries
                .iter()
                .any(|entry| entry["policies"][seat] == "upstream:cha22"),
            "Cha22 must play seat {seat}"
        );
    }
    compare_corpus(&cha22_directory(), "CHA22_PARITY_REPORT");
}

#[test]
fn cha22_oracles_reject_tampered_actions_for_both_seats() {
    let directory = cha22_directory();
    for entry in entries_in(&directory) {
        let rows = read_rows_in(&directory, &entry);
        let seat = entry["policies"]
            .as_array()
            .unwrap()
            .iter()
            .position(|policy| policy == "upstream:cha22")
            .unwrap();
        let mut altered = rows.clone();
        altered[400]["actions"][seat]["market"] = json!([["TASK_CHA22_MUTATION"]]);
        let mismatch = replay(&altered)
            .mismatch
            .expect("tampered action must fail");
        assert!(
            mismatch.contains(&format!("step 399 seat {seat} cha22 action.market")),
            "{}: {mismatch}",
            entry["path"]
        );
    }
}

fn compare_corpus(directory: &Path, report_env: &str) {
    let mut results = Vec::new();
    let mut failures = Vec::new();
    for entry in entries_in(directory) {
        // Kaggle's simulation image runs CPython 3.11; 3.12's compensated float
        // sum() changes R04's decisions, so only 3.11 traces are oracles.
        let runtime = entry["python_runtime"].as_str().unwrap_or_default();
        assert!(
            runtime.starts_with("3.11."),
            "{}: oracle runtime {runtime:?} is not CPython 3.11",
            entry["path"]
        );
        let report = replay(&read_rows_in(directory, &entry));
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
    if let Some(path) = std::env::var_os(report_env) {
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

// ------------------------------------------------ original mid-episode replay

/// Mid-day (day 1 hour 13), a day reset (day 15 hour 0) and the last hour before
/// the final day; each window resumes for one full day.
const REPLAY_POINTS: [u64; 3] = [37, 360, 695];
const REPLAY_RESUME: u64 = 24;

fn replay_document() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/replay/REPLAY.json.gz");
    let output = Command::new("gzip")
        .args(["-dc"])
        .arg(&path)
        .output()
        .expect("gzip required");
    assert!(
        output.status.success(),
        "generate the original-Python replay oracle first: {}",
        path.display()
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

/// Rebuild fresh native controllers from the recorded Python prefix, as the
/// lifecycle requires (a fresh controller only at step zero), then compare their
/// resumed decisions with the frozen decisions of fresh original Python
/// controllers rebuilt from the same prefix. Recorded actions drive the prefix;
/// frozen resumed actions drive the resumed engine steps.
fn resume(case: &Value, rows: &[Value], compared: &mut [usize; 2]) -> Result<(), String> {
    let raw = &rows[0];
    let label = format!(
        "{}@{}",
        case["source"].as_str().unwrap_or("?"),
        case["reconstruct_step"]
    );
    if case["policies"] != raw["source"]["policies"] || case["seed"] != raw["seed"] {
        return Err(format!(
            "{label}: replay case identity differs from its source"
        ));
    }
    let header: TraceHeader = serde_json::from_value(raw.clone()).map_err(|e| e.to_string())?;
    let engine = kaggriculture_engine::Game::new_with_seed_decimal(
        header.configuration.clone(),
        &header.seed.to_string(),
        2,
    )?;
    let mut game = Game::from_engine(engine, &header.configuration)?;
    let kinds = policy_kinds(raw)?;
    let point = case["reconstruct_step"]
        .as_u64()
        .ok_or("reconstruct_step")? as usize;
    let resumed = case["resumed_actions"]
        .as_array()
        .ok_or("resumed_actions")?;
    if resumed.len() as u64 != case["resume_steps"].as_u64().ok_or("resume_steps")? {
        return Err(format!("{label}: resumed action count"));
    }
    let mut seats = [
        SeatController::new(kinds[0], 0, &game)?,
        SeatController::new(kinds[1], 1, &game)?,
    ];
    for step in 0..point + resumed.len() {
        let (phase, expected) = if step < point {
            ("prefix", &rows[1 + step]["actions"])
        } else {
            ("resume", &resumed[step - point])
        };
        for seat in 0..2 {
            let action = seats[seat].action(&game, seat)?;
            if let Some(diff) = difference(
                &action,
                &expected[seat],
                &format!(
                    "{label} {phase} step {step} seat {seat} {} action",
                    kinds[seat].key()
                ),
                false,
            ) {
                return Err(diff);
            }
            if phase == "resume" {
                compared[seat] += 1;
            }
        }
        game.step(expected.as_array().ok_or("actions must be an array")?)?;
    }
    let state = game.snapshot();
    let final_state = &case["final"];
    for (actual, expected, part, order) in [
        (
            serde_json::to_value(&state.public).unwrap(),
            &final_state["public"],
            "public",
            true,
        ),
        (
            serde_json::to_value(&state.privates).unwrap(),
            &final_state["privates"],
            "privates",
            true,
        ),
        (
            json!(state.statuses),
            &final_state["statuses"],
            "statuses",
            false,
        ),
    ] {
        if let Some(diff) = difference(&actual, expected, &format!("{label} final {part}"), order) {
            return Err(diff);
        }
    }
    Ok(())
}

fn replay_sources() -> std::collections::BTreeMap<String, Vec<Value>> {
    entries()
        .iter()
        .map(|entry| (entry["path"].as_str().unwrap().to_owned(), read_rows(entry)))
        .collect()
}

#[test]
fn original_python_mid_episode_replays_match_native_resumed_actions() {
    use sha2::{Digest, Sha256};
    let document = replay_document();
    let manifest = std::fs::read(directory().join("MANIFEST.json")).unwrap();
    assert_eq!(document["format"], "kaggriculture-opponent-replay-v1");
    assert_eq!(document["module_version"], "1.32.7");
    assert_eq!(document["engine_sha256"], ENGINE_SHA);
    let runtime = document["python_runtime"].as_str().unwrap_or_default();
    assert!(
        runtime.starts_with("3.11."),
        "replay runtime {runtime:?} is not CPython 3.11"
    );
    assert_eq!(
        document["oracle_manifest_sha256"],
        format!("{:x}", Sha256::digest(&manifest)),
        "replay oracle was generated from another oracle manifest"
    );
    assert_eq!(document["points"], json!(REPLAY_POINTS));
    assert_eq!(document["resume_steps"], REPLAY_RESUME);
    let sources = replay_sources();
    let cases = document["cases"].as_array().unwrap();
    // Every frozen oracle is replayed at every point, so every bot resumes in
    // both seats at every point.
    let mut covered = std::collections::BTreeSet::new();
    for case in cases {
        let point = case["reconstruct_step"].as_u64().unwrap();
        assert_eq!(case["resume_steps"], REPLAY_RESUME);
        for (seat, policy) in case["policies"].as_array().unwrap().iter().enumerate() {
            covered.insert((policy.as_str().unwrap().to_owned(), seat, point));
        }
    }
    assert_eq!(cases.len(), sources.len() * REPLAY_POINTS.len());
    assert_eq!(covered.len(), 4 * 2 * REPLAY_POINTS.len(), "{covered:?}");
    let mut compared = [0; 2];
    let mut failures = Vec::new();
    for case in cases {
        let rows = &sources[case["source"].as_str().unwrap()];
        if let Err(mismatch) = resume(case, rows, &mut compared) {
            failures.push(mismatch);
        }
    }
    eprintln!(
        "replay cases {} compared resumed actions {compared:?}",
        cases.len()
    );
    if let Some(path) = std::env::var_os("OPPONENT_REPLAY_REPORT") {
        let report = json!({"cases": cases.len(), "compared_actions": compared,
            "failures": failures});
        std::fs::write(path, serde_json::to_string_pretty(&report).unwrap() + "\n").unwrap();
    }
    assert!(
        failures.is_empty(),
        "original Python replay parity failed:\n{}",
        failures.join("\n")
    );
    let expected = cases.len() * REPLAY_RESUME as usize;
    assert_eq!(compared, [expected; 2]);
}

#[test]
fn replay_comparison_rejects_tampered_resumed_actions_and_final_state() {
    let document = replay_document();
    let sources = replay_sources();
    let cases: Vec<&Value> = document["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|case| case["reconstruct_step"] == REPLAY_POINTS[0])
        .collect();
    assert_eq!(cases.len(), sources.len());
    let last = REPLAY_RESUME as usize - 1;
    for case in &cases {
        let rows = &sources[case["source"].as_str().unwrap()];
        for (index, seat) in [(0, 0), (0, 1), (last, 1)] {
            if index == last && case != &cases[0] {
                continue;
            }
            let mut altered = (*case).clone();
            altered["resumed_actions"][index][seat]["farmer"] = json!(["TASK_7_1_MUTATION"]);
            let mismatch =
                resume(&altered, rows, &mut [0; 2]).expect_err("tampered action must fail");
            let step = REPLAY_POINTS[0] as usize + index;
            assert!(
                mismatch.contains(&format!("resume step {step} seat {seat}"))
                    && mismatch.contains("action.farmer"),
                "{mismatch}"
            );
            eprintln!("replay mutation: {mismatch}");
        }
    }
    let mut altered = cases[0].clone();
    altered["final"]["public"]["hour"] = json!(99);
    let mismatch = resume(
        &altered,
        &sources[cases[0]["source"].as_str().unwrap()],
        &mut [0; 2],
    )
    .expect_err("tampered final state must fail");
    assert!(mismatch.contains("final public.hour"), "{mismatch}");
}
