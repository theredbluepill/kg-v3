use kaggriculture_engine::{ATTRIB_FIELDS, Config, ECON_FIELDS, ECON_PASS, Game, TraceHeader};
use serde_json::{Value, json};
use std::path::Path;
use std::process::Command;

fn assert_public(actual: &Value, expected: &Value, context: &str) {
    assert_eq!(actual, expected, "public state: {context}");
}

fn assert_private(actual: &Value, expected: &Value, context: &str) {
    assert_eq!(actual, expected, "private state: {context}");
    for (a, e) in actual.as_array().unwrap().iter().zip(expected.as_array().unwrap()) {
        for field in ["shed", "seeds"] {
            assert_eq!(
                a[field].as_object().unwrap().keys().collect::<Vec<_>>(),
                e[field].as_object().unwrap().keys().collect::<Vec<_>>(),
                "private map key order: {context}: {field}"
            );
        }
        for (a, e) in a["inventories"].as_array().unwrap().iter()
            .zip(e["inventories"].as_array().unwrap())
        {
            assert_eq!(
                a.as_object().unwrap().keys().collect::<Vec<_>>(),
                e.as_object().unwrap().keys().collect::<Vec<_>>(),
                "private map key order: {context}: inventory"
            );
        }
    }
}

#[test]
#[should_panic(expected = "private map key order")]
fn private_order_mismatch_is_rejected() {
    let a: Value = serde_json::from_str(
        r#"[{"shed":{},"seeds":{},"inventories":[{"WHEAT":1,"MILK":2}]}]"#,
    ).unwrap();
    let b: Value = serde_json::from_str(
        r#"[{"shed":{},"seeds":{},"inventories":[{"MILK":2,"WHEAT":1}]}]"#,
    ).unwrap();
    assert_private(&a, &b, "negative comparator case");
}

#[test]
#[should_panic(expected = "public state")]
fn public_number_mismatch_is_rejected() {
    let a: Value = serde_json::from_str(r#"{"prices":{"WHEAT":20}}"#).unwrap();
    let b: Value = serde_json::from_str(r#"{"prices":{"WHEAT":20.0}}"#).unwrap();
    assert_public(&a, &b, "negative comparator case");
}

#[test]
fn kernel_public_api() {
    let config = Config { episode_steps: 3_i64.into(), ..Config::default() };
    let mut game = Game::new(config, 42, 2).unwrap();
    assert_eq!(game.public_state().step, 0);
    assert_eq!(game.econ_counters().unwrap(), &[[0_u64; ECON_FIELDS]; 2]);
    assert_eq!(game.attrib_counters().unwrap(), &[[0_u64; ATTRIB_FIELDS]; 2]);
    let pass = [json!({"farmer": ["PASS"]}), json!({"farmer": ["PASS"]})];
    game.step(&pass).unwrap();
    assert_eq!(game.public_state().step, 1);
    assert!(!game.snapshot().done);
    assert_eq!(game.econ_counters().unwrap()[0][ECON_PASS], 1);
    assert!(game.terminal_banks().is_none());
    game.step_with_market_metrics(&pass).unwrap();
    assert!(game.snapshot().done);
    assert_eq!(game.terminal_banks().unwrap(), &[3000.0, 3000.0]);
}

fn replay(file: &str, episode_id: u64) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(file);
    let output = Command::new("gzip").arg("-dc").arg(&path).output()
        .expect("gzip is required for pinned replay fixtures");
    assert!(output.status.success(), "{}: {}", path.display(), String::from_utf8_lossy(&output.stderr));
    let text = String::from_utf8(output.stdout).unwrap();
    let mut lines = text.lines();
    let raw: Value = serde_json::from_str(lines.next().expect("trace header")).unwrap();
    assert_eq!(raw["type"], "header");
    assert_eq!(raw["source"]["episode_id"].as_u64().unwrap(), episode_id);
    assert_eq!(raw["source"]["module_version"], "1.32.7");
    assert_eq!(raw["source"]["engine_sha256"],
        "bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e");
    let header: TraceHeader = serde_json::from_value(raw.clone()).unwrap();
    assert_eq!(header.transitions, 719);
    let mut game = Game::new(header.configuration.clone(), header.seed.as_i64().unwrap(), 2).unwrap();
    let initial = game.snapshot();
    assert_public(&serde_json::to_value(&initial.public).unwrap(), &raw["initial"]["public"], file);
    assert_private(&serde_json::to_value(&initial.privates).unwrap(), &raw["initial"]["privates"], file);
    assert_eq!(initial.statuses, vec!["ACTIVE", "ACTIVE"]);
    assert_eq!(initial.rewards, vec![0.0, 0.0]);
    assert!(!initial.done);
    let mut count = 0;
    for (index, line) in lines.enumerate() {
        let row: Value = serde_json::from_str(line).unwrap();
        let context = format!("{file}, transition {index}");
        assert_eq!(row["type"], "transition", "{context}");
        assert_eq!(row["from_step"].as_u64().unwrap(), index as u64, "{context}");
        game.step(row["actions"].as_array().unwrap())
            .unwrap_or_else(|error| panic!("{context}: {error}"));
        let actual = game.snapshot();
        assert_public(&serde_json::to_value(&actual.public).unwrap(), &row["expected"], &context);
        assert_private(&serde_json::to_value(&actual.privates).unwrap(), &row["privates"], &context);
        let statuses: Vec<String> = serde_json::from_value(row["statuses"].clone()).unwrap();
        let rewards: Vec<f64> = serde_json::from_value(row["rewards"].clone()).unwrap();
        assert_eq!(actual.statuses, statuses, "{context}");
        assert_eq!(actual.rewards, rewards, "{context}");
        assert_eq!(actual.done, index == 718, "{context}");
        assert_eq!(actual.public.step, index + 1, "{context}");
        count += 1;
    }
    assert_eq!(count, header.transitions);
    assert_eq!(game.terminal_banks().unwrap(), header.terminal_banks.as_slice(), "{file}");
}

#[test]
fn episode_95324500() { replay("fixtures/episode-95324500.jsonl.gz", 95324500); }
#[test]
fn episode_95901360() { replay("fixtures/episode-95901360.jsonl.gz", 95901360); }
#[test]
fn episode_95921764() { replay("fixtures/episode-95921764.jsonl.gz", 95921764); }
#[test]
fn episode_95990191() { replay("fixtures/episode-95990191.jsonl.gz", 95990191); }
