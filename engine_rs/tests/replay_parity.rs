// The vendored library needs these command-line allowances; new tests do not.
#![deny(
    clippy::too_many_arguments,
    clippy::collapsible_if,
    clippy::needless_range_loop
)]

//! Replay parity against Kaggle's Python engine (`kaggle-environments==1.32.7`).
//!
//! One comparator serves three trace sources: the four recorded official episodes,
//! the committed traces that `scripts/kaggriculture_parity/generate_traces.py`
//! produced by running Kaggle's engine live, and an optional sweep directory named
//! by `KAGG_PARITY_TRACES`. Generated traces may also contain `rejected` records:
//! Python's interpreter raised on those actions, so Kaggle kept the old state, and
//! Rust must return an error without mutating its state.

use kaggriculture_engine::{ATTRIB_FIELDS, Config, ECON_FIELDS, ECON_PASS, Game, TraceHeader};
use serde_json::{Value, json};
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;

const MODULE_VERSION: &str = "1.32.7";
const ENGINE_SHA256: &str = "bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e";
const GENERATED_DIR: &str = "fixtures/generated";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DiffKind {
    Value,
    KeyOrder,
}

/// First difference between two JSON trees. Values are compared before key order,
/// so a changed value is reported as such even when it also moves keys.
fn first_difference(actual: &Value, expected: &Value, path: &str) -> Option<(DiffKind, String)> {
    value_difference(actual, expected, path)
        .map(|path| (DiffKind::Value, path))
        .or_else(|| order_difference(actual, expected, path).map(|p| (DiffKind::KeyOrder, p)))
}

fn value_difference(actual: &Value, expected: &Value, path: &str) -> Option<String> {
    match (actual, expected) {
        (Value::Object(a), Value::Object(e)) => {
            for (key, value) in e {
                let child = format!("{path}.{key}");
                match a.get(key) {
                    None => return Some(format!("{child} (missing in Rust)")),
                    Some(other) => {
                        if let Some(found) = value_difference(other, value, &child) {
                            return Some(found);
                        }
                    },
                }
            }
            a.keys()
                .find(|key| !e.contains_key(*key))
                .map(|key| format!("{path}.{key} (extra in Rust)"))
        },
        (Value::Array(a), Value::Array(e)) => {
            for (index, (value, other)) in a.iter().zip(e).enumerate() {
                if let Some(found) = value_difference(value, other, &format!("{path}[{index}]")) {
                    return Some(found);
                }
            }
            (a.len() != e.len()).then(|| format!("{path} (length {} vs {})", a.len(), e.len()))
        },
        _ => (actual != expected).then(|| path.to_string()),
    }
}

/// `Value` equality ignores object key order, but IndexMap insertion order is
/// engine state (market inventory/prices, sheds, seeds, inventories, tiles).
fn order_difference(actual: &Value, expected: &Value, path: &str) -> Option<String> {
    match (actual, expected) {
        (Value::Object(a), Value::Object(e)) => {
            if !a.keys().eq(e.keys()) {
                return Some(path.to_string());
            }
            a.iter()
                .find_map(|(key, value)| order_difference(value, &e[key], &format!("{path}.{key}")))
        },
        (Value::Array(a), Value::Array(e)) => {
            a.iter()
                .zip(e)
                .enumerate()
                .find_map(|(index, (value, other))| {
                    order_difference(value, other, &format!("{path}[{index}]"))
                })
        },
        _ => None,
    }
}

/// Resolve a reported `label.key[index]...` path (annotations after a space are
/// ignored) inside the tree that `label` names.
fn lookup<'a>(root: &'a Value, label: &str, path: &str) -> Option<&'a Value> {
    let rest = path.strip_prefix(label)?;
    let rest = rest.split(' ').next().unwrap_or(rest);
    let mut node = root;
    let mut chars = rest.chars().peekable();
    while let Some(c) = chars.next() {
        let mut token = String::new();
        while let Some(&next) = chars.peek() {
            if next == '.' || next == '[' || (c == '[' && next == ']') {
                break;
            }
            token.push(next);
            chars.next();
        }
        node = match c {
            '.' => node.get(token.as_str())?,
            '[' => {
                chars.next();
                node.get(token.parse::<usize>().ok()?)?
            },
            _ => return None,
        };
    }
    Some(node)
}

fn excerpt(value: Option<&Value>) -> String {
    let text = value.map_or_else(|| "<absent>".to_string(), Value::to_string);
    if text.chars().count() > 200 {
        format!("{}…", text.chars().take(200).collect::<String>())
    } else {
        text
    }
}

/// The first point at which a trace and the Rust kernel disagree.
#[derive(Clone, Debug)]
struct Divergence {
    /// Zero-based JSONL line (0 is the header).
    line: usize,
    from_step: Option<usize>,
    kind: String,
    field: String,
    expected: String,
    actual: String,
}

impl Divergence {
    fn new(line: usize, from_step: Option<usize>, kind: &str, field: &str) -> Self {
        Self {
            line,
            from_step,
            kind: kind.to_string(),
            field: field.to_string(),
            expected: String::new(),
            actual: String::new(),
        }
    }

    fn values(mut self, expected: impl ToString, actual: impl ToString) -> Self {
        self.expected = expected.to_string();
        self.actual = actual.to_string();
        self
    }

    fn to_json(&self) -> Value {
        json!({
            "line": self.line,
            "from_step": self.from_step,
            "kind": self.kind,
            "field": self.field,
            "expected": self.expected,
            "actual": self.actual,
        })
    }
}

impl fmt::Display for Divergence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "line {} from_step {:?}: {} at {}: expected {} actual {}",
            self.line, self.from_step, self.kind, self.field, self.expected, self.actual
        )
    }
}

fn compare_tree(
    actual: &Value,
    expected: &Value,
    label: &str,
    line: usize,
    from_step: Option<usize>,
) -> Result<(), Divergence> {
    match first_difference(actual, expected, label) {
        None => Ok(()),
        Some((kind, path)) => {
            let name = match kind {
                DiffKind::Value => format!("{label} state"),
                DiffKind::KeyOrder => format!("{label} map key order"),
            };
            Err(Divergence::new(line, from_step, &name, &path).values(
                excerpt(lookup(expected, label, &path)),
                excerpt(lookup(actual, label, &path)),
            ))
        },
    }
}

fn assert_public(actual: &Value, expected: &Value, context: &str) {
    if let Err(divergence) = compare_tree(actual, expected, "public", 0, None) {
        panic!("{}: {context}: {divergence}", divergence.kind);
    }
}

fn assert_private(actual: &Value, expected: &Value, context: &str) {
    if let Err(divergence) = compare_tree(actual, expected, "private", 0, None) {
        panic!("{}: {context}: {divergence}", divergence.kind);
    }
}

#[test]
#[should_panic(expected = "private map key order")]
fn private_order_mismatch_is_rejected() {
    let a: Value =
        serde_json::from_str(r#"[{"shed":{},"seeds":{},"inventories":[{"WHEAT":1,"MILK":2}]}]"#)
            .unwrap();
    let b: Value =
        serde_json::from_str(r#"[{"shed":{},"seeds":{},"inventories":[{"MILK":2,"WHEAT":1}]}]"#)
            .unwrap();
    assert_private(&a, &b, "negative comparator case");
}

#[test]
#[should_panic(expected = "public map key order")]
fn public_order_mismatch_is_rejected() {
    let a: Value =
        serde_json::from_str(r#"{"market":{"inventory":{"WHEAT":1,"MILK":2}}}"#).unwrap();
    let b: Value =
        serde_json::from_str(r#"{"market":{"inventory":{"MILK":2,"WHEAT":1}}}"#).unwrap();
    assert_public(&a, &b, "negative comparator case");
}

#[test]
#[should_panic(expected = "private map key order")]
fn nested_private_order_mismatch_is_rejected() {
    let a: Value = serde_json::from_str(r#"[{"seeds":{},"shed":{},"inventories":[]}]"#).unwrap();
    let b: Value = serde_json::from_str(r#"[{"shed":{},"seeds":{},"inventories":[]}]"#).unwrap();
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
fn comparator_names_first_differing_field() {
    let a: Value = serde_json::from_str(r#"{"farms":[{"money":1.0,"hands":[[1,2]]}]}"#).unwrap();
    let b: Value = serde_json::from_str(r#"{"farms":[{"money":1.0,"hands":[[1,3]]}]}"#).unwrap();
    let divergence = compare_tree(&a, &b, "public", 7, Some(6)).unwrap_err();
    assert_eq!(divergence.kind, "public state");
    assert_eq!(divergence.field, "public.farms[0].hands[0][1]");
    assert_eq!(
        (divergence.expected.as_str(), divergence.actual.as_str()),
        ("3", "2")
    );
}

fn view(public: &str, statuses: [&str; 2], rewards: [f64; 2], done: bool) -> StateView {
    StateView {
        public: serde_json::from_str(public).unwrap(),
        privates: json!([{"seeds": {"WHEAT": 1, "CORN": 2}}, {}]),
        statuses: statuses.map(str::to_string).to_vec(),
        rewards: rewards.to_vec(),
        done,
    }
}

#[test]
fn rollback_check_accepts_an_untouched_state() {
    let before = view(r#"{"a":1,"b":2}"#, ["ACTIVE", "ACTIVE"], [0.0, 0.0], false);
    assert!(rollback_difference(&before, &before.clone(), 4, Some(3)).is_ok());
}

#[test]
fn rollback_check_rejects_a_key_order_only_mutation() {
    let before = view(r#"{"a":1,"b":2}"#, ["ACTIVE", "ACTIVE"], [0.0, 0.0], false);
    let after = view(r#"{"b":2,"a":1}"#, ["ACTIVE", "ACTIVE"], [0.0, 0.0], false);
    // serde_json's Value equality ignores key order; the rollback check must not.
    assert_eq!(before.public, after.public);
    let divergence = rollback_difference(&before, &after, 4, Some(3)).unwrap_err();
    assert_eq!(divergence.kind, "rust_mutated public map key order");
    assert_eq!((divergence.line, divergence.from_step), (4, Some(3)));
    let mut private = before.clone();
    let seeds = private.privates[0]["seeds"].as_object_mut().unwrap();
    let wheat = seeds.shift_remove("WHEAT").unwrap();
    seeds.insert("WHEAT".to_string(), wheat);
    let divergence = rollback_difference(&before, &private, 4, Some(3)).unwrap_err();
    assert_eq!(divergence.kind, "rust_mutated private map key order");
    assert_eq!(divergence.field, "private[0].seeds");
}

#[test]
fn rollback_check_rejects_status_reward_and_done_mutations() {
    let before = view(r#"{"a":1}"#, ["ACTIVE", "ACTIVE"], [0.0, 0.0], false);
    for (after, field) in [
        (
            view(r#"{"a":1}"#, ["DONE", "ACTIVE"], [0.0, 0.0], false),
            "statuses",
        ),
        (
            view(r#"{"a":1}"#, ["ACTIVE", "ACTIVE"], [1.0, 0.0], false),
            "rewards",
        ),
        (
            view(r#"{"a":1}"#, ["ACTIVE", "ACTIVE"], [0.0, 0.0], true),
            "done",
        ),
        (
            view(r#"{"a":2}"#, ["ACTIVE", "ACTIVE"], [0.0, 0.0], false),
            "public.a",
        ),
    ] {
        let divergence = rollback_difference(&before, &after, 4, Some(3)).unwrap_err();
        assert!(divergence.kind.starts_with("rust_mutated"), "{divergence}");
        assert_eq!(divergence.field, field);
    }
}

#[test]
fn kernel_public_api() {
    let config = Config {
        episode_steps: 3_i64.into(),
        ..Config::default()
    };
    let mut game = Game::new(config, 42, 2).unwrap();
    assert_eq!(game.public_state().step, 0);
    assert_eq!(game.econ_counters().unwrap(), &[[0_u64; ECON_FIELDS]; 2]);
    assert_eq!(
        game.attrib_counters().unwrap(),
        &[[0_u64; ATTRIB_FIELDS]; 2]
    );
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

#[test]
fn stepped_candidate_preserves_source_and_matches_recorded_games() {
    let mut transitions = 0;
    for episode in [95324500, 95901360, 95921764, 95990191] {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("fixtures/episode-{episode}.jsonl.gz"));
        let text = read_trace(&path);
        let mut lines = text.lines();
        let header: TraceHeader = serde_json::from_str(lines.next().unwrap()).unwrap();
        let mut game =
            Game::new_with_seed_decimal(header.configuration, &header.seed.to_string(), 2).unwrap();
        for line in lines {
            let row: Value = serde_json::from_str(line).unwrap();
            let actions = row["actions"].as_array().unwrap();
            // Debug includes the hidden seed/config and telemetry ledgers; serialized
            // snapshots additionally pin numeric representation and JSON key order.
            let before = format!("{game:?}");
            let (candidate, metrics) = game.stepped_with_market_metrics(actions).unwrap();
            assert_eq!(format!("{game:?}"), before);
            let mutable_metrics = game.step_with_market_metrics(actions).unwrap();
            assert_eq!(format!("{metrics:?}"), format!("{mutable_metrics:?}"));
            assert_eq!(format!("{candidate:?}"), format!("{game:?}"));
            assert_eq!(
                serde_json::to_vec(&candidate.snapshot()).unwrap(),
                serde_json::to_vec(&game.snapshot()).unwrap()
            );
            let (public, privates) = snapshot_values(&candidate);
            compare_tree(&public, &row["expected"], "public", transitions, None).unwrap();
            compare_tree(&privates, &row["privates"], "private", transitions, None).unwrap();
            game = candidate;
            transitions += 1;
        }
        assert!(game.snapshot().done);
    }
    assert_eq!(transitions, 4 * 719);
}

#[test]
fn stepped_candidate_preserves_early_and_deferred_errors() {
    let pass = [json!({"farmer": ["PASS"]}), json!({"farmer": ["PASS"]})];
    let mut config = Config {
        episode_steps: 2_i64.into(),
        weed_spawn_chance: Value::from(0),
        ..Config::default()
    };
    // Failure occurs during market pricing, after unit and telemetry mutations.
    config
        .market_params
        .insert("WHEAT".to_string(), json!({"T": "10"}));
    let mut game = Game::new(config, 7, 2).unwrap();
    let before = format!("{game:?}");
    let error = game.stepped_with_market_metrics(&pass).unwrap_err();
    assert!(error.contains("must be numeric"));
    assert_eq!(game.step_with_market_metrics(&pass).unwrap_err(), error);
    assert_eq!(format!("{game:?}"), before);

    let wrong_count = game.stepped_with_market_metrics(&pass[..1]).unwrap_err();
    assert_eq!(wrong_count, "got 1 actions for 2 farms");
    assert_eq!(
        game.step_with_market_metrics(&pass[..1]).unwrap_err(),
        wrong_count
    );
    assert_eq!(format!("{game:?}"), before);

    let mut completed = Game::new(
        Config {
            episode_steps: 2_i64.into(),
            ..Config::default()
        },
        7,
        2,
    )
    .unwrap();
    completed.step(&pass).unwrap();
    let before = format!("{completed:?}");
    // Completed-game validation still precedes the action-count check.
    let error = completed.stepped_with_market_metrics(&[]).unwrap_err();
    assert_eq!(error, "cannot step a completed game");
    assert_eq!(completed.step_with_market_metrics(&[]).unwrap_err(), error);
    assert_eq!(format!("{completed:?}"), before);
}

/// Counts from one successful replay.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ReplayStats {
    transitions: usize,
    rejected: usize,
}

fn read_trace(path: &Path) -> String {
    if path.extension().is_some_and(|ext| ext == "gz") {
        let output = Command::new("gzip")
            .arg("-dc")
            .arg(path)
            .output()
            .expect("gzip is required for compressed replay fixtures");
        assert!(
            output.status.success(),
            "{}: {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).unwrap()
    } else {
        std::fs::read_to_string(path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
    }
}

fn parse_line(line: &str, index: usize) -> Result<Value, Divergence> {
    serde_json::from_str(line)
        .map_err(|error| Divergence::new(index, None, "format", "json").values("valid JSON", error))
}

fn snapshot_values(game: &Game) -> (Value, Value) {
    let view = state_view(game);
    (view.public, view.privates)
}

/// Everything a Python-rejected step must leave untouched in the Rust game.
#[derive(Clone, Debug, PartialEq)]
struct StateView {
    public: Value,
    privates: Value,
    statuses: Vec<String>,
    rewards: Vec<f64>,
    done: bool,
}

fn state_view(game: &Game) -> StateView {
    let snapshot = game.snapshot();
    StateView {
        public: serde_json::to_value(&snapshot.public).unwrap(),
        privates: serde_json::to_value(&snapshot.privates).unwrap(),
        statuses: snapshot.statuses,
        rewards: snapshot.rewards,
        done: snapshot.done,
    }
}

/// A rejected step must leave the complete state untouched: public and private
/// trees (values and map key order, which `Value` equality ignores), statuses,
/// rewards and completion.
fn rollback_difference(
    before: &StateView,
    after: &StateView,
    line: usize,
    from_step: Option<usize>,
) -> Result<(), Divergence> {
    let mutated = |divergence: Divergence| Divergence {
        kind: format!("rust_mutated {}", divergence.kind),
        ..divergence
    };
    compare_tree(&after.public, &before.public, "public", line, from_step).map_err(mutated)?;
    compare_tree(
        &after.privates,
        &before.privates,
        "private",
        line,
        from_step,
    )
    .map_err(mutated)?;
    let scalar = |field: &str, expected: String, actual: String| {
        Err(Divergence::new(line, from_step, "rust_mutated", field).values(expected, actual))
    };
    if after.statuses != before.statuses {
        return scalar(
            "statuses",
            format!("{:?}", before.statuses),
            format!("{:?}", after.statuses),
        );
    }
    if after.rewards != before.rewards {
        return scalar(
            "rewards",
            format!("{:?}", before.rewards),
            format!("{:?}", after.rewards),
        );
    }
    if after.done != before.done {
        return scalar("done", before.done.to_string(), after.done.to_string());
    }
    Ok(())
}

/// Replay one trace from its configuration and seed, stopping at the first
/// divergence. Nothing from the expected states initializes the Rust game.
fn replay_text(text: &str) -> Result<ReplayStats, Divergence> {
    let mut lines = text.lines();
    let raw = parse_line(lines.next().unwrap_or(""), 0)?;
    let header_error = |field: &str, expected: &dyn ToString, actual: &Value| {
        Divergence::new(0, None, "header", field).values(expected.to_string(), actual)
    };
    if raw["type"] != "header" {
        return Err(header_error("type", &"header", &raw["type"]));
    }
    if raw["source"]["module_version"] != MODULE_VERSION {
        return Err(header_error(
            "source.module_version",
            &MODULE_VERSION,
            &raw["source"]["module_version"],
        ));
    }
    if raw["source"]["engine_sha256"] != ENGINE_SHA256 {
        return Err(header_error(
            "source.engine_sha256",
            &ENGINE_SHA256,
            &raw["source"]["engine_sha256"],
        ));
    }
    let header: TraceHeader = serde_json::from_value(raw.clone()).map_err(|error| {
        Divergence::new(0, None, "header", "schema").values("TraceHeader", error)
    })?;
    let mut game =
        Game::new_with_seed_decimal(header.configuration.clone(), &header.seed.to_string(), 2)
            .map_err(|error| {
                Divergence::new(0, None, "rust_error", "Game::new").values("Ok", error)
            })?;
    let initial = game.snapshot();
    let (public, privates) = snapshot_values(&game);
    compare_tree(&public, &raw["initial"]["public"], "public", 0, None)?;
    compare_tree(&privates, &raw["initial"]["privates"], "private", 0, None)?;
    if initial.statuses != ["ACTIVE", "ACTIVE"] || initial.rewards != [0.0, 0.0] || initial.done {
        return Err(
            Divergence::new(0, None, "initial", "statuses/rewards/done").values(
                "ACTIVE, 0, not done",
                format!("{:?}", (initial.statuses, initial.rewards, initial.done)),
            ),
        );
    }

    let mut stats = ReplayStats::default();
    for (offset, line) in lines.enumerate() {
        let index = offset + 1;
        let row = parse_line(line, index)?;
        let from_step = row["from_step"].as_u64().map(|step| step as usize);
        let current = game.public_state().step;
        if from_step != Some(current) {
            return Err(Divergence::new(index, from_step, "format", "from_step")
                .values(current, &row["from_step"]));
        }
        let Some(actions) = row["actions"].as_array() else {
            return Err(Divergence::new(index, from_step, "format", "actions")
                .values("array", &row["actions"]));
        };
        match row["type"].as_str() {
            Some("rejected") => {
                let before = state_view(&game);
                if game.step(actions).is_ok() {
                    return Err(Divergence::new(index, from_step, "rust_accepted", "step")
                        .values(format!("error ({})", row["python_error"]), "Ok"));
                }
                rollback_difference(&before, &state_view(&game), index, from_step)?;
                stats.rejected += 1;
            },
            Some("transition") => {
                if let Err(error) = game.step(actions) {
                    return Err(
                        Divergence::new(index, from_step, "rust_error", "step").values("Ok", error)
                    );
                }
                stats.transitions += 1;
                let actual = game.snapshot();
                let (public, privates) = snapshot_values(&game);
                compare_tree(&public, &row["expected"], "public", index, from_step)?;
                compare_tree(&privates, &row["privates"], "private", index, from_step)?;
                let statuses: Vec<String> = serde_json::from_value(row["statuses"].clone())
                    .map_err(|e| {
                        Divergence::new(index, from_step, "format", "statuses").values("strings", e)
                    })?;
                let rewards: Vec<f64> =
                    serde_json::from_value(row["rewards"].clone()).map_err(|e| {
                        Divergence::new(index, from_step, "format", "rewards").values("numbers", e)
                    })?;
                if actual.statuses != statuses {
                    return Err(Divergence::new(index, from_step, "statuses", "statuses")
                        .values(format!("{statuses:?}"), format!("{:?}", actual.statuses)));
                }
                if actual.rewards != rewards {
                    return Err(Divergence::new(index, from_step, "rewards", "rewards")
                        .values(format!("{rewards:?}"), format!("{:?}", actual.rewards)));
                }
                let last = stats.transitions == header.transitions;
                if actual.done != last {
                    return Err(
                        Divergence::new(index, from_step, "done", "done").values(last, actual.done)
                    );
                }
                if actual.public.step != current + 1 {
                    return Err(Divergence::new(index, from_step, "step", "public.step")
                        .values(current + 1, actual.public.step));
                }
            },
            _ => {
                return Err(Divergence::new(index, from_step, "format", "type")
                    .values("transition or rejected", &row["type"]));
            },
        }
    }
    if stats.transitions != header.transitions {
        return Err(Divergence::new(0, None, "format", "transitions")
            .values(header.transitions, stats.transitions));
    }
    let banks = game.terminal_banks().unwrap_or(&[]);
    if banks != header.terminal_banks.as_slice() {
        return Err(Divergence::new(0, None, "terminal_banks", "terminal_banks")
            .values(format!("{:?}", header.terminal_banks), format!("{banks:?}")));
    }
    Ok(stats)
}

fn engine_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
}

fn replay(file: &str, episode_id: u64) {
    let text = read_trace(&engine_path(file));
    let raw: Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
    assert_eq!(raw["source"]["episode_id"].as_u64().unwrap(), episode_id);
    assert_eq!(raw["transitions"], 719);
    let stats = replay_text(&text).unwrap_or_else(|divergence| panic!("{file}: {divergence}"));
    assert_eq!(
        stats,
        ReplayStats {
            transitions: 719,
            rejected: 0
        }
    );
}

#[test]
fn episode_95324500() {
    replay("fixtures/episode-95324500.jsonl.gz", 95324500);
}
#[test]
fn episode_95901360() {
    replay("fixtures/episode-95901360.jsonl.gz", 95901360);
}
#[test]
fn episode_95921764() {
    replay("fixtures/episode-95921764.jsonl.gz", 95921764);
}
#[test]
fn episode_95990191() {
    replay("fixtures/episode-95990191.jsonl.gz", 95990191);
}

fn generated_manifest() -> Vec<Value> {
    let path = engine_path(&format!("{GENERATED_DIR}/MANIFEST.json"));
    let manifest: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(manifest["kaggle_environments_version"], MODULE_VERSION);
    assert_eq!(manifest["python_engine_sha256"], ENGINE_SHA256);
    manifest["traces"].as_array().unwrap().clone()
}

fn generated_trace(name: &str) -> String {
    read_trace(&engine_path(&format!("{GENERATED_DIR}/{name}")))
}

/// Traces generated live from Kaggle's engine with seeded random, edge-case,
/// built-in and mixed-seat policies (see the manifest for each game's inputs).
///
/// Entries with `expected_divergence` are minimal repros of documented
/// divergences (D1, D2 in docs/rules-parity-coverage.md): one-step games whose
/// first action is the divergent input, carrying only the divergent field. The
/// vendored kernel is byte-pinned, so they are expected failures: each must still
/// diverge at exactly the recorded line, step, kind and field, or this test fails.
#[test]
fn generated_fixtures_replay() {
    let traces = generated_manifest();
    assert!(traces.len() >= 6, "expected the committed generated set");
    let mut failures = vec![];
    let mut expected_failures = 0;
    for entry in &traces {
        let name = entry["path"].as_str().unwrap();
        let result = replay_text(&generated_trace(name));
        match (&entry["expected_divergence"], result) {
            (Value::Null, Ok(stats)) => {
                assert_eq!(
                    stats.transitions as u64,
                    entry["transitions"].as_u64().unwrap(),
                    "{name}"
                );
                assert_eq!(
                    stats.rejected as u64,
                    entry["rejected"].as_u64().unwrap(),
                    "{name}"
                );
            },
            (Value::Null, Err(divergence)) => failures.push(format!("{name}: {divergence}")),
            (expected, Ok(_)) => failures.push(format!(
                "{name}: expected divergence did not occur ({expected}); update the record"
            )),
            (expected, Err(divergence)) => {
                let reason = expected["reason"].as_str().unwrap_or_default();
                assert!(
                    !reason.trim().is_empty(),
                    "{name}: expected failure needs a reason"
                );
                let observed = json!({
                    "line": divergence.line,
                    "from_step": divergence.from_step,
                    "kind": divergence.kind,
                    "field": divergence.field,
                });
                for key in ["line", "from_step", "kind", "field"] {
                    if observed[key] != expected[key] {
                        failures.push(format!(
                            "{name}: {key} {} != recorded {}",
                            observed[key], expected[key]
                        ));
                    }
                }
                expected_failures += 1;
            },
        }
    }
    eprintln!(
        "{} generated traces, {expected_failures} documented expected divergences",
        traces.len()
    );
    assert!(
        failures.is_empty(),
        "generated parity divergences:\n{}",
        failures.join("\n")
    );
}

fn edit_line(text: &str, line: usize, edit: impl FnOnce(&mut Value)) -> String {
    let mut lines: Vec<String> = text.lines().map(str::to_string).collect();
    let mut row: Value = serde_json::from_str(&lines[line]).unwrap();
    edit(&mut row);
    lines[line] = row.to_string();
    lines.join("\n") + "\n"
}

fn line_for_step(text: &str, record_type: &str, from_step: u64) -> usize {
    text.lines()
        .position(|line| {
            let row: Value = serde_json::from_str(line).unwrap();
            row["type"] == record_type && row["from_step"] == from_step
        })
        .unwrap_or_else(|| panic!("no {record_type} record from step {from_step}"))
}

const PERTURBED: &str = "gen-edge-vs-edge.jsonl.gz";

#[test]
fn generated_trace_perturbed_state_is_rejected() {
    let text = generated_trace(PERTURBED);
    assert!(replay_text(&text).is_ok());
    let line = line_for_step(&text, "transition", 100);
    let perturbed = edit_line(&text, line, |row| {
        let money = row["expected"]["farms"][0]["money"].as_f64().unwrap();
        row["expected"]["farms"][0]["money"] = json!(money + 1.0);
    });
    let divergence = replay_text(&perturbed).unwrap_err();
    assert_eq!((divergence.line, divergence.from_step), (line, Some(100)));
    assert_eq!(divergence.kind, "public state");
    assert_eq!(divergence.field, "public.farms[0].money");
}

#[test]
fn generated_trace_perturbed_private_order_is_rejected() {
    let text = generated_trace(PERTURBED);
    let line = line_for_step(&text, "transition", 50);
    let perturbed = edit_line(&text, line, |row| {
        let seeds = row["privates"][1]["seeds"].as_object_mut().unwrap();
        let key = seeds.keys().next().unwrap().clone();
        let value = seeds.shift_remove(&key).unwrap();
        seeds.insert(key, value);
    });
    let divergence = replay_text(&perturbed).unwrap_err();
    assert_eq!(divergence.from_step, Some(50));
    assert_eq!(divergence.kind, "private map key order");
    assert_eq!(divergence.field, "private[1].seeds");
}

#[test]
fn generated_trace_perturbed_action_is_rejected() {
    let text = generated_trace(PERTURBED);
    let line = line_for_step(&text, "transition", 0);
    let perturbed = edit_line(&text, line, |row| {
        let original = row["actions"][0].clone();
        let replacement =
            json!({"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "WHEAT", 1]]});
        assert_ne!(original, replacement);
        row["actions"][0] = replacement;
    });
    let divergence = replay_text(&perturbed).unwrap_err();
    assert_eq!(divergence.from_step, Some(0), "{divergence}");
}

#[test]
fn generated_trace_perturbed_rewards_are_rejected() {
    let text = generated_trace(PERTURBED);
    let line = line_for_step(&text, "transition", 100);
    let perturbed = edit_line(&text, line, |row| {
        assert_eq!(row["rewards"], json!([0, 0]));
        row["rewards"] = json!([1, 0]);
    });
    let divergence = replay_text(&perturbed).unwrap_err();
    assert_eq!((divergence.line, divergence.from_step), (line, Some(100)));
    assert_eq!(divergence.kind, "rewards", "{divergence}");
}

#[test]
fn generated_trace_perturbed_terminal_banks_are_rejected() {
    let text = generated_trace(PERTURBED);
    let perturbed = edit_line(&text, 0, |header| {
        let bank = header["terminal_banks"][0].as_f64().unwrap();
        header["terminal_banks"][0] = json!(bank + 1.0);
    });
    let divergence = replay_text(&perturbed).unwrap_err();
    assert_eq!((divergence.line, divergence.from_step), (0, None));
    assert_eq!(divergence.kind, "terminal_banks", "{divergence}");
}

#[test]
fn generated_trace_perturbed_transition_count_fails_done() {
    let text = generated_trace(PERTURBED);
    let perturbed = edit_line(&text, 0, |header| {
        let transitions = header["transitions"].as_u64().unwrap();
        header["transitions"] = json!(transitions + 1);
    });
    let divergence = replay_text(&perturbed).unwrap_err();
    assert_eq!(divergence.kind, "done", "{divergence}");
    assert_eq!(divergence.field, "done");
}

#[test]
fn generated_trace_rejection_claims_are_checked() {
    let text = generated_trace(PERTURBED);
    // An accepted step relabelled as rejected: Rust accepts it.
    let line = line_for_step(&text, "transition", 10);
    let relabelled = edit_line(&text, line, |row| row["type"] = json!("rejected"));
    let divergence = replay_text(&relabelled).unwrap_err();
    assert_eq!(
        (divergence.kind.as_str(), divergence.from_step),
        ("rust_accepted", Some(10))
    );
    // A rejected step relabelled as accepted: Rust rejects it, as Python did.
    let (line, step) = text
        .lines()
        .enumerate()
        .find_map(|(index, line)| {
            let row: Value = serde_json::from_str(line).unwrap();
            (row["type"] == "rejected").then(|| (index, row["from_step"].as_u64().unwrap()))
        })
        .expect("the edge trace records Python rejections");
    let relabelled = edit_line(&text, line, |row| row["type"] = json!("transition"));
    let divergence = replay_text(&relabelled).unwrap_err();
    assert_eq!(divergence.kind, "rust_error");
    assert_eq!(divergence.from_step, Some(step as usize));
}

/// Larger local or pod sweeps: `KAGG_PARITY_TRACES=<dir>` replays every
/// `*.jsonl.gz`/`*.jsonl` in that directory; `KAGG_PARITY_REPORT=<file>` receives a
/// JSON report of each trace's counts or first divergence. Unset, this is a no-op.
#[test]
fn env_directory_traces() {
    let Some(directory) = std::env::var_os("KAGG_PARITY_TRACES") else {
        eprintln!("KAGG_PARITY_TRACES is unset; skipping the external trace sweep");
        return;
    };
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&directory)
        .unwrap_or_else(|error| {
            panic!("KAGG_PARITY_TRACES={directory:?} (relative to engine_rs/): {error}")
        })
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            let name = path.file_name().unwrap().to_string_lossy();
            name.ends_with(".jsonl.gz") || name.ends_with(".jsonl")
        })
        .collect();
    paths.sort();
    assert!(
        !paths.is_empty(),
        "KAGG_PARITY_TRACES={directory:?} has no traces"
    );
    let mut report = vec![];
    let mut failures = vec![];
    for path in &paths {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        match replay_text(&read_trace(path)) {
            Ok(stats) => report.push(json!({
                "file": name, "ok": true,
                "transitions": stats.transitions, "rejected": stats.rejected,
            })),
            Err(divergence) => {
                failures.push(format!("{name}: {divergence}"));
                report.push(json!({"file": name, "ok": false, "divergence": divergence.to_json()}));
            },
        }
    }
    if let Some(output) = std::env::var_os("KAGG_PARITY_REPORT") {
        std::fs::write(
            &output,
            serde_json::to_string_pretty(&report).unwrap() + "\n",
        )
        .unwrap_or_else(|error| panic!("KAGG_PARITY_REPORT={output:?}: {error}"));
    }
    eprintln!(
        "replayed {} traces, {} divergent",
        paths.len(),
        failures.len()
    );
    assert!(
        failures.is_empty(),
        "parity divergences:\n{}",
        failures.join("\n")
    );
}
