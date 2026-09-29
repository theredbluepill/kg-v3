//! Stateless, seed-only Kaggle replay adapter. No environment or seed allocator.
use std::fmt;

use kaggriculture_engine::{Config, Game, StepSnapshot, TraceHeader};
use serde_json::{json, Map, Number, Value};

use super::{grammar, ObservationConfig};

#[derive(Debug, Clone)]
pub struct Divergence {
    pub pointer: String,
    /// Zero-based transition; None denotes the initial state/header.
    pub transition: Option<usize>,
    pub message: String,
}
impl Divergence {
    fn new(pointer: impl Into<String>, message: impl Into<String>) -> Self {
        let pointer = pointer.into();
        let mut parts = pointer.split('/');
        parts.next();
        let transition = match parts.next() {
            Some("steps") => parts
                .next()
                .and_then(|p| p.parse::<usize>().ok())
                .and_then(|s| s.checked_sub(1)),
            Some("transitions") => parts.next().and_then(|p| p.parse().ok()),
            _ => None,
        };
        Self {
            pointer,
            transition,
            message: message.into(),
        }
    }
    fn at(mut self, transition: Option<usize>) -> Self {
        self.transition = transition;
        self
    }
}
impl fmt::Display for Divergence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} at {} (transition {:?})",
            self.message, self.pointer, self.transition
        )
    }
}
impl std::error::Error for Divergence {}
type Result<T> = std::result::Result<T, Divergence>;

fn object<'a>(v: &'a Value, p: &str) -> Result<&'a Map<String, Value>> {
    v.as_object()
        .ok_or_else(|| Divergence::new(p, "expected object"))
}
fn array<'a>(v: &'a Value, p: &str) -> Result<&'a Vec<Value>> {
    v.as_array()
        .ok_or_else(|| Divergence::new(p, "expected array"))
}
fn required<'a>(v: &'a Value, key: &str, p: &str) -> Result<&'a Value> {
    v.get(key)
        .ok_or_else(|| Divergence::new(format!("{p}/{key}"), "missing required field"))
}
fn nonnegative_integer(v: &Value, p: &str) -> Result<Number> {
    let n = v
        .as_number()
        .ok_or_else(|| Divergence::new(p, "expected exact nonnegative JSON integer"))?;
    if !n.to_string().bytes().all(|c| c.is_ascii_digit()) {
        return Err(Divergence::new(
            p,
            "expected exact nonnegative JSON integer (float typed seeds forbidden)",
        ));
    }
    Ok(n.clone())
}

/// Resolved public configuration and exact arbitrary-width seed; metadata never
/// enters model inputs. `specification` is supplied by the hash-verified loader.
#[derive(Debug, Clone)]
pub struct SeedHeader {
    pub seed: Number,
    pub configuration: Config,
    configuration_json: Value,
    pub provenance: Value,
    pub specification: Value,
    pub envelope: Value,
}
impl SeedHeader {
    pub fn parse(v: &Value) -> Result<Self> {
        let seed = nonnegative_integer(required(v, "seed", "")?, "/seed")?;
        let configuration_json = required(v, "configuration", "")?.clone();
        let configuration: Config = serde_json::from_value(configuration_json.clone())
            .map_err(|e| Divergence::new("/configuration", e.to_string()))?;
        ObservationConfig::new(&configuration)
            .map_err(|e| Divergence::new("/configuration", e.to_string()))?;
        if configuration_json.get("seed") != Some(&Value::Null) {
            return Err(Divergence::new(
                "/configuration/seed",
                "resolved configuration.seed must be null",
            ));
        }
        let provenance = required(v, "provenance", "")?.clone();
        for key in ["source", "engine"] {
            if provenance[key].as_str().is_none_or(|s| s.is_empty()) {
                return Err(Divergence::new(
                    format!("/provenance/{key}"),
                    "expected nonempty source identity",
                ));
            }
        }
        if provenance["schema_version"] != 1 || provenance["target_framework_version"] != "1.32.7" {
            return Err(Divergence::new(
                "/provenance",
                "expected schema 1, framework 1.32.7",
            ));
        }
        let specification = required(v, "specification", "")?.clone();
        let props = object(&specification["observation"], "/specification/observation")?;
        for key in ["step", "farms", "market", "town", "day", "hour"] {
            if props.get(key).and_then(|v| v.get("shared")) != Some(&Value::Bool(true)) {
                return Err(Divergence::new(
                    format!("/specification/observation/{key}/shared"),
                    "pinned shared field required",
                ));
            }
        }
        for key in ["player", "private"] {
            if !props.contains_key(key) || props[key]["shared"] == true {
                return Err(Divergence::new(
                    format!("/specification/observation/{key}"),
                    "seat-private field must not be shared",
                ));
            }
        }
        let envelope = required(v, "envelope", "")?.clone();
        for key in [
            "id",
            "name",
            "title",
            "description",
            "version",
            "module_version",
            "schema_version",
        ] {
            required(&envelope, key, "/envelope")?;
        }
        if envelope["name"] != "kaggriculture"
            || envelope["module_version"] != "1.32.7"
            || envelope["schema_version"] != 1
        {
            return Err(Divergence::new(
                "/envelope",
                "expected Kaggriculture schema 1, module 1.32.7",
            ));
        }
        Ok(Self {
            seed,
            configuration,
            configuration_json,
            provenance,
            specification,
            envelope,
        })
    }

    /// Required TraceHeader state/schedules are explicit inert placeholders, not
    /// captured answers. Only the two valid player counts are read by the kernel.
    pub fn trace_header(&self) -> Result<TraceHeader> {
        let farm = json!({"money":0.,"tiles":[],"farmer":[],"hands":[],"unlocked_quadrants":[],"hires_today":0});
        let private = json!({"shed":{},"seeds":{},"inventories":[]});
        serde_json::from_value(json!({
            "format":"kaggriculture-re-parity-v1", "seed":self.seed,
            "configuration":self.configuration_json,"shop_schedule":[],"rng_schedule":[],
            "initial":{"public":{"step":0,"day":0,"hour":0,"farms":[farm,farm],
                "market":{"inventory":{},"prices":{}},"town":{"unlocked_shops":[]}},
                "privates":[private,private]},"terminal_banks":[],"transitions":0
        }))
        .map_err(|e| Divergence::new("/header", e.to_string()))
    }
}

#[derive(Debug, Clone)]
pub struct ExecutedTokens {
    pub tokens: Vec<i64>,
    pub length: i64,
}
#[derive(Debug, Clone)]
pub struct TapeTransition {
    pub actions: Vec<Value>,
    pub tokens: Option<Vec<ExecutedTokens>>,
}
#[derive(Debug, Clone)]
pub struct ActionTape {
    pub complete: bool,
    pub transitions: Vec<TapeTransition>,
}
impl ActionTape {
    pub fn parse(v: &Value) -> Result<Self> {
        let complete = v["complete"]
            .as_bool()
            .ok_or_else(|| Divergence::new("/complete", "expected completion claim"))?;
        let mut transitions = Vec::new();
        for (t, row) in array(&v["transitions"], "/transitions")?.iter().enumerate() {
            let p = format!("/transitions/{t}");
            let actions = array(&row["actions"], &format!("{p}/actions"))?.clone();
            if actions.len() != 2 {
                return Err(Divergence::new(
                    format!("{p}/actions"),
                    "exactly two actions required",
                ));
            }
            for (seat, a) in actions.iter().enumerate() {
                let path = format!("{p}/actions/{seat}");
                let map = object(a, &path)?;
                if map.len() != 3
                    || ["farmer", "hands", "market"]
                        .iter()
                        .any(|k| !map.contains_key(*k))
                {
                    return Err(Divergence::new(
                        path,
                        "expected exactly farmer/hands/market action keys",
                    ));
                }
                for key in ["farmer", "hands", "market"] {
                    array(&a[key], &format!("{path}/{key}"))?;
                }
            }
            let tokens = row
                .get("tokens")
                .map(|v| -> Result<Vec<ExecutedTokens>> {
                    let seats = array(v, &format!("{p}/tokens"))?;
                    if seats.len() != 2 {
                        return Err(Divergence::new(
                            format!("{p}/tokens"),
                            "exactly two token rows required",
                        ));
                    }
                    seats
                        .iter()
                        .enumerate()
                        .map(|(seat, v)| {
                            let path = format!("{p}/tokens/{seat}");
                            let tokens = array(&v["tokens"], &path)?
                                .iter()
                                .map(|v| {
                                    v.as_i64().ok_or_else(|| {
                                        Divergence::new(&path, "expected int64 token")
                                    })
                                })
                                .collect::<Result<Vec<_>>>()?;
                            let length = v["length"]
                                .as_i64()
                                .ok_or_else(|| Divergence::new(&path, "expected int64 length"))?;
                            Ok(ExecutedTokens { tokens, length })
                        })
                        .collect()
                })
                .transpose()?;
            transitions.push(TapeTransition { actions, tokens });
        }
        Ok(Self {
            complete,
            transitions,
        })
    }
}

pub fn replay_from_seed(header: &SeedHeader, tape: &ActionTape) -> Result<Vec<StepSnapshot>> {
    let mut game = Game::from_seed_header(&header.trace_header()?)
        .map_err(|e| Divergence::new("/header", e))?;
    let config = ObservationConfig::new(&header.configuration)
        .map_err(|e| Divergence::new("/configuration", e.to_string()))?;
    let mut snapshots = vec![game.snapshot()];
    for (t, row) in tape.transitions.iter().enumerate() {
        let p = format!("/transitions/{t}");
        let pre = &snapshots[t];
        if pre.done {
            return Err(Divergence::new(p, "steps after DONE"));
        }
        if row.actions.len() != 2 {
            return Err(Divergence::new(
                format!("{p}/actions"),
                "exactly two actions required",
            ));
        }
        if let Some(seats) = &row.tokens {
            if seats.len() != 2 {
                return Err(Divergence::new(
                    format!("{p}/tokens"),
                    "exactly two token rows required",
                ));
            }
            for (seat, token) in seats.iter().enumerate() {
                let actors = i64::try_from(pre.public.farms[seat].hands.len() + 1)
                    .map_err(|e| Divergence::new(&p, e.to_string()))?;
                let plan = grammar::plan(
                    actors,
                    config.orders,
                    config.turns_per_day * config.orders + 1,
                )
                .map_err(|e| Divergence::new(&p, e))?;
                let action = grammar::decode(&plan, &token.tokens, token.length)
                    .map_err(|e| Divergence::new(format!("{p}/tokens/{seat}"), e))?;
                if action != row.actions[seat] {
                    return Err(Divergence::new(
                        format!("{p}/actions/{seat}"),
                        "native decoded tokens differ from submitted action",
                    ));
                }
            }
        }
        game.step(&row.actions)
            .map_err(|e| Divergence::new(&p, format!("native step error: {e}")))?;
        let next = game.snapshot();
        if next.public.farms.iter().any(|f| f.hands.len() >= 241) {
            return Err(Divergence::new(p, "observed actor count exceeds 241"));
        }
        snapshots.push(next);
    }
    if tape.complete && !snapshots.last().expect("initial snapshot").done {
        return Err(Divergence::new(
            "/transitions",
            "tape shorter than game while completion claimed",
        )
        .at(tape.transitions.len().checked_sub(1)));
    }
    Ok(snapshots)
}

const EXPORTER: &str = "owl.kaggriculture.native-replay-v1";
const PUBLIC_FIELDS: [&str; 6] = ["step", "day", "hour", "farms", "market", "town"];

fn observation(header: &SeedHeader, snapshot: &StepSnapshot, seat: usize) -> Result<Value> {
    let public = serde_json::to_value(&snapshot.public)
        .map_err(|e| Divergence::new("/snapshot", e.to_string()))?;
    let private = serde_json::to_value(&snapshot.privates[seat])
        .map_err(|e| Divergence::new("/snapshot", e.to_string()))?;
    let props = object(
        &header.specification["observation"],
        "/specification/observation",
    )?;
    let mut obs = Map::new();
    // core.__get_state removes schema-shared defaults for non-first seats.
    // _initialize/_interpreter then append their public assignments to seat 1.
    for (key, schema) in props {
        if seat == 1 && schema["shared"] == true {
            continue;
        }
        let value = match key.as_str() {
            "player" => json!(seat),
            "private" => private.clone(),
            "remainingOverageTime" => required(
                schema,
                "default",
                "/specification/observation/remainingOverageTime",
            )?
            .clone(),
            key if PUBLIC_FIELDS.contains(&key) => public[key].clone(),
            _ => {
                return Err(Divergence::new(
                    format!("/specification/observation/{key}"),
                    "unsupported observation field",
                ))
            },
        };
        obs.insert(key.clone(), value);
    }
    if seat == 1 {
        // This assignment order is pinned kaggriculture.py:268-275, 952-956;
        // `step` is written only on seat 0 by core.__loop_through_interpreter.
        for key in ["farms", "market", "town", "day", "hour"] {
            obs.insert(key.into(), public[key].clone());
        }
    }
    Ok(Value::Object(obs))
}

pub fn export_kaggle_episode(header: &SeedHeader, tape: &ActionTape) -> Result<Value> {
    let snapshots = replay_from_seed(header, tape)?;
    export_snapshots(header, tape, &snapshots)
}
fn export_snapshots(
    header: &SeedHeader,
    tape: &ActionTape,
    snapshots: &[StepSnapshot],
) -> Result<Value> {
    let mut steps = Vec::with_capacity(snapshots.len());
    for (i, snapshot) in snapshots.iter().enumerate() {
        let mut seats = Vec::with_capacity(2);
        for seat in 0..2 {
            let action = if i == 0 {
                required(
                    &header.specification["action"],
                    "default",
                    "/specification/action",
                )?
                .clone()
            } else {
                tape.transitions[i - 1].actions[seat].clone()
            };
            seats.push(
                json!({"action":action,"reward":snapshot.rewards[seat],"info":{},
                "observation":observation(header,snapshot,seat)?,"status":snapshot.statuses[seat]}),
            );
        }
        steps.push(Value::Array(seats));
    }
    let mut result = Map::new();
    for key in [
        "id",
        "name",
        "title",
        "description",
        "version",
        "module_version",
    ] {
        result.insert(key.into(), header.envelope[key].clone());
    }
    result.insert("configuration".into(), header.configuration_json.clone());
    result.insert("specification".into(), header.specification.clone());
    result.insert("steps".into(), Value::Array(steps));
    let last = snapshots.last().expect("initial snapshot");
    result.insert("rewards".into(), json!(last.rewards));
    result.insert("statuses".into(), json!(last.statuses));
    result.insert("schema_version".into(), json!(1));
    result.insert("info".into(),json!({"seed":header.seed,"v3_native_replay":{
        "exporter":EXPORTER,"provenance":header.provenance,"complete":tape.complete,
        "execution":"produced by native replay, not by the Python framework",
        "trace_placeholders":"initial/schedules/terminal_banks/transitions are inert placeholders; only two player counts initialize seed replay"
    }}));
    Ok(Value::Object(result))
}

fn restored_observation(episode: &Value, index: usize, seat: usize) -> Result<Value> {
    let p = format!("/steps/{index}/{seat}/observation");
    let mut obs = object(&episode["steps"][index][seat]["observation"], &p)?.clone();
    if seat == 1 {
        for (key, schema) in object(
            &episode["specification"]["observation"],
            "/specification/observation",
        )? {
            if schema["shared"] == true && !obs.contains_key(key) {
                let value = required(
                    &episode["steps"][index][0]["observation"],
                    key,
                    &format!("/steps/{index}/0/observation"),
                )?;
                obs.insert(key.clone(), value.clone());
            }
        }
    }
    Ok(Value::Object(obs))
}

/// Strict completed-game import. Partial native exports carry an explicit
/// completion=false marker and must end ACTIVE; foreign partials are rejected.
pub fn import_kaggle_episode(episode: &Value) -> Result<(SeedHeader, ActionTape)> {
    object(episode, "")?;
    let native = &episode["info"]["v3_native_replay"];
    let own = native["exporter"] == EXPORTER;
    let complete = if own {
        native["complete"].as_bool().ok_or_else(|| {
            Divergence::new(
                "/info/v3_native_replay/complete",
                "missing completion claim",
            )
        })?
    } else {
        true
    };
    let mut envelope = Map::new();
    for key in [
        "id",
        "name",
        "title",
        "description",
        "version",
        "module_version",
        "schema_version",
    ] {
        envelope.insert(key.into(), required(episode, key, "")?.clone());
    }
    let provenance = if own {
        required(native, "provenance", "/info/v3_native_replay")?.clone()
    } else {
        json!({"source":"imported Kaggle episode", "engine":"kaggle-environments==1.32.7", "schema_version":1,"target_framework_version":"1.32.7"})
    };
    let seed = required(required(episode, "info", "")?, "seed", "/info")?;
    nonnegative_integer(seed, "/info/seed")?;
    let header = SeedHeader::parse(
        &json!({"seed":seed,"configuration":required(episode,"configuration", "")?,
        "provenance":provenance,"specification":required(episode,"specification", "")?,"envelope":envelope}),
    )?;
    let steps = array(&episode["steps"], "/steps")?;
    if steps.is_empty() {
        return Err(Divergence::new("/steps", "missing initial/terminal record"));
    }
    let mut transitions = Vec::new();
    for (i, step) in steps.iter().enumerate() {
        let seats = array(step, &format!("/steps/{i}"))?;
        if seats.len() != 2 {
            return Err(Divergence::new(
                format!("/steps/{i}"),
                "exactly two seats/actions required",
            ));
        }
        let mut actions = Vec::new();
        for (seat, row) in seats.iter().enumerate() {
            let p = format!("/steps/{i}/{seat}");
            let obs = restored_observation(episode, i, seat)?;
            if obs["step"].as_u64() != Some(i as u64) {
                return Err(Divergence::new(
                    format!("{p}/observation/step"),
                    "non-consecutive step",
                ));
            }
            if obs["player"].as_u64() != Some(seat as u64) {
                return Err(Divergence::new(
                    format!("{p}/observation/player"),
                    "wrong private seat",
                ));
            }
            object(
                required(&obs, "private", &format!("{p}/observation"))?,
                &format!("{p}/observation/private"),
            )?;
            // Parse the recorded public/private structures for admission only.
            // They never initialize a Game; replay still uses the seed header.
            let private: kaggriculture_engine::PrivateState =
                serde_json::from_value(obs["private"].clone()).map_err(|error| {
                    Divergence::new(format!("{p}/observation/private"), error.to_string())
                })?;
            let mut public = Map::new();
            for key in PUBLIC_FIELDS {
                public.insert(
                    key.into(),
                    required(&obs, key, &format!("{p}/observation"))?.clone(),
                );
            }
            let public: kaggriculture_engine::PublicState =
                serde_json::from_value(Value::Object(public)).map_err(|error| {
                    Divergence::new(format!("{p}/observation"), error.to_string())
                })?;
            if public.farms.len() != 2 {
                return Err(Divergence::new(
                    format!("{p}/observation/farms"),
                    "exactly two farms required",
                ));
            }
            if private.inventories.len() != public.farms[seat].hands.len() + 1 {
                return Err(Divergence::new(
                    format!("{p}/observation/private/inventories"),
                    "inventory count must equal actor count",
                ));
            }
            let status = if complete && i == steps.len() - 1 {
                "DONE"
            } else {
                "ACTIVE"
            };
            if row["status"] != status {
                return Err(Divergence::new(
                    format!("{p}/status"),
                    "malformed terminal record or steps inconsistent with statuses",
                ));
            }
            let reward = row["reward"]
                .as_f64()
                .filter(|v| v.is_finite())
                .ok_or_else(|| {
                    Divergence::new(format!("{p}/reward"), "raw reward must be finite")
                })?;
            if status == "DONE" && obs["farms"][seat]["money"].as_f64() != Some(reward) {
                return Err(Divergence::new(
                    format!("{p}/reward"),
                    "terminal raw reward differs from bank",
                ));
            }
            if status == "ACTIVE" && reward != 0. {
                return Err(Divergence::new(
                    format!("{p}/reward"),
                    "active raw reward must be zero",
                ));
            }
            actions.push(required(row, "action", &p)?.clone());
        }
        if i > 0 {
            transitions.push(json!({"actions":actions}));
        }
    }
    for (key, seat_key) in [("rewards", "reward"), ("statuses", "status")] {
        let expected = Value::Array(
            steps
                .last()
                .expect("nonempty")
                .as_array()
                .expect("validated")
                .iter()
                .map(|s| s[seat_key].clone())
                .collect(),
        );
        compare_tree(&episode[key], &expected, &format!("/{key}"))?;
    }
    let tape = ActionTape::parse(&json!({"complete":complete,"transitions":transitions}))?;
    let config = ObservationConfig::new(&header.configuration)
        .map_err(|e| Divergence::new("/configuration", e.to_string()))?;
    let required_transitions = usize::try_from((config.episode_steps - 1).max(1))
        .map_err(|e| Divergence::new("/configuration/episodeSteps", e.to_string()))?;
    if complete && tape.transitions.len() != required_transitions {
        return Err(Divergence::new(
            "/steps",
            "steps count inconsistent with terminal configuration/statuses",
        ));
    }
    if !complete && tape.transitions.len() >= required_transitions {
        return Err(Divergence::new(
            "/steps",
            "partial record reaches terminal game length",
        ));
    }
    Ok((header, tape))
}

fn child(p: &str, key: &str) -> String {
    format!("{p}/{}", key.replace('~', "~0").replace('/', "~1"))
}
// JSON integers remain exact, including when compared with a float spelling.
// Normalize decimal coefficient/exponent without ever converting to f64.
fn decimal(n: &Number) -> (String, num_bigint::BigInt) {
    let raw = n.to_string();
    let (mantissa, exponent) = raw
        .split_once(['e', 'E'])
        .map_or((raw.as_str(), num_bigint::BigInt::from(0)), |(m, e)| {
            (m, e.parse::<num_bigint::BigInt>().expect("JSON exponent"))
        });
    let fraction = mantissa.split_once('.').map_or(0, |(_, f)| f.len() as i64);
    let mut digits = mantissa
        .replace('.', "")
        .parse::<num_bigint::BigInt>()
        .expect("JSON mantissa")
        .to_string();
    let mut exponent = exponent - fraction;
    while digits.ends_with('0') {
        digits.pop();
        exponent += 1;
    }
    if digits.is_empty() || digits == "-" {
        return ("0".into(), num_bigint::BigInt::from(0));
    }
    (digits, exponent)
}
fn value_difference(actual: &Value, expected: &Value, p: &str) -> Option<Divergence> {
    match (actual, expected) {
        (Value::Object(a), Value::Object(e)) => {
            for (key, v) in e {
                let path = child(p, key);
                match a.get(key) {
                    Some(other) => {
                        if let Some(d) = value_difference(other, v, &path) {
                            return Some(d);
                        }
                    },
                    None => return Some(Divergence::new(path, "missing field")),
                }
            }
            a.keys()
                .find(|k| !e.contains_key(*k))
                .map(|k| Divergence::new(child(p, k), "extra field"))
        },
        (Value::Array(a), Value::Array(e)) => {
            for (i, (av, ev)) in a.iter().zip(e).enumerate() {
                if let Some(d) = value_difference(av, ev, &format!("{p}/{i}")) {
                    return Some(d);
                }
            }
            (a.len() != e.len()).then(|| Divergence::new(p, "array length differs"))
        },
        (Value::Number(a), Value::Number(e)) => (decimal(a) != decimal(e))
            .then(|| Divergence::new(p, format!("value differs: {a} != {e}"))),
        _ => (actual != expected)
            .then(|| Divergence::new(p, format!("value differs: {actual} != {expected}"))),
    }
}
fn order_difference(actual: &Value, expected: &Value, p: &str) -> Option<Divergence> {
    match (actual, expected) {
        (Value::Object(a), Value::Object(e)) => {
            if !a.keys().eq(e.keys()) {
                return Some(Divergence::new(p, "object key order differs"));
            }
            a.iter()
                .find_map(|(k, v)| order_difference(v, &e[k], &child(p, k)))
        },
        (Value::Array(a), Value::Array(e)) => a
            .iter()
            .zip(e)
            .enumerate()
            .find_map(|(i, (a, e))| order_difference(a, e, &format!("{p}/{i}"))),
        _ => None,
    }
}
pub fn compare_tree(actual: &Value, expected: &Value, p: &str) -> Result<()> {
    if let Some(d) =
        value_difference(actual, expected, p).or_else(|| order_difference(actual, expected, p))
    {
        Err(d)
    } else {
        Ok(())
    }
}

fn number_spelling_difference(actual: &Value, expected: &Value, p: &str) -> Option<Divergence> {
    match (actual, expected) {
        (Value::Object(a), Value::Object(e)) => a
            .iter()
            .find_map(|(key, value)| number_spelling_difference(value, &e[key], &child(p, key))),
        (Value::Array(a), Value::Array(e)) => a
            .iter()
            .zip(e)
            .enumerate()
            .find_map(|(i, (a, e))| number_spelling_difference(a, e, &format!("{p}/{i}"))),
        (Value::Number(a), Value::Number(e)) => {
            let actual = a.to_string();
            let expected = e.to_string();
            (actual != expected).then(|| Divergence::new(p, "canonical number bytes differ"))
        },
        _ => None,
    }
}

/// Compare separately captured native evidence, never manufactured re-export
/// evidence. Missing optional evidence is reported explicitly as uncovered.
pub fn compare_captured(snapshots: &[StepSnapshot], captured: &Value) -> Result<Value> {
    if snapshots.is_empty() {
        return Err(Divergence::new(
            "/captured",
            "replay lacks initial snapshot",
        ));
    }
    let map = object(captured, "/captured")?;
    for key in map.keys() {
        if !["initial", "terminal", "banks", "snapshots"].contains(&key.as_str()) {
            return Err(Divergence::new(
                child("/captured", key),
                "unknown evidence field",
            ));
        }
    }
    let value = |s: &StepSnapshot| {
        serde_json::to_value(s).map_err(|e| Divergence::new("/captured", e.to_string()))
    };
    if let Some(initial) = map.get("initial") {
        compare_tree(&value(&snapshots[0])?, initial, "/captured/initial")?;
    }
    if let Some(terminal) = map.get("terminal") {
        compare_tree(
            &value(snapshots.last().expect("initial"))?,
            terminal,
            "/captured/terminal",
        )
        .map_err(|e| e.at(snapshots.len().checked_sub(2)))?;
    }
    let mut banks_count = 0;
    if let Some(banks) = map.get("banks") {
        let banks = array(banks, "/captured/banks")?;
        if banks.len() != snapshots.len() - 1 {
            return Err(Divergence::new(
                "/captured/banks",
                "bank evidence count must equal transitions",
            ));
        }
        for (t, expected) in banks.iter().enumerate() {
            let actual = json!(snapshots[t + 1]
                .public
                .farms
                .iter()
                .map(|f| f.money)
                .collect::<Vec<_>>());
            compare_tree(&actual, expected, &format!("/captured/banks/{t}"))
                .map_err(|e| e.at(Some(t)))?;
        }
        banks_count = banks.len();
    }
    let mut full_count = 0;
    if let Some(full) = map.get("snapshots") {
        let rows = array(full, "/captured/snapshots")?;
        let mut seen = std::collections::HashSet::new();
        for (i, row) in rows.iter().enumerate() {
            let p = format!("/captured/snapshots/{i}");
            let t = row["transition"]
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .filter(|t| *t < snapshots.len() - 1)
                .ok_or_else(|| Divergence::new(&p, "transition outside replay"))?;
            if !seen.insert(t) {
                return Err(Divergence::new(p, "duplicate snapshot transition"));
            }
            compare_tree(
                &value(&snapshots[t + 1])?,
                required(row, "snapshot", &p)?,
                &format!("{p}/snapshot"),
            )
            .map_err(|e| e.at(Some(t)))?;
        }
        full_count = rows.len();
    }
    Ok(
        json!({"initial":map.contains_key("initial"),"terminal":map.contains_key("terminal"),"banks":banks_count,"snapshots":full_count}),
    )
}

/// Foreign comparison ignores only host/runtime fields: top-level info except
/// seed, per-seat info, remainingOverageTime, actTimeout/runTimeout. core.toJSON
/// copies info; core.__loop_through_interpreter charges measured agent duration
/// against remainingOverageTime. Config time budgets do not affect this kernel.
/// Schema-shared omission is normalized by restoring only marked properties.
fn semantic_view(episode: &Value) -> Result<Value> {
    let mut result = episode.clone();
    result["info"] = json!({"seed":episode["info"]["seed"]});
    if let Some(config) = result["configuration"].as_object_mut() {
        config.shift_remove("actTimeout");
        config.shift_remove("runTimeout");
    }
    for i in 0..array(&episode["steps"], "/steps")?.len() {
        for seat in 0..2 {
            let mut obs = restored_observation(episode, i, seat)?;
            let map = obs.as_object_mut().expect("restored object");
            map.shift_remove("remainingOverageTime");
            // Restored shared fields occupy schema order, not append order.
            // Payload object order stays completely untouched.
            let props = object(
                &episode["specification"]["observation"],
                "/specification/observation",
            )?;
            let mut ordered = Map::new();
            for key in props.keys() {
                if let Some(value) = map.shift_remove(key) {
                    ordered.insert(key.clone(), value);
                }
            }
            ordered.extend(std::mem::take(map));
            result["steps"][i][seat]["observation"] = Value::Object(ordered);
            result["steps"][i][seat]
                .as_object_mut()
                .expect("validated state")
                .shift_remove("info");
        }
    }
    Ok(result)
}

#[derive(Debug)]
pub struct Report {
    pub mode: &'static str,
    pub transitions: usize,
    pub canonical_json: String,
    pub captured: Value,
}

impl Report {
    pub fn to_json(&self) -> Value {
        json!({"ok":true,"mode":self.mode,"transitions":self.transitions,
            "canonical_json":self.canonical_json,"captured":self.captured})
    }
}

pub fn verify_round_trip(episode_json: &str, captured: Option<&Value>) -> Result<Report> {
    let episode: Value = serde_json::from_str(episode_json)
        .map_err(|e| Divergence::new("", format!("JSON parse: {e}")))?;
    let (header, tape) = import_kaggle_episode(&episode)?;
    let snapshots = replay_from_seed(&header, &tape)?;
    let replayed = export_snapshots(&header, &tape, &snapshots)?;
    let canonical_json = replayed.to_string();
    let own = episode["info"]["v3_native_replay"]["exporter"] == EXPORTER;
    if own {
        compare_tree(&replayed, &episode, "")?;
        let input_canonical = episode.to_string();
        if canonical_json.as_bytes() != input_canonical.as_bytes() {
            return Err(number_spelling_difference(&replayed, &episode, "")
                .unwrap_or_else(|| Divergence::new("", "canonical byte serialization differs")));
        }
    } else {
        compare_tree(&semantic_view(&replayed)?, &semantic_view(&episode)?, "")?;
    }
    let evidence = match captured {
        Some(c) => compare_captured(&snapshots, c)?,
        None => json!({"initial":false,"terminal":false,"banks":0,"snapshots":0}),
    };
    Ok(Report {
        mode: if own { "byte" } else { "semantic" },
        transitions: tape.transitions.len(),
        canonical_json,
        captured: evidence,
    })
}
