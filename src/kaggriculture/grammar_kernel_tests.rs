//! Kernel acceptance and replay-state tests for the production grammar.
//!
//! Contract v4.1 moved these from the temporary `engine_rs/tests/grammar_kernel.rs`
//! bridge into root integration at the first production root -> engine edge.
//! Decoded programs execute on the byte-pinned `kaggriculture_engine` kernel.

use super::grammar::{decode, encode, plan, Slot, SLOTS, TOKENS_PER_SEAT};
use kaggriculture_engine::{Config, Game, TraceHeader, ECON_PASS};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

const UNIT_NAMES: [&str; 19] = [
    "NONE",
    "PASS",
    "NORTH",
    "SOUTH",
    "EAST",
    "WEST",
    "PICKUP",
    "PLACE",
    "PLANT",
    "WATER",
    "HARVEST",
    "DROP",
    "BUILD_COOP",
    "BUILD_PASTURE",
    "FEED",
    "FERTILIZE",
    "COLLECT_FERTILIZER",
    "CARE",
    "DIG",
];
const MARKET_NAMES: [&str; 8] = [
    "NONE",
    "HIRE",
    "BUY_LAND",
    "BUY_SEED",
    "BUY_PRODUCT",
    "BUY_ANIMAL",
    "SELL",
    "EMPTY",
];
const ITEM_NAMES: [&str; 13] = [
    "NONE",
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
    "GOOSE",
    "COW",
    "SHEEP",
];

fn read_jsonl(path: &Path) -> Vec<Value> {
    let output = Command::new("gzip").arg("-dc").arg(path).output().unwrap();
    assert!(
        output.status.success(),
        "{}: {}",
        path.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn engine_root() -> PathBuf {
    repo_root().join("engine_rs")
}

fn real_header() -> Value {
    read_jsonl(&engine_root().join("fixtures/episode-95324500.jsonl.gz")).remove(0)
}

fn oracle_records() -> Vec<Value> {
    read_jsonl(&repo_root().join("tests/fixtures/kaggriculture/grammar-v4-reference.jsonl.gz"))
        .into_iter()
        .skip(1)
        .collect()
}

fn explicit_game(header: &Value, seat: usize, actors: usize, money: f64) -> Game {
    let mut raw = header.clone();
    let position = raw["initial"]["public"]["farms"][seat]["farmer"].clone();
    raw["initial"]["public"]["farms"][seat]["hands"] = json!(vec![position; actors - 1]);
    raw["initial"]["public"]["farms"][seat]["money"] = json!(money);
    raw["initial"]["privates"][seat]["inventories"] = json!(vec![json!({}); actors]);
    Game::from_header(&serde_json::from_value::<TraceHeader>(raw).unwrap()).unwrap()
}

fn pass_action(actors: usize) -> Value {
    json!({"farmer": ["PASS"], "hands": vec![json!(["PASS"]); actors - 1], "market": []})
}

fn joint(seat: usize, action: Value) -> [Value; 2] {
    let mut actions = [pass_action(1), pass_action(1)];
    actions[seat] = action;
    actions
}

fn block(rows: &[[i64; SLOTS]]) -> Vec<i64> {
    let mut result = vec![0; TOKENS_PER_SEAT];
    for (row, frame) in rows.iter().enumerate() {
        result[row * SLOTS..(row + 1) * SLOTS].copy_from_slice(frame);
    }
    result
}

fn pass_row(actor: usize) -> [i64; SLOTS] {
    let mut row = [0; SLOTS];
    row[0] = i64::try_from(actor).unwrap();
    row[1] = 1;
    row
}

fn market_row(kind: i64, item: i64, quantity: i64) -> [i64; SLOTS] {
    let mut row = [0; SLOTS];
    row[7] = kind;
    row[8] = item;
    row[9] = quantity / 32;
    row[10] = quantity % 32;
    row
}

fn stop_row() -> [i64; SLOTS] {
    let mut row = [0; SLOTS];
    row[11] = 1;
    row
}

fn assert_order(actual: &Value, expected: &Value, context: &str) {
    match (actual, expected) {
        (Value::Object(a), Value::Object(e)) => {
            assert_eq!(
                a.keys().collect::<Vec<_>>(),
                e.keys().collect::<Vec<_>>(),
                "map key order: {context}"
            );
            for (key, value) in a {
                assert_order(value, &e[key], &format!("{context}.{key}"));
            }
        },
        (Value::Array(a), Value::Array(e)) => {
            for (index, (value, other)) in a.iter().zip(e).enumerate() {
                assert_order(value, other, &format!("{context}[{index}]"));
            }
        },
        _ => {},
    }
}

fn assert_games(actual: &Game, expected: &Game, context: &str) {
    // Full snapshots include public/private state, statuses, rewards, done and step.
    let a = serde_json::to_value(actual.snapshot()).unwrap();
    let e = serde_json::to_value(expected.snapshot()).unwrap();
    assert_eq!(a, e, "snapshot: {context}");
    assert_order(&a, &e, context);
    assert_eq!(
        actual.econ_counters(),
        expected.econ_counters(),
        "economic counters: {context}"
    );
    assert_eq!(
        actual.attrib_counters(),
        expected.attrib_counters(),
        "attribution counters: {context}"
    );
    assert_eq!(
        actual.terminal_banks(),
        expected.terminal_banks(),
        "terminal banks: {context}"
    );
}

fn compare_step(
    game: &Game,
    seat: usize,
    decoded: &Value,
    expected: &Value,
    context: &str,
) -> Game {
    assert_eq!(decoded, expected, "canonical action: {context}");
    let mut actual_game = game.clone();
    let mut expected_game = game.clone();
    actual_game.step(&joint(seat, decoded.clone())).unwrap();
    expected_game.step(&joint(seat, expected.clone())).unwrap();
    assert_games(&actual_game, &expected_game, context);
    actual_game
}

#[test]
fn decoded_programs_feed_kernel() {
    let p = plan(1, 10, 241).unwrap();
    let tokens = block(&[pass_row(0), market_row(2, 0, 0), stop_row()]);
    for seat in 0..2 {
        let action = decode(&p, &tokens, 3).unwrap();
        let mut game = Game::new(Config::default(), 73, 2).unwrap();
        game.step(&joint(seat, action.clone())).unwrap();
        assert_eq!(game.public_state().step, 1);
        // Deliberately before JSON equality: BUY_LAND -> HIRE renderer control
        // must fail on the economically meaningful bank effect (2999 vs 2000).
        assert_eq!(game.public_state().farms[seat].money, 2000.0);
        assert_eq!(
            action,
            json!({"farmer":["PASS"],"hands":[],"market":[["BUY_LAND"]]})
        );
        assert_eq!(
            game.public_state().farms[seat].unlocked_quadrants,
            ["NW", "NE"]
        );
        assert_eq!(game.public_state().farms[1 - seat].money, 3000.0);
    }
}

#[test]
fn decoded_dense_241_and_hire_capacity() {
    let header = real_header();
    for seat in 0..2 {
        let dense = explicit_game(&header, seat, 241, 3000.0);
        let mut rows: Vec<_> = (0..241).map(pass_row).collect();
        rows.extend([market_row(7, 0, 0); 10]);
        rows.push(stop_row());
        assert_eq!(rows.len(), 252);
        let p = plan(241, 10, 241).unwrap();
        let action = decode(&p, &block(&rows), 252).unwrap();
        let mut expected = pass_action(241);
        expected["market"] = json!(vec![json!([]); 10]);
        let after = compare_step(&dense, seat, &action, &expected, "dense 241/10");
        assert_eq!(after.public_state().farms[seat].hands.len(), 240);
        assert_eq!(after.snapshot().privates[seat].inventories.len(), 241);
        assert_eq!(after.econ_counters().unwrap()[seat][ECON_PASS], 241);
        assert_eq!(after.public_state().step, 1);

        let almost = explicit_game(&header, seat, 240, 3000.0);
        let mut rows: Vec<_> = (0..240).map(pass_row).collect();
        rows.extend([market_row(1, 0, 0), stop_row()]);
        let p = plan(240, 10, 241).unwrap();
        let action = decode(&p, &block(&rows), 242).unwrap();
        let mut expected = pass_action(240);
        expected["market"] = json!([["HIRE"]]);
        let hired = compare_step(&almost, seat, &action, &expected, "240 to 241");
        assert_eq!(hired.public_state().farms[seat].hands.len() + 1, 241);
        assert_eq!(hired.snapshot().privates[seat].inventories.len(), 241);
        assert_eq!(hired.public_state().farms[seat].money, 2999.0);
        assert_eq!(hired.public_state().farms[seat].hires_today, 1);
        expected["market"] = json!([["HIRE"], ["HIRE"]]);
        let mut output = vec![-99; TOKENS_PER_SEAT];
        assert!(encode(&p, &expected, &mut output)
            .unwrap_err()
            .starts_with("hire capacity"));
        assert_eq!(output, vec![-99; TOKENS_PER_SEAT]);

        let exhausted = plan(241, 10, 241).unwrap();
        let mut cursor = Some(exhausted.start());
        // Ten EMPTY positions then the forced sentinel: HIRE is absent at each.
        let mut rows: Vec<_> = (0..241).map(pass_row).collect();
        rows.extend([market_row(7, 0, 0); 10]);
        rows.push(stop_row());
        let mut market_positions = 0;
        for (frame, row) in rows.iter().enumerate() {
            for &token in row {
                let state = cursor.unwrap();
                if frame >= 241 && state.slot() == Slot::MarketKind {
                    let mut mask = [false; 8];
                    state.write_mask(&exhausted, &mut mask).unwrap();
                    assert!(!mask[1]);
                    market_positions += 1;
                }
                cursor = state.advance(token, &exhausted).unwrap();
            }
        }
        assert_eq!(market_positions, 11);
        assert!(cursor.is_none());
    }
}

#[test]
fn decoded_actor_ordinals_match_engine_units() {
    let header = real_header();
    for seat in 0..2 {
        let game = explicit_game(&header, seat, 4, 3000.0);
        let mut rows: Vec<_> = (0..4).map(pass_row).collect();
        for (row, kind) in rows.iter_mut().zip([2, 3, 4, 5]) {
            row[1] = kind;
        }
        rows.push(stop_row());
        let action = decode(&plan(4, 10, 241).unwrap(), &block(&rows), 5).unwrap();
        let expected =
            json!({"farmer":["NORTH"],"hands":[["SOUTH"],["EAST"],["WEST"]],"market":[]});
        let after = compare_step(&game, seat, &action, &expected, "actor ordinal");
        let farm = &after.public_state().farms[seat];
        assert_eq!(farm.farmer, [4, 3]);
        assert_eq!(farm.hands, [vec![4, 5], vec![5, 4], vec![3, 4]]);
    }
}

#[test]
fn decoded_hire_masks_ignore_cash() {
    let header = real_header();
    let p = plan(1, 10, 241).unwrap();
    let tokens = block(&[
        pass_row(0),
        market_row(1, 0, 0),
        market_row(7, 0, 0),
        market_row(1, 0, 0),
        stop_row(),
    ]);
    let expected = json!({"farmer":["PASS"],"hands":[],"market":[["HIRE"],[],["HIRE"]]});
    for seat in 0..2 {
        let rich = explicit_game(&header, seat, 1, 3000.0);
        let poor = explicit_game(&header, seat, 1, 0.0);
        let rich_plan = plan(
            i64::try_from(rich.public_state().farms[seat].hands.len() + 1).unwrap(),
            10,
            241,
        )
        .unwrap();
        let poor_plan = plan(
            i64::try_from(poor.public_state().farms[seat].hands.len() + 1).unwrap(),
            10,
            241,
        )
        .unwrap();
        assert_eq!(rich_plan, poor_plan);
        let mut rich_state = Some(rich_plan.start());
        let mut poor_state = Some(poor_plan.start());
        for &token in &tokens[..5 * SLOTS] {
            let r = rich_state.unwrap();
            let q = poor_state.unwrap();
            let mut rm = vec![false; r.slot().width()];
            let mut qm = vec![false; q.slot().width()];
            r.write_mask(&rich_plan, &mut rm).unwrap();
            q.write_mask(&poor_plan, &mut qm).unwrap();
            assert_eq!(rm, qm);
            rich_state = r.advance(token, &rich_plan).unwrap();
            poor_state = q.advance(token, &poor_plan).unwrap();
        }
        let action = decode(&p, &tokens, 5).unwrap();
        let rich_after = compare_step(&rich, seat, &action, &expected, "affordable HIRE");
        let poor_after = compare_step(&poor, seat, &action, &expected, "unaffordable HIRE");
        assert_eq!(rich_after.public_state().farms[seat].hands.len(), 2);
        assert_eq!(rich_after.public_state().farms[seat].money, 2998.0);
        assert_eq!(poor_after.public_state().farms[seat].hands.len(), 0);
        assert_eq!(poor_after.public_state().farms[seat].money, 0.0);
    }
}

#[test]
fn decoded_command_matrices_execute_both_seats() {
    let p = plan(1, 10, 241).unwrap();
    for seat in 0..2 {
        let game = Game::new(Config::default(), 73, 2).unwrap();
        let mut unit_count = 0;
        for (kind, &name) in UNIT_NAMES.iter().enumerate().skip(1) {
            let items: Vec<_> = match kind {
                6 | 7 => (1..=12).collect(),
                8 => (1..=5).collect(),
                _ => vec![0],
            };
            let quantities = if matches!(kind, 6 | 7) {
                vec![None, Some(1), Some(31), Some(32), Some(1023)]
            } else {
                vec![None]
            };
            for item in items {
                for &quantity in &quantities {
                    let mut row = [0; SLOTS];
                    row[1] = i64::try_from(kind).unwrap();
                    row[3] = i64::try_from(item).unwrap();
                    let mut command = vec![json!(name)];
                    if item > 0 {
                        command.push(json!(ITEM_NAMES[item]));
                    }
                    if let Some(q) = quantity {
                        row[4] = 1;
                        row[5] = q / 32;
                        row[6] = q % 32;
                        command.push(json!(q));
                    }
                    let action = decode(&p, &block(&[row, stop_row()]), 2).unwrap();
                    let expected = json!({"farmer":command,"hands":[],"market":[]});
                    let after = compare_step(&game, seat, &action, &expected, name);
                    match kind {
                        2 => assert_eq!(after.public_state().farms[seat].farmer, [4, 3]),
                        3 => assert_eq!(after.public_state().farms[seat].farmer, [4, 5]),
                        4 => assert_eq!(after.public_state().farms[seat].farmer, [5, 4]),
                        5 => assert_eq!(after.public_state().farms[seat].farmer, [3, 4]),
                        12 => assert_eq!(
                            after.public_state().farms[seat].tiles[4][4],
                            json!({"kind":"COOP"})
                        ),
                        13 => assert_eq!(
                            after.public_state().farms[seat].tiles[4][4],
                            json!({"kind":"PASTURE"})
                        ),
                        _ => {},
                    }
                    unit_count += 1;
                }
            }
        }
        assert_eq!(unit_count, 140);
        let mut market_count = 0;
        for (kind, &name) in MARKET_NAMES.iter().enumerate().skip(1) {
            let items: Vec<_> = match kind {
                3 => (1..=5).collect(),
                4 => vec![1, 9],
                5 => (10..=12).collect(),
                6 => (1..=9).collect(),
                _ => vec![0],
            };
            let quantities = if (3..=6).contains(&kind) {
                vec![0, 1, 31, 32, 1023]
            } else {
                vec![0]
            };
            for item in items {
                for &q in &quantities {
                    let command = if item > 0 {
                        json!([name, ITEM_NAMES[item], q])
                    } else if kind == 7 {
                        json!([])
                    } else {
                        json!([name])
                    };
                    let rows = [
                        pass_row(0),
                        market_row(7, 0, 0),
                        market_row(
                            i64::try_from(kind).unwrap(),
                            i64::try_from(item).unwrap(),
                            q,
                        ),
                        market_row(7, 0, 0),
                        stop_row(),
                    ];
                    let action = decode(&p, &block(&rows), 5).unwrap();
                    let expected = json!({"farmer":["PASS"],"hands":[],"market":[[],command,[]]});
                    let after = compare_step(&game, seat, &action, &expected, name);
                    if kind == 1 {
                        assert_eq!(after.public_state().farms[seat].hands.len(), 1);
                    }
                    if kind == 2 {
                        assert_eq!(after.public_state().farms[seat].money, 2000.0);
                    }
                    if (3..=6).contains(&kind) && q == 0 {
                        let mut pass_game = game.clone();
                        pass_game.step(&joint(seat, pass_action(1))).unwrap();
                        assert_eq!(
                            serde_json::to_value(after.snapshot()).unwrap(),
                            serde_json::to_value(pass_game.snapshot()).unwrap()
                        );
                    }
                    if (3..=5).contains(&kind) && q > 0 {
                        assert!(after.public_state().farms[seat].money < 3000.0);
                    }
                    market_count += 1;
                }
            }
        }
        assert_eq!(market_count, 98);
    }
}

#[test]
fn decoded_transfer_quantities_have_inventory_effects() {
    let header = real_header();
    let p = plan(1, 10, 241).unwrap();
    for seat in 0..2 {
        for (kind, name) in [(6, "PICKUP"), (7, "PLACE")] {
            for quantity in [None, Some(1), Some(31), Some(32), Some(1023)] {
                let mut raw = header.clone();
                raw["configuration"]["shedCapacity"] = json!(2048);
                if kind == 6 {
                    raw["initial"]["privates"][seat]["shed"]["WHEAT"] = json!(1023);
                } else {
                    raw["initial"]["privates"][seat]["inventories"][0] = json!({"WHEAT":1023});
                }
                let game = Game::from_header(&serde_json::from_value::<TraceHeader>(raw).unwrap())
                    .unwrap();
                let mut row = [0; SLOTS];
                row[1] = kind;
                row[3] = 1;
                let mut command = vec![json!(name), json!("WHEAT")];
                if let Some(q) = quantity {
                    row[4] = 1;
                    row[5] = q / 32;
                    row[6] = q % 32;
                    command.push(json!(q));
                }
                let expected = json!({"farmer":command,"hands":[],"market":[]});
                let action = decode(&p, &block(&[row, stop_row()]), 2).unwrap();
                let after = compare_step(&game, seat, &action, &expected, name);
                let amount = quantity.unwrap_or(1);
                let private = &after.snapshot().privates[seat];
                let carried = private.inventories[0].get("WHEAT").copied().unwrap_or(0);
                assert_eq!(
                    private.shed["WHEAT"],
                    if kind == 6 { 1023 - amount } else { amount }
                );
                assert_eq!(carried, if kind == 6 { amount } else { 1023 - amount });
            }
        }
    }
}

#[test]
fn decoded_sampled_oracle_programs_execute() {
    let header = real_header();
    let mut count = 0;
    let mut dense_full = 0;
    for record in oracle_records()
        .into_iter()
        .filter(|r| r["source"] == "sampled")
    {
        assert_eq!(record["v4_expected"]["accepted"], true);
        let shape = &record["shape"];
        let actors = shape["actors"].as_i64().unwrap();
        let p = plan(
            actors,
            shape["order_limit"].as_i64().unwrap(),
            shape["hire_limit"].as_i64().unwrap(),
        )
        .unwrap();
        let active: Vec<i64> = serde_json::from_value(record["tokens"].clone()).unwrap();
        let mut tokens = vec![0; TOKENS_PER_SEAT];
        tokens[..active.len()].copy_from_slice(&active);
        let length = record["length"].as_i64().unwrap();
        let action = decode(&p, &tokens, length).unwrap();
        for seat in 0..2 {
            let mut configured_header = header.clone();
            configured_header["configuration"]["maxMarketOrdersPerTurn"] =
                shape["order_limit"].clone();
            let game = explicit_game(
                &configured_header,
                seat,
                usize::try_from(actors).unwrap(),
                3000.0,
            );
            compare_step(
                &game,
                seat,
                &action,
                &record["v4_expected"]["action"],
                record["id"].as_str().unwrap(),
            );
        }
        if actors == 241 && length == 252 {
            dense_full += 1;
        }
        count += 1;
    }
    assert_eq!(count, 256);
    assert!(dense_full >= 16);
}

#[test]
fn decoded_replay_actions_preserve_selected_states() {
    let mut episodes: BTreeMap<u64, Vec<Value>> = BTreeMap::new();
    for record in oracle_records()
        .into_iter()
        .filter(|r| r["source"] == "replay")
    {
        assert_eq!(record["v4_expected"]["accepted"], true);
        episodes
            .entry(record["episode"].as_u64().unwrap())
            .or_default()
            .push(record);
    }
    assert_eq!(episodes.len(), 4);
    let mut count = 0;
    for (episode, records) in episodes {
        assert_eq!(records.len(), 16);
        let trace = read_jsonl(&engine_root().join(format!("fixtures/episode-{episode}.jsonl.gz")));
        let header: TraceHeader = serde_json::from_value(trace[0].clone()).unwrap();
        let mut game = Game::new(header.configuration, header.seed.as_i64().unwrap(), 2).unwrap();
        for transition in trace.iter().skip(1) {
            let step = transition["from_step"].as_u64().unwrap();
            assert_eq!(u64::try_from(game.public_state().step).unwrap(), step);
            let actions = transition["actions"].as_array().unwrap();
            for record in records
                .iter()
                .filter(|r| r["from_step"].as_u64().unwrap() == step)
            {
                let seat = usize::try_from(record["seat"].as_u64().unwrap()).unwrap();
                assert_eq!(record["raw_action"], actions[seat]);
                let shape = &record["shape"];
                assert_eq!(
                    shape["actors"].as_u64().unwrap(),
                    u64::try_from(game.public_state().farms[seat].hands.len() + 1).unwrap()
                );
                let p = plan(
                    shape["actors"].as_i64().unwrap(),
                    shape["order_limit"].as_i64().unwrap(),
                    shape["hire_limit"].as_i64().unwrap(),
                )
                .unwrap();
                let active: Vec<i64> = serde_json::from_value(record["tokens"].clone()).unwrap();
                let mut tokens = vec![0; TOKENS_PER_SEAT];
                tokens[..active.len()].copy_from_slice(&active);
                let decoded = decode(&p, &tokens, record["length"].as_i64().unwrap()).unwrap();
                assert_eq!(decoded, record["raw_action"]);
                let mut original = game.clone();
                let mut replaced = game.clone();
                original.step(actions).unwrap();
                let mut joint_decoded = actions.clone();
                joint_decoded[seat] = decoded;
                replaced.step(&joint_decoded).unwrap();
                assert_games(&replaced, &original, record["id"].as_str().unwrap());
                count += 1;
            }
            game.step(actions).unwrap();
        }
        assert!(game.snapshot().done);
        assert_eq!(game.terminal_banks().unwrap(), header.terminal_banks);
    }
    assert_eq!(count, 64);
}

#[test]
#[should_panic(expected = "map key order")]
fn kernel_comparator_rejects_nested_order_change() {
    let a: Value =
        serde_json::from_str(r#"{"public":{"market":{"prices":{"WHEAT":1,"MILK":2}}}}"#).unwrap();
    let e: Value =
        serde_json::from_str(r#"{"public":{"market":{"prices":{"MILK":2,"WHEAT":1}}}}"#).unwrap();
    assert_eq!(a, e);
    assert_order(&a, &e, "nested negative control");
}
