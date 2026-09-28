//! Fixed-state SELL/HOLD contract checks. No Game::new or Game::step.
use super::*;
use crate::{Config, TraceHeader, TRACE_FORMAT};

fn fixture(animal_location: &str, plant: bool, stock: i64) -> Game {
    let saved: Value = serde_json::from_str(include_str!(
        "../../fixtures/jat-animal-resupply.json"
    )).unwrap();
    let mut farm = saved["farm"].clone();
    let mut private = saved["private"].clone();
    farm["tiles"] = json!(vec![vec![Value::Null; 10]; 10]);
    private["shed"] = json!({"WHEAT": stock, "FERTILIZER": stock, "CARROT": 2,
        "UNKNOWN": 7});
    private["inventories"] = json!([{"WHEAT": 11, "FERTILIZER": 13}]);
    private["seeds"] = json!({"WHEAT": 17});
    match animal_location {
        "tile" => farm["tiles"][0][0] = json!({"kind": "COOP", "animal": "GOOSE"}),
        "shed" => private["shed"]["COW"] = json!(1),
        "inventory" => private["inventories"][0]["SHEEP"] = json!(1),
        "none" => (),
        _ => panic!("unknown fixture"),
    }
    if plant { farm["tiles"][0][1] = json!({"kind": "PLANT", "crop": "WHEAT"}); }
    let header: TraceHeader = serde_json::from_value(json!({
        "format": TRACE_FORMAT, "seed": 0, "configuration": Config::default(),
        "shop_schedule": [], "rng_schedule": [], "terminal_banks": [], "transitions": 0,
        "initial": {"public": {"step": 55, "day": 2, "hour": 7,
            "farms": [farm.clone(), farm], "market": {"inventory": {}, "prices": {}},
            "town": {"unlocked_shops": []}}, "privates": [private.clone(), private]}
    })).unwrap();
    Game::from_header(&header).unwrap()
}

#[test]
fn sell_intent_animals_and_plants_do_not_override_sell() {
    for location in ["none", "tile", "shed", "inventory"] {
        for plant in [false, true] {
            let game = fixture(location, plant, 5);
            let before = serde_json::to_value(game.snapshot()).unwrap();
            for seat in [0, 1] {
                assert_eq!(sweep_step(&game, seat, &[]).unwrap(), json!({
                    "verb": ["PASS"], "orders": [["SELL", "WHEAT", 5],
                        ["SELL", "CARROT", 2], ["SELL", "FERTILIZER", 5]],
                    "done": false, "reason": null
                }), "animal={location} plant={plant} seat={seat}");
            }
            assert_eq!(serde_json::to_value(game.snapshot()).unwrap(), before);
        }
    }
}

#[test]
fn sell_intent_hold_is_product_specific_and_read_only() {
    let game = fixture("tile", true, 5);
    let before = serde_json::to_value(game.snapshot()).unwrap();
    for (hold, expected) in [
        (vec!["WHEAT"], json!([["SELL", "CARROT", 2], ["SELL", "FERTILIZER", 5]])),
        (vec!["FERTILIZER"], json!([["SELL", "WHEAT", 5], ["SELL", "CARROT", 2]])),
        (vec!["WHEAT", "FERTILIZER"], json!([["SELL", "CARROT", 2]])),
        (vec!["WHEAT", "CARROT", "FERTILIZER"], json!([])),
    ] {
        let hold: Vec<String> = hold.into_iter().map(String::from).collect();
        assert_eq!(sweep_step(&game, 0, &hold).unwrap()["orders"], expected);
    }
    assert_eq!(serde_json::to_value(game.snapshot()).unwrap(), before);
}

#[test]
fn sell_intent_ignores_nonpositive_stock_carried_stock_seeds_and_nonproducts() {
    for stock in [-1, 0] {
        let game = fixture("shed", true, stock);
        assert_eq!(sweep_step(&game, 0, &[]).unwrap()["orders"], json!([["SELL", "CARROT", 2]]));
        assert!(sweep_step(&game, 2, &[]).is_err());
    }
}
