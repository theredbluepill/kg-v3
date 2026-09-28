//! Saved Python oracle on synthetic observation sequences: no game transitions.
use kaggriculture_engine::{Game, TraceHeader, native_agents::shoprouter::ShopRouterController};
use serde_json::{Value, json};

fn header_for(fixture: &Value, row: &Value) -> TraceHeader {
    let mut header = fixture["template"].clone();
    let seat = row["seat"].as_u64().unwrap_or(0) as usize;
    let step = row["step"].as_u64().unwrap();
    let positions = row.get("positions").unwrap_or(&fixture["positions"]);
    let public = &mut header["initial"]["public"];
    public["step"] = json!(step);
    public["day"] = json!(step / 24);
    public["hour"] = json!(step % 24);
    public["town"]["unlocked_shops"] = row.get("shops").cloned().unwrap_or(json!([]));
    let farm = &mut public["farms"][seat];
    farm["tiles"] = json!(vec![vec![Value::Null; 10]; 10]);
    farm["farmer"] = positions[0].clone();
    farm["hands"] = json!(&positions.as_array().unwrap()[1..]);
    if let Some(workers) = row.get("weed_workers").and_then(Value::as_array) {
        for worker in workers {
            let position = &positions[worker.as_u64().unwrap() as usize];
            farm["tiles"][position[1].as_u64().unwrap() as usize]
                [position[0].as_u64().unwrap() as usize] = json!({"kind": "WEED"});
        }
    }
    if let Some(prices) = row.get("prices").and_then(Value::as_object) {
        for (item, price) in prices {
            public["market"]["prices"][item] = price.clone();
        }
    }
    let private = &mut header["initial"]["privates"][seat];
    private["shed"] = row.get("shed").cloned().unwrap_or(json!({}));
    private["inventories"] = row
        .get("inventories")
        .cloned()
        .unwrap_or_else(|| json!(vec![json!({}); positions.as_array().unwrap().len()]));
    serde_json::from_value(header).expect("valid explicit synthetic header")
}

#[test]
fn frozen_python_matches_stateful_synthetic_sequences() {
    let fixture: Value =
        serde_json::from_str(include_str!("../fixtures/shop-router-0909-parity.json")).unwrap();
    assert_eq!(fixture["engine_transitions"], 0);
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(
        cases.len(),
        fixture["case_count"].as_u64().unwrap() as usize
    );
    assert_eq!(
        cases
            .iter()
            .filter(|c| c["name"].as_str().unwrap().starts_with("route-"))
            .count(),
        64
    );
    let mut count = 0;
    for case in cases {
        let mut controller = ShopRouterController::default();
        for (index, row) in case["rows"].as_array().unwrap().iter().enumerate() {
            let header = header_for(&fixture, row);
            let game = Game::from_header(&header).unwrap();
            let seat = row["seat"].as_u64().unwrap_or(0) as usize;
            let action = controller.action(&game, seat).unwrap();
            assert_eq!(
                action, row["expected"],
                "case {} row {} step {}",
                case["name"], index, row["step"]
            );
            assert_eq!(
                controller.debug()["plan"],
                row["expected_plan"],
                "case {} row {}",
                case["name"],
                index
            );
            assert_eq!(
                controller.debug()["advanced_sales"],
                row["expected_advanced_sales"],
                "advanced sales case {} row {}",
                case["name"],
                index
            );
            count += 1;
        }
    }
    assert_eq!(count, fixture["row_count"].as_u64().unwrap() as usize);
}
