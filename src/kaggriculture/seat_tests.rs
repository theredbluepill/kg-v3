//! Task 7.4 seat binding: one live seat's view through the training `write_seat`.
use kaggriculture_engine::Config;
use serde_json::{json, Value};

use super::tests::row_bytes;
use super::{
    write_env, ObsBuffersMut, ObsRowMut, ObsStaging, ObservationGame, ObserveErrorKind,
    PreparedObservation, Seat,
};

/// Owned storage for exactly one seat row, poisoned so every write is visible.
struct SingleRow {
    tile_kind: Vec<i64>,
    tile_crop: Vec<i64>,
    tile_animal: Vec<i64>,
    tile_cell: Vec<i64>,
    tile_role: Vec<i64>,
    tiles_int: Vec<i64>,
    tiles_float: Vec<f32>,
    actor_slot: Vec<i64>,
    actor_cell: Vec<i64>,
    actor_role: Vec<i64>,
    actor_mask: Vec<bool>,
    actor_inventory: Vec<i64>,
    actor_inventory_rank: Vec<i64>,
    actors_float: Vec<f32>,
    player_features: Vec<f32>,
    storage_counts: Vec<i64>,
    storage_rank: Vec<i64>,
    banks: Vec<f64>,
    shop_type: Vec<i64>,
    shop_slot: Vec<i64>,
    shop_mask: Vec<bool>,
    market_product: Vec<i64>,
    market_float: Vec<f32>,
    market_int: Vec<i64>,
    global_features: Vec<f32>,
    globals_int: Vec<i64>,
    still_playing: Vec<bool>,
    order_limits: Vec<i64>,
    can_act: Vec<bool>,
}

impl SingleRow {
    fn new() -> Self {
        Self {
            tile_kind: vec![-17; 200],
            tile_crop: vec![-17; 200],
            tile_animal: vec![-17; 200],
            tile_cell: vec![-17; 200],
            tile_role: vec![-17; 200],
            tiles_int: vec![-17; 1400],
            tiles_float: vec![-17.0; 3000],
            actor_slot: vec![-17; 482],
            actor_cell: vec![-17; 482],
            actor_role: vec![-17; 482],
            actor_mask: vec![true; 482],
            actor_inventory: vec![-17; 2892],
            actor_inventory_rank: vec![-17; 2892],
            actors_float: vec![-17.0; 12532],
            player_features: vec![-17.0; 88],
            storage_counts: vec![-17; 17],
            storage_rank: vec![-17; 12],
            banks: vec![-17.0; 2],
            shop_type: vec![-17; 8],
            shop_slot: vec![-17; 8],
            shop_mask: vec![true; 8],
            market_product: vec![-17; 9],
            market_float: vec![-17.0; 18],
            market_int: vec![-17; 18],
            global_features: vec![-17.0; 15],
            globals_int: vec![-17; 16],
            still_playing: vec![true; 1],
            order_limits: vec![-17; 1],
            can_act: vec![true; 252],
        }
    }

    fn borrow(&mut self) -> ObsBuffersMut<'_> {
        ObsBuffersMut {
            tile_kind: &mut self.tile_kind,
            tile_crop: &mut self.tile_crop,
            tile_animal: &mut self.tile_animal,
            tile_cell: &mut self.tile_cell,
            tile_role: &mut self.tile_role,
            tiles_int: &mut self.tiles_int,
            tiles_float: &mut self.tiles_float,
            actor_slot: &mut self.actor_slot,
            actor_cell: &mut self.actor_cell,
            actor_role: &mut self.actor_role,
            actor_mask: &mut self.actor_mask,
            actor_inventory: &mut self.actor_inventory,
            actor_inventory_rank: &mut self.actor_inventory_rank,
            actors_float: &mut self.actors_float,
            player_features: &mut self.player_features,
            storage_counts: &mut self.storage_counts,
            storage_rank: &mut self.storage_rank,
            banks: &mut self.banks,
            shop_type: &mut self.shop_type,
            shop_slot: &mut self.shop_slot,
            shop_mask: &mut self.shop_mask,
            market_product: &mut self.market_product,
            market_float: &mut self.market_float,
            market_int: &mut self.market_int,
            global_features: &mut self.global_features,
            globals_int: &mut self.globals_int,
            still_playing: &mut self.still_playing,
            order_limits: &mut self.order_limits,
            can_act: &mut self.can_act,
        }
    }

    fn bytes(&mut self) -> Vec<u8> {
        row_bytes(&ObsRowMut::from_single_row(self.borrow()).unwrap())
    }
}

fn seat_view(config: &Config, game: &ObservationGame, seat: usize) -> String {
    let snapshot = game.game().snapshot();
    json!({
        "configuration": config,
        "public": snapshot.public,
        "private": snapshot.privates[seat],
    })
    .to_string()
}

fn encode_seat(view: &str, seat: Seat) -> Result<Vec<u8>, super::ObserveError> {
    let prepared = PreparedObservation::from_seat_view(view, seat)?;
    let mut storage = SingleRow::new();
    {
        let mut row = ObsRowMut::from_single_row(storage.borrow())?;
        super::write_seat(&prepared, seat, &mut row);
    }
    Ok(storage.bytes())
}

fn two_seat_rows(game: &ObservationGame) -> [Vec<u8>; 2] {
    let prepared = game.prepare().unwrap();
    let mut staging = ObsStaging::new(1).unwrap();
    let mut buffers = staging.buffers_mut();
    let mut env = buffers.envs_mut().next().unwrap();
    write_env(&prepared, &mut env);
    [row_bytes(&env.seats[0]), row_bytes(&env.seats[1])]
}

/// Scripted legal-looking actions: hires, seed purchases, movement and planting,
/// so states carry hands, inventories, seeds and a non-empty rival private.
fn scripted_action(step: usize, seat: usize) -> Value {
    let moves = ["NORTH", "SOUTH", "EAST", "WEST", "PASS"];
    let unit = |offset: usize| json!([moves[(step + offset + seat) % moves.len()]]);
    let market = match (step + seat) % 6 {
        0 => json!([["HIRE"]]),
        1 => json!([["BUY_SEED", "WHEAT", 2], ["BUY_SEED", "CARROT", 1]]),
        2 => json!([["SELL", "WHEAT", 1]]),
        _ => json!([]),
    };
    let hands: Vec<Value> = (0..step.min(4)).map(unit).collect();
    let farmer = if step % 7 == 3 {
        json!(["PLANT", "WHEAT"])
    } else {
        unit(0)
    };
    json!({"farmer": farmer, "hands": hands, "market": market})
}

#[test]
fn seat_row_equals_two_seat_row_bytewise_over_a_played_game() {
    for (config, seed) in [
        (Config::default(), "7"),
        (
            Config {
                max_market_orders_per_turn: 3_i64.into(),
                turns_per_day: 12_i64.into(),
                farm_hand_cost_mult: 2_i64.into(),
                ..Config::default()
            },
            "20260930",
        ),
    ] {
        let mut game = ObservationGame::from_seed(config.clone(), seed).unwrap();
        let mut rival_private_seen = [false, false];
        for step in 0..120 {
            let expected = two_seat_rows(&game);
            for (seat_index, seat) in [Seat::Zero, Seat::One].into_iter().enumerate() {
                let view = seat_view(&config, &game, seat_index);
                let actual = encode_seat(&view, seat).unwrap();
                assert_eq!(
                    actual, expected[seat_index],
                    "seed {seed} step {step} seat {seat_index}"
                );
                let rival = &game.game().snapshot().privates[1 - seat_index];
                rival_private_seen[seat_index] |=
                    rival.seeds.values().any(|count| *count > 0) || rival.inventories.len() > 1;
            }
            let actions = [scripted_action(step, 0), scripted_action(step, 1)];
            game.step_with_market_metrics(&actions).unwrap();
        }
        // The own row matched while the rival held private state the view lacked.
        assert_eq!(rival_private_seen, [true, true], "seed {seed}");
    }
}

#[test]
fn seat_view_rejects_unknown_configuration_and_view_keys() {
    let config = Config::default();
    let game = ObservationGame::from_seed(config.clone(), "7").unwrap();
    let mut view: Value = serde_json::from_str(&seat_view(&config, &game, 0)).unwrap();
    // Kaggle's local loader injects __raw_path__; stripping it is the Python view's job.
    view["configuration"]["__raw_path__"] = json!("/tmp/main.py");
    let error = PreparedObservation::from_seat_view(&view.to_string(), Seat::Zero)
        .err()
        .unwrap();
    assert_eq!(error.kind, ObserveErrorKind::Config);
    assert_eq!(error.field, "__raw_path__");

    let mut view: Value = serde_json::from_str(&seat_view(&config, &game, 0)).unwrap();
    view["player"] = json!(0);
    let error = PreparedObservation::from_seat_view(&view.to_string(), Seat::Zero)
        .err()
        .unwrap();
    assert_eq!(error.field, "view.player");
}

#[test]
fn seat_view_rejects_missing_parts_bad_seats_and_non_objects() {
    let config = Config::default();
    let game = ObservationGame::from_seed(config.clone(), "7").unwrap();
    for key in ["configuration", "public", "private"] {
        let mut view: Value = serde_json::from_str(&seat_view(&config, &game, 1)).unwrap();
        view.as_object_mut().unwrap().remove(key);
        let error = PreparedObservation::from_seat_view(&view.to_string(), Seat::One)
            .err()
            .unwrap();
        assert_eq!(error.field, format!("view.{key}"));
    }
    for seat in [-1, 2, 7] {
        assert_eq!(Seat::from_index(seat).unwrap_err().field, "seat");
    }
    assert_eq!(Seat::from_index(1).unwrap(), Seat::One);
    let error = PreparedObservation::from_seat_view("[]", Seat::Zero)
        .err()
        .unwrap();
    assert_eq!(error.kind, ObserveErrorKind::Shape);
}

#[test]
fn seat_view_rejects_an_own_private_with_the_wrong_actor_count() {
    let config = Config::default();
    let game = ObservationGame::from_seed(config.clone(), "7").unwrap();
    let mut view: Value = serde_json::from_str(&seat_view(&config, &game, 0)).unwrap();
    view["private"]["inventories"] = json!([{}, {}]);
    let error = PreparedObservation::from_seat_view(&view.to_string(), Seat::Zero)
        .err()
        .unwrap();
    assert_eq!(error.field, "actor_inventory");
}

#[test]
fn single_row_rejects_wrong_lengths_before_any_write() {
    let mut storage = SingleRow::new();
    storage.can_act.push(true);
    let before = storage.can_act.clone();
    let error = ObsRowMut::from_single_row(storage.borrow()).err().unwrap();
    assert_eq!(error.kind, ObserveErrorKind::Shape);
    assert_eq!(error.field, "action_mask.can_act");
    assert_eq!(storage.can_act, before);

    let mut storage = SingleRow::new();
    storage.tiles_int.pop();
    let error = ObsRowMut::from_single_row(storage.borrow()).err().unwrap();
    assert_eq!(error.field, "tiles_int");
}
