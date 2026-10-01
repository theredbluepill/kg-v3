use kaggriculture_engine::{Config, Game, InitialState, TraceHeader, TRACE_FORMAT};
use rayon::prelude::*;

use super::{
    encode_env as native_encode_env, ObsBuffersMut, ObsRowMut, ObsStaging, ObservationConfig,
    ObservationGame, ObserveError, ObserveErrorKind, Seat,
};

#[test]
fn snapshot_both_seats_acquire_once_and_keep_staging_allocations() {
    for dense in [false, true] {
        let header = super::oracle_corpus::timing_header(dense).unwrap();
        let game = ObservationGame::from_header(&header).unwrap();
        let mut output = ObsStaging::new(1).unwrap();
        let allocations = output.allocation_layout();
        let mut expected = None;
        for iteration in 0..4 {
            let before = super::observe::snapshot_acquisitions();
            {
                let mut buffers = output.buffers_mut();
                let mut env = buffers.envs_mut().next().unwrap();
                for row in &mut env.seats {
                    fill_row(row, iteration);
                }
                native_encode_env(&game, &mut env).unwrap();
                assert_eq!(super::observe::snapshot_acquisitions() - before, 1);
                for row in &env.seats {
                    super::check_row(row).unwrap();
                }
            }
            let bytes = all_bytes(&mut output);
            assert_eq!(bytes.len(), 282_246);
            if let Some(expected) = &expected {
                assert_eq!(&bytes, expected);
            } else {
                expected = Some(bytes);
            }
            assert_eq!(output.allocation_layout(), allocations);
        }
    }
}

fn measure_phase<T>(
    name: &str,
    output_bytes: usize,
    body_start: std::time::Instant,
    mut operation: impl FnMut() -> T,
) -> serde_json::Value {
    // The deadline check and black_box overhead are included in every phase.
    for _ in 0..20 {
        assert!(body_start.elapsed().as_secs_f64() < 120.0);
        drop(std::hint::black_box(operation()));
    }
    let start = std::time::Instant::now();
    for _ in 0..200 {
        assert!(body_start.elapsed().as_secs_f64() < 120.0);
        drop(std::hint::black_box(operation()));
    }
    let elapsed = start.elapsed().as_nanos();
    serde_json::json!({
        "phase": name, "repetitions": 200, "warmups": 20,
        "measured_nanoseconds": elapsed,
        "nanoseconds_per_environment": elapsed as f64 / 200.0,
        "nanoseconds_per_seat": elapsed as f64 / 400.0,
        "output_bytes_per_environment": output_bytes,
        "output_bytes_per_seat": output_bytes / 2,
    })
}

#[test]
#[ignore = "bounded optimized-only CPU access-cost diagnostic; no throughput claim"]
fn measure_observe_cost() {
    assert!(
        !std::hint::black_box(cfg!(debug_assertions)),
        "R4 requires an optimized release build"
    );
    let body_start = std::time::Instant::now();
    let mut measurements = Vec::new();
    for (dense, id) in [(false, "official:95324500:0"), (true, "dense:0")] {
        let header = super::oracle_corpus::timing_header(dense).unwrap();
        let game = ObservationGame::from_header(&header).unwrap();
        let snapshot = game.acquire_snapshot();
        let prepared = game.prepare().unwrap();
        let mut output = ObsStaging::new(1).unwrap();
        let allocations = output.allocation_layout();
        {
            let mut buffers = output.buffers_mut();
            let mut env = buffers.envs_mut().next().unwrap();
            native_encode_env(&game, &mut env).unwrap();
            for row in &env.seats {
                super::check_row(row).unwrap();
            }
        }
        let expected = all_bytes(&mut output);
        assert_eq!(expected.len(), 282_246);
        let mut phases = vec![
            measure_phase("snapshot", 0, body_start, || game.acquire_snapshot()),
            measure_phase("validate_existing_snapshot", 0, body_start, || {
                game.validate_existing_snapshot(std::hint::black_box(&snapshot))
                    .unwrap();
            }),
        ];
        {
            let mut buffers = output.buffers_mut();
            let mut env = buffers.envs_mut().next().unwrap();
            phases.push(measure_phase(
                "write_prepared_both_seats",
                expected.len(),
                body_start,
                || {
                    super::write_env(std::hint::black_box(&prepared), &mut env);
                },
            ));
            phases.push(measure_phase(
                "snapshot_validate_write_both_seats",
                expected.len(),
                body_start,
                || {
                    native_encode_env(std::hint::black_box(&game), &mut env).unwrap();
                },
            ));
            for row in &env.seats {
                super::check_row(row).unwrap();
            }
        }
        assert_eq!(all_bytes(&mut output), expected);
        assert_eq!(output.allocation_layout(), allocations);
        measurements.push(serde_json::json!({
            "input_id": id,
            "header_sha256": super::oracle_corpus::hash_bytes(&serde_json::to_vec(&header).unwrap()).unwrap(),
            "config_sha256": super::oracle_corpus::hash_bytes(&serde_json::to_vec(&header.configuration).unwrap()).unwrap(),
            "encoded_sha256": super::oracle_corpus::hash_bytes(&expected).unwrap(),
            "stable_encoded_bytes": true, "stable_output_allocations": true,
            "phases": phases,
        }));
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut source_hashes = serde_json::Map::new();
    for path in [
        "Cargo.toml",
        "Cargo.lock",
        "src/kaggriculture/mod.rs",
        "src/kaggriculture/config.rs",
        "src/kaggriculture/buffers.rs",
        "src/kaggriculture/observe.rs",
        "src/kaggriculture/tests.rs",
        "src/kaggriculture/oracle_corpus.rs",
        "engine_rs/src/lib.rs",
    ] {
        source_hashes.insert(
            path.into(),
            super::oracle_corpus::hash_bytes(&std::fs::read(root.join(path)).unwrap())
                .unwrap()
                .into(),
        );
    }
    let manifest = root.join("tests/fixtures/kaggriculture/observation-v3/manifest.json");
    let corpus_manifest_sha256 = manifest
        .exists()
        .then(|| super::oracle_corpus::hash_bytes(&std::fs::read(manifest).unwrap()).unwrap());
    println!(
        "OBSERVE_TIMING={}",
        serde_json::json!({
            "profile": "release", "fat_lto": true, "codegen_units": 1,
            "test_body_seconds": body_start.elapsed().as_secs_f64(),
            "environments_at_a_time": 1, "seats_per_environment": 2,
            "rust_test_threads": std::env::var("RUST_TEST_THREADS").unwrap(),
            "rayon_threads": std::env::var("RAYON_NUM_THREADS").unwrap(),
            "uses_rayon_in_measured_phases": false,
            "includes_deadline_black_box_and_test_counter_overhead": true,
            "corpus_manifest_sha256": corpus_manifest_sha256,
            "input_scope": "two reproducible producer recipes; full corpus qualification is separate",
            "source_sha256": source_hashes, "measurements": measurements,
        })
    );
}

fn fresh_header(config: Config) -> TraceHeader {
    let snapshot = Game::new(config.clone(), 42, 2).unwrap().snapshot();
    TraceHeader {
        format: TRACE_FORMAT.into(),
        seed: 42.into(),
        configuration: config,
        shop_schedule: vec![],
        rng_schedule: vec![],
        initial: InitialState {
            public: snapshot.public,
            privates: snapshot.privates,
        },
        terminal_banks: vec![],
        transitions: 0,
    }
}

// Deliberately independent owned test buffers with literal contract dimensions.
struct FlatBuffers {
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
impl FlatBuffers {
    fn new(n_envs: usize) -> Self {
        Self {
            tile_kind: vec![-17; n_envs * 2 * 200],
            tile_crop: vec![-17; n_envs * 2 * 200],
            tile_animal: vec![-17; n_envs * 2 * 200],
            tile_cell: vec![-17; n_envs * 2 * 200],
            tile_role: vec![-17; n_envs * 2 * 200],
            tiles_int: vec![-17; n_envs * 2 * 1400],
            tiles_float: vec![-17.0; n_envs * 2 * 3000],
            actor_slot: vec![-17; n_envs * 2 * 482],
            actor_cell: vec![-17; n_envs * 2 * 482],
            actor_role: vec![-17; n_envs * 2 * 482],
            actor_mask: vec![true; n_envs * 2 * 482],
            actor_inventory: vec![-17; n_envs * 2 * 2892],
            actor_inventory_rank: vec![-17; n_envs * 2 * 2892],
            actors_float: vec![-17.0; n_envs * 2 * 12532],
            player_features: vec![-17.0; n_envs * 2 * 88],
            storage_counts: vec![-17; n_envs * 2 * 17],
            storage_rank: vec![-17; n_envs * 2 * 12],
            banks: vec![-17.0; n_envs * 2 * 2],
            shop_type: vec![-17; n_envs * 2 * 8],
            shop_slot: vec![-17; n_envs * 2 * 8],
            shop_mask: vec![true; n_envs * 2 * 8],
            market_product: vec![-17; n_envs * 2 * 9],
            market_float: vec![-17.0; n_envs * 2 * 18],
            market_int: vec![-17; n_envs * 2 * 18],
            global_features: vec![-17.0; n_envs * 2 * 15],
            globals_int: vec![-17; n_envs * 2 * 16],
            still_playing: vec![true; n_envs * 2],
            order_limits: vec![-17; n_envs * 2],
            can_act: vec![true; n_envs * 2 * 252],
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
    fn bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        for value in &self.tile_kind {
            bytes.extend(value.to_le_bytes());
        }
        for value in &self.tile_crop {
            bytes.extend(value.to_le_bytes());
        }
        for value in &self.tile_animal {
            bytes.extend(value.to_le_bytes());
        }
        for value in &self.tile_cell {
            bytes.extend(value.to_le_bytes());
        }
        for value in &self.tile_role {
            bytes.extend(value.to_le_bytes());
        }
        for value in &self.tiles_int {
            bytes.extend(value.to_le_bytes());
        }
        for value in &self.tiles_float {
            bytes.extend(value.to_le_bytes());
        }
        for value in &self.actor_slot {
            bytes.extend(value.to_le_bytes());
        }
        for value in &self.actor_cell {
            bytes.extend(value.to_le_bytes());
        }
        for value in &self.actor_role {
            bytes.extend(value.to_le_bytes());
        }
        bytes.extend(self.actor_mask.iter().map(|v| u8::from(*v)));
        for value in &self.actor_inventory {
            bytes.extend(value.to_le_bytes());
        }
        for value in &self.actor_inventory_rank {
            bytes.extend(value.to_le_bytes());
        }
        for value in &self.actors_float {
            bytes.extend(value.to_le_bytes());
        }
        for value in &self.player_features {
            bytes.extend(value.to_le_bytes());
        }
        for value in &self.storage_counts {
            bytes.extend(value.to_le_bytes());
        }
        for value in &self.storage_rank {
            bytes.extend(value.to_le_bytes());
        }
        for value in &self.banks {
            bytes.extend(value.to_le_bytes());
        }
        for value in &self.shop_type {
            bytes.extend(value.to_le_bytes());
        }
        for value in &self.shop_slot {
            bytes.extend(value.to_le_bytes());
        }
        bytes.extend(self.shop_mask.iter().map(|v| u8::from(*v)));
        for value in &self.market_product {
            bytes.extend(value.to_le_bytes());
        }
        for value in &self.market_float {
            bytes.extend(value.to_le_bytes());
        }
        for value in &self.market_int {
            bytes.extend(value.to_le_bytes());
        }
        for value in &self.global_features {
            bytes.extend(value.to_le_bytes());
        }
        for value in &self.globals_int {
            bytes.extend(value.to_le_bytes());
        }
        bytes.extend(self.still_playing.iter().map(|v| u8::from(*v)));
        for value in &self.order_limits {
            bytes.extend(value.to_le_bytes());
        }
        bytes.extend(self.can_act.iter().map(|v| u8::from(*v)));
        bytes
    }
}

pub(super) fn row_bytes(row: &ObsRowMut<'_>) -> Vec<u8> {
    let mut bytes = Vec::new();
    for value in row.tile_kind.iter() {
        bytes.extend(value.to_le_bytes());
    }
    for value in row.tile_crop.iter() {
        bytes.extend(value.to_le_bytes());
    }
    for value in row.tile_animal.iter() {
        bytes.extend(value.to_le_bytes());
    }
    for value in row.tile_cell.iter() {
        bytes.extend(value.to_le_bytes());
    }
    for value in row.tile_role.iter() {
        bytes.extend(value.to_le_bytes());
    }
    for value in row.tiles_int.iter().flatten() {
        bytes.extend(value.to_le_bytes());
    }
    for value in row.tiles_float.iter().flatten() {
        bytes.extend(value.to_le_bytes());
    }
    for value in row.actor_slot.iter() {
        bytes.extend(value.to_le_bytes());
    }
    for value in row.actor_cell.iter() {
        bytes.extend(value.to_le_bytes());
    }
    for value in row.actor_role.iter() {
        bytes.extend(value.to_le_bytes());
    }
    bytes.extend(row.actor_mask.iter().map(|value| u8::from(*value)));
    for value in row.actor_inventory.iter().flatten() {
        bytes.extend(value.to_le_bytes());
    }
    for value in row.actor_inventory_rank.iter().flatten() {
        bytes.extend(value.to_le_bytes());
    }
    for value in row.actors_float.iter().flatten() {
        bytes.extend(value.to_le_bytes());
    }
    for value in row.player_features.iter().flatten() {
        bytes.extend(value.to_le_bytes());
    }
    for value in row.storage_counts.iter() {
        bytes.extend(value.to_le_bytes());
    }
    for value in row.storage_rank.iter() {
        bytes.extend(value.to_le_bytes());
    }
    for value in row.banks.iter() {
        bytes.extend(value.to_le_bytes());
    }
    for value in row.shop_type.iter() {
        bytes.extend(value.to_le_bytes());
    }
    for value in row.shop_slot.iter() {
        bytes.extend(value.to_le_bytes());
    }
    bytes.extend(row.shop_mask.iter().map(|value| u8::from(*value)));
    for value in row.market_product.iter() {
        bytes.extend(value.to_le_bytes());
    }
    for value in row.market_float.iter().flatten() {
        bytes.extend(value.to_le_bytes());
    }
    for value in row.market_int.iter().flatten() {
        bytes.extend(value.to_le_bytes());
    }
    for value in row.global_features.iter() {
        bytes.extend(value.to_le_bytes());
    }
    for value in row.globals_int.iter() {
        bytes.extend(value.to_le_bytes());
    }
    bytes.push(u8::from(*row.still_playing));
    bytes.extend(row.order_limits.to_le_bytes());
    bytes.extend(row.can_act.iter().map(|value| u8::from(*value)));
    bytes
}

fn field_bytes(out: &mut ObsStaging, seat: Seat) -> Vec<u8> {
    let mut buffers = out.buffers_mut();
    let env = buffers.envs_mut().next().unwrap();
    row_bytes(
        &env.seats[match seat {
            Seat::Zero => 0,
            Seat::One => 1,
        }],
    )
}

fn all_bytes(out: &mut ObsStaging) -> Vec<u8> {
    out.buffers_mut()
        .envs_mut()
        .flat_map(|env| env.seats.iter().flat_map(row_bytes).collect::<Vec<_>>())
        .collect()
}

fn fill_row(row: &mut ObsRowMut<'_>, tag: i64) {
    for (index, value) in row.tile_kind.iter_mut().enumerate() {
        *value = tag * 13000 + index as i64;
    }
    for (index, value) in row.tile_crop.iter_mut().enumerate() {
        *value = tag * 13000 + index as i64 + 1;
    }
    for (index, value) in row.tile_animal.iter_mut().enumerate() {
        *value = tag * 13000 + index as i64 + 2;
    }
    for (index, value) in row.tile_cell.iter_mut().enumerate() {
        *value = tag * 13000 + index as i64 + 3;
    }
    for (index, value) in row.tile_role.iter_mut().enumerate() {
        *value = tag * 13000 + index as i64 + 4;
    }
    for (index, value) in row.tiles_int.iter_mut().flatten().enumerate() {
        *value = tag * 13000 + index as i64 + 5;
    }
    for (index, value) in row.tiles_float.iter_mut().flatten().enumerate() {
        *value = (tag * 13000 + index as i64 + 6) as f32;
    }
    for (index, value) in row.actor_slot.iter_mut().enumerate() {
        *value = tag * 13000 + index as i64 + 7;
    }
    for (index, value) in row.actor_cell.iter_mut().enumerate() {
        *value = tag * 13000 + index as i64 + 8;
    }
    for (index, value) in row.actor_role.iter_mut().enumerate() {
        *value = tag * 13000 + index as i64 + 9;
    }
    for (index, value) in row.actor_mask.iter_mut().enumerate() {
        *value = (tag + index as i64 + 10) % 2 == 0;
    }
    for (index, value) in row.actor_inventory.iter_mut().flatten().enumerate() {
        *value = tag * 13000 + index as i64 + 11;
    }
    for (index, value) in row.actor_inventory_rank.iter_mut().flatten().enumerate() {
        *value = tag * 13000 + index as i64 + 12;
    }
    for (index, value) in row.actors_float.iter_mut().flatten().enumerate() {
        *value = (tag * 13000 + index as i64 + 13) as f32;
    }
    for (index, value) in row.player_features.iter_mut().flatten().enumerate() {
        *value = (tag * 13000 + index as i64 + 14) as f32;
    }
    for (index, value) in row.storage_counts.iter_mut().enumerate() {
        *value = tag * 13000 + index as i64 + 15;
    }
    for (index, value) in row.storage_rank.iter_mut().enumerate() {
        *value = tag * 13000 + index as i64 + 16;
    }
    for (index, value) in row.banks.iter_mut().enumerate() {
        *value = (tag * 13000 + index as i64 + 17) as f64;
    }
    for (index, value) in row.shop_type.iter_mut().enumerate() {
        *value = tag * 13000 + index as i64 + 18;
    }
    for (index, value) in row.shop_slot.iter_mut().enumerate() {
        *value = tag * 13000 + index as i64 + 19;
    }
    for (index, value) in row.shop_mask.iter_mut().enumerate() {
        *value = (tag + index as i64 + 20) % 2 == 0;
    }
    for (index, value) in row.market_product.iter_mut().enumerate() {
        *value = tag * 13000 + index as i64 + 21;
    }
    for (index, value) in row.market_float.iter_mut().flatten().enumerate() {
        *value = (tag * 13000 + index as i64 + 22) as f32;
    }
    for (index, value) in row.market_int.iter_mut().flatten().enumerate() {
        *value = tag * 13000 + index as i64 + 23;
    }
    for (index, value) in row.global_features.iter_mut().enumerate() {
        *value = (tag * 13000 + index as i64 + 24) as f32;
    }
    for (index, value) in row.globals_int.iter_mut().enumerate() {
        *value = tag * 13000 + index as i64 + 25;
    }
    *row.still_playing = (tag + 26) % 2 == 0;
    *row.order_limits = tag * 13000 + 27;
    for (index, value) in row.can_act.iter_mut().enumerate() {
        *value = (tag + index as i64 + 28) % 2 == 0;
    }
}

#[test]
fn shape_product_overflow_is_an_error() {
    assert!(matches!(
        ObsStaging::new(usize::MAX),
        Err(ObserveError {
            kind: ObserveErrorKind::Shape,
            ..
        })
    ));
}

#[test]
fn shape_byte_capacity_overflow_is_an_error() {
    // Every element count fits usize; the largest f32 vector's bytes do not
    // fit the maximum Rust allocation. This must reject before ANY allocation.
    let n_envs = (isize::MAX as usize) / (2 * 482 * 26 * size_of::<f32>()) + 1;
    assert!(matches!(
        ObsStaging::new(n_envs),
        Err(ObserveError {
            kind: ObserveErrorKind::Shape,
            ..
        })
    ));
}

#[test]
fn shape_zero_environments_is_an_error() {
    assert!(matches!(
        ObsStaging::new(0),
        Err(ObserveError {
            kind: ObserveErrorKind::Shape,
            ..
        })
    ));
    let mut flat = FlatBuffers::new(0);
    assert!(matches!(
        flat.borrow().validate(0),
        Err(ObserveError {
            kind: ObserveErrorKind::Shape,
            ..
        })
    ));
}

#[test]
fn shape_every_field_rejects_short_and_long_without_writes() {
    macro_rules! check_field {
        ($field:ident, $name:literal, $extra:expr) => {
            for long in [false, true] {
                let mut flat = FlatBuffers::new(1);
                if long {
                    flat.$field.push($extra);
                } else {
                    flat.$field.pop();
                }
                let before = flat.bytes();
                let Err(error) = flat.borrow().validate(1) else {
                    panic!("accepted {} long={long}", $name);
                };
                assert_eq!(error.kind, ObserveErrorKind::Shape);
                assert_eq!(error.field, $name);
                assert_eq!(flat.bytes(), before, "{} long={long}", $name);
            }
        };
    }
    check_field!(tile_kind, "tile_kind", -17);
    check_field!(tile_crop, "tile_crop", -17);
    check_field!(tile_animal, "tile_animal", -17);
    check_field!(tile_cell, "tile_cell", -17);
    check_field!(tile_role, "tile_role", -17);
    check_field!(tiles_int, "tiles_int", -17);
    check_field!(tiles_float, "tiles_float", -17.0);
    check_field!(actor_slot, "actor_slot", -17);
    check_field!(actor_cell, "actor_cell", -17);
    check_field!(actor_role, "actor_role", -17);
    check_field!(actor_mask, "actor_mask", true);
    check_field!(actor_inventory, "actor_inventory", -17);
    check_field!(actor_inventory_rank, "actor_inventory_rank", -17);
    check_field!(actors_float, "actors_float", -17.0);
    check_field!(player_features, "player_features", -17.0);
    check_field!(storage_counts, "storage_counts", -17);
    check_field!(storage_rank, "storage_rank", -17);
    check_field!(banks, "banks", -17.0);
    check_field!(shop_type, "shop_type", -17);
    check_field!(shop_slot, "shop_slot", -17);
    check_field!(shop_mask, "shop_mask", true);
    check_field!(market_product, "market_product", -17);
    check_field!(market_float, "market_float", -17.0);
    check_field!(market_int, "market_int", -17);
    check_field!(global_features, "global_features", -17.0);
    check_field!(globals_int, "globals_int", -17);
    check_field!(still_playing, "still_playing", true);
    check_field!(order_limits, "order_limits", -17);
    check_field!(can_act, "action_mask.can_act", true);
}

#[test]
fn shape_wrong_environment_count_and_251_frame_mask_are_rejected() {
    let mut flat = FlatBuffers::new(1);
    let before = flat.bytes();
    assert!(matches!(
        flat.borrow().validate(2),
        Err(ObserveError {
            kind: ObserveErrorKind::Shape,
            ..
        })
    ));
    assert_eq!(flat.bytes(), before);
    assert!(matches!(
        flat.borrow().validate(usize::MAX),
        Err(ObserveError {
            kind: ObserveErrorKind::Shape,
            ..
        })
    ));
    assert_eq!(flat.bytes(), before);
    flat.can_act.truncate(251);
    let before = flat.bytes();
    let Err(error) = flat.borrow().validate(1) else {
        panic!("accepted 251 frames");
    };
    assert_eq!(error.field, "action_mask.can_act");
    assert_eq!(flat.bytes(), before);
}

#[test]
fn shape_publish_mismatched_environments_leaves_all_fields_unchanged() {
    for (source_envs, destination_envs) in [(1, 2), (2, 1)] {
        let source = ObsStaging::new(source_envs).unwrap();
        let mut flat = FlatBuffers::new(destination_envs);
        let before = flat.bytes();
        let error = source
            .publish(&mut flat.borrow().validate(destination_envs).unwrap())
            .unwrap_err();
        assert_eq!(error.kind, ObserveErrorKind::Shape);
        assert!(error
            .detail
            .contains(&format!("source n_envs={source_envs}")));
        assert!(error
            .detail
            .contains(&format!("destination n_envs={destination_envs}")));
        assert_eq!(flat.bytes(), before);
    }
}

#[test]
fn shape_serial_and_rayon_cover_identical_ordered_rows() {
    let mut serial = ObsStaging::new(2).unwrap();
    let mut parallel = ObsStaging::new(2).unwrap();
    for (env_index, mut env) in serial.buffers_mut().envs_mut().enumerate() {
        for (seat, row) in env.seats.iter_mut().enumerate() {
            fill_row(row, (2 * env_index + seat + 1) as i64);
        }
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    pool.install(|| {
        parallel
            .buffers_mut()
            .par_envs_mut()
            .enumerate()
            .for_each(|(env_index, mut env)| {
                for (seat, row) in env.seats.iter_mut().enumerate() {
                    fill_row(row, (2 * env_index + seat + 1) as i64);
                }
            })
    });
    assert_eq!(all_bytes(&mut serial), all_bytes(&mut parallel));
    assert_ne!(
        field_bytes(&mut serial, Seat::Zero),
        field_bytes(&mut serial, Seat::One)
    );
    let mut flat = FlatBuffers::new(2);
    serial
        .publish(&mut flat.borrow().validate(2).unwrap())
        .unwrap();
    for (flat_index, value) in flat.tile_kind.iter().enumerate() {
        let tag = (flat_index / 200 + 1) as i64;
        let index = flat_index % 200;
        assert_eq!(
            *value,
            tag * 13000 + index as i64,
            "tile_kind at {flat_index}"
        );
    }
    for (flat_index, value) in flat.tile_crop.iter().enumerate() {
        let tag = (flat_index / 200 + 1) as i64;
        let index = flat_index % 200;
        assert_eq!(
            *value,
            tag * 13000 + index as i64 + 1,
            "tile_crop at {flat_index}"
        );
    }
    for (flat_index, value) in flat.tile_animal.iter().enumerate() {
        let tag = (flat_index / 200 + 1) as i64;
        let index = flat_index % 200;
        assert_eq!(
            *value,
            tag * 13000 + index as i64 + 2,
            "tile_animal at {flat_index}"
        );
    }
    for (flat_index, value) in flat.tile_cell.iter().enumerate() {
        let tag = (flat_index / 200 + 1) as i64;
        let index = flat_index % 200;
        assert_eq!(
            *value,
            tag * 13000 + index as i64 + 3,
            "tile_cell at {flat_index}"
        );
    }
    for (flat_index, value) in flat.tile_role.iter().enumerate() {
        let tag = (flat_index / 200 + 1) as i64;
        let index = flat_index % 200;
        assert_eq!(
            *value,
            tag * 13000 + index as i64 + 4,
            "tile_role at {flat_index}"
        );
    }
    for (flat_index, value) in flat.tiles_int.iter().enumerate() {
        let tag = (flat_index / 1400 + 1) as i64;
        let index = flat_index % 1400;
        assert_eq!(
            *value,
            tag * 13000 + index as i64 + 5,
            "tiles_int at {flat_index}"
        );
    }
    for (flat_index, value) in flat.tiles_float.iter().enumerate() {
        let tag = (flat_index / 3000 + 1) as i64;
        let index = flat_index % 3000;
        assert_eq!(
            *value,
            (tag * 13000 + index as i64 + 6) as f32,
            "tiles_float at {flat_index}"
        );
    }
    for (flat_index, value) in flat.actor_slot.iter().enumerate() {
        let tag = (flat_index / 482 + 1) as i64;
        let index = flat_index % 482;
        assert_eq!(
            *value,
            tag * 13000 + index as i64 + 7,
            "actor_slot at {flat_index}"
        );
    }
    for (flat_index, value) in flat.actor_cell.iter().enumerate() {
        let tag = (flat_index / 482 + 1) as i64;
        let index = flat_index % 482;
        assert_eq!(
            *value,
            tag * 13000 + index as i64 + 8,
            "actor_cell at {flat_index}"
        );
    }
    for (flat_index, value) in flat.actor_role.iter().enumerate() {
        let tag = (flat_index / 482 + 1) as i64;
        let index = flat_index % 482;
        assert_eq!(
            *value,
            tag * 13000 + index as i64 + 9,
            "actor_role at {flat_index}"
        );
    }
    for (flat_index, value) in flat.actor_mask.iter().enumerate() {
        let tag = (flat_index / 482 + 1) as i64;
        let index = flat_index % 482;
        assert_eq!(
            *value,
            (tag + index as i64 + 10) % 2 == 0,
            "actor_mask at {flat_index}"
        );
    }
    for (flat_index, value) in flat.actor_inventory.iter().enumerate() {
        let tag = (flat_index / 2892 + 1) as i64;
        let index = flat_index % 2892;
        assert_eq!(
            *value,
            tag * 13000 + index as i64 + 11,
            "actor_inventory at {flat_index}"
        );
    }
    for (flat_index, value) in flat.actor_inventory_rank.iter().enumerate() {
        let tag = (flat_index / 2892 + 1) as i64;
        let index = flat_index % 2892;
        assert_eq!(
            *value,
            tag * 13000 + index as i64 + 12,
            "actor_inventory_rank at {flat_index}"
        );
    }
    for (flat_index, value) in flat.actors_float.iter().enumerate() {
        let tag = (flat_index / 12532 + 1) as i64;
        let index = flat_index % 12532;
        assert_eq!(
            *value,
            (tag * 13000 + index as i64 + 13) as f32,
            "actors_float at {flat_index}"
        );
    }
    for (flat_index, value) in flat.player_features.iter().enumerate() {
        let tag = (flat_index / 88 + 1) as i64;
        let index = flat_index % 88;
        assert_eq!(
            *value,
            (tag * 13000 + index as i64 + 14) as f32,
            "player_features at {flat_index}"
        );
    }
    for (flat_index, value) in flat.storage_counts.iter().enumerate() {
        let tag = (flat_index / 17 + 1) as i64;
        let index = flat_index % 17;
        assert_eq!(
            *value,
            tag * 13000 + index as i64 + 15,
            "storage_counts at {flat_index}"
        );
    }
    for (flat_index, value) in flat.storage_rank.iter().enumerate() {
        let tag = (flat_index / 12 + 1) as i64;
        let index = flat_index % 12;
        assert_eq!(
            *value,
            tag * 13000 + index as i64 + 16,
            "storage_rank at {flat_index}"
        );
    }
    for (flat_index, value) in flat.banks.iter().enumerate() {
        let tag = (flat_index / 2 + 1) as i64;
        let index = flat_index % 2;
        assert_eq!(
            *value,
            (tag * 13000 + index as i64 + 17) as f64,
            "banks at {flat_index}"
        );
    }
    for (flat_index, value) in flat.shop_type.iter().enumerate() {
        let tag = (flat_index / 8 + 1) as i64;
        let index = flat_index % 8;
        assert_eq!(
            *value,
            tag * 13000 + index as i64 + 18,
            "shop_type at {flat_index}"
        );
    }
    for (flat_index, value) in flat.shop_slot.iter().enumerate() {
        let tag = (flat_index / 8 + 1) as i64;
        let index = flat_index % 8;
        assert_eq!(
            *value,
            tag * 13000 + index as i64 + 19,
            "shop_slot at {flat_index}"
        );
    }
    for (flat_index, value) in flat.shop_mask.iter().enumerate() {
        let tag = (flat_index / 8 + 1) as i64;
        let index = flat_index % 8;
        assert_eq!(
            *value,
            (tag + index as i64 + 20) % 2 == 0,
            "shop_mask at {flat_index}"
        );
    }
    for (flat_index, value) in flat.market_product.iter().enumerate() {
        let tag = (flat_index / 9 + 1) as i64;
        let index = flat_index % 9;
        assert_eq!(
            *value,
            tag * 13000 + index as i64 + 21,
            "market_product at {flat_index}"
        );
    }
    for (flat_index, value) in flat.market_float.iter().enumerate() {
        let tag = (flat_index / 18 + 1) as i64;
        let index = flat_index % 18;
        assert_eq!(
            *value,
            (tag * 13000 + index as i64 + 22) as f32,
            "market_float at {flat_index}"
        );
    }
    for (flat_index, value) in flat.market_int.iter().enumerate() {
        let tag = (flat_index / 18 + 1) as i64;
        let index = flat_index % 18;
        assert_eq!(
            *value,
            tag * 13000 + index as i64 + 23,
            "market_int at {flat_index}"
        );
    }
    for (flat_index, value) in flat.global_features.iter().enumerate() {
        let tag = (flat_index / 15 + 1) as i64;
        let index = flat_index % 15;
        assert_eq!(
            *value,
            (tag * 13000 + index as i64 + 24) as f32,
            "global_features at {flat_index}"
        );
    }
    for (flat_index, value) in flat.globals_int.iter().enumerate() {
        let tag = (flat_index / 16 + 1) as i64;
        let index = flat_index % 16;
        assert_eq!(
            *value,
            tag * 13000 + index as i64 + 25,
            "globals_int at {flat_index}"
        );
    }
    for (flat_index, value) in flat.still_playing.iter().enumerate() {
        let tag = (flat_index + 1) as i64;
        let index = 0_usize;
        assert_eq!(
            *value,
            (tag + index as i64 + 26) % 2 == 0,
            "still_playing at {flat_index}"
        );
    }
    for (flat_index, value) in flat.order_limits.iter().enumerate() {
        let tag = (flat_index + 1) as i64;
        let index = 0_usize;
        assert_eq!(
            *value,
            tag * 13000 + index as i64 + 27,
            "order_limits at {flat_index}"
        );
    }
    for (flat_index, value) in flat.can_act.iter().enumerate() {
        let tag = (flat_index / 252 + 1) as i64;
        let index = flat_index % 252;
        assert_eq!(
            *value,
            (tag + index as i64 + 28) % 2 == 0,
            "can_act at {flat_index}"
        );
    }
}

#[test]
fn shape_staging_allocations_stay_fixed_across_writes_and_publication() {
    let mut source = ObsStaging::new(1).unwrap();
    let mut destination = ObsStaging::new(1).unwrap();
    let source_allocations = source.allocation_layout();
    let destination_allocations = destination.allocation_layout();
    for tag in 0..8 {
        for mut env in source.buffers_mut().envs_mut() {
            for row in &mut env.seats {
                fill_row(row, tag);
            }
        }
        source.publish(&mut destination.buffers_mut()).unwrap();
        assert_eq!(all_bytes(&mut source), all_bytes(&mut destination));
        assert_eq!(source.allocation_layout(), source_allocations);
        assert_eq!(destination.allocation_layout(), destination_allocations);
    }
}

#[test]
fn shape_header_helper_retains_both_private_states() {
    let header = fresh_header(Config::default());
    assert_eq!(header.initial.public.farms.len(), 2);
    assert_eq!(header.initial.privates.len(), 2);
    assert_eq!(header.seed, serde_json::Number::from(42));
}

#[test]
fn shape_error_display_preserves_field_and_optional_context() {
    let error = ObserveError {
        kind: ObserveErrorKind::IntegerRange,
        env: Some(1),
        seat: Some(Seat::Zero),
        field: "market_int[WHEAT,0]".into(),
        detail: "integer outside int64".into(),
    };
    assert_eq!(
        error.to_string(),
        "env=1 seat=0 market_int[WHEAT,0]: integer outside int64"
    );
    let error = ObserveError {
        env: None,
        seat: Some(Seat::One),
        ..error
    };
    assert_eq!(
        error.to_string(),
        "seat=1 market_int[WHEAT,0]: integer outside int64"
    );
    let error = ObserveError {
        seat: None,
        ..error
    };
    assert_eq!(
        error.to_string(),
        "market_int[WHEAT,0]: integer outside int64"
    );
}

#[test]
fn shape_clear_overwrites_every_field_with_positive_zero_or_false() {
    let mut output = ObsStaging::new(1).unwrap();
    let zero = all_bytes(&mut output);
    {
        let mut buffers = output.buffers_mut();
        for mut env in buffers.envs_mut() {
            for row in &mut env.seats {
                fill_row(row, -1);
            }
        }
    }
    assert_ne!(all_bytes(&mut output), zero);
    for mut env in output.buffers_mut().envs_mut() {
        for row in &mut env.seats {
            row.clear();
        }
    }
    assert_eq!(all_bytes(&mut output), zero);
    assert!(zero.iter().all(|byte| *byte == 0));
}

fn encode_env(game: &ObservationGame, out: &mut super::ObsEnvMut<'_>) -> Result<(), ObserveError> {
    native_encode_env(game, out)?;
    for row in &out.seats {
        super::check_row(row)?;
    }
    Ok(())
}

fn encode_header(header: &TraceHeader) -> Result<ObsStaging, ObserveError> {
    let game = ObservationGame::from_header(header)?;
    let mut output = ObsStaging::new(1)?;
    {
        let mut buffers = output.buffers_mut();
        let mut env = buffers.envs_mut().next().unwrap();
        encode_env(&game, &mut env)?;
    }
    Ok(output)
}

fn rejected_header_preserves_output(header: &TraceHeader) -> ObserveError {
    let mut output = FlatBuffers::new(1);
    let before = output.bytes();
    let result = (|| {
        let game = ObservationGame::from_header(header)?;
        let mut buffers = output.borrow().validate(1)?;
        let mut env = buffers.envs_mut().next().unwrap();
        encode_env(&game, &mut env)
    })();
    let error = result.unwrap_err();
    assert_eq!(output.bytes(), before, "rejected {}", error.field);
    error
}

fn config_patch(field: &str, value: serde_json::Value) -> Config {
    let mut configuration = serde_json::to_value(Config::default()).unwrap();
    configuration[field] = value;
    serde_json::from_value(configuration).unwrap()
}

#[test]
fn config_accepts_all_six_corpus_profiles_and_small_episodes() {
    for (episode, day, orders, mult, capacity, unlock, sell, center, money, weed) in [
        (96, 24, 10, 1, 100, 3, 4, 24, 3000, 0.005),
        (96, 12, 4, 3, 64, 2, 3, 12, 7500, 0.0),
        (96, 8, 3, 0, 17, 1, 2, 8, 12345, 0.02),
        (96, 6, 1, 7, 256, 5, 7, 9, 40000, 0.01),
        (96, 30, 8, 2, 500, 4, 5, 30, 9000, 0.125),
        (96, 16, 5, 5, 33, 2, 4, 16, 100000, 1.0),
        (1, 24, 10, 0, 100, 3, 4, 24, 0, 0.0),
        (2, 24, 10, 1, 100, 3, 4, 24, 3000, 0.005),
    ] {
        let config = Config {
            episode_steps: i64::from(episode).into(),
            turns_per_day: i64::from(day).into(),
            max_market_orders_per_turn: i64::from(orders).into(),
            farm_hand_cost_mult: i64::from(mult).into(),
            shed_capacity: i64::from(capacity).into(),
            town_shop_unlock_interval: i64::from(unlock).into(),
            town_shop_sell_interval: i64::from(sell).into(),
            town_center_sell_interval: i64::from(center).into(),
            starting_money: i64::from(money).into(),
            weed_spawn_chance: serde_json::json!(weed),
            ..Config::default()
        };
        ObservationConfig::new(&config).unwrap();
        encode_header(&fresh_header(config)).unwrap();
    }
}

#[test]
fn config_framework_extras_do_not_replace_explicit_seed_or_rule_context() {
    let config = Config {
        weed_spawn_chance: serde_json::json!(0.5),
        ..Config::default()
    };
    let mut with_extras = config.clone();
    for (name, value) in [
        ("seed", serde_json::json!(999)),
        ("actTimeout", serde_json::json!(9)),
        ("runTimeout", serde_json::json!(99)),
    ] {
        with_extras.extra.insert(name.into(), value);
    }
    let mut plain = ObservationGame::from_seed(config, "42").unwrap();
    let mut extras = ObservationGame::from_seed(with_extras, "42").unwrap();
    // Cross a daily RNG boundary; the framework seed cannot silently take over.
    for _ in 0..24 {
        let actions = [serde_json::json!({}), serde_json::json!({})];
        plain.step_with_market_metrics(&actions).unwrap();
        extras.step_with_market_metrics(&actions).unwrap();
    }
    assert_eq!(
        serde_json::to_value(plain.game().snapshot()).unwrap(),
        serde_json::to_value(extras.game().snapshot()).unwrap()
    );
    let mut output_plain = ObsStaging::new(1).unwrap();
    let mut output_extras = ObsStaging::new(1).unwrap();
    encode_env(
        &plain,
        &mut output_plain.buffers_mut().envs_mut().next().unwrap(),
    )
    .unwrap();
    encode_env(
        &extras,
        &mut output_extras.buffers_mut().envs_mut().next().unwrap(),
    )
    .unwrap();
    assert_eq!(all_bytes(&mut output_plain), all_bytes(&mut output_extras));
}

#[test]
fn config_rejects_envelope_violations_without_output_writes() {
    let mut cases = Vec::new();
    for size in [9, 11] {
        cases.push((
            "boardSize",
            serde_json::json!(size),
            ObserveErrorKind::Config,
        ));
    }
    for orders in [0, 11] {
        cases.push((
            "maxMarketOrdersPerTurn",
            serde_json::json!(orders),
            ObserveErrorKind::Config,
        ));
    }
    for field in [
        "episodeSteps",
        "turnsPerDay",
        "shedCapacity",
        "townShopUnlockInterval",
        "townShopSellInterval",
        "townCenterSellInterval",
    ] {
        for value in [0, -1] {
            cases.push((field, serde_json::json!(value), ObserveErrorKind::Config));
        }
    }
    for field in ["startingMoney", "farmHandCostMult"] {
        cases.push((field, serde_json::json!(-1), ObserveErrorKind::Config));
    }
    cases.push((
        "marketParams",
        serde_json::json!({"WHEAT":{}}),
        ObserveErrorKind::Config,
    ));
    cases.push((
        "unsupportedGameOption",
        serde_json::json!(1),
        ObserveErrorKind::Config,
    ));
    for field in [
        "episodeSteps",
        "startingMoney",
        "maxMarketOrdersPerTurn",
        "turnsPerDay",
        "shedCapacity",
        "townShopUnlockInterval",
        "townShopSellInterval",
        "townCenterSellInterval",
        "farmHandCostMult",
    ] {
        cases.push((
            field,
            serde_json::from_str("9223372036854775808").unwrap(),
            ObserveErrorKind::IntegerRange,
        ));
    }
    for (field, value, kind) in cases {
        let config = config_patch(field, value);
        let error = ObservationConfig::new(&config).unwrap_err();
        assert_eq!(error.kind, kind, "{field}: {error}");
        assert!(error.field.contains(field), "{error}");
        let mut header = fresh_header(Config::default());
        header.configuration = config;
        let error = rejected_header_preserves_output(&header);
        assert_eq!(error.kind, kind, "{field}: {error}");
    }
}

#[test]
fn config_day_order_product_has_exact_240_boundary_and_checked_overflow() {
    for (day, orders, accepted) in [
        (24_i64, 10_i64, true),
        (30, 8, true),
        (240, 1, true),
        (241, 1, false),
        (i64::MAX, 10, false),
    ] {
        let config = Config {
            turns_per_day: day.into(),
            max_market_orders_per_turn: orders.into(),
            ..Config::default()
        };
        assert_eq!(
            ObservationConfig::new(&config).is_ok(),
            accepted,
            "D={day} M={orders}"
        );
        if !accepted {
            let mut header = fresh_header(Config::default());
            header.configuration = config;
            assert_eq!(
                rejected_header_preserves_output(&header).kind,
                ObserveErrorKind::Config
            );
        }
    }
}

#[test]
fn config_weed_requires_finite_nonnegative_numeric_f32_without_unit_cap() {
    for value in [
        serde_json::json!(0),
        serde_json::json!(0.125),
        serde_json::json!(1),
        serde_json::json!(2.5),
        serde_json::json!(100),
    ] {
        ObservationConfig::new(&config_patch("weedSpawnChance", value)).unwrap();
    }
    for (value, kind) in [
        (serde_json::json!(-0.001), ObserveErrorKind::Config),
        (serde_json::json!(true), ObserveErrorKind::Config),
        (serde_json::json!("0.1"), ObserveErrorKind::Config),
        (serde_json::Value::Null, ObserveErrorKind::Config),
        (serde_json::json!([]), ObserveErrorKind::Config),
        (serde_json::json!({}), ObserveErrorKind::Config),
        (serde_json::json!(1e100), ObserveErrorKind::NonFinite),
        (
            serde_json::from_str("1e999").unwrap(),
            ObserveErrorKind::NonFinite,
        ),
    ] {
        let config = config_patch("weedSpawnChance", value.clone());
        let error = ObservationConfig::new(&config).unwrap_err();
        assert_eq!(error.kind, kind, "weed={value}: {error}");
        assert_eq!(error.field, "weedSpawnChance");
        let mut header = fresh_header(Config::default());
        header.configuration = config;
        assert_eq!(rejected_header_preserves_output(&header).kind, kind);
    }
}

#[test]
fn config_preserves_maximum_exact_integers_and_large_scaled_values() {
    let config = Config {
        episode_steps: i64::MAX.into(),
        starting_money: i64::MAX.into(),
        shed_capacity: i64::MAX.into(),
        farm_hand_cost_mult: i64::MAX.into(),
        town_shop_unlock_interval: i64::MAX.into(),
        town_shop_sell_interval: i64::MAX.into(),
        town_center_sell_interval: i64::MAX.into(),
        weed_spawn_chance: serde_json::json!(9),
        ..Config::default()
    };
    let mut output = encode_header(&fresh_header(config)).unwrap();
    let mut buffers = output.buffers_mut();
    let env = buffers.envs_mut().next().unwrap();
    for row in &env.seats {
        for channel in [3, 6, 7, 8, 9, 10, 11] {
            assert_eq!(row.globals_int[channel], i64::MAX, "channel {channel}");
        }
        for channel in [6, 7, 8, 9, 10, 11, 12, 13] {
            assert!(row.global_features[channel] > 4.0, "channel {channel}");
        }
        assert!(row.player_features[0][10] > 4.0);
    }
}

#[test]
fn config_context_uses_bound_rules_wide_time_arithmetic_and_both_roles() {
    let config = Config {
        episode_steps: 96_i64.into(),
        turns_per_day: 12_i64.into(),
        max_market_orders_per_turn: 4_i64.into(),
        farm_hand_cost_mult: 3_i64.into(),
        shed_capacity: 64_i64.into(),
        starting_money: 7500_i64.into(),
        town_shop_unlock_interval: 3_i64.into(),
        town_shop_sell_interval: 7_i64.into(),
        town_center_sell_interval: 12_i64.into(),
        weed_spawn_chance: serde_json::json!(2),
        ..Config::default()
    };
    let mut header = fresh_header(config);
    header.initial.public.step = 17;
    header.initial.public.day = 1;
    header.initial.public.hour = 5;
    header.initial.public.farms[0].hires_today = 4;
    header.initial.public.farms[1].hires_today = 5;
    header.initial.public.farms[0].money = -1_234_567.123456789;
    header.initial.public.farms[1].money = 9_876_543.987_654_32;
    header.initial.public.town.unlocked_shops = vec!["BAKERY".into(), "PIZZA_SHOP".into()];
    let mut output = encode_header(&header).unwrap();
    let mut buffers = output.buffers_mut();
    let env = buffers.envs_mut().next().unwrap();
    let expected_floats = [
        17.0 / 96.0,
        5.0 / 12.0,
        1.0 / 8.0,
        78.0 / 96.0,
        12.0 / 24.0,
        4.0 / 10.0,
        64.0 / 1000.0,
        3.0 / 100.0,
        96.0 / 1000.0,
        2.0,
        3.0 / 12.0,
        7.0 / 12.0,
        1.0,
        7500.0 / 200000.0,
        2.0 / 8.0,
    ]
    .map(|value: f64| value as f32);
    for (seat, row) in env.seats.iter().enumerate() {
        assert_eq!(*row.global_features, expected_floats);
        let (own, rival) = if seat == 0 { (4, 5) } else { (5, 4) };
        assert_eq!(
            *row.globals_int,
            [17, 1, 5, 96, 12, 4, 64, 3, 3, 7, 12, 7500, own, rival, 1, 1]
        );
        assert_eq!(*row.order_limits, 4);
        for (role, farm) in [seat, 1 - seat].into_iter().enumerate() {
            assert_eq!(
                row.banks[role].to_bits(),
                header.initial.public.farms[farm].money.to_bits()
            );
            assert_eq!(
                row.player_features[role][0],
                (row.banks[role] / 200000.0) as f32
            );
            assert_eq!(row.player_features[role][1], (1.0_f64 / 241.0) as f32);
            assert_eq!(
                row.player_features[role][9],
                (header.initial.public.farms[farm].hires_today as f64 / 240.0) as f32
            );
        }
    }
    for (episode, step, expected_day, remaining) in [
        (1_i64, 0_usize, 0.0, 1.0),
        (2, 24, 12.0, 0.0),
        (
            1,
            usize::try_from(i64::MAX).unwrap(),
            ((i64::MAX / 24) as f64) / (1.0 / 24.0),
            0.0,
        ),
    ] {
        let mut header = fresh_header(Config {
            episode_steps: episode.into(),
            ..Config::default()
        });
        header.initial.public.step = step;
        header.initial.public.day = step / 24;
        header.initial.public.hour = step % 24;
        let mut output = encode_header(&header).unwrap();
        let mut buffers = output.buffers_mut();
        let env = buffers.envs_mut().next().unwrap();
        assert_eq!(env.seats[0].global_features[2], expected_day as f32);
        assert_eq!(env.seats[0].global_features[3], remaining as f32);
    }
}

#[test]
fn config_rejects_nonempty_explicit_market_params_but_accepts_empty() {
    let mut header = fresh_header(Config::default());
    header.initial.public.market.params = Some(Default::default());
    encode_header(&header).unwrap();
    header
        .initial
        .public
        .market
        .params
        .as_mut()
        .unwrap()
        .insert("WHEAT".into(), Default::default());
    let error = rejected_header_preserves_output(&header);
    assert_eq!(error.kind, ObserveErrorKind::State);
    assert_eq!(error.field, "public.market.params");
}

#[test]
fn config_rejects_farm_count_clock_and_nonfinite_derived_context_transactionally() {
    for count in [0, 1, 3] {
        let mut header = fresh_header(Config::default());
        header
            .initial
            .public
            .farms
            .resize(count, header.initial.public.farms[0].clone());
        header
            .initial
            .privates
            .resize(count, header.initial.privates[0].clone());
        assert_eq!(
            rejected_header_preserves_output(&header).field,
            "public.farms"
        );
    }
    for wrong_day in [false, true] {
        let mut header = fresh_header(Config::default());
        if wrong_day {
            header.initial.public.day = 1;
        } else {
            header.initial.public.hour = 1;
        }
        assert_eq!(
            rejected_header_preserves_output(&header).field,
            if wrong_day {
                "public.day"
            } else {
                "public.hour"
            }
        );
    }
    for money in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::MAX] {
        let mut header = fresh_header(Config::default());
        header.initial.public.farms[1].money = money;
        let error = rejected_header_preserves_output(&header);
        assert_eq!(error.kind, ObserveErrorKind::NonFinite);
        assert_eq!(error.seat, Some(Seat::Zero));
        assert_eq!(error.field, "player_features[1,0]");
    }
    let mut header = fresh_header(Config::default());
    header.initial.public.farms[0].hires_today = usize::MAX;
    assert_eq!(
        rejected_header_preserves_output(&header).kind,
        ObserveErrorKind::IntegerRange
    );
}

#[test]
fn hire_cost_uses_each_public_count() {
    let mut header = fresh_header(Config {
        farm_hand_cost_mult: 7_i64.into(),
        ..Config::default()
    });
    header.initial.public.farms[0].hires_today = 4;
    header.initial.public.farms[1].hires_today = 5;
    let mut output = encode_header(&header).unwrap();
    let mut buffers = output.buffers_mut();
    let env = buffers.envs_mut().next().unwrap();
    for (seat, expected) in [[35.0, 56.0], [56.0, 35.0]].into_iter().enumerate() {
        for (role, cost) in expected.into_iter().enumerate() {
            assert_eq!(
                env.seats[seat].player_features[role][10],
                (cost / 200000.0_f64) as f32
            );
        }
    }
}

#[test]
fn hire_cost_literal_sequence_matches_engine_hire_bank_deltas() {
    let mut game = ObservationGame::from_seed(
        Config {
            farm_hand_cost_mult: 7_i64.into(),
            ..Config::default()
        },
        "42",
    )
    .unwrap();
    for literal in [1, 1, 2, 3, 5, 8] {
        let before = game.game().snapshot();
        let mut output = ObsStaging::new(1).unwrap();
        encode_env(&game, &mut output.buffers_mut().envs_mut().next().unwrap()).unwrap();
        {
            let mut buffers = output.buffers_mut();
            let env = buffers.envs_mut().next().unwrap();
            assert_eq!(
                env.seats[0].player_features[0][10],
                (f64::from(7 * literal) / 200000.0) as f32
            );
            assert_eq!(
                env.seats[1].player_features[1][10],
                (f64::from(7 * literal) / 200000.0) as f32
            );
        }
        game.step_with_market_metrics(&[
            serde_json::json!({"market":[["HIRE"]]}),
            serde_json::json!({}),
        ])
        .unwrap();
        let after = game.game().snapshot();
        assert_eq!(
            before.public.farms[0].money - after.public.farms[0].money,
            f64::from(7 * literal)
        );
        assert_eq!(
            after.public.farms[0].hires_today,
            before.public.farms[0].hires_today + 1
        );
        assert_eq!(after.public.farms[1].money, before.public.farms[1].money);
    }
}

#[test]
fn hire_cost_zero_multiplier_handles_huge_representable_count_without_iteration() {
    let mut header = fresh_header(Config {
        farm_hand_cost_mult: 0_i64.into(),
        ..Config::default()
    });
    header.initial.public.farms[0].hires_today = usize::try_from(i64::MAX).unwrap();
    header.initial.public.farms[1].hires_today = 240;
    let mut output = encode_header(&header).unwrap();
    let mut buffers = output.buffers_mut();
    let env = buffers.envs_mut().next().unwrap();
    for row in env.seats {
        for role in 0..2 {
            assert_eq!(row.player_features[role][10].to_bits(), 0.0_f32.to_bits());
        }
    }
}

#[test]
fn hire_cost_positive_prefix_stops_at_first_f32_overflow() {
    // Independently evaluated integer recurrence: 7 * fib(206) fits after /200000,
    // while 7 * fib(207) = 92266109784618691863425340395105880651619197 does not.
    let mut header = fresh_header(Config {
        farm_hand_cost_mult: 7_i64.into(),
        ..Config::default()
    });
    header.initial.public.farms[0].hires_today = 206;
    let mut output = encode_header(&header).unwrap();
    let mut buffers = output.buffers_mut();
    let env = buffers.envs_mut().next().unwrap();
    assert_eq!(
        env.seats[0].player_features[0][10],
        (57_023_591_856_623_591_583_060_292_457_594_532_272_891_566_f64 / 200000.0) as f32
    );
    header.initial.public.farms[1].hires_today = 207;
    let error = rejected_header_preserves_output(&header);
    assert_eq!(error.kind, ObserveErrorKind::NonFinite);
    assert_eq!(error.field, "player_features[1,10]");
    assert_eq!(error.seat, Some(Seat::Zero));
    assert!(error.detail.contains("207"));
    assert!(error.detail.contains("206"));
}

fn tile_plant(crop: &str) -> serde_json::Value {
    serde_json::json!({
        "kind": "PLANT", "crop": crop, "yield_units": 3,
        "planted_day": 1, "max_lifespan_step": 200,
        "fertilized_until_day": -1, "watered_today": true,
        "consecutive_unwatered": 2,
    })
}

fn tile_animal(animal: &str) -> serde_json::Value {
    serde_json::json!({
        "kind": if animal == "GOOSE" { "COOP" } else { "PASTURE" },
        "animal": animal, "yield_units": 7, "placed_day": 3,
        "consecutive_unfed": 4, "fed_today": true, "cared_today": false,
        "fertilizer_available": true, "pending_care_bonus": 11,
    })
}

fn tile_header() -> TraceHeader {
    let mut header = fresh_header(Config::default());
    header.initial.public.step = 239;
    header.initial.public.day = 9;
    header.initial.public.hour = 23;
    for farm in &mut header.initial.public.farms {
        farm.tiles = vec![vec![serde_json::Value::Null; 10]; 10];
    }
    header
}

#[test]
fn tile_asymmetric_coordinates_and_every_channel() {
    let mut header = tile_header();
    header.initial.public.farms[0].tiles[2][7] = tile_plant("TOMATO");
    header.initial.public.farms[0].tiles[7][2] = tile_animal("GOOSE");
    let mut output = encode_header(&header).unwrap();
    let mut buffers = output.buffers_mut();
    let env = buffers.envs_mut().next().unwrap();
    for (seat, row) in env.seats.into_iter().enumerate() {
        let plant = seat * 100 + 27;
        let animal = seat * 100 + 72;
        assert_eq!(row.tile_kind[plant], 3);
        assert_eq!(row.tile_crop[plant], 3);
        assert_eq!(row.tile_animal[plant], 0);
        assert_eq!(row.tiles_int[plant], [3, 1, 200, -1, 0, 2, 0]);
        assert_eq!(
            row.tiles_float[plant],
            [
                3.0 / 8.0,
                1.0,
                2.0 / 8.0,
                (8.0_f64 / 30.0) as f32,
                1.0,
                (-39.0_f64 / 720.0) as f32,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        );
        assert_eq!(row.tile_kind[animal], 4);
        assert_eq!(row.tile_crop[animal], 0);
        assert_eq!(row.tile_animal[animal], 1);
        assert_eq!(row.tiles_int[animal], [7, 0, 0, 0, 3, 0, 4]);
        assert_eq!(
            row.tiles_float[animal],
            [
                7.0 / 8.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                (6.0_f64 / 30.0) as f32,
                4.0 / 8.0,
                1.0,
                0.0,
                1.0,
                11.0 / 8.0,
            ]
        );
        for token in 0..200 {
            assert_eq!(row.tile_cell[token], (token % 100) as i64);
            assert_eq!(row.tile_role[token], (token / 100) as i64);
        }
    }
}

#[test]
fn tile_all_vocabularies_and_empty_applicability() {
    let crops = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"];
    let animals = ["GOOSE", "COW", "SHEEP"];
    assert_eq!(kaggriculture_engine::CROP_NAMES, crops);
    assert_eq!(kaggriculture_engine::ANIMAL_NAMES, animals);
    let mut header = tile_header();
    for (cell, crop) in crops.into_iter().enumerate() {
        header.initial.public.farms[1].tiles[0][cell] = tile_plant(crop);
    }
    for (cell, animal) in animals.into_iter().enumerate() {
        header.initial.public.farms[1].tiles[1][cell] = tile_animal(animal);
    }
    for (cell, tile) in [
        serde_json::Value::Null,
        serde_json::json!("LOCKED"),
        serde_json::json!({"kind":"WEED"}),
        serde_json::json!({"kind":"COOP"}),
        serde_json::json!({"kind":"PASTURE"}),
    ]
    .into_iter()
    .enumerate()
    {
        header.initial.public.farms[1].tiles[2][cell] = tile;
    }
    let mut output = encode_header(&header).unwrap();
    let mut buffers = output.buffers_mut();
    let env = buffers.envs_mut().next().unwrap();
    for (seat, row) in env.seats.into_iter().enumerate() {
        let base = (1 - seat) * 100;
        for cell in 0..5 {
            assert_eq!(row.tile_kind[base + cell], 3);
            assert_eq!(row.tile_crop[base + cell], cell as i64 + 1);
        }
        for (cell, kind) in [4, 5, 5].into_iter().enumerate() {
            assert_eq!(row.tile_kind[base + 10 + cell], kind);
            assert_eq!(row.tile_animal[base + 10 + cell], cell as i64 + 1);
        }
        for (cell, kind) in [0, 1, 2, 4, 5].into_iter().enumerate() {
            let token = base + 20 + cell;
            assert_eq!(row.tile_kind[token], kind);
            assert_eq!(row.tile_crop[token], 0);
            assert_eq!(row.tile_animal[token], 0);
            assert_eq!(row.tiles_int[token], [0; 7]);
            assert_eq!(row.tiles_float[token].map(f32::to_bits), [0; 15]);
        }
    }
}

#[test]
fn tile_sentinels_and_fertilizer_flags_keep_applicability() {
    let mut header = tile_header();
    for (cell, until) in [-1, 8, 9, 11].into_iter().enumerate() {
        let mut plant = tile_plant("MELON");
        plant["planted_day"] = serde_json::json!(-1);
        plant["max_lifespan_step"] = serde_json::json!(-1);
        plant["fertilized_until_day"] = serde_json::json!(until);
        plant["watered_today"] = serde_json::json!(false);
        header.initial.public.farms[0].tiles[0][cell] = plant;
    }
    let mut animal = tile_animal("SHEEP");
    animal["placed_day"] = serde_json::json!(-1);
    animal["fed_today"] = serde_json::json!(false);
    animal["cared_today"] = serde_json::json!(true);
    animal["fertilizer_available"] = serde_json::json!(false);
    header.initial.public.farms[0].tiles[0][4] = animal;
    let mut output = encode_header(&header).unwrap();
    let mut buffers = output.buffers_mut();
    let env = buffers.envs_mut().next().unwrap();
    let row = &env.seats[0];
    for (cell, (until, flags)) in [
        (-1, [0.0, 0.0, 0.0]),
        (8, [1.0, 0.0, 0.0]),
        (9, [1.0, 1.0, 0.0]),
        (11, [1.0, 1.0, (2.0_f64 / 30.0) as f32]),
    ]
    .into_iter()
    .enumerate()
    {
        assert_eq!(row.tiles_int[cell], [3, -1, -1, until, 0, 2, 0]);
        assert_eq!(row.tiles_float[cell][1], 0.0);
        assert_eq!(&row.tiles_float[cell][3..6], &[0.0; 3]);
        assert_eq!(&row.tiles_float[cell][6..9], &flags);
    }
    assert_eq!(row.tiles_int[4], [7, 0, 0, 0, -1, 0, 4]);
    assert_eq!(&row.tiles_float[4][9..14], &[0.0, 0.5, 0.0, 1.0, 0.0]);
}

#[test]
fn tile_exact_large_integers_and_signed_future_dates_are_not_clamped() {
    let mut header = tile_header();
    let large = i64::MAX;
    let mut plant = tile_plant("WHEAT");
    for field in [
        "yield_units",
        "planted_day",
        "max_lifespan_step",
        "fertilized_until_day",
        "consecutive_unwatered",
    ] {
        plant[field] = serde_json::json!(large);
    }
    header.initial.public.farms[0].tiles[0][0] = plant;
    let mut animal = tile_animal("COW");
    for field in [
        "yield_units",
        "placed_day",
        "consecutive_unfed",
        "pending_care_bonus",
    ] {
        animal[field] = serde_json::json!(large);
    }
    header.initial.public.farms[0].tiles[0][1] = animal;
    let mut output = encode_header(&header).unwrap();
    let mut buffers = output.buffers_mut();
    let env = buffers.envs_mut().next().unwrap();
    let row = &env.seats[0];
    assert_eq!(row.tiles_int[0], [large, large, large, large, 0, large, 0]);
    assert_eq!(row.tiles_int[1], [large, 0, 0, 0, large, 0, large]);
    assert_eq!(
        row.tiles_float[0][3],
        ((9_i128 - i128::from(large)) as f64 / 30.0) as f32
    );
    assert_eq!(
        row.tiles_float[0][5],
        ((i128::from(large) - 239) as f64 / 720.0) as f32
    );
    assert_eq!(
        row.tiles_float[0][8],
        ((i128::from(large) - 9) as f64 / 30.0) as f32
    );
    assert_eq!(row.tiles_float[1][9], row.tiles_float[0][3]);
    assert_eq!(row.tiles_float[1][14], (large as f64 / 8.0) as f32);
    assert!(row.tiles_float[1][14] > 4.0);
    assert!(row.tiles_float.iter().flatten().all(|v| v.is_finite()));
}

#[test]
fn tile_strict_constructor_keys_reject_stale_null_missing_and_unknown_fields() {
    let mut stale_weed = serde_json::json!({"kind":"WEED"});
    stale_weed["yield_units"] = serde_json::json!(0);
    let mut stale_coop = serde_json::json!({"kind":"COOP"});
    stale_coop["planted_day"] = serde_json::json!(-1);
    let mut unknown = tile_plant("WHEAT");
    unknown["not_an_engine_field"] = serde_json::json!(0);
    let mut null_animal = tile_animal("GOOSE");
    null_animal["animal"] = serde_json::Value::Null;
    let mut bad_structure = tile_animal("GOOSE");
    bad_structure["kind"] = serde_json::json!("PASTURE");
    let mut cow_coop = tile_animal("COW");
    cow_coop["kind"] = serde_json::json!("COOP");
    let mut animal_plant = tile_plant("WHEAT");
    animal_plant["animal"] = serde_json::json!("GOOSE");
    let mut missing = tile_plant("WHEAT");
    missing
        .as_object_mut()
        .unwrap()
        .shift_remove("watered_today");
    let mut missing_animal = tile_animal("SHEEP");
    missing_animal
        .as_object_mut()
        .unwrap()
        .shift_remove("pending_care_bonus");
    for tile in [
        stale_weed,
        stale_coop,
        unknown,
        null_animal,
        bad_structure,
        cow_coop,
        animal_plant,
        missing,
        missing_animal,
        serde_json::json!({"kind":"COOP","animal":null}),
    ] {
        for seat in 0..2 {
            let mut header = tile_header();
            header.initial.public.farms[seat].tiles[2][7] = tile.clone();
            let error = rejected_header_preserves_output(&header);
            assert_eq!(error.kind, ObserveErrorKind::State, "{tile}");
            assert!(error.field.contains("tile"));
        }
    }
}

#[test]
fn tile_rejects_bad_representations_enums_and_nonboolean_flags() {
    let mut cases = vec![
        serde_json::json!(0),
        serde_json::json!(true),
        serde_json::json!([]),
        serde_json::json!("EMPTY"),
        serde_json::json!({}),
        serde_json::json!({"kind":"UNKNOWN"}),
    ];
    for (mut tile, field, bad) in [
        (tile_plant("NONE"), "crop", serde_json::json!("NONE")),
        (tile_animal("GOOSE"), "animal", serde_json::json!("DUCK")),
        (tile_plant("WHEAT"), "kind", serde_json::json!(false)),
        (tile_plant("WHEAT"), "crop", serde_json::json!(3)),
    ] {
        tile[field] = bad;
        cases.push(tile);
    }
    for (base, fields) in [
        (tile_plant("WHEAT"), vec!["watered_today"]),
        (
            tile_animal("GOOSE"),
            vec!["fed_today", "cared_today", "fertilizer_available"],
        ),
    ] {
        for field in fields {
            for value in [
                serde_json::json!(0),
                serde_json::json!("true"),
                serde_json::Value::Null,
            ] {
                let mut tile = base.clone();
                tile[field] = value;
                cases.push(tile);
            }
        }
    }
    for tile in cases {
        let mut header = tile_header();
        header.initial.public.farms[1].tiles[2][7] = tile;
        rejected_header_preserves_output(&header);
    }
}

#[test]
fn tile_integer_fields_reject_fractional_wrong_types_ranges_and_negative_counts() {
    let too_large: serde_json::Value = serde_json::from_str("9223372036854775808").unwrap();
    for (base, fields) in [
        (
            tile_plant("WHEAT"),
            vec![
                "yield_units",
                "planted_day",
                "max_lifespan_step",
                "fertilized_until_day",
                "consecutive_unwatered",
            ],
        ),
        (
            tile_animal("GOOSE"),
            vec![
                "yield_units",
                "placed_day",
                "consecutive_unfed",
                "pending_care_bonus",
            ],
        ),
    ] {
        for field in fields {
            let minimum_invalid = if [
                "planted_day",
                "max_lifespan_step",
                "fertilized_until_day",
                "placed_day",
            ]
            .contains(&field)
            {
                -2
            } else {
                -1
            };
            for value in [
                serde_json::json!(1.5),
                serde_json::json!(1.0),
                serde_json::json!("1"),
                serde_json::json!(true),
                serde_json::Value::Null,
                too_large.clone(),
                serde_json::json!(minimum_invalid),
            ] {
                let mut header = tile_header();
                let mut tile = base.clone();
                tile[field] = value;
                header.initial.public.farms[1].tiles[2][7] = tile;
                let error = rejected_header_preserves_output(&header);
                assert!(error.field.contains(field), "{error}");
            }
        }
    }
}

#[test]
fn tile_board_shape_rejection_preserves_all_output() {
    for seat in 0..2 {
        for (axis, long) in [(0, false), (0, true), (1, false), (1, true)] {
            let mut header = tile_header();
            let tiles = &mut header.initial.public.farms[seat].tiles;
            match (axis, long) {
                (0, false) => {
                    tiles.pop();
                },
                (0, true) => tiles.push(vec![serde_json::Value::Null; 10]),
                (1, false) => {
                    tiles[2].pop();
                },
                (1, true) => tiles[2].push(serde_json::Value::Null),
                _ => unreachable!(),
            }
            assert_eq!(
                rejected_header_preserves_output(&header).kind,
                ObserveErrorKind::State
            );
        }
    }
}

#[test]
fn tile_reusing_output_fully_clears_old_applicable_values() {
    let mut header = tile_header();
    header.initial.public.farms[0].tiles[2][7] = tile_plant("TOMATO");
    header.initial.public.farms[1].tiles[7][2] = tile_animal("GOOSE");
    let mut output = FlatBuffers::new(1);
    for current in [&header, &tile_header()] {
        let game = ObservationGame::from_header(current).unwrap();
        encode_env(
            &game,
            &mut output
                .borrow()
                .validate(1)
                .unwrap()
                .envs_mut()
                .next()
                .unwrap(),
        )
        .unwrap();
    }
    assert!(output.tile_kind.iter().all(|v| *v == 0));
    assert!(output.tile_crop.iter().all(|v| *v == 0));
    assert!(output.tile_animal.iter().all(|v| *v == 0));
    assert!(output.tiles_int.iter().all(|v| *v == 0));
    assert!(output.tiles_float.iter().all(|v| v.to_bits() == 0));
    for (token, value) in output.tile_cell.iter().enumerate() {
        assert_eq!(*value, (token % 100) as i64);
    }
}

// Independent literals from the pinned new_plant/new_animal/WEED/build/death
// constructors. This does not call the production tile parser or its constants.
fn official_tile_shape(tile: &serde_json::Value) -> Result<usize, String> {
    use serde_json::Value;
    let (shape, keys): (usize, &[&str]) = match tile {
        Value::Null => return Ok(0),
        Value::String(s) if s == "LOCKED" => return Ok(1),
        Value::Object(map) => match (map.get("kind").and_then(Value::as_str), map.get("animal")) {
            (Some("WEED"), None) => (2, &["kind"]),
            (Some("PLANT"), None) => (
                3,
                &[
                    "kind",
                    "crop",
                    "planted_day",
                    "watered_today",
                    "consecutive_unwatered",
                    "yield_units",
                    "max_lifespan_step",
                    "fertilized_until_day",
                ],
            ),
            (Some("COOP"), None) => (4, &["kind"]),
            (Some("PASTURE"), None) => (5, &["kind"]),
            (Some("COOP"), Some(Value::String(s))) if s == "GOOSE" => (
                6,
                &[
                    "kind",
                    "animal",
                    "placed_day",
                    "yield_units",
                    "consecutive_unfed",
                    "fed_today",
                    "cared_today",
                    "fertilizer_available",
                    "pending_care_bonus",
                ],
            ),
            (Some("PASTURE"), Some(Value::String(s))) if s == "COW" || s == "SHEEP" => (
                if s == "COW" { 7 } else { 8 },
                &[
                    "kind",
                    "animal",
                    "placed_day",
                    "yield_units",
                    "consecutive_unfed",
                    "fed_today",
                    "cared_today",
                    "fertilizer_available",
                    "pending_care_bonus",
                ],
            ),
            _ => return Err(format!("unknown constructor shape: {tile}")),
        },
        _ => return Err(format!("invalid tile representation: {tile}")),
    };
    let map = tile.as_object().ok_or("expected tile object")?;
    if map.len() != keys.len() || keys.iter().any(|key| !map.contains_key(*key)) {
        return Err(format!(
            "constructor keys differ: {tile}; expected {keys:?}"
        ));
    }
    Ok(shape)
}

fn scan_official_tile_shapes(episode: u64) -> Result<[usize; 9], Box<dyn std::error::Error>> {
    use std::io::{BufRead, BufReader};
    use std::process::{Command, Stdio};
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("engine_rs/fixtures/episode-{episode}.jsonl.gz"));
    let mut child = Command::new("gzip")
        .arg("-dc")
        .arg(&path)
        .stdout(Stdio::piped())
        .spawn()?;
    let result = (|| -> Result<[usize; 9], Box<dyn std::error::Error>> {
        let stdout = child.stdout.take().ok_or("gzip stdout unavailable")?;
        let reader = BufReader::new(stdout);
        let mut counts = [0; 9];
        let mut states = 0;
        for line in reader.lines() {
            let row: serde_json::Value = serde_json::from_str(&line?)?;
            let public = if states == 0 {
                if row["type"] != "header"
                    || row["source"]["episode_id"] != episode
                    || row["source"]["module_version"] != "1.32.7"
                    || row["source"]["engine_sha256"]
                        != "bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e"
                    || row["transitions"] != 719
                {
                    return Err(
                        format!("episode {episode}: unexpected header identity/count").into(),
                    );
                }
                &row["initial"]["public"]
            } else {
                if row["type"] != "transition" || row["from_step"] != states - 1 {
                    return Err(
                        format!("episode {episode}: unexpected transition {}", states - 1).into(),
                    );
                }
                &row["expected"]
            };
            if public["step"] != states {
                return Err("nonsequential snapshot step".into());
            }
            let farms = public["farms"].as_array().ok_or("farms must be array")?;
            if farms.len() != 2 {
                return Err("expected two farms".into());
            }
            for farm in farms {
                let rows = farm["tiles"].as_array().ok_or("tiles must be array")?;
                if rows.len() != 10 {
                    return Err("expected ten tile rows".into());
                }
                for row in rows {
                    let cells = row.as_array().ok_or("tile row must be array")?;
                    if cells.len() != 10 {
                        return Err("expected ten tile columns".into());
                    }
                    for tile in cells {
                        counts[official_tile_shape(tile)?] += 1;
                    }
                }
            }
            states += 1;
        }
        if states != 720 {
            return Err(format!("episode {episode}: {states} states, expected 720").into());
        }
        Ok(counts)
    })();
    if result.is_err() {
        let _ = child.kill();
    }
    let status = child.wait()?;
    let counts = result?;
    if !status.success() {
        return Err(format!("gzip failed: {status}").into());
    }
    Ok(counts)
}

#[test]
fn tile_official_fixture_constructor_keys_cover_all_2880_states() {
    let mut counts = [0; 9];
    for episode in [95324500, 95901360, 95921764, 95990191] {
        for (shape, count) in scan_official_tile_shapes(episode)
            .unwrap()
            .into_iter()
            .enumerate()
        {
            counts[shape] += count;
        }
    }
    assert_eq!(counts.iter().sum::<usize>(), 576_000);
    assert!(counts.iter().all(|count| *count > 0));
    println!("official constructor key-set scan: 2880 states, 576000 tiles; [empty, locked, weed, plant, empty coop, empty pasture, goose, cow, sheep] = {counts:?}");
}

const TEST_ITEMS: [&str; 12] = [
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

fn test_counts(entries: &[(&str, i64)]) -> kaggriculture_engine::Counts {
    entries
        .iter()
        .map(|(name, count)| ((*name).into(), *count))
        .collect()
}

fn actor_header() -> TraceHeader {
    let mut header = fresh_header(Config::default());
    header.initial.public.farms[0].farmer = vec![7, 2];
    header.initial.public.farms[0].hands = vec![vec![1, 9], vec![9, 1]];
    header.initial.public.farms[1].farmer = vec![2, 7];
    header.initial.public.farms[1].hands = vec![vec![3, 8]];
    header.initial.privates[0].inventories = vec![
        test_counts(&[("WOOL", 0), ("WHEAT", 7)]),
        test_counts(&[("MILK", 3)]),
        test_counts(&[]),
    ];
    header.initial.privates[1].inventories = vec![
        test_counts(&[("COW", 2), ("EGG", 4)]),
        test_counts(&[("SHEEP", 1)]),
    ];
    header.initial.privates[0].shed = test_counts(&[("WOOL", 0), ("WHEAT", 7), ("COW", 2)]);
    header.initial.privates[1].shed = test_counts(&[("MILK", 4), ("SHEEP", 1)]);
    header.initial.privates[0].seeds = test_counts(&[("TOMATO", 5), ("WHEAT", 3)]);
    header.initial.privates[1].seeds = test_counts(&[("MELON", 2), ("WHEAT", 0)]);
    header
}

#[test]
fn actors_public_positions_roles_and_all_private_channels() {
    let header = actor_header();
    let mut output = encode_header(&header).unwrap();
    let mut buffers = output.buffers_mut();
    let env = buffers.envs_mut().next().unwrap();
    for (seat, row) in env.seats.into_iter().enumerate() {
        for (role, farm_index) in [seat, 1 - seat].into_iter().enumerate() {
            let farm = &header.initial.public.farms[farm_index];
            for (local, position) in std::iter::once(&farm.farmer).chain(&farm.hands).enumerate() {
                let actor = role * 241 + local;
                assert_eq!(row.actor_slot[actor], local as i64);
                assert_eq!(row.actor_cell[actor], 10 * position[1] + position[0]);
                assert_eq!(
                    row.actor_role[actor],
                    2 * role as i64 + i64::from(local > 0)
                );
                assert!(row.actor_mask[actor]);
                assert_eq!(
                    row.actors_float[actor][0],
                    (position[0] as f64 / 9.0) as f32
                );
                assert_eq!(
                    row.actors_float[actor][1],
                    (position[1] as f64 / 9.0) as f32
                );
                if role == 1 {
                    assert_eq!(&row.actors_float[actor][2..], &[0.0; 24]);
                }
            }
        }
        for (actor, inventory) in header.initial.privates[seat].inventories.iter().enumerate() {
            for (item, name) in TEST_ITEMS.iter().enumerate() {
                let expected_count = inventory.get(*name).copied().unwrap_or(0);
                let expected_rank = inventory
                    .get_index_of(*name)
                    .map_or(0, |rank| rank as i64 + 1);
                assert_eq!(row.actor_inventory[actor][item], expected_count);
                assert_eq!(row.actor_inventory_rank[actor][item], expected_rank);
                assert_eq!(
                    row.actors_float[actor][2 + item],
                    (expected_count as f64 / 32.0) as f32
                );
                assert_eq!(
                    row.actors_float[actor][14 + item],
                    (expected_rank as f64 / 12.0) as f32
                );
            }
        }
        assert_eq!(
            row.actor_mask[..241]
                .iter()
                .filter(|present| **present)
                .count() as i64,
            row.globals_int[14]
        );
        assert_eq!(
            row.actor_mask[241..]
                .iter()
                .filter(|present| **present)
                .count() as i64,
            row.globals_int[15]
        );
    }
}

#[test]
fn actors_rank_remove_reinsert_preserves_counts_and_replacement_preserves_rank() {
    let mut header = actor_header();
    let mut original = encode_header(&header).unwrap();
    {
        let mut buffers = original.buffers_mut();
        let env = buffers.envs_mut().next().unwrap();
        assert_eq!(env.seats[0].actor_inventory[0][7], 0);
        assert_eq!(env.seats[0].actor_inventory_rank[0][7], 1);
        assert_eq!(env.seats[0].actor_inventory[0][0], 7);
        assert_eq!(env.seats[0].actor_inventory_rank[0][0], 2);
        assert_eq!(env.seats[0].actor_inventory_rank[0][1], 0);
    }
    let mut replacement = header.clone();
    replacement.initial.privates[0].inventories[0].insert("WOOL".into(), 9);
    let mut updated = encode_header(&replacement).unwrap();
    {
        let mut a = original.buffers_mut();
        let mut b = updated.buffers_mut();
        let a = a.envs_mut().next().unwrap();
        let b = b.envs_mut().next().unwrap();
        assert_eq!(b.seats[0].actor_inventory[0][7], 9);
        assert_eq!(
            a.seats[0].actor_inventory_rank,
            b.seats[0].actor_inventory_rank
        );
    }
    let inventory = &mut header.initial.privates[0].inventories[0];
    inventory.insert("WOOL".into(), 0);
    assert_eq!(
        all_bytes(&mut encode_header(&header).unwrap()),
        all_bytes(&mut original)
    );
    let inventory = &mut header.initial.privates[0].inventories[0];
    let count = inventory.shift_remove("WOOL").unwrap();
    inventory.insert("WOOL".into(), count);
    let mut reordered = encode_header(&header).unwrap();
    let mut a = original.buffers_mut();
    let mut b = reordered.buffers_mut();
    let a = a.envs_mut().next().unwrap();
    let b = b.envs_mut().next().unwrap();
    assert_eq!(a.seats[0].actor_inventory, b.seats[0].actor_inventory);
    assert_eq!(b.seats[0].actor_inventory_rank[0][0], 1);
    assert_eq!(b.seats[0].actor_inventory_rank[0][7], 2);
    assert_eq!(b.seats[0].actor_inventory_rank[0][1], 0);
}

fn dense_actor_header() -> TraceHeader {
    let mut header = fresh_header(Config {
        farm_hand_cost_mult: 0_i64.into(),
        ..Config::default()
    });
    for seat in 0..2 {
        let farm = &mut header.initial.public.farms[seat];
        farm.farmer = vec![seat as i64, 9];
        farm.hands = (1..241)
            .map(|actor| vec![(actor % 10) as i64, ((actor / 10 + seat) % 10) as i64])
            .collect();
        farm.hires_today = 240;
        header.initial.privates[seat].inventories = (0..241)
            .map(|actor| {
                test_counts(&[
                    ("SHEEP", actor as i64),
                    ("WHEAT", 16_777_217 + actor as i64),
                ])
            })
            .collect();
        header.initial.privates[seat].inventories[240].insert("WHEAT".into(), i64::MAX);
    }
    header
}

#[test]
fn actors_all_241_slots_preserve_exact_counts_and_positions() {
    let header = dense_actor_header();
    let mut output = encode_header(&header).unwrap();
    let mut buffers = output.buffers_mut();
    let env = buffers.envs_mut().next().unwrap();
    for (seat, row) in env.seats.into_iter().enumerate() {
        assert!(row.actor_mask.iter().all(|present| *present));
        for (role, farm_index) in [seat, 1 - seat].into_iter().enumerate() {
            let farm = &header.initial.public.farms[farm_index];
            for (local, position) in std::iter::once(&farm.farmer).chain(&farm.hands).enumerate() {
                let actor = role * 241 + local;
                assert_eq!(row.actor_slot[actor], local as i64);
                assert_eq!(row.actor_cell[actor], position[1] * 10 + position[0]);
                assert_eq!(
                    row.actor_role[actor],
                    2 * role as i64 + i64::from(local > 0)
                );
            }
        }
        for actor in 0..241 {
            let count = if actor == 240 {
                i64::MAX
            } else {
                16_777_217 + actor as i64
            };
            assert_eq!(row.actor_inventory[actor][0], count);
            assert_eq!(row.actor_inventory_rank[actor][0], 2);
            assert_eq!(row.actor_inventory_rank[actor][11], 1);
            assert_eq!(row.actors_float[actor][2], (count as f64 / 32.0) as f32);
            assert!(row.actors_float[actor][2] > 4.0);
        }
        assert_eq!(row.globals_int[14..], [241, 241]);
    }
}

#[test]
fn actors_empty_farmer_is_present_and_dense_to_sparse_clears_padding() {
    let mut output = encode_header(&dense_actor_header()).unwrap();
    let allocations = output.allocation_layout();
    let sparse = fresh_header(Config::default());
    let game = ObservationGame::from_header(&sparse).unwrap();
    encode_env(&game, &mut output.buffers_mut().envs_mut().next().unwrap()).unwrap();
    assert_eq!(output.allocation_layout(), allocations);
    let mut buffers = output.buffers_mut();
    let env = buffers.envs_mut().next().unwrap();
    for row in env.seats {
        assert!(row.actor_mask[0]);
        assert!(row.actor_mask[241]);
        assert_eq!(*row.actor_inventory, [[0; 12]; 241]);
        assert_eq!(*row.actor_inventory_rank, [[0; 12]; 241]);
        for actor in 0..482 {
            if actor == 0 || actor == 241 {
                continue;
            }
            assert!(!row.actor_mask[actor]);
            assert_eq!(row.actor_slot[actor], 0);
            assert_eq!(row.actor_cell[actor], 0);
            assert_eq!(row.actor_role[actor], 0);
            assert_eq!(row.actors_float[actor], [0.0; 26]);
        }
    }
}

#[test]
fn actors_reject_malformed_positions_cardinality_and_private_counts_atomically() {
    for seat in 0..2 {
        for position in [
            vec![],
            vec![0],
            vec![0, 0, 0],
            vec![-1, 0],
            vec![0, -1],
            vec![10, 0],
            vec![0, 10],
            vec![i64::MAX, 0],
        ] {
            for farmer in [true, false] {
                let mut header = actor_header();
                if farmer {
                    header.initial.public.farms[seat].farmer = position.clone();
                } else {
                    header.initial.public.farms[seat].hands[0] = position.clone();
                }
                let error = rejected_header_preserves_output(&header);
                assert_eq!(error.kind, ObserveErrorKind::State);
                assert_eq!(
                    error.seat,
                    Some(if seat == 0 { Seat::Zero } else { Seat::One })
                );
            }
        }
        for extra in [false, true] {
            let mut header = actor_header();
            if extra {
                header.initial.privates[seat]
                    .inventories
                    .push(test_counts(&[]));
            } else {
                header.initial.privates[seat].inventories.pop();
            }
            assert_eq!(
                rejected_header_preserves_output(&header).kind,
                ObserveErrorKind::State
            );
        }
        for (name, amount, kind) in [
            ("UNKNOWN", 0, ObserveErrorKind::EnumRange),
            ("WHEAT", -1, ObserveErrorKind::State),
        ] {
            let mut header = actor_header();
            header.initial.privates[seat].inventories[0].insert(name.into(), amount);
            assert_eq!(rejected_header_preserves_output(&header).kind, kind);
        }
        let mut header = dense_actor_header();
        header.initial.public.farms[seat].hands.push(vec![0, 0]);
        header.initial.privates[seat]
            .inventories
            .push(test_counts(&[]));
        assert_eq!(
            rejected_header_preserves_output(&header).kind,
            ObserveErrorKind::State
        );
    }
}

#[test]
fn storage_native_zero_counts_have_all_twelve_insertion_ranks() {
    assert_eq!(&TEST_ITEMS[..9], &kaggriculture_engine::PRODUCTS);
    assert_eq!(&TEST_ITEMS[9..], &kaggriculture_engine::ANIMAL_NAMES);
    let mut output = encode_header(&fresh_header(Config::default())).unwrap();
    for env in output.buffers_mut().envs_mut() {
        for row in env.seats {
            assert_eq!(*row.storage_counts, [0; 17]);
            assert_eq!(*row.storage_rank, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
            for item in 0..12 {
                assert_eq!(
                    row.player_features[0][28 + item],
                    ((item + 1) as f64 / 12.0) as f32
                );
            }
            assert_eq!(row.player_features[0][40], 0.0);
            assert_eq!(row.player_features[0][41], 1.0);
        }
    }
}

#[test]
fn storage_rank_remove_reinsert_changes_order_not_counts() {
    let mut header = actor_header();
    let mut original = encode_header(&header).unwrap();
    {
        let mut buffers = original.buffers_mut();
        let env = buffers.envs_mut().next().unwrap();
        assert_eq!(
            *env.seats[0].storage_rank,
            [2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 3, 0]
        );
    }
    let mut replacement = header.clone();
    replacement.initial.privates[0]
        .shed
        .insert("WHEAT".into(), 19);
    let mut updated = encode_header(&replacement).unwrap();
    {
        let mut a = original.buffers_mut();
        let mut b = updated.buffers_mut();
        let a = a.envs_mut().next().unwrap();
        let b = b.envs_mut().next().unwrap();
        assert_eq!(b.seats[0].storage_counts[0], 19);
        assert_eq!(a.seats[0].storage_rank, b.seats[0].storage_rank);
    }
    header.initial.privates[0].shed.insert("WOOL".into(), 0);
    assert_eq!(
        all_bytes(&mut encode_header(&header).unwrap()),
        all_bytes(&mut original)
    );
    let shed = &mut header.initial.privates[0].shed;
    let count = shed.shift_remove("WOOL").unwrap();
    shed.insert("WOOL".into(), count);
    let mut reordered = encode_header(&header).unwrap();
    let mut a = original.buffers_mut();
    let mut b = reordered.buffers_mut();
    let a = a.envs_mut().next().unwrap();
    let b = b.envs_mut().next().unwrap();
    assert_eq!(a.seats[0].storage_counts, b.seats[0].storage_counts);
    assert_eq!(
        *b.seats[0].storage_rank,
        [1, 0, 0, 0, 0, 0, 0, 3, 0, 0, 2, 0]
    );
}

#[test]
fn storage_all_player_channels_use_public_roles_and_own_private_state() {
    let mut header = actor_header();
    header.initial.public.farms[0].unlocked_quadrants = vec!["SE".into(), "NW".into()];
    header.initial.public.farms[1].unlocked_quadrants = vec!["SW".into(), "NE".into(), "NW".into()];
    header.initial.public.farms[0].tiles[0][0] = serde_json::json!({"kind":"WEED"});
    header.initial.public.farms[0].hires_today = 2;
    header.initial.public.farms[1].hires_today = 4;
    header.initial.public.farms[0].money = -10.000_000_000_000_002;
    header.initial.public.farms[1].money = -10.0;
    assert_eq!(
        header.initial.public.farms[0].money as f32,
        header.initial.public.farms[1].money as f32
    );
    let mut output = encode_header(&header).unwrap();
    let mut buffers = output.buffers_mut();
    let env = buffers.envs_mut().next().unwrap();
    let public_expected = [
        [
            (-10.000_000_000_000_002_f64 / 200000.0) as f32,
            (3.0_f64 / 241.0) as f32,
            0.5,
            1.0,
            0.0,
            0.0,
            1.0,
            0.24,
            0.75,
            (2.0_f64 / 240.0) as f32,
            (2.0_f64 / 200000.0) as f32,
        ],
        [
            (-10.0_f64 / 200000.0) as f32,
            (2.0_f64 / 241.0) as f32,
            0.75,
            1.0,
            1.0,
            1.0,
            0.0,
            0.25,
            0.75,
            (4.0_f64 / 240.0) as f32,
            (5.0_f64 / 200000.0) as f32,
        ],
    ];
    for (seat, row) in env.seats.into_iter().enumerate() {
        for (role, farm_index) in [seat, 1 - seat].into_iter().enumerate() {
            assert_eq!(
                &row.player_features[role][..11],
                &public_expected[farm_index]
            );
            assert_eq!(
                row.banks[role].to_bits(),
                header.initial.public.farms[farm_index].money.to_bits()
            );
        }
        for (item, name) in TEST_ITEMS.iter().enumerate() {
            let private = &header.initial.privates[seat];
            let count = private.shed.get(*name).copied().unwrap_or(0);
            let rank = private
                .shed
                .get_index_of(*name)
                .map_or(0, |rank| rank as i64 + 1);
            assert_eq!(row.storage_counts[item], count);
            assert_eq!(row.storage_rank[item], rank);
            assert_eq!(
                row.player_features[0][11 + item],
                (count as f64 / 100.0) as f32
            );
            assert_eq!(
                row.player_features[0][28 + item],
                (rank as f64 / 12.0) as f32
            );
        }
        for (crop, name) in ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
            .iter()
            .enumerate()
        {
            let count = header.initial.privates[seat]
                .seeds
                .get(*name)
                .copied()
                .unwrap_or(0);
            assert_eq!(row.storage_counts[12 + crop], count);
            assert_eq!(
                row.player_features[0][23 + crop],
                (count as f64 / 32.0) as f32
            );
        }
        let total = [9.0_f64, 5.0][seat];
        assert_eq!(row.player_features[0][40], (total / 100.0) as f32);
        assert_eq!(row.player_features[0][41], ((100.0 - total) / 100.0) as f32);
        assert_eq!(&row.player_features[0][42..], &[0.0; 2]);
        assert_eq!(&row.player_features[1][11..], &[0.0; 33]);
    }
}

#[test]
fn storage_room_uses_wide_nonnegative_remainder_without_clamping_used_ratio() {
    for (total, used, room) in [(0, 0.0, 1.0), (7, 0.7, 0.3), (10, 1.0, 0.0), (15, 1.5, 0.0)] {
        let mut header = fresh_header(Config {
            shed_capacity: 10_i64.into(),
            ..Config::default()
        });
        header.initial.privates[0].shed = test_counts(&[("WHEAT", total)]);
        let mut output = encode_header(&header).unwrap();
        let mut buffers = output.buffers_mut();
        let env = buffers.envs_mut().next().unwrap();
        assert_eq!(env.seats[0].player_features[0][40..42], [used, room]);
    }
    let mut header = fresh_header(Config {
        shed_capacity: 1_i64.into(),
        ..Config::default()
    });
    header.initial.privates[0].shed = TEST_ITEMS
        .into_iter()
        .map(|name| (name.into(), i64::MAX))
        .collect();
    header.initial.privates[0].seeds = test_counts(&[("MELON", i64::MAX), ("WHEAT", 16_777_217)]);
    let mut output = encode_header(&header).unwrap();
    let mut buffers = output.buffers_mut();
    let env = buffers.envs_mut().next().unwrap();
    let row = &env.seats[0];
    assert_eq!(&row.storage_counts[..12], &[i64::MAX; 12]);
    assert_eq!(row.storage_counts[12], 16_777_217);
    assert_eq!(row.storage_counts[16], i64::MAX);
    assert_eq!(
        row.player_features[0][40],
        (110_680_464_442_257_309_684_f64) as f32
    );
    assert_eq!(row.player_features[0][41], 0.0);
    assert!(row.player_features[0].iter().all(|value| value.is_finite()));
}

#[test]
fn storage_rejects_unknown_items_negative_counts_and_bad_quadrants_atomically() {
    for seat in 0..2 {
        for seeds in [false, true] {
            for (name, count, kind) in [
                ("UNKNOWN", 0, ObserveErrorKind::EnumRange),
                ("WHEAT", -1, ObserveErrorKind::State),
            ] {
                let mut header = actor_header();
                let bag = if seeds {
                    &mut header.initial.privates[seat].seeds
                } else {
                    &mut header.initial.privates[seat].shed
                };
                bag.insert(name.into(), count);
                assert_eq!(rejected_header_preserves_output(&header).kind, kind);
            }
        }
        let mut header = actor_header();
        header.initial.privates[seat].seeds.insert("MILK".into(), 0);
        assert_eq!(
            rejected_header_preserves_output(&header).kind,
            ObserveErrorKind::EnumRange
        );
        for quadrants in [vec!["NW", "NW"], vec!["NW", "UNKNOWN"]] {
            let mut header = actor_header();
            header.initial.public.farms[seat].unlocked_quadrants =
                quadrants.into_iter().map(str::to_owned).collect();
            assert!(matches!(
                rejected_header_preserves_output(&header).kind,
                ObserveErrorKind::State | ObserveErrorKind::EnumRange
            ));
        }
    }
}

#[test]
fn privacy_private_values_and_key_order_do_not_cross_seats() {
    for changed in 0..2 {
        for reorder_only in [false, true] {
            let mut header = actor_header();
            let mut before = encode_header(&header).unwrap();
            let private = &mut header.initial.privates[changed];
            if reorder_only {
                for bag in [
                    &mut private.shed,
                    &mut private.seeds,
                    &mut private.inventories[0],
                ] {
                    let (name, count) = bag.shift_remove_index(0).unwrap();
                    bag.insert(name, count);
                }
            } else {
                private.shed.insert("WHEAT".into(), 99);
                private.seeds.insert("TOMATO".into(), 55);
                private.inventories[0].insert("GOOSE".into(), 17);
            }
            let mut after = encode_header(&header).unwrap();
            let owner = if changed == 0 { Seat::Zero } else { Seat::One };
            let observer = if changed == 0 { Seat::One } else { Seat::Zero };
            assert_eq!(
                field_bytes(&mut before, observer),
                field_bytes(&mut after, observer)
            );
            assert!(
                field_bytes(&mut before, owner) != field_bytes(&mut after, owner),
                "private owner row must change"
            );
        }
    }
}

#[test]
fn privacy_public_mutation_changes_both_rows_and_complete_seat_swap_is_symmetric() {
    let header = actor_header();
    let mut original = encode_header(&header).unwrap();
    for changed in 0..2 {
        let mut altered = header.clone();
        altered.initial.public.farms[changed].farmer[0] = 0;
        let mut output = encode_header(&altered).unwrap();
        for seat in [Seat::Zero, Seat::One] {
            assert!(
                field_bytes(&mut original, seat) != field_bytes(&mut output, seat),
                "public position mutation must change both rows"
            );
        }
        let mut buffers = output.buffers_mut();
        let env = buffers.envs_mut().next().unwrap();
        assert_eq!(
            env.seats[changed].actor_cell[0],
            10 * header.initial.public.farms[changed].farmer[1]
        );
        assert_eq!(
            env.seats[1 - changed].actor_cell[241],
            10 * header.initial.public.farms[changed].farmer[1]
        );
    }
    let mut swapped = header.clone();
    swapped.initial.public.farms.swap(0, 1);
    swapped.initial.privates.swap(0, 1);
    let mut output = encode_header(&swapped).unwrap();
    assert_eq!(
        field_bytes(&mut original, Seat::Zero),
        field_bytes(&mut output, Seat::One)
    );
    assert_eq!(
        field_bytes(&mut original, Seat::One),
        field_bytes(&mut output, Seat::Zero)
    );
}

#[test]
fn context_shops_preserve_order_duplicates_zero_categories_and_padding() {
    let mut header = fresh_header(Config::default());
    header.initial.public.town.unlocked_shops =
        vec!["PIZZA_SHOP".into(), "BAKERY".into(), "PIZZA_SHOP".into()];
    let mut output = encode_header(&header).unwrap();
    for env in output.buffers_mut().envs_mut() {
        for row in env.seats {
            assert_eq!(*row.shop_type, [5, 0, 5, 0, 0, 0, 0, 0]);
            assert_eq!(*row.shop_slot, [0, 1, 2, 0, 0, 0, 0, 0]);
            assert_eq!(
                *row.shop_mask,
                [true, true, true, false, false, false, false, false]
            );
            assert_eq!(row.global_features[14], (3.0_f64 / 8.0) as f32);
        }
    }
    header.initial.public.town.unlocked_shops = [
        "BAKERY",
        "BRUNCH_SPOT",
        "FARMERS_MARKET",
        "ICE_CREAM_SHOP",
        "PET_CAFE",
        "PIZZA_SHOP",
        "SMOOTHIE_SHOP",
        "YARN_STORE",
    ]
    .map(String::from)
    .into();
    let mut output = encode_header(&header).unwrap();
    for env in output.buffers_mut().envs_mut() {
        for row in env.seats {
            assert_eq!(*row.shop_type, [0, 1, 2, 3, 4, 5, 6, 7]);
            assert!(row.shop_mask.iter().all(|present| *present));
        }
    }
    header.initial.public.town.unlocked_shops[3] = "UNRECOGNIZED_SHOP".into();
    assert_eq!(
        rejected_header_preserves_output(&header).kind,
        ObserveErrorKind::EnumRange
    );
}

#[test]
fn context_market_preserves_signed_exact_integers_and_unbounded_scales() {
    let mut header = fresh_header(Config::default());
    header
        .initial
        .public
        .market
        .inventory
        .insert("WHEAT".into(), serde_json::json!(-12345));
    header
        .initial
        .public
        .market
        .prices
        .insert("WHEAT".into(), serde_json::json!(7777));
    header
        .initial
        .public
        .market
        .inventory
        .insert("FERTILIZER".into(), serde_json::json!(i64::MIN));
    header
        .initial
        .public
        .market
        .prices
        .insert("FERTILIZER".into(), serde_json::json!(i64::MAX));
    header
        .initial
        .public
        .market
        .prices
        .insert("WOOL".into(), serde_json::json!(-123));
    let mut output = encode_header(&header).unwrap();
    for env in output.buffers_mut().envs_mut() {
        for row in env.seats {
            assert_eq!(*row.market_product, [0, 1, 2, 3, 4, 5, 6, 7, 8]);
            assert_eq!(row.market_int[0], [-12345, 7777]);
            assert_eq!(
                row.market_float[0],
                [(-12345.0_f64 / 10000.0) as f32, (7777.0_f64 / 250.0) as f32]
            );
            assert_eq!(row.market_int[8], [i64::MIN, i64::MAX]);
            assert_eq!(
                row.market_float[8],
                [
                    (i64::MIN as f64 / 10000.0) as f32,
                    (i64::MAX as f64 / 250.0) as f32
                ]
            );
            assert_eq!(row.market_int[7][1], -123);
        }
    }
}

#[test]
fn context_market_rejects_missing_unknown_fractional_and_wrong_type_values() {
    for channel in 0..2 {
        for value in [
            serde_json::json!(1.5),
            serde_json::json!(1.0),
            serde_json::json!("1"),
            serde_json::json!(true),
            serde_json::Value::Null,
            serde_json::from_str("9223372036854775808").unwrap(),
            serde_json::from_str("-9223372036854775809").unwrap(),
        ] {
            let mut header = fresh_header(Config::default());
            let market = &mut header.initial.public.market;
            let values = if channel == 0 {
                &mut market.inventory
            } else {
                &mut market.prices
            };
            values.insert("WHEAT".into(), value);
            let error = rejected_header_preserves_output(&header);
            assert_eq!(error.field, format!("market_int[WHEAT,{channel}]"));
        }
        for missing in [true, false] {
            let mut header = fresh_header(Config::default());
            let market = &mut header.initial.public.market;
            let values = if channel == 0 {
                &mut market.inventory
            } else {
                &mut market.prices
            };
            if missing {
                values.shift_remove("WHEAT");
            } else {
                values.insert("UNKNOWN".into(), serde_json::json!(1));
            }
            rejected_header_preserves_output(&header);
        }
    }
}

#[test]
fn context_order_limits_follow_rule_not_hires_cash_or_actor_count() {
    let mut header = actor_header();
    header.configuration.max_market_orders_per_turn = 3_i64.into();
    for (seat, farm) in header.initial.public.farms.iter_mut().enumerate() {
        farm.hires_today = 100 + seat;
        farm.money = 0.0;
    }
    let mut output = encode_header(&header).unwrap();
    for env in output.buffers_mut().envs_mut() {
        for (seat, row) in env.seats.into_iter().enumerate() {
            assert_eq!(*row.order_limits, 3);
            let actor_count = [3, 2][seat];
            for (frame, present) in row.can_act.iter().enumerate() {
                assert_eq!(*present, frame < actor_count + 3 + 1);
            }
            assert!(*row.still_playing);
        }
    }
    let mut game = Game::new(
        Config {
            max_market_orders_per_turn: 3_i64.into(),
            ..Config::default()
        },
        42,
        2,
    )
    .unwrap();
    game.step(&[
        serde_json::json!({"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"]]}),
        serde_json::json!({}),
    ])
    .unwrap();
    let snapshot = game.snapshot();
    assert_eq!(snapshot.public.farms[0].hands.len(), 3);
    assert_eq!(snapshot.public.farms[0].hires_today, 3);
    assert_eq!(snapshot.public.farms[0].money, 2996.0);
}

#[test]
fn context_can_act_boundaries_and_terminal_snapshots_are_live_rows() {
    let mut header = fresh_header(Config {
        max_market_orders_per_turn: 3_i64.into(),
        ..Config::default()
    });
    for dense in [false, true] {
        if dense {
            header = dense_actor_header();
        }
        let mut output = encode_header(&header).unwrap();
        for env in output.buffers_mut().envs_mut() {
            for row in env.seats {
                assert_eq!(
                    row.can_act.iter().filter(|present| **present).count(),
                    if dense { 252 } else { 5 }
                );
                assert!(*row.still_playing);
            }
        }
    }
    for episode in [1_i64, 2] {
        let mut game = ObservationGame::from_seed(
            Config {
                episode_steps: episode.into(),
                ..Config::default()
            },
            "42",
        )
        .unwrap();
        game.step_with_market_metrics(&[serde_json::json!({}), serde_json::json!({})])
            .unwrap();
        assert!(game.game().snapshot().done);
        let mut output = ObsStaging::new(1).unwrap();
        encode_env(&game, &mut output.buffers_mut().envs_mut().next().unwrap()).unwrap();
        for env in output.buffers_mut().envs_mut() {
            for row in env.seats {
                assert_eq!(row.globals_int[0], 1);
                assert_eq!(row.global_features[3], 0.0);
                assert!(*row.still_playing);
                assert_eq!(row.can_act.iter().filter(|present| **present).count(), 12);
            }
        }
    }
}

#[test]
fn context_check_row_rejects_corruptions_in_every_domain() {
    let mut header = actor_header();
    header.initial.public.farms[0].tiles[0][0] = tile_plant("TOMATO");
    header.initial.public.farms[0].tiles[0][1] = tile_animal("GOOSE");
    header.initial.public.town.unlocked_shops =
        vec!["PIZZA_SHOP".into(), "BAKERY".into(), "PIZZA_SHOP".into()];
    let mut missed = Vec::new();
    for case in 0..72 {
        let mut output = encode_header(&header).unwrap();
        let mut buffers = output.buffers_mut();
        let mut env = buffers.envs_mut().next().unwrap();
        let row = &mut env.seats[0];
        super::check_row(row).unwrap();
        match case {
            0 => row.tile_kind[0] = -1,
            1 => row.tile_kind[0] = 6,
            2 => row.tile_crop[0] = 6,
            3 => row.tile_animal[0] = 4,
            4 => row.tile_cell[0] = 100,
            5 => row.tile_role[0] = 2,
            6 => row.actor_slot[10] = 241,
            7 => row.actor_cell[10] = 100,
            8 => row.actor_role[10] = 4,
            9 => row.shop_type[7] = 8,
            10 => row.shop_slot[7] = 8,
            11 => row.market_product[0] = 9,
            12 => row.tiles_int[0][1] = -2,
            13 => row.tiles_int[2][1] = -1,
            14 => row.tiles_int[0][0] = -1,
            15 => row.actor_inventory_rank[0][0] = 1,
            16 => row.actor_inventory_rank[0][0] = 3,
            17 => row.actor_inventory_rank[0][0] = 0,
            18 => row.storage_rank[0] = 1,
            19 => row.storage_rank[0] = 4,
            20 => row.actors_float[241][2] = 1.0,
            21 => row.player_features[1][11] = 1.0,
            22 => row.can_act[251] = true,
            23 => row.actor_mask[1] = false,
            24 => row.globals_int[14] = 4,
            25 => row.actors_float[0][0] = 0.0,
            26 => row.actor_inventory[0][0] = -1,
            27 => row.storage_counts[12] = -1,
            28 => row.banks[0] = f64::INFINITY,
            29 => row.tiles_float[0][0] = f32::NAN,
            30 => row.actors_float[0][0] = f32::NAN,
            31 => row.player_features[0][0] = f32::NAN,
            32 => row.market_float[0][0] = f32::NAN,
            33 => row.global_features[0] = f32::NAN,
            34 => row.actor_slot[10] = 1,
            35 => row.actors_float[10][0] = -0.0,
            36 => row.shop_type[7] = 3,
            37 => row.tiles_float[2][1] = -0.0,
            38 => row.player_features[0][43] = -0.0,
            39 => *row.still_playing = false,
            40 => *row.order_limits = 0,
            41 => row.global_features[4] = 10.0,
            42 => row.globals_int[2] = 24,
            43 => row.globals_int[12] = -1,
            44 => row.player_features[0][10] = 100.0,
            45 => row.player_features[0][0] = 100.0,
            46 => row.tiles_float[0][0] = 100.0,
            47 => row.tile_crop[2] = 1,
            48 => row.tile_kind[1] = 5,
            49 => row.shop_mask[1] = false,
            50 => row.shop_slot[2] = 1,
            51 => row.market_float[0][1] = 100.0,
            52 => row.player_features[0][11] = 100.0,
            53 => row.player_features[0][2] = 100.0,
            54 => row.global_features[14] = 100.0,
            55 => row.globals_int[6] = 0,
            56 => row.actor_inventory_rank[10][0] = 1,
            57 => row.player_features[1][28] = -0.0,
            58 => row.tiles_float[0][1] = 0.5,
            59 => row.tiles_float[1][12] = 0.5,
            60 => row.tiles_float[1][14] = -1.0,
            61 => row.actors_float[0][14] = 0.0,
            62 => row.storage_rank[0] = 13,
            63 => row.player_features[0][7] = 100.0,
            64 => row.global_features[3] = 100.0,
            65 => row.player_features[1][42] = 1.0,
            66 => row.tiles_float[1][14] = 0.1,
            67 => row.actor_slot[10] = -1,
            68 => row.actor_cell[10] = -1,
            69 => row.actor_role[10] = -1,
            70 => row.shop_type[7] = -1,
            71 => row.shop_slot[7] = -1,
            _ => unreachable!(),
        }
        if super::check_row(row).is_ok() {
            missed.push(case);
        }
    }
    assert!(missed.is_empty(), "accepted corruption cases {missed:?}");
}

fn stage_headers_for_test(
    headers: &[TraceHeader],
    staging: &mut ObsStaging,
    published: &mut FlatBuffers,
    parallel: bool,
) -> Result<(), ObserveError> {
    let rows = staging.buffers_mut().envs_mut().count();
    if rows != headers.len() {
        return Err(ObserveError::new(
            ObserveErrorKind::Shape,
            "n_envs",
            format!(
                "staging n_envs={rows} differs from headers={}",
                headers.len()
            ),
        ));
    }
    let mut output = published.borrow().validate(headers.len())?;
    let encode = |(env, (header, mut row)): (usize, (&TraceHeader, super::ObsEnvMut<'_>))| {
        let result =
            ObservationGame::from_header(header).and_then(|game| encode_env(&game, &mut row));
        result.map_err(|mut error| {
            error.env = Some(env);
            error
        })
    };
    let results: Vec<Result<(), ObserveError>> = if parallel {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        pool.install(|| {
            headers
                .par_iter()
                .zip_eq(staging.buffers_mut().par_envs_mut())
                .enumerate()
                .map(encode)
                .collect()
        })
    } else {
        headers
            .iter()
            .zip(staging.buffers_mut().envs_mut())
            .enumerate()
            .map(encode)
            .collect()
    };
    for result in results {
        result?;
    }
    staging.publish(&mut output)
}

fn assert_published_rows(output: &mut FlatBuffers, n_envs: usize) {
    for env in output.borrow().validate(n_envs).unwrap().envs_mut() {
        for row in env.seats {
            super::check_row(&row).unwrap();
        }
    }
}

#[test]
fn transaction_late_market_failure_preserves_both_environments_then_serial_equals_rayon() {
    let mut headers = [actor_header(), dense_actor_header()];
    headers[1]
        .initial
        .public
        .market
        .prices
        .insert("WHEAT".into(), serde_json::json!(1.5));
    let mut staging = ObsStaging::new(2).unwrap();
    let mut serial = FlatBuffers::new(2);
    let before = serial.bytes();
    for parallel in [false, true] {
        let error =
            stage_headers_for_test(&headers, &mut staging, &mut serial, parallel).unwrap_err();
        assert_eq!(error.env, Some(1));
        assert_eq!(error.field, "market_int[WHEAT,1]");
        assert_eq!(serial.bytes(), before);
    }
    headers[1]
        .initial
        .public
        .market
        .prices
        .insert("WHEAT".into(), serde_json::json!(7777));
    stage_headers_for_test(&headers, &mut staging, &mut serial, false).unwrap();
    let mut parallel = FlatBuffers::new(2);
    stage_headers_for_test(&headers, &mut staging, &mut parallel, true).unwrap();
    assert_published_rows(&mut serial, 2);
    assert_published_rows(&mut parallel, 2);
    assert_eq!(serial.bytes(), parallel.bytes());
    assert_ne!(serial.bytes(), before);
    let before = parallel.bytes();
    let mismatch = ObsStaging::new(1)
        .unwrap()
        .publish(&mut parallel.borrow().validate(2).unwrap())
        .unwrap_err();
    assert_eq!(mismatch.kind, ObserveErrorKind::Shape);
    assert_eq!(parallel.bytes(), before);
    let error = stage_headers_for_test(
        &headers,
        &mut ObsStaging::new(1).unwrap(),
        &mut parallel,
        false,
    )
    .unwrap_err();
    assert_eq!(error.kind, ObserveErrorKind::Shape);
    assert_eq!(parallel.bytes(), before);
}

#[test]
fn transaction_dense_to_sparse_reuse_clears_every_field() {
    let mut dense = dense_actor_header();
    dense.initial.public.town.unlocked_shops = vec!["PIZZA_SHOP".into(); 8];
    dense.initial.public.farms[0].tiles[0][0] = tile_plant("TOMATO");
    dense.initial.public.farms[1].tiles[0][1] = tile_animal("GOOSE");
    let sparse = fresh_header(Config::default());
    let mut staging = ObsStaging::new(2).unwrap();
    let mut reused = FlatBuffers::new(2);
    stage_headers_for_test(&[dense.clone(), dense], &mut staging, &mut reused, true).unwrap();
    let mut reference = FlatBuffers::new(2);
    let mut fresh = ObsStaging::new(2).unwrap();
    stage_headers_for_test(
        &[sparse.clone(), sparse.clone()],
        &mut staging,
        &mut reused,
        true,
    )
    .unwrap();
    stage_headers_for_test(&[sparse.clone(), sparse], &mut fresh, &mut reference, false).unwrap();
    assert_published_rows(&mut reused, 2);
    assert_published_rows(&mut reference, 2);
    assert_eq!(reused.bytes(), reference.bytes());
}

#[test]
fn added_facts_order_only_pair_keeps_counts_and_legacy_but_changes_exact_ranks() {
    let mut before = encode_header(&super::oracle_corpus::dense_header(30).unwrap()).unwrap();
    let mut after = encode_header(&super::oracle_corpus::dense_header(31).unwrap()).unwrap();
    let mut before_buffers = before.buffers_mut();
    let mut after_buffers = after.buffers_mut();
    let before_env = before_buffers.envs_mut().next().unwrap();
    let after_env = after_buffers.envs_mut().next().unwrap();
    for (seat, (left, right)) in before_env.seats.iter().zip(after_env.seats).enumerate() {
        assert_eq!(left.actor_inventory, right.actor_inventory);
        assert_eq!(left.storage_counts, right.storage_counts);
        let old_before = super::oracle_corpus::reconstruct_legacy(left);
        let old_after = super::oracle_corpus::reconstruct_legacy(&right);
        for offset in 0..8176 {
            assert_eq!(
                old_before[offset].to_bits(),
                old_after[offset].to_bits(),
                "order-only pair seat={seat} offset={offset}"
            );
        }
        if seat == 0 {
            // d=30 first inserts MILK (Item 6), including its deliberate zero.
            // Farmer 0 omits one key; its 11 ranks and all 12 shed ranks compact.
            assert_eq!(
                (left.actor_inventory[0][6], right.actor_inventory[0][6]),
                (0, 0)
            );
            assert_eq!(
                (
                    left.actor_inventory_rank[0][6],
                    right.actor_inventory_rank[0][6]
                ),
                (1, 11)
            );
            assert_eq!((left.storage_counts[6], right.storage_counts[6]), (0, 0));
            assert_eq!((left.storage_rank[6], right.storage_rank[6]), (1, 12));
            assert_eq!(left.actors_float[0][20], (1.0_f64 / 12.0) as f32);
            assert_eq!(right.actors_float[0][20], (11.0_f64 / 12.0) as f32);
            assert_eq!(left.player_features[0][34], (1.0_f64 / 12.0) as f32);
            assert_eq!(right.player_features[0][34], 1.0);
            for item in 0..12 {
                if item == 6 {
                    continue;
                }
                let prior = left.actor_inventory_rank[0][item];
                assert_eq!(right.actor_inventory_rank[0][item], (prior - 1).max(0));
                assert_eq!(right.storage_rank[item], left.storage_rank[item] - 1);
            }
            for item in 0..12 {
                assert_eq!(
                    right.actors_float[0][14 + item],
                    (right.actor_inventory_rank[0][item] as f64 / 12.0) as f32
                );
                assert_eq!(
                    right.player_features[0][28 + item],
                    (right.storage_rank[item] as f64 / 12.0) as f32
                );
            }
            // Exact ranks and their two scaled duplicates are the only changes.
            *right.actor_inventory_rank = *left.actor_inventory_rank;
            *right.storage_rank = *left.storage_rank;
            right.actors_float[0][14..26].copy_from_slice(&left.actors_float[0][14..26]);
            right.player_features[0][28..40].copy_from_slice(&left.player_features[0][28..40]);
        }
        assert!(
            row_bytes(left) == row_bytes(&right),
            "seat={seat} bytes differ after rank normalization"
        );
    }
}

#[test]
fn added_facts_legacy_omits_rule_inputs_and_public_rival_hires() {
    let original = fresh_header(Config::default());
    let mut changed = original.clone();
    changed.configuration.weed_spawn_chance = serde_json::json!(2.0);
    changed.configuration.town_shop_unlock_interval = 5_i64.into();
    changed.configuration.town_shop_sell_interval = 11_i64.into();
    changed.configuration.town_center_sell_interval = 13_i64.into();
    changed.configuration.starting_money = 777_777_i64.into();
    changed.initial.public.farms[1].hires_today = 5;
    let mut before = encode_header(&original).unwrap();
    let mut after = encode_header(&changed).unwrap();
    let mut before_buffers = before.buffers_mut();
    let mut after_buffers = after.buffers_mut();
    let left_env = before_buffers.envs_mut().next().unwrap();
    let right_env = after_buffers.envs_mut().next().unwrap();
    let left = &left_env.seats[0];
    let right = &right_env.seats[0];
    let old_before = super::oracle_corpus::reconstruct_legacy(left);
    let old_after = super::oracle_corpus::reconstruct_legacy(right);
    for offset in 0..8176 {
        assert_eq!(
            old_before[offset].to_bits(),
            old_after[offset].to_bits(),
            "offset={offset}"
        );
    }
    assert_eq!(right.globals_int[8..14], [5, 11, 13, 777_777, 0, 5]);
    assert_eq!(
        right.global_features[9..14],
        [
            2.0_f64,
            5.0 / 24.0,
            11.0 / 24.0,
            13.0 / 24.0,
            777_777.0 / 200_000.0
        ]
        .map(|value| value as f32)
    );
    assert_eq!(
        right.banks, left.banks,
        "starting money is a rule input, not a bank overwrite"
    );
    assert_eq!(
        right.player_features[1][10].to_bits(),
        ((8.0_f64 / 200_000.0) as f32).to_bits()
    );
    assert_ne!(
        right.player_features[1][10].to_bits(),
        left.player_features[1][10].to_bits()
    );
    // The other seat's own-hire suffix does retain its changed public count.
    assert_eq!(
        super::oracle_corpus::reconstruct_legacy(&right_env.seats[1])[8171],
        5.0
    );
}

#[test]
fn added_facts_hire_cost_bits_match_an_independent_python_integer_loop() {
    let program = r#"
import json, struct
def exact_cost(mult, hires):
    a, b = 1, 1
    for _ in range(hires):
        a, b = b, a + b
    return mult * a
def bits(cost):
    return struct.unpack('<I', struct.pack('<f', float(cost) / 200000))[0]
print(json.dumps([[m, n, str(exact_cost(m, n)), bits(exact_cost(m, n)),
                   bits(exact_cost(m, n + 1))]
                  for m in (7, 0, 999999)
                  for n in (0, 1, 2, 3, 4, 5, 30, 80, 160)]))
"#;
    let output = std::process::Command::new("python3")
        .args(["-c", program])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Vec<(i64, usize, String, u32, u32)> =
        serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(cases.len(), 27);
    for (multiplier, hires, exact, own_bits, rival_bits) in cases {
        if multiplier == 7 && hires < 6 {
            assert_eq!(exact, [7, 7, 14, 21, 35, 56][hires].to_string());
        }
        let mut header = fresh_header(Config {
            farm_hand_cost_mult: multiplier.into(),
            ..Config::default()
        });
        header.initial.public.farms[0].hires_today = hires;
        header.initial.public.farms[1].hires_today = hires + 1;
        let mut output = encode_header(&header).unwrap();
        for env in output.buffers_mut().envs_mut() {
            for (seat, row) in env.seats.iter().enumerate() {
                let expected = if seat == 0 {
                    [own_bits, rival_bits]
                } else {
                    [rival_bits, own_bits]
                };
                for (role, bits) in expected.into_iter().enumerate() {
                    assert_eq!(
                        row.player_features[role][10].to_bits(),
                        bits,
                        "multiplier={multiplier} hires={hires} seat={seat} role={role}"
                    );
                }
            }
        }
    }
}

#[test]
fn added_facts_large_exact_values_and_wide_totals_exceed_legacy_admission() {
    let mut header = fresh_header(Config {
        episode_steps: i64::MAX.into(),
        starting_money: i64::MAX.into(),
        shed_capacity: i64::MAX.into(),
        farm_hand_cost_mult: 0_i64.into(),
        town_shop_unlock_interval: i64::MAX.into(),
        town_shop_sell_interval: i64::MAX.into(),
        town_center_sell_interval: i64::MAX.into(),
        weed_spawn_chance: serde_json::json!(1.5),
        ..Config::default()
    });
    let step = usize::try_from(i64::MAX - 2).unwrap();
    header.initial.public.step = step;
    header.initial.public.day = step / 24;
    header.initial.public.hour = step % 24;
    header.initial.public.farms[0].hires_today = usize::try_from(i64::MAX).unwrap();
    header.initial.public.farms[0].unlocked_quadrants = vec!["SE".into(), "NE".into()];
    header.initial.privates[0].inventories[0] =
        test_counts(&[("WHEAT", 16_777_217), ("MILK", i64::MAX)]);
    header.initial.privates[0].shed = TEST_ITEMS
        .into_iter()
        .map(|item| (item.into(), i64::MAX))
        .collect();
    assert!(super::oracle_corpus::validate_legacy_domain(&header).is_err());
    let mut output = encode_header(&header).unwrap();
    let mut buffers = output.buffers_mut();
    let env = buffers.envs_mut().next().unwrap();
    let row = &env.seats[0];
    assert_eq!(row.globals_int[0], i64::MAX - 2);
    assert_eq!(row.globals_int[3], i64::MAX);
    assert_eq!(row.globals_int[12], i64::MAX);
    assert_eq!(row.actor_inventory[0][0], 16_777_217);
    assert_eq!(row.actor_inventory[0][6], i64::MAX);
    assert_eq!(
        (
            row.actor_inventory_rank[0][0],
            row.actor_inventory_rank[0][6]
        ),
        (1, 2)
    );
    assert_eq!(&row.storage_counts[..12], &[i64::MAX; 12]);
    assert_eq!(row.player_features[0][40..42], [12.0, 0.0]);
    assert_eq!(row.player_features[0][10].to_bits(), 0);
    assert_eq!(row.player_features[0][3..7], [0.0, 1.0, 0.0, 1.0]);
    assert_eq!(
        row.global_features[3].to_bits(),
        ((1.0_f64 / i64::MAX as f64) as f32).to_bits()
    );
    assert!(row.global_features.iter().all(|value| value.is_finite()));

    let mut near = fresh_header(Config {
        farm_hand_cost_mult: 7_i64.into(),
        ..Config::default()
    });
    near.initial.public.farms[0].hires_today = 206;
    encode_header(&near).unwrap();
    near.initial.public.farms[1].hires_today = 207;
    let error = rejected_header_preserves_output(&near);
    assert_eq!(error.kind, ObserveErrorKind::NonFinite);
    assert_eq!(error.field, "player_features[1,10]");
}

#[test]
fn added_facts_remaining_transitions_follow_actual_small_episode_stop_rule() {
    for episode in [1_i64, 2, 5] {
        let mut game = ObservationGame::from_seed(
            Config {
                episode_steps: episode.into(),
                ..Config::default()
            },
            "42",
        )
        .unwrap();
        let terminal_step = if episode <= 2 { 1 } else { 4 };
        let mut output = ObsStaging::new(1).unwrap();
        for step in 0..=terminal_step {
            let snapshot = game.game().snapshot();
            assert_eq!(snapshot.done, step == terminal_step);
            encode_env(&game, &mut output.buffers_mut().envs_mut().next().unwrap()).unwrap();
            for env in output.buffers_mut().envs_mut() {
                for row in env.seats {
                    assert_eq!(row.globals_int[0], step);
                    assert_eq!(
                        row.global_features[3].to_bits(),
                        (((terminal_step - step) as f64 / episode as f64) as f32).to_bits()
                    );
                    assert!(*row.still_playing);
                }
            }
            if step < terminal_step {
                game.step_with_market_metrics(&[serde_json::json!({}), serde_json::json!({})])
                    .unwrap();
            }
        }
    }
}
