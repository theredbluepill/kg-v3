//! Bounded optimized phase costs; diagnostic states, never training throughput.
use super::{write_env, ObsStaging, ObservationGame};
use serde_json::{json, Value};
use std::hint::black_box;
use std::time::Instant;

fn fixture(dense: bool) -> (ObservationGame, Vec<Value>, String) {
    let header = super::oracle_corpus::timing_header(dense).unwrap();
    let hash = super::oracle_corpus::hash_bytes(&serde_json::to_vec(&header).unwrap()).unwrap();
    let actions = header.initial.public.farms.iter().map(|farm| {
        json!({"farmer":["PASS"],"hands":vec![json!(["PASS"]);farm.hands.len()],"market":[]})
    }).collect();
    (
        ObservationGame::from_header(&header).unwrap(),
        actions,
        hash,
    )
}

#[test]
fn lifecycle_timing_fixtures_step_and_prepare() {
    for dense in [false, true] {
        let (mut game, actions, _) = fixture(dense);
        assert_eq!(
            actions[0]["hands"].as_array().unwrap().len(),
            if dense { 240 } else { 0 }
        );
        game.step_with_market_metrics(&actions).unwrap();
        let prepared = game.prepare().unwrap();
        let mut output = ObsStaging::new(1).unwrap();
        let mut buffers = output.buffers_mut();
        let mut rows = buffers.envs_mut().next().unwrap();
        write_env(&prepared, &mut rows);
        for row in &rows.seats {
            super::check_row(row).unwrap();
        }
    }
}

fn measure(name: &str, deadline: Instant, mut sample: impl FnMut() -> u128) -> Value {
    for _ in 0..20 {
        assert!(deadline.elapsed().as_secs_f64() < 110.0);
        black_box(sample());
    }
    let mut samples = Vec::with_capacity(200);
    for _ in 0..200 {
        assert!(deadline.elapsed().as_secs_f64() < 110.0);
        samples.push(black_box(sample()));
    }
    let total: u128 = samples.iter().sum();
    samples.sort_unstable();
    json!({"phase":name,"warmups":20,"samples":200,
           "total_nanoseconds":total,"mean_nanoseconds":total as f64/200.,
           "median_nanoseconds":(samples[99]+samples[100]) as f64/2.})
}

#[test]
#[ignore = "optimized-only one-game lifecycle phase diagnostic; bounded externally"]
fn measure_lifecycle() {
    assert!(
        !black_box(cfg!(debug_assertions)),
        "measure_lifecycle requires release; never report debug timing"
    );
    let start = Instant::now();
    let mut states = Vec::new();
    for dense in [false, true] {
        let (game, actions, hash) = fixture(dense);
        let snapshot = game.acquire_snapshot();
        let prepared = game.prepare().unwrap();
        let mut output = ObsStaging::new(1).unwrap();
        let phases = vec![
            measure("outer_candidate_clone", start, || {
                let time = Instant::now();
                let copy = black_box(game.clone());
                let ns = time.elapsed().as_nanos();
                drop(copy);
                ns
            }),
            measure("kernel_step_including_inner_clone", start, || {
                // Set up the same candidate outside the timed region each time.
                let mut candidate = game.clone();
                let time = Instant::now();
                black_box(candidate.step_with_market_metrics(&actions).unwrap());
                time.elapsed().as_nanos()
            }),
            measure("prepare_snapshot_acquisition", start, || {
                let time = Instant::now();
                let acquired = black_box(game.acquire_snapshot());
                let ns = time.elapsed().as_nanos();
                drop(acquired);
                ns
            }),
            measure("validate_existing_snapshot", start, || {
                let time = Instant::now();
                game.validate_existing_snapshot(black_box(&snapshot))
                    .unwrap();
                time.elapsed().as_nanos()
            }),
            measure("write_prepared_both_seats", start, || {
                let mut buffers = output.buffers_mut();
                let mut rows = buffers.envs_mut().next().unwrap();
                let time = Instant::now();
                write_env(black_box(&prepared), &mut rows);
                time.elapsed().as_nanos()
            }),
            measure("outer_clone_kernel_step_prepare_write", start, || {
                let mut buffers = output.buffers_mut();
                let mut rows = buffers.envs_mut().next().unwrap();
                let time = Instant::now();
                let mut candidate = game.clone();
                candidate.step_with_market_metrics(&actions).unwrap();
                let next = candidate.prepare().unwrap();
                write_env(&next, &mut rows);
                time.elapsed().as_nanos()
            }),
        ];
        states.push(json!({"state":if dense {"dense_241_actors"} else {"early"},
                           "header_sha256":hash,"phases":phases}));
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut hashes = serde_json::Map::new();
    for path in [
        "Cargo.toml",
        "Cargo.lock",
        "src/kaggriculture/env.rs",
        "src/kaggriculture/reward.rs",
        "src/kaggriculture/admission.rs",
        "src/kaggriculture/observe.rs",
        "src/kaggriculture/buffers.rs",
        "src/kaggriculture/oracle_corpus.rs",
        "src/kaggriculture/lifecycle_timing_tests.rs",
        "engine_rs/src/lib.rs",
    ] {
        hashes.insert(
            path.into(),
            super::oracle_corpus::hash_bytes(&std::fs::read(root.join(path)).unwrap())
                .unwrap()
                .into(),
        );
    }
    println!(
        "LIFECYCLE_TIMING={}",
        json!({"profile":"release","fat_lto":true,
        "codegen_units":1,"logical_environments_at_a_time":1,"native_threads":1,
        "warmups":20,"samples":200,"test_body_seconds":start.elapsed().as_secs_f64(),
        "overflow_policy_label":std::env::var("KG_OVERFLOW_CHECKS_LABEL").unwrap_or_else(|_|"enabled".into()),
        "scope":"phase costs only; no model, Python, pool dispatch, or throughput claim; candidate teardown excluded",
        "source_sha256":hashes,"measurements":states})
    );
}
