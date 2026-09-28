use kaggriculture_engine::{Game, ReplayInput, TraceHeader, Transition};
use serde::Deserialize;
use serde_json::json;
use std::collections::HashSet;
use std::env;
use std::error::Error;
use std::hint::black_box;
use std::io::{self, BufWriter, Read, Write};
use std::time::Instant;

#[derive(Debug, Deserialize)]
struct Branch {
    id: String,
    transitions: Vec<Transition>,
}

#[derive(Debug, Deserialize)]
struct BranchBatchInput {
    header: TraceHeader,
    prefix: Vec<Transition>,
    branches: Vec<Branch>,
}

fn read_stdin() -> Result<Vec<u8>, Box<dyn Error>> {
    let mut bytes = Vec::new();
    io::stdin().lock().read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn load_replay_input(bytes: &[u8]) -> Result<ReplayInput, Box<dyn Error>> {
    let input: ReplayInput = serde_json::from_slice(bytes)?;
    if input.header.transitions != input.transitions.len() {
        return Err(format!(
            "header declares {} transitions but payload has {}",
            input.header.transitions,
            input.transitions.len()
        )
        .into());
    }
    Ok(input)
}

fn load_branch_batch(bytes: &[u8]) -> Result<BranchBatchInput, Box<dyn Error>> {
    let input: BranchBatchInput = serde_json::from_slice(bytes)?;
    if input.branches.is_empty() {
        return Err("branch-batch requires at least one branch".into());
    }
    let mut ids = HashSet::new();
    for branch in &input.branches {
        if !ids.insert(&branch.id) {
            return Err(format!("duplicate branch id {:?}", branch.id).into());
        }
        if branch.transitions.is_empty() {
            return Err(format!("branch {:?} has no successor transition", branch.id).into());
        }
        let transitions = input.prefix.len() + branch.transitions.len();
        if input.header.transitions != transitions {
            return Err(format!(
                "header declares {} transitions but branch {:?} has {} prefix+branch transitions",
                input.header.transitions, branch.id, transitions
            )
            .into());
        }
    }
    Ok(input)
}

fn replay(input: &ReplayInput, native_reset: bool) -> Result<(), Box<dyn Error>> {
    let mut game = if native_reset {
        Game::from_seed_header(&input.header)?
    } else {
        Game::from_header(&input.header)?
    };
    let mut stdout = BufWriter::new(io::stdout().lock());
    serde_json::to_writer(&mut stdout, &game.snapshot())?;
    stdout.write_all(b"\n")?;
    for transition in &input.transitions {
        game.step(&transition.actions)?;
        serde_json::to_writer(&mut stdout, &game.snapshot())?;
        stdout.write_all(b"\n")?;
    }
    stdout.flush()?;
    Ok(())
}

fn benchmark(input: &ReplayInput, iterations: usize) -> Result<(), Box<dyn Error>> {
    if iterations == 0 {
        return Err("iterations must be positive".into());
    }
    // One full reset/replay warmup, matching the Python harness and excluded from timing.
    let mut warmup = Game::from_seed_header(&input.header)?;
    for transition in &input.transitions {
        warmup.step(&transition.actions)?;
    }
    black_box(warmup.terminal_banks());

    let started = Instant::now();
    let mut checksum = 0.0_f64;
    let mut last_banks = None;
    for _ in 0..iterations {
        let mut game = Game::from_seed_header(&input.header)?;
        for transition in &input.transitions {
            game.step(&transition.actions)?;
        }
        let banks = game
            .terminal_banks()
            .ok_or("replay workload did not reach a terminal state")?;
        checksum += banks.iter().sum::<f64>();
        last_banks = Some(banks.to_vec());
    }
    black_box(checksum);
    let elapsed = started.elapsed().as_secs_f64();
    let steps = input.transitions.len() * iterations;
    println!(
        "{}",
        json!({
            "engine": "rust",
            "iterations": iterations,
            "transitions_per_iteration": input.transitions.len(),
            "transitions": steps,
            "elapsed_seconds": elapsed,
            "steps_per_second": steps as f64 / elapsed,
            "terminal_banks": last_banks,
            "checksum": checksum,
        })
    );
    Ok(())
}

fn branch_batch(input: &BranchBatchInput) -> Result<(), Box<dyn Error>> {
    let mut common = Game::from_seed_header(&input.header)?;
    for transition in &input.prefix {
        common.step(&transition.actions)?;
    }
    let common_snapshot = common.snapshot();
    let mut branches = Vec::with_capacity(input.branches.len());
    for branch in &input.branches {
        let mut game = common.clone();
        game.step(&branch.transitions[0].actions)?;
        let successor = game.snapshot();
        for transition in &branch.transitions[1..] {
            game.step(&transition.actions)?;
        }
        branches.push(json!({
            "id": branch.id,
            "successor": successor,
            "final": game.snapshot(),
        }));
    }
    println!(
        "{}",
        json!({
            "schema": "re-branch-batch-v1",
            "prefix_transitions": input.prefix.len(),
            "common": common_snapshot,
            "branches": branches,
        })
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    let mode = args.first().map(String::as_str).unwrap_or("replay");
    let bytes = read_stdin()?;
    match mode {
        "replay" => replay(&load_replay_input(&bytes)?, true),
        "replay-state" => replay(&load_replay_input(&bytes)?, false),
        "bench" => {
            let iterations = args
                .windows(2)
                .find(|pair| pair[0] == "--iterations")
                .map(|pair| pair[1].parse())
                .transpose()?
                .unwrap_or(10);
            benchmark(&load_replay_input(&bytes)?, iterations)
        }
        "branch-batch" => branch_batch(&load_branch_batch(&bytes)?),
        _ => Err(format!(
            "unknown mode {mode:?}; use `replay`, `replay-state`, `branch-batch`, or `bench --iterations N`"
        )
        .into()),
    }
}
