//! Compiled ONLY as an example in the complete, byte-verified reference export.
//! Expected values come from the reference encoder, never the v3 reconstructor.
use kaggriculture_engine::{Game, TraceHeader, myolie_features};
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashSet;
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Source {
    Official {
        episode: u64,
        step: usize,
    },
    Seeded {
        seed: i64,
        profile: usize,
        step: usize,
        policy_sha256: String,
    },
    Dense {
        case: usize,
    },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    record_id: String,
    source: Source,
    #[serde(deserialize_with = "header")]
    header: TraceHeader,
}

fn header<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<TraceHeader, D::Error> {
    let value = Value::deserialize(deserializer)?;
    let object = value
        .as_object()
        .ok_or_else(|| serde::de::Error::custom("header must be object"))?;
    let keys = [
        "format",
        "seed",
        "configuration",
        "shop_schedule",
        "rng_schedule",
        "initial",
        "terminal_banks",
        "transitions",
    ];
    if object.len() != keys.len() || keys.iter().any(|key| !object.contains_key(*key)) {
        return Err(serde::de::Error::custom(
            "complete exact header keys required",
        ));
    }
    serde_json::from_value(value).map_err(serde::de::Error::custom)
}

fn ordered_equal(actual: &Value, expected: &Value, path: &str) -> Result<(), String> {
    if actual != expected {
        return Err(format!("value mismatch at {path}"));
    }
    match (actual, expected) {
        (Value::Object(a), Value::Object(b)) => {
            if a.keys().ne(b.keys()) {
                return Err(format!("key-order mismatch at {path}"));
            }
            for (key, value) in a {
                ordered_equal(value, &b[key], &format!("{path}.{key}"))?;
            }
        },
        (Value::Array(a), Value::Array(b)) => {
            for (i, (a, b)) in a.iter().zip(b).enumerate() {
                ordered_equal(a, b, &format!("{path}[{i}]"))?;
            }
        },
        _ => {},
    }
    Ok(())
}

fn record(input: &Path, output: File) -> Result<(), Box<dyn Error>> {
    let input = BufReader::new(File::open(input)?);
    let mut output = BufWriter::new(output);
    let mut ids = HashSet::new();
    for (index, line) in input.lines().enumerate() {
        let line = line?;
        let record: Record =
            serde_json::from_str(&line).map_err(|error| format!("record line {index}: {error}"))?;
        let id = &record.record_id;
        if !ids.insert(id.clone()) {
            return Err(format!("record {id}: duplicate id").into());
        }
        let h = &record.header;
        let step = h.initial.public.step;
        match &record.source {
            Source::Official {
                episode,
                step: source_step,
            } => {
                if ![95324500, 95901360, 95921764, 95990191].contains(episode)
                    || *source_step != step
                {
                    return Err(format!("record {id}: invalid official source").into());
                }
            },
            Source::Seeded {
                seed,
                profile,
                step: source_step,
                policy_sha256,
            } => {
                if *profile >= 6
                    || *seed != 11001 + *profile as i64
                    || *source_step != step
                    || policy_sha256.len() != 64
                    || !policy_sha256
                        .bytes()
                        .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
                {
                    return Err(format!("record {id}: invalid seeded source").into());
                }
            },
            Source::Dense { case } => {
                if *case >= 32 {
                    return Err(format!("record {id}: invalid dense source").into());
                }
            },
        }
        if !h.shop_schedule.is_empty()
            || !h.rng_schedule.is_empty()
            || !h.terminal_banks.is_empty()
            || h.transitions != 0
        {
            return Err(format!("record {id}: future schedules are forbidden").into());
        }
        let turns: usize = serde_json::to_string(&h.configuration.turns_per_day)?.parse()?;
        if turns == 0
            || (h.initial.public.day, h.initial.public.hour) != (step / turns, step % turns)
        {
            return Err(format!("record {id}: explicit header clock mismatch").into());
        }
        let game = Game::from_header(h).map_err(|error| format!("record {id}: {error}"))?;
        let snapshot = game.snapshot();
        ordered_equal(
            &serde_json::to_value(&snapshot.public)?,
            &serde_json::to_value(&h.initial.public)?,
            "public",
        )
        .map_err(|error| format!("record {id}: {error}"))?;
        ordered_equal(
            &serde_json::to_value(&snapshot.privates)?,
            &serde_json::to_value(&h.initial.privates)?,
            "privates",
        )
        .map_err(|error| format!("record {id}: {error}"))?;
        for (seat, (a, b)) in snapshot
            .public
            .farms
            .iter()
            .zip(&h.initial.public.farms)
            .enumerate()
        {
            if a.money.to_bits() != b.money.to_bits() {
                return Err(format!("record {id} seat {seat}: bank bit mismatch").into());
            }
        }
        for seat in 0..2 {
            let mut values = [0_f32; myolie_features::INVEST_FEATURE_COUNT];
            myolie_features::encode_invest(&game, seat, &mut values)
                .map_err(|error| format!("record {id} seat {seat}: {error}"))?;
            for value in values {
                output.write_all(&value.to_bits().to_le_bytes())?;
            }
        }
    }
    if ids.is_empty() {
        return Err("oracle input contains no records".into());
    }
    output.flush()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() != 3 {
        return Err("usage: observation_v3_oracle STATES_JSONL REFERENCE_F32LE".into());
    }
    let output = Path::new(&args[2]);
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    if let Err(error) = record(Path::new(&args[1]), file) {
        if output.exists() {
            fs::remove_file(output)?;
        }
        return Err(error);
    }
    Ok(())
}
