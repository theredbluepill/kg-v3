//! JSONL access to older controllers whose public interface accepts a `Game`.
//!
//! The reconstructed game contains only the supplied public observation and the
//! acting player's private state. Its dummy seed and empty rival inventory are
//! never consulted by these controllers; no engine transition is performed.

use super::{NativeAgent, NativeAgentKind};
use crate::{Config, Game, InitialState, PrivateState, PublicState, TRACE_FORMAT, TraceHeader};
use serde_json::{Value, json};
use std::io::{self, BufRead, Write};

fn game_from_observation(
    observation: &Value,
    configuration: &Value,
) -> Result<(Game, usize), String> {
    let seat = observation
        .get("player")
        .and_then(Value::as_u64)
        .filter(|seat| *seat < 2)
        .ok_or("observation.player must be 0 or 1")? as usize;
    let public: PublicState = serde_json::from_value(observation.clone())
        .map_err(|error| format!("invalid public observation: {error}"))?;
    if public.farms.len() != 2 {
        return Err("observation.farms must contain exactly two farms".to_string());
    }
    let private: PrivateState = serde_json::from_value(
        observation
            .get("private")
            .ok_or("observation.private is missing")?
            .clone(),
    )
    .map_err(|error| format!("invalid private observation: {error}"))?;
    let configuration: Config = serde_json::from_value(configuration.clone())
        .map_err(|error| format!("invalid configuration: {error}"))?;
    let empty = PrivateState {
        shed: Default::default(),
        seeds: Default::default(),
        inventories: Vec::new(),
    };
    let mut privates = vec![empty; 2];
    privates[seat] = private;
    let header = TraceHeader {
        format: TRACE_FORMAT.to_string(),
        seed: 0.into(),
        configuration,
        shop_schedule: Vec::new(),
        rng_schedule: Vec::new(),
        initial: InitialState { public, privates },
        terminal_banks: Vec::new(),
        transitions: 0,
    };
    Ok((Game::from_header(&header)?, seat))
}

/// Run one controller for the lifetime of a JSONL session, preserving its memory.
pub fn run(kind: NativeAgentKind) -> Result<(), Box<dyn std::error::Error>> {
    let mut agent = NativeAgent::new(kind);
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    for line in io::stdin().lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let input: Value = serde_json::from_str(&line)?;
        let observation = input.get("observation").unwrap_or(&input);
        let config = input.get("configuration").cloned().unwrap_or(json!({}));
        let (game, seat) = game_from_observation(observation, &config).map_err(io::Error::other)?;
        let action = agent.action(&game, seat).map_err(io::Error::other)?;
        serde_json::to_writer(&mut stdout, &action)?;
        writeln!(stdout)?;
        stdout.flush()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(game: &Game, seat: usize) -> Value {
        let mut observation = serde_json::to_value(game.public_state()).unwrap();
        observation["player"] = json!(seat);
        observation["private"] = serde_json::to_value(&game.privates[seat]).unwrap();
        observation
    }

    #[test]
    fn adapter_preserves_visible_state_and_configuration_without_rival_private_data() {
        let mut game = Game::new(Config::default(), 1337, 2).unwrap();
        game.step = 49;
        game.config.shed_capacity = 77_i64.into();
        game.config.farm_hand_cost_mult = 3_i64.into();
        game.config
            .extra
            .insert("custom".into(), json!({"keep": true}));
        for seat in 0..2 {
            game.privates[seat]
                .shed
                .insert("CARROT".into(), 7 + seat as i64);
        }
        let config = serde_json::to_value(&game.config).unwrap();
        for seat in 0..2 {
            let obs = observation(&game, seat);
            let (rebuilt, actual_seat) = game_from_observation(&obs, &config).unwrap();
            assert_eq!(actual_seat, seat);
            assert_eq!(
                serde_json::to_value(rebuilt.public_state()).unwrap(),
                serde_json::to_value(game.public_state()).unwrap()
            );
            assert_eq!(serde_json::to_value(&rebuilt.config).unwrap(), config);
            assert_eq!(
                serde_json::to_value(&rebuilt.privates[seat]).unwrap(),
                obs["private"]
            );
            assert!(rebuilt.privates[1 - seat].shed.is_empty());
            assert!(rebuilt.privates[1 - seat].seeds.is_empty());
            assert!(rebuilt.privates[1 - seat].inventories.is_empty());
        }
    }

    #[test]
    fn stateful_controllers_match_native_calls_on_both_seats() {
        let kinds = [
            NativeAgentKind::R04,
            NativeAgentKind::E776,
            NativeAgentKind::Flex,
            NativeAgentKind::Evgen,
            NativeAgentKind::Ecobot,
            NativeAgentKind::Starter,
            NativeAgentKind::Tetsutani,
            NativeAgentKind::ShopRouter,
        ];
        for kind in kinds {
            for seat in 0..2 {
                let mut direct = NativeAgent::new(kind);
                let mut adapted = NativeAgent::new(kind);
                let mut game = Game::new(Config::default(), 7331, 2).unwrap();
                let config = serde_json::to_value(&game.config).unwrap();
                // Consecutive calls, day changes, route-selection boundary, and
                // episode reset exercise controller memory without playing games.
                for step in [0, 1, 2, 23, 24, 25, 143, 144, 145, 0, 1] {
                    game.step = step;
                    let (rebuilt, player) =
                        game_from_observation(&observation(&game, seat), &config).unwrap();
                    assert_eq!(
                        adapted.action(&rebuilt, player).unwrap(),
                        direct.action(&game, seat).unwrap(),
                        "{kind:?}, seat {seat}, step {step}"
                    );
                    assert_eq!(
                        adapted.debug(),
                        direct.debug(),
                        "{kind:?}, seat {seat}, step {step}: memory differs"
                    );
                }
            }
        }
    }

    #[test]
    fn malformed_observations_and_configuration_fail_closed() {
        let game = Game::new(Config::default(), 0, 2).unwrap();
        let valid = observation(&game, 0);
        for seat in [json!(-1), json!(2), json!("0"), Value::Null] {
            let mut invalid = valid.clone();
            invalid["player"] = seat;
            assert!(game_from_observation(&invalid, &json!({})).is_err());
        }
        for field in ["private", "farms", "market", "step"] {
            let mut invalid = valid.clone();
            invalid.as_object_mut().unwrap().remove(field);
            assert!(
                game_from_observation(&invalid, &json!({})).is_err(),
                "missing {field}"
            );
        }
        assert!(game_from_observation(&valid, &json!({"turnsPerDay": 0})).is_err());
        assert!(game_from_observation(&valid, &json!({"shedCapacity": "bad"})).is_err());
    }
}
