//! Transactional seed stream and native environment lifecycle.
use super::ObserveError;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvError {
    Value(String),
    Overflow(String),
    Panic(String),
}
impl fmt::Display for EnvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Value(s) | Self::Overflow(s) | Self::Panic(s) => f.write_str(s),
        }
    }
}
impl From<ObserveError> for EnvError {
    fn from(error: ObserveError) -> Self {
        Self::Value(error.to_string())
    }
}
impl From<String> for EnvError {
    fn from(error: String) -> Self {
        Self::Value(error)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeedStream {
    pub next: i64,
    pub stride: i64,
}
impl SeedStream {
    pub fn new(seed: i64, stride: i64) -> Result<Self, EnvError> {
        if seed < 0 || stride < 1 {
            return Err(EnvError::Value(
                "seed must be nonnegative and seed_stride positive".into(),
            ));
        }
        Ok(Self { next: seed, stride })
    }
    pub fn reserve(&self, selected: &[bool]) -> Result<(Vec<Option<i64>>, Self), EnvError> {
        let mut next = self.next;
        let mut seeds = Vec::with_capacity(selected.len());
        for (env, selected) in selected.iter().enumerate() {
            seeds.push(if *selected {
                let seed = next;
                next = next.checked_add(self.stride).ok_or_else(|| {
                    EnvError::Overflow(format!("env={env} seed counter overflow"))
                })?;
                Some(seed)
            } else {
                None
            });
        }
        Ok((seeds, Self { next, ..*self }))
    }
}

use super::buffers::checked_lengths;
use super::opponents::{HostedSeat, OpponentKind};
use super::reward::RewardConfig;
use super::{
    grammar, write_env, ObsRowMut, ObsStaging, ObservationConfig, ObservationGame,
    PreparedObservation, ValidatedObsBuffersMut,
};
use kaggriculture_engine::Config;
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::panic::{catch_unwind, AssertUnwindSafe};

pub(super) fn caught<T>(work: impl FnOnce() -> Result<T, EnvError>) -> Result<T, EnvError> {
    catch_unwind(AssertUnwindSafe(work)).map_err(|payload| {
        let detail = if let Some(s) = payload.downcast_ref::<String>() {
            s.as_str()
        } else if let Some(s) = payload.downcast_ref::<&str>() {
            s
        } else {
            "non-string worker panic"
        };
        EnvError::Panic(format!("native lifecycle panic: {detail}"))
    })?
}

#[derive(Clone, Debug, PartialEq)]
pub struct TerminalRecord {
    pub banks: [f64; 2],
    pub margin: f64,
    pub episode_steps: i64,
    pub winner: i64,
    pub econ: [[i64; 32]; 2],
    /// The learned seat of a fixed-opponent game; None in self-play.
    pub learner_seat: Option<usize>,
}

/// Fixed-opponent collection (`env.opponent_mix`): environments `0..envs` host
/// one native scripted seat. The opponent kind is collection bookkeeping only;
/// it never enters an observation, reward, mask or tensor the model reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpponentMix {
    pub kind: OpponentKind,
    pub envs: usize,
}

/// The learned seat of env `env` in its `episode`-th game (0 at construction,
/// +1 at every reset, truncation or auto-reset): seats alternate by env index
/// and by episode, so each env's learner plays both seats in turn.
pub fn learner_seat(env: usize, episode: u64) -> usize {
    ((env % 2) + (episode % 2) as usize) % 2
}

struct EnvSlot {
    game: ObservationGame,
    prepared: PreparedObservation,
    seed: i64,
    actors: [i64; 2],
    orders: i64,
    steps: i64,
    hires: [usize; 2],
    terminal: Option<TerminalRecord>,
    episode: u64,
    /// The hosted scripted controller of a fixed-opponent env; None in self-play.
    bot: Option<HostedSeat>,
}

/// A fresh controller for env `env`'s new game, or None for a self-play env.
fn new_bot(
    mix: Option<OpponentMix>,
    config: &Config,
    env: usize,
    episode: u64,
    prepared: &PreparedObservation,
) -> Result<Option<HostedSeat>, EnvError> {
    let Some(mix) = mix.filter(|mix| env < mix.envs) else {
        return Ok(None);
    };
    let seat = 1 - learner_seat(env, episode);
    HostedSeat::new(mix.kind, seat, config, prepared.snapshot().clone())
        .map(Some)
        .map_err(|e| EnvError::Value(format!("opponent seat={seat}: {e}")))
}

fn next_episode(episode: u64) -> Result<u64, EnvError> {
    episode
        .checked_add(1)
        .ok_or_else(|| EnvError::Overflow("episode counter overflow".into()))
}
#[derive(Clone, Debug, PartialEq)]
pub struct TransitionCache {
    pub rewards: Vec<f32>,
    pub dones: Vec<bool>,
    pub transition_banks_before: Vec<f64>,
    pub transition_banks_after: Vec<f64>,
    pub transition_econ_before: Vec<i64>,
    pub transition_econ_after: Vec<i64>,
}
pub struct TransitionBuffersMut<'a> {
    pub rewards: &'a mut [f32],
    pub dones: &'a mut [bool],
    pub transition_banks_before: &'a mut [f64],
    pub transition_banks_after: &'a mut [f64],
    pub transition_econ_before: &'a mut [i64],
    pub transition_econ_after: &'a mut [i64],
}
impl TransitionCache {
    fn zeros(n: usize) -> Self {
        Self {
            rewards: vec![0.; n * 2],
            dones: vec![false; n * 2],
            transition_banks_before: vec![0.; n * 2],
            transition_banks_after: vec![0.; n * 2],
            transition_econ_before: vec![0; n * 64],
            transition_econ_after: vec![0; n * 64],
        }
    }
    fn publish(&self, out: TransitionBuffersMut<'_>) {
        out.rewards.copy_from_slice(&self.rewards);
        out.dones.copy_from_slice(&self.dones);
        out.transition_banks_before
            .copy_from_slice(&self.transition_banks_before);
        out.transition_banks_after
            .copy_from_slice(&self.transition_banks_after);
        out.transition_econ_before
            .copy_from_slice(&self.transition_econ_before);
        out.transition_econ_after
            .copy_from_slice(&self.transition_econ_after);
    }
}
impl TransitionBuffersMut<'_> {
    fn validate(&self, n: usize) -> Result<(), EnvError> {
        for (name, actual, expected) in [
            ("rewards", self.rewards.len(), n * 2),
            ("dones", self.dones.len(), n * 2),
            (
                "transition_banks_before",
                self.transition_banks_before.len(),
                n * 2,
            ),
            (
                "transition_banks_after",
                self.transition_banks_after.len(),
                n * 2,
            ),
            (
                "transition_econ_before",
                self.transition_econ_before.len(),
                n * 64,
            ),
            (
                "transition_econ_after",
                self.transition_econ_after.len(),
                n * 64,
            ),
        ] {
            if actual != expected {
                return Err(EnvError::Value(format!(
                    "{name}: expected {expected} values, got {actual}"
                )));
            }
        }
        Ok(())
    }
}

pub struct PendingBatch {
    staging: ObsStaging,
    candidates: Vec<Option<EnvSlot>>,
    stream: SeedStream,
    selected: Option<Vec<bool>>,
    transition: Option<TransitionCache>,
    metrics: Vec<(f64, f64, f64, Option<usize>)>,
}
impl PendingBatch {
    /// Per completed game in env order: seat banks, seat-0 margin and, for a
    /// fixed-opponent game, the learned seat.
    pub fn metrics(&self) -> &[(f64, f64, f64, Option<usize>)] {
        &self.metrics
    }
}

pub struct NativeEnv {
    config: Config,
    episode_steps: i64,
    hire_multiplier: i64,
    reward: RewardConfig,
    hire_limit: i64,
    opponent: Option<OpponentMix>,
    pool: ThreadPool,
    slots: Vec<EnvSlot>,
    stream: SeedStream,
    transition: TransitionCache,
    // Taken by prepare_step and returned at commit (or worker failure). Each
    // writer clears its own rows, including padding, before encoding.
    step_staging: Option<ObsStaging>,
    #[cfg(test)]
    fault: Option<Fault>,
}
impl NativeEnv {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n: usize,
        seed: i64,
        stride: i64,
        config: Config,
        reward: RewardConfig,
        threads: usize,
        hire_limit: i64,
        opponent: Option<OpponentMix>,
    ) -> Result<Self, EnvError> {
        caught(|| {
            checked_lengths(n)?;
            if threads == 0 {
                return Err(EnvError::Value("native_threads must be positive".into()));
            }
            if let Some(mix) = opponent {
                if !(1..=n).contains(&mix.envs) {
                    return Err(EnvError::Value(format!(
                        "opponent envs must be in 1..={n}, got {}",
                        mix.envs
                    )));
                }
            }
            grammar::plan(1, 1, hire_limit)?;
            reward.validate()?;
            let admitted = ObservationConfig::new(&config)?;
            let episode_steps = admitted.episode_steps;
            let hire_multiplier = admitted.farm_hand_cost_mult;
            let (seeds, stream) = SeedStream::new(seed, stride)?.reserve(&vec![true; n])?;
            let pool = ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|e| EnvError::Value(format!("native_threads: {e}")))?;
            let mut staging = ObsStaging::new(n)?;
            let results: Vec<Result<EnvSlot, EnvError>> = pool.install(|| {
                staging
                    .buffers_mut()
                    .par_envs_mut()
                    .zip(seeds.par_iter())
                    .enumerate()
                    .map(|(i, (mut rows, seed))| {
                        caught(|| {
                            let seed = seed.expect("all construction rows selected");
                            let game =
                                ObservationGame::from_seed(config.clone(), &seed.to_string())?;
                            let prepared = game.prepare()?;
                            write_env(&prepared, &mut rows);
                            let bot = new_bot(opponent, &config, i, 0, &prepared)?;
                            Ok(EnvSlot {
                                game,
                                prepared,
                                seed,
                                actors: [
                                    rows.seats[0].globals_int[14],
                                    rows.seats[1].globals_int[14],
                                ],
                                orders: *rows.seats[0].order_limits,
                                steps: 0,
                                hires: [0, 0],
                                terminal: None,
                                episode: 0,
                                bot,
                            })
                        })
                        .map_err(|e| at_env(i, e))
                    })
                    .collect()
            });
            let slots = results.into_iter().collect::<Result<Vec<_>, _>>()?;
            let mut transition = TransitionCache::zeros(n);
            for (i, rows) in staging.buffers_mut().envs_mut().enumerate() {
                transition.transition_banks_before[i * 2..i * 2 + 2]
                    .copy_from_slice(rows.seats[0].banks);
                transition.transition_banks_after[i * 2..i * 2 + 2]
                    .copy_from_slice(rows.seats[0].banks);
            }
            Ok(Self {
                config,
                episode_steps,
                hire_multiplier,
                reward,
                hire_limit,
                opponent,
                pool,
                slots,
                stream,
                transition,
                step_staging: Some(staging),
                #[cfg(test)]
                fault: None,
            })
        })
    }
    pub fn n_envs(&self) -> usize {
        self.slots.len()
    }
    pub fn opponent(&self) -> Option<OpponentMix> {
        self.opponent
    }
    /// `[env][seat]` flattened: true where the learner acts on the current
    /// observation. Self-play envs mark both seats; a fixed-opponent env marks
    /// only its learned seat.
    pub fn learner_mask(&self) -> Vec<bool> {
        self.slots
            .iter()
            .flat_map(|slot| match &slot.bot {
                None => [true, true],
                Some(bot) => {
                    let mut mask = [true, true];
                    mask[bot.seat()] = false;
                    mask
                },
            })
            .collect()
    }
    pub fn seed_state(&self) -> (i64, Vec<i64>) {
        (
            self.stream.next,
            self.slots.iter().map(|s| s.seed).collect(),
        )
    }
    pub fn state_snapshot(&self, index: usize) -> Result<String, EnvError> {
        let slot = self
            .slots
            .get(index)
            .ok_or_else(|| EnvError::Value(format!("env_index={index} outside batch")))?;
        serde_json::to_string(&slot.game.game().snapshot())
            .map_err(|e| EnvError::Value(e.to_string()))
    }
    pub fn terminal_metrics(&self, index: usize) -> Result<Option<&TerminalRecord>, EnvError> {
        Ok(self
            .slots
            .get(index)
            .ok_or_else(|| EnvError::Value(format!("env_index={index} outside batch")))?
            .terminal
            .as_ref())
    }
    pub fn prepare_observe(&self) -> Result<PendingBatch, EnvError> {
        caught(|| {
            let mut staging = ObsStaging::new(self.n_envs())?;
            self.pool.install(|| {
                staging
                    .buffers_mut()
                    .par_envs_mut()
                    .zip(self.slots.par_iter())
                    .for_each(|(mut rows, slot)| write_env(&slot.prepared, &mut rows))
            });
            Ok(PendingBatch {
                staging,
                candidates: (0..self.n_envs()).map(|_| None).collect(),
                stream: self.stream,
                selected: None,
                transition: Some(self.transition.clone()),
                metrics: vec![],
            })
        })
    }
    pub fn publish_observe(
        &self,
        mut pending: PendingBatch,
        out: &mut ValidatedObsBuffersMut<'_>,
        transition: TransitionBuffersMut<'_>,
    ) -> Result<(), EnvError> {
        let n = self.n_envs();
        transition.validate(n)?;
        if out.n_envs() != n || pending.staging.buffers_mut().n_envs() != n {
            return Err(EnvError::Value(
                "observation publication n_envs mismatch".into(),
            ));
        }
        let cached = pending
            .transition
            .as_ref()
            .ok_or_else(|| EnvError::Value("missing observation transition cache".into()))?;
        pending.staging.publish(out)?;
        cached.publish(transition);
        Ok(())
    }
    pub fn prepare_reset(
        &mut self,
        mask: &[bool],
        truncate: bool,
    ) -> Result<PendingBatch, EnvError> {
        caught(|| {
            if mask.len() != self.n_envs() || (!truncate && mask.iter().any(|x| !x)) {
                return Err(EnvError::Value(
                    "reset mask: batch mismatch or partial full reset".into(),
                ));
            }
            let (seeds, stream) = self.stream.reserve(mask)?;
            let mut staging = ObsStaging::new(self.n_envs())?;
            let results: Vec<Result<Option<EnvSlot>, EnvError>> = self.pool.install(|| {
                staging
                    .buffers_mut()
                    .par_envs_mut()
                    .zip(seeds.par_iter())
                    .enumerate()
                    .map(|(i, (mut rows, seed))| {
                        caught(|| {
                            let Some(seed) = seed else { return Ok(None) };
                            #[cfg(test)]
                            self.inject(FaultPoint::ResetConstruct, i)?;
                            let game =
                                ObservationGame::from_seed(self.config.clone(), &seed.to_string())?;
                            #[cfg(test)]
                            {
                                self.inject(FaultPoint::ResetPrepare, i)?;
                                self.inject(FaultPoint::ResetPanic, i)?;
                            }
                            let prepared = game.prepare()?;
                            write_env(&prepared, &mut rows);
                            let episode = next_episode(self.slots[i].episode)?;
                            let bot = new_bot(self.opponent, &self.config, i, episode, &prepared)?;
                            Ok(Some(EnvSlot {
                                game,
                                prepared,
                                seed: *seed,
                                actors: [
                                    rows.seats[0].globals_int[14],
                                    rows.seats[1].globals_int[14],
                                ],
                                orders: *rows.seats[0].order_limits,
                                steps: 0,
                                hires: [0, 0],
                                terminal: None,
                                episode,
                                bot,
                            }))
                        })
                        .map_err(|e| at_env(i, e))
                    })
                    .collect()
            });
            let candidates = results.into_iter().collect::<Result<Vec<_>, _>>()?;
            let transition = if truncate {
                None
            } else {
                let mut t = TransitionCache::zeros(self.n_envs());
                for (i, rows) in staging.buffers_mut().envs_mut().enumerate() {
                    t.transition_banks_before[i * 2..i * 2 + 2]
                        .copy_from_slice(rows.seats[0].banks);
                    t.transition_banks_after[i * 2..i * 2 + 2].copy_from_slice(rows.seats[0].banks);
                }
                Some(t)
            };
            Ok(PendingBatch {
                staging,
                candidates,
                stream,
                selected: truncate.then(|| mask.to_vec()),
                transition,
                metrics: vec![],
            })
        })
    }
    pub fn prepare_step(
        &mut self,
        tokens: &[i64],
        lengths: &[i64],
    ) -> Result<PendingBatch, EnvError> {
        caught(|| {
            let n = self.n_envs();
            if tokens.len() != n * 2 * grammar::TOKENS_PER_SEAT || lengths.len() != n * 2 {
                return Err(EnvError::Value(
                    "tokens/lengths: batch shape mismatch".into(),
                ));
            }
            // Admission and decoding are independent per env. Keep their results
            // separate to retain the old error priority: all raw transport,
            // then seed overflow, then all grammar errors, then engine work.
            let admitted: Vec<_> = self.pool.install(|| {
                self.slots
                    .par_iter()
                    .enumerate()
                    .map(|(i, slot)| {
                        caught(|| {
                            let offset = i * 2 * grammar::TOKENS_PER_SEAT;
                            let tokens = &tokens[offset..offset + 2 * grammar::TOKENS_PER_SEAT];
                            let lengths = &lengths[i * 2..i * 2 + 2];
                            admit_transport(i, slot, tokens, lengths)?;
                            let terminal =
                                i128::from(slot.steps) >= i128::from(self.episode_steps) - 2;
                            let actions = caught(|| {
                                decode_actions(i, slot, tokens, lengths, self.hire_limit)
                            });
                            Ok((terminal, actions))
                        })
                    })
                    .collect()
            });
            let admitted = admitted.into_iter().collect::<Result<Vec<_>, _>>()?;
            let terminal: Vec<_> = admitted.iter().map(|(terminal, _)| *terminal).collect();
            let (seeds, stream) = self.stream.reserve(&terminal)?;
            let actions = admitted
                .into_iter()
                .map(|(_, actions)| actions)
                .collect::<Result<Vec<_>, _>>()?;
            // A discarded pending batch is allowed (e.g. Python return allocation
            // failed); only that cold path needs to allocate replacement staging.
            let mut staging = match self.step_staging.take() {
                Some(staging) => staging,
                None => ObsStaging::new(n)?,
            };
            let results: Vec<Result<(EnvSlot, TransitionRow), EnvError>> =
                self.pool.install(|| {
                    actions
                        .into_par_iter()
                        .zip(staging.buffers_mut().par_envs_mut())
                        .enumerate()
                        .map(|(i, (mut actions, mut rows))| {
                            caught(|| {
                                let slot = &self.slots[i];
                                let before_banks = game_banks(&slot.game)?;
                                let before_econ = game_econ(&slot.game)?;
                                let before_hires = executed_hires(&slot.game)?;
                                // The controller is cloned so a failed batch
                                // leaves the committed one untouched.
                                let mut bot = slot.bot.clone();
                                if let Some(bot) = bot.as_mut() {
                                    let seat = bot.seat();
                                    actions[seat] = bot
                                        .action(slot.prepared.snapshot().clone())
                                        .map_err(|e| {
                                            EnvError::Value(format!("opponent seat={seat}: {e}"))
                                        })?;
                                }
                                // The engine returns its single transactional
                                // clone. The committed slot remains untouched.
                                let (mut game, _) =
                                    slot.game.stepped_with_market_metrics(&actions)?;
                                #[cfg(test)]
                                {
                                    self.inject(FaultPoint::StepResult, i)?;
                                    self.inject(FaultPoint::StepPanic, i)?;
                                }
                                let after_banks = game_banks(&game)?;
                                let after_econ = game_econ(&game)?;
                                let after_hires = executed_hires(&game)?;
                                let mut hires = [0; 2];
                                for seat in 0..2 {
                                    hires[seat] = after_hires[seat]
                                        .checked_sub(before_hires[seat])
                                        .ok_or_else(|| {
                                            EnvError::Value(
                                                "HIRE attribution counter decreased".into(),
                                            )
                                        })?;
                                }
                                super::admission::validate_hire_cash(
                                    self.hire_multiplier,
                                    slot.hires,
                                    hires,
                                )?;
                                let done = game.game().terminal_banks().is_some();
                                if done != terminal[i] {
                                    return Err(EnvError::Value(
                                        "terminal seed prediction disagrees with kernel".into(),
                                    ));
                                }
                                let rewards = self.reward.transition(
                                    &before_econ,
                                    &after_econ,
                                    before_banks,
                                    after_banks,
                                    done,
                                )?;
                                let steps = slot.steps.checked_add(1).ok_or_else(|| {
                                    EnvError::Overflow("episode steps outside int64".into())
                                })?;
                                let record = if done {
                                    let margin = after_banks[0] - after_banks[1];
                                    if !margin.is_finite() {
                                        return Err(EnvError::Value(
                                            "terminal margin must be finite".into(),
                                        ));
                                    }
                                    Some(TerminalRecord {
                                        banks: after_banks,
                                        margin,
                                        episode_steps: steps,
                                        winner: if margin > 0. {
                                            0
                                        } else if margin < 0. {
                                            1
                                        } else {
                                            -1
                                        },
                                        econ: after_econ,
                                        learner_seat: bot.as_ref().map(|bot| 1 - bot.seat()),
                                    })
                                } else {
                                    None
                                };
                                let (seed, episode) = if let Some(seed) = seeds[i] {
                                    #[cfg(test)]
                                    self.inject(FaultPoint::AutoReset, i)?;
                                    game = ObservationGame::from_seed(
                                        self.config.clone(),
                                        &seed.to_string(),
                                    )?;
                                    (seed, next_episode(slot.episode)?)
                                } else {
                                    (slot.seed, slot.episode)
                                };
                                #[cfg(test)]
                                self.inject(FaultPoint::StepPrepare, i)?;
                                let prepared = game.prepare()?;
                                write_env(&prepared, &mut rows);
                                if seeds[i].is_some() {
                                    bot = new_bot(
                                        self.opponent,
                                        &self.config,
                                        i,
                                        episode,
                                        &prepared,
                                    )?;
                                }
                                let candidate = EnvSlot {
                                    game,
                                    prepared,
                                    seed,
                                    actors: [
                                        rows.seats[0].globals_int[14],
                                        rows.seats[1].globals_int[14],
                                    ],
                                    orders: *rows.seats[0].order_limits,
                                    steps: rows.seats[0].globals_int[0],
                                    hires: [
                                        rows.seats[0].globals_int[12] as usize,
                                        rows.seats[1].globals_int[12] as usize,
                                    ],
                                    terminal: record,
                                    episode,
                                    bot,
                                };
                                Ok((
                                    candidate,
                                    TransitionRow {
                                        before_banks,
                                        after_banks,
                                        before_econ,
                                        after_econ,
                                        rewards,
                                        done,
                                    },
                                ))
                            })
                            .map_err(|e| at_env(i, e))
                        })
                        .collect()
                });
            // Indexed collection joins every worker before choosing the first
            // error in env order. Return scratch on failure and destroy successful
            // peers on workers; no candidate has touched a committed game/output.
            if let Some(error) = results.iter().find_map(|result| result.as_ref().err()) {
                let error = error.clone();
                self.step_staging = Some(staging);
                self.pool.install(|| results.into_par_iter().for_each(drop));
                return Err(error);
            }
            let results = results.into_iter().map(Result::unwrap);
            let mut transition = TransitionCache::zeros(n);
            let mut candidates = Vec::with_capacity(n);
            let mut metrics = Vec::new();
            for (i, (slot, row)) in results.enumerate() {
                transition.rewards[i * 2..i * 2 + 2].copy_from_slice(&row.rewards);
                transition.dones[i * 2..i * 2 + 2].fill(row.done);
                transition.transition_banks_before[i * 2..i * 2 + 2]
                    .copy_from_slice(&row.before_banks);
                transition.transition_banks_after[i * 2..i * 2 + 2]
                    .copy_from_slice(&row.after_banks);
                for seat in 0..2 {
                    let range = (i * 2 + seat) * 32..(i * 2 + seat + 1) * 32;
                    transition.transition_econ_before[range.clone()]
                        .copy_from_slice(&row.before_econ[seat]);
                    transition.transition_econ_after[range].copy_from_slice(&row.after_econ[seat]);
                }
                if let Some(record) = &slot.terminal {
                    metrics.push((
                        record.banks[0],
                        record.banks[1],
                        record.margin,
                        record.learner_seat,
                    ));
                }
                candidates.push(Some(slot));
            }
            Ok(PendingBatch {
                staging,
                candidates,
                stream,
                selected: None,
                transition: Some(transition),
                metrics,
            })
        })
    }
    pub fn commit(
        &mut self,
        mut pending: PendingBatch,
        out: &mut ValidatedObsBuffersMut<'_>,
        transition: TransitionBuffersMut<'_>,
    ) -> Result<(), EnvError> {
        // Complete preflight before either full or selected publication. Observe
        // never installs replacement slots; truncate never publishes transitions.
        let n = self.n_envs();
        transition.validate(n)?;
        if out.n_envs() != n
            || pending.staging.buffers_mut().n_envs() != n
            || pending.candidates.len() != n
            || pending.selected.as_ref().is_some_and(|m| m.len() != n)
        {
            return Err(EnvError::Value("publication n_envs mismatch".into()));
        }
        // COMMIT: all remaining operations are admitted copies or swaps.
        // Every worker owns disjoint caller rows and drops its replaced game and
        // snapshot here, rather than returning those allocations to the caller.
        self.pool.install(|| {
            pending
                .staging
                .buffers_mut()
                .par_envs_mut()
                .zip(out.par_envs_mut())
                .zip(self.slots.par_iter_mut())
                .zip(pending.candidates.into_par_iter())
                .enumerate()
                .for_each(|(i, (((src, dst), slot), candidate))| {
                    if pending.selected.as_ref().is_none_or(|mask| mask[i]) {
                        for (src, dst) in src.seats.into_iter().zip(dst.seats) {
                            copy_row(src, dst);
                        }
                    }
                    if let Some(candidate) = candidate {
                        *slot = candidate;
                    }
                });
        });
        if let Some(mut next) = pending.transition {
            next.publish(transition);
            std::mem::swap(&mut self.transition, &mut next);
        }
        let old_staging = self.step_staging.replace(pending.staging);
        if let Some(old_staging) = old_staging {
            self.pool.install(|| drop(old_staging));
        }
        std::mem::swap(&mut self.stream, &mut pending.stream);
        Ok(())
    }
}
fn admit_transport(
    env: usize,
    slot: &EnvSlot,
    tokens: &[i64],
    lengths: &[i64],
) -> Result<(), EnvError> {
    let bot_seat = slot.bot.as_ref().map(HostedSeat::seat);
    for (row, (tokens, length)) in tokens
        .chunks_exact(grammar::TOKENS_PER_SEAT)
        .zip(lengths)
        .enumerate()
    {
        if bot_seat == Some(row) {
            // The scripted seat's transport must be the absent program:
            // nothing the learner submits for it is ever executed.
            if *length != 0 || tokens.iter().any(|token| *token != 0) {
                return Err(EnvError::Value(format!(
                            "env={} seat={} is played by the fixed opponent; submit length 0 and zero tokens",
                            env,
                            row
                        )));
            }
            continue;
        }
        if !(1..=grammar::MAX_FRAMES as i64).contains(length) {
            return Err(EnvError::Value(format!(
                "env={} seat={} length outside 1..=252",
                env, row
            )));
        }
        for (i, token) in tokens.iter().enumerate() {
            if *token < 0
                || *token >= grammar::SLOT_WIDTHS[i % grammar::SLOTS] as i64
                || (i >= *length as usize * grammar::SLOTS && *token != 0)
            {
                return Err(EnvError::Value(format!(
                    "env={} seat={} frame={} slot={} invalid token/padding",
                    env,
                    row,
                    i / grammar::SLOTS,
                    grammar::SLOT_NAMES[i % grammar::SLOTS]
                )));
            }
        }
    }
    Ok(())
}

fn decode_actions(
    env: usize,
    slot: &EnvSlot,
    tokens: &[i64],
    lengths: &[i64],
    hire_limit: i64,
) -> Result<Vec<serde_json::Value>, EnvError> {
    let bot_seat = slot.bot.as_ref().map(HostedSeat::seat);
    let mut pair = Vec::with_capacity(2);
    for (seat, &length) in lengths.iter().enumerate() {
        if bot_seat == Some(seat) {
            // Filled by the hosted controller inside the worker.
            pair.push(serde_json::Value::Null);
            continue;
        }
        let offset = seat * grammar::TOKENS_PER_SEAT;
        let plan = grammar::plan(slot.actors[seat], slot.orders, hire_limit)?;
        pair.push(
            grammar::decode(
                &plan,
                &tokens[offset..offset + grammar::TOKENS_PER_SEAT],
                length,
            )
            .map_err(|e| EnvError::Value(format!("env={env} seat={seat} {e}")))?,
        );
    }
    Ok(pair)
}

fn at_env(i: usize, error: EnvError) -> EnvError {
    match error {
        EnvError::Value(s) => EnvError::Value(format!("env={i} {s}")),
        EnvError::Overflow(s) => EnvError::Overflow(format!("env={i} {s}")),
        EnvError::Panic(s) => EnvError::Panic(format!("env={i} {s}")),
    }
}

fn copy_row(src: ObsRowMut<'_>, dst: ObsRowMut<'_>) {
    let ObsRowMut {
        tile_kind,
        tile_crop,
        tile_animal,
        tile_cell,
        tile_role,
        tiles_int,
        tiles_float,
        actor_slot,
        actor_cell,
        actor_role,
        actor_mask,
        actor_inventory,
        actor_inventory_rank,
        actors_float,
        player_features,
        storage_counts,
        storage_rank,
        banks,
        shop_type,
        shop_slot,
        shop_mask,
        market_product,
        market_float,
        market_int,
        global_features,
        globals_int,
        still_playing,
        order_limits,
        can_act,
    } = src;
    *dst.tile_kind = *tile_kind;
    *dst.tile_crop = *tile_crop;
    *dst.tile_animal = *tile_animal;
    *dst.tile_cell = *tile_cell;
    *dst.tile_role = *tile_role;
    *dst.tiles_int = *tiles_int;
    *dst.tiles_float = *tiles_float;
    *dst.actor_slot = *actor_slot;
    *dst.actor_cell = *actor_cell;
    *dst.actor_role = *actor_role;
    *dst.actor_mask = *actor_mask;
    *dst.actor_inventory = *actor_inventory;
    *dst.actor_inventory_rank = *actor_inventory_rank;
    *dst.actors_float = *actors_float;
    *dst.player_features = *player_features;
    *dst.storage_counts = *storage_counts;
    *dst.storage_rank = *storage_rank;
    *dst.banks = *banks;
    *dst.shop_type = *shop_type;
    *dst.shop_slot = *shop_slot;
    *dst.shop_mask = *shop_mask;
    *dst.market_product = *market_product;
    *dst.market_float = *market_float;
    *dst.market_int = *market_int;
    *dst.global_features = *global_features;
    *dst.globals_int = *globals_int;
    *dst.still_playing = *still_playing;
    *dst.order_limits = *order_limits;
    *dst.can_act = *can_act;
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FaultPoint {
    StepResult,
    StepPanic,
    AutoReset,
    StepPrepare,
    ResetConstruct,
    ResetPrepare,
    ResetPanic,
}
#[cfg(test)]
struct Fault {
    point: FaultPoint,
    env: usize,
    hits: std::sync::atomic::AtomicUsize,
}
#[cfg(test)]
impl NativeEnv {
    pub(super) fn set_fault(&mut self, point: FaultPoint, env: usize) {
        self.fault = Some(Fault {
            point,
            env,
            hits: std::sync::atomic::AtomicUsize::new(0),
        });
    }
    pub(super) fn fault_hits(&self) -> usize {
        self.fault
            .as_ref()
            .unwrap()
            .hits
            .load(std::sync::atomic::Ordering::SeqCst)
    }
    pub(super) fn clear_fault(&mut self) {
        self.fault = None;
    }
    fn inject(&self, point: FaultPoint, env: usize) -> Result<(), EnvError> {
        if let Some(f) = &self.fault {
            if f.point == point && f.env == env {
                f.hits.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if matches!(point, FaultPoint::StepPanic | FaultPoint::ResetPanic) {
                    panic!("injected {point:?} env={env}");
                }
                return Err(EnvError::Value(format!("injected {point:?} env={env}")));
            }
        }
        Ok(())
    }
    pub(super) fn replace_game(&mut self, i: usize, game: ObservationGame) {
        assert!(
            self.opponent.is_none(),
            "replace_game would desynchronize a hosted opponent"
        );
        let prepared = game.prepare().unwrap();
        let mut staging = ObsStaging::new(1).unwrap();
        let mut buffers = staging.buffers_mut();
        let mut rows = buffers.envs_mut().next().unwrap();
        write_env(&prepared, &mut rows);
        self.slots[i] = EnvSlot {
            game,
            prepared,
            seed: self.slots[i].seed,
            actors: [rows.seats[0].globals_int[14], rows.seats[1].globals_int[14]],
            orders: *rows.seats[0].order_limits,
            steps: rows.seats[0].globals_int[0],
            hires: [
                rows.seats[0].globals_int[12] as usize,
                rows.seats[1].globals_int[12] as usize,
            ],
            terminal: None,
            episode: self.slots[i].episode,
            bot: None,
        };
    }
}
#[cfg(test)]
impl TransitionCache {
    pub(super) fn buffers_mut(&mut self) -> TransitionBuffersMut<'_> {
        TransitionBuffersMut {
            rewards: &mut self.rewards,
            dones: &mut self.dones,
            transition_banks_before: &mut self.transition_banks_before,
            transition_banks_after: &mut self.transition_banks_after,
            transition_econ_before: &mut self.transition_econ_before,
            transition_econ_after: &mut self.transition_econ_after,
        }
    }
    pub(super) fn test_zeros(n: usize) -> Self {
        Self::zeros(n)
    }
}

struct TransitionRow {
    before_banks: [f64; 2],
    after_banks: [f64; 2],
    before_econ: [[i64; 32]; 2],
    after_econ: [[i64; 32]; 2],
    rewards: [f32; 2],
    done: bool,
}
fn game_banks(game: &ObservationGame) -> Result<[f64; 2], EnvError> {
    let values: [f64; 2] = game
        .game()
        .seat_money()
        .try_into()
        .map_err(|_| EnvError::Value("exactly two banks required".into()))?;
    if values.iter().any(|v| !v.is_finite()) {
        return Err(EnvError::Value("banks must be finite".into()));
    }
    Ok(values)
}
fn game_econ(game: &ObservationGame) -> Result<[[i64; 32]; 2], EnvError> {
    let source = game
        .game()
        .econ_counters()
        .ok_or_else(|| EnvError::Value("exactly two economic counter rows required".into()))?;
    let mut result = [[0; 32]; 2];
    for seat in 0..2 {
        for (index, value) in source[seat].iter().enumerate() {
            result[seat][index] = i64::try_from(*value).map_err(|_| {
                EnvError::Value(format!(
                    "seat={seat} economic counter {index} outside int64"
                ))
            })?;
        }
    }
    Ok(result)
}
fn executed_hires(game: &ObservationGame) -> Result<[u64; 2], EnvError> {
    // Pinned kernel's private-module A_HIRES=52, econ_attrib.rs:29. This
    // diagnostic admission counter never enters actor/critic observations.
    const A_HIRES: usize = 52;
    let source = game
        .game()
        .attrib_counters()
        .ok_or_else(|| EnvError::Value("two HIRE attribution rows required".into()))?;
    Ok([source[0][A_HIRES], source[1][A_HIRES]])
}
