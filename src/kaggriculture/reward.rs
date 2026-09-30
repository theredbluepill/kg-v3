//! Native cumulative economic penalties, absolute own-bank shaping and the
//! reference's two f32 roundings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RewardMode {
    WinLoss,
}

#[derive(Clone, Debug)]
pub struct RewardConfig {
    pub reward_mode: RewardMode,
    pub econ_shaping: f64,
    pub econ_starvation_weight: f64,
    pub econ_drought_weight: f64,
    pub econ_cap: f64,
    pub econ_ineffective_weight: f64,
    pub econ_ineffective_cap: f64,
    /// Own-bank shaping weight `w_b` (owner decision 2026-09-30, term A).
    pub econ_bank_weight: f64,
    /// Bank scale `S`: `w_b * max(0, bank) / S` before the cap.
    pub econ_bank_scale: f64,
    /// Bank cap `cap_b`: the score saturates here and it joins the cap budget.
    pub econ_bank_cap: f64,
}
impl RewardConfig {
    pub fn validate(&self) -> Result<(), String> {
        let Self {
            reward_mode: RewardMode::WinLoss,
            econ_shaping: w,
            econ_starvation_weight: s,
            econ_drought_weight: d,
            econ_cap: cap,
            econ_ineffective_weight: i,
            econ_ineffective_cap: ic,
            econ_bank_weight: wb,
            econ_bank_scale: bs,
            econ_bank_cap: bc,
        } = self;
        if [w, s, d, cap, i, ic, wb, bs, bc]
            .iter()
            .any(|v| !v.is_finite() || **v < 0.)
        {
            return Err("reward coefficients must be finite and nonnegative".into());
        }
        if *w > 0. && (*cap <= 0. || !(*w * *s > 0. || *w * *d > 0.)) {
            return Err(
                "econ_shaping requires a positive cap and a nonzero binary64 event product".into(),
            );
        }
        if *i > 0. && *ic <= 0. {
            return Err("econ_ineffective_weight requires a positive cap".into());
        }
        if *wb > 0. && (*bs <= 0. || *bc <= 0.) {
            return Err(
                "econ_bank_weight requires a positive econ_bank_scale and econ_bank_cap".into(),
            );
        }
        let active = if *w > 0. { *cap } else { 0. }
            + if *i > 0. { *ic } else { 0. }
            + if *wb > 0. { *bc } else { 0. };
        if active >= 1. {
            return Err("active economic caps must sum below one".into());
        }
        Ok(())
    }
    pub fn terminal_scale(&self) -> f64 {
        1. - if self.econ_shaping > 0. {
            self.econ_cap
        } else {
            0.
        } - if self.econ_ineffective_weight > 0. {
            self.econ_ineffective_cap
        } else {
            0.
        } - if self.econ_bank_weight > 0. {
            self.econ_bank_cap
        } else {
            0.
        }
    }
    pub fn penalty(&self, c: &[i64; 32]) -> f64 {
        let death = if self.econ_shaping > 0. {
            // Positive finite W times +inf safely saturates at the cap. Disabled
            // components short-circuit, avoiding 0*inf while retaining R's order.
            (self.econ_shaping
                * (self.econ_starvation_weight * c[0] as f64
                    + self.econ_drought_weight * c[1] as f64))
                .min(self.econ_cap)
        } else {
            0.
        };
        let ineffective = if self.econ_ineffective_weight > 0. {
            (self.econ_ineffective_weight * c[2] as f64).min(self.econ_ineffective_cap)
        } else {
            0.
        };
        death + ineffective
    }
    /// Own-bank score `min(cap_b, (w_b * max(0, bank)) / S)` in binary64, in that
    /// operation order; zero when the term is disabled. An overflowing product
    /// or quotient is +inf and saturates at the cap (never NaN: `S > 0`).
    pub fn bank_score(&self, bank: f64) -> f64 {
        if self.econ_bank_weight > 0. {
            (self.econ_bank_weight * bank.max(0.) / self.econ_bank_scale).min(self.econ_bank_cap)
        } else {
            0.
        }
    }
    /// One transition's rewards. `banks_before` are the banks of the state the
    /// action was taken in (the reset bank on a game's first transition) and
    /// `banks_after` those of the completed transition, before any auto-reset,
    /// so the own-bank term never spans two games.
    pub fn transition(
        &self,
        before: &[[i64; 32]; 2],
        after: &[[i64; 32]; 2],
        banks_before: [f64; 2],
        banks_after: [f64; 2],
        done: bool,
    ) -> Result<[f32; 2], String> {
        if banks_before
            .iter()
            .chain(banks_after.iter())
            .any(|b| !b.is_finite())
        {
            return Err("transition banks must be finite".into());
        }
        if before.iter().flatten().any(|c| *c < 0)
            || before
                .iter()
                .flatten()
                .zip(after.iter().flatten())
                .any(|(b, a)| a < b)
        {
            return Err("economic counters must be nonnegative and monotonic".into());
        }
        let delta: [f64; 2] =
            std::array::from_fn(|s| self.penalty(&after[s]) - self.penalty(&before[s]));
        let r: [f32; 2] = std::array::from_fn(|s| {
            let relative = delta[1 - s] - delta[s];
            // Disabled bank shaping adds nothing, so rewards stay byte-identical
            // to the relative-only reward; enabled, it joins before f32 rounding.
            let economic = if self.econ_bank_weight > 0. {
                (relative + (self.bank_score(banks_after[s]) - self.bank_score(banks_before[s])))
                    as f32
            } else {
                relative as f32
            };
            if done {
                let sign = if banks_after[s] > banks_after[1 - s] {
                    1.
                } else if banks_after[s] < banks_after[1 - s] {
                    -1.
                } else {
                    0.
                };
                (f64::from(economic) + self.terminal_scale() * sign) as f32
            } else {
                economic
            }
        });
        if r.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite reward".into());
        }
        Ok(r)
    }
}
