//! Native cumulative economic penalties and the reference's two f32 roundings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RewardMode { WinLoss }

#[derive(Clone, Debug)]
pub struct RewardConfig {
    pub reward_mode: RewardMode,
    pub econ_shaping: f64,
    pub econ_starvation_weight: f64,
    pub econ_drought_weight: f64,
    pub econ_cap: f64,
    pub econ_ineffective_weight: f64,
    pub econ_ineffective_cap: f64,
}
impl RewardConfig {
    pub fn validate(&self) -> Result<(), String> {
        let Self { reward_mode: RewardMode::WinLoss, econ_shaping:w,
            econ_starvation_weight:s, econ_drought_weight:d, econ_cap:cap,
            econ_ineffective_weight:i, econ_ineffective_cap:ic } = self;
        if [w,s,d,cap,i,ic].iter().any(|v| !v.is_finite() || **v < 0.) {
            return Err("reward coefficients must be finite and nonnegative".into());
        }
        if *w > 0. && (*cap <= 0. || !(*w * *s > 0. || *w * *d > 0.)) {
            return Err("econ_shaping requires a positive cap and a nonzero binary64 event product".into());
        }
        if *i > 0. && *ic <= 0. {
            return Err("econ_ineffective_weight requires a positive cap".into());
        }
        let active = if *w > 0. {*cap} else {0.} + if *i > 0. {*ic} else {0.};
        if active >= 1. { return Err("active economic caps must sum below one".into()); }
        Ok(())
    }
    pub fn terminal_scale(&self) -> f64 {
        1. - if self.econ_shaping > 0. {self.econ_cap} else {0.}
           - if self.econ_ineffective_weight > 0. {self.econ_ineffective_cap} else {0.}
    }
    pub fn penalty(&self, c: &[i64;32]) -> f64 {
        let death = if self.econ_shaping > 0. {
            // Positive finite W times +inf safely saturates at the cap. Disabled
            // components short-circuit, avoiding 0*inf while retaining R's order.
            (self.econ_shaping * (self.econ_starvation_weight * c[0] as f64
                + self.econ_drought_weight * c[1] as f64)).min(self.econ_cap)
        } else {0.};
        let ineffective = if self.econ_ineffective_weight > 0. {
            (self.econ_ineffective_weight * c[2] as f64).min(self.econ_ineffective_cap)
        } else {0.};
        death + ineffective
    }
    pub fn transition(&self, before:&[[i64;32];2], after:&[[i64;32];2], banks:[f64;2], done:bool) -> Result<[f32;2],String> {
        if banks.iter().any(|b| !b.is_finite()) { return Err("transition banks must be finite".into()); }
        if before.iter().flatten().any(|c| *c < 0) || before.iter().flatten().zip(after.iter().flatten()).any(|(b,a)| a < b) {
            return Err("economic counters must be nonnegative and monotonic".into());
        }
        let delta:[f64;2] = std::array::from_fn(|s| self.penalty(&after[s])-self.penalty(&before[s]));
        let r: [f32;2] = std::array::from_fn(|s| {
            let economic = (delta[1-s]-delta[s]) as f32;
            if done {
                let sign = if banks[s] > banks[1-s] {1.} else if banks[s] < banks[1-s] {-1.} else {0.};
                (f64::from(economic) + self.terminal_scale()*sign) as f32
            } else {economic}
        });
        if r.iter().any(|v| !v.is_finite()) { return Err("nonfinite reward".into()); }
        Ok([0.;2])
    }
}
