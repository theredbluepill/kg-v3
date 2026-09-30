//! X1b: cha22 vs cha22 through opponents_rs::play_match (default Config), one line of JSON per seed.
//! Usage: cha22pair <kind0> <kind1> <seed>...
use kaggriculture_opponents::{Config, OpponentKind, play_match};
use std::str::FromStr;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let k0 = OpponentKind::from_str(&args[1]).unwrap();
    let k1 = OpponentKind::from_str(&args[2]).unwrap();
    for s in &args[3..] {
        let seed: i64 = s.parse().unwrap();
        let t = Instant::now();
        let r = play_match(Config::default(), seed, [k0, k1]).unwrap();
        let ctrl_err = r.steps.iter().flat_map(|x| x.seats.iter()).filter(|x| x.controller_error.is_some()).count();
        let eng_err = r.steps.iter().filter(|x| x.engine_error.is_some()).count();
        let (mut sub, mut zero, mut units, mut sells, mut buys, mut hires, mut lands) = (0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
        for st in &r.steps {
            if let Some(m) = &st.market_metrics {
                sub += m.submitted_orders; zero += m.zero_commit_orders; units += m.committed_units;
                sells += m.sell_orders; buys += m.buy_orders; hires += m.hire_orders; lands += m.land_orders;
            }
        }
        println!(
            "{{\"seed\":{},\"kinds\":[\"{}\",\"{}\"],\"completed\":{},\"steps\":{},\"bank0\":{},\"bank1\":{},\"joint\":{},\"winner\":{},\"controller_errors\":{},\"engine_errors\":{},\"joint_submitted_orders\":{},\"joint_zero_commit_orders\":{},\"joint_committed_units\":{},\"joint_sell_orders\":{},\"joint_buy_orders\":{},\"joint_hire_orders\":{},\"joint_land_orders\":{},\"wall_s\":{:.2}}}",
            seed, k0.key(), k1.key(), r.completed, r.steps.len(), r.final_banks[0], r.final_banks[1],
            r.final_banks[0] + r.final_banks[1],
            r.winner.map(|w| w.to_string()).unwrap_or("null".into()), ctrl_err, eng_err,
            sub, zero, units, sells, buys, hires, lands, t.elapsed().as_secs_f64()
        );
    }
}
