"""Differential fuzz: Python KaggricultureRewardConfig vs native constructor admission."""
import itertools, random
from pydantic import ValidationError
from owl import rs
from owl.kaggriculture.rewards import KaggricultureRewardConfig
KEYS = ["econ_shaping","econ_starvation_weight","econ_drought_weight","econ_cap",
        "econ_ineffective_weight","econ_ineffective_cap","econ_bank_weight","econ_bank_scale","econ_bank_cap"]
POOL = [0.0, -0.0, 5e-324, 1e-300, 1e-160, 0.001, 0.1, 0.2, 0.25, 0.3, 1/3, 0.35, 0.5, 0.6, 0.64, 0.65,
        0.75, 0.9, 0.9999999999999999, 1.0, 2.0, 1e5, 1e300, 1.7976931348623157e308]
rng = random.Random(3)
n = agree = acc = 0
mismatch = []
for i in range(60000):
    vals = [rng.choice(POOL) for _ in KEYS]
    # bias toward budget edges
    if i % 3 == 0:
        a, b = rng.choice([(0.25, 0.1), (0.1, 0.25), (0.3, 0.3), (1/3, 1/3), (0.2, 0.2)])
        vals[3], vals[5] = a, b
        vals[8] = rng.choice([1 - a - b, (1 - a) - b, 1 - (a + b), 0.65, 0.64, 0.35, 1/3])
        vals[0] = rng.choice([0.0, 0.2]); vals[4] = rng.choice([0.0, 0.001]); vals[6] = rng.choice([0.0, 0.25, 1e-300])
    cfg = dict(zip(KEYS, vals))
    try:
        KaggricultureRewardConfig.model_validate(cfg); py = True
    except ValidationError:
        py = False
    try:
        rs.KaggricultureEnv(1, 0, 1, "{}", {"reward_mode": "win_loss", **cfg}, 1, hire_limit=241); nat = True
    except ValueError:
        nat = False
    n += 1; acc += py
    if py == nat: agree += 1
    else: mismatch.append((cfg, py, nat))
print(f"cases={n} accepted_py={acc} agree={agree} mismatches={len(mismatch)}")
for m in mismatch[:5]: print(m)
