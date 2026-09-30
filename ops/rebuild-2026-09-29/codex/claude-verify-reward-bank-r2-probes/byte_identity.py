"""Replay the 16 fixture games under randomized relative-reward configs and dump rewards.

Usage: python byte_identity.py <repo_root> <out.npz> <with_bank_keys:0|1>
With keys=1 the dict also carries econ_bank_weight 0 and random inactive S/cap_b.
"""
import importlib.util, sys
from pathlib import Path
import numpy as np
from owl import rs

root = Path(sys.argv[1]); out_path = sys.argv[2]; with_bank = sys.argv[3] == "1"
sys.path.insert(0, str(root))
spec = importlib.util.spec_from_file_location("rec", root / "scripts/record_kaggriculture_env_reference.py")
rec = importlib.util.module_from_spec(spec); sys.modules["rec"] = rec; spec.loader.exec_module(rec)
_, fx = rec.load_fixture()
rng = np.random.default_rng(20260930)
rng2 = np.random.default_rng(7)
def bufs():
    sys.path.insert(0, str(root / "tests"))
    from kaggriculture.test_native_env import buffers
    return buffers(1)
all_rewards = []
configs = []
for trial in range(3):
    for game in range(16):
        w = float(rng.choice([0.0, 0.2, 0.05, 1e-3, 0.37]))
        cap = float(rng.uniform(0.05, 0.4))
        iw = float(rng.choice([0.0, 0.01, 0.3]))
        icap = float(rng.uniform(0.01, 0.2))
        cfg = {"reward_mode": "win_loss", "econ_shaping": w, "econ_starvation_weight": float(rng.uniform(0.1, 5)),
               "econ_drought_weight": float(rng.uniform(0.1, 5)), "econ_cap": cap,
               "econ_ineffective_weight": iw, "econ_ineffective_cap": icap}
        if with_bank:
            cfg |= {"econ_bank_weight": 0.0, "econ_bank_scale": float(rng2.choice([0.0, 1.0, 1e5, 7.3])),
                    "econ_bank_cap": float(rng2.choice([0.0, 0.25, 0.9]))}
        configs.append(repr(cfg))
        env = rs.KaggricultureEnv(1, 17000 + game, 1, "{}", cfg, 1, hire_limit=241)
        out = bufs(); env.observe(**out)
        rewards = np.zeros((719, 2), np.float32)
        for step in range(719):
            tokens = np.zeros((1, 2, 252, 12), dtype=np.int64)
            lengths = fx["lengths"][game, step].reshape(1, 2).copy()
            for seat in range(2):
                index = (game * 719 + step) * 2 + seat
                s, e = fx["program_offsets"][index: index + 2]
                tokens[0, seat, : lengths[0, seat]] = fx["tokens"][s:e]
            env.step(tokens, lengths, **out)
            rewards[step] = out["rewards"][0]
        all_rewards.append(rewards)
arr = np.stack(all_rewards)
np.savez(out_path, rewards=arr)
open(out_path + ".cfg.txt", "w").write("\n".join(configs))
print("nonzero rewards", int((arr != 0).sum()), "of", arr.size, "distinct", len(np.unique(arr)))
