"""Verify r1: apply each seam mutation, run its targeted check, restore it from git.

Run from the repository root on a clean tree:
  OMP_NUM_THREADS=2 CARGO_BUILD_JOBS=2 uv run python ops/rebuild-2026-09-29/reward-bank/mutations-r1.py
A mutation is killed when its check exits nonzero. Rust mutations run the Rust
kaggriculture env tests only (the Python extension is not rebuilt per mutant).
"""

from __future__ import annotations

import subprocess
import sys

RUST = ["cargo", "test", "--lib", "kaggriculture::env_tests", "-q"]
PYTEST = ["uv", "run", "pytest", "-q", "-x", "-p", "no:cacheprovider"]
MUTATIONS = [
    ("V11 relative and bank parts rounded to f32 separately", "src/kaggriculture/reward.rs",
     "(relative + (self.bank_score(banks_after[s]) - self.bank_score(banks_before[s])))\n                    as f32",
     "relative as f32\n                    + (self.bank_score(banks_after[s]) - self.bank_score(banks_before[s])) as f32",
     RUST),
    ("V12 finiteness check reads banks_after twice", "src/kaggriculture/reward.rs",
     "        if banks_before\n            .iter()", "        if banks_after\n            .iter()", RUST),
    ("C5 4-rank bank preset back to w_b 1.0", "configs/kaggriculture_4rank_bc_finetune_bank.yaml",
     "    econ_bank_weight: 0.25\n", "    econ_bank_weight: 1.0\n",
     [*PYTEST, "tests/kaggriculture/test_configs.py"]),
    ("C6 2-rank bank preset back to w_b 1.0", "configs/kaggriculture_2rank_bc_finetune_bank.yaml",
     "    econ_bank_weight: 0.25\n", "    econ_bank_weight: 1.0\n",
     [*PYTEST, "tests/kaggriculture/test_configs.py"]),
    ("P5 zero-sum return drops the absolute value", "python/owl/train/ppo.py",
     "(player_returns[..., 0] - player_returns[..., 1]).abs() / 2",
     "(player_returns[..., 0] - player_returns[..., 1]) / 2",
     [*PYTEST, "tests/kaggriculture/test_training_smoke.py", "-k", "bank"]),
    ("P6 Python oracle rounds the parts to f32 separately", "python/owl/kaggriculture/rewards.py",
     "        economic = economic + bank_rewards(banks_before, banks_after, config)\n",
     "        economic = (\n            economic.float() + bank_rewards(banks_before, banks_after, config).float()\n        ).double()\n",
     [*PYTEST, "tests/kaggriculture/test_rewards.py", "-k", "round_to_float32_once"]),
]

survivors = 0
for name, path, old, new, command in MUTATIONS:
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    if text.count(old) != 1:
        sys.exit(f"{name}: expected exactly one match in {path}")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text.replace(old, new))
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    finally:
        subprocess.run(["git", "checkout", "--", path], check=True)
    tail = (result.stdout + result.stderr).strip().splitlines()[-1:]
    verdict = "KILLED" if result.returncode else "SURVIVED"
    survivors += result.returncode == 0
    print(f"{verdict}: {name} ({path}) exit={result.returncode} :: {' '.join(command)} :: {tail}")
    sys.stdout.flush()
clean = subprocess.run(["git", "status", "--porcelain", "--", "src", "python", "configs"],
                       capture_output=True, text=True, check=True).stdout
print(f"survivors={survivors} of {len(MUTATIONS)}; tree restored: {not clean.strip()}")
