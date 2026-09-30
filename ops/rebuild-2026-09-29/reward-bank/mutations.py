"""Apply each seam mutation, run its targeted check, restore it from git.

Run from the repository root on a clean tree:
  OMP_NUM_THREADS=2 CARGO_BUILD_JOBS=2 uv run python ops/rebuild-2026-09-29/reward-bank/mutations.py
A mutation is killed when its check exits nonzero. Rust mutations run the Rust
kaggriculture env tests only (the Python extension is not rebuilt per mutant).
"""

from __future__ import annotations

import subprocess
import sys

RUST = ["cargo", "test", "--lib", "kaggriculture::env_tests", "-q"]
PYTEST = ["uv", "run", "pytest", "-q", "-x", "-p", "no:cacheprovider"]
MUTATIONS = [
    ("R1 bank term reads the rival's bank", "src/kaggriculture/reward.rs",
     "self.bank_score(banks_after[s]) - self.bank_score(banks_before[s])",
     "self.bank_score(banks_after[1 - s]) - self.bank_score(banks_before[1 - s])", RUST),
    ("R2 terminal_scale ignores the bank cap", "src/kaggriculture/reward.rs",
     "} - if self.econ_bank_weight > 0. {", "} - if self.econ_bank_weight > 1e300 {", RUST),
    ("R3 cap budget ignores the bank cap", "src/kaggriculture/reward.rs",
     "+ if *wb > 0. { *bc } else { 0. };", "+ 0.;", RUST),
    ("R4 positive weight no longer needs a positive scale", "src/kaggriculture/reward.rs",
     "(*bs <= 0. || *bc <= 0.)", "(*bc <= 0.)", RUST),
    ("R5 env passes the after banks as before banks", "src/kaggriculture/env.rs",
     "before_banks,\n                                    after_banks,\n                                    done,",
     "after_banks,\n                                    after_banks,\n                                    done,", RUST),
    ("R6 negative banks are not clamped to zero", "src/kaggriculture/reward.rs",
     "bank.max(0.)", "bank", RUST),
    ("P1 Python terminal_scale ignores the bank cap", "python/owl/kaggriculture/rewards.py",
     "return 1.0 - death_cap - ineffective_cap - bank_cap",
     "return 1.0 - death_cap - ineffective_cap", [*PYTEST, "tests/kaggriculture/test_rewards.py"]),
    ("P2 Python admission drops the scale requirement", "python/owl/kaggriculture/rewards.py",
     "self.econ_bank_scale <= 0 or ", "", [*PYTEST, "tests/kaggriculture/test_rewards.py"]),
    ("P3 adapter metric averages seat 0 only", "python/owl/kaggriculture/env.py",
     "                ).mean()\n", "                )[..., 0].mean()\n",
     [*PYTEST, "tests/kaggriculture/test_env.py", "-k", "bank"]),
    ("P4 common-mode return reads seat 0 only", "python/owl/train/ppo.py",
     "player_returns.mean(dim=-1), both_seats", "player_returns[..., 0], both_seats",
     [*PYTEST, "tests/kaggriculture/test_training_smoke.py", "-k", "bank"]),
    ("C1 4-rank bank preset scale 50,000", "configs/kaggriculture_4rank_bc_finetune_bank.yaml",
     "    econ_bank_scale: 100000.0\n", "    econ_bank_scale: 50000.0\n",
     [*PYTEST, "tests/kaggriculture/test_configs.py"]),
    ("C2 8-rank bank preset keeps recipe J's muon_lr", "configs/kaggriculture_8rank_bc_finetune_bank.yaml",
     "  muon_lr: 0.0001\n", "  muon_lr: 0.0002\n", [*PYTEST, "tests/kaggriculture/test_configs.py"]),
    ("C3 2-rank bank preset drifts vf_coef", "configs/kaggriculture_2rank_bc_finetune_bank.yaml",
     "  vf_coef: 2.0\n", "  vf_coef: 1.0\n", [*PYTEST, "tests/kaggriculture/test_configs.py"]),
    ("C4 4-rank recipe-J base keeps the full adamw_lr", "configs/kaggriculture_4rank_bc_finetune.yaml",
     "  adamw_lr: 1.0e-05\n", "  adamw_lr: 0.0001\n", [*PYTEST, "tests/kaggriculture/test_configs.py"]),
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
