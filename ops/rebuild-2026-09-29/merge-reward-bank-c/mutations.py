"""Merge-seam mutations for the reward-bank landing: apply, run the targeted check, restore.

Run from the staging repository root on the clean merge commit, after `just prepare`:
  OMP_NUM_THREADS=2 CARGO_BUILD_JOBS=2 uv run python ops/rebuild-2026-09-29/merge-reward-bank-c/mutations.py
A mutation is killed when its check exits nonzero. Rust mutations run the Rust
kaggriculture env tests only (the Python extension is not rebuilt per mutant).
"""

from __future__ import annotations

import subprocess
import sys

RUST = ["cargo", "test", "--lib", "kaggriculture::env_tests", "-q"]
PYTEST = ["uv", "run", "pytest", "-q", "-x", "-p", "no:cacheprovider"]
CONFIGS = [*PYTEST, "tests/kaggriculture/test_configs.py"]
MUTATIONS = [
    ("MM1 env.rs passes the after-step banks as banks_before", "src/kaggriculture/env.rs",
     "                                    before_banks,\n                                    after_banks,\n",
     "                                    after_banks,\n                                    after_banks,\n",
     RUST),
    ("MM2 4-rank bank preset keeps recipe J's full muon_lr",
     "configs/kaggriculture_4rank_bc_finetune_bank.yaml",
     "  muon_lr: 0.0001\n", "  muon_lr: 0.0002\n", CONFIGS),
    ("MM3 8-rank bank preset keeps recipe J's full adamw_lr",
     "configs/kaggriculture_8rank_bc_finetune_bank.yaml",
     "  adamw_lr: 5.0e-06\n", "  adamw_lr: 1.0e-05\n", CONFIGS),
    ("MM4 2-rank bank preset turns the bank term off",
     "configs/kaggriculture_2rank_bc_finetune_bank.yaml",
     "    econ_bank_weight: 0.25\n", "    econ_bank_weight: 0.0\n", CONFIGS),
    ("MM5 4-rank recipe-J base drifts from the 4-rank workload (n_envs)",
     "configs/kaggriculture_4rank_bc_finetune.yaml",
     "  n_envs: 64\n", "  n_envs: 32\n", CONFIGS),
    ("MM6 adapter metric sums seats instead of averaging",
     "python/owl/kaggriculture/env.py",
     "                ).mean()\n", "                ).sum()\n",
     [*PYTEST, "tests/kaggriculture/test_env.py"]),
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
