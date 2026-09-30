"""Verify r2 edits: apply each seam mutation, run its targeted check, restore it from git.

Run from the repository root on a clean, committed tree with the extension built:
  OMP_NUM_THREADS=2 CARGO_BUILD_JOBS=2 uv run python ops/rebuild-2026-09-29/reward-bank/mutations-r2.py
A mutation is killed when its check exits nonzero. Rust mutations run the Rust
kaggriculture env tests only (the Python extension is not rebuilt per mutant).
"""

from __future__ import annotations

import subprocess
import sys

RUST = ["cargo", "test", "--lib", "kaggriculture::env_tests", "-q"]
PYTEST = ["uv", "run", "pytest", "-q", "-x", "-p", "no:cacheprovider"]
REWARDS = "tests/kaggriculture/test_rewards.py"
MUTATIONS = [
    ("M11 Rust bank score takes the quotient before the product", "src/kaggriculture/reward.rs",
     "(self.econ_bank_weight * bank.max(0.) / self.econ_bank_scale)",
     "(self.econ_bank_weight * (bank.max(0.) / self.econ_bank_scale))",
     RUST),
    ("O1 Python oracle bank score takes the quotient before the product",
     "python/owl/kaggriculture/rewards.py",
     "raw = config.econ_bank_weight * banks.clamp(min=0.0) / config.econ_bank_scale",
     "raw = config.econ_bank_weight * (banks.clamp(min=0.0) / config.econ_bank_scale)",
     [*PYTEST, REWARDS, "-k", "product_then_the_quotient"]),
    ("Z1 Python oracle adds the terminal term on every step (the pre-r2 oracle)",
     "python/owl/kaggriculture/rewards.py",
     "    return torch.where(dones, terminal_f32, economic_f32)\n",
     "    return (economic_f32.double() + terminal * dones).float()\n",
     [*PYTEST, REWARDS, "-k", "sign_of_zero"]),
    ("Z2 the pre-r2 oracle against the live tiny-W native game",
     "python/owl/kaggriculture/rewards.py",
     "    return torch.where(dones, terminal_f32, economic_f32)\n",
     "    return (economic_f32.double() + terminal * dones).float()\n",
     [*PYTEST, REWARDS, "-k", "extreme_value_native and tiny"]),
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
