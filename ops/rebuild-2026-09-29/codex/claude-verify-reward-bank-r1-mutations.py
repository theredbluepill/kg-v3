"""Independent verifier mutations for kg/rebuild-reward-bank (ab98e73).

Run from the scratch worktree root. Each mutation is applied, its check run,
and the file restored from git. Rust mutants marked NATIVE also rebuild the
extension (maturin develop) and run the Python native-parity tests.
"""

from __future__ import annotations

import subprocess
import sys

RUST = ["cargo", "test", "--lib", "kaggriculture", "-q"]
PYTEST = ["uv", "run", "pytest", "-q", "-x", "-p", "no:cacheprovider"]
PY_REWARDS = [*PYTEST, "tests/kaggriculture/test_rewards.py"]
PY_NATIVE = [*PYTEST, "tests/kaggriculture/test_rewards.py", "tests/kaggriculture/test_native_env.py"]
BUILD = ["uv", "run", "maturin", "develop", "-q"]

STALE_BEFORE = (
    "[self.transition.transition_banks_before[i * 2], "
    "self.transition.transition_banks_before[i * 2 + 1]],"
)
NO_RESET = (
    "{ let c = [self.transition.transition_banks_after[i * 2], "
    "self.transition.transition_banks_after[i * 2 + 1]]; "
    "if c == [0., 0.] { before_banks } else { c } },"
)
CALL = "before_banks,\n                                    after_banks,\n                                    done,"

MUTATIONS = [
    ("V1 drop the cap (Rust bank_score)", "src/kaggriculture/reward.rs",
     ".min(self.econ_bank_cap)", "", "rust"),
    ("V2 drop the cap (Python bank_score)", "python/owl/kaggriculture/rewards.py",
     "return raw.clamp(max=config.econ_bank_cap)", "return raw", "py"),
    ("V3 make it relative (Rust own minus rival increment)", "src/kaggriculture/reward.rs",
     "(relative + (self.bank_score(banks_after[s]) - self.bank_score(banks_before[s])))",
     "(relative + ((self.bank_score(banks_after[s]) - self.bank_score(banks_before[s]))"
     " - (self.bank_score(banks_after[1 - s]) - self.bank_score(banks_before[1 - s]))))",
     "rust"),
    ("V4 make it relative (Python oracle)", "python/owl/kaggriculture/rewards.py",
     "return bank_score(banks_after, config) - bank_score(banks_before, config)",
     "d = bank_score(banks_after, config) - bank_score(banks_before, config)\n"
     "    return d - d.flip(-1)", "py"),
    ("V5 off-by-one previous bank (env uses the previous transition's before bank)",
     "src/kaggriculture/env.rs", CALL,
     STALE_BEFORE + "\n                                    after_banks,\n                                    done,",
     "native"),
    ("V6 forget the autoreset reset (env carries the previous after bank)",
     "src/kaggriculture/env.rs", CALL,
     NO_RESET + "\n                                    after_banks,\n                                    done,",
     "native"),
    ("V7 terminal scale unchanged (Rust)", "src/kaggriculture/reward.rs",
     "} - if self.econ_bank_weight > 0. {", "} - if false {", "rust"),
    ("V8 terminal scale unchanged (Python)", "python/owl/kaggriculture/rewards.py",
     "return 1.0 - death_cap - ineffective_cap - bank_cap",
     "return 1.0 - death_cap - ineffective_cap", "py"),
    ("V9 Python budget uses > instead of >=", "python/owl/kaggriculture/rewards.py",
     "if death_cap + ineffective_cap + bank_cap >= 1:",
     "if death_cap + ineffective_cap + bank_cap > 1:", "py"),
    ("V10 Rust budget counts an inactive bank cap", "src/kaggriculture/reward.rs",
     "+ if *wb > 0. { *bc } else { 0. };", "+ *bc;", "rust"),
    ("V11 Rust rounds the bank term to f32 separately", "src/kaggriculture/reward.rs",
     "(relative + (self.bank_score(banks_after[s]) - self.bank_score(banks_before[s])))\n                    as f32",
     "(relative as f32) + ((self.bank_score(banks_after[s]) - self.bank_score(banks_before[s])) as f32)",
     "native"),
    ("V12 Rust skips the banks_before finiteness check", "src/kaggriculture/reward.rs",
     "if banks_before\n            .iter()\n            .chain(banks_after.iter())",
     "if banks_after\n            .iter()\n            .chain(banks_after.iter())", "rust"),
    ("V13 adapter telemetry swaps before/after (sign flip)", "python/owl/kaggriculture/env.py",
     "                    self._transition_banks_before,\n                    self._transition_banks_after,",
     "                    self._transition_banks_after,\n                    self._transition_banks_before,",
     "pyenv"),
    ("V14 Python oracle ignores w_b gate (adds bank term when off)", "python/owl/kaggriculture/rewards.py",
     "    if config.econ_bank_weight > 0:\n        economic = economic + bank_rewards(",
     "    if True:\n        economic = economic + bank_rewards(", "py"),
]

COMMANDS = {
    "rust": [RUST],
    "py": [PY_REWARDS],
    "pyenv": [[*PYTEST, "tests/kaggriculture/test_env.py", "tests/kaggriculture/test_training_smoke.py"]],
    "native": [RUST, BUILD, PY_NATIVE],
}

survivors = []
for name, path, old, new, kind in MUTATIONS:
    text = open(path, encoding="utf-8").read()
    if text.count(old) != 1:
        sys.exit(f"{name}: expected exactly one match in {path} (found {text.count(old)})")
    open(path, "w", encoding="utf-8").write(text.replace(old, new))
    killed_by = None
    tail = ""
    try:
        for command in COMMANDS[kind]:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            lines = (result.stdout + result.stderr).strip().splitlines()
            tail = " | ".join(l for l in lines[-3:])
            if result.returncode and command is not BUILD:
                killed_by = " ".join(command)
                break
            if result.returncode:
                killed_by = "BUILD FAILED (compile) " + tail
                break
    finally:
        subprocess.run(["git", "checkout", "--", path], check=True)
    if killed_by is None:
        survivors.append(name)
    print(f"{'KILLED' if killed_by else 'SURVIVED'}: {name} [{path}] by={killed_by} :: {tail[-300:]}")
    sys.stdout.flush()
# Rebuild the clean extension so the tree and venv match HEAD again.
subprocess.run(BUILD, check=True)
clean = subprocess.run(["git", "status", "--porcelain", "--", "src", "python", "configs"],
                       capture_output=True, text=True, check=True).stdout
print(f"survivors={len(survivors)} of {len(MUTATIONS)} {survivors}; tree restored: {not clean.strip()}")
