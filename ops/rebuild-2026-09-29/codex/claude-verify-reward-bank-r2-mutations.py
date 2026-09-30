"""Independent r2 verifier mutations for kg/rebuild-reward-bank (8aeba3d).

Run from the scratch worktree root. Each mutation is applied, its checks run in
order, and the file restored from git. A kill records whether it was a compile
error (not a behavioural kill). NATIVE mutants rebuild the extension and run the
Python native-parity/env tests after the Rust tests.
"""
from __future__ import annotations
import subprocess, sys

RUST = ["cargo", "test", "--lib", "kaggriculture", "-q"]
PYTEST = ["uv", "run", "--no-sync", "pytest", "-q", "-x", "-p", "no:cacheprovider", "-m", "not slow"]
BUILD = ["uv", "run", "--no-sync", "maturin", "develop", "-q"]
PY_REW = [*PYTEST, "tests/kaggriculture/test_rewards.py", "tests/kaggriculture/test_native_env.py"]
PY_ENV = [*PYTEST, "tests/kaggriculture/test_env.py", "tests/kaggriculture/test_training_smoke.py",
          "tests/kaggriculture/test_env_reference.py"]
PY_CFG = [*PYTEST, "tests/kaggriculture/test_configs.py"]
KINDS = {"rust": [RUST], "py": [PY_REW], "pyenv": [PY_ENV], "cfg": [PY_CFG],
         "native": [RUST, BUILD, PY_REW, PY_ENV]}

CALL = "before_banks,\n                                    after_banks,\n                                    done,"
R_BANK = "(relative + (self.bank_score(banks_after[s]) - self.bank_score(banks_before[s])))"
M = [
 ("M1 drop the cap (Rust bank_score)", "src/kaggriculture/reward.rs",
  ".min(self.econ_bank_cap)", "", "native"),
 ("M2 drop the cap (Python oracle)", "python/owl/kaggriculture/rewards.py",
  "return raw.clamp(max=config.econ_bank_cap)", "return raw", "py"),
 ("M3 make it relative (Rust: own minus rival score increment)", "src/kaggriculture/reward.rs", R_BANK,
  "(relative + ((self.bank_score(banks_after[s]) - self.bank_score(banks_before[s]))"
  " - (self.bank_score(banks_after[1 - s]) - self.bank_score(banks_before[1 - s]))))", "native"),
 ("M4 make it relative (Python oracle, rival bank difference)", "python/owl/kaggriculture/rewards.py",
  "return bank_score(banks_after, config) - bank_score(banks_before, config)",
  "return bank_score(banks_after - banks_after.flip(-1), config) - bank_score(banks_before - banks_before.flip(-1), config)",
  "py"),
 ("M5 off-by-one previous bank: env passes the post-step bank as 'before' (zero lag)",
  "src/kaggriculture/env.rs", CALL, "after_banks,\n                                    after_banks,\n                                    done,",
  "native"),
 ("M6 off-by-one previous bank: env uses the previous step's published before-bank (t-2)",
  "src/kaggriculture/env.rs", CALL,
  "[self.transition.transition_banks_before[i * 2], self.transition.transition_banks_before[i * 2 + 1]],"
  "\n                                    after_banks,\n                                    done,", "native"),
 ("M7 forget the autoreset reset: carry the previous after-bank (guarded so step 1 is right)",
  "src/kaggriculture/env.rs", CALL,
  "{ let c = [self.transition.transition_banks_after[i * 2], self.transition.transition_banks_after[i * 2 + 1]]; "
  "if c == [0., 0.] { before_banks } else { c } },\n                                    after_banks,\n                                    done,",
  "native"),
 ("M8 terminal scale unchanged (Rust)", "src/kaggriculture/reward.rs",
  "} - if self.econ_bank_weight > 0. {", "} - if false {", "native"),
 ("M9 terminal scale unchanged (Python)", "python/owl/kaggriculture/rewards.py",
  "return 1.0 - death_cap - ineffective_cap - bank_cap", "return 1.0 - death_cap - ineffective_cap", "py"),
 ("M10 terminal sign from banks_before (Rust)", "src/kaggriculture/reward.rs",
  "let sign = if banks_after[s] > banks_after[1 - s] {", "let sign = if banks_before[s] > banks_before[1 - s] {", "native"),
 ("M11 Rust score operation order w_b * (bank / S)", "src/kaggriculture/reward.rs",
  "(self.econ_bank_weight * bank.max(0.) / self.econ_bank_scale)",
  "(self.econ_bank_weight * (bank.max(0.) / self.econ_bank_scale))", "native"),
 ("M12 Python oracle drops the negative-bank clamp", "python/owl/kaggriculture/rewards.py",
  "banks.clamp(min=0.0)", "banks", "py"),
 ("M13 Rust separate f32 rounding (r1 V11 regression)", "src/kaggriculture/reward.rs",
  R_BANK + "\n                    as f32",
  "(relative as f32) + ((self.bank_score(banks_after[s]) - self.bank_score(banks_before[s])) as f32)", "native"),
 ("M14 Rust drops the banks_before finiteness check (r1 V12 regression)", "src/kaggriculture/reward.rs",
  "if banks_before\n            .iter()\n            .chain(banks_after.iter())",
  "if banks_after\n            .iter()\n            .chain(banks_after.iter())", "rust"),
 ("M15 Rust validate drops the cap_b > 0 requirement", "src/kaggriculture/reward.rs",
  "if *wb > 0. && (*bs <= 0. || *bc <= 0.) {", "if *wb > 0. && *bs <= 0. {", "native"),
 ("M16 Python budget counts an inactive bank cap", "python/owl/kaggriculture/rewards.py",
  "self.econ_bank_cap if self.econ_bank_weight > 0 else 0.0,", "self.econ_bank_cap,", "py"),
 ("M17 adapter metric sums instead of averaging seats", "python/owl/kaggriculture/env.py",
  "                ).mean()\n", "                ).sum()\n", "pyenv"),
 ("M18 zero-sum metric drops the /2", "python/owl/train/ppo.py",
  "zero_sum_abs = (player_returns[..., 0] - player_returns[..., 1]).abs() / 2",
  "zero_sum_abs = (player_returns[..., 0] - player_returns[..., 1]).abs()", "pyenv"),
 ("M19 common-mode metric reads seat 0 only", "python/owl/train/ppo.py",
  "self._masked_mean(player_returns.mean(dim=-1), both_seats)",
  "self._masked_mean(player_returns[..., 0], both_seats)", "pyenv"),
 ("M20 8-rank bank preset w_b .25 -> .2", "configs/kaggriculture_8rank_bc_finetune_bank.yaml",
  "econ_bank_weight: 0.25", "econ_bank_weight: 0.2", "cfg"),
 ("M21 2-rank bank preset S 100000 -> 10000", "configs/kaggriculture_2rank_bc_finetune_bank.yaml",
  "econ_bank_scale: 100000.0", "econ_bank_scale: 10000.0", "cfg"),
 ("M22 4-rank bank preset also changes vf_coef-free field gamma? (clip_coef drift)",
  "configs/kaggriculture_4rank_bc_finetune_bank.yaml", "  muon_lr: 0.0001", "  muon_lr: 0.00011", "cfg"),
 ("M23 Rust transition always adds the bank term (w_b gate removed)", "src/kaggriculture/reward.rs",
  "let economic = if self.econ_bank_weight > 0. {", "let economic = if true {", "native"),
]

surv = []
for name, path, old, new, kind in M:
    text = open(path, encoding="utf-8").read()
    if text.count(old) != 1:
        sys.exit(f"{name}: expected one match in {path}, found {text.count(old)}")
    open(path, "w", encoding="utf-8").write(text.replace(old, new))
    killed = None; tail = ""
    try:
        for cmd in KINDS[kind]:
            r = subprocess.run(cmd, capture_output=True, text=True, check=False)
            out = r.stdout + r.stderr
            tail = " | ".join(out.strip().splitlines()[-3:])
            if r.returncode:
                compile_err = "error[E" in out or "could not compile" in out
                killed = ("COMPILE ERROR " if compile_err else "") + " ".join(cmd[-3:] if cmd[0] == "uv" else cmd[:3])
                break
    finally:
        subprocess.run(["git", "checkout", "--", path], check=True)
    if not killed: surv.append(name)
    print(f"{'KILLED' if killed else 'SURVIVED'}: {name} by={killed} :: {tail[-260:]}", flush=True)
subprocess.run(BUILD, check=True)
dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout
print(f"survivors={len(surv)} of {len(M)} {surv}; tree clean: {not dirty.strip()}")
