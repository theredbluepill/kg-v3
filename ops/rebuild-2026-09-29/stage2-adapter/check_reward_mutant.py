"""Compile the exact reward module/table in a dependency-free scratch harness."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import tempfile
from run_check import ARTIFACTS, ROOT, run

source = ROOT / 'src/kaggriculture/reward.rs'
tests = ROOT / 'src/kaggriculture/env_tests.rs'
original = source.read_bytes()
table_source = tests.read_text()
start = table_source.index('#[test]\nfn reward_admission_predicate_cases()')
end = table_source.index('\n#[test]', start + 1)
table = table_source[start:end]
needle = '*cap <= 0. || !(*w * *s > 0. || *w * *d > 0.)'
replacement = '*cap <= 0. || !(*w * *s > 0. || *w * *d > 0. || *i > 0.)'
assert original.decode().count(needle) == 1
with tempfile.TemporaryDirectory(prefix='stage2-reward-') as tmp:
    scratch = Path(tmp)
    module = scratch / 'reward.rs'
    harness = scratch / 'harness.rs'
    harness.write_text('mod reward;\nuse reward::{RewardConfig, RewardMode};\n' + table + '\n')
    module.write_bytes(original)
    binary = scratch / 'reward-tests'
    def check(label: str, expected: int) -> None:
        assert run(label + '-compile', ['rustc', '--edition=2021', '--test', str(harness), '-o', str(binary)], cwd=scratch) == 0
        assert run(label, [str(binary), 'reward_admission_predicate_cases', '--exact', '--nocapture'], cwd=scratch) == expected
    try:
        module.write_text(original.decode().replace(needle, replacement))
        # The original ten cases cannot discriminate the combined rule.
        strengthening = '        (0.2, 0., 0., 0.25, 0.001, 0.1, false),\n'
        assert strengthening in table
        harness.write_text('mod reward;\nuse reward::{RewardConfig, RewardMode};\n' + table.replace(strengthening, '') + '\n')
        check('rust-mutant-old-table', 0)
        harness.write_text('mod reward;\nuse reward::{RewardConfig, RewardMode};\n' + table + '\n')
        check('rust-mutant-strengthening-red', 101)
    finally:
        module.write_bytes(original)
    assert module.read_bytes() == original
    assert source.read_bytes() == original
    check('rust-restored-strengthening-green', 0)
    receipt = {'production_sha256': hashlib.sha256(original).hexdigest(),
               'scratch_restored_byte_exact': module.read_bytes() == original,
               'production_unchanged': source.read_bytes() == original,
               'table_sha256': hashlib.sha256(table.encode()).hexdigest(),
               'mutation': {'from': needle, 'to': replacement}}
    (ARTIFACTS / 'reward-mutation.json').write_text(json.dumps(receipt, indent=2) + '\n')
