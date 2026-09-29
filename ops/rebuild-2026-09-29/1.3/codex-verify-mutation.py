from pathlib import Path
import hashlib
import json
import subprocess

source = Path("src/kaggriculture/observe.rs")
original = source.read_bytes()
before = hashlib.sha256(original).hexdigest()
marker = b"fn write_shops_and_market("
start = original.index(marker)
old = b"[&public.market.inventory, &public.market.prices]"
new = b"[&public.market.prices, &public.market.inventory]"
assert original[start:].count(old) == 1
mutated = original[:start] + original[start:].replace(old, new, 1)
try:
    source.write_bytes(mutated)
    status = subprocess.run([
        "python3", "ops/rebuild-2026-09-29/1.3/bounded.py",
        "--name", "codex-verify-mutant", "--", "cargo", "test", "--locked", "--lib",
        "kaggriculture::oracle_corpus::compare_observation_oracle", "--", "--exact", "--nocapture"
    ], check=False).returncode
finally:
    source.write_bytes(original)
after = hashlib.sha256(source.read_bytes()).hexdigest()
assert before == after and source.read_bytes() == original
result = {"mutation": "swap inventory and price only in production write_shops_and_market",
          "before_sha256": before, "restored_sha256": after, "byte_exact_restored": True,
          "mutant_exit_status": status}
Path("ops/rebuild-2026-09-29/1.3/codex-verify-mutation.json").write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result))
assert status == 101, "must fail an oracle assertion, not a resource or build guard"
