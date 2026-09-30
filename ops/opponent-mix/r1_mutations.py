"""Re-run review r1's surviving mutations M4 and M5 against the new tests.

Each mutation is applied to the working tree, the named test runs, and the
file is restored with ``git checkout``. A mutation is CAUGHT when pytest fails.
Run from the repository root: ``OMP_NUM_THREADS=2 uv run python <this file>``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

MUTATIONS = [
    (
        "M4 eval own/opponent swapped in _evaluate_against_bot",
        Path("scripts/run_ppo.py"),
        "                    own.append(float(banks[seat]))\n"
        "                    opponent.append(float(banks[1 - seat]))\n",
        "                    own.append(float(banks[1 - seat]))\n"
        "                    opponent.append(float(banks[seat]))\n",
        "tests/scripts/test_run_ppo.py::"
        "test_fixed_bot_evaluation_attributes_banks_to_the_learned_seat",
    ),
    (
        "M5 forward_learner_rows values scattered to rows.flip(0)",
        Path("python/owl/train/ppo.py"),
        "    values[rows] = row_values\n",
        "    values[rows.flip(0)] = row_values\n",
        "tests/kaggriculture/test_opponent_mix.py::"
        "test_forward_learner_rows_matches_the_full_batch_rows",
    ),
]


def main() -> None:
    if subprocess.run(["git", "diff", "--quiet", "--", "scripts", "python"]).returncode:
        raise SystemExit("scripts/ or python/ has uncommitted changes; refusing")
    for name, path, old, new, test in MUTATIONS:
        source = path.read_text()
        if source.count(old) != 1:
            raise SystemExit(f"{name}: target text not found exactly once")
        path.write_text(source.replace(old, new))
        try:
            result = subprocess.run(
                ["pytest", "-q", "-p", "no:cacheprovider", test],
                capture_output=True,
                text=True,
            )
        finally:
            subprocess.run(["git", "checkout", "--", str(path)], check=True)
        verdict = "CAUGHT" if result.returncode != 0 else "SURVIVED"
        tail = result.stdout.strip().splitlines()[-1]
        print(f"{verdict}: {name} [{test}] -> {tail}")


if __name__ == "__main__":
    main()
