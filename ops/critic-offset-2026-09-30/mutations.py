"""Source mutations of the critic-offset change; each must fail a named test.

Usage (tree root): OMP_NUM_THREADS=2 python ops/critic-offset-2026-09-30/mutations.py
Each mutation replaces one exact source string, runs its tests, and restores
the file. A mutation is killed when pytest exits non-zero.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

MODEL = Path("python/owl/model/kaggriculture.py")
LORA = Path("python/owl/model/lora.py")
PPO = Path("python/owl/train/ppo.py")
T = "tests/kaggriculture/test_critic_offset.py"

MUTATIONS: list[tuple[str, Path, str, str, list[str]]] = [
    (
        "C1 head reads the trunk without detach",
        MODEL,
        "            own = own.detach()\n",
        "            own = own\n",
        [f"{T}::test_detach_trunk_blocks_the_heads_trunk_gradient"],
    ),
    (
        "C2 output layer not zero-initialised",
        MODEL,
        "        self.zero_critic_offset_output()\n\n    def zero_critic_offset_output",
        "\n    def zero_critic_offset_output",
        [f"{T}::test_fresh_head_output_layer_is_zero_and_its_hidden_layer_is_not"],
    ),
    (
        "C3 loader does not reset the missing head",
        MODEL,
        "    def reset_optional_state(self) -> None:\n        self.zero_critic_offset_output()\n",
        "    def reset_optional_state(self) -> None:\n        pass\n",
        [f"{T}::test_a_checkpoint_without_the_head_loads_and_zeroes_its_output"],
    ),
    (
        "C4 loader accepts a partial head",
        LORA,
        "    if missing_optional != optional:\n",
        "    if False:\n",
        [f"{T}::test_only_the_head_keys_may_be_missing"],
    ),
    (
        "C5 head reads the opponent critic token",
        MODEL,
        "        own = encoded.critic_value_hidden[:, 0]\n",
        "        own = encoded.critic_value_hidden[:, 1]\n",
        [f"{T}::test_the_head_reads_only_its_rows_own_critic_token"],
    ),
    (
        "C6 non-live rows keep the head's output",
        MODEL,
        "        return offsets.masked_fill(~encoded.critic_value_mask[:, 0], 0.0)\n",
        "        return offsets\n",
        [f"{T}::test_non_live_rows_keep_a_zero_offset"],
    ),
    (
        "C7 the offset is not added to the value",
        MODEL,
        "        return values + offsets, winner_log_probs, offsets\n",
        "        return values, winner_log_probs, offsets\n",
        [
            f"{T}::test_value_is_the_winner_value_plus_the_offset_and_actions_ignore_it",
            f"{T}::test_head_fits_a_common_mode_return_the_winner_critic_cannot",
        ],
    ),
    (
        "C8 default config dumps the new fields",
        MODEL,
        "        if not self.critic_offset:\n            del data[\"critic_offset\"]\n",
        "        if False:\n            del data[\"critic_offset\"]\n",
        [
            f"{T}::test_default_config_dumps_exactly_as_before_the_fields",
            f"{T}::test_preset_config_hash_is_unchanged_by_the_default_fields",
        ],
    ),
    (
        "C9 learner rows drop their offsets",
        PPO,
        "        value_offsets[rows] = output.value_offsets.reshape(-1)\n",
        "        pass\n",
        [f"{T}::test_learner_rows_carry_their_offsets_and_scripted_rows_zero"],
    ),
    (
        "C10 ev_common scored by the returns' own mean (no offset)",
        PPO,
        "                    value_offsets.mean(dim=-1),\n",
        "                    torch.zeros_like(value_offsets.mean(dim=-1)),\n",
        [f"{T}::test_offset_telemetry_scores_the_common_mode_by_the_offsets"],
    ),
]


def main() -> int:
    survivors = 0
    for name, path, old, new, tests in MUTATIONS:
        source = path.read_text()
        if source.count(old) != 1:
            print(f"{name}: SETUP ERROR (pattern count {source.count(old)})")
            return 2
        path.write_text(source.replace(old, new))
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-x", *tests],
                capture_output=True,
                text=True,
            )
        finally:
            path.write_text(source)
        killed = result.returncode != 0
        survivors += not killed
        lines = result.stdout.strip().splitlines()
        tail = lines[-1] if lines else ""
        first_error = next((line for line in lines if line.startswith("E ")), "")
        print(f"{name}: {'KILLED' if killed else 'SURVIVED'} ({tail}) {first_error}")
    return 1 if survivors else 0


if __name__ == "__main__":
    raise SystemExit(main())
