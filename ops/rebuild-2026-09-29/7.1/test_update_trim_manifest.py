"""Bounded bookkeeping checks; no engine or opponent execution."""

import json
import unittest

import pytest
import update_trim_manifest as updater


class UpdaterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = (
            json.dumps(
                {
                    "retained": [{"path": "engine_rs/src/lib.rs", "sha256": "kept"}],
                    "authored": [{"path": "engine_rs/tests/replay_parity.rs"}],
                    "excluded": [{"path": "engine_rs/src/native_agents/r04.rs"}],
                    "non_engine_changes": [
                        {"path": "docs/example.md", "reason": "old"}
                    ],
                },
                indent=2,
            ).encode()
            + b"\n"
        )
        self.updates = {"docs/example.md": "new", "ops/receipt.md": "receipt"}

    def test_preserves_engine_inventory_and_records_changes(self) -> None:
        result = json.loads(updater.render(self.base, self.base, self.updates))
        base = json.loads(self.base)
        for key in ("retained", "authored", "excluded"):
            assert result[key] == base[key]
        assert result["non_engine_changes"] == [
            {"path": "docs/example.md", "reason": "old Task 7.1: new"},
            {"path": "ops/receipt.md", "reason": "Task 7.1: receipt"},
        ]

    def test_second_application_is_byte_identical(self) -> None:
        first = updater.render(self.base, self.base, self.updates)
        assert updater.render(first, self.base, self.updates) == first

    def test_unexpected_current_input_is_refused(self) -> None:
        for key in ("retained", "authored", "excluded", "non_engine_changes"):
            changed = json.loads(self.base)
            changed[key].append({"path": "unexpected"})
            with (
                self.subTest(key=key),
                pytest.raises(ValueError, match="unexpected"),
            ):
                updater.render(json.dumps(changed).encode(), self.base, self.updates)


if __name__ == "__main__":
    unittest.main()
