"""Bounded custody-bookkeeping checks; no engine or opponent execution."""

import json
import subprocess
import unittest

import pytest
import update_trim_manifest as updater


def encode(value: object) -> bytes:
    return (json.dumps(value, indent=2) + "\n").encode()


class UpdaterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = encode(
            {
                "retained": [{"path": "engine_rs/src/lib.rs", "sha256": "kept"}],
                "authored": [{"path": "engine_rs/tests/replay_parity.rs"}],
                "excluded": [
                    {"path": path, "reference_sha256": "kept", "reason": "old"}
                    for path in (
                        "engine_rs/src/native_agents/starter.rs",
                        "engine_rs/src/native_agents/r04.rs",
                        "engine_rs/src/native_agents/ecobot.rs",
                        "engine_rs/src/native_agents/e776.rs",
                        "engine_rs/fixtures/e776-kenjo-trace.json",
                        "engine_rs/src/policy_rows.rs",
                    )
                ],
                "non_engine_changes": [{"path": "docs/example.md", "reason": "old"}],
            }
        )
        stopped = json.loads(self.base)
        stopped["non_engine_changes"].append(
            {"path": "ops/run-1.log", "reason": "preserve stopped probe"}
        )
        self.stopped = encode(stopped)
        self.updates = {"docs/example.md": "new", "ops/receipt.md": "receipt"}

    def render(self, current: bytes) -> bytes:
        return updater.render(current, self.base, self.stopped, self.updates)

    def test_accepts_both_committed_inputs_and_preserves_engine_bytes(self) -> None:
        result = self.render(self.base)
        assert self.render(self.stopped) == result
        parsed = json.loads(result)
        for key in ("retained", "authored"):
            assert parsed[key] == json.loads(self.base)[key]
        assert parsed["non_engine_changes"] == [
            {"path": "docs/example.md", "reason": "old Task 7.1: new"},
            {"path": "ops/run-1.log", "reason": "preserve stopped probe"},
            {"path": "ops/receipt.md", "reason": "Task 7.1: receipt"},
        ]

    def test_exactly_five_excluded_reasons_register_the_byte_exact_copies(self) -> None:
        result = json.loads(self.render(self.stopped))
        before = json.loads(self.base)
        for original, actual in zip(
            before["excluded"], result["excluded"], strict=True
        ):
            if original["path"] == "engine_rs/src/policy_rows.rs":
                assert actual == original
                continue
            assert actual["path"] == original["path"]
            assert actual["reference_sha256"] == original["reference_sha256"]
            destination = original["path"].replace("engine_rs/", "opponents_rs/")
            assert actual["reason"] == (
                f"old; byte-exact copy imported to {destination} "
                "under OPPONENT_MANIFEST.json (Task 7.1)"
            )

    def test_second_application_is_byte_identical(self) -> None:
        first = self.render(self.stopped)
        assert self.render(first) == first

    def test_unexpected_input_is_refused_in_every_inventory(self) -> None:
        for current in (self.base, self.stopped, self.render(self.stopped)):
            for key in ("retained", "authored", "excluded", "non_engine_changes"):
                changed = json.loads(current)
                changed[key].append({"path": "unexpected"})
                with (
                    self.subTest(key=key),
                    pytest.raises(ValueError, match="unexpected"),
                ):
                    self.render(encode(changed))

    def test_missing_imported_reference_path_is_refused(self) -> None:
        for missing in range(5):
            base = json.loads(self.base)
            stopped = json.loads(self.stopped)
            base["excluded"].pop(missing)
            stopped["excluded"].pop(missing)
            with pytest.raises(ValueError, match="imported reference"):
                updater.render(encode(stopped), encode(base), encode(stopped), {})

    def test_real_committed_run_one_manifest_is_an_accepted_input(self) -> None:
        def git(revision: str) -> bytes:
            return subprocess.run(
                ["git", "show", f"{revision}:engine_rs/TRIM_MANIFEST.json"],
                cwd=updater.ROOT,
                check=True,
                capture_output=True,
            ).stdout

        base, stopped = git("b8747b6"), git("21d0f45")
        result = updater.render(stopped, base, stopped, updater.CHANGES)
        assert updater.render(result, base, stopped, updater.CHANGES) == result
        for key in ("retained", "authored"):
            assert json.loads(result)[key] == json.loads(stopped)[key]


if __name__ == "__main__":
    unittest.main()
