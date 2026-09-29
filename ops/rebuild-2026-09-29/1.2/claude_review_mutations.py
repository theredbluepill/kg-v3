"""Claude review of Task 1.2: one source mutation per oracle test, then exact restore.

Each control replaces exactly one fragment, runs only the named test, records the
exit code and whether that test reported FAILED, and restores the original bytes
(verified by SHA-256) before the next control. Logs go to logs/claude-review/.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LOGS = Path(__file__).resolve().parent / "logs" / "claude-review"
G = "src/kaggriculture/grammar.rs"
K = "engine_rs/tests/grammar_kernel.rs"
TRIM = "scripts/check_engine_trim.py"
REC = "ops/rebuild-2026-09-29/1.2/oracle/record_reference.py"


def root(name: str) -> list[str]:
    return [
        "cargo",
        "test",
        "--locked",
        "--lib",
        f"kaggriculture::grammar::tests::{name}",
        "--",
        "--exact",
        "--test-threads=1",
    ]


def engine(name: str) -> list[str]:
    return [
        "cargo",
        "test",
        "--manifest-path",
        "engine_rs/Cargo.toml",
        "--locked",
        "--test",
        "grammar_kernel",
        name,
        "--",
        "--exact",
        "--test-threads=1",
    ]


def pytest(selector: str) -> list[str]:
    return [
        "uv",
        "run",
        "--offline",
        "pytest",
        selector,
        "-q",
        "-p",
        "no:cacheprovider",
    ]


# Each control tuple: label, oracle test, file, old fragment, new fragment, argv.
CONTROLS = [
    (
        "constants-item13",
        "grammar_constants",
        G,
        "12 => Ok(Self::Sheep),",
        "12 | 13 => Ok(Self::Sheep),",
        root("grammar_constants"),
    ),
    (
        "tables-low-index",
        "grammar_tables_match_reference_and_reachable_states",
        G,
        "i64::from(high_zero == 0 && present != 0)",
        "i64::from(high_zero != 0 && present != 0)",
        root("grammar_tables_match_reference_and_reachable_states"),
    ),
    (
        "exact-empty-render",
        "grammar_exact_commands",
        G,
        "let mut command = if kind == MarketKind::Empty {",
        "let mut command = if kind == MarketKind::None {",
        root("grammar_exact_commands"),
    ),
    (
        "malformed-padding",
        "grammar_rejects_malformed_i64",
        G,
        "        if value != 0 {\n"
        "            return Err(format!(\n"
        '                "padding',
        "        if value != 0 && active_end > TOKENS_PER_SEAT {\n"
        "            return Err(format!(\n"
        '                "padding',
        root("grammar_rejects_malformed_i64"),
    ),
    (
        "fixture-hire-inclusive",
        "grammar_fixture_oracle",
        G,
        "u16::from(self.hires) < self.shape.hire_limit",
        "u16::from(self.hires) <= self.shape.hire_limit",
        root("grammar_fixture_oracle"),
    ),
    (
        "encode-market-digits",
        "grammar_encode_round_trip",
        G,
        "        frame[Slot::MarketQuantityHigh as usize] = quantity / QUANTITY_BASE;\n"
        "        frame[Slot::MarketQuantity as usize] = quantity % QUANTITY_BASE;",
        "        frame[Slot::MarketQuantityHigh as usize] = quantity % QUANTITY_BASE;\n"
        "        frame[Slot::MarketQuantity as usize] = quantity / QUANTITY_BASE;",
        root("grammar_encode_round_trip"),
    ),
    (
        "encode-write-before-validate",
        "grammar_encode_rejects",
        G,
        "    validate_program(plan, &tokens, length)?;\n"
        "    out.copy_from_slice(&tokens);",
        "    out.copy_from_slice(&tokens);\n"
        "    validate_program(plan, &tokens, length)?;",
        root("grammar_encode_rejects"),
    ),
    (
        "coupled-no-hire-count",
        "grammar_coupled_hire_exact_distribution",
        G,
        "hires: self.hires + u8::from(self.market == MarketPhase::Hire),",
        "hires: self.hires,",
        root("grammar_coupled_hire_exact_distribution"),
    ),
    (
        "replay-plant-item",
        "grammar_replays_reference_program",
        G,
        "if matches!(kind, UnitKind::Pickup | UnitKind::Place | UnitKind::Plant) {",
        "if matches!(kind, UnitKind::Pickup | UnitKind::Place) {",
        root("grammar_replays_reference_program"),
    ),
    (
        "kernel-buy-land-as-hire",
        "decoded_programs_feed_kernel",
        G,
        'Self::BuyLand => "BUY_LAND",',
        'Self::BuyLand => "HIRE",',
        engine("decoded_programs_feed_kernel"),
    ),
    (
        "kernel-dense-hire-inclusive",
        "decoded_dense_241_and_hire_capacity",
        G,
        "u16::from(self.hires) < self.shape.hire_limit",
        "u16::from(self.hires) <= self.shape.hire_limit",
        engine("decoded_dense_241_and_hire_capacity"),
    ),
    (
        "kernel-reverse-hands",
        "decoded_actor_ordinals_match_engine_units",
        G,
        "let hands = tokens[SLOTS..actors * SLOTS]\n        .chunks_exact(SLOTS)",
        "let hands = tokens[SLOTS..actors * SLOTS]\n        .rchunks_exact(SLOTS)",
        engine("decoded_actor_ordinals_match_engine_units"),
    ),
    (
        "kernel-hire-render",
        "decoded_hire_masks_ignore_cash",
        G,
        'Self::Hire => "HIRE",',
        'Self::Hire => "BUY_LAND",',
        engine("decoded_hire_masks_ignore_cash"),
    ),
    (
        "kernel-north-render",
        "decoded_command_matrices_execute_both_seats",
        G,
        'Self::North => "NORTH",',
        'Self::North => "SOUTH",',
        engine("decoded_command_matrices_execute_both_seats"),
    ),
    (
        "kernel-unit-digit-swap",
        "decoded_transfer_quantities_have_inventory_effects",
        G,
        "QUANTITY_BASE * frame[Slot::UnitQuantityHigh as usize]\n"
        "                + frame[Slot::UnitQuantity as usize],",
        "QUANTITY_BASE * frame[Slot::UnitQuantity as usize]\n"
        "                + frame[Slot::UnitQuantityHigh as usize],",
        engine("decoded_transfer_quantities_have_inventory_effects"),
    ),
    (
        "kernel-market-item-slot",
        "decoded_sampled_oracle_programs_execute",
        G,
        "ActionItem::try_from(frame[Slot::MarketItem as usize])?.as_str(),",
        "ActionItem::try_from(frame[Slot::UnitItem as usize])?.as_str(),",
        engine("decoded_sampled_oracle_programs_execute"),
    ),
    (
        "kernel-market-digit-swap",
        "decoded_replay_actions_preserve_selected_states",
        G,
        "QUANTITY_BASE * frame[Slot::MarketQuantityHigh as usize]\n"
        "                + frame[Slot::MarketQuantity as usize],",
        "QUANTITY_BASE * frame[Slot::MarketQuantity as usize]\n"
        "                + frame[Slot::MarketQuantityHigh as usize],",
        engine("decoded_replay_actions_preserve_selected_states"),
    ),
    (
        "kernel-order-comparator",
        "kernel_comparator_rejects_nested_order_change",
        K,
        "(Value::Object(a), Value::Object(e)) => {",
        "(Value::Object(a), Value::Object(e)) if a.is_empty() && e.is_empty() => {",
        engine("kernel_comparator_rejects_nested_order_change"),
    ),
    (
        "trim-authored-unchecked",
        "test_check_engine_trim.py",
        TRIM,
        '        "Task 1.2 authored set",\n    )',
        '        "Task 1.2 authored set",\n    ) if False else None',
        pytest("tests/tools/test_check_engine_trim.py"),
    ),
    (
        "recorder-duplicate-id",
        "test_record_grammar_reference.py",
        REC,
        '            and row["id"] not in ids,',
        "            and True,",
        pytest("tests/tools/test_record_grammar_reference.py"),
    ),
]


def main() -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ, CARGO_BUILD_JOBS="2", CARGO_NET_OFFLINE="true", UV_OFFLINE="true"
    )
    only = set(sys.argv[1:])
    results = []
    for label, oracle, rel, old, new, command in CONTROLS:
        if only and label not in only:
            continue
        path = ROOT / rel
        before = path.read_bytes()
        if before.count(old.encode()) != 1:
            raise SystemExit(
                f"{label}: fragment count {before.count(old.encode())} != 1"
            )
        try:
            path.write_bytes(before.replace(old.encode(), new.encode()))
            run = subprocess.run(
                command, cwd=ROOT, env=env, capture_output=True, text=True
            )
        finally:
            path.write_bytes(before)
        restored = hashlib.sha256(path.read_bytes()).hexdigest()
        assert restored == hashlib.sha256(before).hexdigest()
        output = run.stdout + run.stderr
        (LOGS / f"{label}.log").write_text(
            f"$ {' '.join(command)}\n# mutation {rel}: {old!r} -> {new!r}\n"
            f"# exit {run.returncode}; restored sha256 {restored}\n{output}"
        )
        failed = run.returncode != 0 and ("FAILED" in output or " failed" in output)
        results.append(
            {
                "label": label,
                "oracle": oracle,
                "file": rel,
                "exit": run.returncode,
                "detected": failed,
                "restored_sha256": restored,
            }
        )
        print(json.dumps(results[-1]), flush=True)
    with (LOGS / "summary.jsonl").open("a") as handle:
        for row in results:
            handle.write(json.dumps(row) + "\n")
    return 0 if all(r["detected"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
