"""Run independent, isolated Rust guard mutations only in the archive scratch copy.

No command runs on import. Use --list to inspect the inventory or repeat --only
to choose cases. Every case restores its source bytes in finally; build errors
and watchdog stops are recorded and do not count as killed mutants.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = Path("/Users/poonszesen/kg-v3-env")
SCRATCH = ROOT / ".codex-tmp/verify-env-r3-core"
ENV_FILE = "src/kaggriculture/env.rs"
REWARD_FILE = "src/kaggriculture/reward.rs"
ADMISSION_FILE = "src/kaggriculture/admission.rs"
BINDINGS_FILE = "src/kaggriculture/bindings.rs"


@dataclass(frozen=True)
class Mutation:
    name: str
    path: str
    test: str
    old: str
    new: str
    reason: str


MUTATIONS = [
    Mutation(
        "reward-own-counters", REWARD_FILE,
        "reward_uses_only_own_counters_and_raw_banks",
        "self.econ_starvation_weight * c[0] as f64",
        "self.econ_starvation_weight * c[3] as f64",
        "Move starvation penalty to an irrelevant telemetry counter.",
    ),
    Mutation(
        "reward-telescoping", REWARD_FILE,
        "reward_episode_telescopes_with_explicit_output_rounding_budget",
        "        Ok(r)\n", "        Ok([0.0; 2])\n",
        "Return zero rollout rewards; the independent endpoint budget must reject.",
    ),
    Mutation(
        "reward-underflow-products", REWARD_FILE, "reward_admission_predicate_cases",
        "!(*w * *s > 0. || *w * *d > 0.)", "!(*s > 0. || *d > 0.)",
        "Admit coefficients whose binary64 event products both underflow.",
    ),
    Mutation(
        "reward-death-cap", REWARD_FILE, "reward_admission_predicate_cases",
        "(*cap <= 0. || !(*w * *s > 0. || *w * *d > 0.))",
        "(!(*w * *s > 0. || *w * *d > 0.))",
        "Drop the active death-shaping positive-cap requirement.",
    ),
    Mutation(
        "reward-ineffective-cap", REWARD_FILE, "reward_admission_predicate_cases",
        "if *i > 0. && *ic <= 0. {", "if false && *i > 0. && *ic <= 0. {",
        "Drop the active ineffective-shaping positive-cap requirement.",
    ),
    Mutation(
        "reward-active-cap-sum", REWARD_FILE,
        "reward_rejects_invalid_coefficients_and_handles_huge_counts",
        "if active >= 1. {", "if false && active >= 1. {",
        "Admit active economic caps totaling one or more.",
    ),
    Mutation(
        "reward-nonfinite", REWARD_FILE,
        "reward_rejects_invalid_coefficients_and_handles_huge_counts",
        ".any(|v| !v.is_finite() || **v < 0.)", ".any(|v| **v < 0.)",
        "Remove explicit NaN/infinity coefficient admission.",
    ),
    Mutation(
        "reward-negative", REWARD_FILE,
        "reward_rejects_invalid_coefficients_and_handles_huge_counts",
        ".any(|v| !v.is_finite() || **v < 0.)", ".any(|v| !v.is_finite())",
        "Remove explicit negative coefficient admission.",
    ),
    Mutation(
        "seed-stride", ENV_FILE, "seed_construction_reset_stride_and_overflow",
        "next.checked_add(self.stride)", "next.checked_add(1)",
        "Advance every stream by one regardless of the rank stride.",
    ),
    Mutation(
        "seed-overflow", ENV_FILE, "seed_construction_reset_stride_and_overflow",
        "next.checked_add(self.stride)", "Some(next.wrapping_add(self.stride))",
        "Wrap the seed counter instead of rejecting exhaustion.",
    ),
    Mutation(
        "step-live-seed", ENV_FILE, "batch_failure_preserves_every_published_byte",
        "    pub fn prepare_step(\n"
        "        &mut self,\n"
        "        tokens: &[i64],\n"
        "        lengths: &[i64],\n"
        "    ) -> Result<PendingBatch, EnvError> {\n",
        "    pub fn prepare_step(\n"
        "        &mut self,\n"
        "        tokens: &[i64],\n"
        "        lengths: &[i64],\n"
        "    ) -> Result<PendingBatch, EnvError> {\n"
        "        self.stream.next += self.stream.stride; // MUTATION: early live write\n",
        "Mutate the live seed counter before any failing step admission.",
    ),
    Mutation(
        "reset-live-seed", ENV_FILE,
        "reset_and_truncate_failures_preserve_every_published_byte",
        "    pub fn prepare_reset(\n"
        "        &mut self,\n"
        "        mask: &[bool],\n"
        "        truncate: bool,\n"
        "    ) -> Result<PendingBatch, EnvError> {\n",
        "    pub fn prepare_reset(\n"
        "        &mut self,\n"
        "        mask: &[bool],\n"
        "        truncate: bool,\n"
        "    ) -> Result<PendingBatch, EnvError> {\n"
        "        self.stream.next += self.stream.stride; // MUTATION: early live write\n",
        "Mutate the live seed counter before failing reset/truncate staging.",
    ),
    Mutation(
        "reset-terminal-clear", ENV_FILE,
        "reset_and_truncate_failures_preserve_every_published_byte",
        "            let (seeds, stream) = self.stream.reserve(mask)?;",
        "            for (slot, selected) in self.slots.iter_mut().zip(mask) {\n"
        "                if *selected { slot.terminal = None; }\n"
        "            }\n"
        "            let (seeds, stream) = self.stream.reserve(mask)?;",
        "Clear selected live terminal records before staging can fail.",
    ),
    Mutation(
        "truncate-whole-publish", ENV_FILE, "truncate_commits_only_selected_rows",
        "            commit_selected_rows(&mut pending.staging.buffers_mut(), out, selected);",
        "            let _ = selected;\n            pending.staging.publish(out)?;",
        "Replace selected-row publication with a whole-batch observation copy.",
    ),
    Mutation(
        "truncate-transition-clear", ENV_FILE, "truncate_preserves_transition_and_no_winner",
        "            let transition = if truncate {\n                None\n",
        "            let transition = if truncate {\n"
        "                Some(TransitionCache::zeros(self.n_envs()))\n",
        "Publish zero transitions during truncation, overwriting the saved transition.",
    ),
    Mutation(
        "horizon-off-by-one", ENV_FILE, "default_horizon_terminates_at_719",
        "i128::from(s.steps) >= i128::from(self.episode_steps) - 2",
        "i128::from(s.steps) >= i128::from(self.episode_steps) - 1",
        "Predict autoreset one turn late against the real kernel horizon.",
    ),
    Mutation(
        "hire-admission-wiring", ENV_FILE,
        "executed_unbounded_hire_cast_rejects_before_publication",
        "                                super::admission::validate_hire_cash(\n"
        "                                    self.hire_multiplier,\n"
        "                                    slot.hires,\n"
        "                                    hires,\n"
        "                                )?;",
        "                                let _ = (self.hire_multiplier, slot.hires, hires);",
        "Bypass post-execution HIRE cast admission in the real candidate lifecycle.",
    ),
    Mutation(
        "hire-no-execution", ADMISSION_FILE,
        "hire_cash_no_execution_does_not_cap_configuration",
        "if executed[seat] == 0 {", "if false && executed[seat] == 0 {",
        "Inspect unaffordable/unexecuted past-cost indices and overrestrict the envelope.",
    ),
    Mutation(
        "hire-safe-sequences", ADMISSION_FILE,
        "hire_cash_admits_safe_executed_sequences",
        "cost >= UINT64_CASH_LIMIT", "cost >= UINT64_CASH_LIMIT / 2.0",
        "Overrestrict valid executed HIRE cash at the signed-integer bound.",
    ),
    Mutation(
        "hire-rounded-threshold", ADMISSION_FILE,
        "hire_cash_rejects_cost_that_rounds_up_to_two_to_64",
        "cost >= UINT64_CASH_LIMIT", "cost > UINT64_CASH_LIMIT",
        "Admit an exact cost below u64::MAX whose f64 conversion equals 2^64.",
    ),
    Mutation(
        "hire-later-seat", ADMISSION_FILE,
        "hire_cash_checks_later_executed_hire_and_both_seats",
        "for seat in 0..2 {", "for seat in 0..1 {",
        "Omit seat one's executed HIRE admission.",
    ),
    Mutation(
        "hire-oversized", ADMISSION_FILE,
        "hire_cash_rejects_oversized_exact_cost_without_float_saturation",
        "cost >= UINT64_CASH_LIMIT", "cost >= UINT64_CASH_LIMIT * 2.0",
        "Move the upper bound above an oversized exact executed cost.",
    ),
    Mutation(
        "hire-negative-multiplier", ADMISSION_FILE,
        "hire_cash_rejects_negative_multiplier_explicitly",
        "if multiplier < 0 {", "if false && multiplier < 0 {",
        "Drop the negative-multiplier helper admission.",
    ),
    Mutation(
        "observe-holds-gil", BINDINGS_FILE,
        "native_observe_releases_gil_with_controlled_latch",
        "fn native_work<T: Send, F: FnOnce() -> T + Send>(py: Python<'_>, work: F) -> T {\n"
        "    py.detach(|| {\n"
        "        #[cfg(test)]\n"
        "        detached_test::await_python_progress();\n"
        "        work()\n"
        "    })\n"
        "}",
        "fn native_work<T: Send, F: FnOnce() -> T + Send>(_py: Python<'_>, work: F) -> T {\n"
        "    #[cfg(test)]\n"
        "    detached_test::await_python_progress();\n"
        "    work()\n"
        "}",
        "Keep the interpreter attached while the actual observation latch waits.",
    ),
]


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def classify(receipt: dict[str, Any], log: str) -> str:
    if receipt.get("stop_reason") is not None:
        return "resource_guard"
    if re.search(r"test result: FAILED\. \d+ passed; [1-9]\d* failed;", log):
        # Distinguish an assertion/expected-error discriminator from arbitrary
        # execution failure, while retaining the complete log for inspection.
        markers = (
            "assertion", "missing rollback failure", "missing reset/truncate failure",
            "unbounded hire must reject", "Python thread could not progress",
            "called `Result::unwrap_err()` on an `Ok` value",
            "terminal seed prediction disagrees with kernel",
            "unselected observation bytes changed",
        )
        return "killed_assertion" if any(marker in log for marker in markers) else "test_failure_review"
    # Cargo ends a completed failing test with "error: test failed". Inspect
    # the actual harness outcome first so that trailer is not a compiler error.
    if "could not compile" in log or re.search(r"^error(?:\[E\d+\])?:", log, re.M):
        return "compile_error"
    if receipt.get("exit_status") == 0:
        return "survived" if re.search(r"test result: ok\. [1-9]\d* passed;", log) else "no_tests"
    return "execution_error"


def run_one(mutation: Mutation, seconds: float) -> dict[str, Any]:
    source = SCRATCH / mutation.path
    primary = ROOT / mutation.path
    if source.is_symlink() or source.resolve().is_relative_to(ROOT / "src"):
        raise RuntimeError(f"mutation source escapes archive: {source}")
    if os.path.samefile(source, primary):
        raise RuntimeError(f"mutation source aliases tracked primary source: {source}")
    original = source.read_bytes()
    if original != primary.read_bytes():
        raise RuntimeError(f"scratch is not pristine before mutation: {mutation.path}")
    record: dict[str, Any] = {
        "name": mutation.name,
        "path": mutation.path,
        "test": mutation.test,
        "reason": mutation.reason,
        "original_sha256": digest(original),
        "classification": "mutation_setup_error",
    }
    prefix = HERE / f"core-{mutation.name}"
    try:
        text = original.decode("utf-8")
        count = text.count(mutation.old)
        if count != 1:
            raise RuntimeError(f"expected one mutation anchor, found {count}")
        mutant = text.replace(mutation.old, mutation.new, 1).encode("utf-8")
        record["mutant_sha256"] = digest(mutant)
        patch = "".join(difflib.unified_diff(
            text.splitlines(keepends=True), mutant.decode("utf-8").splitlines(keepends=True),
            fromfile=f"original/{mutation.path}", tofile=f"mutant/{mutation.path}",
        ))
        prefix.with_suffix(".diff").write_text(patch)
        record["diff"] = str(prefix.with_suffix(".diff"))
        source.write_bytes(mutant)
        command = [
            sys.executable, str(HERE / "bounded.py"), "--seconds", str(seconds),
            "--name", str(prefix), "--", "cargo", "test", "--locked", "--offline",
            "--lib", mutation.test, "--", "--test-threads=1",
        ]
        record["command"] = command
        # Remove previous run receipts before execution, so a runner failure
        # cannot accidentally be classified from stale output.
        for suffix in (".json", ".log", ".runner.log"):
            prefix.with_suffix(suffix).unlink(missing_ok=True)
        env = os.environ | {
            "CARGO_BUILD_JOBS": "2", "CARGO_NET_OFFLINE": "true",
            "UV_OFFLINE": "true", "RUST_TEST_THREADS": "1",
        }
        # All detailed output goes to artifacts; the caller receives one concise
        # result per mutation. The watchdog owns the child process group.
        with prefix.with_suffix(".runner.log").open("w") as output:
            result = subprocess.run(command, cwd=SCRATCH, env=env, stdout=output,
                                    stderr=subprocess.STDOUT, check=False)
        record["runner_exit_status"] = result.returncode
        receipt_path = prefix.with_suffix(".json")
        record["bounded_receipt"] = str(receipt_path)
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            log = prefix.with_suffix(".log").read_text()
            record["classification"] = classify(receipt, log)
            record["bounded"] = receipt
        else:
            record["classification"] = "execution_error"
            record["error"] = "watchdog did not publish a receipt"
    except Exception as error:
        record["error"] = repr(error)
    finally:
        source.write_bytes(original)
        restored = source.read_bytes()
        record["restored_sha256"] = digest(restored)
        record["restored_byte_exact"] = restored == original
        record["primary_unchanged"] = primary.read_bytes() == original
        prefix.with_suffix(".mutation.json").write_text(json.dumps(record, indent=2) + "\n")
        if not record["restored_byte_exact"] or not record["primary_unchanged"]:
            raise RuntimeError(f"source custody failure: {mutation.name}")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="print inventory without editing or running")
    parser.add_argument("--only", action="append", default=[], metavar="NAME")
    parser.add_argument("--seconds", type=float, default=115)
    args = parser.parse_args()
    names = {mutation.name for mutation in MUTATIONS}
    unknown = set(args.only) - names
    if unknown:
        parser.error(f"unknown mutations: {sorted(unknown)}")
    selected = [mutation for mutation in MUTATIONS if not args.only or mutation.name in args.only]
    if args.list:
        for mutation in selected:
            print(f"{mutation.name}: {mutation.path} -> {mutation.test}")
        return 0
    if not (SCRATCH / "Cargo.toml").is_file() or (SCRATCH / ".git").exists():
        parser.error("expected the non-worktree archive scratch copy")
    results = []
    for mutation in selected:
        record = run_one(mutation, args.seconds)
        results.append(record)
        (HERE / "core-mutations.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps({key: record[key] for key in
                          ("name", "classification", "restored_byte_exact")}), flush=True)
    # A survived mutant, build error, missing test, or resource limit all require
    # review. Do not misrepresent any of them as an assertion discriminator.
    return 0 if all(record["classification"] == "killed_assertion" for record in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
