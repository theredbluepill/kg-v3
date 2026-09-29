---
type: "Reference"
title: "Failed training reports status before distributed cleanup"
description: "The rebuilt runner forwards failure exit codes to W&B and flushes rank-tagged tracebacks before process-group destruction; offline unit tests establish the local exception contract."
tags: ["kaggriculture-v3", "adaptation", "diagnostics"]
status: "verified-scoped"
generated: {"by": "openai/codex", "at": "2026-09-29"}
sources:
  - resource: "repository:python/owl/train/logging.py"
  - resource: "repository:python/owl/train/distributed.py"
  - resource: "repository:scripts/run_ppo.py"
  - resource: "repository:tests/owl/train/test_logging.py"
  - resource: "repository:tests/owl/train/test_distributed.py"
  - resource: "repository:tests/scripts/test_run_ppo.py"
  - resource: "repository:README.md"
  - resource: "repository:ops/rebuild-2026-09-29/briefs/0.3.md"
  - resource: "repository:ops/rebuild-2026-09-29/0.3-results.md"
  - resource: "reference-branch:kg/reference-2026-09-29/ops/gap-closure-2026-09-29/plan.md"
---

# Failed training reports status before distributed cleanup

This is rebuild Task 0.3 on Isaiah's clean base, serving the
[[../decisions/restart-the-port-from-isaiahs-clean-base|restart Decision]].
The reference plan's Task 1.1 supplies the implementation; the historical
[[shared-ppo-adapts-game-batches-without-a-second-loop|shared PPO Reference]]
records the upstream failure-status defect. Searches for failed runs, tracebacks,
cleanup and exit codes found that existing historical concept, but no current
rebuild contract. This separate Reference keeps the new contract distinct from
the earlier port's training and performance claims.

## Contract and adaptation inventory

- `python/owl/train/logging.py`: `MetricLogger.close` accepts keyword-only
  `exit_code: int = 0`. `WandbLogger` forwards it to `Run.finish`; `DebugLogger`
  accepts it without side effects.
- `scripts/run_ppo.py`: `_logger_session` replaces `closing` around the existing
  main-rank logger. Normal completion closes with 0; any escaping `BaseException`
  closes with 1 and is re-raised. The PPO loop and worker path are unchanged.
- `python/owl/train/distributed.py`: exceptions from context construction or
  the yielded body print `[rank{RANK}] training failed:` and the traceback to
  stderr, flush, then propagate through the existing destruction in `finally`.
- The three source-matched test files cover W&B forwarding, DebugLogger,
  runner wiring, RuntimeError/KeyboardInterrupt/SystemExit propagation, exact
  close/cleanup counts and stderr availability at cleanup entry. README documents
  the behavior; the task brief and result receipt retain the local evidence.

Future runner changes must preserve both exit status and traceback ordering;
logger finalization success alone does not establish successful training.

## Verification and limits

TDD on this version produced **9 expected failures and 2 passing controls**, then
**11 passes** after implementation. The required Python regression command and
the final `py-prepare` each pass **722 tests**, with **3 hardware/backend skips**.
Python preparation also passes formatting, lint, 3.11 syntax, mypy over 48 source
files and documentation freshness. A separate native Codex review finds no
task-scope blocker; Claude's cross-review remains pending.

`cargo test` and the full `prepare` attempt each report **148 passes, 7 failures,
2 ignored** because this checkout lacks the generation and replay parity fixtures.
No Rust source changed, no fixture download ran, and parity requirements were not
disabled. The receipt records the initial test-style lint failure and its repair.

W&B is a local test double and CUDA/NCCL entry points are mocked. This does not
qualify live telemetry, multi-rank execution, CUDA fault recovery, performance or
playing strength. The report precedes process-group destruction, but follows
logger finalization. Failures in logger finalization or distributed cleanup can
supersede the original exception while retaining its Python exception chain.
Hard kills and distributed startup failures before the existing `try` remain
outside the contract. Reopen if live evidence shows failure information is lost
at one of those boundaries; no training run is needed for this code change.
