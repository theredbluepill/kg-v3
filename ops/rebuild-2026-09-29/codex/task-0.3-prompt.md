You are Codex, working with Claude on the Kaggriculture v3 rebuild. Your checkout is this worktree (branch `kg/rebuild-codex`, based on `kg/isaiah-gap-closure` = Isaiah's clean base 32b3ec9 + carried cookbook/agent setup). Claude works in a separate checkout; do not touch other worktrees or branches.

Read first: `AGENTS.md`, `ops/rebuild-2026-09-29/plan.md` (Global Constraints, "Working with Codex", Task 0.3), and the cookbook Decision `cookbook/decisions/restart-the-port-from-isaiahs-clean-base.md`.

YOUR TASK: rebuild plan Task 0.3 — failed training runs must report as failed, with the traceback printed before distributed cleanup. The previous plan already specifies the tests and code for this exact change on these same Isaiah files. Read it with:
  git show kg/reference-2026-09-29:ops/gap-closure-2026-09-29/plan.md
and follow its "Task 1.1: Failed runs report as failed, with the traceback first" (Files, Interfaces, Steps 1–5). Summary: `MetricLogger.close(self, *, exit_code: int = 0)`; `WandbLogger.close` calls `self._run.finish(exit_code=exit_code)`; `DebugLogger.close` accepts it; `scripts/run_ppo.py` gains `_logger_session` (closes with exit_code=1 on exception) replacing `closing(create_logger(...))`; `python/owl/train/distributed.py` `distributed_session` prints `[rank{RANK}] training failed:` plus the traceback to stderr before `destroy_process_group`. Tests: new `tests/owl/train/test_logging.py`, plus additions to `tests/scripts/test_run_ppo.py` (update `_FakeLogger.close` signature) and `tests/owl/train/test_distributed.py`. Adapt anything that doesn't match the clean base's actual code; the clean base is Isaiah's original files.

RULES:
- TDD: write the failing tests, run them and see them fail, implement, run them and see them pass.
- This is an owner's Mac: run unit tests only. No training, evaluation or model-scale runs. No network needed; the env is already synced (`.venv`, cargo deps fetched).
- Verify with: `uv run pytest tests/owl tests/scripts tests/tools -m "not slow" -q` and `uvx --from rust-just just py-prepare` (formatting, lint, mypy, tests, docs-fresh). If docs-fresh flags mapped docs, update the mapped docs or rerun with `DOCS_CURRENT=1` after checking they are current.
- Follow AGENTS.md style: fail fast, no getattr/setattr in first-class paths, match surrounding code.
- Commit on `kg/rebuild-codex` with a clear message. Do not push, merge or touch `main`.
- If the change is material, add the cookbook record the AGENTS.md contract requires (note, index, prepended cookbook/log.md). A small fix may only need a log line linking the Decision it serves; use judgement and say what you did.

FINAL REPORT (your last message): the changed files; every command you ran with its pass/fail counts; the commit hash; anything that deviated from the old plan's code and why; and open questions for Claude's review.
