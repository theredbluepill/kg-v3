---
type: "Workflow"
title: "Run codex exec with closed stdin and wait for its verdict"
description: "Rebuild-plan rules for delegating to Codex (commit 9bfa0a0): close stdin with </dev/null, run codex exec in the foreground or poll until it exits, accept a run only when its -o report ends with a verdict, and resume a usage-limited run from its partial commit with a separate resume prompt. Supported by transcripts from this rebuild; the termination cause in two incomplete reviews is operator-reported and inferred."
tags: ["kaggriculture-v3", "workflows", "codex"]
status: "verified-scoped"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-09-29"}
sources: [{"resource": "repository:ops/rebuild-2026-09-29/plan.md"}, {"resource": "repository:ops/rebuild-2026-09-29/codex/brief-2.3-r3-transcript.log"}]
---

# Run codex exec with closed stdin and wait for its verdict

This workflow covers how the rebuild delegates briefs, implementations and reviews to Codex. Commit `9bfa0a0` added these rules to `ops/rebuild-2026-09-29/plan.md` ("Working with Codex"). Claude, as orchestrator, adopted them after episodes in this rebuild. They are not an owner directive.

## Rules

```bash
codex exec -C <worktree> -s <workspace-write|read-only> \
  -o ops/rebuild-2026-09-29/codex/<task>.md "<task prompt>" \
  </dev/null 2>&1 | tee ops/rebuild-2026-09-29/codex/<task>-transcript.log
```

1. **Close stdin (`</dev/null`).** With a non-tty stdin, `codex exec` prints "Reading additional input from stdin..." and reads stdin before it starts. A pipe that never closes therefore holds the run.
2. **Run in the foreground, or poll until the process exits.** A verifier or orchestrator agent must not return while `codex exec` is still running.
3. **Accept a run only from its report.** The `-o` report must exist and end with a verdict line. A transcript that contains `VERDICT` is not enough: prompts and earlier reports the reviewer reads also contain it.
4. **After a usage limit, resume.** Keep the failed attempt's transcript as `<task>-attemptN-<reason>-transcript.log`. Then resume from the partial commit with a separate `<task>-resume-prompt.md` instead of rerunning from scratch.

## Evidence

- **Stdin.** The tracked `codex/brief-2.3-r3-transcript.log` begins with "Reading additional input from stdin...". That run still finished with a verdict. The stall itself is operator-reported; no transcript records a hung run.
- **Incomplete reviews.** Two local, uncommitted transcripts stop mid-review without a verdict of their own: `verify-1.1b-r2-attempt2-killed-transcript.log` and `verify-2.3-r2-attempt1-killed-transcript.log`. Their `VERDICT` lines are quoted from the prompt or from earlier reports the reviewer read. The operator reports that the parent agent's return terminated the child. No termination signal or parent-lifecycle receipt was retained, so that cause is inferred.
- **Usage limit.** The local transcript `verify-1.1b-r2-attempt1-usage-limit-transcript.log` ends with Codex's "You’ve hit your usage limit" error. The resume pattern is shown by the local `task-1.2-impl-resume-prompt.md`, which continues brief 1.2 from the committed partial work.

## Verification

- On 2026-09-29 I read the cited transcripts on this checkout: line 1 of the stdin transcript, the ends of both incomplete-review transcripts and the location of every `VERDICT` line in them, the last lines of the usage-limit transcript, and the resume prompt. That confirms what the transcripts contain, not why the processes stopped.
- Only `brief-2.3-r3-transcript.log` and the plan are committed. The other transcripts are local working files in `ops/rebuild-2026-09-29/codex/`.

## Limits and gaps

- The rules were not tested with a controlled reproduction, such as an open pipe or an early parent return.
- No orchestration receipt ties a parent's return to a child's exit. The child's exit status (`${PIPESTATUS[0]}`) records only the child's own outcome. Tying the two needs a process-lifecycle log with timestamps for the parent's return and the child's exit.

Concept search before writing: `cookbook/` had no note on `codex exec`, stdin handling or usage limits. The [[profile-cuda-bottlenecks-with-nsight-systems|Nsight workflow]] is the only other Workflow and covers GPU profiling, not agent delegation.
