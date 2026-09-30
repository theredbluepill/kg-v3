# Pause checkpoint — 2026-09-29 ~09:58Z (owner travelling 1–2 h)

All workflows were stopped, and no Codex process was left running. The pod `w7ia3zvxqsvs3g` is RUNNING and idle (both GPUs at 0 MiB and 0 %, no processes). Uncommitted work stays in each worktree; nothing was reset.

## Integration branch
- `kg/isaiah-gap-closure` (`/Users/poonszesen/kg-v3-int`) is at `b8747b6`. The **1.3 merge landed** (Codex verify-merge-1-3-r2 APPROVE).

## Resume queue
1. **Merge chain (track A):** the teacher merge (`8fde43c`), then the evidence and cookbook merge (`kg/rebuild-model` at `8093d51`). Both are staged and Codex-verified before landing. Neither has started.
2. **1.4 native env:** worktree `kg-v3-env`, branch `kg/rebuild-env`, base `e197528`, 11 dirty files. Codex was mid-implementation.
   - Resume with `codex exec resume 01a0ec7f-ece8-7391-b9f6-e5c0209f9ee4`: same sandbox `-c` flags as the 1.2/1.3 resume, stdin from a resume prompt, run in the foreground and poll.
   - Then: Claude review, then Codex verify.
3. **1.5 stage 1:** worktree `kg-v3-adapter`, branch `kg/rebuild-adapter`, 24 dirty files.
   - Codex finished and wrote `codex/task-1.5-s1-report.md` (session `01a0ec82-358e-7e40-956c-bc4687aedfc2`), but it is not committed.
   - Next: commit Codex's work, Claude review, then stage 2 after 1.4 is approved.
4. **8-rank plan:**
   - Worktree `kg-v3-8rank`, branch `kg/rebuild-8rank`, implementation committed at `939d5ea`. Codex verify r1 was interrupted (session `01a0ec98-536f-72c0-a321-3a88caff667a`); rerun it.
   - Also correct the plan note: `winner_ce_6m` is not an Isaiah-aligned fallback. It changes the value loss to `winner_ce`, needs the `win_only` reward, and uses a different LR schedule. Our economic reward and the 3.2/3.4 guards reject it. A batch-shape-only variant (512×128) would be a deviation.
5. **GPU receipts driver race (F3):** worktree `kg-v3-gpuchecks`, 5 dirty files. The fix was in progress: block signals around spawn and registration, plus a regression test. Then run the Codex confirm r3.
6. **Value-gap diagnostic:** never reached the pod. Worktree `kg-v3-valuegap` holds at most a draft run statement. Relaunch the same workflow script (`wf_7d16c05c-ea4`).
7. **Afterwards:** stage 1.4 and 1.5 into integration, run the 2-rank smoke (teacher off), then BC 5.1/5.2 and Phase 6 (2-rank, then 8-rank; get owner approval and state the price before creating the 8-GPU pod).

Workflow scripts for resume are under `~/.claude/projects/-Users-poonszesen-kg-v3/.../workflows/scripts/`: `parallel-merges-1-4-1-5-wf_e135e6fc-800.js`, `value-gap-diagnostic-wf_7d16c05c-ea4.js`, `plan-for-8-rank-main-run-wf_19590eff-1b8.js` and `driver-race-fix-wf_f7dfad56-cfa.js`. Re-running with `resumeFromRunId` replays completed agents from cache.
