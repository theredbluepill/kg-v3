You are Codex, INDEPENDENT VERIFIER, re-verify round 2 of the staging branch kg/merge-7-1 (this worktree, /Users/poonszesen/kg-v3-m-7-1, HEAD cb7006b) against BASE faed717 (kg/isaiah-gap-closure). Leave no tracked modification in any worktree: scratch only under /tmp or /private/tmp, restore anything you touch byte-for-byte. Do not modify /Users/poonszesen/kg-v3, /Users/poonszesen/kg-v3-t71 or any other worktree. No training or GPU work; tiny CPU checks only (OMP_NUM_THREADS=2, CARGO_BUILD_JOBS=2).

Round 1 (/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/codex/verify-merge-7-1-r1.md, full report verify-merge-7-1-r1-full.md beside it) returned REQUEST CHANGES with one P2: ops/rebuild-2026-09-29/codex/verify-7.1-r2/regeneration/replay-summary.json was listed as committed in evidence-custody.json but was absent from HEAD because .gitignore:15 (replay-*.json) ignored it. The fix commit cb7006b force-adds that file.

Check:
1. Mark the r1 finding RESOLVED / PARTIAL / UNRESOLVED with evidence: every path in evidence-custody.json committed.landed_with_task_7_1 (promoted_from_defer and post_inventory_copies) exists in HEAD (git cat-file -e HEAD:<path>) and its blob SHA-256 equals the recorded sha256; the committed counts and bytes equal evidence-custody.md's "Task 7.1 landing" section (46 / 115,127 post-inventory; 11 / 677,342 promoted). Also check no other file under ops/ that the landing meant to commit is still ignored (git status --ignored, git check-ignore).
2. Review the delta 3895180..cb7006b for anything else, and confirm nothing from either parent (faed717, 908c73f) was lost relative to r1's state (the delta adds one file only).
3. Guard mutation: on a scratch copy (not this worktree), delete or alter the added replay-summary.json and show your custody hash check detects it, then restore and show it passes.
4. Run the relevant checks in a scratch copy or non-rewriting components here: uv run pytest -q tests/tools/test_check_opponent_import.py tests/scripts/test_kaggriculture_parity.py, uv run python scripts/check_engine_trim.py, uv run --offline python scripts/check_opponent_import.py, uv run python scripts/check_doc_freshness.py; report counts. Confirm `git status --short` here is empty afterwards.

End with findings (severity P0-P3, file:line, fix), prior-finding status, a checks table with counts, and exactly one final line:
VERDICT: APPROVE / APPROVE WITH EDITS / REQUEST CHANGES
(APPROVE WITH EDITS only when every finding is P3.)
