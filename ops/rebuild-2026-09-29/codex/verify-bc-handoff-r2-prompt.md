You are Codex, INDEPENDENT VERIFIER, re-verify round 2 of the BC-to-PPO handoff on branch kg/rebuild-bc-handoff (this worktree, /Users/poonszesen/kg-v3-bchandoff, HEAD ca51222) against BASE 218a05b (kg/rebuild-bc-now). Leave no tracked modification in any worktree: scratch only under /tmp or /private/tmp, restore anything you touch byte-for-byte. Do not modify /Users/poonszesen/kg-v3 or any other worktree. No training or GPU work; tiny CPU checks only (OMP_NUM_THREADS=2). BC stays lean: this is the last round; report blocking defects first.

Round 1 (/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/codex/verify-bc-handoff.md) returned REQUEST CHANGES with:
- P2: scripts/run_ppo.py _load_model_weights (teacher_init / initial last-best loader) accepted top-level opponent_id / hidden_state keys.
- P3: critic-evidence wording (700/720 seat values, 345/360 turns both seats; the gradient probe flips each model's own prediction; seat detection is a hypothesis; Isaiah had no BC-to-PPO path) in the Reference, README and ops README, plus index/log.
The fix commit ca51222 adds ppo.reject_unknown_checkpoint_keys, called by ppo._checkpoint_metadata and run_ppo._load_model_weights, extends tests/kaggriculture/test_bc.py::test_ppo_load_rejects_prohibited_checkpoint_state, and rewords the notes.

Check:
1. Mark each r1 finding RESOLVED / PARTIAL / UNRESOLVED with evidence. Reproduce your r1 teacher-loader probes (top-level opponent_id and hidden_state, on a full and on a minimal {"model": ...} checkpoint) and confirm they are rejected, while tests/scripts/test_run_ppo.py::test_load_model_weights_allows_base_checkpoint_for_lora_model and a minimal {"model": ...} teacher load still pass. Check every other checkpoint loader in scripts/run_ppo.py and python/owl/train/ppo.py for an unknown-key path.
2. Guard mutation on a scratch copy: remove the reject_unknown_checkpoint_keys call from run_ppo._load_model_weights and show the targeted test fails, then passes on restore.
3. Review the delta 5349e96..ca51222 for anything else, and confirm the r1-approved parts are unchanged.
4. Run: KG_V3_BC_BEST=/tmp/kg-v3-bc-best/checkpoint_bc_best.pt KG_V3_BC_SHARDS=/tmp/kg-v3-bc-best/shards-top1 uv run pytest -q tests/kaggriculture/test_bc.py tests/scripts/test_run_ppo.py tests/kaggriculture/test_teacher.py; uv run mypy python/owl scripts; uv run ruff check python scripts tests; uv run python scripts/check_doc_freshness.py; the cookbook lint on the changed notes (echo '{"tool_name":"Write","tool_input":{"file_path":"<abs path>"}}' | CLAUDE_PROJECT_DIR=$PWD node .claude/hooks/cookbook-lint.mjs). Report counts. Confirm `git status --short` here is empty afterwards.

End with findings (severity P0-P3, file:line, fix), prior-finding status, a checks table with counts, and exactly one final line:
VERDICT: APPROVE / APPROVE WITH EDITS / REQUEST CHANGES
(APPROVE WITH EDITS only when every finding is P3.)
