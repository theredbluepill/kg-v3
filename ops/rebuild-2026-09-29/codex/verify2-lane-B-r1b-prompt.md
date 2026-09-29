You are Codex, the INDEPENDENT VERIFIER of lane B (Claude's model branch kg/rebuild-model), checked out detached at 4fdb526 in this worktree. Leave no tracked modification.
Verify the diff kg/isaiah-gap-closure..HEAD (Tasks 2.1 encoder, 2.2 critic, and the fixes for your earlier findings in ops/rebuild-2026-09-29/codex/verify-stream-b-2.1.md) against ops/rebuild-2026-09-29/briefs/2.1-encoder.md, briefs/2.2-critic.md, docs/kaggriculture-contract.md (v4) and the plan (I0/I0b, L6).
Check:
- each of your findings 1–7 is actually resolved;
- the critic follows Isaiah's design (an OutputProjectionMLP over the two critic-value tokens, winner softmax, value 2p-1, critic gain and Muon exclusion);
- the tests are non-vacuous (try one or two mutations, then restore);
- the contract-valid fixture matches the contract.
Run: uv run pytest tests/kaggriculture -q; uv run pytest tests/owl tests/scripts tests/tools -m "not slow" -q; uv run mypy python/owl scripts; and report counts.
End with VERDICT: APPROVE / APPROVE WITH EDITS / REJECT, with findings (severity, file:line, fix).
