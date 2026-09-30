# Verify the BC single-player filter (one round)

Review commit `42a8b39` on `kg/rebuild-bc-now` in `/Users/poonszesen/kg-v3-bcnow`: `git diff 356d19f 42a8b39`. The prior preparer reviews are `/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/codex/verify-5.1-prepare.md` (r1) and `verify-5.1-prepare-r2.md` (APPROVE).

Owner direction: BC should imitate **one player** from the 7 days, not every episode's winner. `--team NAME` makes only that team's seats policy seats, winning ones only unless `--include-losses`. The other seat still carries an observation for the critic and a PASS placeholder.

Check for blocking correctness issues only; the owner says BC stays lean:
1. **Seat selection:** it is correct for seat 0, seat 1, a mirror game (the team in both seats), draws, absent teams and malformed `TeamNames`. The policy-seat mask the trainer reads matches.
2. **Name hygiene:** the name never persists. Records, identity, manifest and the recorded command hold only `team_sha256` and a redacted argv. Look for any other leak path, such as exceptions, logs, pairing.json or print statements.
3. **Resume identity:** it now binds `team_sha256` and `include_losses`. Confirm a rerun with a different team fails.
4. **Denominators:** rejection counts separate "team absent" from "team lost". Totals stay consistent.
5. **Stateless policy:** filtering data by team is a collection choice, not a model input (see `cookbook/decisions/the-policy-is-stateless-and-observation-only.md`). Confirm no identity reaches shards or tensors.

Run the BC and preparer tests plus mypy and ruff on the changed files. Scratch output goes only under /tmp; do not modify tracked files. No downloads, GPU or training. Cite file:line.

End with exactly one line: `VERDICT: APPROVE`, `VERDICT: APPROVE WITH EDITS`, or `VERDICT: REQUEST CHANGES`.
