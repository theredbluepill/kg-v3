# Verify the Task 5.1 BC shard preparer (one round)

Review commit `49255ac` on branch `kg/rebuild-bc-now` in worktree `/Users/poonszesen/kg-v3-bcnow`.
Diff scope: `git -C /Users/poonszesen/kg-v3-bcnow diff 8d29ae3 49255ac`:

- new: `scripts/kaggriculture_prepare_bc.py`, `tests/scripts/test_kaggriculture_prepare_bc.py`;
- changed: `python/owl/kaggriculture/bc_data.py`, `python/owl/train/bc.py`, `tests/kaggriculture/test_bc.py`.

Governing brief (Codex r4 APPROVE): `/Users/poonszesen/kg-v3-bcbrief/ops/rebuild-2026-09-29/briefs/5.1-bc-data.md`.
The 5.2 trainer that consumes these shards is `scripts/train_bc.py` and `python/owl/train/bc.py` in the same worktree.

## Owner directives that bind this review

- "kaggle obs shd be step+1". Observation `steps[t]` is labeled with the action recorded at `steps[t+1]`. An earlier draft read `steps[t]`; the commit fixes that line (around line 391). Confirm the fix is complete. Check every place that derives labels, placeholders, actor counts, market or farmer tokens, and turn ranges.
- Data: 7 days (2026-09-22..09-28), winning seat only, single player. Draws may keep both seats; say whether that is sound.
- "for BC we don't have to be perfectly/over-engineered." This is one verification round. Report **blocking correctness findings** (P1/P2), meaning anything that would produce wrong labels, wrong observations, a wrong seat, a leaked or mixed split, shards the trainer cannot load or misreads, silent data loss, or an unusable resume. Mark non-blocking polish P3 and keep it short. No mutation sweeps.

## Checks to run

1. `cd /Users/poonszesen/kg-v3-bcnow && OMP_NUM_THREADS=2 uv run pytest tests/kaggriculture/test_bc.py tests/scripts/test_kaggriculture_prepare_bc.py -q`, plus mypy on the changed Python files.
2. Read-trace end to end, one episode through `admit_turns` → encoding → shard write → manifest → `bc_data.py` reader → the tensors `bc.py` trains on. Confirm:
   - the observation and label turn indices align (`t` and `t+1`);
   - the seat is the winner by final bank;
   - illegal or undecodable actions are handled explicitly;
   - the manifest keys and dtypes match what the reader validates.
3. If a local Kaggriculture episode JSON or day archive exists (the brief-5.1 review used local episode `114406062`; search `~/kaggriculture-v2` read-only), run the preparer on 1–3 episodes into a scratch dir under `/tmp`. Include `--pairing-sample` or `--pairing-only`. Report the pairing-check match counts with the recorded `info.seed`. If no local data exists, say so; do not download anything.
4. Confirm the CLI works with a directory of symlinked day zips (the pod layout symlinks 7 `kaggriculture-episodes-2026-09-DD.zip` files into one dir) and with `--days 22-28`.

## Constraints

- Do not modify tracked files, commit, push, or touch `~/kaggriculture-v2` (read-only). Write scratch output only under `/tmp`.
- No GPU, no network downloads, no training.
- Cite file:line for every finding.

End the report with exactly one line: `VERDICT: APPROVE`, `VERDICT: APPROVE WITH EDITS`, or `VERDICT: REQUEST CHANGES`.
