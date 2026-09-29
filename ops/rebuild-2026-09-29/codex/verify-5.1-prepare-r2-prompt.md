# Re-verify the Task 5.1 BC shard preparer (r2)

Your r1 report is `/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/codex/verify-5.1-prepare.md` (REQUEST CHANGES) and its evidence is under `/tmp/bc51-review-0z513d/`.
The fix is commit `356d19f` on `kg/rebuild-bc-now` in `/Users/poonszesen/kg-v3-bcnow`. Review `git diff 49255ac 356d19f`.

1. **P2 (resume identity).** `records/identity.json` now binds records to:
   - the checkout identity (`_source_identity`, including `prepare_sha256` and the git head);
   - the month, days, validation fraction, turn stride, hire limit and label pairing;
   - every archive's SHA-256.

   A mismatch, or records without an identity file, raises. Rerun your r1 winner-swap resume reproduction against `356d19f` in a scratch dir under `/tmp` and confirm it now fails. Say whether any label- or seat-affecting input is still unbound.
2. **P3s.** Is `--days` now defaulting to `22-28`? Are mypy and ruff clean on the changed files?
3. **Pod evidence to interpret; no action on the pod.** The production prep is running on the pod from `49255ac`: a fresh out_dir, `--days 22-28 --turn-stride 10 --workers 12`, with 7 symlinked day zips. Its `--pairing-sample 8` result on real 09-22..09-28 episodes was 5743/5752. All 9 mismatches differ only in one seat's `private` state, never public, at turns 335, 383, 431, 455, 527 and 623:

   | Episode | Matches | Mismatch turns |
   |---|---|---|
   | 111810488 | 718/719 | 383 (private1) |
   | 112862572 | 717/719 | 383 (private1), 527 (private0) |
   | 114483925 | 715/719 | 335 (p0), 431 (p1), 455 (p1), 623 (p1) |
   | 113243404 | 718/719 | 623 (p0) |
   | 114001770 | 718/719 | 527 (p1) |
   | 112158749, 112490397, 113637101 | 719/719 | none |

   From the pinned engine source, explain the likely cause. For example, a periodic RNG-driven private event that re-stepping from a single `steps[t]` cannot reproduce. Say whether it affects label correctness. Labels are the recorded `steps[t+1]` actions, not predicted states.
   Question: since fresh-run shards are unaffected by the resume fix, may the production prep from `49255ac` be used for BC training?

One round; blocking findings only (owner: no over-engineering for BC). Do not modify tracked files; scratch only under `/tmp`; no downloads, GPU or training. Cite file:line.

End with exactly one line: `VERDICT: APPROVE`, `VERDICT: APPROVE WITH EDITS`, or `VERDICT: REQUEST CHANGES`.
