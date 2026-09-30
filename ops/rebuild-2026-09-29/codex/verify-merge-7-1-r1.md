**P2 — A receipt declared committed is absent from HEAD.** [evidence-custody.json:424](/Users/poonszesen/kg-v3-m-7-1/ops/rebuild-2026-09-29/evidence-custody.json:424) lists `ops/rebuild-2026-09-29/codex/verify-7.1-r2/regeneration/replay-summary.json` as committed. Its 273 local bytes match the source and recorded hash, but `git cat-file -e HEAD:<path>` fails. `git check-ignore -v` identifies `.gitignore:15`, `replay-*.json`.

Consequently, **45 compact copies / 114,854 bytes** are committed, versus the claimed **46 / 115,127**. Fix by force-adding that exact receipt, or reclassifying it as local and correcting the counts. No other actionable findings.

Counts below use **BASE / opponent parent / HEAD**.

| Check and command | Result |
|---|---|
| `uv run pytest tests --collect-only -q` | **1,701 / 1,724 / 1,793**; HEAD exactly equals both parents’ union |
| Root-wide collection with `--import-mode=importlib` and updater directory on `PYTHONPATH` | **1,934 / 1,732 / 2,034**; exact union, zero lost IDs |
| `cargo test --offline --locked --manifest-path … -- --list` | Engine **69/69/69**; root **258/258/258**; opponents **absent/22/22**; zero lost names |
| Parent diffs and cookbook sections | **33/30/34** concepts; none lost. All **28** recipes retained. **22** References indexed once; **130** unique log headings, newest-first |
| Trim updater and both custody checkers | Pass; regenerated manifest byte-identical |
| Custody source/hash audit | All **56 actual committed copies** match source, manifest and Git blobs. **12/12** new left-local files and both oversized transcripts match |
| Independent secret scan | **186 files**, including **9 decompressed fixtures**; no secret or pod-host findings. **51/51** synthetic detection controls pass |
| Size/extensions | No additions over **512 KiB**; no weights/array extensions. Eight oracle `.jsonl.gz` fixtures and one replay `.json.gz`, individually listed in the report |
| Cookbook lint and ancestry | Required metadata valid; **40/40** tracked sources resolve. **42** phase rows / **116** cited commit occurrences checked; no false merged claims |
| Scratch guard mutations | Controller and oracle byte mutations both fail; both pass after restoration. Updater tests: **8 passed, 12 subtests passed** |
| `CARGO_BUILD_JOBS=2 OMP_NUM_THREADS=2 uvx --from rust-just just prepare` in scratch | **PASS**: engine **69**; root **254**, four ignored; opponents **22**; Python **1,771**, 22 skipped. Formatting, lint, typing and docs checks pass |
| Owner constraints | No learned identity connection, v2 model import or second trainer introduced |
| Final staging `git status --short` | **Empty**; HEAD unchanged |

The literal root-wide pytest collection encounters the same six pre-existing module-name collisions at BASE and HEAD; the successful collections above resolve that ambiguity. The optional original-source reread could not run because `/Users/poonszesen/kaggriculture` is absent. Default custody and native parity tests passed.

[Full report, commands, exact fixture list and receipts](/tmp/verify-merge-7-1-UGqrrq/REPORT.md). All scratch work stayed under `/tmp`; no original worktree was modified.

VERDICT: REQUEST CHANGES