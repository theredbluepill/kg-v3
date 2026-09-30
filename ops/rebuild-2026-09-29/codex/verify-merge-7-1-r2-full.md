Independent verification, round 2: `kg/merge-7-1` at `cb7006b49d72aa2de39ecda28818e476e3c25446`, against BASE `faed71773fa9f6414e4379ace904780349979cd8`. Round 1 state: `3895180d47cd0848a24b6a76daa7a67db90eff3b`; opponent parent: `908c73fce0be298f81e229b08bf0ca082ad2e075`.

## Findings

No new P0–P3 findings. No fix required.

## Prior finding

**RESOLVED — r1 P2**, `ops/rebuild-2026-09-29/evidence-custody.json:424`: the receipt declared committed but previously absent is now in HEAD at `ops/rebuild-2026-09-29/codex/verify-7.1-r2/regeneration/replay-summary.json`. It is 273 bytes; SHA-256 `a796a5a8c4e100479da399cd93091b48a13461c70d86da4f90a2c8fe376ffcc7`, matching the manifest. The applied fix is the force-add in cb7006b.

For each of all 57 landing records, the verifier ran `git cat-file -e HEAD:<target_path>`, read its Git blob with `git cat-file blob HEAD:<target_path>`, and compared its byte count and SHA-256 to the manifest. All 57 match. The post-inventory records total 46 files / 115,127 bytes; promoted records total 11 / 677,342. Both equal the Markdown landing section. The checker rejects duplicate paths as well.

`git status --short --ignored --untracked-files=all -- ops/` and `git ls-files --others --ignored --exclude-standard -- ops/` show only one pre-existing `__pycache__/record_reference.cpython-312.pyc`, unrelated to the landing. `git check-ignore -v --stdin` over all 57 targets yields no ignored paths (exit 1). With `--no-index`, only the now-tracked receipt matches `.gitignore:15:replay-*.json`. Thus the existing ignore rule still matches the name, but it no longer prevents this receipt from being committed. No other intended landing receipt remains ignored.

## Delta and parent preservation

Tree comparison of 3895180 and cb7006b shows 4,132 to 4,133 entries: exactly one addition (the receipt, mode 100644), zero deletions, and zero changes to any pre-existing mode/type/blob tuple. Both parents and r1 are ancestors of HEAD. All 3,957 BASE paths and all 1,750 opponent-parent paths occur in r1; every one retains its r1 entry in HEAD. No parent content was lost relative to r1. This is incremental preservation, not a claim that the merge had no intentional differences from its parents.

The added JSON parses and equals the already committed `regeneration/replay.log` JSON. Its referenced fixture independently matches the committed replay fixture: SHA-256 `0d6fee093244de7e9b89757c9a2594fc16cc8392f5f12ff523d4e96a07987391`, 23,323 bytes, 24 cases. Recorded generation time, RSS and Python version remain historical receipt data; they were not re-measured here. `git diff --check 3895180..cb7006b` passes.

## Executed checks

All execution used `OMP_NUM_THREADS=2 CARGO_BUILD_JOBS=2`. The four requested commands ran in an independent scratch checkout of cb7006b with a copied, retargeted Python environment and copied native extension. `UV_NO_SYNC=1 UV_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1` prevented dependency synchronization, network access through uv and bytecode writes. Python, owl and owl.rs import paths were checked to resolve to scratch. The Git clone reads the source object store through an alternate; writes stay in the scratch clone. No training, GPU work, broad rebuild, formatter, source regeneration or original-worktree mutation ran.

| Check / command | Result |
|---|---|
| Landing custody Git blob audit | 57/57 existence, size and SHA-256 matches; promoted 11 / 677,342 bytes; post-inventory 46 / 115,127 bytes |
| Ignored-file scan and `git check-ignore` | 0 missing/ignored intended landing files; only 1 unrelated pre-existing Python cache ignored |
| `3895180..cb7006b` tree audit | 1 addition, 0 deletions, 0 existing entry changes; all 4,132 r1 entries retained |
| Scratch guard baseline | 57/57 matches, exit 0 |
| Scratch receipt deletion | Exit 1; exactly the missing receipt reported; 56/57 match |
| Restore after deletion | 57/57 matches, exit 0 |
| Scratch same-size alteration (`cases: 24` to `25`) | Exit 1; exactly the receipt hash mismatch reported, still 273 bytes; 56/57 match |
| Restore after alteration | 57/57 matches, exit 0; all 59 copied files (57 receipts + custody JSON/Markdown) equal their originals |
| `uv run pytest -q tests/tools/test_check_opponent_import.py tests/scripts/test_kaggriculture_parity.py` | 99 passed, 2 skipped in 4.23s, exit 0 |
| `uv run python scripts/check_engine_trim.py` | 1/1 invocation passed, exit 0; `engine trim manifest: OK` |
| `uv run --offline python scripts/check_opponent_import.py` | 1/1 invocation passed, exit 0; entry pins/custody only |
| `uv run python scripts/check_doc_freshness.py` | 1/1 invocation passed, exit 0; no pending changed/untracked paths, no DOCS_CURRENT override |
| Final staging and scratch status | Both `git status --short` empty; unstaged/staged diffs exit 0; HEAD unchanged |

Pytest skips: original-submission sources live in the sibling repository (`tests/tools/test_check_opponent_import.py:396`); full committed-trace regeneration exceeds the Mac two-live-game bound (`tests/scripts/test_kaggriculture_parity.py:371`). No new action-parity or fresh original-Python regeneration claim is made. Docs freshness examines pending changes, so its clean-tree pass does not independently review the committed merge's documentation.

## Receipts

- [Custody checker source](check_landing_custody.py), [all HEAD path/hash results](custody-head.json).
- [Ignored-file commands and outputs](ignore-audit.json).
- [Delta and parent tree audit](delta-audit.json).
- [Mutation driver](run_guard_mutations.py), [five guard trials and restoration](guard-summary.json); each trial also has a detailed JSON/log beside it.
- [Pytest output](pytest.log), [trim output](engine-trim.log), [opponent import output](opponent-import.log), [docs freshness output](doc-freshness.log) (empty output, exit 0).
- [Final source/scratch status](final-status.json).

All generated artifacts stay beneath `/private/tmp/verify-merge-7-1-r2-uaIEBC`. No original worktree was modified.

VERDICT: APPROVE
