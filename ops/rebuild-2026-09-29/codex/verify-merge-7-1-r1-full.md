Independent verification of `kg/merge-7-1` HEAD `3895180d47cd0848a24b6a76daa7a67db90eff3b`, BASE `faed71773fa9f6414e4379ace904780349979cd8`, opponent parent `908c73fce0be298f81e229b08bf0ca082ad2e075`.

## Findings

**P2 — A receipt declared committed is absent from Git.** `ops/rebuild-2026-09-29/evidence-custody.json:424` lists `ops/rebuild-2026-09-29/codex/verify-7.1-r2/regeneration/replay-summary.json` under `committed.landed_with_task_7_1.post_inventory_copies`. The file exists locally, is 273 bytes, and its SHA-256 matches the stated source and manifest. However, `git ls-files --error-unmatch PATH` and `git cat-file -e HEAD:PATH` fail; `git check-ignore -v PATH` identifies `.gitignore:15` (`replay-*.json`). A clean clone therefore lacks it. `evidence-custody.md:73` claims 46 compact files / 115,127 bytes committed, while only 45 / 114,854 bytes are committed. All 11 DEFER copies are committed; two new prepare logs make 58 actual ops additions in 3895180.

Concrete fix: force-add this exact receipt (`git add -f ops/rebuild-2026-09-29/codex/verify-7.1-r2/regeneration/replay-summary.json`) after verifying its hash, or explicitly reclassify it as local and correct the counts and landing claim. No fix was applied by this verifier.

## Checks and commands

All writes and executions that could format/build files occurred in independent clones beneath `/tmp/verify-merge-7-1-UGqrrq`. All CPU commands used `OMP_NUM_THREADS=2 CARGO_BUILD_JOBS=2`. Venvs/build caches were copied with copy-on-write (not shared writable links to source); editable imports were retargeted and checked. No training or GPU work ran.

| Check / command | Results |
|---|---|
| `git show -s --format=%P b6cd4f2` | Exactly faed717 and 908c73f; both ancestors of HEAD. |
| `uv run pytest tests --collect-only -q` at BASE / opponent / HEAD | 1,701 / 1,724 / 1,793 IDs. HEAD exactly equals parent union, zero lost. |
| Literal `uv run pytest --collect-only -q` | BASE: 1,830 collected plus 6 errors; opponent: 1,732/pass; HEAD: 1,930 plus the same 6 errors. Existing ops/test module-name collisions, not a merge regression. |
| Root-wide `uv run pytest --collect-only -q --import-mode=importlib`, with `PYTHONPATH=<scratch>/ops/rebuild-2026-09-29/7.1` | 1,934 / 1,732 / 2,034 IDs; exact union, zero lost. Importlib alone needs the updater's local import directory on opponent/HEAD. |
| `cargo test --offline --locked --manifest-path <crate>/Cargo.toml -- --list` | Engine: 69 / 69 / 69; root: 258 / 258 / 258 (four ignored); opponent: absent / 22 / 22. Exact union, zero lost names. |
| `git diff faed717 HEAD`, `git diff 908c73f HEAD`, cookbook section/name comparison | All conflicted and auto-merged sections accounted for. 33 / 30 / 34 concepts; zero lost. Manifest, justfile, generator, parity tests match opponent parent exactly and had no intervening integration edits. |
| References / cookbook log / just recipes | 22 References indexed once each; 130 unique log headings, dated newest-first with landing on top; all 125 BASE and 104 opponent-parent sections preserved; all 28 recipes retained. |
| `python ops/rebuild-2026-09-29/7.1/update_trim_manifest.py` on guard clone | Byte-identical 61,282-byte manifest, SHA-256 `e460abbb56e4c25239ad3e4b2b2d0f69ac1f2f0c904314c64c9d75f366323927`. |
| `python scripts/check_engine_trim.py`; `python scripts/check_opponent_import.py` | Both pass. |
| Two guard-input mutations in scratch | One byte changed independently in Starter source and oracle-00 gzip; each checker exits 1 with import-hash/trace-hash diagnostic. Byte restoration then checker exits 0; scratch Git status empty. Merge landing introduces no new guard. |
| `uv run pytest -q ops/rebuild-2026-09-29/7.1/test_update_trim_manifest.py` | 8 passed, 12 subtests passed. |
| `CARGO_BUILD_JOBS=2 OMP_NUM_THREADS=2 UV_NO_SYNC=1 UV_OFFLINE=1 uvx --from rust-just just prepare` | Exit 0. Root 254 passed/4 ignored; engine 69 passed; opponents 22 passed; Python 1,771 passed/22 skipped. Formatting, Clippy, Ruff, docs lint, mypy (65 files), docs-fresh pass. Matches committed prepare log and coverage doc. |
| Cookbook hook via stdin JSON with `tool_name: Write` and absolute `tool_input.file_path` | `{}`/exit 0 for the single added/changed concept. All six fields present, first tag correct; all 40 repository sources tracked/resolvable. A file_path-only payload would not invoke lint; hook validates neither source existence nor first tag, so checked separately. |
| `git merge-base --is-ancestor` for phase tracker/plan | 42 phase rows / 116 cited commit occurrences; zero falsely merged claims. Task 7.1 tick and r2 approval match Git/reports. |
| Custody hashes/source existence | All 57 copy records match source/local bytes; 56 tracked, one absent (finding). 12 new left-local files (2,287,580 bytes) and two large Task 7.1 transcripts match. MANIFEST: 261 regular files match +1 symlink; DEFER: 89/90 match. The one stale verify-7.3-r2 transcript is already explicitly disclosed as changed after the inventory. Counts/bytes in Markdown and JSON agree arithmetically; actual committed count differs as above. |
| Independent secret scan | 186 changed tracked files (175 added, 11 modified), 4,581,566 raw bytes; nine gzip files expanded, 91,639,967 decoded bytes scanned. Zero credentials/private keys/provider keys/Bearer/SSH-public/Fernet/IPv4/pod-redaction-map hits. 335 host:port candidates are source-file:line references. Positive-control pattern checks 51/51 pass (43 main and eight quoted-provider supplements). No added file exceeds 512 KiB; no weights/array extensions. |
| Owner boundaries | Model, PPO trainer, run_ppo, root Rust/Cargo unchanged. Opponent identity/state stays in standalone scripted evaluator; no learned-input/loss/reward/normalization/selection connection introduced. No v2 neural model or second trainer. Future learned-seat integration remains explicitly unqualified. |
| Final `git status --short`, `git diff --quiet`, `git diff --cached --quiet` in staging | Empty / exit 0. HEAD unchanged. No original worktree modified. |

## Corpus-extension additions

No weights/array files were added. The eight `.jsonl.gz` files total 1,779,187 bytes; the replay `.json.gz` is also listed below. These are committed parity fixtures, not learned weights or a training corpus.

| File | Bytes |
|---|---:|
| `opponents_rs/fixtures/oracle/oracle-00-starter-vs-r04.jsonl.gz` | 178,455 |
| `opponents_rs/fixtures/oracle/oracle-01-r04-vs-ecobot.jsonl.gz` | 265,001 |
| `opponents_rs/fixtures/oracle/oracle-02-ecobot-vs-e776.jsonl.gz` | 268,960 |
| `opponents_rs/fixtures/oracle/oracle-03-e776-vs-starter.jsonl.gz` | 185,228 |
| `opponents_rs/fixtures/oracle/oracle-04-starter-vs-r04.jsonl.gz` | 178,049 |
| `opponents_rs/fixtures/oracle/oracle-05-r04-vs-ecobot.jsonl.gz` | 256,224 |
| `opponents_rs/fixtures/oracle/oracle-06-ecobot-vs-e776.jsonl.gz` | 263,133 |
| `opponents_rs/fixtures/oracle/oracle-07-e776-vs-starter.jsonl.gz` | 184,137 |
| `opponents_rs/fixtures/replay/REPLAY.json.gz` | 23,323 |

## Limits and receipts

An extra `check_opponent_import.py --original-sources` attempt could not re-read `/Users/poonszesen/kaggriculture`, which is absent on this host. The required default custody check and native original-oracle comparisons passed. Fresh original-Python generation was not claimed; the prior r2 reports and their committed evidence were inspected. The 22 Python skips retain the documented unavailable bindings/CUDA/backend/source and bounded-Mac limits.

- Full executed prepare: [prepare.log](prepare.log).
- Parent IDs/names and exact commands: [inventory/REPORT.md](inventory/REPORT.md), [pytest comparison](inventory/pytest-comparison.json), [wide comparison](inventory/pytest-wide-comparison.json), [Rust comparison](inventory/rust-comparison.json).
- Merge sections, note/source checks and ancestry: [docs/report.md](docs/report.md).
- Custody detailed report: [custody/report.md](custody/report.md).
- Custody details and scanner: [custody/custody_summary.json](custody/custody_summary.json), [scan summary](custody/secret_scan_summary.json), [positive controls](custody/secret_scan_positive_controls.json).
- Guard commands/mutations/restoration: [guards-results.json](guards-results.json); updater tests: [updater-tests.log](updater-tests.log).
- Final status: [final-status.json](final-status.json).

VERDICT: REQUEST CHANGES
