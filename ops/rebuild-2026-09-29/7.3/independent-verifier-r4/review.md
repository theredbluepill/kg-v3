Independent verification of Task 7.3, round 4

Reviewed `kg/rebuild-7-3` at exact HEAD `8a8bc48744fbc7295f45ae584ec2e8f0a0db99de`, parent `f23cd4fcfab3187ad008d0bfca2408d5f8f20fdf`, against base `0b8cf98ef57fc49a329dca4c8368c630586c4752` (also the merge base). Scope is the two r3 findings, not a new qualification of all Task 7.3. Both the requested external r3 report and the committed full r3 review were read. No tracked implementation, tests, documentation, cookbook or lockfiles were changed. No training, GPU, network or pod work was performed.

**Finding 1 — P2, false successful custody: RESOLVED for the reported failure states.**

`python/owl/kaggriculture/replay_export.py:50` stages bytes in a dot-prefixed exclusive temporary file, flushes and fsyncs them, and hard-links without replacement. Its `finally` removes staging files. `_publish` at line 81 records the paths it created and rolls them back on ordinary publication failures. `_write` at line 552 orders the episode before custody; directory fsync follows both links. `_publish_failure_custody` at line 565 removes the hash and verification claims, sets both completion flags false, attempts error custody and retires only after publication succeeds. When error custody fails, it adds a note and keeps the game active. A separate double-fault limitation is the P3 below; the universal “on any failure” language is broader than that evidence.

Independent probes in [test_r4_probes.py](test_r4_probes.py) inject faults into the staged path, rather than the no-longer-opened final episode path. Each case runs the unchanged source over the real native evaluator with `episodeSteps=3`, seed 91, stride 3 and one or two selected games:

| Injected fault | One game | Two active games | Observed persisted state |
| --- | --- | --- | --- |
| Episode staging open raises `PermissionError` | PASS | PASS | Only error custody; no episode or temporary file |
| Episode staging writes 20 bytes, then raises `OSError` | PASS | PASS | Only error custody; no partial episode or temporary file |
| Custody staging writes 20 bytes, then raises `OSError` | PASS | PASS | Published episode rolled back; only error custody |
| Custody staging close raises `OSError` | PASS | PASS | Published episode rolled back; only error custody |

All eight preserve the exact injected exception object. Game 0 records publication failure; in two-game cases game 1 records evaluation abort. Every sidecar has `status: error`, both completion flags false, no `episode_sha256` or `verification`, and the recorder has no active games afterward. Per-case `probe-*.json` receipts preserve complete records and the actual injected staging path.

Three additional probes fail the first staged-file fsync, second staged-file fsync, or final directory fsync. Each preserves exception identity, removes the attempt's episode/staging files, and leaves only error custody. A twelfth probe checks the hidden staging name and absence of the final file while it opens. **12/12 independent probes pass** ([log](probes.log), [command receipt](probes.json)). The requested live regression `test_partial_episode_write_during_live_evaluation_leaves_error_custody` also passes in the required 67-test suite. The existing failed-error-custody test verifies that the game stays active, the failure is a note and a later retry succeeds.

**Finding 2 — P2, canonical evaluation incomplete: DEFERRED (recorded), nonblocking by this review's instruction.**

All three required records name the Task 3.1 dependency and reopening:

- `ops/rebuild-2026-09-29/7.3/r3-fixes/receipt.md:53`: explicit deferral; at line 60, after Task 3.1 lands, call `native_evaluation.evaluate_native_games` through `_evaluate_games` and unskip acceptance for eight complete, byte-verified episodes with seed, seat and checkpoint custody.
- `ops/rebuild-2026-09-29/7.3/results.md:146`: explicitly deferred to Task 3.1, with acceptance kept skipped until it lands and a link to the detailed receipt.
- `cookbook/references/native-replay-export-preserves-kaggle-episodes.md:268`: Task 3.1 and reopening when `_create_eval_env` builds `KaggricultureEnv`; line 408 explicitly records the deferral.

`scripts/run_ppo.py:1606` retains the `NotImplementedError`, and `tests/scripts/test_run_ppo.py:3061` retains the skipped canonical acceptance. Receipt lines 66–69 explain retaining the older Task 1.5/model-policy guard text while the other lane implements Task 3.1. Nothing was implemented at this seam in this review; no canonical export completion is claimed.

**New finding — P3, rollback failure replaces the original publication exception.**

Location: `python/owl/kaggriculture/replay_export.py:90` (new `_publish` rollback loop).

If the final directory fsync raises and removal of the published custody file also raises, the unlink exception escapes the cleanup loop. `_write` and the evaluator then propagate that secondary exception; the original fsync exception survives only as `__context__`. This violates the new guarantee that the original publication exception is re-raised.

The independent [live double-fault probe](double_fault_probe.py) reproduces this over two native games. The raised exception is the injected rollback `PermissionError`, rather than the injected fsync `OSError`. Game 0 retains its complete custody and episode; game 1 gets error custody. Crucially, game 0's episode still matches its hash and independently passes byte verification. This requires two faults and does not reproduce successful custody for a missing or corrupt episode, so it is nonblocking P3 rather than a new P2. See [receipt](double-fault-live-result.json) and [log](double-fault.log).

Fix: catch rollback errors, attach them as notes to the original publication exception, and re-raise the original. Preserve consistency if successful custody cannot be removed: retain its matching episode rather than blindly continuing to remove it. Add a regression for this combined fsync/unlink failure. No fix was applied during verification.

**Independent source mutations.**

All **11/11 mutations were detected**; the combined baseline passes eight cases, and every mutation's targeted tests pass again after byte-for-byte restoration. The final run uses `/tmp/kg-t73-r4-mutations-9bih42mi/replay_export.py`, loaded explicitly before pytest collection. The actual repository module is never edited. This is an independently written runner, not a rerun of the author's mutation receipt.

Test keys used below:

- S: independent `test_r4_probes.py::test_dot_staging_precedes_visible_file`.
- D: `tests/kaggriculture/test_replay_export.py::test_episode_is_durable_before_custody_claims_it`.
- N: `tests/kaggriculture/test_replay_export.py::test_publication_never_replaces_a_path_created_meanwhile`.
- F: `tests/kaggriculture/test_replay_export.py::test_publication_failure_never_leaves_successful_custody` (episode/custody × write/close).
- A: `tests/kaggriculture/test_replay_export.py::test_unpublishable_error_custody_leaves_no_files_and_the_game_active`.

| Mutation | Test detecting it | Mutant result | Restored result |
| --- | --- | --- | --- |
| Replace the dot prefix with a visible staging filename | S | 1 failed | 1 passed |
| Remove staged-file fsync | D | 1 failed | 1 passed |
| Remove directory fsync | D | 1 failed | 1 passed |
| Replace no-replace hard link with `os.replace` | N | 1 failed | 1 passed |
| Publish custody before the episode | D | 1 failed | 1 passed |
| Remove published-path rollback | F, custody write and close cases | 2 failed, 2 passed | 4 passed |
| Suppress error custody after failed publication | F, all cases | 4 failed | 4 passed |
| Leave staging files behind | F, all cases | 4 failed | 4 passed |
| Keep episode hash in error custody | F, all cases | 4 failed | 4 passed |
| Keep verification in error custody | F, all cases | 4 failed | 4 passed |
| Retire the game when error custody also fails | A | 1 failed | 1 passed |

[Detailed mutations](mutations.json) name the exact source replacements, test node IDs, mutant hashes, log files, return codes and per-mutation restored hashes. [Final command receipt](mutations-final.json), [runner](run_mutations.py), and [restoration receipt](mutation-restoration.json) bind them to source SHA-256 `c8170e303fc9eda760f49fab4483720090f72afc5d6dae6bb6bdf9dbb1d5f7c8`. The first run used Python's system temporary directory; it is retained under `initial-mutations/`. The final qualifying run repeats all mutations under the requested `/tmp` location.

**Required checks and resource limits.**

All command runs use `CARGO_BUILD_JOBS=2 OMP_NUM_THREADS=2 UV_OFFLINE=1 WANDB_MODE=offline`. Python subprocesses have a verifier-only peak-RSS watchdog; each command has a 300-second timeout. `ps` is denied by the sandbox, so receipts report OS `getrusage` maximum child RSS, not aggregate simultaneous process-tree memory. The initial monitor setup failed; the required pytest command was rerun successfully with the replacement watchdog and metered receipt. No network fallback was attempted.

| Required command | Independently observed result | Wall time | Peak child RSS |
| --- | --- | ---: | ---: |
| `uv run --offline pytest tests/kaggriculture/test_replay_export.py tests/kaggriculture/test_replay_export_integration.py tests/owl/test_replay.py -q` | **67 passed**, 0 failed, 0 skipped | 8.420 s | 358,825,984 bytes |
| `uv run --offline mypy python/ scripts/` | **No issues in 68 source files** | 1.126 s | 98,992,128 bytes |
| `uvx --from rust-just just py-prepare` | **INCOMPLETE: memory safety stop**, exit 86; 2,219 collected. Progress shows 234 passes and 1 skip before stopping, not a final pytest total. Formatting (131 files unchanged), lint, syntax and mypy passed. | 69.514 s | 902,021,120 bytes |

Required receipts: [pytest](pytest.json), [pytest output](pytest.log), [mypy](mypy.json), [mypy output](mypy.log), [py-prepare](py-prepare.json), [py-prepare output](py-prepare.log).

Allocator/GC retries and a partitioned attempt also stopped under the memory ceiling; none is counted as a full pass. The in-process fresh mypy typing test and model tests reached the watchdog on those attempts. The maximum recorded peak across all commands was **989,675,520 bytes**, below 1 GB. All commands completed or stopped within 72 seconds. [Preparation attempts](preparation-attempts.json) records all collected/progress counts, durations and peaks; modified execution conditions live in the corresponding receipts and scratch plugins. The malloc attempt also sets `PYTHONMALLOC=malloc`. The default full `py-prepare` pass remains unverified in this round; the author's earlier 2,207 passes / 12 skips are not presented as independently repeated. The deferred canonical acceptance is among that broader suite's expected skips.

Because `py-prepare` did not reach its final recipe, `uv run --offline python scripts/check_doc_freshness.py` was run separately: PASS, “No doc updates required” ([receipt](docs-fresh.json)). The 12 independent probes and live double-fault reproduction used peaks of 346,734,592 and 383,057,920 bytes respectively; the final mutation run used 445,513,728 bytes and 40.970 seconds. These are bounded correctness checks, not performance or gameplay claims.

**Tracked-tree custody.**

[Final custody receipt](restoration-custody.json) confirms exact SHA-256 equality for all **3,418 tracked files**, unchanged tracked-path inventory and unchanged HEAD. `git diff --quiet` and `git diff --cached --quiet` both return zero. `git status --short` contains only `?? ops/rebuild-2026-09-29/7.3/independent-verifier-r4/`.

The before/after canonical tracked-file manifest SHA-256 is identical:
`a5fffadfd341c31d840fbaaca7a96ec7765c5087122e2b4ed67ec1d2b6501644`.
Both manifests and the final git status are retained beside this report. No durable cookbook adaptation was made because this was a verification-only task requiring no tracked modification.

VERDICT: APPROVE WITH EDITS
