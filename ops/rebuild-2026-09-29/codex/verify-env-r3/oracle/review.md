# Independent Task 1.4 oracle and recorder verification

Scope: `e197528...1e63597957ed4995dc3eac47e996349a485670a9`, branch `kg/rebuild-env`. Read the Task 1.4 brief, current native-buffer Reference, prior r2 finding and pinned TrainingBatch implementation. All mutations used `/private/tmp/kg-env-r3-oracle`; no tracked repository file or installed extension was changed. The lane's target and stopping conditions are in `run-statement.md`.

## Prior findings

- **RESOLVED — r2 P3 size/hash test masking**, formerly `tests/tools/test_record_kaggriculture_env_reference.py:255,269`. Current tests at lines 264–331 lower real size caps, alter declared sizes within their budgets, flip a same-length archive byte, or change exactly one digest. They assert the exact expected diagnostic before `np.load`. All six separate guard omissions fail a named current test. In particular, removing the compressed cap, archive hash or expanded archive hash now fails; no pinned-source hash mismatch was counted as a mutation kill.
- **RESOLVED — inherited r1 P3 semantic inventory guard masking.** The 17 coherent invalid-array cases refresh array metadata; the nine hash-only cases remain separate. All 14 semantic/inventory/hash guard omissions fail named current tests.
- **RESOLVED — inherited r1 P3 stale constructor status.** `docs/rl-api-specs.md:1060–1062` explicitly says Q1 was approved and incorporated by contract v4.2.

## Finding

**P3 — Final fixture-pair publication can leave a partial target after a filesystem error.** `scripts/record_kaggriculture_env_reference.py:369–370` replaces the NPZ and JSON separately. A scratch probe injecting `OSError` only on the second replacement proves:

- With no target, `publish_fixture` raises but leaves the NPZ without its JSON.
- With a previously loader-valid target, it raises after replacing the NPZ while preserving the previous JSON, so `load_fixture` rejects the resulting pair (`fixture compressed size differs`).

Both inputs were fully validated and source-consistent before the final replacement. This is narrower than worker rollback: the supervised worker correctly stages privately, and its failure cannot publish to the requested target. The problem is failure during the parent's final pair publication. The loader fails closed, so this does not permit false oracle acceptance or affect native runtime semantics. It does fall short of the brief's unqualified “no partial fixture publication” at line 630.

**Fix:** preserve the previous target pair and restore it if the second replacement raises; remove the first replacement for a fresh target on failure. Add tests for both fresh and existing targets. If the intended guarantee also includes an uncatchable kill between replacements, use a single atomic generation/commit-pointer publication design; exception rollback alone cannot provide that guarantee. Keep the exact worker-budget guarantee distinguished from final filesystem publication semantics.

Receipt: `pair-publication-probe-details.json`, `pair-publication-probe-runner.log`, reproducible script `probe_pair_publication.py`. The injected `os.replace` override was restored in `finally`; only disposable scratch files were written.

## Fresh checks and mutations

| Campaign | Fresh result | Receipt |
| --- | --- | --- |
| Restored copied recorder suite | 58 passed | `restored-recorder.log` |
| Final restored recorder plus complete replay | 59 passed, 18.42 s pytest / 18.77 s wall, 340,164,608 sampled peak RSS bytes | `final-restored-suite.log`, `.json` |
| Semantic inventory and per-array hash omissions | 14/14 killed by current tests | `guard-removal-details.json`, `guard-00.log` through `guard-13.log` |
| Size/hash omissions | 6/6 killed by current tests | `size-custody-mutations.json`, `mutant-*-cap/size/hash.log` |
| Source identity, ZIP inventory, exported-byte drift omissions | 3/3 killed by current tests | `size-custody-mutations.json` |
| Replay transition output and scalar math perturbations | 7/7 killed | `replay-mutations.log`, `replay-mutant-*.log` |
| Premature publication before validation/worker success | 2/2 killed by current tests | `publication-mutations.log`, `premature-*.log` |
| Coherent coverage, array schema, manifest, loader size/hash and CLI probes | 17/17 omissions distinguished | `extra-probes.log`, `run_extra_probes.py` |
| Fixed recipe and supervisor RSS probes | 2/2 omissions distinguished | `supervision-probes.log`, `run_supervision_probes.py` |
| ZIP timestamp, NPY version/layout/dimension/expanded-total boundaries | 5/5 omissions distinguished | `loader-guard-probes.log`, `run_loader_probes.py` |
| Reference archive path/type, resolved pin, source drift immediately before publication | 4/4 omissions distinguished | `source-boundary-probes.log`, `run_source_boundary_probes.py` |
| Second replacement failure | 2/2 cases expose partial publication; loader rejects both | `pair-publication-probe-details.json` |

Campaign counts deliberately overlap and must not be added as unique guards. The seven replay mutations perturb copied test outputs/math, not rebuilt native code; each transition mutation reaches the first-divergence diagnostic at game 0 / seed 17000 / step 0 and the intended field. The scalar mathematical oracle fails its separate assertion. Supplementary probes use inert load/archive/worker sentinels and never launch a reference build or recorder. The dimension-guard omission reaches the independent expanded-total guard; other header omissions reach the forbidden NumPy-load sentinel. These distinguish check ordering without claiming NumPy would accept malformed input.

The final 59-test check overlaps the parent's full requested test count and is not additive. Every campaign ran under the 115 s / 960 MiB watchdog and completed within its bounds.

## Independent provenance and corpus check

`verify_source_pin.py` independently obtains the complete reference inventory with `git ls-tree` and hashes each of **127 pinned files** via direct `git show`, comparing the exact inventory and every hash with the manifest. It also verifies the archive hash, **five local source hashes**, compressed/expanded fixture hashes and all array metadata/hashes. This is separate from trusting the recorder's own `source_identity()` implementation.

The pin is `65f0eac5bb00b18a9d3acce319c2a231cbd5dff0`; its `engine_rs/src/training.rs` hash is `1d6105cbf72c4ac655063edbc1ed42aab42c74d37c219493c7d88df571038518`. The Rust example calls the actual pinned `TrainingBatch::from_json` and `step`; encoding comes from its exported actor codec. The fixed policy reads only the current clock and hand count, and the two reward recipes match the brief. Source exports are checked after build and before publication. No new reference regeneration is claimed.

The existing frozen fixture contains **16 complete games × 719 = 11,504 transitions**, **23,008 seat programs**, **46,528 active frames**, and positive values for every required coverage counter in every game. It is **310,365 bytes compressed / 17,201,128 bytes expanded**. Its SHA-256 is `494bbf2c80af9adba66cbcfbccdfd7c638c7cc3bb74e438517dad3d39bb5c976`. The final full replay passes bitwise reward/done/bank/counter, seed and terminal comparisons; the independent scalar math allowance remains separate from exact-reference equality. Details: `source-pin-corpus.json`, `source-pin.log`.

## Restoration and harness limits

`restoration-custody.json` verifies all **12 copied sources/tests/fixtures** against their before hashes and their main-worktree counterparts; every file is byte-identical and unchanged. Recorder hash remains `156bee305dd86d0c8f33584dab0f6175810e52908723577418f6d9f5536f3499`; the copied replay test is also restored exactly. The tracked diff is empty.

The first copied-baseline invocation accidentally overlapped the first scratch mutation campaign. Four cases correctly stopped at changing source custody; this is preserved in `baseline-recorder.log` and excluded from behavioral results. All subsequent mutations were serialized; the fully restored recorder then passed all 58 tests and final recorder-plus-replay passed all 59. This was a harness scheduling error, not a repository failure.

Lane recommendation: **APPROVE WITH EDITS** for the P3 final-publication failure-safety issue. The prior r2 size/hash finding and both inherited r1 findings are resolved. No native-runtime or trajectory-equivalence defect was found in this lane.
