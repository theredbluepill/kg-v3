Re-verified `aba04ff` on `kg/rebuild-7-4-brief`, including `3fcfb51..aba04ff`. **All prior findings are resolved at the design level. No new findings.**

Below, **B** means [7.4-packaging.md](/Users/poonszesen/kg-v3-t74/ops/rebuild-2026-09-29/briefs/7.4-packaging.md); cached sources are from the specified kaggle-environments archive.

| Finding | Status | Evidence |
|---|---|---|
| R5 normalization/status/counts | RESOLVED | B:143,267–273,450–456,614 requires raw validation, per-step inspection, 719 actual calls, zero caught errors and capture before terminal overwrite. |
| RR1 raw-action validator | RESOLVED | B:220–231,587–590 correctly specifies nested commands and integer quantities, plus native encode/re-decode with matching actor/order/hire context. Matches cached `kaggriculture.json:130–133`, `grammar.rs:721–968` and `briefs/1.4.md:48`. Scratch checks passed. |
| RR2 terminal faults | RESOLVED | B:143,591–592,614 explicitly captures `Agent.act` results before `Environment.step`. Cached `kaggriculture.py:960–963` overwrites status/reward before `core.py:636–637`; scratch exception and timeout probes confirmed this. |
| RR3 companion Reference | RESOLVED | [Reference:74](/Users/poonszesen/kg-v3-t74/cookbook/references/kaggle-packaging-reuses-the-starter-submission-path.md:74) reconciles normalization and terminal overwrite; :90–93 records resource limits. `cookbook/references/index.md:5` and prepended `cookbook/log.md:3–5` are updated. |
| RR4 T6 label | RESOLVED | B:353 now says T6, matching :603. |
| R1 Docker fetch | RESOLVED | No regression: B:285–291,593–594 explicitly requires copying `engine_rs` before fetch. |
| R2 runtime isolation | RESOLVED | No regression: B:302–317 retains fresh unmodified runtime, extracted files and both module-origin assertions. |
| R3 resources/equal quotas | RESOLVED | No regression: B:82–88,318–326,376–380 retains limits, receipt, size rejection and identical quotas. |
| R4 timing/process/watchdog | RESOLVED | No regression: B:330–337,360–379 retains whole-call timing, fresh processes, watchdog and reconciled overage arithmetic. |
| R6 single-seat/autoreset | RESOLVED | No regression: B:199–209,410–414 retains separate validation and excludes the autoreset row. |
| R7 custody/ignored outputs | RESOLVED | No regression: B:469–494 retains host verification, commit archive and ignored bulk outputs. |
| S1 `hire_limit` | RESOLVED | No regression: B:213–218 derives it from checkpoint `action_spec`. |
| S2 RNG states | RESOLVED | No regression: B:421–426 compares post-call states. |
| S3 late imports | RESOLVED | No regression: B:140 retains the appropriately qualified bundled-import risk. |
| S4 production ordering | RESOLVED | No regression: B:147,642–647 keeps production ordering explicitly unverified. |
| W1 telemetry | RESOLVED | No regression: B:514–536 retains shared logging, parent-only telemetry, v3 identity and visible outages. The shared logger’s hardcoded project remains an explicit implementation dependency. |

Scratch mutation results:

| Probe | Result |
|---|---|
| RR1 positive | An offline scratch build of byte-identical `grammar.rs` rendered an action containing hand commands, `BUY_SEED`, `SELL`, `HIRE`, `BUY_LAND`, an empty market command and integer quantities. Structural validation and native round trip accepted it. |
| RR1 negatives | **10 rejected:** string quantity, flat hands, boolean quantity, exceeded order/hire limits, wrong actor count, missing key, `None`, list and string returns. |
| RR1 guard mutations | Restoring the old string-list rule rejected the valid action. Removing the native round trip admitted an over-limit action. |
| RR2 | **7 checks passed:** final exception and timeout, corresponding mid-episode controls, two disabled-capture mutations and a clean final-call control. Final faults became `DONE` with numeric rewards; the harness retained both faults. Disabling capture exposed the status-only false pass. |
| R5 normalization | Cached schema normalized all **3** non-dict cases to PASS; raw validation rejected them. |

RR2 used the cached `Agent.act`, core step/interpreter processing and terminal interpreter code, with unrelated economics helpers stubbed and synthetic clocks. These probes establish guard behavior, not completed packaging integration.

The requested `py-prepare` command ran with `UV_OFFLINE=1`:

- **1,625 passed, 7 skipped**, in 47.36 seconds.
- Formatting: **116 files unchanged**; lint, syntax and docs freshness passed.
- Mypy: **63 source files passed**.
- Skips: native grammar binding, two CUDA pinned-memory cases, two FlashAttention cases, quantized backend and native Kaggriculture environment.
- All **5 cached source hashes matched**; diff whitespace check passed.
- Requested prohibited-text grep: **zero matches** across the brief and Reference.
- No submission is presumed; observation-only stateless inference, no v2 model code and one canonical trainer remain preserved.

All scratch files were removed. Initial and final `git status --porcelain` were both empty; HEAD remained `aba04ff`. No `.git` writes, Docker, training, GPU work, network installs or submissions occurred.

VERDICT: APPROVE