Reviewed `3fcfb51` on `kg/rebuild-7-4-brief`. **Changes are still required:** the new action validator would reject valid Kaggriculture actions. Most prior findings are resolved at the design level.

Below, **B** means [7.4-packaging.md](/Users/poonszesen/kg-v3-t74/ops/rebuild-2026-09-29/briefs/7.4-packaging.md); **K** means the specified cached `kaggle_environments/` directory. Repository paths are relative to the worktree.

| Prior finding | Status | Evidence |
|---|---|---|
| R1 Docker fetch layer | RESOLVED | B:275–281,578 explicitly copies `engine_rs` before fetch. `Dockerfile.kaggle:51–57` currently omits it; `Cargo.toml:15` requires it. Scratch offline fetch reproduced and repaired the failure. |
| R2 fresh runtime load | RESOLVED | B:292–307 requires an unmodified runtime container, extracted files, runtime interpreter and both module-origin assertions. `Dockerfile.kaggle:44–49,64–66` confirms why checking the build environment was insufficient. |
| R3 resources/equal CPU | RESOLVED | B:80–86,308–315,366–370 specifies limits, a dated receipt, size rejection and equal quotas. The indexed [official FAQ](https://www.kaggle.com/competitions/kaggriculture/overview/citation) independently confirms 1.6 vCPUs, 6.5 GiB RAM, 8 GiB disk and 100 MiB. |
| R4 timing/processes/watchdog/budget | RESOLVED | B:320–327,350–363 covers the billed call, fresh processes and watchdog; 20 seconds initially means ≤19 seconds overage, with ≤29 seconds total. Supported by K/`agent.py:191–224` and K/`core.py:325–332,629–632`. |
| R5 normalization/statuses/counts | PARTIAL | B:141,257–263,440–446 adds the requested checks, consistent with K/`utils.py:150–195`. However, B:218–220 defines the wrong action structure; the terminal reward claim also needs qualification. Findings below. |
| R6 single-seat/autoreset alignment | RESOLVED | B:199–207,400–404,564–566 scopes a separate validator and excludes the reset row. `python/owl/kaggriculture/types.py:174–207` retains strict two-seat checks; `briefs/1.4.md:541` specifies reset before publishing terminal observations. That lifecycle remains a dependency, not executed evidence. |
| R7 Docker custody/ignored outputs | RESOLVED | B:459–484,578–585 specifies host verification, commit archive, identity transfer and ignored outputs. Supported by `.dockerignore:1`, `.gitignore:24`, and `scripts/build_kaggle_submission.sh:37`. |
| S1 `hire_limit` | RESOLVED | B:211–216,570 reads checkpoint `action_spec`; `types.py:78–80` defines it and `python/owl/model/kaggriculture.py:694` uses it. |
| S2 post-call RNG states | RESOLVED | B:411–416 explicitly compares post-call states. `python/owl/model/kaggriculture_actor.py:75–86` consumes RNG; the scratch extra-draw mutation demonstrated why action equality alone is insufficient. |
| S3 late imports | RESOLVED | B:138 appropriately says bundled imports *can* fail. K/`agent.py:51–59` temporarily adds the directory; the scratch loader mutation confirmed the distinction from standard-library imports. |
| S4 production ordering | RESOLVED | B:145,626–631 explicitly retains the unverified risk. K/`utils.py:105–121` supports local ordering; K/`agent.py:91` establishes only client serialization. |

The remaining findings are:

1. **P2 — The raw-action validator rejects canonical actions.** [B:218](/Users/poonszesen/kg-v3-t74/ops/rebuild-2026-09-29/briefs/7.4-packaging.md:218) requires every value to be “a list of strings.” `hands` and `market` contain nested command lists, and quantities are integers. See K/`envs/kaggriculture/kaggriculture.json:130–133` and [grammar.rs:730](/Users/poonszesen/kg-v3-t74/src/kaggriculture/grammar.rs:730), `:754–784`. **Fix:** describe the actual nested structure and validate through the native grammar with actor/order/hire context. Add positive tests containing hands, market orders and numeric quantities.

2. **P3 — F5 overstates failure rewards at termination.** [B:141](/Users/poonszesen/kg-v3-t74/ops/rebuild-2026-09-29/briefs/7.4-packaging.md:141) says exceptions/timeouts set rewards to `None`. K/`kaggriculture.py:960–963` replaces both status and reward before K/`core.py:636–637` checks them. The scratch terminal mutation reproduced `ERROR → DONE` with a numeric reward. **Fix:** qualify F5 and explicitly capture exceptional/timeout results from `Agent.act` before `Environment.step`, including a final-call fault test.

3. **P3 — The companion cookbook Reference contradicts the revision.** [Reference:74](/Users/poonszesen/kg-v3-t74/cookbook/references/kaggle-packaging-reuses-the-starter-submission-path.md:74) still says non-dict returns lose; `:81–82` still calls resource limits unknown. **Fix:** reconcile these claims with the revised brief and update its index/log according to the cookbook contract.

4. **P3 — Incorrect task reference.** [B:343](/Users/poonszesen/kg-v3-t74/ops/rebuild-2026-09-29/briefs/7.4-packaging.md:343) labels latency measurement T7; `:588` assigns it T6. **Fix:** change that reference to T6.

W1 is feasible through the shared logger. [B:504–526](/Users/poonszesen/kg-v3-t74/ops/rebuild-2026-09-29/briefs/7.4-packaging.md:504) specifies parent-only logging, v3 identifiers, separate experiment/attempt identity, visible outages and source-bound local receipts, consistent with [the governing Decision:16](/Users/poonszesen/kg-v3-t74/cookbook/decisions/evaluation-preserves-generality-and-evidence.md:16).

The claimed dependency is accurate at both `b8747b6` and this head: [logging.py:65](/Users/poonszesen/kg-v3-t74/python/owl/train/logging.py:65) explicitly passes `project="orbit-wars"`. The planned extension must flow through `create_logger` as well as `WandbLogger`. Current initialization errors propagate, so the promised outage handling still needs implementation. The lazy import at `:57` permits keeping W&B outside the agent process; the packaging script copies `owl`, not the external `wandb` package. No live telemetry or tarball isolation was verified.

Scratch mutations produced these results; they exercise proposed guards, not a completed 7.4 implementation:

| Family | Mutation result |
|---|---|
| R1 fetch | Missing `engine_rs`: offline locked fetch exited 101. Copying it: exited 0. |
| R2 origin | Synthetic extracted modules passed; resolving them outside extraction failed. No ABI claim. |
| R3 resources | Changing the second CPU quota from 1.6 to 2 failed equality. A sparse 100 MiB file passed; one extra byte failed. |
| R4 process/watchdog | Sleeping child was killed externally. Repeating Torch interop setup in one process failed; two fresh processes passed. |
| R5 actions/status | Cached schema normalized `None`, list and string to PASS. Proposed guard caught those but rejected a valid nested/quantity action. An isolated terminal interpreter call erased injected `ERROR`; unrelated economics helpers were stubbed. |
| R6 validator | Scratch single-seat adaptation accepted `[1,1]`, rejected the other specified shapes, and retained NaN rejection. Existing two-seat validator rejected `[1,1]`. |
| R7 custody/ignore | Bulk artifact paths were already ignored; root tarball became ignored after adding the proposed line to a scratch ignore file. Commit archive bytes remained unchanged by a scratch working-copy mutation. No `.git` writes. |
| S1/S2 | `hire_limit=2` survived; zero failed validation. An extra RNG draw preserved the returned value but changed post-call state. |
| S3 imports | Moving a bundled import into `agent()` caused failure through the real cached loader; late standard-library import passed. |
| W1 outage | Shared DEBUG mode never invoked W&B initialization. Injected initialization failure propagated; the parent receipt retained source identity and recorded offline status. |

The requested command completed successfully:

`OMP_NUM_THREADS=2 uvx --from rust-just just py-prepare`

- **1,625 passed, 7 skipped**, 46.24 seconds.
- Formatting: 116 files unchanged; lint and Python syntax checks passed.
- Mypy: 63 source files passed; docs freshness passed.
- Skips: native grammar binding, two CUDA pinned-memory checks, two FlashAttention checks, quantized backend and native Kaggriculture environment.
- All five cached source hashes matched.
- Requested prohibited-text grep: **zero matches**.
- The design preserves observation-only stateless inference, uses no v2 model implementation and introduces no second trainer.
- No Docker, training, GPU work, network installs or submission occurred. All scratch files were removed. Initial and final `git status --porcelain` were both empty.

VERDICT: REQUEST CHANGES