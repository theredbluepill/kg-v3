---
type: "Reference"
title: "Rebuild data preparation preserves replay identity"
description: "The engine-independent selector reproduces the historical 252-episode split; the refreshed BC data brief binds preparation to the merged encoder and grammar, with only the 1.4 codec binding blocking tokens; replay/opponent briefs bound the remaining evaluation work."
tags: ["kaggriculture-v3", "adaptation", "replays", "evaluation"]
status: "verified-scoped"
generated: {"by": "openai/codex", "at": "2026-09-29"}
sources:
  - resource: "repository:scripts/kaggriculture_bc/select_replays.py"
  - resource: "repository:tests/scripts/test_kaggriculture_select_replays.py"
  - resource: "repository:ops/rebuild-2026-09-29/briefs/5.1-bc-data.md"
  - resource: "repository:ops/rebuild-2026-09-29/briefs/7.1-opponents.md"
  - resource: "repository:ops/rebuild-2026-09-29/briefs/7.3-replay-export.md"
  - resource: "repository:ops/rebuild-2026-09-29/checks/stream-d-source-audit.json"
  - resource: "repository:ops/rebuild-2026-09-29/checks/stream-d-results.md"
  - resource: "repository:ops/rebuild-2026-09-29/checks/5.1-brief-refresh/ref-admission-probe.log"
  - resource: "repository:ops/rebuild-2026-09-29/codex/brief-5.1-rereview.md"
  - resource: "repository:src/kaggriculture/mod.rs"
  - resource: "repository:src/kaggriculture/grammar.rs"
  - resource: "repository:docs/kaggriculture-contract.md"
  - resource: "reference-branch:kg/reference-2026-09-29/ops/bc-bootstrap-2026-09-29/select_replays.py"
  - resource: "reference-branch:kg/reference-2026-09-29/ops/bc-bootstrap-2026-09-29/prepare.py"
  - resource: "reference-branch:kg/reference-2026-09-29/ops/bc-bootstrap-2026-09-29/data-manifest.json"
  - resource: "reference-branch:kg/reference-2026-09-29/python/owl/kaggriculture/actor_codec.py"
  - resource: "external-repository:/Users/poonszesen/kaggriculture-v2/ops/kaggle-public-episodes-2026-09-28/manifest.json"
---

# Rebuild data preparation preserves replay identity

The [[../decisions/restart-the-port-from-isaiahs-clean-base|restart Decision]]
classifies replay selection as reviewed porting and preparation as a rebuild.
Searches for replay selection, split leakage, source volume, seed export and
native opponents found the historical [[bc-bootstrap-uses-native-replay-features-and-current-heads|BC]]
and [[shared-ppo-adapts-game-batches-without-a-second-loop|PPO]] References, not a
current-tree data pipeline. This record covers stream D's bounded preparation
work; it does not promote the old model, admission run or evaluation results.

## Adaptation inventory and consequence

- `scripts/kaggriculture_bc/select_replays.py` ports the standard-library
  selection recipe with types and an explicit source-root argument. It retains
  sorted JSON names, `20260929 + day` RNG seeds, September 21–27, first four
  validation and next 32 train episodes per day, payload hashes and manifest
  fields. Episode basename duplication is rejected before export; this guard
  previously lived in preparation. Short days and duplicate ZIP entries now
  fail explicitly. Existing outputs are not replaced. No v2 implementation
  names, absolute legacy paths or engine imports remain in the selector.
- `tests/scripts/test_kaggriculture_select_replays.py` supplies synthetic ZIPs
  and manifests, golden shuffled IDs, leakage/count/hash checks, isolated CLI
  execution, no-overwrite and interrupted-copy cleanup. Tests require no corpus
  or native engine. Future selection changes must retain the audited identity
  or explain the resulting dataset change.
- `ops/rebuild-2026-09-29/briefs/5.1-bc-data.md` records source volume custody
  and the preparation data flow against the merged APIs (Claude refresh at
  `2390c8e`). Explicit per-turn `TraceHeader`s are encoded for both seats
  through the merged `owl.rs.encode_kaggriculture_headers_into`. Recorded
  actions are admitted per seat through 1.4's planned `kaggriculture_encode`
  and `kaggriculture_decode`, which are the only blocked step. The brief also
  sets shard schema `kaggriculture-bc-shard-v1`, a custody manifest and
  data-pod placement. Three findings bind future preparation. First, the Rust
  `Config` is `serde(default)`, so the full configuration key set must be
  checked before encoding. Second, the reference's `raw.get(k) or []`
  silently normalized every falsy `hands`/`market` value (absent, `null`,
  `false`, zero, `""`, `{}`) to `[]`, while the grammar requires the exact
  keys, so that rule is replicated and counted by kind; Codex's rereview
  corrected the earlier absent-or-null-only claim with an in-memory probe of
  the pinned reference codec. Third, the new grammar checks in a different
  order, which can move first-rejection categories but not admission. The
  old flat features/context arrays cannot be loaded as the new schema. After
  that rereview, a run must come from a clean immutable checkout with
  per-file source hashes rechecked before the manifest is published, and a
  compact per-shard path/bytes/SHA-256 inventory is committed while NPZ bulk
  stays external.
- `ops/rebuild-2026-09-29/briefs/7.3-replay-export.md` distinguishes Kaggle
  episode JSON from native JSONL oracles. Export must retain the resolved seed,
  full config, action timing/order and completed state before auto-reset; seed
  replay must not load expected snapshots or diagnostic RNG schedules as truth.
- `ops/rebuild-2026-09-29/briefs/7.1-opponents.md` recommends starter, R04,
  EcoBot and E776 by source behavior and dependency footprint, with hashes and
  explicit lifecycle/parity gaps. The four source modules plus E776's policy
  tape total 335,571 bytes before shared support. This is a recommendation,
  not an adopted or qualified panel; no bot code is imported.
- The companion source audit, fixture custody and result receipt under
  `ops/rebuild-2026-09-29/checks/stream-d-*` bind checks to source and disclose
  missing live evidence. The historical BC note corrects the false claim that
  omitted units become NONE: preparation inserts `None`, which the codec rejects.
- `docs/kaggriculture-contract.md` gains the missing blank line before its token
  list so repository Markdown lint passes. No v4 contract semantics change.

## Independent checks and limits

The synced source corpus inventory, separate from the reference selection
receipt, reproduces all **252 ordered episode IDs, split assignments and byte
lengths** through the ported selector: 224 train / 28 validation. No raw payload
was opened or rehashed. The receipt's 158,772 admitted / 22,416 rejected
paired turns are historical arithmetic, not new admission evidence.

The source is network volume **4llk4uaf20, EU-RO-1**. A historical reader mounted
it at `/data`, copied the selected ZIP to the GPU pod, then terminated. The
original volume path was under `/workspace`; the prepared GPU arrays were at
`/workspace/kg-v3/replays/bc-bootstrap/arrays`. Today's pod reachability and
artifact retention remain unverified under the no-network task scope. These
paths and the prior ZIP hash guide custody recovery, not an assumption of a live
mount.

Four pinned native fixtures were structurally inspected: **2,876 transitions**,
with consecutive steps and matching final banks/statuses. Archived official
engine source hashes match their engine pin. This is not engine execution or an
export/import round trip. Opponent source sizes/dependencies and policy JSON were
inspected, without playing games or inheriting strength claims.

The final `py-prepare` and `prepare` each pass **739 Python tests / 3 backend
skips**, including **17 selector cases**; `prepare` also passes **155 Rust tests /
2 ignored**. The result receipt retains commands, logs and the initial Markdown
failure. Rust checks concern the retained Orbit starter, not the absent
Kaggriculture engine.
The refreshed brief's per-turn differential oracle loads the stdlib-only
reference `actor_codec.py` by path. Run on one local episode outside the slice
(114406062), it admitted 719 of 719 paired turns in 0.2 s
(`ops/rebuild-2026-09-29/checks/5.1-brief-refresh/ref-admission-probe.log`).
That shows the oracle is cheap and runnable. It is not evidence about the
slice.
Preparation, native replay export, opponent import and full-payload admission
remain deferred until their native seams exist. Reopen when the engine lands or
custody changes; qualify these interfaces before BC or panel results can claim
current-tree parity.
