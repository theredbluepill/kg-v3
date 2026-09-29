---
type: "Reference"
title: "Frozen engine API blocks standalone opponent import"
description: "Task 7.1 stops at its explicit visibility boundary: byte-exact controllers require private engine items and accessors excluded with policy_rows; no opponent is qualified."
tags: ["kaggriculture-v3", "adaptation", "opponents", "parity"]
status: "blocked-by-engine-api"
generated: {"by": "openai/codex", "at": "2026-09-29"}
sources:
  - resource: "user-directive:2026-09-29:task-7.1-standalone-crate-stop-on-private-engine-item"
  - resource: "repository:engine_rs/src/lib.rs"
  - resource: "repository:engine_rs/TRIM_MANIFEST.json"
  - resource: "repository:ops/rebuild-2026-09-29/briefs/7.1-opponents.md"
  - resource: "repository:ops/rebuild-2026-09-29/7.1/native-api-probe.json"
  - resource: "repository:ops/rebuild-2026-09-29/7.1/native-api-probe-red.log"
  - resource: "repository:ops/rebuild-2026-09-29/7.1/results.md"
  - resource: "repository:ops/rebuild-2026-09-29/7.1/update_trim_manifest.py"
  - resource: "repository:docs/rules-parity-coverage.md"
  - resource: "reference-branch:65f0eac5bb00b18a9d3acce319c2a231cbd5dff0:engine_rs/src/policy_rows.rs"
---

# Frozen engine API blocks standalone opponent import

The Task 7.1 placement instruction requires a standalone `opponents_rs` crate,
byte-exact controller files and a frozen engine. It explicitly requires stopping
that path if a required engine item is not public. At integration
`b8747b6e8acece5f561d09a75bb914364a60ac05`, an isolated edition-2024 compile
probe fails with `E0603` for private `fib`, `E0616` for private `Game.config`,
and `E0599` for missing `Game::{farms, privates, market, town, step_index}`.
The five accessors existed in the reference's excluded `policy_rows.rs`
(`impl Game`, lines 1105–1122), rather than in the retained rules kernel.

This is a compile-time API boundary finding, independent of controller playing
behaviour. All four controller blobs and E776 policy data in the scratch probe
match the brief's SHA-256s. The compiler checks both library and test targets;
Starter's inline tests cannot run. Other E776 type errors remain unattributed
because missing accessors prevent type resolution; they are not established
independent controller defects. No production opponent crate, root dependency,
engine source change, replacement Game or copied accessor was introduced.

## Consequence and verification boundary

Reopen implementation only with a revised placement/API contract that resolves
these exact items. Neither existing kernel replay parity nor historical native
opponent comments establish this task's action parity. There are zero new
opponent traces, actions compared or coverage observations. Openings, day resets,
weeds, shortages/rejected orders, hires, final-day sales and mid-episode replay
are all uncovered. Lifecycle, visibility perturbation, deterministic match
execution and default-configuration-only qualification remain unimplemented.
Task 1.4 binding-dependent tests also remain unimplemented after this stop.

Original-submission custody is a separate remaining condition. Read-only checks
found all three specified Python entry hashes and E776's 14 manifest-listed
files match sibling commit `e8884aae82eddeb7a1aeae99ecceeca7c830d67e`.
EcoBot and E776 provenance explicitly leave software licenses unresolved and
warn against general-purpose redistribution; R04 has no `PROVENANCE.md` at its
agent path. No Python source is copied. Source availability does not resolve
the license/notice gap or qualify the controllers.

The adaptation consists solely of this note/index/log, the coverage correction,
the compact blocker/check receipts and the idempotent trim-manifest updater.
That updater records only these non-engine changes; retained/authored entries
and all excluded reasons remain unchanged because no production import occurred.
Actual commands, failures and remaining checks are in the result receipt. There
is no green opponent implementation or mutation-based oracle check to claim.

Existing-concept search covered opponent import, native accessors, private
engine fields, parity, custody and negative evidence. The
[[rebuild-data-preparation-preserves-replay-identity|data-preparation Reference]]
records the recommendation, and the
[[live-differential-parity-checks-the-rust-kernel|kernel parity Reference]]
records engine behaviour, but neither identifies this external-crate API gap.
This source audit plus actual compiler failure supports a Reference; it is not
a strength result, general lesson or reason to create a result board.
