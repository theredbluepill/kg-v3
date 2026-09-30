---
type: "Reference"
title: "Cha22 opponent imports a view-adapted closure with light parity"
description: "opponents_rs registers cha22, the full ig_agent port with its V43/V47/V48/Farm2945/Metav4/Pipe16 closure from pin 65f0eac5 (27 byte-exact files, 8 changed only at three Game-view accessor lines, 3 byte-exact fixtures) and Apache-2.0 notices. On three CPython 3.11 oracle games against Starter it matches 4,314/4,314 original-Python actions and 2,157 transitions. Merged into kg/rebuild-opponent-mix, env.opponent_mix hosts it against the learned seat, and the anchor presets configs/kaggriculture_{4,2}rank_vs_cha22.yaml train against it; nothing trained."
tags: ["kaggriculture-v3", "adaptation", "opponents", "parity", "cha22"]
status: "implemented"
generated: {"by": "anthropic/claude", "at": "2026-09-30"}
sources:
  - resource: "repository:opponents_rs/src/native_agents.rs"
  - resource: "repository:opponents_rs/src/native_agents/cha22/mod.rs"
  - resource: "repository:opponents_rs/src/registry.rs"
  - resource: "repository:opponents_rs/src/lib.rs"
  - resource: "repository:opponents_rs/src/view_tests.rs"
  - resource: "repository:opponents_rs/tests/lifecycle.rs"
  - resource: "repository:opponents_rs/tests/oracle_parity.rs"
  - resource: "repository:opponents_rs/OPPONENT_MANIFEST.json"
  - resource: "repository:opponents_rs/README.md"
  - resource: "repository:opponents_rs/fixtures/oracle-cha22/MANIFEST.json"
  - resource: "repository:opponents_rs/notices/cha22/NOTICE.md"
  - resource: "repository:opponents_rs/notices/cha22/UPSTREAM-SOURCE-COMMENTS.txt"
  - resource: "repository:scripts/check_opponent_import.py"
  - resource: "repository:tests/tools/test_check_opponent_import.py"
  - resource: "repository:scripts/kaggriculture_parity/generate_traces.py"
  - resource: "repository:tests/scripts/test_kaggriculture_parity.py"
  - resource: "repository:tests/owl/kaggriculture/test_opponents.py"
  - resource: "repository:engine_rs/TRIM_MANIFEST.json"
  - resource: "repository:docs/rules-parity-coverage.md"
  - resource: "repository:ops/cha22-opponent-import-2026-09-30/results.md"
  - resource: "repository:ops/cha22-opponent-import-2026-09-30/regeneration.log"
  - resource: "repository:ops/cha22-opponent-import-2026-09-30/cha22-parity.json"
  - resource: "repository:ops/opponent-mix/review-r1.md"
  - resource: "repository:ops/cha22-opponent-import-2026-09-30/opponents-test.log"
  - resource: "repository:ops/cha22-opponent-import-2026-09-30/original-sources-cha22.log"
  - resource: "repository:ops/cha22-opponent-import-2026-09-30/mutation.log"
  - resource: "repository:ops/cha22-opponent-import-2026-09-30/prepare.log"
  - resource: "repository:opponents_rs/src/hosted.rs"
  - resource: "repository:opponents_rs/tests/hosted.rs"
  - resource: "repository:configs/kaggriculture_4rank_vs_cha22.yaml"
  - resource: "repository:tests/scripts/test_run_ppo.py"
  - resource: "external-repository:/Users/poonszesen/kaggriculture-v2/ops/cha22-opponent-import-2026-09-24/README.md"
  - resource: "external-repository:/Users/poonszesen/kaggriculture-v2/opponents/cha22/README.md"
---

# Cha22 opponent imports a view-adapted closure with light parity

The owner chose Cha22 as the fixed opponent: "OK, for fixed bot, we can use
cha22 (check ~/kaggriculture-v2)." and then asked "can we acceleerate this
setup?". `opponents_rs` now registers `cha22` beside the four Task 7.1 bots
([[snapshot-view-isolates-byte-exact-evaluation-opponents|snapshot view]]),
behind the same v3 controller view and seat lifecycle. As with those bots, its
identity and internal memory stay in evaluator bookkeeping. They never reach
the learned actor/critic inputs, rewards, normalization or checkpoint selection
([[../decisions/the-policy-is-stateless-and-observation-only|stateless policy]]).

## Import and custody

The port follows the original submission's full `ig_agent` entry (SHA-256
`127ed3e6…`; `kaggle_agent` aliases it) through Metav4, Farm2945, V47, V43, V48
and Pipe16. All 35 closure files come from the repository's own pin
`65f0eac5bb00b18a9d3acce319c2a231cbd5dff0`. 27 are byte-exact. 8 change only
the three Game-view accessor lines: a borrowed public snapshot, `privates()`,
and a new `Game::configuration()`. The three fixtures the files embed are
byte-exact. The checker's `adapted` section re-derives each adapted file from
its pinned blob, so a rehashed decision edit still fails. `engine_rs/TRIM_MANIFEST.json`
points each excluded reference path at its copy.

A stopped agent's WIP (`6653148`) had copied the older kaggriculture-v2
`30a3ac47` closure and left the manifest stale, so `just prepare` failed its
custody check. The pinned closure adds v2's execution-recovery layer, which
stays inert unless v2's `drive_executed` injection fills it. Parity is identical
with both closures. Only `v43-routes.json` (4.9 MB) is read at runtime. The two
sell libraries (2.7 MB) only let the byte-exact `farm2945/race.rs` compile; a
load probe saw no read over four full matches.

Cha22 is Apache-2.0. `opponents_rs/notices/cha22/` keeps v2's notice files
byte-exact and every comment line of the original main.py, which carries the
license text and the attributions of its layers. The original source is not
copied here. The byte-pinned Rust headers cite `agents/cha22/main.py`, a v2
working name present in neither repository; `opponents_rs/README.md` maps it
to the original's SHA-256 (`127ed3e6…`). The license text exists only as those
comment lines, so redistribution would first need the plain text and a change
statement (review r1 P3-3, `ops/opponent-mix/review-r1.md`).

## Verification

Parity is light by design. Three default-config games against Starter (seeds
20260937–20260939; Cha22 in seat 0 twice and seat 1 once) were generated from
the original submission on Kaggle 1.32.7 under CPython 3.11.15. Native replay
matches 4,314 / 4,314 actions (2,157 Cha22). All 2,157 transitions agree on
public/private state, statuses, rewards and terminal banks. Claude regenerated
the traces in a fresh venv under three hash seeds and got byte-identical files.
A tampered Cha22 action fails each trace. Skipping Cha22's DAWN market layer
in a scratch copy (restored afterwards) fails the oracle test, but only in one
of the three games, at step 434 (`mutation.log`). The light corpus therefore
exercises some layers in only one game. Custody,
generator and lifecycle tests were added, and Cha22 beats Starter in a full
native match in both seats. `just prepare` passes (`prepare.log`).

## Limits and reopening

- Not rerun here: Cha22 mid-episode Python replay, custom configurations and
  playing strength. v2's fuller import checks are cited, not repeated.
- Python orders equal-price ADV candidates by hash seed, while Rust keeps tape
  order. These three games never reached a differing tie.
- The inactive PIPE opening alternatives are not exposed.
- The full `--original-sources` mode now fails on this Mac because the 7.1
  sibling is no longer a Git repository. The Cha22 part was checked directly.
- Reopen parity if a run meets a hash-seed tie or a state far from these games.

The import branch itself had no trainer seat. Merged into
`kg/rebuild-opponent-mix`, the bot-agnostic `env.opponent_mix`
([[../decisions/train-ppo-against-a-fixed-opponent-with-a-learner-mask|fixed-opponent Decision]])
hosts Cha22 through `HostedSeat` against the learned seat, still through
`scripts/run_ppo.py` alone. The hosted view carries the configuration Cha22
reads. `opponents_rs/tests/hosted.rs` reproduces a Cha22 mirror `play_match`,
and the native kernel-reference test replays Cha22 in both seats. The anchor
presets `configs/kaggriculture_{4,2}rank_vs_cha22.yaml` train against it under
term M. Nothing has been trained with it, and its stepping throughput is
unmeasured. Existing-concept search covered opponents, imports, parity,
custody, notices and anchors. This is a new opponent claim, so the 7.1
Reference stays as it is. No board is warranted.
