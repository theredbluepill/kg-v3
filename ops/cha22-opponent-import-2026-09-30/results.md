# Cha22 opponent import — completion (Track B), 2026-09-30

Owner, verbatim and in order: "OK, for fixed bot, we can use cha22 (check
~/kaggriculture-v2)."; "can we acceleerate this setup?"; "implement the new
rewrad first before we revisit the cha22 anchor setup."; "is anchor thing
ready?". Interpretation (not an owner statement): the anchor setup trains PPO
against a fixed Cha22 seat; this change supplies only the opponent (Track B).
Light parity follows "can we acceleerate this setup?".

## Question, mechanism and stopping condition

Can v3 run the original Cha22 `ig_agent` decisions natively behind the v3
controller view? The mechanism is the v2 Rust port and its dependency closure,
adapted only at the Game-view accessor lines. Discriminating observation:
action-level and transition-level agreement with the original Python on three
default-config games against Starter, Cha22 in both seats. Stop at the first
mismatch and preserve it.

## Review of the stopped agent's WIP (`6653148`)

- **Wrong pin (corrected).** The WIP copied the closure from kaggriculture-v2
  `30a3ac47`. Task 7.1 and `engine_rs/TRIM_MANIFEST.json` use this repository's
  own pinned commit `65f0eac5…`, whose closure is newer: `cha22/mod.rs` gains
  v2's execution-recovery layer (`execution.rs`, `execution_tests.rs`),
  `farm2945/orderpri2.rs` gains `observe_executed` and `v43/mod.rs` gains
  `cancel_terminal_plan`. The recovery layer is inert unless the v2-only
  `drive_executed` injection populates it; the registry never calls it. Now all
  35 closure files come from `65f0eac5`: 27 byte-exact, 8 with only the three
  Game-view accessor lines changed (including `execution_tests.rs`, whose 7
  tests now run). With the pinned blobs in-repo, the default custody check
  needs no sibling repository.
- **Custody incomplete (completed).** The WIP left `OPPONENT_MANIFEST.json`
  unchanged, so `scripts/check_opponent_import.py` (run by `just prepare`)
  failed with 40 undeclared files. The checker now has an `adapted` section
  (re-derives each file from its pinned blob with exactly the three accessor
  substitutions), a `notices` section, the Cha22 Python oracle pin and a second
  oracle directory with its own MANIFEST.
- **No license notices (added).** `opponents_rs/notices/cha22/` holds v2's
  NOTICE.md, UPSTREAM-NOTEBOOK.md and UPSTREAM_README.md byte-exact (v2
  `30a3ac47`) and every comment line of the original main.py
  (UPSTREAM-SOURCE-COMMENTS.txt: the Apache-2.0 text and layer attributions).
- **Test log truncated (rerun).** `wip-opponents-test-truncated.log` stopped
  mid-lifecycle. The full suite now passes: 35 tests (22 unit including 7
  execution tests, 6 lifecycle, 7 oracle) in `opponents-test.log` (before the
  winner assertion) and `prepare.log` (final tree; `just prepare` exit 0,
  root Rust 284 passed with 5 ignored, Python 2,850 passed with 20 skipped).
- **Generator.** `getattr(module, CHA22_ENTRY)` became `module.ig_agent`
  (repo attribute-access rule); Kaggle's last callable `kaggle_agent` is an
  alias of `ig_agent` in the source. Tests added for the specs, the source-hash
  refusal and per-seat module isolation.
- **Lifecycle non-vacuity.** The Cha22 full-match test now requires Cha22 to
  win (seed 17: 129,037 vs 3,699 in both seat orders).

## Parity (independently rerun)

Oracle generation: fresh `uv venv` (CPython 3.11.15, kaggle-environments
1.32.7), original main.py SHA-256 `127ed3e6…` (matches v2's recorded
source-manifest hash), `--preset cha22`. Under `PYTHONHASHSEED` 0, 12345 and
random all three traces and the MANIFEST are byte-identical to the committed
ones (`regeneration.log`). Rust replay (`CHA22_PARITY_REPORT`, `cha22-parity.json`):

| Trace | Seed | Seat 0 | Seat 1 | Actions matched | Transitions |
| --- | ---: | --- | --- | --- | ---: |
| oracle-cha22-00 | 20260937 | cha22 | starter | 719/719, 719/719 | 719 |
| oracle-cha22-01 | 20260938 | starter | cha22 | 719/719, 719/719 | 719 |
| oracle-cha22-02 | 20260939 | cha22 | starter | 719/719, 719/719 | 719 |

4,314 / 4,314 actions (2,157 Cha22) and 2,157 / 2,157 transitions of
public/private state, statuses, rewards and terminal banks. Python terminal
banks: 174,272 vs 3,684; 3,731 vs 179,279; 164,412 vs 3,506. The parity result
is the same with the WIP's `30a3ac47` closure and with the `65f0eac5` closure.
Tampering Cha22's step-399 market fails each trace at that step and seat.
Controller mutation (scratch, restored; `mutation.log`): skipping the DAWN
market layer fails `oracle-cha22-00` at step 434 (native 0 orders, Python 1)
while the other two traces still match, so some layers are exercised by only
one of the three games.
Traces: 564,323 B; with the 7.1 corpus 2,343,510 / 4,000,000 B.

## Large fixtures

All three are byte-exact at `65f0eac5` and in the manifest's `imported`.
`v43-routes.json` (4.9 MB) is read on every Cha22 turn (a scratch probe counted
723,128 loads over four full matches). The two sell libraries (2.7 MB) are
needed only to compile the byte-exact `farm2945/race.rs`: the probe saw zero
library loads, and Cha22's path through Metav4 never calls Farm2945's predict
layer. Removing them would require editing imported decision code.

## Limits

- Light corpus only: no Cha22 mid-episode Python replay, no custom configs, no
  strength claim. v2's fuller checks (5,752 actions, 237 direct cases, 64
  clones) are cited from kaggriculture-v2
  `ops/cha22-opponent-import-2026-09-24/`, not rerun.
- Python's equal-price ADV ordering depends on the hash seed (Rust keeps tape
  order); these games never reached a differing tie. Inactive PIPE opening
  alternatives are not exposed.
- `--original-sources` fails before reaching Cha22 on this Mac because
  `/Users/poonszesen/kaggriculture` is no longer a Git repository (7.1's R04,
  EcoBot and E776 sources; pre-existing). The Cha22 part was checked directly
  (`original-sources-cha22.log`).
- Not ready as an anchor run: no trainer path puts an `opponents_rs` seat
  against the learned policy yet (the learned-seat tests still skip). That is
  the remaining Track A work.

## Receipts

`regeneration.log`, `cha22-parity.json`, `opponents-test.log`,
`original-sources-cha22.log`, `prepare.log`, `mutation.log`,
`update_manifest.py`, `update_trim_manifest.py`; the stopped agent's receipts
are kept as `wip-*`.
