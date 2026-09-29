# Task 7.3 independent oracle checks

Target: distinguish agreement with independent Python engine state from circular
native export/import agreement. Stop once four full pinned fixture replays, one
tiny live framework game, the canonical-byte round trip and captured-evidence
checks pass with their respective mutation failures.

Expected checks (not results):

- Reconstruct each official fixture's Kaggle envelope from its own recorded
  Python header/transition data, using the real framework's envelope layout.
  Compare native seed replay with all 719 full post-transition snapshots plus
  initial/terminal snapshots and every pair of raw banks. Require all values and
  key ordering. Mutate final raw reward into 0.8 to demonstrate a failure.
- Run one real pinned framework game: `episodeSteps=10`, `turnsPerDay=3`,
  `maxMarketOrdersPerTurn=4`, `townShopUnlockInterval=1`, `weedSpawnChance=0.2`,
  resolved seed `1208925819614629174706195` (`2**80+19`). Nine scripted steps
  include bought inventory, explicit zero, omitted quantity and an empty market
  entry. Require three day boundaries, weed rolls, three shop unlocks and DONE.
  Compare its own `toJSON()` with native export and native import/re-export.
  Change the seed to prove the oracle comparison is live.
- Require native export/import/re-export byte equality and exact wide-seed
  survival; remove a submitted explicit zero to prove a changed tape fails.
- Change independently captured transition banks to prove that comparator is
  not only replay versus itself. Remove specification-shared fields to test
  reconstruction; substitute the other seat's private state and reverse private
  map keys to demonstrate privacy and insertion-order checks.

The tests do not use the production framework loader to form expected envelopes.
They independently verify installed source bytes before importing the framework.
They use no model, panel, trainer or live native environment binding.

This test's independent Python comparison allows only top-level `info` except
`seed`, per-state `info`, and observation `remainingOverageTime`: pinned `core.py`
`toJSON()` copies
arbitrary caller `env.info`/seat-state metadata, while
`__loop_through_interpreter` adjusts time budget from agent-log durations.
Native production provenance therefore differs and native replay has no measured
agent clock. No action, configuration, game observation, reward or status is
allowlisted in this additional comparison. Production also allows
`configuration.actTimeout` and `configuration.runTimeout`, framework execution
budgets used by `core.py` `__loop_through_interpreter`/`run` rather than native
game transitions. The test headers preserve their source values, so the stricter
comparison can require equality. Shared-field restoration is dictated by
`specification.observation` and is not a license to ignore a differing present
field.

# Initial actual observations and reds

- Local package version is `1.32.7`; all four hashes below match the brief.
  The brief's statement that this Mac has 1.29.0 without Kaggriculture is stale.
- Four fixtures each contain one header and 719 transitions and match the brief's
  compressed-byte hashes below.
- Real `toJSON()` duplicates farms/market/town/day/hour into seat 1 even though
  marked shared: the Kaggriculture interpreter writes those fields to both
  seats. Only shared `step` is omitted there. Seat 0 observation order is
  remainingOverageTime, step, player, farms, private, market, town, day, hour;
  seat 1 is remainingOverageTime, player, private, farms, market, town, day, hour.
  State keys are action, reward, info, observation, status. Private keys are
  shed, seeds, inventories. Initial action is canonical PASS, reward 0.
- Initial `uv run --offline pytest` attempted uv's automatic editable release
  rebuild and failed because the test-first Rust module had not been implemented
  (`oracle-red-autobuild.log`). No extension rebuild was explicitly requested in
  this lane. Subsequent focused commands use `--no-sync` until root's debug
  rebuild. A targeted stop attempt after that completed failure was unavailable
  because the sandbox does not permit process listing.
- `uv run --offline --no-sync pytest tests/kaggriculture/test_replay_export_oracles.py
  -q -s`: nine test cases failed on missing `owl.rs` exporter/verifier symbols,
  after independent fixture conversion and framework setup succeeded
  (`oracle-red.log`). The one live fixture game took 0.016589 seconds with
  process peak RSS 423,510,016 bytes in this red run.

| Installed package file | SHA-256 |
|---|---|
| core.py | 0922c4599a1b6e0d8c3dadf06ae5297f98138d859d686f00daee6f36e6d45d0e |
| utils.py | 537b627b11784d424147ef57ebb0369b039bf83c9f891e81f10486b1f552334b |
| envs/kaggriculture/kaggriculture.py | bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e |
| envs/kaggriculture/kaggriculture.json | a82c89c1a2315b93f39775d8e025471a01b738647c9772658368ee6b1b6f4867 |

| Fixture | SHA-256 |
|---|---|
| episode-95324500.jsonl.gz | 47cdfa489b7a80edf8ec1361f2f55cd033c75824d624cd7c9e9daaa3137affd7 |
| episode-95901360.jsonl.gz | e80653f445570a3778a3fb9026a66614b1ecaa2ea417cf8358d3f1850d725281 |
| episode-95921764.jsonl.gz | bc3e01cd12ff70fd2f78bfbe7129d5caca468a6324c124c25c9efebc86fbd3f2 |
| episode-95990191.jsonl.gz | 4bf1a3b09c644719c8b36a619289844e52c3458d0e25dc0a0a1429bc6eaa0d1b |

# First implementation run and corrective evidence

`oracle-green-attempt1.log`: **9 passed, 1 failed in 132.77 seconds**. All four
fixture comparisons passed over all public/private payloads and their key order,
status, raw reward, initial/terminal snapshots, 719 captured bank pairs and 719
captured successor snapshots per fixture. Timing including independent evidence
comparison and Python canonical-output comparison:

| Fixture | Full positive comparison seconds | Raw-reward mutation |
|---|---:|---|
| 95324500 | 16.355816 | `/steps/719/0/reward`, transition 718 |
| 95901360 | 23.847805 | `/steps/719/0/reward`, transition 718 |
| 95921764 | 17.268194 | `/steps/719/0/reward`, transition 718 |
| 95990191 | 23.894932 | `/steps/719/0/reward`, transition 718 |

Each raw-reward mutation also changes the corresponding top-level reward and
terminal farm money in both public seat views. The mutated envelope is internally
consistent, so native replay comparison, not import bank/reward admission, is
what rejects it.

Other successful mutation detections in that run:

- Framework seed changed by one: `/steps/3/0/observation/farms/0/tiles/0/1`,
  transition 2 (weed state differs).
- Captured bank changed by one: `/captured/banks/1/0`, transition 1.
- Seat 0 private shed copied to seat 1:
  `/steps/1/1/observation/private/shed/WHEAT`, transition 0.
- Initial private seeds map order reversed:
  `/steps/0/0/observation/private/seeds`, initial state (`transition None`).
- Official actor inventory key order reversed:
  `/steps/22/1/observation/private/inventories/3`, transition 21. This additional
  test's missing-API red is retained as `oracle-inventory-red.log`.

The failed test was an ineffective mutation: deleting explicit zero from a SELL
command did not change this tape's game outcome and the alternative tape still
formed a valid replay. No native comparison can reconstruct the original
ineffective command without independent tape custody. The production exporter
does preserve the original explicit zero and empty entry, now asserted directly
against the source tape. This failed mutation is not credited as killed.

The byte-only mutation instead replaces initial float64 money `3000.0` with the
semantically equal integer spelling `3000`. It exposed a real diagnostic gap:
the byte comparison rejected it but gave only an empty root pointer, rather than
the first numeric-spelling difference. `oracle-byte-pointer-red.log` records that
red; the required pointer is `/steps/0/0/observation/farms/0/money`, initial state.

The single live framework game is now run in a subprocess with a 120-second
timeout; its own peak RSS is checked below 1 GB, independently of allocations in
earlier pytest cases. The child imports the same test helper, with no pytest
conftest imports and no extra live game. The byte-pointer red measured
**0.014567 seconds / 198,279,168 bytes**. The full framework game remains nine
transitions at the wide seed above and requires no `slow` marker.
