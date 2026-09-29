# Task 7.3 oracle results

Final command, with the required offline/thread-limit environment and
`UV_NO_SYNC=1` after the parent's explicit debug extension rebuild:

```text
uv run --offline pytest tests/kaggriculture/test_replay_export_oracles.py -q -s
10 passed in 98.89s (0:01:38), exit 0
```

Full transcript: `oracle-green-final.log`. Test-first and intermediate evidence:
`oracle-red.log` (nine absent-API reds), `oracle-inventory-red.log` (additional
inventory-order absent-API red), `oracle-byte-pointer-red.log` (actual byte
divergence lacked its precise pointer), `oracle-green-attempt1.log` (nine passed,
one ineffective mutation), and `oracle-red-autobuild.log` (uv attempted an
editable build before the Rust implementation existed). The distinction between
planned tests, actual results and the ineffective mutation is retained in
`oracle-expectations.md`. Ruff formatting/checking passed for the final test file.

## Independent fixture round trips

All four compressed fixture hashes matched the brief before replay. Their
Kaggle envelopes are built in tests from their own recorded Python initial
public/private state and 719 recorded actions, public/private successor states,
statuses and rewards, using the real framework envelope layout. They are not
constructed by the native exporter. Native import, seed replay, export and an
additional ordered Python comparison all pass. Each also compares captured
initial and terminal snapshots, 719 bank pairs and all 719 full successor
snapshots.

| Fixture | Transitions | Positive comparison wall seconds | Final raw banks |
| --- | ---: | ---: | --- |
| 95324500 | 719 | 11.368691 | 97126 / 32640 |
| 95901360 | 719 | 14.083801 | 143344 / 151788 |
| 95921764 | 719 | 15.644422 | 7843 / 94230 |
| 95990191 | 719 | 16.688783 | 87792 / 99703 |

Total: **2,876 transitions**. Timings include native verification, full captured
evidence checks and the additional ordered Python comparison, excluding each
mutation run. All run by default; no ignored or `slow` cases in this file.

## Live pinned framework and byte round trip

All four installed source hashes match the archived pins and installed version
is **1.32.7**, superseding the brief's old Mac runtime gap. The one live game is
isolated in a subprocess with a 120-second timeout and its own RSS check below
1 GB. Actual game elapsed time: **0.014333 seconds**, peak process RSS
**197,558,272 bytes**. The helper imports pinned files before starting the game
timer; the timeout bounds the whole subprocess.

Configuration overrides: `episodeSteps=10`, `turnsPerDay=3`,
`maxMarketOrdersPerTurn=4`, `townShopUnlockInterval=1`, `weedSpawnChance=0.2`.
Resolved seed: **1208925819614629174706195 (`2**80 + 19`)**. Nine deterministic
transitions span three day boundaries, produce weeds and three unlocked shops,
and terminate DONE. Other fields are the pinned framework's resolved defaults.
The tape includes a purchase, explicit zero quantity, an omitted quantity and an
empty market entry. Both framework `toJSON()` versus native export and framework
import/native replay/re-export agree. Exact wide seed, source action payload and
native canonical byte round trip pass.

The additional Python comparison only strips arbitrary `info` except `seed`,
per-seat `info`, and `remainingOverageTime`. This is stricter than production,
which also allowlists `configuration.actTimeout`/`runTimeout`: all tested headers
retain those original framework budget values. Pinned `core.py` copies host
metadata in `toJSON()`, consumes agent duration from `remainingOverageTime` in
`__loop_through_interpreter`, and uses act/run timeout for framework execution.
Native replay has no agent execution clock. No game state, action, reward,
status or seed is allowlisted. Shared-field restoration follows the episode
specification and never copies private state.

## Mutation failures

Every following mutation failed with the shown first pointer and transition in
the final green command; tests require those failures.

| Mutation | First divergent JSON pointer | Transition |
| --- | --- | ---: |
| Each of four fixtures: raw final reward and matching public bank replaced by 0.8 | `/steps/719/0/reward` | 718 |
| Live framework seed increased by one | `/steps/3/0/observation/farms/0/tiles/0/1` | 2 |
| Native initial float money 3000.0 spelled as integer 3000, semantic values equal | `/steps/0/0/observation/farms/0/money` | initial (`None`) |
| Independently captured bank increased by one | `/captured/banks/1/0` | 1 |
| Seat 0 private shed copied to seat 1 | `/steps/1/1/observation/private/shed/WHEAT` | 0 |
| Initial private seeds key order reversed | `/steps/0/0/observation/private/seeds` | initial (`None`) |
| Official actor inventory COW/WHEAT key order reversed | `/steps/22/1/observation/private/inventories/3` | 21 |

The earlier SELL-zero deletion mutation was **not killed**: it was an equally
ineffective action tape and therefore a valid alternate replay. Explicit-zero
preservation is asserted against the source tape; the final byte-only mutation
tests canonical byte equality independently. No strength or pod-scale evaluation
result is claimed. Task 1.4 live native seed/reset/terminal capture and evaluation
wiring remain outside this independent oracle lane.
