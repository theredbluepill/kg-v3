Review commit: **`2b62f84b3fe940047bd449a6c3b3ed2aac5b2e95`**, available in this verified [Git bundle](/private/tmp/kg-v3-stream-d-20260929.bundle).

The sandbox blocked this checkout’s Git index. The commit is on `kg/rebuild-codex-data` in a temporary local clone; **this checkout retains the identical, uncommitted files**.

Delivered:

- Typed [selector](/Users/poonszesen/kg-v3-codex-data/scripts/kaggriculture_bc/select_replays.py) and synthetic tests.
- [BC data/preparation brief](/Users/poonszesen/kg-v3-codex-data/ops/rebuild-2026-09-29/briefs/5.1-bc-data.md).
- [Replay-export brief](/Users/poonszesen/kg-v3-codex-data/ops/rebuild-2026-09-29/briefs/7.3-replay-export.md).
- [Opponent brief](/Users/poonszesen/kg-v3-codex-data/ops/rebuild-2026-09-29/briefs/7.1-opponents.md), recommending starter, R04, EcoBot and E776.
- Cookbook records, hashes and [verification receipt](/Users/poonszesen/kg-v3-codex-data/ops/rebuild-2026-09-29/checks/stream-d-results.md).

Checks passed: **17 selector cases; 739 Python tests, 3 skips; 155 Rust tests, 2 ignored**. Both preparation targets pass. Offline inventory comparison matches all **252 historical episode IDs and splits**.

BC data historically lives on volume **`4llk4uaf20`, EU-RO-1**, under `/workspace/kaggriculture-v2/public-episodes-2026-09-14-to-2026-09-27` (`/data/...` on the temporary reader). Transfer to the GPU pod previously succeeded. **Current pod reachability and artifact retention remain unverified**; no network was used.

Native preparation, replay round trips and opponent parity await the engine. No bot or engine code was imported.