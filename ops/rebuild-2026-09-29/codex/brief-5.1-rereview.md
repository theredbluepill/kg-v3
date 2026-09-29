Reviewed `kg/rebuild-bc-brief@bcefd02` read-only. Three edits are needed:

1. **P2 — Correct the normalization-equivalence claim.** [Lines 133–139](/Users/poonszesen/kg-v3-bcbrief/ops/rebuild-2026-09-29/briefs/5.1-bc-data.md:133) normalize only absent/null `hands` and `market`. Reference `prepare.py` uses `raw.get(k) or []`, which also normalizes `false`, `0`, `""`, and `{}`. An in-memory probe against the pinned reference codec admitted all eight field/value combinations with a PASS farmer and no hands. Either reproduce and count that normalization, or explicitly document/test the stricter admission rule. Its incidence in the selected slice remains unknown.

2. **P2 — Identify the actual preparation source.** [Lines 236–239](/Users/poonszesen/kg-v3-bcbrief/ops/rebuild-2026-09-29/briefs/5.1-bc-data.md:236) record HEAD, a dirty flag, locks and extension hash. These cannot identify modified Python preparation, allocator or oracle code. Require an immutable clean checkout for both execution routes, or preserve exact source bytes/hashes; recheck before publishing the manifest.

3. **P3 — Retain a tracked shard custody inventory.** [Lines 248–250](/Users/poonszesen/kg-v3-bcbrief/ops/rebuild-2026-09-29/briefs/5.1-bc-data.md:248) keep the manifest outside git and commit only its hash. Commit the compact per-shard path/bytes/hash inventory, as required by the [custody Decision](/Users/poonszesen/kg-v3-bcbrief/cookbook/decisions/evaluation-preserves-generality-and-evidence.md:28). Keep NPZ bulk external.

The main design checks out: observation/action timing, seat-private header construction, merged encoder API, approved codec signatures, actor/order inputs, paired admission, STOP/padding and shard dtypes. Selection and receipt arithmetic match: **252 episodes, 224/28 split, 158,772 admitted and 22,416 rejected**. The differential oracle should explicitly evaluate both reference seats independently before deriving the historical first rejection.

Compression measurement and data-pod placement are reasonable. Full-slice encoder compatibility, compression, runtime and current artifact availability remain unverified acceptance work. No files changed.

**VERDICT: APPROVE WITH EDITS**