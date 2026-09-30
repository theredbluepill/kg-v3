# Landing digests: stagger + credit onto the critic-offset integration

Pre-landing tree `78dfd1f` (critic offset landed) versus the landed tree
(merge `a807752` plus the preset follow-up). Same Mac, same `rs.abi3.so`
(no Rust change on either side), torch 2.9.0, Python 3.12,
`OMP_NUM_THREADS=2`. Scripts: `ops/stagger-credit-2026-09-30/baseline_digest.py`
and `config_digest.py`, run from each tree's root.

| Check | `78dfd1f` | Landed | Result |
| --- | --- | --- | --- |
| Native env digest | `257eae38864aa2aa26373c7751be97b3b0d3c6d9d81ee80782036df41159a590` | same | equal |
| Trainer digest (2 updates, default off) | `3ffd53a026b8b3be079c64b111226a6cb313dbdad586c81fbd28c974bd322fa2` | same | equal |
| `config_sha256` of every preset | 43 presets | the same 43 values plus the two credit presets (`landing-config-digest.json`) | equal on every shared preset |

The native and trainer digests also equal the stagger branch's base `3e89425`
receipt (`ops/stagger-credit-2026-09-30/digests.md`), so the merged default
path (stagger off, critic offset off) computes what `3e89425` computed.
Mutation receipt for the new offset-bootstrap test: `landing-mutation.log`.
