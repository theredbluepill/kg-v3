Created both briefs:

- [Task 1.4](/Users/poonszesen/kg-v3-envbriefs/ops/rebuild-2026-09-29/briefs/1.4.md)
- [Task 1.5](/Users/poonszesen/kg-v3-envbriefs/ops/rebuild-2026-09-29/briefs/1.5.md)

They specify:

- Root PyO3 environment with full signatures/stub, 29 observation buffers and six transition outputs.
- Two Python-owned buffer generations with CUDA event fences, because current PPO copies are nonblocking without an explicit reuse fence.
- Ordered validation, seed reservation, decoding, candidate execution and staging before atomic publication.
- Rust reward execution; Python typed configuration and independent oracle.
- `base_seed + rank`, stride `world_size`, checked i64 exhaustion.
- Native grammar/table/codec bindings; adapter and factory in 1.5, trainer storage/copy integration in 3.1.
- A bounded local check and separate Claude-operated two-rank smoke without teacher or BC.

Checks run: `git show/log`, source searches, AST-based schema/signature comparisons, Python-snippet parsing and whitespace checks. All 29 observation layouts match `types.py`; all 35 writable parameters match the stub; shared ABI sections are byte-identical. Git status shows only the two requested files.

No implementation, builds, repository tests, training, commits or pod operations ran.

Claude’s review questions cover constructor refinements, seed range, overflow admission, reward rounding, the Orbit config discriminator default and smoke scope.