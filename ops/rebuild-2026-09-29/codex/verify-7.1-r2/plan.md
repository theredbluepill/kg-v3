# Independent Task 7.1 verification r2

Target: review kg/rebuild-7-1 at 908c73fce0be298f81e229b08bf0ca082ad2e075 against b8747b6e8acece5f561d09a75bb914364a60ac05...HEAD. Stop after source review, requested checks, original-oracle reproduction, effective mutation detection and tracked-byte preservation checks; no production edits.

Inputs: Task 7.1 brief; prior r1 findings at /Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/codex/verify-7.1-r1.md; cookbook index/log and opponent Reference; four pinned native controllers, their dedicated manifest, eight default-game original-Python traces and 24 replay cases.

Questions and discriminating checks:

- Does the new replay close prior P2? Inspect generation of fresh original controllers and native reconstruction. Reproduce replay from pinned originals with CPython 3.11.15; expect all bytes equal and 1,152 resumed actions matched.
- Are all eight continuous oracles and four bots tied to their original behavior? Regenerate at most two games per invocation, with pinned source hashes; expect the eight gzip files and manifest byte-identical and native 11,504 actions/5,752 transitions matched.
- Can comparisons reject altered expected/actual actions? Mutate each continuous fixture independently at step 37; mutate every replay case five steps into resume (combined batch with all 24 mismatch labels required); mutate each native controller at step 700 independently. Expect semantic parity failures, not compile failures. Restore every scratch input byte-for-byte in finally blocks; finish with full restored opponent suite.
- Is the broader checkout healthy? Engine locked/offline Cargo, root Cargo, opponent Cargo, relevant pytest, mypy, engine trim and original-source custody. Existing Task 1.4 skips remain explicit.
- Does shortage language reflect actual engine semantics? Read scarcity pricing and BuyProduct execution, then count BUY_PRODUCT orders with predecision inventory<I0 in the existing frozen corpus. This is a census, no new blind seed sweep.

Receipts are local untracked files here. Mutation copies and build output stay in .codex-tmp. The original r1 report is preserved; r2 report carries a status for every prior finding. Independent read-only subagents review the oracle and boundary/custody code while root owns execution and final verdict.
