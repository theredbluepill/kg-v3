Created [ops/rebuild-2026-09-29/briefs/1.2.md](/Users/poonszesen/kg-v3-grammar/ops/rebuild-2026-09-29/briefs/1.2.md). Only this file changed; no commit.

Read the plan, v4 contract, 1.1/2.1/2.3 briefs, AGENTS/CLAUDE, cookbook context, trim manifest/checker/provenance, kernel and episode fixtures, Isaiah’s RL modules/API docs, and all requested reference grammar, decoder, bridge, codec and test files.

Ran read-only Git/source inspection, gzip/JSON inspection and SHA-256 checks. Each episode contains 719 transitions and reaches 13 actors and ten market orders. Brief structure, links, 78 source citations, whitespace and docs-freshness checks passed. No builds, unit tests, fixture generation or training ran.

Five review questions are recorded with recommendations:

1. Follow v4’s HIRE-limited grammar; preserve both reference decoders’ differing verdicts.
2. Reject nonzero padding, then ignore padding during decoding.
3. Keep length-zero inactive rows outside native step.
4. Replace the binary DFA interface with typed plans and direct mask tables.
5. Compile the grammar under `src/kaggriculture/`, reuse its source in an authored engine test, and defer L4 until the production engine dependency arrives.