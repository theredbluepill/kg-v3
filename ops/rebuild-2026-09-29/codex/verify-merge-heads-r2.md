Confirmed `e1458d2` on clean branch `kg/merge-heads`. All three r1 findings are resolved:

1. [Architecture wording](/Users/poonszesen/kg-v3-merge2/docs/model-architecture.md:748) and the Python comment now distinguish measured input-side overflow from the strict training design bound justified by backward.
2. The [Reference](/Users/poonszesen/kg-v3-merge2/cookbook/references/compiled-gemm-template-overflows-above-2-21-rows.md:48) correctly describes implemented heads, `actor_input_proj`, and chunking. The documented widths and 11,096-row head limit match the code.
3. Historical rejection is scoped to `1ddc71d` in the Reference, frontmatter and index; current packed chunking is explicitly CPU-tested, with GPU behavior unmeasured.

No new overclaim found; production compliance remains explicitly unproven.

The Python change is comment-only: non-comment tokens and AST are identical to the parent. `git diff --check` passes. No files changed; no test suite or GPU workload run.

**VERDICT: APPROVE**