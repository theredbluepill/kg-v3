# Independent verification of kg/merge-8rank

Reviewed commit `a3a5cb8aa40d87f0e42cb0d5dc200a73c3ec277f` using the requested three-dot diff against `0b8cf98ef57fc49a329dca4c8368c630586c4752`. The second parent is `7ad45fc6e209e4ad46712644e62fbf3e228ff119`; their merge base is `b51b0c0382a13a72854860ae17d7798f0875e4ee`.

No implementation regression, lost-parent test, or semantic conflict-resolution defect was found. One inherited documentation inconsistency remains in a touched Reference.

## Finding

**P3 — Reconcile the configs Reference with the merged GPU chunk-count evidence.**

`cookbook/references/kaggriculture-configs-follow-isaiahs-scaling-6m-recipe.md:59` categorically says chunking is CPU-tested only and has not been measured on GPU. This is contradicted by `cookbook/references/model-only-sps-ceiling-bounds-per-rank-throughput.md:38`: the GPU teacher proxy at `e1458d2` counted 1/2/3 trunk chunks for sparse/mid/dense 16,384-row workloads. The integration's corrected compiled-GEMM Reference already distinguishes this timing/count evidence from correctness and production qualification (`cookbook/references/compiled-gemm-template-overflows-above-2-21-rows.md:62`).

**Fix:** replace the blanket unmeasured claim with that scoped GPU measurement and link its evidence. Retain the open numerical-correctness comparison and real teacher/production-integration gaps. This wording is unchanged from both parents; it is an inherited inconsistency found by the requested current documentation audit, not a merge-loss regression. No tracked fix was made.

## Parent preservation and resolution

- All 247 code/config/test paths in the parent union resolve to parent blobs or the integration's earlier deletions: 204 match both, 36 integration only, five branch only, and two retired branch paths. Runtime Python/Rust code, engine manifest and lockfiles equal integration.
- Python test definitions: integration 926, branch 859, merge 930. No integration names are missing. The branch trim-inventory test was intentionally renamed for Task 1.3; nine grammar-kernel test names survive at their root integration home. Rust definitions remain 327 (69 engine and 258 root, including four ignored).
- Four affected or glob-driven test modules collect 233 cases at integration versus 250 after merge: 17 added, none removed. Twelve are explicit shape/startup cases, four are existing training guards now applied to the 8-rank YAML, and one is the config loader's new YAML case.
- Cookbook log has 117 unique headings, preserving both parents. The multi-GPU sources are the exact ordered deduplicated union: 11 integration entries plus 13 new branch entries. The integration's allocated/reserved-memory paragraph and the branch's eight-rank section are preserved.
- README, model architecture, Decisions index and plan match independent clean `git merge-file` results byte-for-byte. The plan retains the memory metric, adds world-size-8 seed coverage and Task 6.3b, and correctly keeps full winner_ce recipe adoption separate from a batch-only deviation. References index changes only the intended configs line.
- Two resolution-note accounting details are inaccurate but do not indicate tree defects: the pre-branch integration-only log group contains six entries, not seven; and TRIM_MANIFEST changed on integration since the fork for Task 1.3. HEAD correctly retains that manifest; no regeneration was needed for this config merge.

Detailed receipts: `merge-audit.md`, `parent-inventory.md`, `parent_inventory.json`, `new_collected_cases.json`.

## Checks run on this version

| Command | Result |
|---|---|
| `cargo test --manifest-path engine_rs/Cargo.toml --locked --offline` | 69 passed (41 + 9 + 19); 0 failed |
| `cargo test` | 254 passed, 4 ignored; 0 failed |
| `uv run python scripts/check_engine_trim.py` | OK |
| `OMP_NUM_THREADS=2 uv run pytest tests/kaggriculture tests/owl tests/scripts tests/tools -m 'not slow' -q` | 1,690 passed, 11 skipped; 0 failed; 48.47 seconds |
| `uv run mypy python/owl scripts` | No issues in 63 source files |
| Ruff check and format check for the two changed test files | Pass |
| Python 3.11 syntax check | Pass |
| docs-fresh | Pass; scans working changes, so parent/branch documentation was separately audited |
| Direct cookbook lint on the three changed concept notes | No diagnostics; structural/source syntax check only |
| `git diff --check 0b8cf98...HEAD` | Pass |

The Python suite ran in one process; no resource guard or sharding workaround was needed. The 11 skips are one native grammar binding, two CUDA pin-memory cases, four pending teacher/trainer/run_ppo cases, two FlashAttention/CUDA cases, one unavailable quantized backend, and one native evaluation-env case. No GPU, eight-rank DDP, training, or competitive qualification was performed.

## Mutation checks

Twenty targeted scratch-copy mutations were all detected by the intended tests, with green baselines and green restored runs. They cover:

- per-rank env and minibatch shapes and the fixed teacher chunk setting;
- both world-size divisibility guards, partial optimizer minibatches, and partial teacher chunks;
- cross-rank env/model/optimizer/PPO equality and equality to the scaling_6m workload/optimizer;
- rollout, minibatch, teacher and evaluation row calculations;
- removing the startup workload caller;
- reward-recipe drift and removing the model-factory dispatch.

Every mutated scratch file was restored byte-for-byte and checked against its original SHA-256; tracked originals were never mutated. The four newly collected YAML guard cases exercise existing runtime guards, not new guard implementations. Full mutation commands, changes, failure signatures and hashes are in `mutations/results.json`, `mutations/extended-results.json` and the accompanying logs and coverage matrix.

## Custody

Tracked working-tree and index diffs are empty. Only untracked verification receipts under this directory were added; scratch copies are under `.codex-tmp/`. No implementation or cookbook adaptation was made by this read-only review.

VERDICT: APPROVE WITH EDITS
