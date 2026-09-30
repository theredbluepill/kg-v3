# Independent merge, documentation and receipt audit

Scope: HEAD `666deec789b2e75f56afb6dbb7b7acd40171c6b1` against three-dot base `ca370893eff25f98a1a00a0bbcde82750eff1c4c`. Original merge: `d793d16f23ee529f9fc49120558c8e9731b843f1`; incoming parent: `24380f3eb0e9c7090d874f8f1a45c0d4e3477f40`. This audit ran no GPU job and no full test suite. Evidence files here are untracked; no tracked edit was made.

## Findings

No actionable merge, documentation or retained-evidence finding.

## Parent preservation and test names

`audit.py` independently builds complete path-to-mode/type/blob maps from Git. The parents' common ancestor is `a78f609e36d40d5254ff909066fe837f2d2540be`. Their changed path sets have no overlap. The original merge tree equals the exact union of their independent changes: **zero unexpected entries**, including file modes. Exactly five commits from the incoming parent are outside the base.

The user's original resolution numbers are correct for `d793d16`: **113 changed files, 16,325 added lines, zero deleted lines**. `results.md` is an existing file with a pure appended addition, rather than a newly added path. HEAD additionally includes the two review-fix commits; its three-dot diff contains **198 files**. All production source, tests, scripts, configs and docs under `src/`, `python/`, `engine_rs/`, `tests/`, `scripts/`, `configs/` and `docs/` are still byte/mode-identical to base. This includes the trim manifest and parity coverage page.

The fresh Python 3.12 AST-qualified/Rust attribute-name inventory covers live `tests/`, `src/` and `engine_rs/`, excluding historical copies under `ops/`. Base, original merge and HEAD each contain **1,257 unique path-qualified declarations (930 Python, 327 Rust)**, with **zero additions/losses between those three trees**. The incoming parent and common ancestor each have 775 declarations (618 Python, 157 Rust). Exactly three incoming names are absent from HEAD, and all three were already replaced in the base before this merge:

- `test_every_api_rejects_hidden_state_and_heads_are_deferred` → `test_every_api_rejects_hidden_state`: heads now exist; the base also tests invalid `dones`.
- `test_kmax_is_max_of_embed_and_mlp_hidden` → `test_trunk_gemm_width_matches_every_block_linear` and `test_preset_trunk_gemm_width`: width is checked against actual compiled block Linear dimensions.
- `test_packed_overflow_boundary_rejects_before_trunk` → `test_packed_path_chunks_at_the_overflow_boundary`, with additional ordered/chunk-equality and unfittable-row checks: current behavior chunks at row boundaries.

These are declarations, not parameterized runtime test counts. The comparison found **zero merge-induced test loss**. Whole source-tree equality independently covers test bodies and parameterizations.

## Receipt custody and numerical reductions

- All **94** retained `pod/attempt*/` files are byte/mode-identical to the incoming parent and original merge. Subsequent fixes are confined to separate current scripts and documentation.
- All **116** file hashes in `MANIFEST.sha256` match, with no unlisted bundle files (excluding Python cache files and the manifest itself). The manifest's comment identifies the deliberately uncopied 4,123,148-byte attempt-2 compile-cache hash list with SHA-256; this audit did not access that remote artifact or bulk caches.
- Both as-run script receipts match: **9/9 hashes per attempt**, 18 total.
- Regenerating `summary.json` into this evidence directory produces byte-identical output, SHA-256 `319020791014d3aede72761f609be1e9ef096548479df12b6f136ae73a42f97b`.
- Independently reduced the retained **24 C4 timing records** (20 samples each): exact median and recorded order-statistic p90, guard chunk counts and FlashAttention flags agree. The six 2/4/8-rank-shape split update-wall/global-SPS/per-rank-SPS calculations agree with the derived JSON.
- All **12 backend records** retain identical start/end settings. Every numeric field in the **21 retained C2 JSONL records** is finite. This checks the recorded evidence, not unretained tensor contents.
- Base `results.md` is an exact prefix of HEAD's file. Original-merge `results.md` equals the incoming parent's version. Six current result headings are unique and ordered; GPU checks (09:15–09:29Z) follow the ATEN A/B (07:06–07:15Z).

## Documentation and prior findings

The requested external r2 report and the committed copy are byte-identical, SHA-256 `8621dc3e975523bbf246e0b0b0e66c720a28f1e2a0651eb7f27505b9571f6c64`.

The r1 documentation findings carried in r2 remain **RESOLVED**:

- Pending-merge provenance: the multi-GPU Decision now cites the merged local receipt bundle, with its source list reconciled; it keeps GPU 0-only timing and component-only throughput limits.
- Missing GPU evidence: the compiled-GEMM Reference credits depth-1 real-trunk ATEN backward above the bound while keeping depth-8 and production backend-claim verification open. The teacher Reference credits C3/C4 GPU evidence and measured component memory while retaining same-batch equality, trained-policy, real multi-rank and full trainer integration limits. The References index follows both revised descriptions.

README, results and the newest cookbook log consistently distinguish as-run artifacts from post-run launcher/C2 fixes. The r2 code findings and mutation results are being verified separately by the parent reviewer.

Direct invocation of the installed cookbook hook on the three modified concept notes returned `{}` with exit 0. Separate checks confirm their first tag and all **73 declared repository sources**. `uv run python scripts/check_doc_freshness.py` passed: `No doc updates required`. Shape/source lint and path existence do not establish truth, remote-source access, live hook discovery or trust; this audit separately checked the substantive GPU claims against retained receipts as described above.

Evidence: `merge.json`, `test-names.json`, `custody.json`, `additional-custody.json`, `results-order.json`, `cookbook-lint.json`, `docs-fresh.log`; `audit.py` is the independent reproducible audit.
