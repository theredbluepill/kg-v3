Independent verification of `kg/merge-8rank` at `ca370893eff25f98a1a00a0bbcde82750eff1c4c`, using the three-dot diff from `0b8cf98ef57fc49a329dca4c8368c630586c4752`. Merge `a3a5cb8` has that integration parent and `7ad45fc6e209e4ad46712644e62fbf3e228ff119` as its 8-rank parent. The later tip fixes the r1 documentation finding and preserves its evidence.

No new actionable findings. The merge preserves both parents' intended content, and its resolutions are semantically sound.

The bounded review target was parent preservation, config/plan/cookbook consistency and sensitivity of each added guard or oracle. Inputs were the three parent/tip trees, changed config and tests, retained source-bound evidence, and the requested local check commands. Expected discriminating observations were intact parent inventories, passing unmodified checks, targeted failures under scratch mutations, and byte-exact restoration. All were observed; no GPU or training run was launched.

Prior findings from `/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/codex/verify-merge-8rank-r1.md`:

| Severity and location | Status | Fix verified |
| --- | --- | --- |
| P3 — `cookbook/references/kaggriculture-configs-follow-isaiahs-scaling-6m-recipe.md:59` | RESOLVED | The Reference now credits the e1458d2 GPU teacher-proxy timing and 1/2/3 trunk chunks, while retaining numerical-correctness, real-teacher integration and config-level CPU-only gaps. Its description and cookbook log agree. No further fix needed. |

The prior report contains exactly one finding; none are PARTIAL or UNRESOLVED. Its bytes were left unchanged. See [independent semantic review](docs-audit/report.md).

Parent preservation evidence:

- All 926 integration Python test definitions and 327 Rust definitions survive; the tip has 930 Python definitions and the same Rust inventory. The branch's apparent missing nine Rust names moved earlier from `engine_rs/tests/grammar_kernel.rs` to `src/kaggriculture/grammar_kernel_tests.rs`. The one Python rename records the earlier grammar-bridge retirement; its replacement and a rejection test survive. These integration changes predate this merge.
- Fresh selected collection grows from 233 integration cases to 250 tip cases: 17 additions and zero removals, including config-glob cases induced by the new YAML.
- The five branch config/test files are byte-identical to their branch-parent versions. Production Python/Rust, manifests and lockfiles are unchanged from integration.
- Both parents' cookbook log blocks survive exactly, with no duplicate headings. The multi-GPU Decision has the exact ordered 24-source union, keeps the integration body and its allocated-versus-reserved resource-fit paragraph, and appends the branch's eight-rank section unchanged. Only the intended configs line changes in the References index.
- Repeating the clean three-way merges of README, model architecture, Decisions index and plan produces the current bytes. The plan retains Task 6.3b, the world-size-8 seed case, and the distinct, unadopted c1/c2 recipe options.
- The trim manifest is byte-identical to integration and passes its checker. The statement that neither side touched it should be scoped to this merge's net changes: integration had already updated it after the common fork for Task 1.3. No regeneration was needed here.

See [parent and resolution audit](merge-audit/report.md) and its machine-readable inventories.

Requested checks were freshly executed on this tip:

| Command | Result |
| --- | --- |
| `cargo test --manifest-path engine_rs/Cargo.toml --locked --offline` | 69 passed (41 + 9 + 19), 0 failed |
| `cargo test` | 254 passed, 4 ignored, 0 failed |
| `uv run python scripts/check_engine_trim.py` | Passed |
| `uv run pytest tests/kaggriculture tests/owl tests/scripts tests/tools -m 'not slow' -q` | 1,690 passed, 11 skipped; no sharding or resource-guard workaround needed |
| `uv run mypy python/owl scripts` | Clean, 63 source files |

Read-only ruff lint, formatting check (119 files), Python 3.11 syntax check and docs-fresh also passed. Direct cookbook lint returned no shape/source-syntax findings for the three changed concepts; semantic truth was reviewed separately. Exact commands, exit codes and wall times are in [checks.json](checks.json), with adjacent raw logs. The Python skips and their reasons are preserved in [python.log](python.log).

All **20/20 scratch mutations were detected** by the intended test or explicit-error contract, with no collection/import-only failures. Coverage includes all four new divisibility/whole-chunk guards, derived rank shapes and fixed teacher setting, cross-rank env/model/optimizer/PPO equality, all four eight-rank workload rows, teacher clamp/call counts, the startup caller and the expanded recipe/model cases. Baseline and restored runs each passed 53 tests. All 118 physical scratch files were restored byte-for-byte. See the [mutation-to-oracle map](mutations/README.md), exact replacements and SHA-256 receipts.

All 2,044 tracked worktree files and their index entries match the initial snapshot. Working-tree and staged tracked diffs are empty. Only this untracked verification directory was added; [custody.json](custody.json) records the comparison. No repository adaptation was made, so no tracked cookbook update was required.

Limits remain as documented: these are CPU/config/integration-seam checks, not eight-rank CUDA/DDP execution or end-to-end trainer qualification. Existing native-env, teacher/trainer integration and GPU numerical-correctness gaps remain open. No new blocker was found within this merge's scope.

VERDICT: APPROVE
