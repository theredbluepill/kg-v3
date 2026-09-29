# Vendored engine provenance

This crate was copied from the current working tree at
`/Users/poonszesen/kaggriculture/engine_rs` on 2026-09-20. The source tree was
based on sibling commit `e8884aae82eddeb7a1aeae99ecceeca7c830d67e` and had
material tracked and untracked changes, including the post-V43 native agents.
It was therefore copied by bytes rather than reconstructed from Git.

The tree was copied twice on that date. The second copy, taken after the
sibling added the Metav4 and Pipe16 native controllers, contained 102
non-build files and 18,756,097 bytes. The copied `Cargo.toml`, `Cargo.lock`,
and `src/**/*.rs` source digest under the shared adapter’s length-delimited hashing contract
is `172f042b119fe84b9b984dcc0a5cdbd95b2a67969f398c20aa5222405d51e6ae`.
Receipts that record `49cfdd3ac5b5696165988f4b355010c42ebb8bdde55786b0bac84f70576e4019`
refer to the first copy (94 files), which v2 commit `5a5c0de6` preserves.

The crate targets `kaggle-environments==1.32.7` and Python game source SHA-256
`bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e`.

Generated `target/` output and `.omc/` runtime state were deliberately not
copied. `scripts/re_batch.py` is vendored at the v2 repository root for the
ctypes ABI. Four source test fixtures remain at their original repository-
relative `data/intel/...` and `tests/fixtures/...` paths so the Rust source
bytes stay unchanged.

## Incremental opponent import — 2026-09-24

V56 and its required shared controller/FFI changes were copied byte-exact from
the sibling working tree. The module registry additionally exports the local
JSONL adapter. Eight earlier controllers gained locally authored
JSONL entry points and a shared observation adapter. This is an incremental
import, not a replacement of the game-transition implementation.

That V56 import’s `Cargo.toml`, `Cargo.lock`, `src/**/*.rs` digest:
`c02e7207010f4648250f13a472cdd9704b262235e3840219ff3065e7165b9c9d`.
The opponent registry separately pins its nine embedded runtime data fixtures:
`333e76be89fdf232d14ec38585d09c5ff391e09fd5455bf9c1588f2c97f7b881`.

Exact imported/local paths, prior source bytes, source/data hashes, binary hashes
and current build/test receipts live in `ops/opponent-import-2026-09-24/`.
The original copy counts and digests above remain historical. Fifteen retained
engine fixtures match the sibling bytes. The shared batch bridge is unchanged.

## Cha22 incremental import — 2026-09-24

Cha22’s four controller files, JSONL entry, oracle and12 required shared files
were copied byte-exact from the verified sibling working tree. Native enum and
FFI slots were patched locally, retaining v2’s JSONL adapter. Current source
digest: `8001a3e0b734f479ccd057f3c20498f7de91e5ef1007c164384b0e7b3f21b565`.
Embedded data remains `333e76be89fdf232d14ec38585d09c5ff391e09fd5455bf9c1588f2c97f7b881`.

Custody, preserved prior files, build/tests and exact replay receipts:
`ops/cha22-opponent-import-2026-09-24/`. All19 retained opponents preserve
27,322 recorded actions; Cha22 matches5,752 source actions,237 direct cases
and64 clones. These are bounded execution checks; no strength result follows.

## Task 1.1 rules-only trim — 2026-09-29

The sections above describe historical full imports, not this trimmed package.
This copy is pinned to kg/reference-2026-09-29 at
65f0eac5bb00b18a9d3acce319c2a231cbd5dff0. TRIM_MANIFEST.json accounts for every
reference path, source hash, permitted line edit and excluded component.
The only rules-source edits remove lib.rs reference lines 19, 21–25 and 27:
ffi, joint_matching, myolie_features, myolie_sampler, native_agents,
policy_rows and training declarations. py_random.rs, econ_attrib.rs and the
RNG integration tests are unchanged. Cargo drops the cdylib/binary targets
and unused Rayon dependency; Cargo regenerates the lockfile. This provenance
note is append-only. No bot, v2 collector, trainer, model or C ABI is imported.
Four episode traces remain as rules oracles; the new replay integration test
starts from configuration and seed. Historic opponent/ABI/performance claims
above do not qualify this trim. See ../docs/rules-parity-coverage.md for current
checks and limits, and ../ops/rebuild-2026-09-29/briefs/1.1-rules-kernel.md
for the reviewed import plan. The root crate does not yet depend on this package.
