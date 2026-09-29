# Evidence custody sweep — 2026-09-29

This is a working record of which local rebuild evidence was committed and which stays outside git. The machine-readable manifest is [`evidence-custody.json`](evidence-custody.json). It lists every MANIFEST and DEFER entry with its path, bytes, SHA-256, location and reason, and every EXCLUDE entry with its path and reason only.

- **Branch:** `kg/merge-custody`, worktree `/Users/poonszesen/kg-v3-m-custody`.
- **BASE:** `kg/isaiah-gap-closure` at `666deec`, after `kg/merge-gpu-receipts` landed.
- **Inventory:** `/Users/poonszesen/kg-v3-m-custody-inventory/inventory.json`, SHA-256 `a53fb12b…79ac`, generated at 2026-09-29T13:48:20Z against integration `ca37089`. It covers 1,916 entries from ten sources (the main checkout and nine worktrees). The scripts that produced it are in `/Users/poonszesen/kg-v3-m-custody-inventory/tools/`. Neither the inventory nor its scripts are committed.
- **Sources were only read.** Nothing in the main checkout or any other worktree was modified, moved or deleted, and the main checkout's uncommitted cookbook edits are still in place.

## What was committed

| Item | Files | Bytes |
|---|---:|---:|
| COMMIT-class copies to the same repo-relative path | 1,537 | 27,521,530 |
| DEFER entries promoted because gpu-receipts has landed | 9 | 830,834 |
| Architecture PNG (MANIFEST override, see below) | 1 | 1,361,061 |
| Main-checkout cookbook edits applied by hand | 3 | — |

- **Where the copies went:** `ops/rebuild-2026-09-29/codex/` (581), `verify-b8747b6-independent/` (333), `verify-e197528-independent/` (277), `1.3/` (208), `verify-merge-1.3/` (135), the pause checkpoint and the architecture image receipt and prompt.
- **Transcripts:** 62 Codex transcripts of 512 KiB or less are included (16,001,189 B). Larger transcripts stay in MANIFEST.
- **Copy-time checks:** every source was re-hashed against the inventory, and all 1,546 matched. Every file is 512 KiB or less and none has a weights, array or corpus extension (`npz`, `pt`, `ckpt`, `pth`, `safetensors`, `bin`, `npy`, `parquet`). Each was scanned with strict credential patterns (private-key blocks, W&B, Kaggle, GitHub, OpenAI/Anthropic, Hugging Face, AWS and RunPod keys, Bearer tokens, password and secret assignments, SSH public keys), plus Fernet blobs and the pod's SSH host and port (taken from the tracked redaction map and not repeated here). Nothing matched.
- **Broad keyword pass on the copies:** the only hits were `wandb_run_id` identifiers, test names, prose ("Kaggle framework keys") and one Codex system-prompt line naming an API-key skill. No IP-like strings appeared apart from package version numbers.
- **Promoted from DEFER:** `codex/verify-gpu-bundle-r1..r3` (prompt, report, transcript). The inventory deferred them to the gpu-receipts landing. That work has since merged, and the merged `gpu-checks-2026-09-29/README.md` cites these reviews. The same checks passed.
- **Architecture PNG:** `ops/v3-architecture-image-2026-09-29/v3-agent-architecture.png` is 1,361,061 B, above the inventory's 512 KiB text cap. It is committed because the encoder Reference cites it as a repository source. Its SHA-256 matches its receipt. It is an image, not weights or a corpus.
- **Two copies were already in place:** the image receipt and prompt were added by the architecture-image commit, so the copy step found them identical.

## What stays out of git

| Class | Files | Bytes | Why |
|---|---:|---:|---|
| MANIFEST | 262 (261 still local) | 67,738,146 | Bulky or reconstructible; kept at the listed local path |
| DEFER | 90 (62 still deferred) | 16,890,641 | Evidence for in-progress work; lands with its own merge. Task 7.1 landed: 11 committed, 2 transcripts above 512 KiB stay local |
| EXCLUDE | 3 | not recorded | Failed secret triage; only path and reason are recorded |
| SKIP | 21 | 1,375,311 | Already tracked on integration with identical content |

**MANIFEST by reason**

| Reason | Files | Bytes |
|---|---:|---:|
| Non-transcript file > 512 KiB (includes the committed PNG, a 15 MB `rs.abi3.so` build product and two verifier `inventory.json` files) | 12 | 39,587,072 |
| Codex transcript > 512 KiB (largest: `task-1.3-impl-resume-transcript.log`, 8,757,062 B) | 14 | 23,887,837 |
| Verifier scratch source copies (reconstructible from git) | 144 | 1,883,013 |
| Binary (`.jsonl.gz` engine fixtures copied by a verifier) | 5 | 1,301,683 |
| Copies of tracked files at a named commit | 13 | 414,248 |
| Integration files collected by a verifier | 63 | 224,217 |
| Reconstructed copies of tracked files | 4 | 222,372 |
| Mutation or superseded copies (`*.original*`, `*.stale`) | 6 | 217,704 |
| Symlink to a worktree's `scripts/` | 1 | 0 |

**DEFER by task**

| Task | Files | Status |
|---|---:|---|
| 5.x BC | 30 | Deferred |
| Phase 7 (7.3, 7.4) | 11 | Deferred |
| Phase 7 (7.1) | 11 | Committed with `kg/merge-7-1` (see "Task 7.1 landing") |
| Phase 7 (7.1) | 2 | Local: `task-7.1-impl-transcript.log` and `task-7.1-impl-r2-transcript.log` exceed 512 KiB |
| 1.4 native env | 18 | Deferred |
| 1.5 Python adapter | 3 | Deferred |
| GPU checks (gpu-receipts) | 6 | Already tracked on BASE, identical (landed with `kg/merge-gpu-receipts`) |
| GPU checks (gpu-receipts) | 9 | Committed in this sweep (see above) |

**EXCLUDE.** All three are Codex transcripts in the main checkout's `ops/rebuild-2026-09-29/codex/`:

- `verify-flash-attn-r2-transcript.log` and `verify-merge-evidence-r2-transcript.log` contain the pod's unredacted SSH host and port, which the tracked flash-attn evidence deliberately redacts.
- `verify-merge-1.1b-r2-transcript.log` contains opaque Fernet-style encrypted blobs from Codex session records, which cannot be shown to be non-secret.

## Task 7.1 landing

`kg/merge-7-1` merged `kg/rebuild-7-1` (Codex `verify-7.1-r2` APPROVE at `908c73f`) and committed its compact evidence. The JSON's `committed.landed_with_task_7_1` lists every file with source, bytes and SHA-256.

- **Promoted from DEFER:** 11 of the 13 Task 7.1 entries (677,342 B): the implementation and verification prompts and reports, the view-probe source and the two verification transcripts (431,881 B and 217,598 B, under the 512 KiB cap). Each source re-hashed equal to this manifest. The two implementation transcripts (1,140,805 B and 1,197,029 B) stay local in the main checkout; their entries now say so.
- **Written after the inventory:** Codex's `verify-7.1-r2/` evidence directory in `/Users/poonszesen/kg-v3-t71` was unclassified. 46 compact files (115,127 B) are committed at the same paths. 12 files (2,287,580 B) stay local: the two tracked-file inventories (reconstructible from git) and the regenerated oracle, oracle manifest and replay copies, which `regeneration/byte-comparison.json` records as byte-equal to the committed fixtures.
- **Copy-time checks:** every copy hashed equal to its source; the strict credential patterns, Fernet blobs and the pod SSH host and port matched nothing.

## Custody gaps

- **Local-only copies.** MANIFEST and DEFER files exist only in the listed worktrees. Removing a worktree before its evidence lands (DEFER) or is archived elsewhere (MANIFEST) loses it. The JSON keeps SHA-256 values so that a later copy can be checked.
- **Coverage cutoff.** The inventory is a snapshot from 13:48:20Z. Files created or changed after that are not classified here. One example is `verify-7.3-r2-transcript.log`, a DEFER entry whose source changed after the inventory; later Codex reports in the main checkout are others.
- **Integration-side finding.** The inventory found that the integration already tracks `flash-attn-setup-2026-09-29/post-run/extract_operator_transcript.py`, whose redaction map contains the pod's literal public SSH host. That host is not a credential. This sweep does not change it.
- **Missing citations.** Of 66 `ops/` paths that integration cookbook notes cite but the integration lacks, 41 now resolve in this tree: 40 from the COMMIT copies and `codex/verify-gpu-bundle-r3.md` from the promoted DEFER files. 22 exist only on `kg/reference-2026-09-29` and 3 waited for their in-progress landings: `brief-7.4-review.md`, `task-1.5-s1-report.md` and `task-7.1-impl-report.md`. The inventory's `cited_but_missing` lists them all; this was rechecked against this tree. `task-7.1-impl-report.md` resolves since the Task 7.1 landing; the other two still wait.
