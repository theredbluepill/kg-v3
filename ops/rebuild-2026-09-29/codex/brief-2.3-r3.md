All seven prior findings and the additional document-replacement edit are substantively addressed. No blocking issue or superseded v1/v2 requirement remains.

**Low severity — qualify the actor ordinal.** [Brief line 16](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/briefs/2.3-action-heads.md:16) says `unit_actor` is “the frame ordinal.” This applies only to unit frames. Replace it with:

> Unit-frame actor ordinal; zero in market, STOP, padding and inactive frames.

The reference Rust grammar explicitly requires zero after the unit phase (`myolie_sampler.rs:153`); reference token assembly makes the same distinction (`kaggriculture.py:693–712`). The canonical-equality requirement already provides the intended guard, so this is a wording correction.

The requested edits check out as follows:

| Prior finding | Verified in v3 |
|---|---|
| 1 — Mask indexing and quantity cases | Lines 33–52, 134–140 |
| 2 — Complete, safely indexed replay validation | Lines 85–108 |
| 3 — Runtime context, placement, sentinel and HIRE capacity | Lines 54–69, 133 |
| 4 — Saved per-slot replay comparisons, malformed inputs, compiled gradients and GPU handoff | Lines 126–157, 171 |
| 5 — Embedding-weight registration, explicit output reset and optimizer membership | Lines 23–25, 159–163 |
| 6 — Log-probability and entropy layouts | Lines 112–120 |
| 7 — Independent Gumbels, raw/final HIRE counts, STOP density and actual-implementation enumeration | Lines 71–83, 142–149 |
| Additional edit — Replace contradictory old requirements and separate synchronization removal from L6 remediation | Lines 5, 108, 171 |

Checked against all requested reference files, Isaiah’s model, the current Kaggriculture model and contract v4. Read-only source review; no files changed or tests executed.

VERDICT: APPROVE WITH EDITS