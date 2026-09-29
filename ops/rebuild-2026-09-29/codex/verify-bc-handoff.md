1. **P2 — Prohibited state still accepted by teacher loader.** [scripts/run_ppo.py:1085](/Users/poonszesen/kg-v3-bchandoff/scripts/run_ppo.py:1085) accepts valid model checkpoints containing top-level `opponent_id` or `hidden_state`; both cases reproduced. This is pre-existing and disclosed, but violates the explicit checkpoint-rejection contract. Share the allowed-key validation across loaders while preserving minimal `{"model": ...}` and LoRA checkpoints; add focused rejection tests.

2. **P3 — Correct the critic evidence wording.** [Reference:23](/Users/poonszesen/kg-v3-bchandoff/cookbook/references/bc-best-starts-ppo-with-a-fresh-critic-head.md:23), [README:495](/Users/poonszesen/kg-v3-bchandoff/README.md:495), and [ops README:43](/Users/poonszesen/kg-v3-bchandoff/ops/rebuild-2026-09-29/bc-handoff/README.md:43) misstate the denominator: **700/720 seat-values saturate (97.22%)**, while both seats saturate on 345/360 turns (95.83%). The gradient targets are each model’s opposite prediction, not a shared flipped outcome. Learned team-identity detection remains a hypothesis. Reconcile the note/index/log wording, and clarify at Reference line 25 that Isaiah lacked a BC-to-PPO path, not a critic.

The fresh-head implementation otherwise checks out. All four configs have equal model/observation/action sections. All 12 real-checkpoint mode/config combinations preserve the expected tensors and outputs. Parameter identities remain intact; only `model_and_optimizer` restores optimizer state. Initialization precedes DDP construction, restoration follows its synchronization, and last-best copies the restored student. DDP/compile consistency was source-reviewed, not GPU-tested.

The critic probe reproduced **0.118016 versus 12.364379** gradient norms, zero fresh-head saturation, and identical actor outputs across all 360 rows. Freshening the head is a reasonable documented adaptation using Isaiah’s initializer; better PPO learning remains unmeasured.

| Check | Result |
|---|---:|
| Opt-in handoff selection | 14 passed, 30 deselected |
| Requested three-file pytest suite | 193 passed, 6 skipped |
| Real checkpoint: four configs × three modes | 12 passed |
| Actual PPO load/resume schema probes | 8 passed |
| Teacher prohibited-key probes | 2 improper acceptances |
| Minimal LoRA checkpoint test | 1 passed |
| Mutation (a): disable restoration | 1 failed → 1 passed restored |
| Mutation (b): remove key guard | 1 failed → 1 passed restored |
| Mutation (c): empty mode keys | 1 failed → 1 passed restored |
| Combined mutation baseline/restoration | 3 passed / 3 passed |
| Mypy | 69 files, no issues |
| Ruff | No issues |
| Doc freshness | Passed; 21-path BASE→HEAD mapping also checked |
| Cookbook lint | 3 concepts passed; index/log exempt |
| Checkpoint/manifest/shard custody | 3 hashes matched |
| Kaggriculture pre-environment stop | Preserved and exercised |

The six skips include the real-checkpoint test executed separately and five existing integration gaps. CPU only, `OMP_NUM_THREADS=2`; no production training or GPU work.

[Full report and evidence](/tmp/kg-bc-verify-r1.RFBAc7/verification.md). Scratch mutations were restored byte-for-byte. Final `git status --short` is empty; no worktree was modified.

VERDICT: REQUEST CHANGES