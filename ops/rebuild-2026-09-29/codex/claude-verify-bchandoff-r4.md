Reviewer: independent Claude subagent (substitute for Codex during its usage limit; owner-approved). Not a Codex verdict.

# Verify: BC best to PPO warm start (kg/rebuild-bc-handoff), round 4

Date: 2026-09-30. Mac, CPU only. No training and no GPU work.

## Scope

- Branch `kg/rebuild-bc-handoff` at tip `41c95f7`. Diff reviewed: `git diff 218a05b...HEAD` (30 files, +1559/-41). I read the whole code diff (`python/owl/train/ppo.py`, `scripts/run_ppo.py`, `scripts/slurm/launch-train.sbatch`, the test changes) and `41c95f7` (the r3 fixes) line by line, plus the README, `docs/containerization.md` and the three touched cookbook concepts.
- Target: `--load-model-weights-mode model_fresh_critic_head` and `_fresh_state_keys_for_mode`, `PPOTrainer.load_model_weights(..., fresh_state_keys=...)`, `CHECKPOINT_KEYS` / `reject_unknown_checkpoint_keys`, the `warm_start.json` custody record, and the critic decision.
- Prior reviews: `claude-verify-bchandoff-r1.md` (REQUEST CHANGES), `-r2.md`, `-r3.md` (APPROVE WITH EDITS).
- Scratch copy: detached worktree `/tmp/cv-bch-r4` at `41c95f7`, owl.rs built with `uv sync`. It has been removed (`git worktree remove` + `prune`). `/Users/poonszesen/kg-v3-bchandoff` is still at `41c95f7` with no tracked modifications.

## Prior findings

| Finding | Status | Evidence |
|---|---|---|
| r1 P2-1: `main`'s mode wiring untested (M8/M9 survived) | **RESOLVED** | Re-run here as Q3 (`main` passes `"model_only"`): 2 failed. Q4 (`load_optimizer = mode != "model_only"`): 1 failed. |
| r1 P2-2: no record of the warm-start checkpoint | **RESOLVED** | `scripts/run_ppo.py` `_warm_start_record` + `warm_start.json` + `warm_start/*` summary keys. Q5 and Q8 (below) are caught by the parametrized `main` test. |
| r1 P3-1: SLURM rejected the new mode | RESOLVED | `launch-train.sbatch` allow-list includes `model_fresh_critic_head`; `docs/containerization.md` documents it. |
| r1 P3-2: loader-scope wording | RESOLVED | Reference scopes the claim to `run_ppo`/`PPOTrainer` loaders and names the three unchecked Orbit loaders; `ppo._checkpoint_metadata` raises a named missing-keys error. |
| r1 P3-3: two allow-lists | RESOLVED | `run_ppo._checkpoint_metadata` derives from `ppo.CHECKPOINT_KEYS` / `OPTIONAL_CHECKPOINT_KEYS`. |
| r1 P3-4: 4-rank claim | RESOLVED | Both notes say "eager, 2- and 8-rank … 4-rank by equal model sections". |
| r2 P3-1: digest oracle vacuous | RESOLVED | Test writes `(1<<20)+17` random bytes; r3 re-ran R3/R14 (3 failed each). Not re-run here; the test is unchanged since r3 except the path line. |
| r2 P3-2: path resolution untested | **RESOLVED** (was PARTIAL in r3) | `tests/scripts/test_run_ppo.py:1001-1008` now passes `sub/../checkpoint.pt` and asserts its absolute form differs from the resolved one. Q5 (`.resolve()` → `.absolute()`, r3's N1) now gives **3 failed** (all three mode parametrizations), matching `mutations-r3fix.txt`. |
| r2 P3-3: 4-rank claim in the top-1-team note | RESOLVED | `top-1-team-bc-checkpoint-is-selected-by-held-out-nll-only.md:31`. |
| r2 P3-4: "14 targeted tests" | RESOLVED | Restated as 13 reproducible, 14th unidentified. |
| r3 P3-1: path oracle checks "absolute" not "resolved" | **RESOLVED** | Same evidence as r2 P3-2 above. |
| r3 P3-2: actor drift unwatched | **RESOLVED** | Reference (new paragraph after the launch command, and the reopen bullet) and ops README name `optimizer/grad_norm` against 10.0, `policy/approx_kl`, `policy/clipfrac`, held-out BC NLL against 0.480, beside `train/explained_variance` and `loss/value_loss`. I checked every metric name exists in `ppo.py` (lines 726, 1157, 3240, 3247, 3248), that `optimizer/grad_norm` is the mean over minibatches of `clip_grad_norm_`'s return (the pre-clip norm), and that no clip-rate metric exists, as stated. `max_grad_norm: 10.0` and `vf_coef: 2.0` match the eager, 2- and 4-rank configs. The risk is labelled "unmeasured". |
| r3 P3-3: stale "r2 fixes not re-verified" | **RESOLVED** | Reference, index line, log and ops README cite r3. They now say "the r3 fixes are not re-verified by a separate reviewer", which this report does (see P3-1). |

## Checks (scratch worktree, `OMP_NUM_THREADS=2`, `CARGO_BUILD_JOBS=2`)

| Check | Result |
|---|---|
| `uvx --from rust-just just py-prepare` | rc 0. Ruff "All checks passed!" twice; mypy "Success: no issues found in 70 source files"; pytest **2,196 passed, 12 skipped** (78.8 s), matching `py-prepare-r3fix.log`. `check_doc_freshness` rc 0 (vacuous in a clean checkout, as r2 noted; r3 ran the branch-wide variant, and `41c95f7` touches only a test and cookbook/ops files). |
| Three-file shard `test_run_ppo.py test_bc.py test_teacher.py` | **196 passed, 6 skipped** (26.7 s; peak memory footprint 1.24 GB, over the 1 GB tiny-check budget). Mutation baseline. |
| Opt-in real-checkpoint test (`KG_V3_BC_BEST`, `KG_V3_BC_SHARDS` pointing at `/tmp/kg-v3-bc-best`) | 1 passed. |
| Real checkpoint, all three modes (2-rank config on CPU, via the test helpers) | Top-level keys equal `CHECKPOINT_KEYS`. `model_fresh_critic_head` fresh keys are exactly `critic_head.{up,out}.{weight,bias}`; after the load those equal the pre-load init and differ from the checkpoint, and every other tensor equals the checkpoint. `model_only` / `model_and_optimizer` load every tensor. `env_steps` 0 in all modes. |
| Custody | `shasum -a 256 -c ops/rebuild-2026-09-29/bc-handoff/SHA256SUMS`: all 5 OK (checkpoint `fd854587…6f51`). |
| Note sources | Every `repository:` source in the three touched concepts exists at the tip. |
| Source review | Order in `main` unchanged since r3: `reset_parameters` → compile → DDP wrap → `load_model_weights`, so the captured fresh head is the post-broadcast state. `load_state_dict(fresh_state, strict=False)` cannot drop an unknown key silently because `fresh_state_keys` is checked against the model's state keys first. Unknown model tensors are rejected by `load_model_state_dict_allowing_lora`; unknown top-level keys by `reject_unknown_checkpoint_keys`. Nothing adds opponent identity or between-turn state; `run_ppo.py` stays the one trainer; no v2 model code. |

## Mutations (each applied alone to the scratch copy by exact-string replacement asserted unique, the three-file shard run, the file restored and `git diff --quiet` checked)

| # | Mutation | Result |
|---|---|---|
| Q1 | **drop the fresh-critic reset** (`if fresh_state: model.load_state_dict(...)` → `pass`) | CAUGHT: 3 failed (`test_bc_best_loads_through_ppo_load_model_weights[model_fresh_critic_head-*]`) |
| Q2 | **accept unknown keys** (`reject_unknown_checkpoint_keys` computes an empty set) | CAUGHT: 1 failed (`test_ppo_load_rejects_prohibited_checkpoint_state`) |
| Q3 | **wrong mode from `main()`**: `_fresh_state_keys_for_mode(..., "model_only")` | CAUGHT: 2 failed (`main` test, `model_and_optimizer` and `model_fresh_critic_head` params) |
| Q4 | `main` loads optimizer state for the fresh-critic mode (`!= "model_only"`) | CAUGHT: 1 failed |
| Q5 | warm-start record uses `.absolute()` instead of `.resolve()` (r3 N1) | CAUGHT: 3 failed |
| Q6 | fresh keys keep only the `critic_head` biases | CAUGHT: 4 failed |
| Q7 | restore the fresh state before the checkpoint load (order swap) | CAUGHT: 3 failed |
| Q8 | warm-start record skipped for `model_only` | CAUGHT: 1 failed |
| Q9 | `teacher_init` loader (`run_ppo._load_model_weights`) accepts unknown keys | CAUGHT: 1 failed |
| Q10 | unknown-`fresh_state_keys` guard disabled | CAUGHT: 1 failed |

10 of 10 caught. Q2 and Q9 are each guarded by one test only (`test_ppo_load_rejects_prohibited_checkpoint_state`); that is enough to fail, but it is a single point of detection.

## Claims labelled "inferred, not measured"

- "That the fresh head learns better than the BC head is inferred from the gradient probe, not measured." Labelled; acceptable.
- The actor-drift risk ("This is unmeasured"). Labelled; acceptable. See P3-2 for how strong its one number is.
- "Under DDP that state is rank 0's … (PyTorch default; not run here)." Labelled; acceptable.

## Findings

No P1. No P2.

### P3-1: the records will again call the latest fixes unverified
- Where: `cookbook/references/bc-best-starts-ppo-with-a-fresh-critic-head.md` (Checks and limits, first bullet: "The r3 fixes are not re-verified by a separate reviewer"), the 2026-09-30 r3-fix entry in `cookbook/log.md`, and the ops README Limits.
- Issue: this report re-verifies them (Q5 3 failed, py-prepare 2,196/12). Same pattern as r3 P3-3.
- Fix: optional. Cite this report (a Claude stand-in, not a Codex verdict) when the note is next edited, with its index line and a log entry, as the cookbook contract requires. Not worth a round of its own.

### P3-2: the 12.4 versus 10.0 comparison is indicative, not like for like
- Where: the Reference's actor-drift paragraph and the ops README ("The fresh head's probe gradient norm (12.4) is above the configs' global clip").
- Issue: `critic_probe.txt` measures the `critic_head`-only gradient norm of an unweighted MSE against a *flipped* outcome over one game's 360 rows. The clip applies to the global norm of the full PPO loss (with `vf_coef` 2.0, the policy and teacher terms, and all trunk parameters) per minibatch. The number could be higher or lower in the real first update. The paragraph already says "could" and "unmeasured", so it does not overclaim, and the prescribed observables (`optimizer/grad_norm` against 10.0) measure the real thing.
- Fix: optional. Add "(critic-head norm on a flipped-outcome probe, not the PPO loss's global norm)" after "12.4".

### P3-3 (informational): the custody hash covers the main rank's file, not each rank's bytes
- Where: `scripts/run_ppo.py` `_warm_start_record` (main rank, before the load).
- Issue: the SHA-256 is taken from the main rank's copy at launch; each rank then `torch.load`s its own path later. On the documented single-node 2-rank launch this is the same file, so the record is correct. With a multi-node launch using node-local copies, or a file replaced between hash and load, the record would not prove what non-main ranks loaded.
- Fix: none needed for Phase 6.2. If multi-node launches are added, hash on every rank and assert equality (or broadcast the digest and compare).

## Assessment

Every r1, r2 and r3 finding is resolved. The one r3 survivor (N1) now fails 3 tests. The three required mutations (drop the fresh-critic reset, accept unknown keys, wire the wrong mode from `main()`) and seven others are all caught, and `just py-prepare` reproduces 2,196 passed / 12 skipped. The real checkpoint loads correctly in all three modes, with exactly the four critic-head tensors fresh, and custody checksums pass. The labelled inferences are acceptable. The remaining items are optional wording (P3-1, P3-2) and one informational multi-node custody limit (P3-3). None blocks.

VERDICT: APPROVE
