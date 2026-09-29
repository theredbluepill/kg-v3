Reviewer: independent Claude subagent (substitute for Codex during its usage limit; owner-approved). Not a Codex verdict.

# Verify: BC best to PPO warm start (kg/rebuild-bc-handoff), round 2

Date: 2026-09-30. Mac, CPU only. No training and no GPU work.

## Scope

- Branch `kg/rebuild-bc-handoff` at tip `2d00284`. Reviewed diff: `git diff 218a05b...HEAD` (commits `5349e96`, `ca51222`, `a814544`, `2d00284`; 26 files, +1192/-38).
- Target: `--load-model-weights-mode model_fresh_critic_head` and `_fresh_state_keys_for_mode` (`scripts/run_ppo.py`); `PPOTrainer.load_model_weights(..., fresh_state_keys=...)`, `CHECKPOINT_KEYS`/`OPTIONAL_CHECKPOINT_KEYS` and `reject_unknown_checkpoint_keys` (`python/owl/train/ppo.py`); the new `warm_start.json` and `warm_start/*` summary custody; the critic decision; and the tests, docs, ops evidence and cookbook claims.
- Governing sources: `cookbook/references/bc-best-starts-ppo-with-a-fresh-critic-head.md` (the spec), Codex r1 `ops/rebuild-2026-09-29/codex/verify-bc-handoff.md`, Claude r1 `ops/rebuild-2026-09-29/codex/claude-verify-bchandoff-r1.md`, and the stateless-policy and recipe-aligns-to-Isaiah rules in `CLAUDE.md`.
- Scratch copy: detached worktree `/tmp/cv-bchandoff-r2` at `2d00284`. Fixtures copied from the main checkout, and owl.rs built by `uv sync`. The scratch copy is now removed. `/Users/poonszesen/kg-v3-bchandoff` has no tracked modifications and is still at `2d00284`.

## Prior findings (Claude r1)

| Finding | Status | Evidence |
|---|---|---|
| P2-1: the `run_ppo.main` mode wiring was untested (M8, M9 survived) | **RESOLVED** | `test_fresh_launch_from_checkpoint_uses_starting_checkpoint_as_teacher` is now parametrized over all three modes (`tests/scripts/test_run_ppo.py:957-1134`). It runs `main` through a fake trainer and asserts the `(model, mode)` passed to `_fresh_state_keys_for_mode`, the `fresh_state_keys` it receives, and `load_optimizer is expect_load_optimizer`. Re-run here: M8 (pass `"model_only"`) gives 2 failures and M9 (`!= "model_only"`) gives 1, the same counts as `mutations-r1fix.txt`. |
| P2-2: no in-run record of the warm-start checkpoint, its hash or the mode | **RESOLVED** (with a weak test oracle, see P3-1) | `scripts/run_ppo.py:207-212,339-354,406-408`: on a fresh launch with `--load-model-weights`, the main rank writes `warm_start.json` (resolved path, SHA-256, mode) and sets `warm_start/*` summary keys. I ran `_warm_start_record` directly on the real checkpoint, given as a relative path from `/tmp`. It returned `/private/tmp/kg-v3-bc-best/checkpoint_bc_best.pt`, `fd8545872aca…8e6f51` (equal to `SHA256SUMS`) and `model_fresh_critic_head`. Mutations dropping the summaries, the session hand-off, or the mode are caught. |
| P3-1 (r1): SLURM rejects the new mode | RESOLVED | `launch-train.sbatch:210-216` accepts it and `bash -n` passes. `docs/containerization.md` documents it. The sbatch file has no automated test (it had none before this branch either). |
| P3-2 (r1): loader-scope wording | RESOLVED | The Reference now scopes the claim to `run_ppo`/`PPOTrainer`, names the three unchecked Orbit loaders, and scopes minimal checkpoints to `teacher_init`. `ppo._checkpoint_metadata` raises a named `checkpoint is missing keys` error, tested at `test_bc.py:1121` (mutation R8 caught). I enumerated every `torch.load` in `run_ppo.py`/`ppo.py`; all four go through a key check. |
| P3-3 (r1): two allow-lists | RESOLVED | `run_ppo._checkpoint_metadata` derives from `ppo.CHECKPOINT_KEYS`/`OPTIONAL_CHECKPOINT_KEYS`, and `test_run_ppo_checkpoint_keys_derive_from_the_trainer_key_set` pins it (R13 caught, 6 failures). |
| P3-4 (r1): the 4-rank claim was not in the branch's own evidence | **PARTIAL** | Fixed in the handoff Reference, its index line and the log. It remains in `top-1-team-bc-checkpoint-is-selected-by-held-out-nll-only.md:31` (see P3-3 below). |

## Checks (scratch worktree, `OMP_NUM_THREADS=2`, `CARGO_BUILD_JOBS=2`)

| Check | Result |
|---|---|
| `uvx --from rust-just just py-prepare` | exit 0. Ruff isort, format and lint all passed. mypy: no issues in 70 source files. pytest: **2,196 passed, 12 skipped**, matching `py-prepare-r1fix.log`. Peak RSS 2.25 GB for the whole target (over the 1 GB tiny-check budget; recorded, not omitted). |
| `docs-fresh` | Vacuous in a clean checkout, because it diffs the working tree against HEAD. I re-ran it with the index soft-reset to `218a05b` so the whole branch counted as changed: "No doc updates required", exit 0. HEAD was restored to `2d00284` afterwards. |
| Three-file shard `test_bc.py test_run_ppo.py test_teacher.py` | 196 passed, 6 skipped (the opt-in real-checkpoint test and 5 existing Task 1.4/3.1 gaps). This is the mutation baseline. |
| Handoff selection `-k "bc_best or ranked or prohibited or fresh_critic"` (collect-only) | 13 of 44 at HEAD, and 13 of 44 at `5349e96`. See P3-4 for the recorded "14". |
| Custody `shasum -c ops/.../SHA256SUMS` | All 5 files in `/tmp/kg-v3-bc-best` OK, including the checkpoint `fd854587…6f51`. |
| `_warm_start_record` on the real 52 MB checkpoint, relative path | Correct resolved path and SHA-256, with a multi-chunk read. Peak RSS 0.28 GB. |
| Cookbook `--staged-sources --require-log` (branch index staged against `218a05b`) | rc 0. Every `repository:` source in the new Reference exists at the tip. The PostToolUse payload mode returned `{}` even for a deliberately bad note, so that invocation did not exercise the lint. I read the frontmatter of the three touched concepts by hand instead: they have all six fields and first tag `kaggriculture-v3`. |
| Source review | Order is `reset_parameters`, compile (in place, so state_dict keys are unchanged), DDP wrap, then `load_model_weights`. The capture/restore keeps the post-broadcast fresh head, and the last-best teacher is copied from the restored student. The optimizer is fresh unless the mode is `model_and_optimizer`. A non-default mode without `--load-model-weights` is rejected (`run_ppo.py:680-683`). The BC checkpoint has `env_steps` 0, so the PPO schedule starts fresh. Nothing adds opponent identity or between-turn state. `scripts/run_ppo.py` stays the one trainer, and no v2 model code enters. |

## Mutations (each applied alone in the scratch copy, the three-file shard run, the file restored and byte-compared)

| # | Mutation | Result |
|---|---|---|
| R1 (M8) | `main` passes `"model_only"` to `_fresh_state_keys_for_mode` | CAUGHT: 2 failed |
| R2 (M9) | `load_optimizer = mode != "model_only"` | CAUGHT: 1 failed |
| R3 | `_warm_start_record` never feeds bytes to the digest (`digest.update(chunk)` → `pass`) | **SURVIVED**: 196 passed, 6 skipped |
| R4 | `warm_start.json` records the unresolved CLI path | **SURVIVED**: 196 passed, 6 skipped |
| R5 | `main` passes `warm_start=None` to the session | CAUGHT: 3 failed |
| R6 | the recorded mode is a constant | CAUGHT: 2 failed |
| R7 | drop `reject_unknown_checkpoint_keys` from `run_ppo._load_model_weights` | CAUGHT: 1 failed |
| R8 | drop the named missing-key error in `ppo._checkpoint_metadata` | CAUGHT: 1 failed |
| R9 | drop the fresh-state restore in `PPOTrainer.load_model_weights` | CAUGHT: 3 failed |
| R10 | drop the `warm_start/*` `set_summary` calls | CAUGHT: 1 failed |
| R11 | `_fresh_state_keys_for_mode` returns only `critic_head.out.*` | CAUGHT: 4 failed |
| R12 | drop `reject_unknown_checkpoint_keys` from `ppo._checkpoint_metadata` | CAUGHT: 1 failed |
| R13 | `run_ppo._checkpoint_metadata` treats `total_active_entities` as required | CAUGHT: 6 failed |
| R14 | hash only the first 1 MiB chunk | **SURVIVED**: no failures |

11 of 14 are caught. All three survivors are in the warm-start custody record, and the implementation itself is correct on the real checkpoint (see Checks).

## Findings

No P1. No P2.

### P3-1: the warm-start digest oracle is vacuous
- Where: `tests/scripts/test_run_ppo.py:995` (`checkpoint_path.touch()`) and `:1131` (`hashlib.sha256(b"").hexdigest()`), which test `scripts/run_ppo.py:345-350`.
- Issue: the checkpoint file in the `main` test is empty, so the expected digest is the hash of zero bytes. An implementation that never reads the file (R3), or that reads only the first chunk (R14), gives the same value, and the suite passes. `mutations-r1fix.txt:17` and the Reference say that a digest mutation is caught. That is true only for the constant-string mutation tried there. The code is correct today: it reproduces `fd854587…6f51` on the real checkpoint. The Phase 6.2 run statement's required cite of `fd854587…` would also expose a wrong hash. That downstream check is why this is P3 and not P2.
- Fix: write more than 1 MiB of non-constant bytes into `checkpoint_path` (for example `os.urandom((1 << 20) + 17)`) and assert `hashlib.sha256(content).hexdigest()`. R3 and R14 then fail.

### P3-2: path resolution is not tested
- Where: `scripts/run_ppo.py:345` and the test at `tests/scripts/test_run_ppo.py:1128-1129`.
- Issue: pytest's `tmp_path` is already resolved and the test passes an absolute path, so dropping `.resolve()` (R4) survives. A relative `--load-model-weights runs/.../checkpoint_bc_best.pt`, the documented launch form, would then leave a relative path in the custody record. I checked that the current code resolves a relative path correctly.
- Fix: `monkeypatch.chdir(tmp_path)` and pass `checkpoint.pt` as a relative path (or go through a symlink). Then assert the absolute resolved path.

### P3-3: the 4-rank claim remains in the top-1-team note (r1 P3-4 residue)
- Where: `cookbook/references/top-1-team-bc-checkpoint-is-selected-by-held-out-nll-only.md:31` ("into the 2-, 4- and 8-rank models with equal outputs (CPU)").
- Issue: the handoff Reference now says "eager, 2- and 8-rank (4-rank by equal model sections)", but this sibling note still states the 4-rank load as observed. The fact is supported by Codex r1's 12-case run (4 configs × 3 modes), which the note does not cite. The two notes disagree.
- Fix: use the handoff Reference's wording, or cite Codex r1's 12-case real-checkpoint run.

### P3-4: the "14 targeted tests" figure is not reproducible
- Where: `cookbook/references/bc-best-starts-ppo-with-a-fresh-critic-head.md:40`, `ops/rebuild-2026-09-29/bc-handoff/README.md:84` and `targeted-tests.log` (which records no command).
- Issue: the stated selection `-k "bc_best or ranked or prohibited or fresh_critic"` collects 13 of 44 tests both at `5349e96`, where the log was taken, and at HEAD. The 1 + 9 + 1 + 1 + 1 inventory is also 13. The same note then reports "the 13 handoff tests … passed again", which implies the same set. Codex r1 also printed 14/30. The extra test is unidentified.
- Fix: record the exact command in the log, and either identify the 14th test or restate the figure as 13 with the discrepancy noted.

## Assessment

Both r1 P2s are resolved, and the M8/M9 re-runs reproduce the committed counts. The fresh-head load, the key allow-list and the mode guards are correct, and each is guarded by a test that fails under mutation. The critic decision is recorded as an implementer choice with its limits: the improvement is inferred, not measured, and the saturation figure comes from one game. What remains is test-oracle strength in the new custody record and two evidence-wording inconsistencies. All four findings are P3 edits.

VERDICT: APPROVE WITH EDITS
