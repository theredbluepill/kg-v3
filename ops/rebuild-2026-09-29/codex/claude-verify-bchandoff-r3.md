Reviewer: independent Claude subagent (substitute for Codex during its usage limit; owner-approved). Not a Codex verdict.

# Verify: BC best to PPO warm start (kg/rebuild-bc-handoff), round 3

Date: 2026-09-30. Mac, CPU only. No training and no GPU work.

## Scope

- Branch `kg/rebuild-bc-handoff` at tip `b26cb50`. Diff reviewed: `git diff 218a05b...HEAD` (commits `5349e96`, `ca51222`, `a814544`, `2d00284`, `b26cb50`; 28 files, +1375/-41). I read the whole diff, and read `b26cb50` (the r2 fixes) line by line.
- Target: `--load-model-weights-mode model_fresh_critic_head` and `_fresh_state_keys_for_mode` (`scripts/run_ppo.py`), `PPOTrainer.load_model_weights(..., fresh_state_keys=...)`, `CHECKPOINT_KEYS`/`OPTIONAL_CHECKPOINT_KEYS`/`reject_unknown_checkpoint_keys` (`python/owl/train/ppo.py`), the `warm_start.json` and `warm_start/*` custody, and the critic decision. Also the tests, docs, ops evidence and cookbook claims.
- Governing sources: `cookbook/references/bc-best-starts-ppo-with-a-fresh-critic-head.md` (the spec), Codex r1 `ops/rebuild-2026-09-29/codex/verify-bc-handoff.md`, Claude r1 and r2 (`claude-verify-bchandoff-r1.md`, `-r2.md`), and the stateless-policy and recipe-aligns-to-Isaiah rules in `CLAUDE.md`.
- Scratch copy: detached worktree `/tmp/cv-bchandoff-r3` at `b26cb50`. I copied the fixtures in from the main checkout and built owl.rs with `uv sync`. The scratch copy has since been removed. `/Users/poonszesen/kg-v3-bchandoff` is still at `b26cb50` and has no tracked modifications.

## Prior findings

| Finding | Status | Evidence |
|---|---|---|
| r1 P2-1: `main`'s mode wiring was untested | **RESOLVED** | Re-run here: M8 (`main` passes `"model_only"`) gives 2 failed, 194 passed. M9 (`load_optimizer = mode != "model_only"`) gives 1 failed, 195 passed. Both match `mutations-r1fix.txt`. |
| r1 P2-2: no record of the warm-start checkpoint | **RESOLVED** | `scripts/run_ppo.py:207-212,341-354,406-408`. I ran `_warm_start_record` on the real checkpoint, passed as `shards-top1/../checkpoint_bc_best.pt` relative to `/tmp/kg-v3-bc-best`. It returned `/private/tmp/kg-v3-bc-best/checkpoint_bc_best.pt`, sha256 `fd8545872aca…8e6f51` and `model_fresh_critic_head`. `shasum -c SHA256SUMS` passes for all 5 files. |
| r2 P3-1: the digest oracle checked nothing | **RESOLVED** | The test writes `(1<<20)+17` random bytes (`tests/scripts/test_run_ppo.py` ~996-999). R3 (no bytes hashed) and R14 (first 1 MiB only) each give 3 failed, 193 passed, 6 skipped here, which matches `mutations-r2fix.txt`. |
| r2 P3-2: path resolution was untested | **PARTIAL** | R4 (drop `.resolve()`) now gives 3 failed, as recorded. But `.resolve()` → `.absolute()` (N1) still passes 196/196: after `chdir(tmp_path)` the absolute and resolved paths are the same. See P3-1 below. |
| r2 P3-3: 4-rank claim in the top-1 team note | **RESOLVED** | `top-1-team-bc-checkpoint-is-selected-by-held-out-nll-only.md:31` now reads "eager, 2- and 8-rank … the 4-rank model follows by equal model sections". No stale "2-, 4- and 8-rank" wording is left in `cookbook/` or the ops README. |
| r2 P3-4: the "14 targeted tests" figure | **RESOLVED** | The Reference (line 40) and the ops README restate it as 13 (reproducible) with the 14th unidentified, and give the exact command. I did not identify the 14th either. |

## Checks (scratch worktree, `OMP_NUM_THREADS=2`, `CARGO_BUILD_JOBS=2`)

| Check | Result |
|---|---|
| `uvx --from rust-just just py-prepare` | rc 0. Ruff: "All checks passed!" twice. mypy: "Success: no issues found in 70 source files". pytest: **2,196 passed, 12 skipped** (79 s), which matches `py-prepare-r2fix.log`. `check_doc_freshness`: "No doc updates required". Memory was not measured on this run; r2 measured 2.25 GB peak for the whole target, which is over the 1 GB tiny-check budget. |
| docs-fresh over the whole branch (index soft-reset to `218a05b`, HEAD then restored to `b26cb50`) | "No doc updates required", rc 0. |
| Three-file shard `tests/scripts/test_run_ppo.py tests/kaggriculture/test_bc.py tests/kaggriculture/test_teacher.py` | **196 passed, 6 skipped** (26 s). This is the mutation baseline. |
| Real checkpoint, fresh mode, no forward (2-rank config, `force_flash_attn` false) | `_fresh_state_keys_for_mode` returns exactly `critic_head.{up,out}.{weight,bias}`. After `PPOTrainer.load_model_weights`, every non-critic tensor equals the checkpoint, and every critic tensor equals the pre-load fresh init and differs from the checkpoint. The optimizer state is empty and `env_steps` is 0. |
| Custody | `shasum -a 256 -c ops/rebuild-2026-09-29/bc-handoff/SHA256SUMS`: all 5 OK. The checkpoint's top-level keys equal `CHECKPOINT_KEYS`. |
| Cookbook lint `--staged-sources --require-log` (branch staged against `218a05b`) | rc 0. Limit: this invocation also returned rc 0 when I staged a deliberate `repository:scripts/does_not_exist.py` source together with the log, so it does not check that sources exist. I checked instead that every `repository:` source in the three touched concept notes exists at the tip (none missing), and that their frontmatter has the required six fields with first tag `kaggriculture-v3`. |
| Source review | Order in `main`: `reset_parameters` → compile → DDP wrap → `load_model_weights`. So the captured fresh head is the post-broadcast state, and the `last_best` teacher is copied from the restored student. The optimizer stays fresh unless the mode is `model_and_optimizer`. A non-default mode without `--load-model-weights` is rejected. LoRA cannot combine with the mode, because LoRA supports only `StatelessTransformerV1` and the mode requires `KaggricultureTransformer`. The BC `wandb_run_id` (`kvl4rfda`) is returned in the metadata but not used by a fresh launch (`resume_run_id` stays None). Nothing adds opponent identity or between-turn state. `run_ppo.py` stays the one trainer, and no v2 model code enters. |

## Mutations (each applied alone, the three-file shard run, the file restored and `git diff --quiet` checked)

| # | Mutation | Result |
|---|---|---|
| M8 | `main` passes `"model_only"` to `_fresh_state_keys_for_mode` | CAUGHT: 2 failed |
| M9 | `load_optimizer = mode != "model_only"` | CAUGHT: 1 failed |
| R3 | digest never fed (`digest.update(chunk)` → `pass`) | CAUGHT: 3 failed |
| R4 | `.resolve()` dropped | CAUGHT: 3 failed |
| R14 | hash only the first 1 MiB | CAUGHT: 3 failed |
| N1 | `.resolve()` → `.absolute()` | **SURVIVED**: 196 passed, 6 skipped |
| N2 | fresh-state capture without `.clone()` (aliases the live parameter, so the restore becomes a no-op) | CAUGHT: 3 failed |
| N3 | `reject_unknown_checkpoint_keys` lets `opponent_id` through | CAUGHT: 1 failed |
| N4 | unknown-`fresh_state_keys` guard disabled | CAUGHT: 1 failed |
| N5 | `KaggricultureTransformer` isinstance guard disabled | CAUGHT: 1 failed |
| N6 | fresh keys widened with `critic_value_tokens` | CAUGHT: 4 failed. A first variant that added `value*` keys matched no key, so it was an equivalent mutant and I replaced it. |
| N7 | restore only the `critic_head.out.*` tensors | CAUGHT: 3 failed |
| N8 | `main` never builds the warm-start record | CAUGHT: 3 failed |
| N9 | named missing-metadata-key error disabled | CAUGHT: 1 failed |
| N10 | `main` passes the wrapped `model` instead of `unwrap_model(model)` | Survived, but it is equivalent on CPU without DDP. Under DDP the isinstance guard would raise loudly at launch, so no silent error follows. Not a finding. |

13 of 14 non-equivalent mutants are caught. The one real survivor is N1.

## Findings

No P1. No P2.

### P3-1: the path oracle checks "absolute", not "resolved"
- Where: `tests/scripts/test_run_ppo.py:1001-1002,1137`, which tests `scripts/run_ppo.py:345`. The claim appears in the handoff Reference line 39, in `cookbook/log.md` (the 2026-09-30 r2-fix entry) and in the index line ("checks … the resolved path").
- Issue: after `monkeypatch.chdir(tmp_path)`, `Path("checkpoint.pt").absolute()` is the same as `.resolve()`, because `tmp_path` is already a realpath. So replacing `.resolve()` with `.absolute()` (N1) passes the whole shard. That implementation would record a path such as `runs/x/../bc/checkpoint_bc_best.pt`, or a symlink rather than its target. The recorded SHA-256 still pins the bytes, so this does not affect custody. That is why it is P3.
- Fix: pass the checkpoint through a `..` segment (e.g. make `sub/` and pass `sub/../checkpoint.pt`) or through a symlink (`link.pt -> checkpoint.pt`), then assert `str(checkpoint_path.resolve())`. Or narrow the claim to "absolute".

### P3-2: the reopen conditions watch the critic but not the BC actor it shares a trunk with
- Where: `cookbook/references/bc-best-starts-ppo-with-a-fresh-critic-head.md:43`, and the Phase 6.2 launch paragraph.
- Issue: the fresh head's gradient norm on the 30-row probe was 12.4. The configs clip the global gradient norm at `max_grad_norm: 10.0` with `vf_coef: 2.0` (`configs/kaggriculture_2rank.yaml:60-62`). Early value-loss gradients therefore flow through the shared trunk (value tokens attend to it), and they can dominate the clipped global step. That could scale down or perturb the BC actor's updates before the critic settles. This is an unmeasured risk, not an observed defect. The note's only reopen condition is critic EV/value loss, so an early erosion of the BC policy would not trigger reopening.
- Fix: the Phase 6.2 run statement should name actor-side observables for the first updates: grad-norm clip rate, approx-KL/clip fraction, and held-out BC NLL at the first checkpoints against the BC best's 0.480. The Reference's reopen condition should add "or the actor drifts from the BC start faster than a comparison start". No code change is needed.

### P3-3: the records still say the r2 fixes are unverified
- Where: `cookbook/references/bc-best-starts-ppo-with-a-fresh-critic-head.md:39` ("The r2 fixes are not re-verified by a separate reviewer"), `cookbook/log.md` (the 2026-09-30 entry), and `ops/rebuild-2026-09-29/bc-handoff/README.md` Limits.
- Issue: this r3 report re-verifies them (R3/R4/R14 at 3 failed each, py-prepare 2,196/12). The statements go stale once this report lands.
- Fix: cite `ops/rebuild-2026-09-29/codex/claude-verify-bchandoff-r3.md` (a Claude stand-in, not a Codex verdict), with its N1 residue. Make the cookbook edit together with its index and log entry, as the contract requires.

## Assessment

Both r1 P2s and three of r2's four P3s are resolved. Their mutation counts reproduce exactly, and `just py-prepare` gives 2,196 passed and 12 skipped. The fresh-head restore, the key allow-list, the mode guards and the custody record are correct on the real checkpoint. Each is guarded by a test that fails under mutation, including the subtler no-clone aliasing mutant. What remains: one weaker-than-claimed path oracle, one unmeasured interaction to state for the Phase 6.2 run, and record wording to update. All three are P3.

VERDICT: APPROVE WITH EDITS
