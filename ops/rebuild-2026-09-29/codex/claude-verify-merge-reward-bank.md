Reviewer: independent Claude subagent (substitute for Codex during its usage limit; owner-approved). Not a Codex verdict.

# Verify-merge: own-bank reward shaping onto the integration (label `reward-bank`)

The landing Claude agent wrote this pass. It is not independent of the merge it checks.

- **Staging:** worktree `/Users/poonszesen/kg-v3-m-reward-bank`, branch `kg/merge-reward-bank-c`.
- **BASE:** the integration tip `kg/isaiah-gap-closure` `4731c496d90453bcece605ea32e546b927550d8e` (the 8-rank prep landing).
- **Merged:** `kg/rebuild-reward-bank` `b73b693` (`56ec719` native + oracle term A, `8de8f0d` presets, `ab98e73` cookbook, `3d10f52`/`8aeba3d` verify r1 fixes, `71a964d`/`b73b693` verify r2 edits).
- **Merge commit:** `e85cf5b` (`--no-ff`).
- **Owner, 2026-09-30, verbatim:** "A is good + decrease the LR by half?"

## Checks

1. **Merge shape.** The branch was cut from BASE (`git merge-base` = `4731c49`), so nothing conflicted. The merge tree `85182b0` equals the tree of `b73b693`. No semantic resolution was needed.
2. **Term A as the owner chose it.** `src/kaggriculture/reward.rs` adds `B(bank) = min(cap_b, w_b * max(0, bank) / S)` per seat. Seat `s` receives `B(after_s) - B(before_s)` added to its relative economic term before the first f32 rounding. The rival's bank does not enter it, so the term is absolute and not zero-sum, which is what option A asked for ("a small per-step reward for own bank growth ... capped like the other terms"). `cap_b` joins the cap budget, so `terminal_scale` falls to `1 - .25 - .25 = 0.5` in the bank presets, and a complete-game return stays in [-1, 1].
3. **Before/after banks at the seam.** In `src/kaggriculture/env.rs` both bank reads bracket `game.step_with_market_metrics` inside one slot, before any auto-reset. So a delta never spans two games, and no per-seat bank state is carried between steps. MM1 below (after-banks passed as before-banks) is killed.
4. **Halved learning rates.** A normalized diff of `configs/kaggriculture_4rank_bc_finetune.yaml` against `..._4rank_bc_finetune_bank.yaml` differs in exactly four values: `econ_bank_weight` 0 -> .25, `econ_bank_cap` 0 -> .25, `adamw_lr` 1e-5 -> 5e-6 and `muon_lr` 2e-4 -> 1e-4. The 2- and 8-rank bank presets carry the same four values. The recipe-J presets without `_bank` keep recipe J unchanged. The new 4-rank recipe-J base differs from the 8-rank one only in `n_envs` (64 vs 32) and `segments_per_minibatch`, which keeps the global workload.
5. **Off by default, byte-identical.** Every other Kaggriculture config sets `0.0 / 100000.0 / 0.0` explicitly. With `w_b = 0` the native code adds nothing before the f32 cast. `tests/kaggriculture/test_env_reference.py` replays the recorded 16-game fixture bit-exactly under it, and it passes in this merge's `just prepare`.
6. **Stateless policy.** Outside `reward.rs`, `rewards.py`, the bindings and the tests, the only reader of the bank fields is the adapter's `reward_bank_mean` telemetry (`python/owl/kaggriculture/env.py`). No observation, model, head or loss code reads the reward config. Opponent identity is not involved.
7. **Launch path against the required-field migration.** The three fields are required, so a pre-change `config.yaml` no longer loads as `FullConfig`. `scripts/run_ppo.py` reads a sibling `config.yaml` only for `rl.teacher_init` (`_checkpoint_config_path`) and fixed teachers. The planned `--load-model-weights <BC best> --load-model-weights-mode model_only` launch seeds the student and the last-best teacher without reading one. The Decision and `README.md` state the migration (add the three off values). No launch was run here.
8. **Evidence spot-check.** I read `/Users/poonszesen/kg-v3-pod4/ops/rebuild-2026-09-29/pod4-2026-09-30/main-J-4rank/pod-receipts/iterations.txt` and `watchdog.log`. They give `own_bank_mean` 72.6k (it. 12), 73.1k (23), 74.3k (34), 68.4k (45), 52.6k (57), 17.9k (68) and 5,684 (79). The watchdog stopped the run at 00:46:34Z on its 20k floor. The printed LR peaks at 2.00e-4 at iteration 63. This matches the Decision. The decline therefore began during warm-up (1.44e-4 at iteration 45), before the peak. The bank still held at iteration 34 (1.09e-4), close to the halved peak of 1e-4. That is one seed, and it does not separate the LR from the objective, as the Decision states.
9. **Prior branch reviews.** `codex/claude-verify-reward-bank-r1.md` REQUEST CHANGES (it corrected `w_b` 1.0 -> .25; see finding P2 below), applied in `3d10f52`. `codex/claude-verify-reward-bank-r2.md` APPROVE WITH EDITS, applied in `71a964d`.
10. **`just prepare` on `e85cf5b`.**
    - Command: `CARGO_BUILD_JOBS=2 OMP_NUM_THREADS=2 uvx --from rust-just just prepare`. It exits 0 on the first attempt.
    - Rust `owl` lib: 284 passed, 5 ignored, plus the engine and opponent crates. Python: 2,831 passed, 18 skipped. Ruff, markdown lint, mypy (77 files) and docs freshness also pass.
    - The git-ignored Orbit fixtures were copied from `/Users/poonszesen/kg-v3-int/tests/fixtures` first (`diff -rq` identical).
    - Receipt: `ops/rebuild-2026-09-29/merge-reward-bank-c/prepare.log`.
11. **Merge-seam mutations.** Six were applied to `e85cf5b`, each run against its targeted check and then restored from git. All six were killed, and the tree was restored clean. Receipt: `ops/rebuild-2026-09-29/merge-reward-bank-c/mutations.log`, with `mutations.py` beside it.
    - MM1: `env.rs` passes the after-banks as `banks_before` (Rust env tests).
    - MM2: the 4-rank bank preset at full `muon_lr`.
    - MM3: the 8-rank bank preset at full `adamw_lr`.
    - MM4: the 2-rank bank preset with the term off.
    - MM5: the 4-rank recipe-J base drifts in `n_envs`.
    - MM6: the adapter's `reward_bank_mean` sums seats instead of averaging them.
12. **Cookbook contract.** The Decision `cookbook/decisions/add-absolute-own-bank-shaping-and-halve-the-recipe-j-learning-rates.md` quotes the owner verbatim in `decider` and in the body. Its first tag is `kaggriculture-v3`, and it has repository sources. The Decisions index and prepended `cookbook/log.md` entries accompany it. It labels the values `w_b` .25, `S` 100,000 and `cap_b` .25 as agent-proposed, not owner-given. No board note exists, so no board edit is needed.
13. **Scope.** Code, config, tests and docs only. Nothing trained, no pod was created, read or touched, and `~/kaggriculture-v2` was not touched. Nothing was pushed.

## Findings

None blocking.

- **P2 (owner visibility, not a code defect).** The owner accepted term A but gave no values. The agent's original message proposed `w_b = 1.0` and described consequences that only `w_b = .25` produces. The shipped `.25` realises those described consequences (70k scores .175, saturation at 100k), not the literal `1.0`. The Decision records this. The owner should see it before a bank-preset launch.
- **P3 (attribution).** The bank presets change the objective and the LR together. A collapse or a hold on them cannot be attributed to either change alone. The Decision cites the J/2 LR-only control (`nw3klj2s`) as the separating arm, but that run's outcome is not checked here.
- **P3 (critic).** The zero-sum winner critic cannot represent the term's common mode. It is instrumented (`train/reward_bank_mean`, `train/return_common_mean`, `train/return_zero_sum_abs_mean`) but not measured on any run. The per-step float64 oracle cost of `reward_bank_mean` with the term on is also unmeasured against the rollout step time.

## Not checked

- No GPU, `torchrun`, nsys or pod execution. The halved-LR bank presets have never trained.
- The J/2 control run and W&B `gq94cyyp` itself; only the local receipts were read.
- An inert positive `w_b` whose score underflows for every reachable bank still passes validation (a Decision limit).

VERDICT: APPROVE
