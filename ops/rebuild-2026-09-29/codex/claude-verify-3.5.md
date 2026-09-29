# Task 3.5 verification: bounded local functional check

Reviewer: independent Claude subagent (substitute for Codex during its usage limit; owner-approved). Not a Codex verdict.

Target: `/Users/poonszesen/kg-v3-t31`, branch `kg/rebuild-3-1`, commits `268b1b4`
(test, evidence, cookbook) and `266c5e7` (mutation evidence), on base `79ae216`.
Mac CPU only; no push.

VERDICT: APPROVE (no blocking findings; P3 notes below)

## Plan items checked

| Plan item | Where covered | Result |
|---|---|---|
| Tiny model, 2 envs, 2 updates, CPU | `test_kaggriculture_two_update_functional_check_through_main[promoted\|held]` runs `run_ppo.main()` on `configs/kaggriculture.yaml` (`kaggriculture_cpu`, 33,615 params, 2 envs, horizon 2, `--max-env-steps 8`); asserts logged steps `[4, 4, 8, 8]` and 4 optimizer steps | covered |
| Finite losses | every training and evaluation metric asserted finite, `loss/total_loss` present; CLI history finite (total 0.0017179, 0.0013280) | covered |
| Checkpoint written and loadable | exact set of periodic (4, 8), final and last-best files; final reloads through `_create_eval_model_for_config` + `_load_model_from_checkpoint` with `env_steps=8` and equal weights; CLI checkpoint reloads with `env_steps=64` | covered |
| Evaluation plus promotion branch | real native evaluation after each update (2 games to step 5); threshold 0.0 promotes (last_best rewritten at step 8, teacher active, teacher cache in update 2), 1.5 holds (last_best stays step 0, no teacher) | covered |
| `eval_replay_games=0` | asserted in the loaded config; no `eval_replays/` directory; the >0 rejection is covered by the existing `test_main_rejects_kaggriculture_replay_before_allocation` | covered |
| CLI with `--wandb-mode offline` | `ops/rebuild-2026-09-29/3.5/cli-*`: exit 0, 2.95 s wall, 502.3 MB max RSS, project `kg-v3`, not synced | covered, without evaluation (see P3-1) |
| Under 2 min and 1 GB | test 0.70 s / 349.7 MB; CLI 2.95 s / 502.3 MB | met; `just py-prepare` as a whole peaked at 2.10 GB (P3-3) |

## Mutations (strictly adversarial)

`ops/rebuild-2026-09-29/3.5/verify/mutate.py` applied each mutation to
`scripts/run_ppo.py`, ran the two functional-check cases, and restored the file
with `git checkout`, then checked its hash (`31fe4e55...b03e`). Log:
`ops/rebuild-2026-09-29/3.5/verify/mutations.log`. The baseline passed 2 of 2.

| Mutation | Result |
|---|---|
| promotion comparison `>=` changed to `>` | killed (`promoted`) |
| last_best refresh skipped | killed (`promoted`) |
| promoted last_best checkpoint written to another path | killed (`promoted`) |
| teacher activation on promotion skipped | killed (`promoted`) |
| final checkpoint written with `env_steps=0` | killed (both) |
| `replay_dir` passed even when `eval_replay_games == 0` | survived, equivalent: `_evaluate_games` writes replays only when `replay_games > 0` |

Five of six were killed. The survivor is an equivalent mutant, so it is not a test gap.

## Other checks

- The test patches only launch plumbing: the release-build assert, `configure_torch`, the single-process session, the probed compile stack and the logger. It also patches `FullConfig.from_file`, which lowers `rl.checkpoint_freq` to one update only after the real validated load, and asserts the loaded value is 1,000 first. The env, model, trainer, evaluation, promotion, teacher and checkpoint code are real. `monkeypatch` restores the inherited classmethod afterwards.
- The patched cadence is justified by measurement. Two updates at 2 envs reach Isaiah's 1,000-step floor only at horizon ≥ 250. A horizon-500 trial measured 5.09 GB max RSS, recorded in `functional-check.md`, because CPU attention materializes the 709-token score matrix.
- `just py-prepare` passed: 2,391 tests passed, 6 hardware/backend skips, and docs-fresh passed. The cookbook lint hook passed on the three edited notes, and the staged-source check with the log requirement passed. The pre-commit hook passed on both commits.
- The committed evidence contains no credentials. `runs/`, `wandb/` and `*.pt` stay gitignored, and `custody-sha256.txt` hashes them.
- Stateless policy, one trainer and no v2 code: the change adds only a test and evidence. No trainer or model code changed.

## P3 notes (non-blocking)

1. The CLI run reaches no evaluation, because the validated `checkpoint_freq` floor is not crossed in 64 env steps. The evaluation and promotion branch is proven only in-process through `main()` with a patched cadence. This is stated in the evidence and the Reference.
2. `main` asserts a release build, so the native extension was rebuilt in release mode first (28.6 s, 0.90 GB). The local `.so` is now a release build. That is why the existing native smoke ran in 0.05 s here, against the 4.6 s recorded earlier under the debug build.
3. `just py-prepare` as a whole peaked at 2.10 GB RSS, above the 1 GB bound. The functional test and CLI run stayed under it.
4. The brief said the Mac has no W&B key, but `~/.netrc` holds an `api.wandb.ai` entry. Offline mode sent nothing. The evidence records this without the secret.
5. The losses carry no learning claim. The 1,000-step warmup keeps both updates' learning rate tiny. Evaluation seat assignment still uses the unseeded global torch RNG (inherited), and the test seeds torch before launching.
