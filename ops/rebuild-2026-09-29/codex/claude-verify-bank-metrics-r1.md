Reviewer: independent Claude subagent (substitute for Codex during its usage limit; owner-approved). Not a Codex verdict.

# Verify r1: learner-perspective bank telemetry (`kg/rebuild-bank-metrics`)

- Target: `kg/rebuild-bank-metrics` at `7425db7750efea6d702057c857b9d217d955c228`,
  one commit on top of merge-base with `kg/rebuild-3-1` ("Log learner-perspective
  raw-bank telemetry to W&B during Kaggriculture PPO"). 13 files, +636/-5.
- Owner request under review: carry margin/own_bank into W&B during PPO.
- Method: scratch worktree `git worktree add --detach /tmp/cv-bank-r1 HEAD`, with
  the branch's own built `python/owl/rs.abi3.so` copied in (the branch changes no
  Rust) and the branch venv's Python on `PYTHONPATH=/tmp/cv-bank-r1/python`. The
  scratch worktree was removed afterwards (`git worktree remove --force` plus
  `prune`; `/tmp/cv-bank-r1` no longer exists; it had no tracked changes).
- Machine: Mac CPU only. No CUDA, no multi-rank process group, no live W&B.

## Checks run (this version)

| Check | Result |
| --- | --- |
| `pytest -m "not slow" tests/kaggriculture tests/owl/train/test_ppo.py tests/scripts/test_run_ppo.py` | 1315 passed, 3 skipped (CUDA/pinned memory) in 36.6 s |
| Targeted subset (`-k "bank or telemetry or smoke or truncation_bootstraps or no_teacher"`) | 23 passed |
| `ruff check` + `ruff format --check` on the 8 changed Python files | clean / already formatted |
| `mypy python scripts` (as `just py-static` runs it) | no issues in 71 source files |
| Independent mixed-seat eval sign check (below) | correct |
| 12 mutations (below) | 12/12 killed |

`just py-prepare` / `docs-fresh` were not rerun here (`just` is not installed on
this shell's PATH); the author's note claims it passed.

## Math verification

**Seat mapping, training.** The native step (`src/kaggriculture/env.rs:633`)
pushes one `(record.banks[0], record.banks[1], record.margin)` tuple per completed
game, and `bindings.rs:830-842` emits `terminal_bank_0`/`terminal_bank_1` as
index-aligned lists from the same tuples. `_collect_rollout` extends them in
lockstep per step; `_self_play_bank_metrics` gathers `(bank_0_list, bank_1_list)`
per rank and flattens both in the same rank order, so pairing is preserved
across ranks. `self_play_bank_metrics` reshapes to `[2, games]`; own_bank pools
`banks.flatten()` (both seats, two values per game), which is correct because
both seats are the learner. The seat-symmetry test pins that swapping seats
changes nothing.

**Percentiles.** `torch.quantile` default (linear) on float64; tests compare
against an independent order-statistic helper. Checked by hand for a single
game (`[100, 40]` -> p10 46, p90 94) and for mixed-seat margins below.

**Draws.** `decisive = banks[0] != banks[1]`; `draw_rate = mean(~decisive)`;
`winner_bank_mean`/`loser_bank_mean` use max/min over decisive games only and
are omitted when every game drew, so no NaN. Exact float equality is the same
rule `terminal_seat_banks` uses for the eval winner, so the draw definition is
consistent with promotion.

**Absolute self-play margin.** `margin_abs = |bank_0 - bank_1|` per game;
`margin_abs_{mean,p50}`. The rationale (signed learner margin is identically
zero in self-play; `train/terminal_margin_0` only measures seat asymmetry) is
correct and documented.

**Evaluation sign convention.** `opponent_bank_metrics` receives
`env_metrics["candidate_bank"]` and `env_metrics["last_best_bank"]`, which the
pre-existing `_candidate_bank_metrics` fills per game from
`banks[seats.index(MODEL_CURRENT)]` / `banks[seats.index(MODEL_LAST_BEST)]`, so
"own" is the candidate whichever seat it holds. Both lists are appended once per
game in the same `_extend_single_env_metrics` call, so they stay aligned. Signed
margin = own - opponent.

Independent check (not in the branch's tests): three games through the real
`_candidate_bank_metrics` then `opponent_bank_metrics` - candidate seat 0 winning
3000-1000, candidate seat 1 losing (banks 3000/1000), candidate seat 1 winning
(banks 500/4500) - gave `candidate_bank [3000, 1000, 4500]`,
`last_best_bank [1000, 3000, 500]`, margins `[2000, -2000, 4000]`,
`eval/margin_mean 1333.33`, `p10 -1200`, `p50 2000`, `p90 3600`, all matching
hand calculation.

In the branch's end-to-end test (`test_kaggriculture_bank_telemetry_reaches_the_logger`)
I instrumented `_candidate_bank_metrics`: under `torch.manual_seed(41)` both
evaluation games assign `[1, 0]`, i.e. the candidate (`MODEL_CURRENT = 0`) sits
in seat 1 in both games, with banks `[12, 14]` and `[31, 0]`. So the seat-1 path
is exercised end-to-end, but only by this RNG draw (see finding 1).

## W&B reach and Orbit isolation

- Training: `train_iteration` adds the keys to `metrics` inside
  `isinstance(self._obs, KaggricultureObsBatch)`; `_run_training_loop` passes
  the dict unchanged to `logger.log`, and the W&B logger calls
  `wandb.log(metrics, step=step)` (`python/owl/train/logging.py:99-100`). No key
  allowlist sits in between.
- Evaluation: added inside `isinstance(cfg.env.obs_spec, KaggricultureObsConfig)`
  in `_evaluate_against_last_best`; the eval dict is broadcast and logged with
  `eval/promoted`.
- The integration test drives a real native trainer and real last-best
  evaluation through `_run_training_loop` into `_FakeLogger` and asserts every
  new train/eval key, finiteness, and equality of the new eval means to the old
  `eval/candidate_bank`, `eval/last_best_bank`, `eval/candidate_bank_margin`.
- Orbit: `test_trainer_smoke_keeps_metrics_finite_and_updates_parameters` asserts
  no `bank`/`margin` key in Orbit training metrics;
  `test_orbit_evaluation_logs_no_bank_telemetry` does the same for Orbit eval.
  Mutations M11/M12 (remove each gate) are killed.
- No key collisions: new names (`bank_games`, `own_bank_*`, `opponent_bank_*`,
  `margin_{mean,p10,p50,p90}`, `margin_abs_*`, `winner/loser_bank_mean`,
  `draw_rate`) do not overlap existing `terminal_bank_{0,1}`,
  `terminal_margin_0`, `eval/bank_{0,1}`, `eval/margin_0`, `eval/candidate_*`.

## Leakage into model inputs, rewards, selection

- `grep` for every new key and for `kaggriculture.telemetry` across `python/`,
  `scripts/`, `src/` finds only the two producers (`ppo.py:19`, `run_ppo.py:21`)
  and the call sites. Nothing reads the keys back.
- The training hook runs after `_update` and GAE, on host lists copied from the
  step metrics; it writes nothing to the rollout, observation buffers, rewards,
  normalizers, or trainer state.
- Promotion still reads only `eval/win_rate_against_last_best`
  (`run_ppo.py:549-551`).
- No opponent identity enters any input: `opponent_bank_metrics` takes only a
  key prefix as a label.

## Mutations (each reverted after its run)

| # | Mutation | Result | Killing tests |
| --- | --- | --- | --- |
| M1 | drop `abs()` in train margin | killed | pooled-seats, seat-symmetry, native smoke |
| M2 | seat-0 bank used for both seats in own_bank | killed | 5 tests incl. native smoke and run_ppo integration |
| M3 | swap candidate/opponent lists in eval | killed | run_ppo integration |
| M4 | eval uses seat-order `bank_0`/`bank_1` instead of candidate seat | killed | run_ppo integration (only because the RNG draw put the candidate in seat 1) |
| M5 | eval margin sign flipped | killed | 3 tests |
| M6 | p10 computed at q=0.9 | killed | 3 tests |
| M7 | winner/loser means include draws | killed | pooled-seats test |
| M8 | training skips the cross-rank gather | killed | rank-pooling test |
| M9 | training passes `terminal_bank_0` for both seats | killed | 3 tests |
| M10 | draw_rate counts decisive games | killed | 3 tests |
| M11 | training telemetry emitted for Orbit | killed | Orbit trainer smoke |
| M12 | eval telemetry emitted for Orbit | killed | Orbit eval test |

## Findings

1. **Minor (test robustness) - seat-1 candidate coverage of the new eval wiring
   depends on an unasserted RNG draw.**
   `tests/scripts/test_run_ppo.py::test_kaggriculture_bank_telemetry_reaches_the_logger`
   is the only test that exercises `_evaluate_against_last_best ->
   opponent_bank_metrics`. It catches M4 (seat-order banks instead of candidate
   banks) only because `torch.manual_seed(41)` happens to put the candidate in
   seat 1 in both games. A change to the seed, `n_envs`, or the upstream RNG
   consumption could put the candidate in seat 0 in both games, after which M4
   would survive silently. No test runs a mixed-seat evaluation end to end.
   Edit: pin the assignment (for example, monkeypatch `_assign_eval_models` or
   assert the per-game seats observed) or add a direct unit test feeding
   `_candidate_bank_metrics` for mixed seats into `opponent_bank_metrics`, and
   assert `eval/margin_*` against hand values. The independent check above
   shows the current code is correct.

2. **Minor (throughput, unmeasured) - one extra `all_gather_object` per update on
   multi-rank runs.** `_self_play_bank_metrics` adds a second pickled object
   collective per update beside `_distributed_mean_env_metrics`' existing key-set
   gather. Payload is small (one float pair per completed game), and the
   author's note already lists it as unmeasured. It could be folded into the
   existing gather if it ever appears in an `nsys` update profile. No action
   required now.

3. **Nit (naming) - `eval/margin_*` (candidate minus last-best) sits beside
   `eval/margin_0` (seat-0 mean from the terminal record) and
   `eval/candidate_bank_margin`.** Semantics are correct and documented in
   `docs/rl-api-specs.md` and `README.md`, but dashboards could confuse
   `eval/margin_mean` with `eval/margin_0`. Optional: a W&B panel note, or a
   clearer prefix if a fixed-opponent panel is added.

4. **Nit (custody wording).** The cookbook Reference says "No independent review
   ran in this change." After this review, that line is stale. Update it to cite
   this report with the reviewer caveat (Claude substitute, not Codex). Its
   "nine mutations" count covers the author's set; this review killed 12 more,
   independently.

No correctness defect found in the seat mapping, percentiles, draw handling,
absolute self-play margin, evaluation sign convention, W&B reach, or Orbit
isolation. Nothing reaches model inputs, rewards, losses, normalization, or
promotion.

## Limits

CPU only. A real multi-rank gather (tested only with a stubbed collective),
CUDA, a live W&B upload, and the per-update gather cost at production scale
were not exercised. I did not rerun `just py-prepare`/`docs-fresh`. Ruff, format
and mypy were run directly, along with the four touched test modules.

VERDICT: APPROVE WITH EDITS
