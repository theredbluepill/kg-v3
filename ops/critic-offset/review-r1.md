Reviewer: independent Claude subagent (substitute for Codex during its usage limit; owner-approved). Not a Codex verdict.

# Review r1: per-seat critic offset head (`model.critic_offset`) and the bank-critic preset

- **Target.** `kg/rebuild-critic-offset` at HEAD `732438c`, compared with base `3e89425`. One commit: 25 files, +2459/−22.
- **Request context.** Owner, verbatim, 2026-09-30: "per-player critic might be the way out?". Earlier: "let's switch back to self play no matter what, and think about how do we get the agent to earn moneny for real?". The design sources are `kg-v3-int/ops/earn-money-2026-09-30/plan.md` §4 F4, `design-signal.md` C2 and `critique-signal.md`. I read all three.
- **Method.**
  - I worked in scratch detached worktrees: `/private/tmp/claude-501/critic-review-r1` at `732438c` and `/private/tmp/claude-501/critic-review-base` at `3e89425`.
  - Both used the existing `.venv` interpreter, with `PYTHONPATH` pointed at each worktree's `python/`, and a copy of the built `owl.rs` extension. The branch has no Rust change.
  - I did not edit `/Users/poonszesen/kg-v3-critic`. The only write there is this uncommitted report. I ran fast tests only.

## Checks run (this version, this machine, Apple Silicon CPU)

| Check | Result |
|---|---|
| Default-off byte identity: `ops/critic-offset-2026-09-30/baseline_digest.py`, re-run independently on base and HEAD at `OMP_NUM_THREADS=2` | native `257eae38…`, trainer `3ffd53a0…` and model `1e4690be…` are identical on base and HEAD. The config hash and `config.yaml` bytes are identical for all 14 existing presets. The only difference is the new preset's line. My base run matches the committed `baseline-digest-3e89425.json` byte for byte |
| The same trainer and model digest at `OMP_NUM_THREADS=1` | base = HEAD (trainer `2fd98367…`, model `1e4690be…`). The flag-off identity holds at a second thread count |
| Step-0 identity through one full update. Tiny model and native env with term A; off model against on model (detached head) holding the off weights; SGD; clipping disabled | Shared-parameter update is identical (Σ\|Δ\| 1.75947999 on both). The zero head's out-layer gradient sends nothing back into the trunk |
| `pytest tests/kaggriculture tests/scripts/test_run_ppo.py` | 1575 passed, 4 skipped (CUDA and BC-data skips) |
| `pytest tests/owl` | 933 passed, 3 skipped |
| `mypy` on `kaggriculture.py`, `lora.py` and `ppo.py`; `ruff check` / `ruff format --check` on `python/owl`, `scripts`, `tests` | clean. Ruff finds 2 nits in `ops/` scripts, which are outside `all_py_code` |
| Cookbook lint hook on the new Decision and the revised term-A Decision | no output (pass). Every `repository:` source path exists. The m17 figure (82.0k → 86.8k) matches `evidence-v2.md:91` |
| Author's receipts `mutations.log` (C1-C10) and `prepare.log` (2969 passed, 9 skipped) | read, not re-run, apart from the overlap below |

## Mutations (8 of my own, independent of the author's C1-C10; 7 killed, 1 survived)

Run against `test_critic_offset.py`, `test_configs.py` and `test_run_ppo.py::test_critic_offset_with_the_own_bank_reward_two_update_run_through_main` (165 tests). Every mutation was reverted afterwards, and `git status` is clean.

| # | Mutation | Result |
|---|---|---|
| R1 | `compute_value` (the horizon `last_values` and truncation bootstrap) returns the winner value only, without the offset | **SURVIVED** (165 passed). See P2-1 |
| R2 | `_evaluation_from` (the PPO update's `new_values`, plain and teacher paths) drops the offset | KILLED (`test_two_updates_with_the_head_and_the_own_bank_term`, the run_ppo test) |
| R3 | validator removed (`critic_offset_detach_trunk` accepted without `critic_offset`) | KILLED (`test_default_config_dumps_exactly_as_before_the_fields`) |
| R4 | serializer always omits the fields, so `config.yaml` loses the flag | KILLED (3 tests, including the strict round trip and the run_ppo reload) |
| R5 | `ev_common` scored over steps with any seat, not both | KILLED (2 tests) |
| R6 | `critic_offset_head.out` left out of `get_output_layers` (it would go to Muon) | KILLED |
| R7 | trainer never detects the head, so there is no offset buffer | KILLED (4 tests) |
| R8 | head always detached, whatever the flag says | KILLED (`test_detach_trunk_blocks_the_heads_trunk_gradient[False]`) |

With the author's 10, 17 of 18 source mutations are killed.

## Review of the requested properties

- **Default-off byte identity.** Holds (above). With the flag off, `critic_offset_head` is `None`. `_value_parts` computes the same `2·exp(logp)−1` expression. `ModelOutput.value_offsets` defaults to `None`. The wrap serializer deletes both fields, so `config_sha256` and `config.yaml` bytes are unchanged. No offset buffer is allocated, and no new metric key appears.
- **Zero-init identity at step 0.** `reset_parameters` skips the head's out layer in the gain loop and then zeroes its weight and bias (`kaggriculture.py:347-359`). The hidden layer keeps a nonzero init, so the head can train. Values, actions, log-probs, `evaluate_actions` and `compute_value` are bit-identical (test at `test_critic_offset.py:113`). A full update leaves the shared parameters identical when clipping does not bind (measured above).
- **The value sum is used consistently.** All of these route through `_value_parts` / `_values`:
  - rollout (`forward`, `kaggriculture.py:657`);
  - update and teacher replay (`_evaluation_from`, `:707`, which `_teacher_evaluation` also uses);
  - bootstrap (`compute_value`, `:1209`, used at `ppo.py:1198` for `last_values` and `ppo.py:1261` for truncation);
  - learner-row scatter (`ppo.py:3472-3479`).

  GAE (`segments.values`), the MSE value loss, `explained_variance` and the `value_clip_anchor` all see the sum. `vf_clip_coef` is null in the preset. Kaggriculture refuses `winner_ce`, so no path treats `values` as a probability. Correct by inspection, but the bootstrap leg is unprotected by tests (P2-1).
- **Teacher value distillation stays well-defined.** The student's `winner_log_probabilities` are compared with the teacher's winner softmax. The cached path reads `teacher_targets.winner_probabilities`; the combined path calls `teacher._winner_log_probabilities`. The offset is never in the CE, so it is well-defined whether or not either side has a head.
- **`detach_trunk`.** `own.detach()` on `critic_value_hidden[:, 0]` before the head (`:1196-1197`) is correct for gradient flow. The test checks that no trunk parameter gets a gradient with the detach and that some do without it (R8 and C1 both killed). However, the head's gradient still enters the global clip norm (P2-2).
- **Loader rule, in both directions.**
  - `_admit_missing_optional_state` (`lora.py:148-167`) admits only the case where every `critic_offset_head.*` key is missing, and then zeroes the out layer.
  - A partial head fails, any other missing key fails, and an unexpected key fails.
  - A head checkpoint loaded into a model without the head fails on its unexpected keys.
  - `model_and_optimizer` from a headless checkpoint fails on the optimizer param count (tested).
  - Resume loads the model strictly (`ppo.py:1005`), so a headless resume into a head config fails. That is correct, but the docs say otherwise (P3-1).
- **last_best, promotion and teacher architectures match.**
  - The initial last-best is either `_create_eval_model_from_weights(student)` (`run_ppo.py:396`) or `_initial_last_best_model`, which builds from the student config and loads a headless `teacher_init` through the admitting loader (`run_ppo.py:1258-1263`).
  - The promotion refresh (`run_ppo.py:1509`) and the eval copy (`:1499`) do a strict `load_state_dict` between same-architecture models.
  - A fixed teacher keeps its own config, and only its winner softmax is read.
  - The run_ppo test asserts that the teacher has the head and that every checkpoint reloads strictly from `config.yaml`. All models are built on a real device with `reset_parameters`, so an admitted headless load never leaves the head's hidden layer uninitialised.
- **No opponent identity and no cross-seat leakage.** The head reads only token 0 (self) of the row's own critic pair, and the test pins the hook input to `critic_value_hidden[:, 0]`. A single-env batch reproduces the first rows, so rows are independent. R5, C5 and C9 cover the related telemetry and routing. The row's self token attends to its own current observation only, which the stateless-policy Decision allows. There is no identity input, and the mix label never reaches the head.
- **Kaggle packaging is unaffected.**
  - No agent or packaging file changed.
  - The policy is independent of the head: randomised head weights leave actions and log-probs bit-identical.
  - The Orbit `Agent` builds from `config.yaml` and loads strictly (`agent.py:180-236`), and `config.yaml` records `critic_offset: true` when it is on (R4 killed).
  - A Kaggriculture agent is still design-only (Task 7.4).
  - Minor note: `checkpoint_quantization._is_critic_tensor_name` (`:229-230`) does not match `critic_offset_head.*`, so the structured NF3/NF4 format would spend a little of its NF4 budget on the unused head. That is harmless to actions.
- **The preset's effective reward and `terminal_scale`.**
  - `kaggriculture_4rank_bank_critic.yaml` differs from `kaggriculture_4rank_margin.yaml` only in the six `econ_bank_*` / `econ_margin_*` values and in `model: kaggriculture_critic_offset`. I checked this with a comment-stripped diff, and a test pins it.
  - The term weights are .25/150k/.25 for bank, .25/100k/.25 for margin and 0 for `econ_shaping`, so `terminal_scale` = .5 (asserted).
  - The return from reset lies in about [−0.755, 0.995]: B starts at ≈.005 and is capped at .25, M is ±.25 and the sign term is ±.5. The header states this correctly.
  - The model yaml sets `critic_offset_detach_trunk: false`, which departs from plan F4 (P3-2).
- **Docs and the cookbook contract.**
  - The new Decision has all six fields, the first tag is `kaggriculture-v3`, and the decider is quoted with its relay caveat. It labels the interpretation as not owner adoption and carries an adaptation inventory, actual checks and gaps.
  - The index and a prepended log entry are present.
  - The correction to the term-A Decision is sound: the two seat rows are separate softmaxes, so V0 + V1 = 0 is not structural.
  - There is no board yet, so no fourth edit is due.
  - Inaccuracies: P3-1 and P3-3.

## Findings

### P1

None.

### P2

- **P2-1. The bootstrap leg of "the value sum everywhere" is untested** (`python/owl/model/kaggriculture.py:1209`; `tests/kaggriculture/test_critic_offset.py:131`).
  - **Problem.** Mutation R1 makes `compute_value` return only `2p−1`. All 165 targeted tests still pass. That function feeds `last_values` at every horizon cut (`ppo.py:1198`; 64-turn segments at γ = 1, λ = .9) and the truncation bootstrap (`ppo.py:1261`). The only assertion on `compute_value` runs at a zero head, where the mutant behaves identically.
  - **Why it matters.** With the offset missing from the bootstrap, every segment's tail advantage silently loses the common-mode continuation value. That is exactly the bias this head exists to remove. `ev_common` and the loss would still look healthy.
  - **Current code.** Correct by inspection.
  - **Fix.**
    - In `test_value_is_the_winner_value_plus_the_offset_and_actions_ignore_it`, assert with the randomised head that `on.compute_value(obs) == a.values + offsets`, and that `on.evaluate_actions(obs, a.actions).values` equals the same sum.
    - Optionally, in `test_two_updates_…`, assert that the `last_values` returned by `_collect_rollout()` equals the winner value plus `compute_value`'s offset.
- **P2-2. The detached arm does not keep M5 out: the head's gradient still shares the global clip** (`python/owl/train/ppo.py:1697-1698`; `configs/kaggriculture_4rank_bank_critic.yaml:32-34`; Decision "Gaps" bullet).
  - **Problem.** `clip_grad_norm_(self.model.parameters(), max_grad_norm)` includes the offset head's parameters. With `critic_offset_detach_trunk: true`, the head's value gradient still enlarges the total norm, so every shared (trunk, actor, winner) update shrinks whenever clipping binds.
  - **Measured.** One update with the tiny model, native env with term A, SGD at lr 1e-2 and an off model against an on model (detached) holding the same weights:

    | | Unclipped | clip 1e-3 |
    |---|---|---|
    | Shared-parameter Σ\|Δ\|, off | 1.75947999 | 3.8409e-4 |
    | Shared-parameter Σ\|Δ\|, on (detached) | 1.75947999 | 3.0341e-4 (−21%) |
    | Grad norm | 4.486 off / 5.643 on | same |

  - **Why it matters.** Plan F4 (`plan.md:237`) asks for the first arm to "stop the head's gradient from reaching the trunk, to keep M5 out of the comparison". M5 (`plan.md:44`) is defined partly by the clip: norms of 13-15 clipped to 10 in 66-100% of iterations. The critique (`critique-signal.md:79-81`) names "the share of the clipped norm … left for the policy" as the confound. The recipe's own `max_grad_norm` is 10. Muon's orthogonalisation and AdamW's normalisation weaken the effect, but they do not remove it: per-step clip factors reweight momentum and the second moments. So a detached arm read as "M5-free" would be over-read.
  - **Fix (cheapest first).**
    - Log the head's share, for example `optimizer/grad_norm_critic_offset` beside `optimizer/grad_norm`.
    - When detached, clip the head's parameters separately, or exclude them from the global norm, so the shared update is exactly the flag-off one.
    - State the residual in the Decision's gaps and in `docs/rl-api-specs.md`'s `critic_offset_detach_trunk` bullet.

### P3

- **P3-1. The docs say resume goes through the admitting loader; it doesn't** (`docs/rl-api-specs.md:1269-1270`, "every `--load-model-weights` mode, `rl.teacher_init`, last-best and resume"; the Decision's "Loader rule" bullet). The resume model load is a strict `load_state_dict` (`ppo.py:1005`). Only the resume's last-best load (`run_ppo.py:360`) uses `load_model_state_dict_allowing_lora`. The behaviour is stricter than documented, which is fine, but the wording should say "the resume's last-best".
- **P3-2. The preset's attached trunk departs from the design source, and its provenance is vague.**
  - Plan F4 (`plan.md:237`) and the critique (`critique-signal.md:120`) both make the first arm detached.
  - The preset's model yaml (`configs/model/kaggriculture_critic_offset.yaml:18`) sets `false`.
  - The Decision (`add-a-per-seat-critic-offset-for-the-own-bank-reward.md:86`) says "as instructed" without naming who instructed it, and the owner's quoted words do not specify it.

  Name the instruction's source, or make the preset detached to match F4 (with P2-2 addressed, or its residual stated).
- **P3-3. "FP32 output" overstates the precision** (`kaggriculture.py:1185` docstring; `docs/model-architecture.md:791`; the Decision's "Head" bullet).
  - Under the presets' `dtype: bfloat16` autocast, the head's Linear layers compute and round in bf16. Measured: `critic_offset_head(...)` returns `torch.bfloat16`, and only `.float()` afterwards makes it FP32.
  - At \|o\| ≈ .125-.25 the bf16 step is ≈ 1e-3. A 1k bank change pays ≈ .0017 under term A.
  - The winner head has the same property, so this is not a new error class.

  Either say "cast to FP32", or run the head's out layer outside autocast, which is cheap. Consider measuring the effect on δ before relying on small purchase credits.
- **P3-4. The head shifts the global RNG stream.** Head construction and init draw random numbers. Measured: `torch.rand(1)` after `_tiny(seed=5)` gives 0.59498, and 0.62053 with the head. The step-0 identity is therefore of deterministic outputs on fixed weights, not of a launch's sampled rollouts: an A/B launch against a flag-off twin at the same seed will not share trajectories. Document this, or build the head under `torch.random.fork_rng()`.
- **P3-5. Nits.**
  - `ruff check` flags `ops/critic-offset-2026-09-30/baseline_digest.py:26` (unused `noqa`) and `mutations.py:37` (E501). Both are outside `just prepare`'s `all_py_code`.
  - `_is_critic_tensor_name` does not cover the offset head (see Kaggle above).

## Residual unknowns

- Nothing is trained. These remain unmeasured:
  - whether the offset learns the common mode at the preset's `adamw_lr` 5e-6 on its out layer (the Decision records this);
  - the winner/offset split;
  - the purchase-row δ;
  - any own-bank effect.
- DDP, CUDA, `torch.compile` and bf16 paths were not exercised here (CPU only). The head sits outside the compiled trunk in eager code, and DDP sees it used in every forward.
- The author's full `just prepare` result (`prepare.log`) was not re-run in full. I re-ran the targeted suites, `tests/owl`, mypy on the changed modules and ruff.

VERDICT: APPROVE
