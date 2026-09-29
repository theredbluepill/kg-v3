---
type: "Reference"
title: "Kaggriculture model joins Isaiah's factory, trunk compile and masked critic"
description: "Task 3.1 model side: KaggricultureTransformerConfig is in the shared ModelConfig union and create_model with game-checked specs, the trunk compile target dispatches through a nominal TrunkCompileAPI that compiles only the blocks behind _run_trunk's overflow guard, and the winner softmax uses Isaiah's masked form; CPU tests only, FullConfig still rejects the model until a Kaggriculture env config exists."
tags: ["kaggriculture-v3", "model", "training", "adaptation"]
status: "verified-scoped"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-09-29"}
sources: [{"resource": "repository:python/owl/model/config.py"}, {"resource": "repository:python/owl/model/factory.py"}, {"resource": "repository:python/owl/model/base.py"}, {"resource": "repository:python/owl/model/kaggriculture.py"}, {"resource": "repository:python/owl/model/stateless_transformer_v1.py"}, {"resource": "repository:python/owl/model/__init__.py"}, {"resource": "repository:python/owl/train/utils.py"}, {"resource": "repository:python/owl/train/config.py"}, {"resource": "repository:tests/kaggriculture/test_model_registration.py"}, {"resource": "repository:tests/kaggriculture/test_model_compile.py"}, {"resource": "repository:tests/kaggriculture/test_model_encoder.py"}, {"resource": "repository:tests/owl/model/test_model_config_files.py"}, {"resource": "repository:docs/model-architecture.md"}, {"resource": "repository:docs/kaggriculture-model.md"}, {"resource": "repository:README.md"}, {"resource": "repository:ops/rebuild-2026-09-29/plan.md"}, {"resource": "repository:ops/rebuild-2026-09-29/trainer-model/compile-red.log"}, {"resource": "repository:ops/rebuild-2026-09-29/trainer-model/critic-red.log"}, {"resource": "repository:ops/rebuild-2026-09-29/trainer-model/py-prepare.log"}, {"resource": "repository:ops/rebuild-2026-09-29/trainer-model/isaiah-suites.log"}]
---

# Kaggriculture model joins Isaiah's factory, trunk compile and masked critic

Branch `kg/rebuild-trainer-model`, based on `kg/isaiah-gap-closure` at `e1458d2`. The [[kaggriculture-encoder-reuses-isaiah-stateless-layers|encoder]] and [[kaggriculture-grammar-heads-sit-behind-isaiahs-actor-projection|heads]] References left three items open: shared registration, trainer compile wiring, and the critic's masked softmax. This note records all three.

## What changed

- **Registration.**
  - `python/owl/model/config.py`: `ModelConfig` now has three members, discriminated by `model_arch`: `StatelessTransformerV1Config`, `RecurrentTransformerV1Config` and `KaggricultureTransformerConfig`. The first two are also named `OrbitModelConfig`.
  - `python/owl/model/factory.py`: `create_model` matches on the config class and ends in `assert_never`, as Isaiah's factory does. Each architecture belongs to one game. An Orbit model given Kaggriculture specs raises `TypeError`, and so does the reverse. Overloads give Orbit callers `BaseModelAPI` and Kaggriculture callers `KaggricultureTransformer`. A caller holding the full union gets `BaseModelAPI[Any, Any, Any]`.
  - `python/owl/train/config.py`: `FullConfig.env` is still Orbit's `EnvConfig`, so its validator rejects `kaggriculture_transformer` with an explicit error. It never reads Orbit-only fields (`actor`, `value_mode`, `critic_mode`) from the Kaggriculture config.
- **Trunk compile.**
  - `TrunkCompileAPI` (`python/owl/model/base.py`) is a nominal ABC with `compile_transformer_trunk(*, mode) -> int`. `StatelessTransformerV1` and `KaggricultureTransformer` both implement it.
  - `configure_model_compile` accepts any `nn.Module`. For `trunk` it dispatches with `isinstance` (no `getattr`). The recurrent model keeps Isaiah's specific error, and any other model raises.
  - Kaggriculture compiles only `_forward_transformer_trunk` (blocks plus `final_norm`). `_run_trunk` calls it after the overflow guard and chunking.
  - `mlp` compiles `blocks[i].mlp` in place, and those modules run only inside `_run_trunk`.
- **Critic.** The Task 2.2 brief asked for a masked winner softmax, and Isaiah uses `masked_softmax(logits, still_playing)`. The code used a plain `log_softmax`.
  - `_winner_log_probabilities` now follows Isaiah's `_critic_distillation`. It fills masked logits with the dtype minimum, then applies `log_softmax`.
  - The mask is the critic tokens' own token mask, `KaggricultureEncoded.critic_value_mask`, which is the row's `still_playing` on both tokens.
  - Live rows are unchanged. An all-masked row gets Isaiah's uniform masked result (value 0).
  - **Recorded deviation:** Isaiah's `_critic_logits` raises on a row with no live player; this model does not. The heads already emit empty programs for inactive rows, and contract v4 keeps `still_playing` true. The difference is listed in `docs/kaggriculture-model.md` (resolved toward Isaiah's output, not escalated).

## Checks (this version, owner's Mac, CPU, tiny shapes)

- `tests/kaggriculture/test_model_registration.py` (7 tests):
  - the preset round-trips through the union (Python, dump and JSON)
  - an unknown field is rejected
  - the factory builds the preset on `meta` with 6,252,223 parameters
  - a union-typed config dispatches
  - spec mismatches raise in both directions
  - `FullConfig` rejects the model
- `tests/kaggriculture/test_model_compile.py` (11 tests):
  - the trunk target compiles exactly the bound `_forward_transformer_trunk` with `dynamic=True`
  - over sampling, replay and value calls, only block submodules and `final_norm` run inside the compiled region. Stems, critic, actor projection and heads run outside it.
  - the compiled callable is called once per chunk, and the guard raises before it runs
  - `mlp` compiles exactly the block MLPs, and they run only inside `_run_trunk`
  - no target compiles the whole model
  - an AST pin lists every compile call site on the trainer and model paths
  - Against the base dispatch, 4 of these fail (`ops/rebuild-2026-09-29/trainer-model/compile-red.log`).
- Critic tests in `tests/kaggriculture/test_model_encoder.py`:
  - the mask is the critic token mask
  - with distinct non-zero critic tokens on every row, the probabilities equal Isaiah's `masked_softmax`, live rows equal the unmasked softmax, and finished rows are uniform with value 0
  - reverting to a plain `log_softmax` fails the test (`critic-red.log`)
- `just py-prepare` (`uvx --from rust-just just py-prepare`) passes: Ruff, mypy, 1,324 passed with 4 hardware skips, and docs freshness (`py-prepare.log`).
- Isaiah's suites, `pytest tests/owl tests/scripts tests/tools -m "not slow"`: 1,048 passed, 3 skipped (`isaiah-suites.log`). The plan's `--extra reference` flag no longer exists in this tree, so it was dropped. No Rust changed, so `cargo test` was not run.

## Limits

- Nothing here runs a real `torch.compile` or CUDA. Every compile test uses a recording stand-in, so Inductor graph breaks, recompiles and the flash path in the compiled trunk remain Phase 6 GPU qualification.
- The trainer cannot build this model yet, because `FullConfig` has no Kaggriculture env config. That seam, rollout storage and the Kaggriculture observation mapping stay open in Task 3.1. When it lands, replace the `FullConfig` rejection with the env pairing.
- Reopen the no-raise critic choice if a real batch can carry a row with `still_playing = False`. Contract v4 says it cannot.
