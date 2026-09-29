"""Read-only independent oracle and failure-path probes for 530b8cc (r1 findings repaired)."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

ROOT = Path.cwd()
OUT = ROOT / 'ops/rebuild-2026-09-29/codex/verify-3.2-3.3-r2'
spec = importlib.util.spec_from_file_location('review_test_run_ppo', ROOT / 'tests/scripts/test_run_ppo.py')
assert spec is not None and spec.loader is not None
t = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = t
spec.loader.exec_module(t)
r = t.run_ppo

reference = subprocess.check_output(['git', 'show', 'kg/reference-2026-09-29:scripts/run_ppo.py'], text=True)
node = next(n for n in ast.parse(reference).body if isinstance(n, ast.FunctionDef) and n.name == '_evaluation_outcome')
module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
ns = {'torch': torch, 'KaggricultureObsConfig': t.KaggricultureObsConfig}
exec(compile(ast.fix_missing_locations(module), '<literal-reference-evaluation-outcome>', 'exec'), ns)
cases = 0
for banks in [(3000., 2000.), (2000., 3000.), (2500., 2500.), (2.**40 + .5, 2.**40 + .25)]:
    metrics = {'bank_0': banks[0], 'bank_1': banks[1], 'margin_0': banks[0] - banks[1]}
    returns = torch.tensor([-0.4, 0.4])
    cfg = t._kaggriculture_eval_config()
    expected = ns['_evaluation_outcome'](cfg, metrics, returns)
    for assignment in (torch.tensor([0, 1]), torch.tensor([1, 0])):
        actual, _ = r._evaluation_scores_and_metrics(cfg, metrics, returns, assignment)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        cases += 1
orbit_returns = torch.tensor([1., -1., 0., 0.])
assert ns['_evaluation_outcome'](t._full_config(), {}, orbit_returns) is orbit_returns
assert r._evaluation_scores_and_metrics(t._full_config(), {}, orbit_returns, torch.tensor([0, 1, -1, -1]))[0] is orbit_returns

seed_pairs = [(0, 0), (1, 2131737497183550101), (0, 787325655728545358)]
seeds = [r._evaluation_seed(base_seed=b, env_steps=s) for b, s in seed_pairs]
assert seeds[0] == seeds[1] and seeds[2] == seeds[0] + 1

failures = []
for phase in ('refresh', 'checkpoint'):
    class Trainer(t._FakeTrainer):
        def write_checkpoint(self, path, *, env_steps, wandb_run_id=None, model=None):
            if phase == 'checkpoint' and path.name == 'checkpoint_last_best.pt' and env_steps > 0:
                raise OSError('injected last-best checkpoint write failure')
            super().write_checkpoint(path, env_steps=env_steps, wandb_run_id=wandb_run_id, model=model)
    trainer = Trainer()
    logger = t._FakeLogger()
    with pytest.MonkeyPatch.context() as patch:
        t._patch_eval_model_from_weights(patch)
        patch.setattr(r, '_evaluate_against_last_best', lambda **kwargs: {'eval/win_rate_against_last_best': .7, 'eval/games': 2.})
        if phase == 'refresh':
            def fail_refresh(*args, **kwargs):
                raise RuntimeError('injected incumbent refresh failure')
            patch.setattr(r, '_refresh_eval_model_from_weights', fail_refresh)
        try:
            r._run_training_loop(trainer=trainer, logger=logger, run_dir=OUT / 'no-checkpoint-written', cfg=t._full_config(checkpoint_freq=1000), env_steps_per_iteration=1000, max_env_steps=1000, max_runtime_seconds=None, dist_ctx=t.DistributedContext.single_process_cpu())
        except (RuntimeError, OSError) as exc:
            error = str(exc)
        else:
            raise AssertionError('fault was not exercised')
    promotion_logs = [m for m, _ in logger.logged if 'eval/promoted' in m]
    saved = [(p.name, s) for p, s, _ in trainer.checkpoints if p.name == 'checkpoint_last_best.pt' and s > 0]
    assert promotion_logs == [] and saved == []
    failures.append({'phase': phase, 'error': error, 'logged': promotion_logs, 'completed_promoted_checkpoints': saved})

results = {
    'source_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'reference_sha': subprocess.check_output(['git', 'rev-parse', 'kg/reference-2026-09-29'], text=True).strip(),
    'isaiah_sha': subprocess.check_output(['git', 'rev-parse', '32b3ec9'], text=True).strip(),
    'oracle': {'literal_function': '_evaluation_outcome', 'reference_source_sha256': hashlib.sha256(reference.encode()).hexdigest(), 'kaggriculture_exact_cases': cases, 'orbit_identity_preserved': True},
    'seed_counterexamples': [{'base_seed': b, 'env_steps': s, 'seed': v} for (b, s), v in zip(seed_pairs, seeds, strict=True)],
    'promotion_failures': failures,
}
(OUT / 'probe.json').write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps(results, indent=2))
