import argparse, contextlib, hashlib, importlib.util, json, os, platform, statistics, sys, time
from pathlib import Path
import torch
from owl.kaggriculture.env import KaggricultureVectorizedEnv
from owl.kaggriculture.types import (
    KaggricultureObsBatch,
    KaggricultureActionMask,
    KaggricultureObsConfig,
    KaggricultureActionConfig,
)

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--model-source", default="python/owl/model/kaggriculture.py")
args = ap.parse_args()
spec = importlib.util.spec_from_file_location(
    "owl.model.kaggriculture_benchmark", args.model_source
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
KaggricultureTransformer, KaggricultureTransformerConfig = (
    module.KaggricultureTransformer,
    module.KaggricultureTransformerConfig,
)
torch.set_num_threads(1)
torch.manual_seed(20260928)
env = KaggricultureVectorizedEnv(n_envs=8, pin_memory=False, seed=73)
base = env.reset()
model = KaggricultureTransformer(
    KaggricultureTransformerConfig(),
    obs_spec=KaggricultureObsConfig(),
    action_spec=KaggricultureActionConfig(),
)
result = {
    "torch": torch.__version__,
    "platform": platform.platform(),
    "cpu_threads": 1,
    "n_envs": 8,
    "perspectives": 16,
    "model_parameters": sum(p.numel() for p in model.parameters()),
    "model_sha256": hashlib.sha256(Path(args.model_source).read_bytes()).hexdigest(),
    "warmups": 2,
    "measured_repeats": 5,
    "scenarios": [],
}
for actors, label in [(1, "native_reset"), (16, "synthetic_16_actors")]:
    obs = KaggricultureObsBatch(
        features=base.features.clone(),
        context=base.context.clone(),
        entity_mask=base.entity_mask.clone(),
        still_playing=base.still_playing.clone(),
        action_mask=KaggricultureActionMask(base.action_mask.can_act.clone()),
    )
    if actors != 1:
        obs.context[:, 1:3] = actors
        obs.features[..., 1025:1027] = actors
        raw = obs.features[..., 3827:5273].reshape(8, 2, 2, 241, 3)
        raw.zero_()
        raw[..., :actors, 0] = 1
        raw[..., :actors, 1] = torch.arange(actors) % 10
        raw[..., :actors, 2] = (torch.arange(actors) // 10) % 10
        obs.entity_mask.copy_(torch.arange(241)[None, None, :] < actors)
        obs.action_mask.can_act.copy_(torch.arange(252)[None, None, :] < actors + 11)
    with torch.no_grad():
        saved = model(obs).actions
    metrics = {}
    for name, run, scope in [
        ("rollout_forward", lambda: model(obs), torch.no_grad),
        (
            "evaluate_autograd",
            lambda: model.evaluate_actions(obs, saved),
            contextlib.nullcontext,
        ),
    ]:
        for _ in range(2):
            with scope():
                value = run()
            del value
        timings = []
        for _ in range(5):
            start = time.perf_counter()
            with scope():
                value = run()
            timings.append((time.perf_counter() - start) * 1000)
            del value
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as prof:
            with scope():
                value = run()
            del value
        counts = {e.key: e.count for e in prof.key_averages()}
        metrics[name] = {
            "mean_ms": statistics.mean(timings),
            "min_ms": min(timings),
            "max_ms": max(timings),
            "scalar_extractions": counts.get("aten::_local_scalar_dense", 0),
            "arange_calls": counts.get("aten::arange", 0),
        }
    result["scenarios"].append(
        {
            "label": label,
            "actors_per_seat": actors,
            "saved_mean_frames": saved.lengths.float().mean().item(),
            "saved_max_frames": saved.lengths.max().item(),
            "metrics": metrics,
        }
    )
    print(label, metrics, flush=True)
Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
print(args.out)
