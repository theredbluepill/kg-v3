"""Phase 6.0 smoke: flash-attn 2.8.3 kernel and the model's packed flash path.

Run from the v3 checkout root on the pod with CUDA_VISIBLE_DEVICES=0:
    .venv/bin/python <this> --out <dir>
Part (a): flash_attn_varlen_func vs per-sequence SDPA on packed BF16 q/k/v.
Part (b): KaggricultureTransformer (preset config, force_flash_attn=True) on a
contract-valid 256-row batch: eager flash, compiled flash, eager padded SDPA.
Run-statement: ops/rebuild-2026-09-29/run-statements/pod-flash-attn-setup.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from pydantic import BaseModel

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

TOL_ABS, TOL_REL = 0.02, 0.02


def _diff(a: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    a, ref = a.float(), ref.float()
    d = (a - ref).abs()
    outside = d > TOL_ABS + TOL_REL * ref.abs()
    return {
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "ref_max_abs": float(ref.abs().max()),
        "frac_outside_tol": float(outside.float().mean()),
        "n": int(d.numel()),
        "finite": bool(torch.isfinite(a).all()),
    }


def _kernel_names(prof: torch.profiler.profile) -> list[str]:
    names = {
        e.key
        for e in prof.key_averages()
        if e.device_type == torch.autograd.DeviceType.CUDA
    }
    return sorted(names)


def part_a(device: torch.device) -> dict[str, Any]:
    from flash_attn import flash_attn_varlen_func

    g = torch.Generator().manual_seed(0)
    heads, head_dim, n_seq = 8, 32, 256
    lengths = [214, 709] + torch.randint(650, 710, (n_seq - 2,), generator=g).tolist()
    total = sum(lengths)
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    q, k, v = (
        torch.randn((total, heads, head_dim), generator=g).to(device, torch.bfloat16)
        for _ in range(3)
    )
    cu = cu.to(device)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        out = flash_attn_varlen_func(
            q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
            max_seqlen_q=max(lengths), max_seqlen_k=max(lengths),
            dropout_p=0.0, causal=False,
        )  # fmt: skip
        torch.cuda.synchronize()
    ref32 = torch.empty_like(out, dtype=torch.float32)
    ref16 = torch.empty_like(out)
    for i in range(n_seq):
        s, e = int(cu[i]), int(cu[i + 1])
        qi, ki, vi = (t[s:e].transpose(0, 1).unsqueeze(0) for t in (q, k, v))
        ref32[s:e] = (
            F.scaled_dot_product_attention(qi.float(), ki.float(), vi.float())
            .squeeze(0)
            .transpose(0, 1)
        )
        ref16[s:e] = (
            F.scaled_dot_product_attention(qi, ki, vi).squeeze(0).transpose(0, 1)
        )
    kernels = _kernel_names(prof)
    return {
        "heads": heads,
        "head_dim": head_dim,
        "n_seq": n_seq,
        "total_tokens": total,
        "min_len": min(lengths),
        "max_len": max(lengths),
        "out_dtype": str(out.dtype),
        "flash_vs_sdpa_fp32": _diff(out, ref32),
        "sdpa_bf16_vs_sdpa_fp32_noise_baseline": _diff(ref16, ref32),
        "flash_vs_sdpa_bf16": _diff(out, ref16),
        "flash_kernels": [n for n in kernels if "flash" in n.lower()],
    }


def _to(obj: BaseModel, device: torch.device) -> BaseModel:
    fields: dict[str, Any] = {}
    for name in type(obj).model_fields:
        value = getattr(obj, name)
        if isinstance(value, torch.Tensor):
            fields[name] = value.to(device)
        elif isinstance(value, BaseModel):
            fields[name] = _to(value, device)
        else:
            fields[name] = value
    return type(obj)(**fields)


def part_b(device: torch.device, out_dir: Path) -> dict[str, Any]:
    from owl.kaggriculture import types as kt
    from owl.model import kaggriculture as km
    from owl.model.attn import flash_attn_available, use_flash_attn
    from owl.model import stateless_transformer_v1 as stv1
    from owl.train.utils import configure_torch

    from tests.kaggriculture.conftest import make_obs

    configure_torch()  # TF32 on, as run_ppo
    assert flash_attn_available(), "flash-attn not importable"
    config = km.KaggricultureTransformerConfig.from_file(
        ROOT / "configs/model/kaggriculture.yaml"
    )
    assert config.force_flash_attn is True
    torch.manual_seed(0)
    model = km.KaggricultureTransformer(
        config,
        obs_spec=kt.KaggricultureObsConfig(),
        action_spec=kt.KaggricultureActionConfig(),
    ).to(device)
    model.eval()
    g = torch.Generator().manual_seed(1)
    envs = 128
    own = [1, 241] + torch.randint(215, 242, (envs - 2,), generator=g).tolist()
    rival = [1, 241] + torch.randint(215, 242, (envs - 2,), generator=g).tolist()
    shops = [0, 8] + torch.randint(0, 9, (envs - 2,), generator=g).tolist()
    t0 = time.time()
    obs = make_obs(envs, own_actors=own, rival_actors=rival, shops=shops)
    make_obs_s = time.time() - t0
    obs = _to(obs, device)
    assert isinstance(obs, kt.KaggricultureObsBatch)
    counts = model.count_non_masked_tokens(obs)

    pack_calls: list[tuple[int, int]] = []
    real_pack = km.pack_sequence

    def counting_pack(x: torch.Tensor, mask: torch.Tensor, **kw: Any) -> Any:
        px, packed = real_pack(x, mask, **kw)
        pack_calls.append((int(px.shape[0]), int(packed.max_seqlen)))
        return px, packed

    km.pack_sequence = counting_pack
    varlen_calls = [0]
    real_varlen = stv1.varlen_attention

    def counting_varlen(*a: Any, **kw: Any) -> torch.Tensor:
        varlen_calls[0] += 1
        return real_varlen(*a, **kw)

    stv1.varlen_attention = counting_varlen

    def run(tag: str) -> tuple[torch.Tensor, torch.Tensor, list[str], float]:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            x, mask = model._assemble_tokens(obs)
            flash_flag = use_flash_attn(x)
            torch.cuda.synchronize()
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA]
            ) as prof:
                t = time.time()
                enc = model.encode_observations(obs)
                torch.cuda.synchronize()
                dt = time.time() - t
        info[f"{tag}_use_flash_attn_x"] = flash_flag
        info[f"{tag}_x_dtype"] = str(x.dtype)
        return enc.hidden, enc.token_mask, _kernel_names(prof), dt

    info: dict[str, Any] = {
        "config": config.model_dump(),
        "rows": int(obs.still_playing.numel()),
        "make_obs_s": make_obs_s,
        "present_tokens_total": int(counts),
        "padded_seq_len": km.sequence_length(config),
    }
    # eager flash
    h_eager, mask, k_eager, t_eager = run("eager_flash")
    seqlens = mask.sum(1)
    info["seqlen_min"] = int(seqlens.min())
    info["seqlen_max"] = int(seqlens.max())
    info["seqlen_total"] = int(seqlens.sum())
    info["eager_flash"] = {
        "pack_calls": list(pack_calls),
        "varlen_attention_python_calls": varlen_calls[0],
        "flash_kernels": [n for n in k_eager if "flash" in n.lower()],
        "sdpa_like_kernels": [
            n for n in k_eager if "fmha" in n.lower() or "efficient" in n.lower()
        ],
        "wall_s_profiled": t_eager,
    }
    pack_calls.clear()
    varlen_calls[0] = 0
    # restore before compiling so the counter cannot add graph breaks
    stv1.varlen_attention = real_varlen

    # compiled flash (first call compiles + autotunes)
    assert model.compile_transformer_trunk(mode="max-autotune-no-cudagraphs") == 1
    t = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        h_c0 = model.encode_observations(obs).hidden
    torch.cuda.synchronize()
    compile_s = time.time() - t
    pack_calls.clear()
    h_comp, _, k_comp, t_comp = run("compiled_flash")
    info["compiled_flash"] = {
        "first_call_compile_s": compile_s,
        "pack_calls": list(pack_calls),
        "flash_kernels": [n for n in k_comp if "flash" in n.lower()],
        "triton_kernel_count": sum(1 for n in k_comp if n.startswith("triton")),
        "wall_s_profiled": t_comp,
        "repeat_consistency_max_abs": float((h_c0 - h_comp).float().abs().max()),
    }
    pack_calls.clear()
    # compiled at a second (smaller) shape to exercise dynamic=True
    sub = _to(
        make_obs(32, own_actors=own[:32], rival_actors=rival[:32], shops=shops[:32]),
        device,
    )
    assert isinstance(sub, kt.KaggricultureObsBatch)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        hs_c = model.encode_observations(sub)
        compiled_trunk = model._compiled_transformer_trunk
        model._compiled_transformer_trunk = None
        hs_e = model.encode_observations(sub)
        model._compiled_transformer_trunk = compiled_trunk
    m = hs_e.token_mask
    info["compiled_vs_eager_64rows"] = _diff(hs_c.hidden[m], hs_e.hidden[m])

    # padded SDPA reference (same weights, flash dispatch disabled, eager trunk)
    real_use, real_req = km.use_flash_attn, km._requires_flash_attn
    km.use_flash_attn = lambda *_: False
    km._requires_flash_attn = lambda *_, **__: False
    model._compiled_transformer_trunk = None
    pack_calls.clear()
    h_pad, mask_pad, k_pad, _ = run("eager_padded")
    info["eager_padded"] = {
        "pack_calls": list(pack_calls),
        "flash_kernels": [n for n in k_pad if "flash" in n.lower()],
    }
    km.use_flash_attn, km._requires_flash_attn = real_use, real_req
    assert torch.equal(mask, mask_pad)
    info["compiled_vs_eager_flash"] = _diff(h_comp[mask], h_eager[mask])
    info["eager_flash_vs_eager_padded_sdpa"] = _diff(h_eager[mask], h_pad[mask])
    info["compiled_flash_vs_eager_padded_sdpa"] = _diff(h_comp[mask], h_pad[mask])
    info["masked_positions_zero"] = bool(
        (h_eager[~mask] == 0).all() and (h_comp[~mask] == 0).all()
    )
    km.pack_sequence = real_pack
    (out_dir / "kernels_eager_flash.txt").write_text("\n".join(k_eager) + "\n")
    (out_dir / "kernels_compiled_flash.txt").write_text("\n".join(k_comp) + "\n")
    (out_dir / "kernels_eager_padded.txt").write_text("\n".join(k_pad) + "\n")
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda", 0)
    result: dict[str, Any] = {
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "tolerance": f"|d| <= {TOL_ABS} + {TOL_REL}|ref|",
    }
    t = time.time()
    result["a_kernel"] = part_a(device)
    result["a_wall_s"] = time.time() - t
    print(json.dumps(result["a_kernel"], indent=2), flush=True)
    t = time.time()
    result["b_model"] = part_b(device, args.out)
    result["b_wall_s"] = time.time() - t
    result["max_memory_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
    (args.out / "smoke_flash.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["b_model"], indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
