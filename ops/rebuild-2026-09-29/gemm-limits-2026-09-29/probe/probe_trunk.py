"""Real KaggricultureTransformer trunk limit probe (one path per process).

Imports the model from a copy of python/owl at kg/rebuild-model (PYTHONPATH),
builds the default config (D=256, depth 8, 8 heads, mlp_ratio 2.0, 709 tokens),
fp32 parameters under torch.autocast(bfloat16) with TF32 on, exactly as
run_ppo's autocast_context/configure_torch do, and compiles the trunk with the
model's own compile_transformer_trunk(mode="max-autotune-no-cudagraphs"),
which uses dynamic=True. The trunk input x is bf16 [rows, 709, 256].

--path packed: force_flash_attn=True. The pod venv has no flash-attn package,
so owl.model.attn._FLASH_ATTN_VARLEN_FUNC is replaced by torch's own varlen
aten flash (or efficient) attention kernel. Only the attention kernel differs;
every GEMM, the pack/unpack and the _run_trunk guard are the branch code.
--path padded: force_flash_attn=False and no flash backend, so _run_trunk
takes the padded, row-chunked SDPA path (the path the reference run used).
--bypass-guard patches kaggriculture._GEMM_ELEMENT_LIMIT to 2**62 so neither the
packed raise nor the padded chunking engages (negative control).

Each point: eager reference via _run_trunk with the compiled callable removed,
then the compiled _run_trunk; records trunk-call M values, raise/ok, and
compiled-vs-eager differences over valid tokens.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

import owl
import owl.model.attn as attn_mod

TOKENS = 709
ACTOR_SLOTS = 482
TILES = 200
SHOP_SLOTS = 8
ROW_CHUNK = 512


def emit(fh, record: dict) -> None:
    fh.write(json.dumps(record) + "\n")
    fh.flush()


def flash_shim(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
               dropout_p, causal):
    return torch.ops.aten._flash_attention_forward(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        dropout_p, causal, False,
    )[0]


def efficient_shim(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                   dropout_p, causal):
    return torch.ops.aten._efficient_attention_forward(
        q[None], k[None], v[None], None, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k, dropout_p, 0, False,
    )[0][0]


def reference_varlen(q, k, v, cu):
    outs = []
    for i in range(cu.numel() - 1):
        s, e = int(cu[i]), int(cu[i + 1])
        o = torch.nn.functional.scaled_dot_product_attention(
            q[s:e].transpose(0, 1).float(), k[s:e].transpose(0, 1).float(),
            v[s:e].transpose(0, 1).float(),
        )
        outs.append(o.transpose(0, 1))
    return torch.cat(outs)


def pick_shim() -> tuple[str, dict]:
    gen = torch.Generator(device="cuda").manual_seed(1)
    lens = torch.tensor([709, 219, 300], dtype=torch.int32)
    cu = torch.nn.functional.pad(lens.cumsum(0, dtype=torch.int32), (1, 0)).cuda()
    t = int(lens.sum())
    q, k, v = (torch.randn((t, 8, 32), device="cuda", generator=gen,
                           dtype=torch.bfloat16) for _ in range(3))
    ref = reference_varlen(q, k, v, cu)
    errors = {}
    for name, fn in (("aten_flash_varlen", flash_shim),
                     ("aten_efficient_varlen", efficient_shim)):
        try:
            out = fn(q, k, v, cu, cu, 709, 709, 0.0, False)
            err = float((out.float() - ref).abs().max())
            if err < 0.05:
                attn_mod._FLASH_ATTN_VARLEN_FUNC = fn
                return name, {"max_abs_vs_sdpa": err, "errors": errors}
            errors[name] = f"max_abs {err}"
        except Exception as exc:  # noqa: BLE001 - recorded, next backend tried
            errors[name] = repr(exc)[:300]
    raise RuntimeError(f"no varlen attention backend works: {errors}")


def make_mask(spec: str, gen: torch.Generator) -> torch.Tensor:
    kind, _, value = spec.partition(":")
    if kind == "sparse":  # always-on 219 tokens + a few actors/shops per row
        rows = int(value)
        mask = torch.zeros((rows, TOKENS), dtype=torch.bool)
        actors = torch.randint(0, 41, (rows,), generator=gen)
        shops = torch.randint(0, SHOP_SLOTS + 1, (rows,), generator=gen)
        slot = torch.arange(ACTOR_SLOTS)
        mask[:, :ACTOR_SLOTS] = slot[None] < actors[:, None]
        mask[:, ACTOR_SLOTS:ACTOR_SLOTS + TILES] = True
        sh = torch.arange(SHOP_SLOTS)
        mask[:, ACTOR_SLOTS + TILES:ACTOR_SLOTS + TILES + SHOP_SLOTS] = (
            sh[None] < shops[:, None]
        )
        mask[:, ACTOR_SLOTS + TILES + SHOP_SLOTS:] = True
        return mask
    if kind == "dense":  # every actor slot and shop live: 709 tokens per row
        return torch.ones((int(value), TOKENS), dtype=torch.bool)
    if kind == "packed":  # dense rows, total valid tokens exactly value
        total = int(value)
        rows = -(-total // TOKENS)
        mask = torch.ones((rows, TOKENS), dtype=torch.bool)
        last = total - TOKENS * (rows - 1)
        mask[-1, last:] = False
        return mask
    raise ValueError(spec)


def compare(out: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor) -> dict:
    max_abs = 0.0
    bad25 = bad100 = nonfinite = 0
    first_bad_row = last_bad_row = None
    for s in range(0, out.shape[0], ROW_CHUNK):
        m = mask[s:s + ROW_CHUNK]
        d = (out[s:s + ROW_CHUNK].float() - ref[s:s + ROW_CHUNK].float()).abs()
        nonfinite += int((~torch.isfinite(out[s:s + ROW_CHUNK][m])).sum())
        d = torch.nan_to_num(d, nan=1e30, posinf=1e30).amax(-1)  # per token
        d = torch.where(m, d, torch.zeros_like(d))
        max_abs = max(max_abs, float(d.max()))
        bad25 += int((d > 0.25).sum())
        bad100 += int((d > 1.0).sum())
        rows_bad = (d > 0.25).any(-1).nonzero().flatten()
        if rows_bad.numel():
            if first_bad_row is None:
                first_bad_row = int(rows_bad[0]) + s
            last_bad_row = int(rows_bad[-1]) + s
    return {"max_abs_diff": max_abs, "tokens_diff_gt_0.25": bad25,
            "tokens_diff_gt_1.0": bad100, "nonfinite": nonfinite,
            "first_bad_row": first_bad_row, "last_bad_row": last_bad_row}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", choices=("packed", "padded"), required=True)
    ap.add_argument("--points", required=True, help="';'-separated mask specs")
    ap.add_argument("--bypass-guard", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    case = f"trunk_{args.path}" + ("_bypass" if args.bypass_guard else "")

    fh = open(args.out, "a")
    shim = None
    shim_info: dict = {}
    if args.path == "packed":
        shim, shim_info = pick_shim()
    import owl.model.kaggriculture as kg
    from owl.kaggriculture import types as kt

    if "/gemm-limits-src-1ddc71d/" not in kg.__file__:
        raise RuntimeError(f"model imported from the wrong tree: {kg.__file__}")

    if args.bypass_guard:
        kg._GEMM_ELEMENT_LIMIT = 2**62
    cfg = kg.KaggricultureTransformerConfig(force_flash_attn=args.path == "packed")
    assert kg.sequence_length(cfg) == TOKENS
    torch.manual_seed(0)
    model = kg.KaggricultureTransformer(
        cfg, obs_spec=kt.KaggricultureObsConfig(),
        action_spec=kt.KaggricultureActionConfig(),
    ).cuda().eval()
    model.compile_transformer_trunk(mode="max-autotune-no-cudagraphs")
    compiled = model._compiled_transformer_trunk
    calls: list[int] = []

    def recording(x, token_mask, packed):
        calls.append(int(x.shape[0]) if x.dim() == 2 else int(x.shape[0] * x.shape[1]))
        return compiled(x, token_mask, packed)

    emit(fh, {"event": "process", "case": case, "torch": torch.__version__,
              "owl_file": owl.__file__, "kg_file": kg.__file__,
              "attention_shim": shim, "shim_check": shim_info,
              "gemm_limit": kg._GEMM_ELEMENT_LIMIT,
              "gemm_kmax": kg.gemm_kmax(cfg),
              "rows_per_chunk": kg.rows_per_chunk(tokens=TOKENS, kmax=kg.gemm_kmax(cfg)),
              "use_flash_attn_bf16": attn_mod.use_flash_attn(
                  torch.empty(1, device="cuda", dtype=torch.bfloat16))})
    specs = ["dense:8"] + [p for p in args.points.split(";") if p]
    for i, spec in enumerate(specs):
        label = "warm" if i == 0 else "target"
        gen = torch.Generator().manual_seed(1000 + i)
        mask_cpu = make_mask(spec, gen)
        rows = mask_cpu.shape[0]
        emit(fh, {"event": "start", "case": case, "label": label, "spec": spec,
                  "rows": rows, "packed_tokens": int(mask_cpu.sum())})
        mask = mask_cpu.cuda()
        x = torch.randn((rows, TOKENS, cfg.embed_dim), device="cuda",
                        dtype=torch.bfloat16,
                        generator=torch.Generator(device="cuda").manual_seed(i))
        rec: dict = {"event": "result", "case": case, "label": label, "spec": spec,
                     "rows": rows, "packed_tokens": int(mask_cpu.sum()),
                     "padded_token_rows": rows * TOKENS}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model._compiled_transformer_trunk = None
            try:
                ref = model._run_trunk(x, mask)
                rec["eager"] = "ok"
            except ValueError as exc:
                ref = None
                rec["eager"] = f"ValueError: {exc}"
            calls.clear()
            model._compiled_transformer_trunk = recording
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                out = model._run_trunk(x, mask)
                torch.cuda.synchronize()
                rec["compiled"] = "ok"
            except ValueError as exc:
                out = None
                rec["compiled"] = f"ValueError: {exc}"
            rec["compiled_s"] = time.perf_counter() - t0
            rec["trunk_call_M"] = list(calls)
            rec["trunk_call_M_x_kmax_over_2p31"] = [
                c * kg.gemm_kmax(cfg) / 2**31 for c in calls]
        if out is not None and ref is not None:
            rec.update(compare(out, ref, mask))
        emit(fh, rec)
        del x, mask, ref, out
        torch.cuda.empty_cache()
    fh.close()


if __name__ == "__main__":
    main()
