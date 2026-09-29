import json, platform, sys
import torch, triton, flash_attn
import torch.nn.functional as F
from flash_attn import flash_attn_func

out = {
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "triton": triton.__version__,
    "flash_attn": flash_attn.__version__,
    "gpus": [],
}
assert torch.__version__ == "2.9.0+cu128", torch.__version__
assert triton.__version__ == "3.5.0", triton.__version__
assert flash_attn.__version__ == "2.8.3", flash_attn.__version__
for dev in range(torch.cuda.device_count()):
    torch.manual_seed(0)
    d = torch.device("cuda", dev)
    cap = torch.cuda.get_device_capability(d)
    B, L, H, D = 4, 256, 8, 64
    q, k, v = (torch.randn(B, L, H, D, device=d, dtype=torch.bfloat16) for _ in range(3))
    fa = flash_attn_func(q, k, v, causal=False)
    ref = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
    fa_c = flash_attn_func(q, k, v, causal=True)
    ref_c = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True).transpose(1, 2)
    err = (fa.float() - ref.float()).abs().max().item()
    err_c = (fa_c.float() - ref_c.float()).abs().max().item()
    torch.cuda.synchronize(d)
    ok = err < 2e-2 and err_c < 2e-2 and torch.isfinite(fa).all().item()
    out["gpus"].append({"index": dev, "name": torch.cuda.get_device_name(d), "capability": f"sm_{cap[0]}{cap[1]}",
                        "shape_BLHD": [B, L, H, D], "dtype": "bf16",
                        "max_abs_err_noncausal": err, "max_abs_err_causal": err_c, "ok": ok})
    assert cap == (12, 0), cap
    assert ok, out["gpus"][-1]
print("sm arch list:", torch.cuda.get_arch_list())
out["torch_arch_list"] = torch.cuda.get_arch_list()

import owl.rs  # noqa
import owl
out["owl_rs"] = owl.rs.__file__
import wandb
api = wandb.Api()
out["wandb_version"] = wandb.__version__
out["wandb_default_entity"] = api.default_entity
print(json.dumps(out, indent=2))
