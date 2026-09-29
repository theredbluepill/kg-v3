import json, sys, hashlib, pathlib, importlib.metadata as md
import torch, triton, flash_attn
so = [p for p in pathlib.Path(flash_attn.__file__).parent.parent.glob("flash_attn_2_cuda*.so")]
out = {
  "python": sys.version, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
  "cudnn": torch.backends.cudnn.version(), "cxx11abi": torch._C._GLIBCXX_USE_CXX11_ABI,
  "triton": triton.__version__, "flash_attn": flash_attn.__version__,
  "flash_attn_dist": md.version("flash-attn"), "einops": md.version("einops"),
  "flash_attn_so": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size} for p in so],
  "device": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0),
  "torch_arch_list": torch.cuda.get_arch_list(),
}
import flash_attn_2_cuda
out["flash_attn_2_cuda_file"] = flash_attn_2_cuda.__file__
print(json.dumps(out, indent=2))
