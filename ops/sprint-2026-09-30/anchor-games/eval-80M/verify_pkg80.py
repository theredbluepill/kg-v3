"""Verify pkg-80M: slim tensors == 80M checkpoint model state; compare with 70M; manifest diff vs pkg-70M."""
import json, sys, torch
from pathlib import Path
W = Path(__file__).resolve().parents[1]
R = Path("/Users/poonszesen/kg-v3-runs/earn720-r30e01w30-8xh200-from-60M-20261001")
orig = torch.load(R / "ckpt-80M/checkpoint_00_080_063_744.pt", map_location="cpu", weights_only=False)
slim = torch.load(W / "pkg-80M/models/primary/checkpoint.pt", map_location="cpu", weights_only=False)
s70 = torch.load(W / "pkg-70M/models/primary/checkpoint.pt", map_location="cpu", weights_only=False)["model"]
m, s = orig["model"], slim["model"]
assert m.keys() == s.keys() == s70.keys()
out = {"orig_keys": sorted(orig), "slim_keys": sorted(slim), "tensors": len(s),
       "all_torch_equal_to_80M_model": all(torch.equal(m[k], s[k]) for k in m),
       "tensors_differing_from_70M": sum(not torch.equal(s[k], s70[k]) for k in s)}
a = json.loads((W / "pkg-70M/manifest.json").read_text()); b = json.loads((W / "pkg-80M/manifest.json").read_text())
out["manifest_files_differing_vs_pkg70M"] = sorted(k for k in set(a["files"]) | set(b["files"]) if a["files"].get(k) != b["files"].get(k))
out["source_equal"] = a["source"] == b["source"]; out["native_equal"] = a["native_module"] == b["native_module"]
out["builder_equal"] = a["builder"] == b["builder"]; out["source"] = b["source"]; out["checkpoint"] = b["checkpoint"]
print(json.dumps(out, indent=1)); Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")
