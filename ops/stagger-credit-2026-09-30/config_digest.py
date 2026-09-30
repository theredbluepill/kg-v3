"""config_sha256 of every loadable preset; equal output before and after the
stagger/credit change shows the default-off config dump is unchanged."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from owl.train.config import FullConfig
from owl.train.logging import config_sha256

out: dict[str, str] = {}
for path in sorted(Path("configs").glob("*.yaml")):
    if path.name in sys.argv[1:]:
        continue
    try:
        cfg = FullConfig.from_file(path)
    except Exception as exc:  # noqa: BLE001 - receipt records non-loading presets
        out[path.name] = f"unloadable: {type(exc).__name__}"
        continue
    out[path.name] = config_sha256(cfg)
print(json.dumps(out, indent=1, sort_keys=True))
