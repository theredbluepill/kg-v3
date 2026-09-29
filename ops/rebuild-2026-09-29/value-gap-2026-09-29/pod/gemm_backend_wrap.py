"""Run a probe/bench script with an explicit Inductor GEMM-backend setting.

Usage: python gemm_backend_wrap.py --backends {default,ATEN} --record R.json
       -- SCRIPT [ARGS...]

``default`` leaves torch._inductor.config.max_autotune_gemm_backends untouched
(torch 2.9.0 default "ATEN,TRITON,CPP"); ``ATEN`` sets it to "ATEN" before any
compile. The value is recorded before and after the script runs (R.json), and
the TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS env var must be unset so the
setting comes only from this wrapper. The script runs in this process via
runpy with run_name="__main__" and its own argv.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
import time

import torch
import torch._inductor.config as inductor_config


def main() -> None:
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit("usage: gemm_backend_wrap.py --backends B --record R -- SCRIPT ...")
    split = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", choices=("default", "ATEN"), required=True)
    ap.add_argument("--record", required=True)
    args = ap.parse_args(argv[:split])
    script, *script_args = argv[split + 1:]
    if "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS" in os.environ:
        raise SystemExit("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS must be unset")
    before = inductor_config.max_autotune_gemm_backends
    if args.backends == "ATEN":
        inductor_config.max_autotune_gemm_backends = "ATEN"
    rec = {"requested": args.backends, "value_before_set": before,
           "value_at_start": inductor_config.max_autotune_gemm_backends,
           "script": script, "script_args": script_args,
           "torch": torch.__version__, "torch_git": torch.version.git_version,
           "start_unix": time.time()}
    with open(args.record, "w") as fh:
        json.dump(rec, fh, indent=1)
    sys.argv = [script, *script_args]
    try:
        runpy.run_path(script, run_name="__main__")
    finally:
        rec["value_at_end"] = inductor_config.max_autotune_gemm_backends
        rec["end_unix"] = time.time()
        with open(args.record, "w") as fh:
            json.dump(rec, fh, indent=1)


if __name__ == "__main__":
    main()
