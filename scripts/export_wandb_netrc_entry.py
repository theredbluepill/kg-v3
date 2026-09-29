#!/usr/bin/env python3
"""Write only the W&B netrc entry to a pipe, for installing it on a pod.

Usage (operator's machine, see
``cookbook/workflows/install-the-wandb-credential-before-any-pod-launch.md``)::

    uv run python scripts/export_wandb_netrc_entry.py | ssh <pod> '<install>'

Reads ``NETRC`` (default ``~/.netrc``) and emits one ``machine <host> login
<login> password <key>`` line for the ``WANDB_BASE_URL`` host (default
``api.wandb.ai``), never another machine or ``default``. It refuses to write to
a terminal so the key cannot be printed by mistake.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from owl.train.logging import MissingWandbCredentialsError, wandb_netrc_entry


def main() -> int:
    if sys.stdout.isatty():
        print(
            "refusing to print the W&B key to a terminal; pipe this into ssh",
            file=sys.stderr,
        )
        return 2
    try:
        entry = wandb_netrc_entry(os.environ, home=Path.home())
    except MissingWandbCredentialsError as error:
        print(f"nothing exported: {error}", file=sys.stderr)
        return 1
    try:
        sys.stdout.write(entry)
        sys.stdout.flush()
    except BrokenPipeError:
        # The receiver refused before reading (for example, an existing netrc).
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        print("the receiver closed the pipe; nothing was sent", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
