"""Kaggle file-agent wrapper for v2's native V56 port (V56 Smarter Seeds and Fertilizer) (read-only use).

Documented equivalent (~/kaggriculture-v2/opponents/v56/README.md): the
JSONL entry program reads one {"observation", "configuration"} line per turn
and writes the action; one process per game (the controller is stateful).
Mirrors opponents/registry.py NativeOpponent. Any failure raises.
"""

import json
import subprocess

BINARY = "/root/anchor-eval/bin/v56_agent"  # Linux build of v2 23f75800 engine_rs (see cpu-pod/README.md)
_PROCESS = []


def _process():
    if not _PROCESS:
        _PROCESS.append(
            subprocess.Popen(
                [BINARY],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
            )
        )
    return _PROCESS[0]


def _plain(value):
    return json.loads(json.dumps(value))


def agent(observation, configuration):
    process = _process()
    request = json.dumps(
        {"observation": _plain(observation), "configuration": _plain(configuration or {})},
        ensure_ascii=False,
        allow_nan=False,
    )
    process.stdin.write(request + "\n")
    process.stdin.flush()
    line = process.stdout.readline()
    if not line:
        raise RuntimeError(
            f"v56_agent stopped (exit {process.poll()}): {process.stderr.read().strip()}"
        )
    return json.loads(line)
