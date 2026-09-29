"""Extract Phase 6.0 operator-session tool calls/results that back the run statement's
pre-run authorship, idle checks and source-state claims.

Usage: python3 extract_operator_transcript.py <transcript.jsonl> > operator_transcript_excerpts.txt
The pod's public SSH host/port are redacted; nothing else is changed.
"""

import hashlib
import json
import sys

ENTRIES = {  # transcript line index -> purpose
    42: "pre-run idle/pre-check (tool call)",
    43: "pre-run idle/pre-check (result)",
    86: "run statement Write (content hashed, not reproduced)",
    92: "first pod mutation: bundle transfer, clone, detached checkout, status (tool call)",
    93: "first pod mutation (result)",
    153: "pre-smoke idle re-check (tool call)",
    154: "pre-smoke idle re-check (result)",
    176: "end-of-run check incl. tracked-file git status (tool call)",
    177: "end-of-run check (result)",
    211: "post-commit idle check (tool call)",
    212: "post-commit idle check (result)",
}
REDACT = {"157.157.221.177": "<pod-ssh-host>", "-p 10854": "-p <pod-ssh-port>", "-P 10854": "-P <pod-ssh-port>"}


def redact(s: str) -> str:
    for k, v in REDACT.items():
        s = s.replace(k, v)
    return s


path = sys.argv[1]
raw = open(path, "rb").read()
print(f"# source transcript: {path.replace('/Users/poonszesen', '~')}")
print(f"# source transcript sha256 at extraction: {hashlib.sha256(raw).hexdigest()} ({len(raw)} bytes)")
lines = raw.decode().splitlines()
for i, purpose in ENTRIES.items():
    d = json.loads(lines[i])
    for b in d["message"]["content"]:
        print(f"\n===== line {i} | {d['timestamp']} | {purpose}")
        if b.get("type") == "tool_use":
            inp = b["input"]
            if b["name"] == "Write":
                c = inp["content"].encode()
                print(f"Write {inp['file_path'].replace('/Users/poonszesen', '~')}")
                print(f"content sha256 {hashlib.sha256(c).hexdigest()} ({len(c)} bytes)")
            else:
                print(f"{b['name']} {b['id']}\n{redact(str(inp.get('command')))}")
        elif b.get("type") == "tool_result":
            cc = b.get("content")
            t = cc if isinstance(cc, str) else "\n".join(x.get("text", "") for x in cc if isinstance(x, dict))
            print(f"result for {b['tool_use_id']}\n{redact(t)}")
