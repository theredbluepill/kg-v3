"""Replay a recorded game's anchor-seat observations through a native anchor binary.

usage: anchor_parity.py BINARY REPLAY_GZ ANCHOR_SEAT OUT_JSONL
Builds the wrapper's request line per step (seat observation plus the shared
keys stored on seat 0, e.g. `step`), feeds all lines to one process (the
controller is stateful, one process per game, as in anchors/*/main.py), writes
the returned actions and reports how many equal the replay's recorded action.
"""
import gzip, json, subprocess, sys

binary, replay_path, seat, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
replay = json.loads(gzip.decompress(open(replay_path, "rb").read()))
steps = replay["steps"]
config = replay["configuration"]
lines, recorded = [], []
for t in range(len(steps) - 1):
    if steps[t][seat]["status"] != "ACTIVE":
        break
    obs = dict(steps[t][0]["observation"])
    obs.update(steps[t][seat]["observation"])
    obs["player"] = seat
    lines.append(json.dumps({"observation": obs, "configuration": config}, ensure_ascii=False, allow_nan=False))
    recorded.append(steps[t + 1][seat]["action"])
proc = subprocess.run([binary], input="\n".join(lines) + "\n", capture_output=True, text=True, check=True)
actions = [json.loads(l) for l in proc.stdout.splitlines()]
with open(out, "w") as f:
    for a in actions:
        f.write(json.dumps(a, sort_keys=True) + "\n")
same = sum(a == r for a, r in zip(actions, recorded))
first = next((i for i, (a, r) in enumerate(zip(actions, recorded)) if a != r), None)
print(json.dumps({"binary": binary, "replay": replay_path.rsplit("/", 1)[-1], "requests": len(lines),
                  "outputs": len(actions), "equal_to_replay": same, "first_mismatch_step": first}))
