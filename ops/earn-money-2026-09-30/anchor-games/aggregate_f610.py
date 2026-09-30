"""Aggregate the best-f610 receipts (cha22 and smaller_market_shock only) into games-best-f610.jsonl.

aggregate.py parses the anchor as label.split("-")[1], which is "f610" for the
"best-f610-<anchor>-s<seed>-seat<n>" labels, so this wrapper re-derives the anchor
and reports the paired margin against the candidate (fc6b) and BC on the same games.
"""

from __future__ import annotations

import json

from aggregate import W, load, summarize, table

POLICY = "best-f610"
ANCHORS = ["cha22", "smaller_market_shock"]


def main() -> None:
    rows = load(POLICY)
    for r in rows:
        r["anchor"] = r["label"][len(POLICY) + 1 :].split("-s9")[0]
    with (W / f"games-{POLICY}.jsonl").open("w") as handle:
        for r in rows:
            handle.write(json.dumps(r, sort_keys=True) + "\n")
    by = {a: summarize([r for r in rows if r["anchor"] == a]) for a in ANCHORS}
    by["ALL"] = summarize(rows)
    print(table(POLICY, by))
    print()
    key = lambda r: r["label"].split("-", 1)[1] if not r["label"].startswith(POLICY) else r["label"][len(POLICY) + 1 :]  # noqa: E731
    mine = {key(r): r for r in rows}
    print("| anchor | versus | paired games | mean own-bank diff | paired margin diff | f610 margin higher |")
    print("|---|---|---|---|---|---|")
    for other in ("candidate", "bc"):
        theirs = {key(r): r for r in load(other)}
        for a in [*ANCHORS, "ALL"]:
            keys = [k for k in mine if k in theirs and (a == "ALL" or mine[k]["anchor"] == a)]
            own = sum(mine[k]["own_bank"] - theirs[k]["own_bank"] for k in keys) / len(keys)
            mar = sum(mine[k]["margin"] - theirs[k]["margin"] for k in keys) / len(keys)
            hi = sum(mine[k]["margin"] > theirs[k]["margin"] for k in keys)
            print(f"| {a} | {other} | {len(keys)} | {own:+,.0f} | {mar:+,.0f} | {hi}/{len(keys)} |")


if __name__ == "__main__":
    main()
