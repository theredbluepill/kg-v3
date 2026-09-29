"""Print the loss and timing history of an offline W&B run directory."""

import glob
import json
import sys

from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore

PREFIXES = (
    "loss/",
    "train/env_steps",
    "_step",
    "eval/",
    "teacher/cache",
    "train/total_games",
    "time/",
    "perf/sps",
    "train/terminal",
)

store = datastore.DataStore()
store.open_for_scan(glob.glob(sys.argv[1] + "/*.wandb")[0])
rows = []
while (data := store.scan_data()) is not None:
    record = pb.Record()
    record.ParseFromString(data)
    if record.WhichOneof("record_type") == "history":
        rows.append(
            {
                item.key or "/".join(item.nested_key): json.loads(item.value_json)
                for item in record.history.item
            }
        )
out = [{k: v for k, v in row.items() if k.startswith(PREFIXES)} for row in rows]
print(json.dumps(out, indent=1, sort_keys=True))
