"""Compare native grammar_tables() (dumped by main.rs.txt) with the heads' stand-in."""

import json
import sys

import numpy as np

from owl.kaggriculture import gpu_grammar as g

native = json.load(open(sys.argv[1]))
tables = g.grammar_tables_from_arrays(
    {name: np.array(value, dtype=bool) for name, value in native.items()}
)
expected = g.expected_grammar_tables().as_dict()
total = mismatched = 0
for name, value in tables.as_dict().items():
    diff = int((value != expected[name]).sum())
    total += value.numel()
    mismatched += diff
    print(name, tuple(value.shape), "mismatch", diff)
print("total bits", total, "mismatched", mismatched)
