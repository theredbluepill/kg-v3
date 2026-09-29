Found **one blocking correctness issue** in `49255ac`.

- **P2 — Resume can silently reuse stale labels and the wrong winner.** [kaggriculture_prepare_bc.py:779](/Users/poonszesen/kg-v3-bcnow/scripts/kaggriculture_prepare_bc.py:779) accepts completed records by filename without checking current source or preparation settings. I simulated an interrupted run, exchanged the terminal-bank winner in a **scratch copy** of its input, and resumed. Preparation succeeded; the trainer loaded the cached seat-1 winner mask although the current input’s winner was seat 0. This also permits reusing records generated before the pairing correction. Validate preparation/source identity before reuse; reject or rebuild mismatches. [Reproduction evidence](/tmp/bc51-review-0z513d/resume-result.json).

Two non-blocking items:

- **P3:** The default remains `21-27`, despite the requested `22-28`; explicit `--days 22-28` works. [CLI:718](/Users/poonszesen/kg-v3-bcnow/scripts/kaggriculture_prepare_bc.py:718).
- **P3:** Mypy reports seven errors: untyped Kaggle imports, inferred collection indexing, and a missing annotation. [Script:508](/Users/poonszesen/kg-v3-bcnow/scripts/kaggriculture_prepare_bc.py:508), [script:557](/Users/poonszesen/kg-v3-bcnow/scripts/kaggriculture_prepare_bc.py:557), [test:110](/Users/poonszesen/kg-v3-bcnow/tests/scripts/test_kaggriculture_prepare_bc.py:110). [Full output](/tmp/bc51-review-0z513d/mypy.log).

The fresh preparation path passed verification:

- Requested pytest suites: **52 passed**.
- The `t`/`t+1` correction is complete for fresh runs: observations, actor counts and placeholders use `t`; farmer, hand and market labels use `t+1`; turns cover `0..718`.
- Both shards loaded through the actual reader. All **1,438 winning-seat programs** decoded to their recorded `steps[t+1]` actions; all losing-seat placeholders matched PASS programs sized from `steps[t]`. Sampled observation tensors matched re-encoding.
- Manifest keys and array dtypes match reader validation. Gather restores integer tensors to `int64`; `policy_seat` reaches the trainer’s policy-loss mask. Malformed winning actions are explicitly rejected and counted.
- Winners match final banks. Keeping both seats on draws is sound under your allowance: both remain in the same episode split, with critic targets `0.5/0.5`.

| Local episode | Recorded `info.seed` | Pairing matches | Winning seat |
|---|---:|---:|---:|
| 114406062 | 863363664 | 719/719 | 1 |
| 114461109 | 1596988104 | 719/719 | 0 |

The CLI succeeded with **seven symlinked day ZIPs and `--days 22-28`**. These were scratch archives containing the two unchanged local payloads plus six empty archives; this verifies the layout, not the full seven-day corpus.

[Detailed report and evidence](/tmp/bc51-review-0z513d/report.md). Repository remained clean; no source-data writes, downloads, GPU execution, or training beyond the requested unit tests.

VERDICT: REQUEST CHANGES