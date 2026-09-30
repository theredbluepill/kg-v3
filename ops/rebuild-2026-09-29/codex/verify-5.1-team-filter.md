Found two correctness issues in `42a8b39`:

- **P2 — Accepted option abbreviations leak the name.** Argparse accepts `--tea NAME` and `--te=NAME`, but [the redactor:817](/Users/poonszesen/kg-v3-bcnow/scripts/kaggriculture_prepare_bc.py:817) handles only exact `--team` spellings. An actual CLI invocation persisted the clear name in `manifest.json` through [run.command:959](/Users/poonszesen/kg-v3-bcnow/scripts/kaggriculture_prepare_bc.py:959). Redaction must cover every accepted spelling.
- **P2 — Draw totals become incorrect.** [build_manifest:704](/Users/poonszesen/kg-v3-bcnow/scripts/kaggriculture_prepare_bc.py:704) equates two policy seats with a draw. Team filtering breaks that equivalence: three admitted draws were reported as one; with `--include-losses`, a decisive mirror game was reported as a draw. Count equal terminal banks instead.

The remaining checks passed:

- Seat 0, seat 1, mirrors, draws, absent teams and malformed `TeamNames`: **32 independent cases**, plus preparation through the actual shard reader. Masks, selected actions, both-seat observations and PASS placeholders matched. The trainer applies the mask at [bc.py:320](/Users/poonszesen/kg-v3-bcnow/python/owl/train/bc.py:320).
- Different-team and changed-`include_losses` resumes fail before cached records are read; unchanged resumes succeed. [Identity binding:916](/Users/poonszesen/kg-v3-bcnow/scripts/kaggriculture_prepare_bc.py:916).
- Absent/lost rejection counts and episode/turn/policy-row totals reconcile. No additional name leak was found in exceptions, logs or pairing output.
- No name or team hash reaches shard tensors or model inputs. [Shard fields:223](/Users/poonszesen/kg-v3-bcnow/python/owl/kaggriculture/bc_data.py:223), [model call:332](/Users/poonszesen/kg-v3-bcnow/python/owl/train/bc.py:332).

**55 tests passed; mypy and ruff passed on both changed files.** Tracked files remain unchanged; all scratch evidence is in the [review report](/tmp/bc-team-review-tZFUFb/report.md).

VERDICT: REQUEST CHANGES