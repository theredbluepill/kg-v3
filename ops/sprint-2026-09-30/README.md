# Final sprint evidence (2026-09-30 15:30Z – 2026-09-30 23:59Z)

Compact, source-bound evidence for the final Kaggriculture v3 sprint: the
self-play continuation from c50 to 210M, the fixed-shop anchor panel, the
throughput work and the GPU head-parity test. Numbers in the cookbook should
cite these files. `sprint-facts.md` is the consolidated fact sheet written at
the end of the sprint; this directory holds the evidence behind it.

Only small text was copied (no replays, checkpoints, packages, tarballs or
native binaries; nothing over 1 MB). Every file left out is listed with its size
and sha256 in [`MANIFEST-skipped.tsv`](MANIFEST-skipped.tsv). Per-game replays
and per-game stdout/stderr logs are counted there by directory. The IP address
of the terminated CPU eval pod was replaced with `<terminated-cpu-pod-ip>` (or
`<terminated-pod-ip-redacted>` in `pod.env`). No other edits were made to copied
files.

## Layout

| Path | What it is | Source |
|---|---|---|
| `sprint-facts.md` | Fact sheet: recipe, runs, checkpoint shas, panel table, submissions, throughput, rules, custody | session scratchpad |
| `clock_keeper.sh` | The SCHED_IDLE per-CPU clock keeper used on the 8x H200 pod, with its measured effect and caveats | facts file + `keepers_start.sh` |
| `anchor-games/cpu-pod/` | One-command CPU-pod panel harness (`eval_ckpt.sh`, `setup_pod.sh`, `pod.env`, `summarize.py`, `README.md`) and the per-checkpoint panel logs `eval-60M-crosscheck.log` … `eval-210M.log` | `kg-v3-int/ops/earn-money-2026-09-30/anchor-games/cpu-pod/` |
| `anchor-games/cpu-pod/pod/harness/` | The game runner shipped to the pod (`run_game.py`, `run_games.sh`, `anchor_parity.py`) | same |
| `anchor-games/cpu-pod/pod/anchors/{cha22,v56}/` | Thin anchor wrappers (the `smaller_market_shock` source is hashed in the manifest, not copied) | same |
| `anchor-games/cpu-pod/parity/linux-vs-mac-replays.jsonl` | Linux-pod vs Mac anchor replay parity result | same |
| `anchor-games/eval-60M/`, `eval-70M/`, `eval-80M/` | Mac fast-path panels: scripts, per-game JSONL, summaries, tables, the rule-off derivation and its 60M validation, spot checks | `anchor-games/eval-*` |
| `anchor-games/endgame/` | Rule 1 / rule 2 A/B scripts and tables (`ab.py`, `cf.py`, `tables.md`, `ab_rule2_tables.md`); `kaggle/ids.txt` lists the ladder episodes whose replays were analysed | `anchor-games/endgame/` |
| `anchor-games/games-fixedshop-linux/<ckpt>-ft-on/` | Linux CPU-pod panels for 60M–210M (rule 1 on): `summary.{json,md}`, `eval.json`, `run.log`, `package/{build,verify,manifest}.json` + `PACKAGE.md`, and 48 per-game `receipts/*.json` (80M has only `run.log` and receipts) | `anchor-games/games-fixedshop-linux/` |
| `anchor-games/games-fixedshop/<label>/` | Mac panels and earlier labels (bc, best-f610, candidate, c50, c50-ft-on, c50-r12-on, p3-60f2, p4, 60M, 60M-ft-on, 70M/80M arms): per-game `receipts/*.json`, rule 2 `blocks/*.json`, 70M `prerule/*.json` | `anchor-games/games-fixedshop/` |
| `throughput/sps-diag-evidence/` | 1x H200 diagnostic: training-run logs for the diagnostic arms (`sps-diag-{off,on,native-off,native-on,native-actor}-20261001.log`), `/proc/stat` snapshots, keeper vs no-keeper token benchmarks, `sps_train.sh`, `summ.py`, `maturin-17b3068d.log` | session scratchpad |
| `throughput/parity-gpu/` | Compiled vs eager actor-head parity on 1x H200 (driver 570.211.01, 110M weights, bf16): `gpu_head_parity{,2}.py`, `results{,2}.json`, run logs | session scratchpad |

The small files in the session's `sps-diag-pod-archive/` were byte-identical to
`throughput/sps-diag-evidence/`, so they are not duplicated; its checkpoint,
token tensors and native module are hashed in the manifest.

## Related evidence already in the tree

- `ops/sps-2026-10-01/` — Codex's native parallel step and Python switches:
  briefs, parity/benchmark logs, verification receipts (branch `kg/sps-combined`).
- `ops/h200-driver-gate-2026-10-01/` — ATEN-only GEMM probe on driver 570.211.01.
- `ops/late-invest-2026-10-01/` — rule 2 implementation checks and its c50 A/B.
- `ops/package-60m-2026-10-01/` — the 60M package with rule 1 baked on (not submitted).
- `ops/submit-90m-2026-10-01/`, `ops/submit-170m-2026-10-01/`,
  `ops/submit-210m-2026-10-01/` — Kaggle submission receipts.
- `ops/rebuild-2026-09-29/sprint-8gpu/` — the 8-GPU sprint kit (launch, watchdog,
  copy-off, stop) with the sprint reward recipe.
- `ops/earn-money-2026-09-30/anchor-games/` — the earlier (bc, best-f610,
  candidate) anchor games and `results.md`.

## Not archived

The 8x H200 pod was terminated right after the final submission; the last run's
(W&B `r4zqqs49`) training log and receipts were not copied off. Its metrics are
in W&B, and checkpoints 140M–210M plus `last_best` are on the Mac under
`/Users/poonszesen/kg-v3-runs/` with `SHA256SUMS`.
