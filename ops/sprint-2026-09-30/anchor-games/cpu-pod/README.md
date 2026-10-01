# Linux CPU-pod fixed-shop anchor eval (48 games per checkpoint)

Date 2026-10-01. Owner, verbatim: "Let's start a cpu pod as you said and evaluate the pt."; earlier "let anchors speak will be good", "let's not apply rule2."
Nothing was committed, pushed, uploaded or submitted. The H200 training pod (port 15795) and the diagnostic pod (port 15537) were not touched, and `pod.env` refuses both ports.

## One command

```
./eval_ckpt.sh CKPT CONFIG LABEL [--rule1-off] [--package PKG_OUT_DIR]
```
1. Package on the Mac with `$KG_PKG_REPO/scripts/package_checkpoint.sh ... --final-turn-liquidation` (default `KG_PKG_REPO=/Users/poonszesen/kg-v3-pkg60`, a clean worktree on kg/package-60m). Output goes to its git-ignored `artifacts/anchor-eval/<OUT>-<ckpt sha12>/`. The script then checks the checkpoint/config sha, that rule 1 is baked on (or off with `--rule1-off`) and that rule 2 is on its env switch.
2. Ship the Linux tarball to `/root/anchor-eval/pkgs/<OUT>-<archive sha12>/agent`. This step first checks the pod pins (engine sha, anchor binary shas) and refuses to mix receipts from two packages in one games folder.
3. On the pod, `pod/harness/run_games.sh` runs 8 seeds 93001-93008 x 2 seats x {smaller_market_shock, cha22, v56} = 48 games, PARALLEL=30, 1 torch thread each, nice 10, strict agent, with `KAGGRICULTURE_AGENT_BLOCK_LATE_INVESTMENTS=0` (rule 2 off). It is detached and resumable (existing receipts are skipped).
4. Copy back as one tar stream to `../games-fixedshop-linux/<OUT>/{receipts,logs,replays,run.log}`, along with `package/` (PACKAGE.md, verify.json, build.json, manifest.json), `summary.md`/`summary.json` (from `summarize.py`) and `eval.json` (ids and stage wall times).

`OUT` is `LABEL-ft-on` (rule 1 on) or `LABEL` (rule 1 off), matching the Mac arms. Labels are `<LABEL>-<anchor>-s<seed>-seat<seat>` and receipts come from the same `run_game.py`, so `summarize.py DIR --compare OTHER` pairs Linux and Mac folders by (anchor, seed, seat).
A fresh pod needs `./setup_pod.sh` first. It is idempotent: set `POD_HOST`/`POD_PORT` in the environment or in `pod.env`.

## Pod and setup (`setup_pod.sh`)
- RunPod CPU pod `<terminated-cpu-pod-ip>:32670`: 32 vCPU (cgroup cpuset of a 384-CPU AMD EPYC 9965 host, host load average 100-160 during runs), 754 GB RAM visible, Ubuntu 24.04.2, system Python 3.12.3.
- uv 0.12.21 and rustup minimal (rustc/cargo 1.98.1).
- venv `/root/anchor-eval/venv` on Python 3.11.13 with torch 2.6.0+cpu (download.pytorch.org/whl/cpu), numpy 2.4.6, pyyaml 6.0.3 and kaggle-environments 1.32.7 from PyPI (for its dependencies). The Mac venv-kaggle had Py 3.11.15, torch 2.6.0 and numpy 2.4.6.
- Fixed-shop engine: the Mac `scratchpad/kenv-fixedshop` is copied to `/root/anchor-eval/kenv-fixedshop` and put first on PYTHONPATH, exactly as on the Mac. It has no packaging metadata, so an editable install is not possible. `kaggriculture.py` sha256 is `f73d27ce04a6fd5ddcdd59c618e1da8c9b6fb2f27f5cf9cdc27efad942345004`, the same as the Mac receipts and all 96 Linux receipts. The input tarball is cached at `~/.cache/kg-v3/anchor-eval/kenv-fixedshop.tar.gz` (sha256 `3b692d62…1a3e`).
- Anchors (`pod/anchors/`):
  - smaller_market_shock: `main.py` is pure Python and is copied unchanged (`V92_SELL_LIB` stays unset, as on the Mac).
  - cha22 and v56: both wrap v2 native binaries. Only the `BINARY` path changed, to `/root/anchor-eval/bin/*`.
  - The binaries are built on the pod from `git archive 23f758007c3c53befc9bcab15a5bc7661d8b97e0 engine_rs` of `~/kaggriculture-v2`. That is read-only use; the archive is cached as `engine_rs-23f75800.tar.gz`, sha256 `a2eeee59…d6e6`. Build: `cargo build --release --locked --bin v56_agent --bin cha22_agent`.
  - Linux shas: `cha22_agent` `bd7ffaa0b97b829a6ea3217178f12a1aa0f64cf8f58a750849c79ca94088c5ca`, `v56_agent` `34458b2a35f77be6431a3b4b692aaffd0bcb6bb3fbcf1356849555f54478f119`. The Mac binaries are `82334d26…ed0a` and `9216782d…0702`.
  - The Mac binaries predate the v2 worktree's uncommitted `ffi.rs`/`Cargo.toml` edits (PyO3 only). `engine_rs/src/native_agents` and `src/bin` are identical between the worktree and 23f75800.

## Equivalence evidence
- Anchor binaries: `pod/harness/anchor_parity.py` feeds every anchor-seat observation of the 32 Mac 60M-ft-on cha22/v56 replays into one Linux process per game. Result: 23,008/23,008 actions equal the recorded Mac actions (`parity/linux-vs-mac-replays.jsonl`). The Mac binaries reproduced 4 of those replays 719/719 (`parity/mac-*.jsonl`).
- Whole harness: the existing 60M Linux package (`kg-v3-pkg60/artifacts/60m-r1`, de26cd68, rule 1 baked, manifest `469a8a25…`) was run through `eval_ckpt.sh --package`. Against the Mac 60M-ft-on arm (pkg-60M, 4c99768a, rule 1 by env, macOS arm64), **48/48 games have identical final banks and 48/48 replays have identical action sequences for both seats at every step**. Receipts are in `../games-fixedshop-linux/60M-ft-on/` and the comparison is `vs-mac-60M-ft-on.json`. So Linux-vs-Mac platform differences do not move these games, and Linux results pair directly with Mac arms.

## 70M result (`../games-fixedshop-linux/70M-ft-on/`)
- Checkpoint `earn720-r30e01w30-8xh200-from-60M-20261001/ckpt-70M/checkpoint_00_070_041_344.pt`, sha256 `4c8b9483…e9`, config `f7a1349a…377c`.
- Package: kg/package-60m `cfc45f0e`, clean tree, rule 1 baked on, rule 2 env (off).
  - Archive `7cbb97ed…6d26`, inner manifest `e02a51cb…314d`, slim checkpoint `addbdb38…dfba`, 210/210 tensors equal.
  - Native module `3e5e4e55…664e` (source 9a743fad, no fixed opponents).
  - The 40-turn Kaggle-image episode qualified.
  - An earlier identical scratch build had archive `47800f5c…` and manifest `e3315f4f…`: the tarball and manifest are not byte-reproducible across builds, so check custody by these ids.
- Health: 48/48 qualified. Every game had 719 calls, 0 exceptions, 0 invalid raw actions, 0 default passes and 0 bad statuses. Turn 0 took at most 2.00 s, steady p99 at most 0.497 s, and minimum remaining overage was 59.0 s (host CPU, not Kaggle hardware).

| anchor | 70M W-L | 70M margin (SE seeds) | 60M W-L | 60M margin | 70M better on seeds | flips L->W / W->L |
| --- | --- | --- | --- | --- | --- | --- |
| cha22 | 4-12 | -1789.8 (1908.3) | 2-14 | -4492.8 | 7/8 | 2 / 0 |
| smaller_market_shock | 6-10 | -1168.8 (1845.9) | 4-12 | -1215.5 | 5/8 | 4 / 2 |
| v56 | 6-10 | -1702.2 (1188.3) | 2-14 | -4204.4 | 5/8 | 4 / 0 |
| ALL | 16-32 | -1553.6 (1534.6) | 8-40 | -3304.2 | 7/8 | 10 / 2 |

The paired margin difference for 70M minus 60M is +1750.6, with SE 985.4 over the 8 seeds (`vs-linux-60M-ft-on.json`). This is a selection-panel measurement, not held-out qualification. 70M still loses on average to all three anchors.

## Wall time per 48 games (PARALLEL=30)
- 70M total was 727 s:
  - package 48 s (the 40-turn Kaggle-image episode is 41 s of that)
  - ship 37 s
  - games 366 s by the pod run.log (406 s including 30 s polling)
  - copy-back 237 s
- The copy-back used `scp -r` in that run. It was then replaced by a tar stream, and the 60M cross-check copy-back took 14 s (total 426 s, games 348 s, no package build).
- Per game: 165 s mean and 189 s max on this pod at 30-way concurrency, versus about 90 s on the Mac at 5-way. The run is two waves (30 + 18 games).
