---
type: "Reference"
title: "Earn-money pod runs use run-local launch, copy-off and switch scripts"
description: "Adaptation inventory, 2026-09-30: the 4-rank earn-money runs (h3lpxy6q, ssoc84zg, pw6qjsz3, 4h9c3d6g, cmwjclbe, pz3xhg9e, pcy5knet) and the from-scratch and vs-cha22 runs before them were launched by run-local bash scripts at code 0f70773 with configs/kaggriculture_4rank_margin.yaml plus -o overrides (no new preset). Each has a Mac copy-off loop that verifies checkpoint sha256 against the pod. switch.sh chained cmwjclbe into pz3xhg9e and has a detection flaw: its last_best mtime test can fire before the promotion evaluation ends; the outcome was checked by weights. Memory: horizon 720 at 16 envs/rank used 90.96 of 96 GB per GPU, so 12 envs/rank (68.5 GB) is the chosen setting. The local Kaggle-harness anchor tooling is inventoried too. Checks are the receipts and hash comparisons named here; the scripts carry no tests."
tags: ["kaggriculture-v3", "adaptation", "training", "operations", "custody"]
status: "verified-scoped"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-09-30"}
sources:
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/earn-bank-credit-4rank/run_earn.sh"
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/earn-bank-credit-4rank/copyoff.sh"
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/earn-bank-credit-4rank/launch.md"
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/earn512/run_earn512.sh"
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/earn512/launch.md"
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/earn720/run_earn720.sh"
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/earn720/launch.md"
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/earn720-12env/run.sh"
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/earn720-12env/copyoff.sh"
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/earn720-12env/launch.md"
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/earnlr/switch.sh"
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/earnlr/switch.log"
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/earnlr/launch.md"
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/earnB/run.sh"
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/earnB/launch.md"
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/scratch-selfplay-4rank/run_scratch.sh"
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/scratch-selfplay-4rank/make_teacher_init.py.txt"
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/scratch-bank-4rank/run_scratch_bank.sh"
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/scratch-lr2e3-4rank/run_scratch_lr.sh"
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/vs-cha22-4rank/run_cha22.sh"
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/vs-cha22-4rank/stop.md"
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/A-bank-lr2-4rank/main_probe.py"
  - resource: "repository:ops/rebuild-2026-09-29/pod4-2026-09-30/A-bank-lr2-4rank/watchdog.py"
  - resource: "repository:configs/kaggriculture_4rank_margin.yaml"
  - resource: "repository:scripts/run_ppo.py"
  - resource: "repository:ops/earn-money-2026-09-30/anchor-games/pkg_local_mac.py"
  - resource: "repository:ops/earn-money-2026-09-30/anchor-games/run_game.py"
  - resource: "repository:ops/earn-money-2026-09-30/anchor-games/run_all.sh"
  - resource: "repository:ops/earn-money-2026-09-30/anchor-games/aggregate.py"
  - resource: "repository:ops/earn-money-2026-09-30/anchor-games/aggregate_f610.py"
  - resource: "repository:ops/earn-money-2026-09-30/anchor-games/manifests/best-f610.json"
  - resource: "local-untracked:~/kg-v3-runs/earn-bank-credit-4rank-20260930/receipts/hashes.sha256"
  - resource: "local-untracked:~/kg-v3-runs/earn720-12env-from-promoted-4rank-20260930/receipts/hashes.sha256"
  - resource: "local-untracked:~/kg-v3-runs/earn720-lr1e4-from-promoted2-4rank-20260930/receipts/hashes.sha256"
  - resource: "wandb-run:spoon/kg-v3/4h9c3d6g"
---

# Earn-money pod runs use run-local launch, copy-off and switch scripts

## What was adapted

On 2026-09-30 the 4-GPU pod (`abl4mvr5w1mmn4`) ran a chain of self-play runs. They are launched by run-local scripts, not by a new preset or a second trainer. The canonical trainer `scripts/run_ppo.py` is unchanged. The receipts live under `ops/rebuild-2026-09-29/pod4-2026-09-30/`, one folder per run, merged from `kg/rebuild-pod4-evidence`.

**Why.** The owner asked for launches before the stagger and critic-offset branches landed ("can we implement this asap?", "Can we get this running?"). So the long credit window and the reward were applied as command-line overrides on already-landed code. The finding these runs support is in [[a-long-credit-window-turned-bc-start-self-play-from-sliding-to-improving|the credit-window Reference]].

## Inventory

**Launch scripts.** Each is a bash wrapper started with `setsid nohup … & disown` in `/root/kg-v3-anchor`, which was checked out at `0f70773` with 0 porcelain lines (Mac receipts `git_state.txt`). Each wrapper:
- writes `hashes.sha256` over the config, model config, trainer, reward and env sources, the native `.so`, the lockfiles, the start checkpoint, the wrapper and the watchdog;
- refuses to launch on a busy GPU;
- samples `nvidia-smi` every 2 s for the first 15 min, then every 60 s;
- runs `torchrun --nproc-per-node 4 main_probe.py scripts/run_ppo.py <config> <run dir> … --log-mode wandb --wandb-mode online` with no step or time cap.

| Folder | Script | W&B | Config and overrides (margin preset sha256 prefix `b7fa7f9d5736`, equal at `0f70773` and on this branch) |
| --- | --- | --- | --- |
| `scratch-selfplay-4rank` | `run_scratch.sh` | `wk142q4b` | margin preset; random init; teacher seeded by `make_teacher_init.py.txt` via `rl.teacher_init` |
| `scratch-bank-4rank` | `run_scratch_bank.sh` | `d6sh0akf` | + reward .25 bank /150k, .25 margin /100k, shaping 0 |
| `scratch-lr2e3-4rank` | `run_scratch_lr.sh` | `uujuarkx` | + Muon 2e-3 / AdamW 1e-4 |
| `vs-cha22-4rank` | `run_cha22.sh` | `kifqbyx5` | `configs/kaggriculture_4rank_vs_cha22.yaml` (term M, cha22 in every env), BC start |
| `earn-bank-credit-4rank` | `run_earn.sh` | `h3lpxy6q` | margin preset; BC start; the reward; `env.n_envs=16 rl.horizon=256 rl.segments_per_minibatch=1 rl.gae_lambda=1.0` |
| `earn512` | `run_earn512.sh` | `ssoc84zg` | fc6b start; `rl.horizon=512 env.n_envs=8` |
| `earn720` | `run_earn720.sh` | `pw6qjsz3` | fc6b start; `rl.horizon=720 env.n_envs=6` |
| `earn720-12env` | `run.sh` | `cmwjclbe` | fc6b start; `rl.horizon=720 env.n_envs=12` |
| `earnlr` | generated by `switch.sh` | `pz3xhg9e` | f610 start; + `optimizer.muon_lr=0.002 optimizer.adamw_lr=0.0001` |
| `earnB` | `run.sh` | `pcy5knet` | f610 start; 720, 12 envs, Muon 1e-4 / AdamW 5e-6 |

All the earn runs load their start with `--load-model-weights-mode model_only`: a fresh optimizer, a new 1,000-step LR warm-up, and the start as both teacher and first `last_best`. `main_probe.py` (`26ba5b0c…`) and the nonfinite-only `watchdog.py` (`7a8c0e98…`) are byte-identical to A2's committed copies. The Mac receipts of `h3lpxy6q`, `cmwjclbe` and `pcy5knet` confirm this.

**Copy-off loops.** Each folder has a `copyoff.sh`. Every 10 min it rsyncs the log, the watchdog log and the receipts to `~/kg-v3-runs/<name>/`. It copies only finished `checkpoint_*.pt` files (unchanged for 60 s), compares each file's sha256 with the pod's, deletes a mismatched copy for retry, and rewrites `SHA256SUMS`. It holds the pod host and port, but no credential. Flaw: the header comments were copied from a template, and some still name `scratch-bank-4rank` (for example `earn720-12env/copyoff.sh`). The `DEST` and `NAME` variables are correct.

**Switch script (`earnlr/switch.sh`, log `switch.log`).** The owner said: "let eval complete, I pretty much sure it will promote, but let's run 2e-3 /1e-4 on the new run". The script:
1. waits for `cmwjclbe`'s next checkpoint;
2. treats a `checkpoint_last_best.pt` mtime at or after the checkpoint's as a promotion;
3. copies the file to `/root/promoted-B/`;
4. stops the run by process group;
5. writes `/root/earnlr/run.sh` by `sed` from `earn720-12env/run.sh`, and launches it.

- **Detection flaw.** `run_ppo` writes `last_best` from the start model when the file is absent at the first checkpoint, so the mtime test can fire before the evaluation ends. It fired at 11:04:08Z, and the file's final mtime was 11:04:18Z.
- **Outcome check.** The copied weights are bit-identical to `checkpoint_00_020_033_024` and differ from fc6b, so the copy was the promotion. `cmwjclbe`'s evaluation metrics never reached W&B because the run was stopped about 40 s later. The generated `earnlr/run.sh` is not committed: only its `sed` recipe is.
- **Reuse condition.** Before reusing the script, test for the evaluation's log record, or compare the weights, instead of the mtime.

**Memory finding at horizon 720.** Per GPU (RTX PRO 6000, 96 GB):

| envs/rank | GPU memory | env steps/s | Run |
| --- | --- | --- | --- |
| 6 | 40.9-48.6 GB | about 1,900 | `pw6qjsz3` |
| 12 | 68.5 GB | about 2,410-2,460 | `cmwjclbe` |
| 16 | 90.96 GB | 2,615 | `4h9c3d6g`, stopped at iteration about 3 |

Sixteen envs left too little headroom for an unattended run whose rank 0 also hosts the 10M evaluation, so 12 envs per rank is the working setting. These are launch-time samples, not an Nsight profile.

**Local anchor-game tooling** (`ops/earn-money-2026-09-30/anchor-games/`):
- `pkg_local_mac.py` mirrors the ship builder with a macOS arm64 `owl.rs`.
- `run_game.py` is the ship's local-episode runner with strict mode and per-call receipts.
- `run_all.sh`, `run_sms.sh` and `run_cha22.sh` run the policy, anchor, seed and seat grid.
- `aggregate.py` writes the per-game JSONL and the tables. It parses the anchor as `label.split("-")[1]`, which is wrong for `best-f610-…` labels. `aggregate_f610.py` corrects this for f610 and adds paired differences.
- The package manifests are copied to `manifests/`. The packages, replays and third-party anchor sources are not committed.

## Checks

- **Hashes.** The wrapper, watchdog, config and code identity were compared against the Mac receipts for three runs, and the preset hash against `0f70773` (this session).
- **f610 weights.** The weights in `/root/promoted-B` were compared with the 20M checkpoint on the pod (`earnlr/launch.md`).
- **f610 table.** `aggregate_f610.py` was run on the 32 committed receipts.
- None of the scripts has tests.

## Gaps

- `ssoc84zg` and `pw6qjsz3` receipts were not re-hashed here.
- The watchdog stops a run only on a nonfinite loss. Bank and KL collapse, as in `pz3xhg9e`, are judged and stopped by a person.
- The 16-env memory figure is one launch sample.
