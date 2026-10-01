---
type: "Workflow"
title: "Evaluate each checkpoint on the anchor panel with a CPU pod"
description: "Final-sprint workflow (2026-09-30): build the real Linux Kaggle package for a checkpoint, ship it to a 32-vCPU CPU pod, play the 48-game fixed-shop anchor panel (8 seeds x 2 seats x smaller_market_shock, cha22, v56) up to 30 games in parallel, and copy the receipts back, with one command (eval_ckpt.sh). About 1 min packaging plus about 6 min of games per checkpoint, against about 25 min on the Mac. The pod reproduced the Mac's 60M and 70M panels game for game (48/48 identical final banks). Selection panel, not held-out; it saturated at 48-0 from 170M on."
tags: ["kaggriculture-v3", "workflows", "evaluation", "pods", "kaggle"]
status: "verified-scoped"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-10-01"}
sources:
  - resource: "user-directive:2026-09-30:start-a-cpu-pod-and-evaluate-the-pt"
  - resource: "user-directive:2026-09-30:let-anchors-speak-will-be-good"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/cpu-pod/README.md"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/cpu-pod/eval_ckpt.sh"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/cpu-pod/setup_pod.sh"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/cpu-pod/summarize.py"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/cpu-pod/pod/harness/run_game.py"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/cpu-pod/pod/harness/anchor_parity.py"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/cpu-pod/parity/linux-vs-mac-replays.jsonl"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/cpu-pod/eval-60M-crosscheck.log"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/cpu-pod/eval-70M.log"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/cpu-pod/eval-210M.log"
  - resource: "repository:scripts/package_checkpoint.sh"
---

# Evaluate each checkpoint on the anchor panel with a CPU pod

**Why.** At about 8,300 env steps/s the 8x H200 run produced a 10M checkpoint every ~20 minutes, while the Mac needed ~25–40 minutes per 48-game panel. The owner asked: "Let's start a cpu pod as you said and evaluate the pt." Promotion and submission choices in the sprint leaned on this panel ("let anchors speak will be good").

## Steps

1. **Pod (once).** RunPod CPU pod `cpu3c`, 32 vCPU (AMD EPYC 9965), $0.96/hr. Run `ops/sprint-2026-09-30/anchor-games/cpu-pod/setup_pod.sh`: uv, a Python 3.11 venv with torch 2.6.0 CPU (the Kaggle runtime), the fixed-shop copy of kaggle-environments on `PYTHONPATH`, and Linux builds of the cha22 and v56 anchors from a read-only export of the v2 `engine_rs` (commit `23f75800`; binary shas in the README). smaller_market_shock is pure Python.
2. **Per checkpoint.** `eval_ckpt.sh CKPT CONFIG LABEL [--rule1-off] [--package DIR]`:
   - builds the Linux Kaggle package with `scripts/package_checkpoint.sh --final-turn-liquidation` (rule 1 on, rule 2 off, clean native module), or reuses an existing one with `--package` (used for 210M so the submitted archive itself was evaluated);
   - ships it, plays the 48 games with up to 30 in parallel (one torch thread per game), and copies receipts, logs and the package manifest back to `games-fixedshop-linux/<LABEL>-ft-on/` as one tar stream;
   - refuses the training-pod ports, checks the pinned engine and anchor shas, and resumes where it left off.
3. **Read the result** with `summarize.py` or a paired script: per-anchor W-L and margins, paired delta against earlier checkpoints with SE over the 8 seed means.

## Verification (this sprint)

- **Parity with the Mac harness.** 60M and 70M played on the pod matched the Mac games 48/48 with identical final banks and actions (`parity/linux-vs-mac-replays.jsonl`, `eval-60M-crosscheck.log`). Replaying 23,008 recorded anchor actions through the Linux anchor builds reproduced all of them.
- **Speed.** About 6 minutes of games per panel (each game ~120–200 s on the shared host, 30 in parallel) plus about 1 minute of packaging; copy-back fell from 4 min to 14 s after switching to a single tar stream.
- **Use.** Every sprint checkpoint from 70M to 210M was evaluated this way (`eval-*.log`).

## Limits

- It is a **selection panel** (3 anchors, 8 seeds; seat mirrors often play the same game), not held-out qualification, and from 170M on our checkpoints won 48-0, so it no longer separates strong checkpoints. Next step: add our own past checkpoints as opponents (league evaluation).
- The fixed-shop engine separates the shop draw from the weed RNG, so results differ from the official engine's market path.
- Two operational slips: one `ssh` copy-back dropped (receipts were still complete), and the packaged 720-turn Kaggle-image game is skipped in this mode (the 48 full games in the same package cover it).

Related: [[../episodes/the-final-sprint-took-720-turn-self-play-from-c50-to-a-48-0-anchor-panel|final-sprint episode]], [[../decisions/hold-the-learning-rate-and-let-the-anchor-panel-speak|LR hold and anchor panel]], [[../decisions/evaluation-preserves-generality-and-evidence|evaluation contract]].
