# 08bc Kaggle probe submission receipt

- Owner request (verbatim, 2026-09-30): "can you package the 08bc and submit to
  kaggle for probing? Note we have 4 submissions left, only use 1 of it."
- Submissions used: exactly 1. Before submitting, the competition listing showed
  one submission already made on 2026-09-30 UTC (ref 56695632, the BC agent) and
  nothing pending. After the submit the CLI printed "3 submissions remaining today."

## Identity

| Item | Value |
| --- | --- |
| Competition | `kaggriculture` |
| Archive | `artifacts/08bc/submission.tar.gz`, 25.9 MB |
| Archive sha256 | `8283e67691befa88ca66706322bf86f35aec39fb54ccd20c1bc7951b86e6d33c` |
| Inner `manifest.json` sha256 | `e77ea68d2de5d440ff4d8068e95069439bf307aa23ba6404eb94d2803d6f7bd5` |
| Checkpoint | `kg-v3-runs/earn720-lr1e4-from-promoted2-4rank-20260930/promoted-40M/checkpoint_last_best.pt` |
| Checkpoint sha256 (original) | `08bc19aed8c0647002ea60e281f39b217f59b82deba54f45f53b839b2e574600` |
| Slim checkpoint in archive sha256 | `664e9470608b8e17955cdbbd08b244e6afce6b027768f1f08b074b26a47f1635` |
| Config sha256 (run and archive) | `62e0b5c118305b6ffeb277d4eb68a13ef03e53f3ad31e72a3c5b556a249675f7` |
| Source | branch `kg/submit-08bc`, commit `9a743fad991593fdd372115a48e3b372acf45b8b`, tree `aaf9d4d2`, clean |
| Native module | `owl/rs.abi3.so` sha256 `2bbdd2f0...8312`; built on pod abl4mvr5w1mmn4 with `maturin --compatibility linux`; highest GLIBC symbol 2.35 (`hypotf`); see `native-module-receipt.json`, `glibc-symbols.txt` |

## Validation before upload

- Kaggle-image Docker episode (Python 3.11.13, torch 2.6.0, kaggle-environments
  1.32.7, glibc 2.35), self-play seed 7, strict mode, run from this archive's
  manifest `e77ea68d`: qualified; 720 steps, both statuses DONE, 0 bad statuses,
  0 exceptions, 0 invalid raw actions, 0 default passes; steady p99 0.29 s/turn
  and max 0.42 s/turn (host CPU with no vCPU quota, not Kaggle hardware).
  File: `validation/pre-upload/kaggle-image-episode-self-seed7.json`.
- Pod Kaggle-mode episodes against the starter opponent, CPU only, 4 games (seeds
  20261100/20261101 in `validation/`, 20261200/20261201 in `validation/pre-upload/`),
  both seats: all ok, 719/719 calls, 0 caught errors, 0 budget passes, 0 invalid
  actions, steady p99 about 0.12 s/turn, one turn-0 call over 1 s (about 0.25 s
  of the 60 s overage used). The packaged agent won all 4 games. These are
  packaging checks, not strength claims.

## Submission

- Command: `kaggle competitions submit -c kaggriculture -f artifacts/08bc/submission.tar.gz -m "v3 PPO 08bc (4th promotion; 720-window self-play from BC; ckpt 08bc19ae; pkg sha 8283e676; branch kg/submit-08bc 9a743fa)"`
- Submitted 2026-09-30T15:14:16Z UTC. CLI output: "Successfully submitted to Kaggriculture".
- Submission ref **56711278**. Its status right after submitting was
  `SubmissionStatus.PENDING` with no score yet.

## Gaps

- The native module was built on the pod (glibc 2.39, with `--compatibility linux`
  and a symbol check), not inside Kaggle's image. The Docker image episode did
  load it and play a full game.
- The module is 14.3 MB, against 3.5 MB for the BC build. The cause was not
  investigated.
- No Kaggle score or episode results yet. The public score is probe evidence,
  not held-out qualification.
