# Kaggle package receipt

Built by `scripts/package_checkpoint.sh`. Nothing was uploaded or submitted.

| Item | Value |
| --- | --- |
| Command | `scripts/package_checkpoint.sh /Users/poonszesen/kg-v3-runs/earn720-r30e01w30-8xh200-from-60M-20261001/ckpt-90M/checkpoint_00_090_086_144.pt /Users/poonszesen/kg-v3-runs/earn720-r30e01w30-8xh200-from-60M-20261001/ckpt-90M/config.yaml /Users/poonszesen/kg-v3-pkg60/artifacts/anchor-eval/90M-ft-on-6b196ef33b3d --episode-steps 40 --seed 7 --final-turn-liquidation` |
| Archive | `submission.tar.gz`, 24724581 bytes |
| Archive sha256 | `5e460d20ddfc0514e24c072cdb6d1846c2f7eda46a1bdfaef518d9a7482d44d2` |
| Inner manifest.json sha256 | `317d06bb4411e6a7a85c1c9bbc24e4d1312e527be28f11dfea1fdd6875c89275` |
| Checkpoint | `/Users/poonszesen/kg-v3-runs/earn720-r30e01w30-8xh200-from-60M-20261001/ckpt-90M/checkpoint_00_090_086_144.pt` |
| Checkpoint sha256 | `6b196ef33b3db9dab74220421f147d4bb9cdb49562115e3e7ac822947a11d19a` |
| Slim checkpoint sha256 | `f5cedc1647a2122cda1abd043ef5faf57f58b455f504a570813c064ddc23fd39` |
| Config | `/Users/poonszesen/kg-v3-runs/earn720-r30e01w30-8xh200-from-60M-20261001/ckpt-90M/config.yaml`, sha256 `f7a1349a651904e9374c39a3aff4fa2460613eaf011d9fc319017af24eae377c` |
| Source | commit `cfc45f0e4d0e57819f27ea061472a250486bd938`, clean tree |
| Native module | `3e5e4e556a58a3cb0595f422ec09dff4a2e2d73f018469583617f571a6bc664e` from native-cache (source commit `9a743fad991593fdd372115a48e3b372acf45b8b`, native sources identical at HEAD); fixed-opponent controllers compiled in: false |
| Endgame rules in main.py | rule 1 final-turn liquidation: baked on; rule 2 late-investment filter: environment (off) |
| Builder | `/Users/poonszesen/.cache/kg-v3/package-venv-py311-torch2.6.0/bin/python` |
| Kaggle image | `v700-kaggle-environments-official:28b6d8a` (`sha256:7ace6fabd2b9fe304fce3d3ded92d6817f5c5706b403fe43b3f353ed28711378`), linux/amd64 emulated, --network none --cpus=1.6 --memory=6.5g |

## Checks

- Verify: ok=True; 67 files re-hashed against the inner manifest; 210/210 model tensors equal to checkpoint['model'] (6,252,223 parameters); problems: none.
- Kaggle-image episode (strict, self-play, seed 7, episode_steps 40): qualified=True; 40 steps, 0 bad statuses, final ['DONE', 'DONE']; seat 0: 39 calls, 0 exceptions, 0 invalid raw, 0 default passes, turn 0 1.05 s, steady p99 0.292 s, max 0.292 s; seat 1: 39 calls, 0 exceptions, 0 invalid raw, 0 default passes, turn 0 0.91 s, steady p99 0.303 s, max 0.303 s; banks [698.0, 698.0]; wall 21.847 s; owl.rs loaded from /kaggle_simulations/agent/owl/rs.abi3.so. Receipt: validation/kaggle-image-episode-self-seed7-steps40.json.

## Stage wall times (s)

| venv | preflight | build | verify | image | total |
| --- | --- | --- | --- | --- | --- |
| 0.8 | 0.6 | 1.9 | 0.7 | 26.9 | 31.0 |

## Limits

- Packaging and legality evidence only, not strength. Episode timings are emulated amd64 on the Mac, not Kaggle hardware.
- That Kaggle production runs this image is not established (see the runtime receipt).
- The native module was not rebuilt here; its custody is native-cache/manifest.json.
- Bounded episode (40 turns); run with --full-episode for a 720-turn game.
