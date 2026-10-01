# Kaggle package receipt

Built by `scripts/package_checkpoint.sh`. Nothing was uploaded or submitted.

| Item | Value |
| --- | --- |
| Command | `scripts/package_checkpoint.sh /Users/poonszesen/kg-v3-runs/earn720-r30e01w30-8xh200-from-130M-sps-20261001/ckpt-210M-submit/checkpoint_00_210_009_344.pt /Users/poonszesen/kg-v3-runs/earn720-r30e01w30-8xh200-from-130M-sps-20261001/ckpt-210M-submit/config.yaml /Users/poonszesen/kg-v3-pkg60/artifacts/submit-210m --episode-steps 40 --seed 7 --final-turn-liquidation` |
| Archive | `submission.tar.gz`, 24721884 bytes |
| Archive sha256 | `45efe071e0c92954995440109cc4b0f7d034b746540ed4d63b36616084f9c36a` |
| Inner manifest.json sha256 | `c1530dd19df8ebc94d51fee0988c468ad61930b76f345bf8ea028c466c7b039e` |
| Checkpoint | `/Users/poonszesen/kg-v3-runs/earn720-r30e01w30-8xh200-from-130M-sps-20261001/ckpt-210M-submit/checkpoint_00_210_009_344.pt` |
| Checkpoint sha256 | `790b64d80d60e8e81c0736b3a9ad925a63f072abcced8a2c100075e3677118cd` |
| Slim checkpoint sha256 | `c1553285fd2b3b2151163bb4137506737cda8f22694b14db6da73edb3059941a` |
| Config | `/Users/poonszesen/kg-v3-runs/earn720-r30e01w30-8xh200-from-130M-sps-20261001/ckpt-210M-submit/config.yaml`, sha256 `87c0aebe34cf5f0ff4e845c29ae7640f8e865fd2f346b99549de7472d30870ec` |
| Source | commit `a22e6c4d7415f7b98fc763c35aa45607ef9a0bb0`, clean tree |
| Native module | `3e5e4e556a58a3cb0595f422ec09dff4a2e2d73f018469583617f571a6bc664e` from native-cache (source commit `9a743fad991593fdd372115a48e3b372acf45b8b`, native sources identical at HEAD); fixed-opponent controllers compiled in: false |
| Endgame rules in main.py | rule 1 final-turn liquidation: baked on; rule 2 late-investment filter: environment (off) |
| Builder | `/Users/poonszesen/.cache/kg-v3/package-venv-py311-torch2.6.0/bin/python` |
| Kaggle image | `v700-kaggle-environments-official:28b6d8a` (`sha256:7ace6fabd2b9fe304fce3d3ded92d6817f5c5706b403fe43b3f353ed28711378`), linux/amd64 emulated, --network none --cpus=1.6 --memory=6.5g |

## Checks

- Verify: ok=True; 67 files re-hashed against the inner manifest; 210/210 model tensors equal to checkpoint['model'] (6,252,223 parameters); problems: none.
- Kaggle-image episode (strict, self-play, seed 7, episode_steps 40): qualified=True; 40 steps, 0 bad statuses, final ['DONE', 'DONE']; seat 0: 39 calls, 0 exceptions, 0 invalid raw, 0 default passes, turn 0 0.94 s, steady p99 0.244 s, max 0.244 s; seat 1: 39 calls, 0 exceptions, 0 invalid raw, 0 default passes, turn 0 0.76 s, steady p99 0.219 s, max 0.219 s; banks [572.0, 572.0]; wall 18.199 s; owl.rs loaded from /kaggle_simulations/agent/owl/rs.abi3.so. Receipt: validation/kaggle-image-episode-self-seed7-steps40.json.

## Stage wall times (s)

| venv | preflight | build | verify | image | total |
| --- | --- | --- | --- | --- | --- |
| 0.8 | 0.5 | 1.6 | 0.6 | 22.9 | 26.4 |

## Limits

- Packaging and legality evidence only, not strength. Episode timings are emulated amd64 on the Mac, not Kaggle hardware.
- That Kaggle production runs this image is not established (see the runtime receipt).
- The native module was not rebuilt here; its custody is native-cache/manifest.json.
- Bounded episode (40 turns); run with --full-episode for a 720-turn game.
