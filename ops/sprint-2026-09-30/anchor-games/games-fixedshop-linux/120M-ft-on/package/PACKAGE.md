# Kaggle package receipt

Built by `scripts/package_checkpoint.sh`. Nothing was uploaded or submitted.

| Item | Value |
| --- | --- |
| Command | `scripts/package_checkpoint.sh /Users/poonszesen/kg-v3-runs/earn720-r30e01w30-8xh200-from-100M-sps-20261001/ckpt-120M/checkpoint_00_120_038_144.pt /Users/poonszesen/kg-v3-runs/earn720-r30e01w30-8xh200-from-100M-sps-20261001/ckpt-120M/config.yaml /Users/poonszesen/kg-v3-pkg60/artifacts/anchor-eval/120M-ft-on-b4b7d78433cb --episode-steps 40 --seed 7 --final-turn-liquidation` |
| Archive | `submission.tar.gz`, 24723650 bytes |
| Archive sha256 | `7ead9c1e7dc01ba4cb137600e1f2df6b159a373dddcef16329502e69189f6189` |
| Inner manifest.json sha256 | `8792a08de03077cb0025610a07e9e0ac2f58b843570f0af5d783738a67c91f13` |
| Checkpoint | `/Users/poonszesen/kg-v3-runs/earn720-r30e01w30-8xh200-from-100M-sps-20261001/ckpt-120M/checkpoint_00_120_038_144.pt` |
| Checkpoint sha256 | `b4b7d78433cb6cf4f0007a8ccf79e059583b959d794f6ef5b769d7e59ad25096` |
| Slim checkpoint sha256 | `04d86a012208b839eda11051e2764d91b839d3743efd5e19a094d664f67b40aa` |
| Config | `/Users/poonszesen/kg-v3-runs/earn720-r30e01w30-8xh200-from-100M-sps-20261001/ckpt-120M/config.yaml`, sha256 `87c0aebe34cf5f0ff4e845c29ae7640f8e865fd2f346b99549de7472d30870ec` |
| Source | commit `31443af308ae61625a5c7387ec2ca4af29bf5376`, clean tree |
| Native module | `3e5e4e556a58a3cb0595f422ec09dff4a2e2d73f018469583617f571a6bc664e` from native-cache (source commit `9a743fad991593fdd372115a48e3b372acf45b8b`, native sources identical at HEAD); fixed-opponent controllers compiled in: false |
| Endgame rules in main.py | rule 1 final-turn liquidation: baked on; rule 2 late-investment filter: environment (off) |
| Builder | `/Users/poonszesen/.cache/kg-v3/package-venv-py311-torch2.6.0/bin/python` |
| Kaggle image | `v700-kaggle-environments-official:28b6d8a` (`sha256:7ace6fabd2b9fe304fce3d3ded92d6817f5c5706b403fe43b3f353ed28711378`), linux/amd64 emulated, --network none --cpus=1.6 --memory=6.5g |

## Checks

- Verify: ok=True; 67 files re-hashed against the inner manifest; 210/210 model tensors equal to checkpoint['model'] (6,252,223 parameters); problems: none.
- Kaggle-image episode (strict, self-play, seed 7, episode_steps 40): qualified=True; 40 steps, 0 bad statuses, final ['DONE', 'DONE']; seat 0: 39 calls, 0 exceptions, 0 invalid raw, 0 default passes, turn 0 0.84 s, steady p99 0.264 s, max 0.264 s; seat 1: 39 calls, 0 exceptions, 0 invalid raw, 0 default passes, turn 0 0.73 s, steady p99 0.248 s, max 0.248 s; banks [621.0, 621.0]; wall 18.826 s; owl.rs loaded from /kaggle_simulations/agent/owl/rs.abi3.so. Receipt: validation/kaggle-image-episode-self-seed7-steps40.json.

## Stage wall times (s)

| venv | preflight | build | verify | image | total |
| --- | --- | --- | --- | --- | --- |
| 0.8 | 0.5 | 1.6 | 0.6 | 23.0 | 26.5 |

## Limits

- Packaging and legality evidence only, not strength. Episode timings are emulated amd64 on the Mac, not Kaggle hardware.
- That Kaggle production runs this image is not established (see the runtime receipt).
- The native module was not rebuilt here; its custody is native-cache/manifest.json.
- Bounded episode (40 turns); run with --full-episode for a 720-turn game.
