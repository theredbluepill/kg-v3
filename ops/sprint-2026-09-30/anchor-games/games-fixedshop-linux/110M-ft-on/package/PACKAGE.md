# Kaggle package receipt

Built by `scripts/package_checkpoint.sh`. Nothing was uploaded or submitted.

| Item | Value |
| --- | --- |
| Command | `scripts/package_checkpoint.sh /Users/poonszesen/kg-v3-runs/earn720-r30e01w30-8xh200-from-100M-sps-20261001/ckpt-110M/checkpoint_00_110_015_744.pt /Users/poonszesen/kg-v3-runs/earn720-r30e01w30-8xh200-from-100M-sps-20261001/ckpt-110M/config.yaml /Users/poonszesen/kg-v3-pkg60/artifacts/anchor-eval/110M-ft-on-4c85eb6a7e3d --episode-steps 40 --seed 7 --final-turn-liquidation` |
| Archive | `submission.tar.gz`, 24723848 bytes |
| Archive sha256 | `6206b7c5a21ac31816f7abd04c3d6231b5fb884c428656de9f164cdea9233437` |
| Inner manifest.json sha256 | `d4fd94208b28f6c48cfa0c186c970eec3fe979bbc3267d9a27959a796224803d` |
| Checkpoint | `/Users/poonszesen/kg-v3-runs/earn720-r30e01w30-8xh200-from-100M-sps-20261001/ckpt-110M/checkpoint_00_110_015_744.pt` |
| Checkpoint sha256 | `4c85eb6a7e3dcd7a82e05480d8ffad5e6888dfb286cdb1f25f34decf5c250b0b` |
| Slim checkpoint sha256 | `e15f4ccf901e58a21930e331862c34e5d3d3f2ae4920e4dc3175cfb780f50e80` |
| Config | `/Users/poonszesen/kg-v3-runs/earn720-r30e01w30-8xh200-from-100M-sps-20261001/ckpt-110M/config.yaml`, sha256 `87c0aebe34cf5f0ff4e845c29ae7640f8e865fd2f346b99549de7472d30870ec` |
| Source | commit `31443af308ae61625a5c7387ec2ca4af29bf5376`, clean tree |
| Native module | `3e5e4e556a58a3cb0595f422ec09dff4a2e2d73f018469583617f571a6bc664e` from native-cache (source commit `9a743fad991593fdd372115a48e3b372acf45b8b`, native sources identical at HEAD); fixed-opponent controllers compiled in: false |
| Endgame rules in main.py | rule 1 final-turn liquidation: baked on; rule 2 late-investment filter: environment (off) |
| Builder | `/Users/poonszesen/.cache/kg-v3/package-venv-py311-torch2.6.0/bin/python` |
| Kaggle image | `v700-kaggle-environments-official:28b6d8a` (`sha256:7ace6fabd2b9fe304fce3d3ded92d6817f5c5706b403fe43b3f353ed28711378`), linux/amd64 emulated, --network none --cpus=1.6 --memory=6.5g |

## Checks

- Verify: ok=True; 67 files re-hashed against the inner manifest; 210/210 model tensors equal to checkpoint['model'] (6,252,223 parameters); problems: none.
- Kaggle-image episode (strict, self-play, seed 7, episode_steps 40): qualified=True; 40 steps, 0 bad statuses, final ['DONE', 'DONE']; seat 0: 39 calls, 0 exceptions, 0 invalid raw, 0 default passes, turn 0 0.83 s, steady p99 0.219 s, max 0.219 s; seat 1: 39 calls, 0 exceptions, 0 invalid raw, 0 default passes, turn 0 0.72 s, steady p99 0.223 s, max 0.223 s; banks [701.0, 701.0]; wall 17.859 s; owl.rs loaded from /kaggle_simulations/agent/owl/rs.abi3.so. Receipt: validation/kaggle-image-episode-self-seed7-steps40.json.

## Stage wall times (s)

| venv | preflight | build | verify | image | total |
| --- | --- | --- | --- | --- | --- |
| 0.7 | 0.5 | 1.5 | 0.6 | 22.1 | 25.5 |

## Limits

- Packaging and legality evidence only, not strength. Episode timings are emulated amd64 on the Mac, not Kaggle hardware.
- That Kaggle production runs this image is not established (see the runtime receipt).
- The native module was not rebuilt here; its custody is native-cache/manifest.json.
- Bounded episode (40 turns); run with --full-episode for a 720-turn game.
