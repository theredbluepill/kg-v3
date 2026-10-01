# Kaggle package receipt

Built by `scripts/package_checkpoint.sh`. Nothing was uploaded or submitted.

| Item | Value |
| --- | --- |
| Command | `scripts/package_checkpoint.sh /Users/poonszesen/kg-v3-runs/earn720-r30e01w30-8xh200-from-90M-20261001/ckpt-100M/checkpoint_00_100_108_544.pt /Users/poonszesen/kg-v3-runs/earn720-r30e01w30-8xh200-from-90M-20261001/ckpt-100M/config.yaml /Users/poonszesen/kg-v3-pkg60/artifacts/anchor-eval/100M-ft-on-8e520566c705 --episode-steps 40 --seed 7 --final-turn-liquidation` |
| Archive | `submission.tar.gz`, 24723793 bytes |
| Archive sha256 | `ec801999c7c770728bc4231dabdbe154cd9e99cb5295b56246fe204dba0240b4` |
| Inner manifest.json sha256 | `5a3d948f31066208f240c34d40de9b4eb659d9378d76f1fd7570d5f316dab475` |
| Checkpoint | `/Users/poonszesen/kg-v3-runs/earn720-r30e01w30-8xh200-from-90M-20261001/ckpt-100M/checkpoint_00_100_108_544.pt` |
| Checkpoint sha256 | `8e520566c705cb0afa083962766d6879942f4ff16cf3844bb29e2bcb0a138d81` |
| Slim checkpoint sha256 | `c702f5346157eb026f4c64139e782ff0cbc30a690176773de525adae8da607d0` |
| Config | `/Users/poonszesen/kg-v3-runs/earn720-r30e01w30-8xh200-from-90M-20261001/ckpt-100M/config.yaml`, sha256 `f7a1349a651904e9374c39a3aff4fa2460613eaf011d9fc319017af24eae377c` |
| Source | commit `31443af308ae61625a5c7387ec2ca4af29bf5376`, clean tree |
| Native module | `3e5e4e556a58a3cb0595f422ec09dff4a2e2d73f018469583617f571a6bc664e` from native-cache (source commit `9a743fad991593fdd372115a48e3b372acf45b8b`, native sources identical at HEAD); fixed-opponent controllers compiled in: false |
| Endgame rules in main.py | rule 1 final-turn liquidation: baked on; rule 2 late-investment filter: environment (off) |
| Builder | `/Users/poonszesen/.cache/kg-v3/package-venv-py311-torch2.6.0/bin/python` |
| Kaggle image | `v700-kaggle-environments-official:28b6d8a` (`sha256:7ace6fabd2b9fe304fce3d3ded92d6817f5c5706b403fe43b3f353ed28711378`), linux/amd64 emulated, --network none --cpus=1.6 --memory=6.5g |

## Checks

- Verify: ok=True; 67 files re-hashed against the inner manifest; 210/210 model tensors equal to checkpoint['model'] (6,252,223 parameters); problems: none.
- Kaggle-image episode (strict, self-play, seed 7, episode_steps 40): qualified=True; 40 steps, 0 bad statuses, final ['DONE', 'DONE']; seat 0: 39 calls, 0 exceptions, 0 invalid raw, 0 default passes, turn 0 0.82 s, steady p99 0.218 s, max 0.218 s; seat 1: 39 calls, 0 exceptions, 0 invalid raw, 0 default passes, turn 0 0.71 s, steady p99 0.221 s, max 0.221 s; banks [748.0, 748.0]; wall 17.942 s; owl.rs loaded from /kaggle_simulations/agent/owl/rs.abi3.so. Receipt: validation/kaggle-image-episode-self-seed7-steps40.json.

## Stage wall times (s)

| venv | preflight | build | verify | image | total |
| --- | --- | --- | --- | --- | --- |
| 0.8 | 0.5 | 1.6 | 0.6 | 22.1 | 25.6 |

## Limits

- Packaging and legality evidence only, not strength. Episode timings are emulated amd64 on the Mac, not Kaggle hardware.
- That Kaggle production runs this image is not established (see the runtime receipt).
- The native module was not rebuilt here; its custody is native-cache/manifest.json.
- Bounded episode (40 turns); run with --full-episode for a 720-turn game.
