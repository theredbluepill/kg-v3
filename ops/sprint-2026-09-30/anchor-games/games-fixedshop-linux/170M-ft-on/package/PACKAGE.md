# Kaggle package receipt

Built by `scripts/package_checkpoint.sh`. Nothing was uploaded or submitted.

| Item | Value |
| --- | --- |
| Command | `scripts/package_checkpoint.sh /Users/poonszesen/kg-v3-runs/earn720-r30e01w30-8xh200-from-130M-sps-20261001/ckpt-170M/checkpoint_00_170_034_944.pt /Users/poonszesen/kg-v3-runs/earn720-r30e01w30-8xh200-from-130M-sps-20261001/ckpt-170M/config.yaml /Users/poonszesen/kg-v3-pkg60/artifacts/anchor-eval/170M-ft-on-693e30e7f79f --episode-steps 40 --seed 7 --final-turn-liquidation` |
| Archive | `submission.tar.gz`, 24722723 bytes |
| Archive sha256 | `428a63d706cf528e2c0cad17587452e07c853a66971eb14a1ad219ca770ce3c3` |
| Inner manifest.json sha256 | `57d7f47fa27c88c05bbd461a1e53c24cfa22b11acd3f013a49ce5ccf0625dfac` |
| Checkpoint | `/Users/poonszesen/kg-v3-runs/earn720-r30e01w30-8xh200-from-130M-sps-20261001/ckpt-170M/checkpoint_00_170_034_944.pt` |
| Checkpoint sha256 | `693e30e7f79f70df3646566919d72bd285d9d77764e1a5181379650e5f486ca3` |
| Slim checkpoint sha256 | `e665075c4eba1ae3799bb8442555651dcac1075139d17b0853ffe32335a5ce9a` |
| Config | `/Users/poonszesen/kg-v3-runs/earn720-r30e01w30-8xh200-from-130M-sps-20261001/ckpt-170M/config.yaml`, sha256 `87c0aebe34cf5f0ff4e845c29ae7640f8e865fd2f346b99549de7472d30870ec` |
| Source | commit `31443af308ae61625a5c7387ec2ca4af29bf5376`, clean tree |
| Native module | `3e5e4e556a58a3cb0595f422ec09dff4a2e2d73f018469583617f571a6bc664e` from native-cache (source commit `9a743fad991593fdd372115a48e3b372acf45b8b`, native sources identical at HEAD); fixed-opponent controllers compiled in: false |
| Endgame rules in main.py | rule 1 final-turn liquidation: baked on; rule 2 late-investment filter: environment (off) |
| Builder | `/Users/poonszesen/.cache/kg-v3/package-venv-py311-torch2.6.0/bin/python` |
| Kaggle image | `v700-kaggle-environments-official:28b6d8a` (`sha256:7ace6fabd2b9fe304fce3d3ded92d6817f5c5706b403fe43b3f353ed28711378`), linux/amd64 emulated, --network none --cpus=1.6 --memory=6.5g |

## Checks

- Verify: ok=True; 67 files re-hashed against the inner manifest; 210/210 model tensors equal to checkpoint['model'] (6,252,223 parameters); problems: none.
- Kaggle-image episode (strict, self-play, seed 7, episode_steps 40): qualified=True; 40 steps, 0 bad statuses, final ['DONE', 'DONE']; seat 0: 39 calls, 0 exceptions, 0 invalid raw, 0 default passes, turn 0 0.85 s, steady p99 0.236 s, max 0.236 s; seat 1: 39 calls, 0 exceptions, 0 invalid raw, 0 default passes, turn 0 0.71 s, steady p99 0.225 s, max 0.225 s; banks [718.0, 718.0]; wall 18.078 s; owl.rs loaded from /kaggle_simulations/agent/owl/rs.abi3.so. Receipt: validation/kaggle-image-episode-self-seed7-steps40.json.

## Stage wall times (s)

| venv | preflight | build | verify | image | total |
| --- | --- | --- | --- | --- | --- |
| 0.8 | 0.5 | 1.6 | 0.6 | 22.5 | 26.0 |

## Limits

- Packaging and legality evidence only, not strength. Episode timings are emulated amd64 on the Mac, not Kaggle hardware.
- That Kaggle production runs this image is not established (see the runtime receipt).
- The native module was not rebuilt here; its custody is native-cache/manifest.json.
- Bounded episode (40 turns); run with --full-episode for a 720-turn game.
