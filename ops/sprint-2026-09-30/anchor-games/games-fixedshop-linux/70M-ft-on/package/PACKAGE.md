# Kaggle package receipt

Built by `scripts/package_checkpoint.sh`. Nothing was uploaded or submitted.

| Item | Value |
| --- | --- |
| Command | `scripts/package_checkpoint.sh /Users/poonszesen/kg-v3-runs/earn720-r30e01w30-8xh200-from-60M-20261001/ckpt-70M/checkpoint_00_070_041_344.pt /Users/poonszesen/kg-v3-runs/earn720-r30e01w30-8xh200-from-60M-20261001/ckpt-70M/config.yaml /Users/poonszesen/kg-v3-pkg60/artifacts/anchor-eval/70M-ft-on-4c8b94831dd2 --episode-steps 40 --seed 7 --final-turn-liquidation` |
| Archive | `submission.tar.gz`, 24723951 bytes |
| Archive sha256 | `7cbb97ed38e5058b423f7e4beed4f9c767bf5315f68033f9c0ccd1e2c6b16d26` |
| Inner manifest.json sha256 | `e02a51cbf9da2254727e0216f4b24a24eeeaf0f3ac51759e7bf87fb294f1314d` |
| Checkpoint | `/Users/poonszesen/kg-v3-runs/earn720-r30e01w30-8xh200-from-60M-20261001/ckpt-70M/checkpoint_00_070_041_344.pt` |
| Checkpoint sha256 | `4c8b94831dd2a7b0688646555e19fa47170da538bd37d14296f529b9745238e9` |
| Slim checkpoint sha256 | `addbdb38de0d508c71527d80f19485e60d0080ef77052e1ee4892f7b6783dfba` |
| Config | `/Users/poonszesen/kg-v3-runs/earn720-r30e01w30-8xh200-from-60M-20261001/ckpt-70M/config.yaml`, sha256 `f7a1349a651904e9374c39a3aff4fa2460613eaf011d9fc319017af24eae377c` |
| Source | commit `cfc45f0e4d0e57819f27ea061472a250486bd938`, clean tree |
| Native module | `3e5e4e556a58a3cb0595f422ec09dff4a2e2d73f018469583617f571a6bc664e` from native-cache (source commit `9a743fad991593fdd372115a48e3b372acf45b8b`, native sources identical at HEAD); fixed-opponent controllers compiled in: false |
| Endgame rules in main.py | rule 1 final-turn liquidation: baked on; rule 2 late-investment filter: environment (off) |
| Builder | `/Users/poonszesen/.cache/kg-v3/package-venv-py311-torch2.6.0/bin/python` |
| Kaggle image | `v700-kaggle-environments-official:28b6d8a` (`sha256:7ace6fabd2b9fe304fce3d3ded92d6817f5c5706b403fe43b3f353ed28711378`), linux/amd64 emulated, --network none --cpus=1.6 --memory=6.5g |

## Checks

- Verify: ok=True; 67 files re-hashed against the inner manifest; 210/210 model tensors equal to checkpoint['model'] (6,252,223 parameters); problems: none.
- Kaggle-image episode (strict, self-play, seed 7, episode_steps 40): qualified=True; 40 steps, 0 bad statuses, final ['DONE', 'DONE']; seat 0: 39 calls, 0 exceptions, 0 invalid raw, 0 default passes, turn 0 1.56 s, steady p99 0.500 s, max 0.500 s; seat 1: 39 calls, 0 exceptions, 0 invalid raw, 0 default passes, turn 0 1.37 s, steady p99 0.488 s, max 0.488 s; banks [643.0, 643.0]; wall 33.913 s; owl.rs loaded from /kaggle_simulations/agent/owl/rs.abi3.so. Receipt: validation/kaggle-image-episode-self-seed7-steps40.json.

## Stage wall times (s)

| venv | preflight | build | verify | image | total |
| --- | --- | --- | --- | --- | --- |
| 1.1 | 0.9 | 2.8 | 1.1 | 41.4 | 47.4 |

## Limits

- Packaging and legality evidence only, not strength. Episode timings are emulated amd64 on the Mac, not Kaggle hardware.
- That Kaggle production runs this image is not established (see the runtime receipt).
- The native module was not rebuilt here; its custody is native-cache/manifest.json.
- Bounded episode (40 turns); run with --full-episode for a 720-turn game.
