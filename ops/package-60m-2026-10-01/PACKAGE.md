# Kaggle package receipt

Built by `scripts/package_checkpoint.sh`. Nothing was uploaded or submitted.

| Item | Value |
| --- | --- |
| Command | `scripts/package_checkpoint.sh /Users/poonszesen/kg-v3-runs/earn720-r30e01w30-from-c50-4rank-20260930/ckpt-60M/checkpoint_00_060_018_944.pt /Users/poonszesen/kg-v3-runs/earn720-r30e01w30-from-c50-4rank-20260930/ckpt-60M/config.yaml /Users/poonszesen/kg-v3-pkg60/artifacts/60m-r1 --full-episode --seed 7 --final-turn-liquidation` |
| Archive | `submission.tar.gz`, 24725520 bytes |
| Archive sha256 | `7b72346f6c359d5856f30adbcdfe7cbf5513355f427766c2a0cc0ebddb7f8db9` |
| Inner manifest.json sha256 | `469a8a25a0ad7ef0de6790bc20d173c889842a3cb092d7fbc464ec74cbf08e62` |
| Checkpoint | `/Users/poonszesen/kg-v3-runs/earn720-r30e01w30-from-c50-4rank-20260930/ckpt-60M/checkpoint_00_060_018_944.pt` |
| Checkpoint sha256 | `20b1f795a58acf693cb581cf51447463370110d921e227038d53ed31b6da91a2` |
| Slim checkpoint sha256 | `ad06d493ebde02c019f05f04c0398dec31c38787b3549c90627ee2e21924f6bc` |
| Config | `/Users/poonszesen/kg-v3-runs/earn720-r30e01w30-from-c50-4rank-20260930/ckpt-60M/config.yaml`, sha256 `a60a577e6c202803eb83211c0e40c599e747f24f1ce124fac4245dc75079e27c` |
| Source | commit `de26cd68f48e157a58f134f32665d12593e42818`, clean tree |
| Native module | `3e5e4e556a58a3cb0595f422ec09dff4a2e2d73f018469583617f571a6bc664e` from native-cache (source commit `9a743fad991593fdd372115a48e3b372acf45b8b`, native sources identical at HEAD); fixed-opponent controllers compiled in: false |
| Endgame rules in main.py | rule 1 final-turn liquidation: baked on; rule 2 late-investment filter: environment (off) |
| Builder | `/Users/poonszesen/.cache/kg-v3/package-venv-py311-torch2.6.0/bin/python` |
| Kaggle image | `v700-kaggle-environments-official:28b6d8a` (`sha256:7ace6fabd2b9fe304fce3d3ded92d6817f5c5706b403fe43b3f353ed28711378`), linux/amd64 emulated, --network none --cpus=1.6 --memory=6.5g |

## Checks

- Verify: ok=True; 67 files re-hashed against the inner manifest; 210/210 model tensors equal to checkpoint['model'] (6,252,223 parameters); problems: none.
- Kaggle-image episode (strict, self-play, seed 7, episode_steps None): qualified=True; 720 steps, 0 bad statuses, final ['DONE', 'DONE']; seat 0: 719 calls, 0 exceptions, 0 invalid raw, 0 default passes, turn 0 1.44 s, steady p99 0.616 s, max 0.766 s; seat 1: 719 calls, 0 exceptions, 0 invalid raw, 0 default passes, turn 0 1.34 s, steady p99 0.584 s, max 0.789 s; banks [64622.0, 64622.0]; wall 559.335 s; owl.rs loaded from /kaggle_simulations/agent/owl/rs.abi3.so. Receipt: validation/kaggle-image-episode-self-seed7-full.json.

## Stage wall times (s)

| venv | preflight | build | verify | image | total |
| --- | --- | --- | --- | --- | --- |
| 1.0 | 0.7 | 2.6 | 1.0 | 567.1 | 572.5 |

## Limits

- Packaging and legality evidence only, not strength. Episode timings are emulated amd64 on the Mac, not Kaggle hardware.
- That Kaggle production runs this image is not established (see the runtime receipt).
- The native module was not rebuilt here; its custody is native-cache/manifest.json.
