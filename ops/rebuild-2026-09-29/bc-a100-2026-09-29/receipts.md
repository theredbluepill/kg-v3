# BC data receipts: A100 pod, 2026-09-29 (top-1 team, winning seats)

Working artifact for the BC-now lane. It holds counts and hashes only; the
shards stay on the pod's network volume.

## Scope

- Imitated player: the live Kaggle leaderboard #1 team "M & M & P & Q"
  (public leaderboard, 3086.5 on 2026-09-29). Only that team's winning seats
  are policy seats (both seats on a draw; none occurred).
- Days 2026-09-22..28 from `/root/bc-archives` (7 day archives, 4198 episodes
  listed; archive SHA-256 values are in the pod manifest under
  `source.archive_sha256`).
- Source: `kg/rebuild-bc-now` at `89ca39c` (Codex verify-5.1-team-filter-r2
  APPROVE). The pod checkout `/root/kg-v3-bc` was fast-forwarded from `49255ac`
  by bundle; `git diff --stat 49255ac 89ca39c` touches only
  `scripts/kaggriculture_prepare_bc.py` and its test, so the existing
  `owl_rs` build was reused without a maturin rebuild.

## Superseded mixed-winner shards

`/workspace/kg-v3-bc-2026-09-29/shards/` holds the earlier mixed-winner
preparation (4198 episodes, stride 10, manifest SHA-256
`bdc5978b1b49d0b822936d9df88c41173b9485432ffc281b438db153d1393986`). It was
prepared but superseded by the owner's one-player update and was not trained
on. It is left in place, untouched, for reference.

## Pairing

Not rerun. `pairing.json` is a copy of the earlier run's
`/workspace/kg-v3-bc-2026-09-29/shards/pairing.json` (SHA-256
`a2c2db5786f91129897aa6c60e757561ca2887a688b66548ee9bc9a5a7128add`):
5743/5752 transitions match (0.9984) over 8 episodes, kaggle_environments
1.32.7. The nine mismatches name only private0/private1 at day-end turns.
This is a pairing diagnostic; it does not establish parity or prove label
correctness.

## Memory check and stride

- `bytes_per_turn_row` = 216612 (printed at startup).
- Stride 1 would give 523 x 719 = 376037 rows, about 81.5 GB (75.9 GiB) resident,
  over the about-60 GB budget of the pod's 125 GB container limit.
- Deviation from the brief: `--resident-budget-gib 60` was **not** used.
  `stride_for_budget` counts every listed episode (4198 x 719), not the team's
  kept episodes, so it would have chosen stride 11 (about 34k rows). The same
  formula applied to the 523 kept episodes gives
  ceil(376037 / floor(60 GiB / 216612) = 297418) = 2, so the run used
  `--turn-stride 2` explicitly. Resident size: 188021 rows x 216612 = about
  40.7 GB.
- Open gap: `--resident-budget-gib` over-strides whenever `--team` filters most
  episodes. It was not changed here.

## Run

- Command: `run-top1.sh` (launched with nohup; started 2026-09-29T14:07:59Z,
  manifest written 14:13:58Z, about 6 minutes, 12 workers).
- Output: `/workspace/kg-v3-bc-2026-09-29/shards-top1/`.
- Log: `/workspace/kg-v3-bc-2026-09-29/logs/prepare-top1.log` (copied as
  `prepare-top1.log`).

## Manifest

- SHA-256 of `shards-top1/manifest.json`:
  `ba5fe1c417741c587473b5696a6ca55227240b394236948cb4f4f5c4ed9a3036`.
- `run.git_head` `89ca39cfd259de328338c1b15c5251f9a9be8b57`; `turn_stride` 2;
  `validation_fraction` 0.03; `team_sha256`
  `bdf5243c27ea6e5caf82529c401a47e30f7b9e20731c385ea6a626ac09d8ea24`; the
  recorded command redacts the team value as `<team>`.

| Quantity | Count |
| --- | ---: |
| Episodes listed | 4198 |
| Team episodes seen (kept + team lost) | 842 |
| Admitted (team won) | 523 |
| Rejected: team absent | 3356 |
| Rejected: team lost | 319 |
| Turn rejections / normalizations | 0 / 0 |
| Train episodes / rows | 507 / 182273 |
| Validation episodes / rows | 16 / 5748 |
| Draw episodes (train / validation) | 0 / 0 |
| Shard bytes, train | 320964518 |
| Shard bytes, validation | 10248233 |
| Shard bytes, total | 331212751 |

Validation has 16 episodes, so no `--validation-fraction 0.05` rerun was needed.

## Files here

- `manifest-summary.json`: the manifest's `schema`, `run`, `totals`,
  `episode_rejections`, `turn_rejections` and `normalizations` (no per-episode
  lists).
- `pairing.json`, `prepare-top1.log`, `run-top1.sh`.
