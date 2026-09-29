# PPO collapse ablation receipts (frozen)

**Frozen on 2026-09-30; later corrections go in a new file.** Do not edit any
file listed in `SHA256SUMS`. A correction, a late arm (for example an
ablation K) or a Codex review goes in a new, separately named file in this
directory, which cites the file it corrects.

- **What is here:** the 6.2 control (`../6.2/`) and ablations A–J of the
  pre-landing PPO collapse study on `kg/pod-ppo-prelanding` `e74d67e`, pod
  `aki4vy8kpfldpa` (2x RTX PRO 6000). Each arm's `run-statement.md` was
  committed before launch and its `result.md` ends with a "Receipt close"
  section: outcome, denominators, W&B link, spend and gaps.
- **Read first:** `final-report.md` (all arms, ranked causes, recipe J and
  its loss conditions), then `attribution.md` (control and A–I) and
  `comparison.md` (round 1).
- **Custody:** `SHA256SUMS` covers every file in this directory tree and in
  `../6.2/`, except `SHA256SUMS` itself. Check it from this directory with
  `shasum -a 256 -c SHA256SUMS` (or `sha256sum -c SHA256SUMS`). Checkpoints
  are not in git. They stay on the pod under `/root/runs/<arm>/`, and
  their hashes are in each arm's `pod-receipts/checkpoints.sha256`.
- **Status:** pre-landing diagnostic, not Codex-verified. Nothing here
  supports selection, ranking or submission.
