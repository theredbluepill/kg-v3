---
type: "Decision"
title: "Spend Kaggle submission slots only on owner-agreed checkpoints"
description: "Owner rule for the final sprint (2026-09-30/10-01): \"do not spend submission slot unless we agreed tgt.\" Each Kaggle submission needs the owner's explicit agreement for that checkpoint; packaging or a panel result alone authorizes nothing. Outcome: 4 submissions, each separately agreed. 08bc probe (56711278, rule 1 off, score 1004.1 at ~19:30Z); 90M (56716929, panel 38-10, +1.94k); 170M (56720629, 48-0, +6.90k); 210M (56722061, the final slot at 23:50:02Z, submitted before its panel, which then went 48-0 at +7.51k on the same archive). 60M was packaged but, on the owner's word, not submitted. Final ladder scores of 90M, 170M and 210M were not recorded before the sprint closed."
tags: ["kaggriculture-v3", "decisions", "kaggle-runtime", "evaluation"]
status: "adopted"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-10-01"}
decider: "Owner, 2026-09-30/10-01 (Claude Code session): \"do not spend submission slot unless we agreed tgt.\". Per-checkpoint approvals are quoted in the body."
sources:
  - resource: "user-directive:2026-10-01:do-not-spend-submission-slot-unless-we-agreed-tgt"
  - resource: "user-directive:2026-09-30:package-the-08bc-and-submit-to-kaggle-for-probing"
  - resource: "user-directive:2026-10-01:when-90m-finished-anchors-submit-it-to-kaggle"
  - resource: "user-directive:2026-10-01:submit-170m-to-kaggle-after-anchor-panel"
  - resource: "user-directive:2026-10-01:package-210m-submit-then-eval"
  - resource: "user-directive:2026-10-01:you-can-package-but-not-need-to-submit-60m"
  - resource: "repository:ops/submit-90m-2026-10-01/receipt.md"
  - resource: "repository:ops/submit-170m-2026-10-01/receipt.md"
  - resource: "repository:ops/submit-210m-2026-10-01/receipt.md"
  - resource: "repository:ops/package-60m-2026-10-01/PACKAGE.md"
  - resource: "repository:ops/sprint-2026-09-30/sprint-facts.md"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/games-fixedshop-linux/90M-ft-on/summary.md"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/games-fixedshop-linux/170M-ft-on/summary.md"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/games-fixedshop-linux/210M-ft-on/summary.md"
  - resource: "repository:ops/sprint-2026-09-30/anchor-games/games-fixedshop-linux/210M-ft-on/package/PACKAGE.md"
  - resource: "repository:scripts/package_checkpoint.sh"
---

# Spend Kaggle submission slots only on owner-agreed checkpoints

## Decision

The owner, verbatim (Claude Code session, 2026-09-30/10-01):

> do not spend submission slot unless we agreed tgt.

This sharpens the standing rule of the [[evaluation-preserves-generality-and-evidence|evaluation Decision]], under which the owner decides what is submitted.

- A submission needs the owner's explicit agreement for that specific checkpoint.
- Agreement is not implied by a package, a panel result, a trainer promotion or a remaining daily slot.
- An agent may package and evaluate without asking. It submits only what was agreed, in the way it was agreed.

## Outcome: four submissions, each agreed separately

| Checkpoint | Owner's words (verbatim) | Ref | Archive | Time (UTC) | Evidence at submission |
| --- | --- | --- | --- | --- | --- |
| 08bc probe | "can you package the 08bc and submit to kaggle for probing? Note we have 4 submissions left, only use 1 of it." | 56711278 | `8283e676…` | 09-30 15:14 | Ran with rule 1 off; scored 1004.1 at about 19:30. Its panel was 0-48, −11.9k. |
| 90M | "when 90m finished anchors, submit it to kaggle to spend 1 slot." | 56716929 | `5e460d20…` | 09-30 19:26 | Panel 38-10, +1,943; +2,494 ± 761 vs 80M. |
| 170M | "submit 170m to kaggle after anchor panel" | 56720629 | `428a63d7…` | 09-30 22:47 | Panel 48-0, +6,904. |
| 210M | "沒事，打包210m, 提交，然後eval." (package 210M, submit, then evaluate) | 56722061 | `45efe071…` | 09-30 23:50:02 | Submitted before its panel, as asked; that was the final slot. The panel afterwards went 48-0, +7,505 on the same archive. |

- **Not submitted.** 60M was packaged with rule 1 baked on, and the owner said "you can package, but not need to submit 60m". It was not submitted.
- **Rules 1 and 2.** The 90M, 170M and 210M archives have rule 1 baked on and rule 2 off ([[../references/final-turn-liquidation-sells-the-shed-on-the-last-resolved-turn|rule 1 Reference]], [[../references/late-investment-filter-drops-only-purchases-that-cannot-sell-in-time|rule 2 Reference]]).
- **Leaderboard context** (from the sprint fact sheet). At 19:36Z there were 10,225 teams, and the team ranked 2,987 at 23:03Z. The deadline was 2026-09-30 23:59:00 UTC.
- **Packaging receipts** are in the [[../references/kaggle-packaging-reuses-the-starter-submission-path|packaging Reference]].

## Limits

- **Scores.** The final ladder scores and ratings of 90M, 170M and 210M are not recorded here. Only the 08bc score was read during the sprint.
- **No ladder A/B.** The ladder does not separate checkpoint strength from rule 1.
- **Panel limits.** The panels are selection evidence: three anchors and eight seeds, not held-out.
- **210M order.** The 210M submission preceded its own panel, by the owner's choice at the deadline.
