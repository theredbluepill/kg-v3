## Checks

```json
{
 "games": 48,
 "b_qualified": 48,
 "b_calls_719": 48,
 "b_exceptions": 0,
 "b_invalid": 0,
 "b_default_pass": 0,
 "b_bad_statuses": 0,
 "b_single_manifest": 1,
 "b_single_engine": 1,
 "tap_installed": 48,
 "switch_lines_both_on": 48,
 "rule1_liquidation_line_once": 48,
 "tap_count_equals_stdout_count": 48,
 "blocked_orders_total": 24,
 "games_with_blocks": 16,
 "no_block_games_identical_to_a": 32,
 "no_block_games": 32,
 "block_games_first_diff_is_first_block": 16,
 "block_games_first_block_market_equals_a": 16,
 "every_blocked_order_rederives_a_reason": 48,
 "d_b_a_negative": 8,
 "d_b_a_zero": 32,
 "d_b_a_positive": 8,
 "win_to_loss_vs_a": 0,
 "loss_to_win_vs_a": 0
}
```

## Summary

| anchor | n | W-L OFF | W-L A (r1) | W-L B (r1+r2) | mean gap A | mean gap B | B − A mean ± SE | B − A min / max | B − OFF mean ± SE | games with blocks | blocked orders | unsold shed/carried/tile A | B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| smaller_market_shock | 16 | 4-12 | 4-12 | 4-12 | -3,425 | -3,455 | -30 ± 27 | -220 / +111 | +917 ± 218 | 6 | 10 | 0.0 / 8.8 / 9.3 | 0.0 / 11.9 / 8.6 |
| cha22 | 16 | 2-14 | 2-14 | 2-14 | -8,777 | -8,774 | +3 ± 15 | -69 / +90 | +681 ± 124 | 4 | 6 | 0.0 / 10.9 / 12.6 | 0.0 / 10.5 / 12.5 |
| v56 | 16 | 0-16 | 0-16 | 0-16 | -9,356 | -9,331 | +25 ± 33 | -114 / +404 | +1,021 ± 226 | 6 | 8 | 0.0 / 14.3 / 10.9 | 0.0 / 14.8 / 11.9 |
| all | 48 | 6-42 | 6-42 | 6-42 | -7,186 | -7,187 | -1 ± 21 | -220 / +404 | +873 ± 152 | 16 | 24 | 0.0 / 11.3 / 10.9 | 0.0 / 12.4 / 11.0 |

## Blocked orders by category

| anchor | category | blocked orders |
|---|---|---|
| cha22 | BUY_SEED WHEAT | 6 |
| smaller_market_shock | BUY_SEED WHEAT | 10 |
| v56 | BUY_SEED WHEAT | 8 |

## Every blocked order

| game | step (day h) | blocked order | reason |
|---|---|---|---|
| c50-smaller_market_shock-s93003-seat0 | 680 (d28 h8) | `["BUY_SEED", "WHEAT", 1]` | BUY_SEED WHEAT: no harvest can sell by step 718 |
| c50-smaller_market_shock-s93003-seat1 | 680 (d28 h8) | `["BUY_SEED", "WHEAT", 1]` | BUY_SEED WHEAT: no harvest can sell by step 718 |
| c50-smaller_market_shock-s93006-seat0 | 680 (d28 h8) | `["BUY_SEED", "WHEAT", 1]` | BUY_SEED WHEAT: no harvest can sell by step 718 |
| c50-smaller_market_shock-s93006-seat1 | 680 (d28 h8) | `["BUY_SEED", "WHEAT", 1]` | BUY_SEED WHEAT: no harvest can sell by step 718 |
| c50-smaller_market_shock-s93008-seat0 | 677 (d28 h5) | `["BUY_SEED", "WHEAT", 1]` | BUY_SEED WHEAT: no harvest can sell by step 718 |
| c50-smaller_market_shock-s93008-seat0 | 677 (d28 h5) | `["BUY_SEED", "WHEAT", 1]` | BUY_SEED WHEAT: no harvest can sell by step 718 |
| c50-smaller_market_shock-s93008-seat0 | 680 (d28 h8) | `["BUY_SEED", "WHEAT", 1]` | BUY_SEED WHEAT: no harvest can sell by step 718 |
| c50-smaller_market_shock-s93008-seat1 | 677 (d28 h5) | `["BUY_SEED", "WHEAT", 1]` | BUY_SEED WHEAT: no harvest can sell by step 718 |
| c50-smaller_market_shock-s93008-seat1 | 677 (d28 h5) | `["BUY_SEED", "WHEAT", 1]` | BUY_SEED WHEAT: no harvest can sell by step 718 |
| c50-smaller_market_shock-s93008-seat1 | 680 (d28 h8) | `["BUY_SEED", "WHEAT", 1]` | BUY_SEED WHEAT: no harvest can sell by step 718 |
| c50-cha22-s93006-seat0 | 681 (d28 h9) | `["BUY_SEED", "WHEAT", 1]` | BUY_SEED WHEAT: no harvest can sell by step 718 |
| c50-cha22-s93006-seat1 | 681 (d28 h9) | `["BUY_SEED", "WHEAT", 1]` | BUY_SEED WHEAT: no harvest can sell by step 718 |
| c50-cha22-s93008-seat0 | 678 (d28 h6) | `["BUY_SEED", "WHEAT", 1]` | BUY_SEED WHEAT: no harvest can sell by step 718 |
| c50-cha22-s93008-seat0 | 679 (d28 h7) | `["BUY_SEED", "WHEAT", 1]` | BUY_SEED WHEAT: no harvest can sell by step 718 |
| c50-cha22-s93008-seat1 | 678 (d28 h6) | `["BUY_SEED", "WHEAT", 1]` | BUY_SEED WHEAT: no harvest can sell by step 718 |
| c50-cha22-s93008-seat1 | 679 (d28 h7) | `["BUY_SEED", "WHEAT", 1]` | BUY_SEED WHEAT: no harvest can sell by step 718 |
| c50-v56-s93005-seat0 | 677 (d28 h5) | `["BUY_SEED", "WHEAT", 1]` | BUY_SEED WHEAT: no harvest can sell by step 718 |
| c50-v56-s93005-seat1 | 677 (d28 h5) | `["BUY_SEED", "WHEAT", 1]` | BUY_SEED WHEAT: no harvest can sell by step 718 |
| c50-v56-s93007-seat0 | 677 (d28 h5) | `["BUY_SEED", "WHEAT", 1]` | BUY_SEED WHEAT: no harvest can sell by step 718 |
| c50-v56-s93007-seat1 | 677 (d28 h5) | `["BUY_SEED", "WHEAT", 1]` | BUY_SEED WHEAT: no harvest can sell by step 718 |
| c50-v56-s93008-seat0 | 679 (d28 h7) | `["BUY_SEED", "WHEAT", 1]` | BUY_SEED WHEAT: no harvest can sell by step 718 |
| c50-v56-s93008-seat0 | 681 (d28 h9) | `["BUY_SEED", "WHEAT", 1]` | BUY_SEED WHEAT: no harvest can sell by step 718 |
| c50-v56-s93008-seat1 | 679 (d28 h7) | `["BUY_SEED", "WHEAT", 1]` | BUY_SEED WHEAT: no harvest can sell by step 718 |
| c50-v56-s93008-seat1 | 681 (d28 h9) | `["BUY_SEED", "WHEAT", 1]` | BUY_SEED WHEAT: no harvest can sell by step 718 |

## Per game

| game | blocked | first diff idx | gap OFF | gap A | gap B | B − A | bank A → B | opp A → B | unsold A → B |
|---|---|---|---|---|---|---|---|---|---|
| c50-smaller_market_shock-s93001-seat0 | 0 | None | -4,304 | -4,209 | -4,209 | +0 | 94,421 → 94,421 | 98,630 → 98,630 | 0/3/3 → 0/3/3 |
| c50-smaller_market_shock-s93001-seat1 | 0 | None | -4,304 | -4,209 | -4,209 | +0 | 94,421 → 94,421 | 98,630 → 98,630 | 0/3/3 → 0/3/3 |
| c50-smaller_market_shock-s93002-seat0 | 0 | None | -9,026 | -8,148 | -8,148 | +0 | 102,594 → 102,594 | 110,742 → 110,742 | 0/7/0 → 0/7/0 |
| c50-smaller_market_shock-s93002-seat1 | 0 | None | -9,026 | -8,148 | -8,148 | +0 | 102,594 → 102,594 | 110,742 → 110,742 | 0/7/0 → 0/7/0 |
| c50-smaller_market_shock-s93003-seat0 | 1 | 681 | -6,490 | -5,769 | -5,833 | -64 | 112,856 → 112,798 | 118,625 → 118,631 | 0/27/7 → 0/27/10 |
| c50-smaller_market_shock-s93003-seat1 | 1 | 681 | -7,284 | -6,203 | -6,195 | +8 | 112,296 → 112,304 | 118,499 → 118,499 | 0/13/26 → 0/13/26 |
| c50-smaller_market_shock-s93004-seat0 | 0 | None | -6,126 | -4,181 | -4,181 | +0 | 127,136 → 127,136 | 131,317 → 131,317 | 0/7/7 → 0/7/7 |
| c50-smaller_market_shock-s93004-seat1 | 0 | None | -6,126 | -4,181 | -4,181 | +0 | 127,136 → 127,136 | 131,317 → 131,317 | 0/7/7 → 0/7/7 |
| c50-smaller_market_shock-s93005-seat0 | 0 | None | +382 | +869 | +869 | +0 | 86,664 → 86,664 | 85,795 → 85,795 | 0/13/6 → 0/13/6 |
| c50-smaller_market_shock-s93005-seat1 | 0 | None | +382 | +869 | +869 | +0 | 86,664 → 86,664 | 85,795 → 85,795 | 0/13/6 → 0/13/6 |
| c50-smaller_market_shock-s93006-seat0 | 1 | 681 | -3,533 | -2,522 | -2,411 | +111 | 93,494 → 93,381 | 96,016 → 95,792 | 0/11/15 → 0/26/13 |
| c50-smaller_market_shock-s93006-seat1 | 1 | 681 | -2,818 | -2,421 | -2,514 | -93 | 93,371 → 93,503 | 95,792 → 96,017 | 0/26/13 → 0/12/15 |
| c50-smaller_market_shock-s93007-seat0 | 0 | None | -7,796 | -7,266 | -7,266 | +0 | 104,394 → 104,394 | 111,660 → 111,660 | 0/0/4 → 0/0/4 |
| c50-smaller_market_shock-s93007-seat1 | 0 | None | -8,013 | -7,265 | -7,265 | +0 | 104,397 → 104,397 | 111,662 → 111,662 | 0/0/4 → 0/0/4 |
| c50-smaller_market_shock-s93008-seat0 | 3 | 678 | +2,066 | +3,991 | +3,771 | -220 | 71,470 → 71,549 | 67,479 → 67,778 | 0/2/24 → 0/26/17 |
| c50-smaller_market_shock-s93008-seat1 | 3 | 678 | +2,066 | +3,991 | +3,771 | -220 | 71,470 → 71,549 | 67,479 → 67,778 | 0/2/24 → 0/26/17 |
| c50-cha22-s93001-seat0 | 0 | None | -14,130 | -13,993 | -13,993 | +0 | 92,577 → 92,577 | 106,570 → 106,570 | 0/3/18 → 0/3/18 |
| c50-cha22-s93001-seat1 | 0 | None | -14,130 | -13,993 | -13,993 | +0 | 92,577 → 92,577 | 106,570 → 106,570 | 0/3/18 → 0/3/18 |
| c50-cha22-s93002-seat0 | 0 | None | -21,381 | -20,481 | -20,481 | +0 | 95,478 → 95,478 | 115,959 → 115,959 | 0/8/0 → 0/8/0 |
| c50-cha22-s93002-seat1 | 0 | None | -21,381 | -20,481 | -20,481 | +0 | 95,478 → 95,478 | 115,959 → 115,959 | 0/8/0 → 0/8/0 |
| c50-cha22-s93003-seat0 | 0 | None | -9,442 | -9,038 | -9,038 | +0 | 108,326 → 108,326 | 117,364 → 117,364 | 0/3/0 → 0/3/0 |
| c50-cha22-s93003-seat1 | 0 | None | -9,550 | -9,039 | -9,039 | +0 | 108,325 → 108,325 | 117,364 → 117,364 | 0/3/0 → 0/3/0 |
| c50-cha22-s93004-seat0 | 0 | None | -16,062 | -14,959 | -14,959 | +0 | 120,395 → 120,395 | 135,354 → 135,354 | 0/34/37 → 0/34/37 |
| c50-cha22-s93004-seat1 | 0 | None | -16,062 | -14,959 | -14,959 | +0 | 120,395 → 120,395 | 135,354 → 135,354 | 0/34/37 → 0/34/37 |
| c50-cha22-s93005-seat0 | 0 | None | -2,936 | -2,649 | -2,649 | +0 | 87,765 → 87,765 | 90,414 → 90,414 | 0/5/10 → 0/5/10 |
| c50-cha22-s93005-seat1 | 0 | None | -2,936 | -2,649 | -2,649 | +0 | 87,765 → 87,765 | 90,414 → 90,414 | 0/5/10 → 0/5/10 |
| c50-cha22-s93006-seat0 | 1 | 682 | -12,181 | -11,299 | -11,209 | +90 | 94,967 → 95,064 | 106,266 → 106,273 | 0/16/7 → 0/0/16 |
| c50-cha22-s93006-seat1 | 1 | 682 | -12,264 | -11,299 | -11,209 | +90 | 94,968 → 95,064 | 106,267 → 106,273 | 0/14/9 → 0/0/16 |
| c50-cha22-s93007-seat0 | 0 | None | -3,677 | -2,901 | -2,901 | +0 | 103,746 → 103,746 | 106,647 → 106,647 | 0/3/2 → 0/3/2 |
| c50-cha22-s93007-seat1 | 0 | None | -3,677 | -2,901 | -2,901 | +0 | 103,746 → 103,746 | 106,647 → 106,647 | 0/3/2 → 0/3/2 |
| c50-cha22-s93008-seat0 | 2 | 679 | +4,260 | +5,105 | +5,036 | -69 | 71,036 → 71,010 | 65,931 → 65,974 | 0/16/26 → 0/28/17 |
| c50-cha22-s93008-seat1 | 2 | 679 | +4,260 | +5,105 | +5,036 | -69 | 71,036 → 71,010 | 65,931 → 65,974 | 0/16/26 → 0/28/17 |
| c50-v56-s93001-seat0 | 0 | None | -16,457 | -16,347 | -16,347 | +0 | 89,870 → 89,870 | 106,217 → 106,217 | 0/13/17 → 0/13/17 |
| c50-v56-s93001-seat1 | 0 | None | -16,457 | -16,347 | -16,347 | +0 | 89,870 → 89,870 | 106,217 → 106,217 | 0/13/17 → 0/13/17 |
| c50-v56-s93002-seat0 | 0 | None | -18,662 | -17,265 | -17,265 | +0 | 98,977 → 98,977 | 116,242 → 116,242 | 0/5/0 → 0/5/0 |
| c50-v56-s93002-seat1 | 0 | None | -18,662 | -17,265 | -17,265 | +0 | 98,977 → 98,977 | 116,242 → 116,242 | 0/5/0 → 0/5/0 |
| c50-v56-s93003-seat0 | 0 | None | -4,808 | -3,048 | -3,048 | +0 | 111,802 → 111,802 | 114,850 → 114,850 | 0/27/28 → 0/27/28 |
| c50-v56-s93003-seat1 | 0 | None | -5,781 | -3,413 | -3,413 | +0 | 111,442 → 111,442 | 114,855 → 114,855 | 0/37/25 → 0/37/25 |
| c50-v56-s93004-seat0 | 0 | None | -10,078 | -8,464 | -8,464 | +0 | 128,740 → 128,740 | 137,204 → 137,204 | 0/26/2 → 0/26/2 |
| c50-v56-s93004-seat1 | 0 | None | -10,078 | -8,464 | -8,464 | +0 | 128,740 → 128,740 | 137,204 → 137,204 | 0/26/2 → 0/26/2 |
| c50-v56-s93005-seat0 | 1 | 678 | -1,940 | -1,283 | -1,172 | +111 | 85,799 → 85,721 | 87,082 → 86,893 | 0/13/25 → 0/8/35 |
| c50-v56-s93005-seat1 | 1 | 678 | -1,940 | -1,283 | -1,172 | +111 | 85,799 → 85,721 | 87,082 → 86,893 | 0/13/25 → 0/8/35 |
| c50-v56-s93006-seat0 | 0 | None | -15,393 | -14,392 | -14,392 | +0 | 90,801 → 90,801 | 105,193 → 105,193 | 0/3/2 → 0/3/2 |
| c50-v56-s93006-seat1 | 0 | None | -15,336 | -14,526 | -14,526 | +0 | 90,638 → 90,638 | 105,164 → 105,164 | 0/6/4 → 0/6/4 |
| c50-v56-s93007-seat0 | 1 | 678 | -12,390 | -12,025 | -12,016 | +9 | 96,724 → 96,733 | 108,749 → 108,749 | 0/0/7 → 0/0/7 |
| c50-v56-s93007-seat1 | 1 | 678 | -13,374 | -12,380 | -11,976 | +404 | 96,371 → 96,588 | 108,751 → 108,564 | 0/14/8 → 0/14/6 |
| c50-v56-s93008-seat0 | 2 | 680 | -2,134 | -1,598 | -1,712 | -114 | 72,065 → 71,938 | 73,663 → 73,650 | 0/14/6 → 0/23/5 |
| c50-v56-s93008-seat1 | 2 | 680 | -2,134 | -1,598 | -1,712 | -114 | 72,065 → 71,938 | 73,663 → 73,650 | 0/14/6 → 0/23/5 |
