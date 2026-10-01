## Health (played arms)

```json
{
 "60M-off": {
  "games": 48,
  "qualified": 48,
  "calls_719": 48,
  "exceptions": 0,
  "invalid_raw_actions": 0,
  "default_pass_fallbacks": 0,
  "bad_statuses": 0,
  "manifest_sha256": [
   "2d48dd16885e16ce261a72b044cb9109442f39f2cb2e60a2af83ed185f44d58b"
  ],
  "kaggriculture_py_sha256": [
   "f73d27ce04a6fd5ddcdd59c618e1da8c9b6fb2f27f5cf9cdc27efad942345004"
  ],
  "mean_wall_s": 93.2
 },
 "60M-ft": {
  "games": 48,
  "qualified": 48,
  "calls_719": 48,
  "exceptions": 0,
  "invalid_raw_actions": 0,
  "default_pass_fallbacks": 0,
  "bad_statuses": 0,
  "manifest_sha256": [
   "2d48dd16885e16ce261a72b044cb9109442f39f2cb2e60a2af83ed185f44d58b"
  ],
  "kaggriculture_py_sha256": [
   "f73d27ce04a6fd5ddcdd59c618e1da8c9b6fb2f27f5cf9cdc27efad942345004"
  ],
  "mean_wall_s": 90.3
 },
 "70M-ft": {
  "games": 48,
  "qualified": 48,
  "calls_719": 48,
  "exceptions": 0,
  "invalid_raw_actions": 0,
  "default_pass_fallbacks": 0,
  "bad_statuses": 0,
  "manifest_sha256": [
   "761ff250ba287f1c8815c32b483e56965b172381c3815afea9c555579c25e8df"
  ],
  "kaggriculture_py_sha256": [
   "f73d27ce04a6fd5ddcdd59c618e1da8c9b6fb2f27f5cf9cdc27efad942345004"
  ],
  "mean_wall_s": 90.6,
  "manifest_is_pkg_70M": true
 }
}
```

## 70M-off derivation checks

```json
{
 "games": 48,
 "prerule_record_obs718": 48,
 "rule_action_equals_on_replay": 48,
 "rule_changed_action": 48,
 "prefix_actions_equal_on_0_717": 48,
 "opp_final_action_equal_on": 48,
 "own_final_action_is_policy": 48,
 "derived_bad_statuses": 0,
 "cf_market_model_matches_engine": 48,
 "opp_bank_unchanged_vs_on": 37,
 "validation_on_60M_derived_equals_played_off": "48/48",
 "spot_check_played_off": {
  "games": 3,
  "qualified": 3,
  "banks_equal": 3,
  "all_actions_equal": 3,
  "all_obs_equal_ex_overage": 3
 }
}
```

## Per arm and anchor

| arm | anchor | n | W-L | own bank | anchor bank | margin | unsold shed/carried/tile |
|---|---|---|---|---|---|---|---|
| 60M-off | smaller_market_shock | 16 | 2-14 | 95,732 | 97,745 | -2,013 | 16.5 / 14.9 / 8.4 |
| 60M-off | cha22 | 16 | 0-16 | 96,188 | 101,392 | -5,204 | 21.3 / 12.9 / 11.2 |
| 60M-off | v56 | 16 | 2-14 | 95,123 | 100,431 | -5,308 | 20.9 / 14.5 / 13.9 |
| 60M-off | all | 48 | 4-44 | 95,681 | 99,856 | -4,175 | 19.6 / 14.1 / 11.2 |
| 70M-off (derived) | smaller_market_shock | 16 | 6-10 | 97,035 | 98,849 | -1,813 | 17.6 / 15.1 / 7.8 |
| 70M-off (derived) | cha22 | 16 | 4-12 | 97,267 | 99,576 | -2,308 | 17.4 / 16.4 / 10.1 |
| 70M-off (derived) | v56 | 16 | 6-10 | 96,914 | 99,068 | -2,154 | 17.0 / 14.8 / 9.8 |
| 70M-off (derived) | all | 48 | 16-32 | 97,072 | 99,164 | -2,092 | 17.3 / 15.4 / 9.2 |
| 60M-ft | smaller_market_shock | 16 | 4-12 | 96,526 | 97,741 | -1,216 | 0.0 / 14.9 / 8.4 |
| 60M-ft | cha22 | 16 | 2-14 | 96,898 | 101,391 | -4,493 | 0.0 / 12.9 / 11.2 |
| 60M-ft | v56 | 16 | 2-14 | 96,226 | 100,431 | -4,204 | 0.0 / 14.5 / 13.9 |
| 60M-ft | all | 48 | 8-40 | 96,550 | 99,854 | -3,304 | 0.0 / 14.1 / 11.2 |
| 70M-ft | smaller_market_shock | 16 | 6-10 | 97,680 | 98,849 | -1,169 | 0.0 / 15.1 / 7.8 |
| 70M-ft | cha22 | 16 | 4-12 | 97,786 | 99,576 | -1,790 | 0.0 / 16.4 / 10.1 |
| 70M-ft | v56 | 16 | 6-10 | 97,366 | 99,068 | -1,702 | 0.0 / 14.8 / 9.8 |
| 70M-ft | all | 48 | 16-32 | 97,611 | 99,164 | -1,554 | 0.0 / 15.4 / 9.2 |

## Paired: 70M-ft vs 60M-ft (rule 1 on, both played)

| anchor | W-L 70M-ft | W-L 60M-ft | flips L->W / W->L | Δ margin mean ± SE (8 seeds) | better on k/8 seeds | Δ own bank ± SE | Δ anchor bank ± SE | min / max Δ margin (game) |
|---|---|---|---|---|---|---|---|---|
| smaller_market_shock | 6-10 | 4-12 | 4 / 2 | **+47 ± 1,335** | 5/8 | +1,154 ± 1,037 | +1,108 ± 1,799 | -6,967 / +6,777 |
| cha22 | 4-12 | 2-14 | 2 / 0 | **+2,703 ± 1,955** | 7/8 | +888 ± 1,069 | -1,815 ± 1,784 | -8,802 / +9,518 |
| v56 | 6-10 | 2-14 | 4 / 0 | **+2,502 ± 1,324** | 5/8 | +1,140 ± 1,750 | -1,363 ± 1,650 | -889 / +10,352 |
| all | 16-32 | 8-40 | 10 / 2 | **+1,751 ± 985** | 7/8 | +1,060 ± 905 | -690 ± 1,540 | -8,802 / +10,352 |

## Paired: 70M-off (derived) vs 60M-off (played)

| anchor | W-L 70M-off | W-L 60M-off | flips L->W / W->L | Δ margin mean ± SE (8 seeds) | better on k/8 seeds | Δ own bank ± SE | Δ anchor bank ± SE | min / max Δ margin (game) |
|---|---|---|---|---|---|---|---|---|
| smaller_market_shock | 6-10 | 2-14 | 4 / 0 | **+200 ± 1,317** | 5/8 | +1,304 ± 1,188 | +1,104 ± 1,801 | -7,022 / +5,566 |
| cha22 | 4-12 | 0-16 | 4 / 0 | **+2,895 ± 1,875** | 7/8 | +1,079 ± 1,159 | -1,816 ± 1,785 | -8,089 / +8,930 |
| v56 | 6-10 | 2-14 | 4 / 0 | **+3,154 ± 1,191** | 7/8 | +1,791 ± 1,686 | -1,363 ± 1,650 | -1,038 / +10,322 |
| all | 16-32 | 4-44 | 12 / 0 | **+2,083 ± 911** | 7/8 | +1,391 ± 981 | -692 ± 1,541 | -8,089 / +10,322 |

## Paired: 70M-ft vs 70M-off (rule-1 effect on 70M)

| anchor | W-L 70M-ft | W-L 70M-off | flips L->W / W->L | Δ margin mean ± SE (8 seeds) | better on k/8 seeds | Δ own bank ± SE | Δ anchor bank ± SE | min / max Δ margin (game) |
|---|---|---|---|---|---|---|---|---|
| smaller_market_shock | 6-10 | 6-10 | 0 / 0 | **+645 ± 164** | 8/8 | +645 ± 164 | +0 ± 0 | +22 / +1,397 |
| cha22 | 4-12 | 4-12 | 0 / 0 | **+519 ± 126** | 8/8 | +519 ± 126 | +0 ± 0 | +95 / +1,206 |
| v56 | 6-10 | 6-10 | 0 / 0 | **+451 ± 126** | 8/8 | +451 ± 126 | +0 ± 0 | +146 / +1,214 |
| all | 16-32 | 16-32 | 0 / 0 | **+538 ± 96** | 8/8 | +538 ± 96 | +0 ± 0 | +22 / +1,397 |

## Running table: mean margin (W-L) per anchor, 16 games each, fixed-shop engine

Rule-1-off arms first (all played except 70M, derived), then rule-1-on arms.

| checkpoint | smaller_market_shock | cha22 | v56 | all 48 |
|---|---|---|---|---|
| BC | -62,879 (0-16) | -58,720 (0-16) | -60,196 (0-16) | -60,598 (0-48) |
| fc6b | -42,090 (0-16) | -38,185 (0-16) | -43,094 (0-16) | -41,123 (0-48) |
| f610 | -21,870 (0-16) | -23,366 (0-16) | -18,676 (0-16) | -21,304 (0-48) |
| 60f2 | -14,277 (0-16) | -14,075 (0-16) | -13,003 (0-16) | -13,785 (0-48) |
| 08bc | -11,577 (0-16) | -11,162 (0-16) | -13,091 (0-16) | -11,943 (0-48) |
| c50 | -4,372 (4-12) | -9,456 (2-14) | -10,352 (0-16) | -8,060 (6-42) |
| 60M | -2,013 (2-14) | -5,204 (0-16) | -5,308 (2-14) | -4,175 (4-44) |
| 70M (derived off) | -1,813 (6-10) | -2,308 (4-12) | -2,154 (6-10) | -2,092 (16-32) |
| c50 ft | -3,425 (4-12) | -8,777 (2-14) | -9,356 (0-16) | -7,186 (6-42) |
| 60M ft | -1,216 (4-12) | -4,493 (2-14) | -4,204 (2-14) | -3,304 (8-40) |
| 70M ft | -1,169 (6-10) | -1,790 (4-12) | -1,702 (6-10) | -1,554 (16-32) |

Running-table health: {"BC": {"qualified": 48, "kaggriculture_py": ["f73d27ce"]}, "fc6b": {"qualified": 48, "kaggriculture_py": ["f73d27ce"]}, "f610": {"qualified": 48, "kaggriculture_py": ["f73d27ce"]}, "60f2": {"qualified": 48, "kaggriculture_py": ["f73d27ce"]}, "08bc": {"qualified": 48, "kaggriculture_py": ["f73d27ce"]}, "c50": {"qualified": 48, "kaggriculture_py": ["f73d27ce"]}, "60M": {"qualified": 48, "kaggriculture_py": ["f73d27ce"]}, "c50 ft": {"qualified": 48, "kaggriculture_py": ["f73d27ce"]}, "60M ft": {"qualified": 48, "kaggriculture_py": ["f73d27ce"]}}

## Per game margins

| game | 60M-off | 70M-off (derived) | 60M-ft | 70M-ft | 70M unsold shed/carried/tile off -> ft |
|---|---|---|---|---|---|
| smaller_market_shock s93001 seat0 | -292 | -2,635 | +76 | -1,773 | 21/9/2 -> 0/9/2 |
| smaller_market_shock s93001 seat1 | -292 | -2,635 | +76 | -1,773 | 21/9/2 -> 0/9/2 |
| smaller_market_shock s93002 seat0 | -4,498 | -1,327 | -2,476 | -719 | 15/25/8 -> 0/25/8 |
| smaller_market_shock s93002 seat1 | -4,498 | -1,327 | -2,476 | -719 | 15/25/8 -> 0/25/8 |
| smaller_market_shock s93003 seat0 | -2,628 | +2,938 | -2,442 | +4,335 | 39/31/16 -> 0/31/16 |
| smaller_market_shock s93003 seat1 | -2,536 | +2,152 | -2,432 | +3,431 | 29/51/31 -> 0/51/31 |
| smaller_market_shock s93004 seat0 | -5,956 | -4,860 | -3,951 | -3,613 | 16/0/8 -> 0/0/8 |
| smaller_market_shock s93004 seat1 | -5,956 | -4,860 | -3,951 | -3,613 | 16/0/8 -> 0/0/8 |
| smaller_market_shock s93005 seat0 | -884 | +716 | -555 | +879 | 10/5/0 -> 0/5/0 |
| smaller_market_shock s93005 seat1 | -884 | +716 | -555 | +879 | 10/5/0 -> 0/5/0 |
| smaller_market_shock s93006 seat0 | -2,100 | -3,840 | -1,988 | -3,502 | 20/0/4 -> 0/0/4 |
| smaller_market_shock s93006 seat1 | -2,332 | -3,759 | -2,144 | -3,505 | 16/2/3 -> 0/2/3 |
| smaller_market_shock s93007 seat0 | -4,110 | -11,132 | -3,856 | -10,813 | 9/5/3 -> 0/5/3 |
| smaller_market_shock s93007 seat1 | -4,110 | -10,845 | -3,856 | -10,823 | 6/5/3 -> 0/5/3 |
| smaller_market_shock s93008 seat0 | +4,435 | +5,842 | +5,541 | +6,314 | 19/35/14 -> 0/35/14 |
| smaller_market_shock s93008 seat1 | +4,435 | +5,842 | +5,541 | +6,314 | 19/35/14 -> 0/35/14 |
| cha22 s93001 seat0 | -8,236 | -6,634 | -8,102 | -5,428 | 32/14/7 -> 0/14/7 |
| cha22 s93001 seat1 | -8,236 | -6,634 | -8,102 | -5,428 | 32/14/7 -> 0/14/7 |
| cha22 s93002 seat0 | -7,151 | -1,276 | -5,782 | -1,138 | 11/0/5 -> 0/0/5 |
| cha22 s93002 seat1 | -7,151 | -1,276 | -5,782 | -1,138 | 11/0/5 -> 0/0/5 |
| cha22 s93003 seat0 | -9,913 | -1,261 | -9,898 | -380 | 15/36/31 -> 0/36/31 |
| cha22 s93003 seat1 | -9,999 | -1,069 | -9,937 | -431 | 15/37/29 -> 0/37/29 |
| cha22 s93004 seat0 | -5,848 | -4,476 | -4,909 | -3,858 | 12/16/5 -> 0/16/5 |
| cha22 s93004 seat1 | -5,848 | -4,476 | -4,909 | -3,858 | 12/16/5 -> 0/16/5 |
| cha22 s93005 seat0 | -3,761 | +3,947 | -3,258 | +4,294 | 17/3/5 -> 0/3/5 |
| cha22 s93005 seat1 | -3,761 | +3,947 | -3,258 | +4,294 | 17/3/5 -> 0/3/5 |
| cha22 s93006 seat0 | -2,607 | -1,399 | -2,084 | -1,304 | 10/27/9 -> 0/27/9 |
| cha22 s93006 seat1 | -2,558 | -1,745 | -1,996 | -1,262 | 18/35/4 -> 0/35/4 |
| cha22 s93007 seat0 | -3,787 | -11,876 | -2,890 | -11,692 | 10/4/2 -> 0/4/2 |
| cha22 s93007 seat1 | -3,943 | -11,876 | -2,951 | -11,692 | 10/4/2 -> 0/4/2 |
| cha22 s93008 seat0 | -229 | +4,585 | +987 | +5,192 | 28/27/20 -> 0/27/20 |
| cha22 s93008 seat1 | -229 | +4,585 | +987 | +5,192 | 28/27/20 -> 0/27/20 |
| v56 s93001 seat0 | -9,022 | +1,300 | -8,896 | +1,456 | 16/4/3 -> 0/4/3 |
| v56 s93001 seat1 | -9,022 | +1,300 | -8,896 | +1,456 | 16/4/3 -> 0/4/3 |
| v56 s93002 seat0 | -6,364 | -3,420 | -4,282 | -2,206 | 18/3/11 -> 0/3/11 |
| v56 s93002 seat1 | -6,364 | -3,420 | -4,282 | -2,206 | 18/3/11 -> 0/3/11 |
| v56 s93003 seat0 | -3,322 | -3,175 | -2,077 | -2,707 | 18/12/17 -> 0/12/17 |
| v56 s93003 seat1 | -2,143 | -3,181 | -2,088 | -2,705 | 20/12/17 -> 0/12/17 |
| v56 s93004 seat0 | -6,635 | -6,035 | -4,455 | -5,344 | 15/13/6 -> 0/13/6 |
| v56 s93004 seat1 | -6,635 | -6,035 | -4,455 | -5,344 | 15/13/6 -> 0/13/6 |
| v56 s93005 seat0 | -802 | +1,024 | -598 | +1,401 | 20/32/11 -> 0/32/11 |
| v56 s93005 seat1 | -802 | +1,024 | -598 | +1,401 | 20/32/11 -> 0/32/11 |
| v56 s93006 seat0 | -7,474 | -2,945 | -6,380 | -2,682 | 21/4/8 -> 0/4/8 |
| v56 s93006 seat1 | -7,304 | -3,092 | -6,380 | -2,660 | 23/4/8 -> 0/4/8 |
| v56 s93007 seat0 | -11,102 | -6,610 | -10,617 | -6,464 | 10/3/0 -> 0/3/0 |
| v56 s93007 seat1 | -10,895 | -6,610 | -10,620 | -6,464 | 10/3/0 -> 0/3/0 |
| v56 s93008 seat0 | +1,483 | +2,709 | +3,677 | +2,916 | 16/47/22 -> 0/47/22 |
| v56 s93008 seat1 | +1,483 | +2,709 | +3,677 | +2,916 | 16/47/22 -> 0/47/22 |
