# Task 7.3 Claude review receipts

Review of Codex run 1 (`517edc4`). Planned expectation per fix, then actual log.

| Fix | Expectation | Red | Green / mutation |
| --- | --- | --- | --- |
| ACTIVE rewards as Kaggle integer `0` | export test fails on `0.0` | `rust-red.log` (active_rewards...) | `rust-green.log` |
| Semantic number kinds (`0` vs `0.0`) | Rust unit and Python framework oracle fail before fix | `rust-red.log`, `python-red.log` | `python-green.log`, `python-number-kind-green.log`; `mutation-number-kind.log` (kind check removed -> DID NOT RAISE) |
| Derived hire cap documented | maximal-hiring day decodes; cap - 1 fails | none (behaviour kept) | `rust-green.log`; `mutation-derived-cap.log` |
| Recorder malformed evidence | ValueError + error custody | `recorder-evidence-red.log` | `recorder-evidence-green.log` |

`rust-red.log` and `rust-red-attempt1.log` are the same first run. Its
`token_decode_uses_the_recorded_hire_limit` failure rested on a wrong premise
(hands persist across days; engine `end_of_day` clears them), so that change was
reverted and replaced by the derived-cap test. `prepare.log`: full
`just prepare`, exit 0 (1,741 Python passed / 16 skipped; 269 root Rust passed /
4 ignored; 69 engine passed).
