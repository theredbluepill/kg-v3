# Task 1.4 kernel cast and saturation audit

This is source analysis, not a measured test receipt. Task D's command receipts
record the actual red/green and release outcomes separately. No vendored bytes
are changed by this audit or the root admission helper.

## Scope and source identity

The production path starts at `ObservationGame::from_seed` and reaches
`Game::step_with_market_metrics`, its private transition helpers,
`econ_attrib.rs`, and the integer-seeded `py_random.rs` path. There is no
production lifecycle header importer. Configuration is admitted by
`ObservationConfig::new`; every published current observation has passed
`prepare`. The action grammar admits at most 241 actors, ten market orders,
and quantity 1023. The deployment targets have 64-bit `usize`.

Audited SHA-256 values, before Task D implementation:

| Path | SHA-256 |
|---|---|
| `engine_rs/src/lib.rs` | `c4b9bac5057be3a435d2f1035aae17bcd15e7f95ea8557322e4929877c8231fd` |
| `engine_rs/src/econ_attrib.rs` | `861b7bbaf61ac065bb18797bc389c2acbf6e15c37decaeeeff7eeaeac71e05ac` |
| `engine_rs/src/py_random.rs` | `40ff0c5e5a79a8ad9d580af4957e028f95ff180603259d8b81be2b357cd73356` |
| `src/kaggriculture/config.rs` | `82edc952a0446f5a5b2379ba25b1071150c3e992e0fe7405bae0c98170b0ac0f` |
| `src/kaggriculture/observe.rs` | `0d84eb9df01f699767fc674996e5eae14be31c09f70b058b5af35fce36365749` |
| `src/kaggriculture/grammar.rs` | `5b660b560731b15b746795fec3875eed911943038b3cea3d45ca912032a0426d` |

Enumeration command (read-only; regular expression includes float casts and
explicit saturation/wrapping beyond the brief's minimum integer pattern):

```sh
rg -n '\bas (?:i[0-9]+|u[0-9]+|usize|isize|f[0-9]+)\b|saturating_|wrapping_' engine_rs/src/lib.rs engine_rs/src/econ_attrib.rs engine_rs/src/py_random.rs
```

The three files contain 99 numeric `as` expressions: 78 in `lib.rs` including
eight in its test module, ten in `econ_attrib.rs`, and eleven in `py_random.rs`.
There are seven `saturating_sub` expressions in `lib.rs`, including the unused
wealth diagnostic, and seven explicit MT19937 wrapping operations. Every
enumerated source line appears below; repeated casts on one line are stated.
An unbounded intermediate addition is not made safe by its later cast: those
additions require the separate dependency overflow-check policy and rollback.

## Cast classification

`B` means bounded by the admitted configuration, grammar, prepared seeded state,
or an explicit checked conversion. `U` means an executable conversion lacks a
u64 range bound without the additional admission described below. `X` means
excluded from this production path. `M` means intentional modular arithmetic.

### engine_rs/src/lib.rs

| Line(s) | Expression family | Class and reason |
|---|---|---|
| 534 | i128 numeric difference to f64 | B: difference of i64/u64-sized values is finite; no integer narrowing |
| 607 | market delta to f64 | X: float-market branch; raw default markets remain integer-valued |
| 658, 660 | usize bound to f64; integral f64 to usize | X: configuration deserializer, not step; root then requires boardSize exactly 10 |
| 1477 | day i64 to usize | B: nonnegative quotient; checked usize-to-i64 immediately before at 1455 |
| 1496 | nonnegative seed inventory i64 to u64 | B: exact nonnegative conversion; summation separately overflow-checked |
| 1499, 1501 | animal flags bool to u64 | B: 0 or 1 |
| 1500 | nonnegative pending bonus i64 to u64 | B: exact nonnegative conversion |
| 1505 | land prices i64 to u64 | B: constants 1000, 2000, 4000 |
| 1537 | idle hand count usize to u64 | B: at most 240 admitted hands |
| 1633 | y/x i64 to usize (two casts) | B: nonnegative guards; seeded positions lie in 0..9 |
| 1648, 1649 | y/x and returned x/y i64 to usize (four casts) | B: negative guard; seeded positions lie in 0..9 |
| 1673, 1676, 1677 | clipped/missed/dug units i64 to u64 | B: max(0) before exact widening; preceding arithmetic separately checked |
| 1703 | malformed flag bool to u64 | B: 0 or 1 |
| 1945 | sale/buy counters to f64 (two casts) | X: vendored test module |
| 1963 | test position x/y to usize (two casts) | X: vendored test module |
| 1977, 2225 | test cash counter to f64 | X: vendored test module |
| 2134 | test cash delta to u64 | X: vendored test module |
| 2189 | test weeds counter to usize | X: vendored test module |
| 2827, 2831, 2836, 2845, 2848, 2851, 2852, 2857 | wealth integer operands to f64 (nine casts; two at 2836) | X: `seat_wealth` is not called by native lifecycle stepping |
| 2896 | board half usize to i64 | B: boardSize = 10 |
| 2908 | default spawn x/y i64 to usize (two casts) | B: shed access coordinates 4 or 5 |
| 3059 | ready units i64 to u64 | B: guarded positive value |
| 3144, 3153 | boardSize to i64 (four casts) | B: boardSize = 10 |
| 3156 | position x/y i64 to usize (two casts) | B: explicit nonnegative and board-bound checks |
| 3756 | PRICE_FLOOR to f64 | B: constant 1 |
| 3844, 3847 | land cost to f64 | B: constants 1000, 2000, 4000 |
| 3936 | ignored market entries usize to u64 | B: canonical grammar submits no more than the configured limit |
| 3966 | requested count i64 to u64 | B: max(0), grammar quantity at most 1023 |
| 4018 | wasted HIRE float cash delta to u64 | U: multiplier is i64 but Fibonacci-scaled hire cost and bank need not fit u64 |
| 4107 | base price to f64; sale shortfall f64 to u64 (two casts) | B: default price floor 1 and base at most 250 bound shortfall |
| 4108 | floor-sale flag bool to u64 | B: 0 or 1 |
| 4112 | base price to f64; premium f64 to u64 (two casts) | B: BUY_SEED fixed prices; BUY_PRODUCT restricted to bounded WHEAT/FERTILIZER quotes; proof below |
| 4122 | rounded SELL price f64 to u64 fallback | B: quoted integer fits u64, so `as_u64` succeeds; proof below |
| 4185 | lost crop units i64 to u64 | B: max(0) before conversion |
| 4298 | care/clipping losses i64 to u64 (two casts) | B: max(0) before conversions; additions separately checked |
| 4325, 4334, 4336, 4341 | death/clipped/missed units i64 to u64 | B: max(0) and min/max operands; arithmetic separately checked |
| 4346 | wasted-care bool to u64 | B: 0 or 1 |
| 4348, 4349 | held units/pending bonus i64 to u64 | B: max(0) before conversion |
| 4350 | fertilizer flag bool to u64 | B: 0 or 1 |
| 4389 | requested quantity i64 to u64 | B: positive filter and grammar bound 1023 |
| 4410 | nonnegative shed + carried sum i64 to u64 | B: exact once prior i64 sums/addition succeed under overflow checks |
| 4420 | terminal tile units i64 to u64 | B: max(0) before conversion |
| 4461, 4462, 4468, 4472 | end-of-day usize to i64 | B: same checked day passed through 1477; no intervening increment |

### engine_rs/src/econ_attrib.rs

| Line(s) | Expression family | Class and reason |
|---|---|---|
| 69 | rounded quoted price f64 to u64 fallback | B: same default-quote proof as lib.rs:4122 |
| 104, 208, 214, 215, 217 | flags bool to u64 | B: 0 or 1 |
| 123, 206 | tile yield i64 to u64 | B: max(0) before conversion |
| 137 | requested transfer i64 to u64 | B: positive filter and grammar bound 1023 |
| 179 | investment cash delta f64 to u64 | U for HIRE; B for BUY_LAND's fixed prices |

### engine_rs/src/py_random.rs

| Line(s) | Expression family | Class and reason |
|---|---|---|
| 148 | i128 seed limb to u32 | X/M: unused convenience integer-seed entry; intentional low-limb extraction |
| 202 | 53-bit random integer to f64 | B: exactly representable 53-bit numerator |
| 235 | requested bit count u32 to usize | X: unused getrandbits_u64 convenience entry; it bounds width to 64 |
| 245 | usize bit width u32 to usize | B: at most 64 |
| 251 | randbelow bound usize to u64 | B: 64-bit host; daily shop choice has eight entries |
| 252 | candidate u64 to usize | B: rejection sampler proves candidate < bound |
| 292, 320 | MT state indices usize to u32 | B: at most 623 |
| 305 | seed key index usize to u32 | B: i64 seed times 1000003 XOR admitted day needs at most three 32-bit limbs |
| 368 | decimal multiply-add low word u64 to u32 | M: deliberate radix-2^32 limb extraction, with high bits carried separately |
| 372 | decimal multiply-add carry u64 to u32 | B: decimal multiplier 10 bounds carry to 9 |

## Saturation and related conversions

| Source | Classification |
|---|---|
| lib.rs:1688,4499 `before.saturating_sub(after)` | B: intended nonnegative discarded-goods measurement; underlying i64 aggregate sums checked separately |
| lib.rs:2856 quadrant-count `saturating_sub(1)` | X: unused wealth diagnostic |
| lib.rs:3076 submitted minus committed | B: intended nonnegative count, at most 241 commands |
| lib.rs:3839 extra quadrants | B: zero to three extra quadrants |
| lib.rs:3936 ignored market entries | B: grammar restricts input to configured orders |
| lib.rs:4136 requested minus committed | B: intended unfilled count; grammar quantity at most 1023 |
| py_random.rs:291,292,303,304,305,319,320 wrapping arithmetic | M: specified MT19937 u32 recurrence, intentionally unchanged by overflow policy |
| lib.rs:353-361 `python_int_i64` fallback to MIN/MAX | B: grammar admits numeric quantities only in 0..1023, so fallback is unreachable |
| lib.rs:380-382 `PyInt::capped_usize` fallback | B: admitted market order count is 1..10 |
| lib.rs:400-407 `room_from_i64_total` fallback to MAX | B: capacity is positive i64; successful checked sums of nonnegative seeded goods leave room in 0..i64::MAX |
| lib.rs:388-399 `div_rem_usize`/`divides_usize` fallback | B: positive admitted i64 intervals fit 64-bit usize |
| lib.rs:1419 `seat_wealth` checked day fallback to i64::MAX | X: unused wealth diagnostic |
| lib.rs:3751-3762 rounded market price conversion | B: `ToPrimitive::to_i64` is checked; larger finite values become BigInt, never a saturating cast; publication still requires i64 market values |
| lib.rs:3790-3835 Fibonacci and hire cost conversion | U's source: exact BigInt cost, finite f64 conversion; successful HIRE cost now receives the root admission below |

The `python_int`, `schema_integer`, `numeric`, and market arithmetic helpers use
checked `ToPrimitive`/`FromPrimitive`, finite tests, or arbitrary-precision
integers. Serde `as_i64`/`as_u64` are checked accessors. None silently narrows an
otherwise admitted grammar value. Ordinary game additions/sums remain the
separate checked-arithmetic obligation.

## Default-quote proof

The prepared current market inventory and prices fit i64 (`observe.rs:960-1014`),
and custom parameters are rejected (`config.rs:70-77`). Default tables are at
`lib.rs:161-201`. Between market quotes, products other than WHEAT/FERTILIZER
can only be sold; default prices are nonincreasing as inventory increases.
Their quoted prices therefore do not exceed the admitted current i64 price.
WHEAT/FERTILIZER purchases can remove at most `2 * 10 * 1023 = 20460` units per
step. Even starting at i64::MIN inventory, WHEAT's default sqrt quote stays
below 3.1 billion and FERTILIZER's linear quote below 1.85e18. Those are below
i64::MAX and u64::MAX. BUY_SEED/BUY_ANIMAL use fixed small positive prices.
Town consumption runs after trading; its refreshed prices undergo observation
admission before publication and do not feed a cash cast during that step.

This proof would not hold for custom markets, a wider quantity/order grammar,
unvalidated imported state, or a new BUY_PRODUCT item. Those changes reopen it.

## Required admission: two HIRE cash sites, no configuration cap

There are **two reachable-unbounded cast sites**, lib.rs:4018 and
econ_attrib.rs:179. The former is visible to the required economic u64-to-i64
output check only for wasted hires. A successful mid-day hire also updates the
private attribution ledger and otherwise could saturate silently.

`src/kaggriculture/admission.rs::validate_hire_cash(multiplier, hires_before,
executed)` checks actual successful hires in an unpublished candidate. The
caller obtains `executed[seat]` by checked subtraction of public
`Game::attrib_counters()` slot 52 (`A_HIRES`, econ_attrib.rs:29,179-184). The
ledger persists when end-of-day resets `hires_today` and hands. Failed HIREs
never increment `hires_today`; actual successful costs are exactly
`multiplier * fib(hires_before[seat] + j)`, with kernel fib(0)=fib(1)=1.
The helper computes BigInt costs and rejects their f64 conversion at or above
2^64. In particular `2 * i64::MAX` is less than u64::MAX but rounds to 2^64,
so checking only the exact integer bound would be wrong.

This runs immediately after the real candidate step, before autoreset and
publication. It adds no configuration bound and does not reject merely
submitted, unaffordable hires. Zero executed hires and a zero multiplier need
no cost calculation. No ambiguous comparison with a saturated final ledger
value is used. Existing checked econ conversion, finite banks, dependency
overflow catch, and batch rollback remain required.

### Payment-rounding proof and limits

Seeded initial money is at most i64::MAX, whose f64 conversion is at most 2^63.
For a candidate that passes economic output admission, cumulative exact sale
cash is at most i64::MAX. For positive integer sale price p, conversion to
binary64 has relative error at most 2^-52, and adding a positive binary64
amount to a nonnegative representable bank changes the bank by at most twice
that amount. Thus each sale changes the bank by less than 2.5p. Purchases and
investments cannot increase it. Every intermediate bank is consequently less
than `3.5 * 2^63` in an admissible seeded candidate.

For a successful HIRE with rounded cost c and bank b, if c >= b/2 then b-c is
exact by Sterbenz; computing b-(b-c) rounds back to the representable c. If
c < b/2, then c < `1.75 * 2^63`; the subtraction's rounding error, bounded by
one bank ULP (less than 8192 here), cannot approach the remaining gap to 2^64.
Therefore c < 2^64 cannot saturate either payment cast. An executed c >= 2^64
lies in the exact-subtraction case and is rejected before publication. Integral
payments at this scale make the attribution `.round()` immaterial.

This is a seeded-state proof conditioned on the required final economic range
check. It is not a claim of unrestricted huge-configuration support or imported
header safety. A synthetic header can exercise a rejection without claiming
that a short bounded seeded trajectory reaches it. Numerical tests establish
the helper's boundary decisions; lifecycle tests must separately establish its
wiring and rollback. None of those checks has been run by this audit writer.

## Release discriminator

The proposed release test imports a test-only two-step-horizon header with seat
0 shed WHEAT=i64::MAX and FERTILIZER=1, no carried goods, then submits PASS for
both seats. Wide observation totals admit those individual exact fields. The
terminal path calls `held_goods`, whose i64 shed sum at lib.rs:4408 wraps to MIN
without dependency overflow checks, then max(0) erases the goods. With the root
package release override it panics and must become a worker error, preserving
all published bytes. Assert the panic-specific path, not any generic error.
The header is a test probe, not a production import API or seeded reachability
claim. Actual release status belongs in the Task D receipt; no result is
predicted here.
