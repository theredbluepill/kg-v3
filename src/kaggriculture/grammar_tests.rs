use super::*;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::process::Command;

use serde_json::Value;

#[test]
fn grammar_constants() {
    assert_eq!(MAX_ACTORS, 241);
    assert_eq!(MAX_ORDERS, 10);
    assert_eq!(MAX_FRAMES, 252);
    assert_eq!(TOKENS_PER_SEAT, 3024);
    assert_eq!(DEFAULT_HIRE_LIMIT, 241);
    assert_eq!(QUANTITY_BASE, 32);
    assert_eq!(MAX_QUANTITY, 1023);
    assert_eq!(GRAMMAR_TABLES_VERSION, 1);
    let names = [
        "unit_actor",
        "unit_kind",
        "unit_target",
        "unit_item",
        "unit_quantity_present",
        "unit_quantity_high",
        "unit_quantity",
        "market_kind",
        "market_item",
        "market_quantity_high",
        "market_quantity",
        "stop",
    ];
    let widths = [241, 20, 128, 16, 2, 32, 32, 8, 16, 32, 32, 2];
    assert_eq!(SLOT_NAMES, names);
    assert_eq!(SLOT_WIDTHS, widths);
    for i in 0..SLOTS {
        let slot = Slot::try_from(i).unwrap();
        assert_eq!(slot as usize, i);
        assert_eq!(slot.name(), names[i]);
        assert_eq!(slot.width(), widths[i]);
    }
    assert!(Slot::try_from(12).is_err());
    assert!(Slot::try_from(usize::MAX).is_err());
    for i in 0..19 {
        assert_eq!(UnitKind::try_from(i).unwrap() as i64, i);
    }
    for i in 0..8 {
        assert_eq!(MarketKind::try_from(i).unwrap() as i64, i);
    }
    for i in 0..13 {
        assert_eq!(ActionItem::try_from(i).unwrap() as i64, i);
    }
    for i in [-1, 19, 20, i64::MAX] {
        assert!(UnitKind::try_from(i).is_err());
    }
    for i in [-1, 8, i64::MAX] {
        assert!(MarketKind::try_from(i).is_err());
    }
    for i in [-1, 13, 14, 15, i64::MAX] {
        assert!(ActionItem::try_from(i).is_err());
    }
    assert_eq!(UnitKind::CollectFertilizer.as_str(), "COLLECT_FERTILIZER");
    assert_eq!(MarketKind::BuyProduct.as_str(), "BUY_PRODUCT");
    assert_eq!(ActionItem::Fertilizer.as_str(), "FERTILIZER");
}

fn grammar_fixture_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    match env!("CARGO_PKG_NAME") {
        "owl" => manifest,
        "kaggriculture-engine" => manifest.parent().unwrap().to_path_buf(),
        package => panic!("unsupported grammar fixture package {package}"),
    }
}

fn grammar_reference_header() -> Value {
    let path =
        grammar_fixture_root().join("tests/fixtures/kaggriculture/grammar-v4-reference.jsonl.gz");
    let output = Command::new("gzip").arg("-dc").arg(&path).output().unwrap();
    assert!(
        output.status.success(),
        "gzip {}: {:?}",
        path.display(),
        output
    );
    assert!(output.stdout.len() <= 4 * 1024 * 1024);
    let text = std::str::from_utf8(&output.stdout).unwrap();
    let header: Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
    let keys: HashSet<_> = header
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        keys,
        HashSet::from([
            "type",
            "schema_version",
            "reference_commit",
            "slot_names",
            "slot_widths",
            "tables",
        ])
    );
    assert_eq!(header["type"], "header");
    assert_eq!(header["schema_version"], 1);
    assert_eq!(
        header["reference_commit"],
        "65f0eac5bb00b18a9d3acce319c2a231cbd5dff0"
    );
    assert_eq!(header["slot_names"], serde_json::json!(SLOT_NAMES));
    assert_eq!(header["slot_widths"], serde_json::json!(SLOT_WIDTHS));
    header
}

fn grammar_tables_json(tables: &GrammarTables) -> Value {
    serde_json::json!({
        "unit_kind": tables.unit_kind,
        "unit_item": tables.unit_item,
        "unit_quantity_present": tables.unit_quantity_present,
        "unit_quantity_high": tables.unit_quantity_high,
        "unit_quantity_low": tables.unit_quantity_low,
        "market_kind": tables.market_kind,
        "market_item": tables.market_item,
        "market_quantity": tables.market_quantity,
    })
}

fn support_bits(width: usize, admitted: impl Fn(usize) -> bool) -> Vec<bool> {
    (0..width).map(admitted).collect()
}

fn singleton(width: usize, token: usize) -> Vec<bool> {
    support_bits(width, |value| value == token)
}

// This table oracle is the explicit §4 command support, independent of State
// and the production extraction routine; the recorded native tables add a
// separately generated source oracle for every bit.
fn specified_tables_json() -> Value {
    let unit_item: Vec<_> = (0..20)
        .map(|kind| {
            support_bits(16, |item| match kind {
                6 | 7 => (1..=12).contains(&item),
                8 => (1..=5).contains(&item),
                _ => item == 0,
            })
        })
        .collect();
    let unit_present: Vec<_> = (0..20)
        .map(|kind| support_bits(2, |bit| bit == 0 || matches!(kind, 6 | 7)))
        .collect();
    let unit_high: Vec<_> = (0..2)
        .map(|present| support_bits(32, |digit| present == 1 || digit == 0))
        .collect();
    let unit_low: Vec<_> = (0..2)
        .map(|present| {
            (0..2)
                .map(|high_zero| {
                    support_bits(32, |digit| {
                        if present == 0 {
                            digit == 0
                        } else {
                            high_zero == 0 || digit > 0
                        }
                    })
                })
                .collect::<Vec<_>>()
        })
        .collect();
    let market_item: Vec<_> = (0..8)
        .map(|kind| {
            support_bits(16, |item| match kind {
                3 => (1..=5).contains(&item),
                4 => matches!(item, 1 | 9),
                5 => (10..=12).contains(&item),
                6 => (1..=9).contains(&item),
                _ => item == 0,
            })
        })
        .collect();
    let market_quantity: Vec<_> = (0..8)
        .map(|kind| support_bits(32, |digit| (3..=6).contains(&kind) || digit == 0))
        .collect();
    serde_json::json!({
        "unit_kind": support_bits(20, |kind| (1..=18).contains(&kind)),
        "unit_item": unit_item,
        "unit_quantity_present": unit_present,
        "unit_quantity_high": unit_high,
        "unit_quantity_low": unit_low,
        "market_kind": vec![true; 8],
        "market_item": market_item,
        "market_quantity": market_quantity,
    })
}

fn boolean_count(value: &Value) -> usize {
    match value {
        Value::Bool(_) => 1,
        Value::Array(values) => values.iter().map(boolean_count).sum(),
        Value::Object(values) => values.values().map(boolean_count).sum(),
        other => panic!("table contains a non-boolean leaf: {other}"),
    }
}

fn specified_state_mask(state: State, p: &GrammarPlan) -> Vec<bool> {
    let ready = usize::from(state.units) == p.shape().actors();
    support_bits(state.slot().width(), |token| match state.slot() {
        Slot::UnitActor => token == if ready { 0 } else { usize::from(state.units) },
        Slot::UnitKind => {
            if ready {
                token == 0
            } else {
                (1..=18).contains(&token)
            }
        },
        Slot::UnitTarget => token == 0,
        Slot::UnitItem => match state.unit {
            UnitPhase::Transfer => (1..=12).contains(&token),
            UnitPhase::Plant => (1..=5).contains(&token),
            UnitPhase::None | UnitPhase::Ordinary => token == 0,
        },
        Slot::UnitQuantityPresent => state.unit == UnitPhase::Transfer || token == 0,
        Slot::UnitQuantityHigh => state.quantity_present || token == 0,
        Slot::UnitQuantity => {
            if state.quantity_present {
                !state.quantity_high_zero || token > 0
            } else {
                token == 0
            }
        },
        Slot::MarketKind => {
            if ready
                && state.unit == UnitPhase::None
                && usize::from(state.orders) < p.shape().order_limit()
            {
                token != 1 || p.shape().actors() + usize::from(state.hires) < p.shape().hire_limit()
            } else {
                token == 0
            }
        },
        Slot::MarketItem => match state.market {
            MarketPhase::Seed => (1..=5).contains(&token),
            MarketPhase::Product => matches!(token, 1 | 9),
            MarketPhase::Animal => (10..=12).contains(&token),
            MarketPhase::Sell => (1..=9).contains(&token),
            MarketPhase::None | MarketPhase::Hire | MarketPhase::Other => token == 0,
        },
        Slot::MarketQuantityHigh | Slot::MarketQuantity => {
            state.market == MarketPhase::Seed || token == 0
        },
        Slot::Stop => {
            token
                == usize::from(
                    ready && state.unit == UnitPhase::None && state.market == MarketPhase::None,
                )
        },
    })
}

fn table_overlay_mask(state: State, p: &GrammarPlan, tables: &GrammarTables) -> Vec<bool> {
    let ready = usize::from(state.units) == p.shape().actors();
    match state.slot() {
        Slot::UnitKind if !ready => tables.unit_kind.to_vec(),
        Slot::UnitItem => tables.unit_item[match state.unit {
            UnitPhase::Transfer => 6,
            UnitPhase::Plant => 8,
            UnitPhase::None | UnitPhase::Ordinary => 0,
        }]
        .to_vec(),
        Slot::UnitQuantityPresent => {
            tables.unit_quantity_present[if state.unit == UnitPhase::Transfer {
                6
            } else {
                0
            }]
            .to_vec()
        },
        Slot::UnitQuantityHigh => {
            tables.unit_quantity_high[usize::from(state.quantity_present)].to_vec()
        },
        Slot::UnitQuantity => tables.unit_quantity_low[usize::from(state.quantity_present)]
            [usize::from(state.quantity_high_zero)]
        .to_vec(),
        Slot::MarketKind
            if ready
                && state.unit == UnitPhase::None
                && usize::from(state.orders) < p.shape().order_limit() =>
        {
            let mut mask = tables.market_kind.to_vec();
            mask[1] = p.shape().actors() + usize::from(state.hires) < p.shape().hire_limit();
            mask
        },
        Slot::MarketItem => tables.market_item[match state.market {
            MarketPhase::Seed => 3,
            MarketPhase::Product => 4,
            MarketPhase::Animal => 5,
            MarketPhase::Sell => 6,
            MarketPhase::None | MarketPhase::Hire | MarketPhase::Other => 0,
        }]
        .to_vec(),
        Slot::MarketQuantityHigh | Slot::MarketQuantity => {
            tables.market_quantity[if state.market == MarketPhase::Seed {
                3
            } else {
                0
            }]
            .to_vec()
        },
        // Ordinals, phase availability and the STOP bit are runtime overlays,
        // not exported local command tables.
        _ => specified_state_mask(state, p),
    }
}

fn specified_transition(state: State, token: i64) -> Option<State> {
    if state.slot() == Slot::Stop {
        return (token == 0).then_some(State {
            shape: state.shape,
            units: state.units + u16::from(state.unit != UnitPhase::None),
            orders: state.orders + u8::from(state.market != MarketPhase::None),
            hires: state.hires + u8::from(state.market == MarketPhase::Hire),
            slot: Slot::UnitActor,
            unit: UnitPhase::None,
            market: MarketPhase::None,
            quantity_present: false,
            quantity_high_zero: false,
        });
    }
    let mut next = state;
    next.slot = Slot::try_from(state.slot() as usize + 1).unwrap();
    match state.slot() {
        Slot::UnitKind => {
            next.unit = match token {
                0 => UnitPhase::None,
                6 | 7 => UnitPhase::Transfer,
                8 => UnitPhase::Plant,
                _ => UnitPhase::Ordinary,
            }
        },
        Slot::UnitItem if state.unit == UnitPhase::Plant => next.unit = UnitPhase::Ordinary,
        Slot::UnitQuantityPresent => {
            next.quantity_present = token == 1;
            if state.unit == UnitPhase::Transfer && token == 0 {
                next.unit = UnitPhase::Ordinary;
            }
        },
        Slot::UnitQuantityHigh => next.quantity_high_zero = state.quantity_present && token == 0,
        Slot::UnitQuantity => {
            next.quantity_present = false;
            next.quantity_high_zero = false;
            if state.unit == UnitPhase::Transfer {
                next.unit = UnitPhase::Ordinary;
            }
        },
        Slot::MarketKind => {
            next.market = match token {
                0 => MarketPhase::None,
                1 => MarketPhase::Hire,
                3 => MarketPhase::Seed,
                4 => MarketPhase::Product,
                5 => MarketPhase::Animal,
                6 => MarketPhase::Sell,
                2 | 7 => MarketPhase::Other,
                _ => panic!("unadmitted market token {token}"),
            }
        },
        Slot::MarketItem
            if matches!(
                state.market,
                MarketPhase::Product | MarketPhase::Animal | MarketPhase::Sell
            ) =>
        {
            next.market = MarketPhase::Seed
        },
        Slot::MarketQuantity if state.market == MarketPhase::Seed => {
            next.market = MarketPhase::Other
        },
        _ => {},
    }
    Some(next)
}

fn check_reachable_states(p: &GrammarPlan, tables: &GrammarTables) {
    let start = p.start();
    assert_eq!(start.shape, p.shape());
    assert_eq!(
        (start.units, start.orders, start.hires, start.slot()),
        (0, 0, 0, Slot::UnitActor)
    );
    let mut pending = vec![start];
    let mut seen = HashSet::new();
    let mut predecessors: HashMap<State, HashSet<State>> = HashMap::new();
    let mut terminal_parents = Vec::new();
    while let Some(state) = pending.pop() {
        if !seen.insert(state) {
            continue;
        }
        let expected = specified_state_mask(state, p);
        assert_eq!(
            table_overlay_mask(state, p, tables),
            expected,
            "table overlay {state:?}"
        );
        assert!(expected.iter().any(|&bit| bit), "dead state {state:?}");
        for initial in [false, true] {
            let mut mask = vec![initial; state.slot().width()];
            state.write_mask(p, &mut mask).unwrap();
            assert_eq!(mask, expected, "mask {state:?}");
        }
        for (choice, &admitted) in expected.iter().enumerate() {
            let token = i64::try_from(choice).unwrap();
            assert_eq!(state.allows(token, p), admitted, "allows {state:?} {token}");
            let advanced = state.advance(token, p);
            if !admitted {
                assert!(
                    !advanced.unwrap_err().is_empty(),
                    "empty error {state:?} {token}"
                );
                continue;
            }
            let advanced = advanced.unwrap();
            assert_eq!(
                advanced,
                specified_transition(state, token),
                "transition {state:?} {token}"
            );
            match advanced {
                Some(next) => {
                    predecessors.entry(next).or_default().insert(state);
                    pending.push(next);
                },
                None => {
                    assert_eq!((state.slot(), token), (Slot::Stop, 1));
                    terminal_parents.push(state);
                },
            }
        }
        for token in [
            -1,
            i64::try_from(state.slot().width()).unwrap(),
            65537,
            i64::MIN,
            i64::MAX,
        ] {
            assert!(!state.allows(token, p), "out of range {state:?} {token}");
            assert!(
                state.advance(token, p).is_err(),
                "out of range {state:?} {token}"
            );
        }
    }
    assert!(
        seen.len() <= MAX_ACTORS * 24 + 66 * 32,
        "state bound {:?}: {}",
        p.shape(),
        seen.len()
    );
    // Every reachable state can finish at STOP; do not infer this merely from
    // finding one terminal on an otherwise possibly dead graph.
    let mut can_finish = HashSet::new();
    while let Some(state) = terminal_parents.pop() {
        if !can_finish.insert(state) {
            continue;
        }
        if let Some(parents) = predecessors.get(&state) {
            terminal_parents.extend(parents);
        }
    }
    assert_eq!(can_finish, seen, "STOP reachability for {:?}", p.shape());
}

fn advance_prefix(mut state: State, p: &GrammarPlan, prefix: &[i64]) -> State {
    for &token in prefix {
        state = state.advance(token, p).unwrap().unwrap();
    }
    state
}

fn check_actor_ordinals_and_can_act() {
    for actors in 1..=MAX_ACTORS {
        for orders in [1, MAX_ORDERS] {
            let p = plan(
                i64::try_from(actors).unwrap(),
                i64::try_from(orders).unwrap(),
                241,
            )
            .unwrap();
            assert_eq!(
                (
                    p.shape().actors(),
                    p.shape().order_limit(),
                    p.shape().hire_limit()
                ),
                (actors, orders, 241)
            );
            let expected: Vec<_> = (0..MAX_FRAMES)
                .map(|frame| frame < actors + orders + 1)
                .collect();
            for initial in [false, true] {
                let mut mask = vec![initial; MAX_FRAMES];
                write_can_act(&p, &mut mask).unwrap();
                assert_eq!(mask, expected);
            }
            let mut state = p.start();
            for actor in 0..actors {
                let mut mask = vec![true; UNIT_ACTOR_WIDTH];
                state.write_mask(&p, &mut mask).unwrap();
                assert_eq!(mask, singleton(UNIT_ACTOR_WIDTH, actor));
                let frame = [
                    i64::try_from(actor).unwrap(),
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ];
                state = advance_prefix(state, &p, &frame);
            }
            let mut ordinal = vec![true; UNIT_ACTOR_WIDTH];
            state.write_mask(&p, &mut ordinal).unwrap();
            assert_eq!(ordinal, singleton(UNIT_ACTOR_WIDTH, 0));
            let stop_state = advance_prefix(state, &p, &[0; 11]);
            assert_eq!(stop_state.advance(1, &p).unwrap(), None);
            // can_act is observation capacity, independent of early STOP.
            let mut after_stop = vec![true; MAX_FRAMES];
            write_can_act(&p, &mut after_stop).unwrap();
            assert_eq!(after_stop, expected);
        }
    }
}

fn check_mask_errors_and_plan_affinity() {
    let p = plan(1, 10, 241).unwrap();
    let mut state = p.start();
    let pass = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    for token in pass {
        for len in [0, state.slot().width() - 1, state.slot().width() + 1] {
            let mut output: Vec<_> = (0..len).map(|index| index % 2 == 0).collect();
            let before = output.clone();
            assert!(!state.write_mask(&p, &mut output).unwrap_err().is_empty());
            assert_eq!(output, before, "failed mask write changed output");
        }
        state = state.advance(token, &p).unwrap().unwrap();
    }
    for len in [0, MAX_FRAMES - 1, MAX_FRAMES + 1] {
        let mut output: Vec<_> = (0..len).map(|index| index % 2 == 0).collect();
        let before = output.clone();
        assert!(!write_can_act(&p, &mut output).unwrap_err().is_empty());
        assert_eq!(output, before, "failed can_act write changed output");
    }
    let dense = plan(241, 10, 241).unwrap();
    let equal = plan(241, 10, 241).unwrap();
    let other = plan(1, 1, 1).unwrap();
    let cursor = advance_prefix(dense.start(), &dense, &[0, 6, 0, 1]);
    assert_eq!(cursor.slot(), Slot::UnitQuantityPresent);
    for token in [-1, 0, 1, 2, i64::MAX] {
        assert!(!cursor.allows(token, &other));
        assert_eq!(
            cursor.advance(token, &other).unwrap_err(),
            "grammar plan/state shape mismatch"
        );
    }
    for len in [0, cursor.slot().width(), cursor.slot().width() + 1] {
        let mut output = vec![true; len];
        let before = output.clone();
        assert_eq!(
            cursor.write_mask(&other, &mut output).unwrap_err(),
            "grammar plan/state shape mismatch"
        );
        assert_eq!(output, before);
    }
    let mut mask = vec![false; cursor.slot().width()];
    cursor.write_mask(&equal, &mut mask).unwrap();
    assert_eq!(mask, vec![true, true]);
    for token in [0, 1] {
        assert!(cursor.allows(token, &equal));
        assert_eq!(cursor.advance(token, &dense), cursor.advance(token, &equal));
    }
}

fn check_market_digit_equality() {
    let p = plan(1, 1, 241).unwrap();
    let after_unit = advance_prefix(p.start(), &p, &[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    let kind_state = advance_prefix(after_unit, &p, &[0; 7]);
    assert_eq!(kind_state.slot(), Slot::MarketKind);
    for kind in 0..8 {
        let item_state = kind_state.advance(kind, &p).unwrap().unwrap();
        let mut items = vec![false; MARKET_ITEM_WIDTH];
        item_state.write_mask(&p, &mut items).unwrap();
        for (item, allowed) in items.into_iter().enumerate() {
            if !allowed {
                continue;
            }
            let high_state = item_state
                .advance(i64::try_from(item).unwrap(), &p)
                .unwrap()
                .unwrap();
            let mut high_mask = vec![false; MARKET_QUANTITY_HIGH_WIDTH];
            high_state.write_mask(&p, &mut high_mask).unwrap();
            for (high, &admitted) in high_mask.iter().enumerate() {
                if !admitted {
                    continue;
                }
                let low_state = high_state
                    .advance(i64::try_from(high).unwrap(), &p)
                    .unwrap()
                    .unwrap();
                let mut low_mask = vec![false; MARKET_QUANTITY_WIDTH];
                low_state.write_mask(&p, &mut low_mask).unwrap();
                assert_eq!(
                    low_mask, high_mask,
                    "market digits differ: kind={kind} item={item} high={high}"
                );
            }
        }
    }
}

#[test]
fn grammar_tables_match_reference_and_reachable_states() {
    let tables = grammar_tables().unwrap();
    let actual = grammar_tables_json(&tables);
    assert_eq!(boolean_count(&actual), 964);
    assert_eq!(
        actual,
        grammar_tables_json(&grammar_tables().unwrap()),
        "nondeterministic tables"
    );
    assert_eq!(
        actual,
        specified_tables_json(),
        "independent command supports"
    );
    assert_eq!(
        actual,
        grammar_reference_header()["tables"],
        "recorded reference-plan bits"
    );
    for orders in 1..=10 {
        check_reachable_states(&plan(1, orders, 11).unwrap(), &tables);
    }
    for (actors, orders, hires) in [
        (2, 10, 1),
        (17, 10, 16),
        (231, 10, 241),
        (240, 10, 241),
        (241, 10, 241),
    ] {
        check_reachable_states(&plan(actors, orders, hires).unwrap(), &tables);
    }
    // A changes ordinal/ready/capacity, and O changes queue availability. The
    // schedule covers all local categories, not every numeric A/O/H triple.
    check_actor_ordinals_and_can_act();
    check_mask_errors_and_plan_affinity();
    check_market_digit_equality();
}

fn pass_block(actors: usize) -> (Vec<i64>, i64) {
    let mut tokens = vec![0; TOKENS_PER_SEAT];
    for actor in 0..actors {
        tokens[actor * SLOTS] = i64::try_from(actor).unwrap();
        tokens[actor * SLOTS + 1] = 1;
    }
    tokens[actors * SLOTS + 11] = 1;
    (tokens, i64::try_from(actors + 1).unwrap())
}

fn block(frames: &[[i64; SLOTS]]) -> (Vec<i64>, i64) {
    let mut tokens = vec![0; TOKENS_PER_SEAT];
    for (target, source) in tokens.chunks_exact_mut(SLOTS).zip(frames) {
        target.copy_from_slice(source);
    }
    (tokens, i64::try_from(frames.len()).unwrap())
}

// Literal reference command matrices, independent of masks and enum rendering.
const COMMAND_UNITS: [&str; 19] = [
    "NONE",
    "PASS",
    "NORTH",
    "SOUTH",
    "EAST",
    "WEST",
    "PICKUP",
    "PLACE",
    "PLANT",
    "WATER",
    "HARVEST",
    "DROP",
    "BUILD_COOP",
    "BUILD_PASTURE",
    "FEED",
    "FERTILIZE",
    "COLLECT_FERTILIZER",
    "CARE",
    "DIG",
];
const COMMAND_MARKETS: [&str; 8] = [
    "NONE",
    "HIRE",
    "BUY_LAND",
    "BUY_SEED",
    "BUY_PRODUCT",
    "BUY_ANIMAL",
    "SELL",
    "EMPTY",
];
const COMMAND_ITEMS: [&str; 13] = [
    "NONE",
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
    "GOOSE",
    "COW",
    "SHEEP",
];

fn command_cases() -> Vec<(Vec<i64>, i64, Value)> {
    let mut cases = Vec::new();
    let mut stop = [0; SLOTS];
    stop[11] = 1;
    for (kind, name) in COMMAND_UNITS.iter().enumerate().skip(1) {
        let items: Vec<usize> = match kind {
            6 | 7 => (1..=12).collect(),
            8 => (1..=5).collect(),
            _ => vec![0],
        };
        let quantities = if matches!(kind, 6 | 7) {
            vec![None, Some(1), Some(31), Some(32), Some(1023)]
        } else {
            vec![None]
        };
        for item in items {
            for quantity in &quantities {
                let mut frame = [0; SLOTS];
                frame[1] = i64::try_from(kind).unwrap();
                frame[3] = i64::try_from(item).unwrap();
                let mut expected = vec![serde_json::json!(name)];
                if item != 0 {
                    expected.push(serde_json::json!(COMMAND_ITEMS[item]));
                }
                if let Some(quantity) = quantity {
                    frame[4] = 1;
                    frame[5] = quantity / 32;
                    frame[6] = quantity % 32;
                    expected.push(serde_json::json!(quantity));
                }
                let (tokens, length) = block(&[frame, stop]);
                cases.push((
                    tokens,
                    length,
                    serde_json::json!({"farmer":expected,"hands":[],"market":[]}),
                ));
            }
        }
    }
    assert_eq!(cases.len(), 140);
    let mut pass = [0; SLOTS];
    pass[1] = 1;
    let mut empty = [0; SLOTS];
    empty[7] = 7;
    for (kind, name) in COMMAND_MARKETS.iter().enumerate().skip(1) {
        let items: Vec<usize> = match kind {
            3 => (1..=5).collect(),
            4 => vec![1, 9],
            5 => (10..=12).collect(),
            6 => (1..=9).collect(),
            _ => vec![0],
        };
        let quantities = if (3..=6).contains(&kind) {
            vec![0, 1, 31, 32, 1023]
        } else {
            vec![0]
        };
        for item in items {
            for quantity in &quantities {
                let mut frame = [0; SLOTS];
                frame[7] = i64::try_from(kind).unwrap();
                frame[8] = i64::try_from(item).unwrap();
                frame[9] = quantity / 32;
                frame[10] = quantity % 32;
                let mut expected = if kind == 7 {
                    vec![]
                } else {
                    vec![serde_json::json!(name)]
                };
                if item != 0 {
                    expected.extend([
                        serde_json::json!(COMMAND_ITEMS[item]),
                        serde_json::json!(quantity),
                    ]);
                }
                let (tokens, length) = block(&[pass, empty, frame, empty, stop]);
                cases.push((
                    tokens,
                    length,
                    serde_json::json!({"farmer":["PASS"],"hands":[],"market":[[],expected,[]]}),
                ));
            }
        }
    }
    assert_eq!(cases.len(), 238);
    cases
}

#[test]
fn grammar_exact_commands() {
    let p = plan(1, 10, 241).unwrap();
    for (tokens, length, expected) in command_cases() {
        let before = tokens.clone();
        let actual = decode(&p, &tokens, length).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(actual.to_string(), expected.to_string());
        assert_eq!(tokens, before);
    }
    let p = plan(17, 10, 16).unwrap();
    let (tokens, length) = pass_block(17);
    assert_eq!(
        decode(&p, &tokens, length).unwrap()["hands"]
            .as_array()
            .unwrap()
            .len(),
        16
    );
}

#[test]
fn grammar_rejects_malformed_i64() {
    let p = plan(1, 10, 241).unwrap();
    let (base, length) = pass_block(1);
    for (slot, &width) in SLOT_WIDTHS.iter().enumerate() {
        for bad in [-1, i64::try_from(width).unwrap(), 65537, i64::MIN, i64::MAX] {
            for frame in [0, 2, 251] {
                let mut tokens = base.clone();
                tokens[frame * SLOTS + slot] = bad;
                let before = tokens.clone();
                let error = decode(&p, &tokens, length).unwrap_err();
                assert!(error.contains(SLOT_NAMES[slot]), "{error}");
                assert!(error.contains(&format!("frame {frame}")), "{error}");
                assert_eq!(tokens, before);
            }
        }
    }
    for (index, value) in [
        (0, 1),
        (1, 0),
        (1, 19),
        (2, 1),
        (3, 1),
        (4, 1),
        (5, 1),
        (6, 1),
        (7, 1),
        (8, 1),
        (9, 1),
        (10, 1),
        (11, 1),
        (23, 0),
        (13, 1),
        (15, 1),
    ] {
        let mut tokens = base.clone();
        tokens[index] = value;
        let error = decode(&p, &tokens, length).unwrap_err();
        assert!(error.contains("support"), "{index}={value}: {error}");
        assert!(error.contains(SLOT_NAMES[index % SLOTS]), "{error}");
    }
    for item in 13..=15 {
        let mut tokens = base.clone();
        tokens[1] = 6;
        tokens[3] = item;
        let error = decode(&p, &tokens, length).unwrap_err();
        assert!(error.contains("unit_item"));
    }
    let mut zero = base.clone();
    zero[1] = 6;
    zero[3] = 1;
    zero[4] = 1;
    let error = decode(&p, &zero, length).unwrap_err();
    assert!(error.contains("unit_quantity"));
    let mut after_stop = base.clone();
    after_stop[35] = 1;
    assert!(decode(&p, &after_stop, 3).unwrap_err().contains("STOP"));
    // Both complete frames are individually admitted, but HIRE is not STOP.
    // This reaches the final-cursor check instead of an earlier length/support check.
    let mut missing_sentinel = base.clone();
    missing_sentinel[SLOTS + Slot::MarketKind as usize] = 1;
    missing_sentinel[SLOTS + Slot::Stop as usize] = 0;
    let before = missing_sentinel.clone();
    assert_eq!(
        decode(&p, &missing_sentinel, 2).unwrap_err(),
        "program lacks distinct final STOP"
    );
    assert_eq!(missing_sentinel, before);
    for bad_length in [-1, 0, 1, 13, 253, i64::MIN, i64::MAX] {
        let error = decode(&p, &base, bad_length).unwrap_err();
        assert!(error.contains("length"));
    }
    for size in [3023, 3025] {
        let error = decode(&p, &vec![0; size], length).unwrap_err();
        assert!(error.contains("shape"));
    }
    for frame in [2, 251] {
        let mut tokens = base.clone();
        tokens[frame * SLOTS + 1] = 1;
        let error = decode(&p, &tokens, length).unwrap_err();
        assert!(
            error.contains("padding")
                && error.contains(&format!("frame {frame}"))
                && error.contains("unit_kind"),
            "{error}"
        );
    }
    for (a, o, h, field) in [
        (-1, 1, 1, "actors"),
        (0, 1, 1, "actors"),
        (242, 1, 1, "actors"),
        (i64::MAX, 1, 1, "actors"),
        (1, -1, 1, "order_limit"),
        (1, 0, 1, "order_limit"),
        (1, 11, 1, "order_limit"),
        (1, 1, -1, "hire_limit"),
        (1, 1, 0, "hire_limit"),
        (1, 1, 242, "hire_limit"),
        (1, 1, i64::MAX, "hire_limit"),
    ] {
        let error = plan(a, o, h).unwrap_err();
        assert!(error.contains("shape") && error.contains(field), "{error}");
    }
}

fn fixture_records() -> Vec<Value> {
    // Header verification is shared with the independently recorded table test.
    grammar_reference_header();
    let output = Command::new("gzip")
        .arg("-dc")
        .arg(
            grammar_fixture_root()
                .join("tests/fixtures/kaggriculture/grammar-v4-reference.jsonl.gz"),
        )
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(output.stdout.len() <= 4 * 1024 * 1024);
    let mut ids = HashSet::new();
    std::str::from_utf8(&output.stdout)
        .unwrap()
        .lines()
        .skip(1)
        .map(|line| {
            let row: Value = serde_json::from_str(line).unwrap();
            let mut keys = HashSet::from([
                "type",
                "id",
                "source",
                "shape",
                "tokens",
                "length",
                "padding_edits",
                "sampler",
                "training_decoder",
                "python_codec",
                "v4_expected",
            ]);
            match row["source"].as_str().unwrap() {
                "sampled" => keys.extend(["seed", "selection_mode"]),
                "replay" => keys.extend(["episode", "from_step", "seat", "raw_action"]),
                "mutation" => {
                    keys.insert("category");
                },
                "replay_rejected" => {
                    keys.extend(["episode", "from_step", "seat", "raw_action", "category"])
                },
                other => panic!("unknown fixture source {other}"),
            }
            assert_eq!(
                row.as_object()
                    .unwrap()
                    .keys()
                    .map(String::as_str)
                    .collect::<HashSet<_>>(),
                keys
            );
            assert_eq!(row["type"], "program");
            assert!(ids.insert(row["id"].as_str().unwrap().to_owned()));
            assert_eq!(
                row["shape"]
                    .as_object()
                    .unwrap()
                    .keys()
                    .map(String::as_str)
                    .collect::<HashSet<_>>(),
                HashSet::from(["actors", "order_limit", "hire_limit"])
            );
            assert_eq!(
                row["v4_expected"]
                    .as_object()
                    .unwrap()
                    .keys()
                    .map(String::as_str)
                    .collect::<HashSet<_>>(),
                HashSet::from(["accepted", "action", "reason"])
            );
            for name in ["sampler", "training_decoder", "python_codec"] {
                let verdict = &row[name];
                let keys = verdict
                    .as_object()
                    .unwrap()
                    .keys()
                    .map(String::as_str)
                    .collect::<HashSet<_>>();
                if verdict.get("not_applicable").is_some() {
                    assert_eq!(keys, HashSet::from(["not_applicable", "reason"]));
                    assert_eq!(verdict["not_applicable"], true);
                    assert!(!verdict["reason"].as_str().unwrap().is_empty());
                } else {
                    assert_eq!(keys, HashSet::from(["accepted", "action", "error"]));
                    if verdict["accepted"].as_bool().unwrap() {
                        assert!(verdict["error"].is_null());
                        assert!(verdict["action"].is_object());
                    } else {
                        assert!(verdict["action"].is_null());
                        assert!(!verdict["error"].as_str().unwrap().is_empty());
                    }
                }
            }
            row
        })
        .collect()
}

fn fixture_program(row: &Value) -> (GrammarPlan, Vec<i64>, i64) {
    let shape = &row["shape"];
    let p = plan(
        shape["actors"].as_i64().unwrap(),
        shape["order_limit"].as_i64().unwrap(),
        shape["hire_limit"].as_i64().unwrap(),
    )
    .unwrap();
    let tokens: Vec<i64> = serde_json::from_value(row["tokens"].clone()).unwrap();
    let mut padded = vec![0; TOKENS_PER_SEAT];
    padded[..tokens.len()].copy_from_slice(&tokens);
    for edit in row["padding_edits"].as_array().unwrap() {
        let f = usize::try_from(edit[0].as_u64().unwrap()).unwrap();
        let s = usize::try_from(edit[1].as_u64().unwrap()).unwrap();
        padded[f * SLOTS + s] = edit[2].as_i64().unwrap();
    }
    (p, padded, row["length"].as_i64().unwrap())
}

fn fixture_action_matches(actual: &Value, row: &Value) -> Result<(), String> {
    if actual != &row["v4_expected"]["action"] {
        return Err(format!("expected action mismatch: {}", row["id"]));
    }
    for oracle in ["sampler", "training_decoder", "python_codec"] {
        if row[oracle]["accepted"] != true || row[oracle]["action"] != *actual {
            return Err(format!("{oracle} action mismatch"));
        }
    }
    Ok(())
}

#[test]
fn grammar_fixture_oracle() {
    let mut scheduled = 0;
    let mut sampled = 0;
    let mut replays = 0;
    let mut dense = 0;
    let mut dense_full = 0;
    let mut negatives = 0;
    let mut positive_control = 0;
    let mut mutation_checked = false;
    for row in fixture_records() {
        if row["source"] == "replay_rejected" {
            continue;
        }
        let (p, tokens, length) = fixture_program(&row);
        let result = decode(&p, &tokens, length);
        if row["v4_expected"]["accepted"].as_bool().unwrap() {
            let actual = result.unwrap();
            fixture_action_matches(&actual, &row).unwrap();
            if row["source"] == "sampled" || row["source"] == "replay" {
                scheduled += 1;
                sampled += usize::from(row["source"] == "sampled");
                replays += usize::from(row["source"] == "replay");
                dense += usize::from(p.shape().actors() == 241);
                dense_full += usize::from(p.shape().actors() == 241 && length == 252);
            } else {
                positive_control += 1;
                assert_eq!(row["category"], "padding-zero");
            }
            if !mutation_checked {
                let mut corrupt = row.clone();
                corrupt["v4_expected"]["action"]["farmer"] = serde_json::json!(["NOT_A_COMMAND"]);
                assert!(
                    fixture_action_matches(&actual, &corrupt).is_err(),
                    "in-memory expected-action corruption escaped"
                );
                fixture_action_matches(&actual, &row).unwrap();
                mutation_checked = true;
            }
        } else {
            assert!(result.is_err(), "admitted negative {}", row["id"]);
            negatives += 1;
            let category = row["category"].as_str().unwrap();
            if category.starts_with("padding-") {
                for oracle in ["sampler", "training_decoder", "python_codec"] {
                    assert_eq!(row[oracle]["accepted"], true);
                }
            } else if category == "missing-stop" {
                assert_eq!(row["sampler"]["accepted"], false);
                for oracle in ["training_decoder", "python_codec"] {
                    assert_eq!(row[oracle]["not_applicable"], true);
                }
            } else {
                assert_eq!(row["sampler"]["accepted"], false);
                assert_eq!(row["python_codec"]["accepted"], false);
                assert_eq!(
                    row["training_decoder"]["accepted"],
                    category.starts_with("hire-capacity-")
                );
            }
        }
    }
    assert_eq!(
        (
            scheduled,
            sampled,
            replays,
            dense,
            dense_full,
            negatives,
            positive_control
        ),
        (320, 256, 64, 64, 22, 43, 1)
    );
    assert!(mutation_checked);
}

#[test]
fn grammar_encode_round_trip() {
    let p = plan(1, 10, 241).unwrap();
    for (tokens, length, expected) in command_cases() {
        let mut out = vec![-99; TOKENS_PER_SEAT];
        assert_eq!(encode(&p, &expected, &mut out).unwrap(), length);
        assert_eq!(out, tokens);
        assert_eq!(decode(&p, &out, length).unwrap(), expected);
    }
    let mut replay_count = 0;
    for row in fixture_records() {
        if !row["v4_expected"]["accepted"].as_bool().unwrap() {
            continue;
        }
        let (p, tokens, length) = fixture_program(&row);
        let action = decode(&p, &tokens, length).unwrap();
        let mut out = vec![-1; TOKENS_PER_SEAT];
        assert_eq!(encode(&p, &action, &mut out).unwrap(), length);
        assert_eq!(out, tokens, "{}", row["id"]);
        assert_eq!(decode(&p, &out, length).unwrap(), action);
        if row["source"] == "replay" {
            replay_count += 1;
            out.fill(-5);
            assert_eq!(encode(&p, &row["raw_action"], &mut out).unwrap(), length);
            assert_eq!(out, tokens, "raw replay {}", row["id"]);
        }
    }
    assert_eq!(replay_count, 64);
}

fn assert_encode_error(p: &GrammarPlan, action: Value, prefix: &str) {
    let mut out: Vec<_> = (0..TOKENS_PER_SEAT)
        .map(|i| -i64::try_from(i).unwrap() - 1)
        .collect();
    let before = out.clone();
    let error = encode(p, &action, &mut out).unwrap_err();
    assert!(
        error.starts_with(prefix),
        "expected {prefix:?}, got {error:?}; {action}"
    );
    assert_eq!(out, before, "partial write on {error}");
}

#[test]
fn grammar_encode_rejects() {
    let p = plan(1, 10, 241).unwrap();
    let base = serde_json::json!({"farmer":["PASS"],"hands":[],"market":[]});
    for bad in [
        Value::Null,
        serde_json::json!([]),
        serde_json::json!({}),
        serde_json::json!({"farmer":["PASS"],"hands":[]}),
        serde_json::json!({"farmer":["PASS"],"hands":[],"market":[],"other":0}),
    ] {
        assert_encode_error(&p, bad, "action keys");
    }
    for field in ["farmer", "hands", "market"] {
        let mut bad = base.clone();
        bad.as_object_mut().unwrap().remove(field);
        assert_encode_error(&p, bad, "action keys");
    }
    for value in [
        Value::Null,
        serde_json::json!({}),
        serde_json::json!("not an array"),
    ] {
        let mut bad = base.clone();
        bad["hands"] = value.clone();
        assert_encode_error(&p, bad, "hands syntax");
        let mut bad = base.clone();
        bad["market"] = value;
        assert_encode_error(&p, bad, "market syntax");
    }
    let mut bad = base.clone();
    bad["hands"] = serde_json::json!([["PASS"]]);
    assert_encode_error(&p, bad, "extra hand commands");
    assert_encode_error(&plan(2, 10, 241).unwrap(), base.clone(), "hand count");
    for command in [
        serde_json::json!([]),
        Value::Null,
        serde_json::json!("PASS"),
        serde_json::json!(["PASS", 1]),
        serde_json::json!(["PLANT"]),
        serde_json::json!(["PLANT", "WHEAT", 1]),
        serde_json::json!(["PICKUP"]),
        serde_json::json!(["PICKUP", "WHEAT", 1, 2]),
    ] {
        let mut bad = base.clone();
        bad["farmer"] = command;
        assert_encode_error(&p, bad, "unsupported unit command arguments");
    }
    for name in ["pass", "NONE", "STOP", "EMPTY", "UNKNOWN"] {
        let mut bad = base.clone();
        bad["farmer"] = serde_json::json!([name]);
        assert_encode_error(&p, bad, "unknown unit command");
        let mut bad = base.clone();
        bad["market"] = serde_json::json!([[name]]);
        assert_encode_error(&p, bad, "unknown market command");
    }
    for command in [Value::Null, serde_json::json!(3), serde_json::json!("HIRE")] {
        let mut bad = base.clone();
        bad["market"] = serde_json::json!([command]);
        assert_encode_error(&p, bad, "market command syntax");
    }
    for command in [
        serde_json::json!(["HIRE", 1]),
        serde_json::json!(["BUY_SEED", "WHEAT"]),
        serde_json::json!(["SELL", "WHEAT", 1, 2]),
    ] {
        let mut bad = base.clone();
        bad["market"] = serde_json::json!([command]);
        assert_encode_error(&p, bad, "unsupported market arguments");
    }
    for item in [
        serde_json::json!("NONE"),
        serde_json::json!("wheat"),
        serde_json::json!("UNKNOWN"),
        serde_json::json!(3),
        Value::Null,
    ] {
        let mut bad = base.clone();
        bad["farmer"] = serde_json::json!(["PLANT", item]);
        assert_encode_error(&p, bad, "unit item syntax");
        let mut bad = base.clone();
        bad["market"] = serde_json::json!([["BUY_SEED", item, 1]]);
        assert_encode_error(&p, bad, "market item syntax");
    }
    let mut bad = base.clone();
    bad["farmer"] = serde_json::json!(["PLANT", "GOOSE"]);
    assert_encode_error(&p, bad, "unit item syntax");
    let mut bad = base.clone();
    bad["market"] = serde_json::json!([["BUY_PRODUCT", "CARROT", 1]]);
    assert_encode_error(&p, bad, "market item syntax");
    for quantity in [
        serde_json::json!(-1),
        serde_json::json!(1024),
        serde_json::json!(i64::MAX),
        serde_json::json!(u64::MAX),
        serde_json::json!(1.0),
        serde_json::json!("1"),
        serde_json::json!(true),
        Value::Null,
    ] {
        let mut bad = base.clone();
        bad["farmer"] = serde_json::json!(["PICKUP", "WHEAT", quantity]);
        assert_encode_error(&p, bad, "quantity range");
        let mut bad = base.clone();
        bad["market"] = serde_json::json!([["SELL", "WHEAT", quantity]]);
        assert_encode_error(&p, bad, "quantity range");
    }
    let mut bad = base.clone();
    bad["farmer"] = serde_json::json!(["PLACE", "WHEAT", 0]);
    assert_encode_error(&p, bad, "quantity range");
    let mut bad = base.clone();
    bad["market"] = serde_json::json!(vec![vec!["HIRE"]; 11]);
    assert_encode_error(&p, bad, "market queue limit");
    for (actors, hire_limit, hires) in [(1, 1, 1), (17, 16, 1), (240, 241, 2), (241, 241, 1)] {
        let action = serde_json::json!({"farmer":["PASS"],"hands":vec![vec!["PASS"];actors-1],"market":vec![vec!["HIRE"];hires]});
        assert_encode_error(
            &plan(i64::try_from(actors).unwrap(), 10, hire_limit).unwrap(),
            action,
            "hire capacity",
        );
    }
    for size in [0, 3023, 3025] {
        let mut out = vec![7; size];
        let before = out.clone();
        assert!(encode(&p, &base, &mut out).unwrap_err().contains("shape"));
        assert_eq!(out, before);
    }
    for row in fixture_records() {
        if row["source"] == "replay_rejected" {
            let (p, _, _) = fixture_program(&row);
            assert_encode_error(
                &p,
                row["raw_action"].clone(),
                row["category"].as_str().unwrap(),
            );
        }
    }
}

fn complete_first(mut state: State, p: &GrammarPlan) -> Option<State> {
    loop {
        let mut mask = vec![false; state.slot().width()];
        state.write_mask(p, &mut mask).unwrap();
        let token = i64::try_from(mask.iter().position(|&bit| bit).unwrap()).unwrap();
        let next = state.advance(token, p).unwrap();
        if state.slot() == Slot::Stop {
            return next;
        }
        state = next.unwrap();
    }
}

fn market_start(p: &GrammarPlan) -> State {
    let mut state = p.start();
    for actor in 0..p.shape().actors() {
        state = advance_prefix(
            state,
            p,
            &[
                i64::try_from(actor).unwrap(),
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        );
    }
    advance_prefix(state, p, &[0; 7])
}

type QueueLaw = std::collections::BTreeMap<Vec<usize>, f64>;
fn sequential_law(p: &GrammarPlan, rows: &[[f64; 8]; 3]) -> QueueLaw {
    fn visit(
        p: &GrammarPlan,
        rows: &[[f64; 8]; 3],
        state: State,
        queue: Vec<usize>,
        mass: f64,
        out: &mut QueueLaw,
    ) {
        assert_eq!(state.slot(), Slot::MarketKind);
        let mut mask = [false; 8];
        state.write_mask(p, &mut mask).unwrap();
        if queue.len() == 3 {
            assert_eq!(
                mask,
                [true, false, false, false, false, false, false, false]
            );
            assert!(complete_first(state.advance(0, p).unwrap().unwrap(), p).is_none());
            *out.entry(queue).or_default() += mass;
            return;
        }
        let row = rows[queue.len()];
        let z: f64 = row
            .iter()
            .zip(mask)
            .filter_map(|(weight, ok)| ok.then_some(*weight))
            .sum();
        for (kind, &ok) in mask.iter().enumerate() {
            if !ok {
                continue;
            }
            let after_kind = state
                .advance(i64::try_from(kind).unwrap(), p)
                .unwrap()
                .unwrap();
            let next = complete_first(after_kind, p);
            let next_mass = mass * row[kind] / z;
            if kind == 0 {
                assert!(next.is_none());
                *out.entry(queue.clone()).or_default() += next_mass;
            } else {
                let mut next_queue = queue.clone();
                next_queue.push(kind);
                visit(
                    p,
                    rows,
                    advance_prefix(next.unwrap(), p, &[0; 7]),
                    next_queue,
                    next_mass,
                    out,
                );
            }
        }
    }
    let mut out = QueueLaw::new();
    visit(p, rows, market_start(p), vec![], 1.0, &mut out);
    out
}

#[test]
fn grammar_coupled_hire_exact_distribution() {
    let mut rows = [
        [1., 5., 2., 3., 4., 2., 1., 6.],
        [5., 7., 1., 3., 2., 4., 6., 1.],
        [2., 6., 4., 1., 5., 3., 7., 2.],
    ];
    for row in &mut rows {
        let z: f64 = row.iter().sum();
        for value in row {
            *value /= z;
        }
    }
    let races: Vec<Vec<(usize, usize, f64)>> = rows
        .iter()
        .map(|row| {
            let mut pairs = Vec::new();
            for (kind, &mass) in row.iter().enumerate() {
                if kind != 1 {
                    pairs.push((kind, kind, mass));
                    pairs.push((1, kind, row[1] * mass / (1. - row[1])));
                }
            }
            assert_eq!(pairs.len(), 14);
            pairs
        })
        .collect();
    for budget in [0, 1, 2, 3, 10] {
        let mut parallel = QueueLaw::new();
        let mut corrected_none = false;
        let mut corrected_empty = false;
        for &first in &races[0] {
            for &second in &races[1] {
                for &third in &races[2] {
                    let outcomes = [first, second, third];
                    let mut hires = 0;
                    let mut queue = vec![];
                    for &(raw, nonhire, _) in &outcomes {
                        let choice = if hires < budget { raw } else { nonhire };
                        corrected_none |= raw == 1 && choice == 0;
                        corrected_empty |= raw == 1 && choice == 7;
                        hires += usize::from(raw == 1);
                        if choice == 0 {
                            break;
                        }
                        queue.push(choice);
                    }
                    *parallel.entry(queue).or_default() +=
                        outcomes.iter().map(|o| o.2).product::<f64>();
                }
            }
        }
        let p = plan(1, 3, i64::try_from(1 + budget).unwrap()).unwrap();
        let sequential = sequential_law(&p, &rows);
        assert_eq!(
            parallel.keys().collect::<Vec<_>>(),
            sequential.keys().collect::<Vec<_>>(),
            "queue keys budget={budget}"
        );
        for (queue, mass) in &parallel {
            assert!(
                (mass - sequential[queue]).abs() < 1e-12,
                "budget={budget},queue={queue:?}"
            );
        }
        for law in [&parallel, &sequential] {
            assert!((law.values().sum::<f64>() - 1.).abs() < 1e-12);
        }
        assert!(parallel.contains_key(&vec![7, 7, 7]));
        let within_budget = parallel
            .keys()
            .all(|queue| queue.iter().filter(|&&kind| kind == 1).count() <= budget);
        assert!(within_budget);
        if budget < 3 {
            assert!(corrected_none && corrected_empty);
        }
        if budget == 0 {
            let above_limit = sequential_law(&plan(17, 3, 16).unwrap(), &rows);
            assert_eq!(above_limit, sequential);
        }
    }
    let p = plan(240, 10, 241).unwrap();
    let state = market_start(&p);
    assert!(state.allows(1, &p));
    let mut current = state.advance(1, &p).unwrap().unwrap();
    while current.slot() != Slot::Stop {
        assert_eq!(current.hires, 0);
        current = current.advance(0, &p).unwrap().unwrap();
    }
    assert_eq!(current.hires, 0);
    let closed = current.advance(0, &p).unwrap().unwrap();
    assert_eq!(closed.hires, 1);
    let next = advance_prefix(closed, &p, &[0; 7]);
    assert!(!next.allows(1, &p));
    let error = next.advance(1, &p).unwrap_err();
    assert!(error.starts_with("hire capacity"));
    let full = plan(241, 10, 241).unwrap();
    assert!(!market_start(&full).allows(1, &full));
}

fn walk_recorded(p: &GrammarPlan, tokens: &[i64], length: i64) {
    let active = usize::try_from(length).unwrap() * SLOTS;
    let mut state = Some(p.start());
    for (index, &token) in tokens[..active].iter().enumerate() {
        let current = state.expect("terminal before last token");
        assert_eq!(current.slot() as usize, index % SLOTS);
        let mut mask = vec![false; current.slot().width()];
        current.write_mask(p, &mut mask).unwrap();
        assert!(mask[usize::try_from(token).unwrap()], "token {index}");
        state = current.advance(token, p).unwrap();
        assert_eq!(state.is_none(), index + 1 == active);
    }
    assert!(state.is_none());
}

#[test]
fn grammar_replays_reference_program() {
    let p = plan(2, 10, 241).unwrap();
    let mut tokens = vec![0; TOKENS_PER_SEAT];
    tokens[Slot::UnitKind as usize] = 8; // PLANT (pinned contract enum)
    tokens[Slot::UnitItem as usize] = 1; // WHEAT
    tokens[SLOTS + Slot::UnitActor as usize] = 1;
    tokens[SLOTS + Slot::UnitKind as usize] = 1; // PASS
    tokens[2 * SLOTS + Slot::MarketKind as usize] = 1; // HIRE
    tokens[3 * SLOTS + Slot::MarketKind as usize] = 3; // BUY_SEED
    tokens[3 * SLOTS + Slot::MarketItem as usize] = 1;
    tokens[3 * SLOTS + Slot::MarketQuantityHigh as usize] = 1;
    tokens[3 * SLOTS + Slot::MarketQuantity as usize] = 1;
    tokens[4 * SLOTS + Slot::Stop as usize] = 1;
    walk_recorded(&p, &tokens, 5);
    assert_eq!(
        decode(&p, &tokens, 5).unwrap(),
        serde_json::json!({
            "farmer": ["PLANT", "WHEAT"], "hands": [["PASS"]],
            "market": [["HIRE"], ["BUY_SEED", "WHEAT", 33]]
        })
    );
    let mut count = 0;
    let mut dense_full = 0;
    for row in fixture_records() {
        if !row["v4_expected"]["accepted"].as_bool().unwrap() {
            continue;
        }
        let (p, tokens, length) = fixture_program(&row);
        walk_recorded(&p, &tokens, length);
        assert_eq!(
            decode(&p, &tokens, length).unwrap(),
            row["v4_expected"]["action"]
        );
        count += 1;
        dense_full += usize::from(p.shape().actors() == 241 && length == 252);
    }
    assert_eq!((count, dense_full), (321, 22));
}
