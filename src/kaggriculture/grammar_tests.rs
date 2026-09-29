use super::*;

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
    let names = ["unit_actor", "unit_kind", "unit_target", "unit_item", "unit_quantity_present", "unit_quantity_high", "unit_quantity", "market_kind", "market_item", "market_quantity_high", "market_quantity", "stop"];
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
    for i in 0..19 { assert_eq!(UnitKind::try_from(i).unwrap() as i64, i); }
    for i in 0..8 { assert_eq!(MarketKind::try_from(i).unwrap() as i64, i); }
    for i in 0..13 { assert_eq!(ActionItem::try_from(i).unwrap() as i64, i); }
    for i in [-1, 19, 20, i64::MAX] { assert!(UnitKind::try_from(i).is_err()); }
    for i in [-1, 8, i64::MAX] { assert!(MarketKind::try_from(i).is_err()); }
    for i in [-1, 13, 14, 15, i64::MAX] { assert!(ActionItem::try_from(i).is_err()); }
    assert_eq!(UnitKind::CollectFertilizer.as_str(), "COLLECT_FERTILIZER");
    assert_eq!(MarketKind::BuyProduct.as_str(), "BUY_PRODUCT");
    assert_eq!(ActionItem::Fertilizer.as_str(), "FERTILIZER");
}
