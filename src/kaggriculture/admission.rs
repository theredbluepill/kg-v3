//! Admission for the two HIRE cash casts identified by the Task 1.4 audit.

use num_bigint::BigInt;
use num_traits::ToPrimitive;

// u64::MAX rounds up to this value in f64. The cast-safe upper bound is strict.
const UINT64_CASH_LIMIT: f64 = 18_446_744_073_709_551_616.0;

/// Admit actual successful hires before publishing an unpublished candidate.
///
/// `executed` is the checked delta of the kernel's cumulative A_HIRES ledger,
/// captured before an autoreset. Failed hires never advance the Fibonacci
/// index, so this reproduces only the costs that the kernel actually charged.
/// See ops/rebuild-2026-09-29/1.4/cast-audit.md for the seeded-state rounding
/// proof and the required companion economic-counter range checks.
pub(super) fn validate_hire_cash(
    multiplier: i64,
    hires_before: [usize; 2],
    executed: [u64; 2],
) -> Result<(), String> {
    if multiplier < 0 {
        return Err("farmHandCostMult must be nonnegative".into());
    }
    if multiplier == 0 {
        return Ok(());
    }
    let multiplier = BigInt::from(multiplier);
    for seat in 0..2 {
        if executed[seat] == 0 {
            continue;
        }
        let count = usize::try_from(executed[seat])
            .map_err(|_| format!("seat {seat}: executed hire count exceeds usize"))?;
        let end = hires_before[seat]
            .checked_add(count)
            .ok_or_else(|| format!("seat {seat}: executed hire index overflows usize"))?;
        let (mut a, mut b) = (BigInt::from(1), BigInt::from(1));
        for index in 0..end {
            let cost = (&multiplier * &a).to_f64();
            if cost.is_none_or(|cost| !cost.is_finite() || cost >= UINT64_CASH_LIMIT) {
                // Fibonacci costs never decrease. If the threshold is reached
                // before the first executed index, that first hire is unsafe
                // too; stop without constructing arbitrarily large integers.
                let first_unsafe = index.max(hires_before[seat]);
                return Err(format!(
                    "seat {seat}: executed hire index {first_unsafe} exceeds uint64 cash conversion range"
                ));
            }
            (a, b) = (b.clone(), a + b);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::validate_hire_cash;

    #[test]
    fn hire_cash_no_execution_does_not_cap_configuration() {
        assert_eq!(validate_hire_cash(i64::MAX, [240, 240], [0, 0]), Ok(()));
        assert_eq!(validate_hire_cash(0, [240, 240], [10, 10]), Ok(()));
    }

    #[test]
    fn hire_cash_admits_safe_executed_sequences() {
        assert_eq!(validate_hire_cash(1, [0, 4], [10, 3]), Ok(()));
        assert_eq!(validate_hire_cash(i64::MAX, [0, 1], [2, 1]), Ok(()));
        assert_eq!(validate_hire_cash(1_i64 << 62, [3, 0], [1, 0]), Ok(()));
    }

    #[test]
    fn hire_cash_rejects_cost_that_rounds_up_to_two_to_64() {
        // 2 * i64::MAX is below u64::MAX, but its f64 conversion is 2^64.
        let error = validate_hire_cash(i64::MAX, [2, 0], [1, 0]).unwrap_err();
        assert!(error.contains("seat 0"), "{error}");
        assert!(error.contains("hire index 2"), "{error}");
        assert!(error.contains("uint64 cash"), "{error}");
    }

    #[test]
    fn hire_cash_checks_later_executed_hire_and_both_seats() {
        let error = validate_hire_cash(i64::MAX, [0, 0], [0, 3]).unwrap_err();
        assert!(error.contains("seat 1"), "{error}");
        assert!(error.contains("hire index 2"), "{error}");
    }

    #[test]
    fn hire_cash_rejects_oversized_exact_cost_without_float_saturation() {
        let error = validate_hire_cash(1_i64 << 62, [4, 0], [1, 0]).unwrap_err();
        assert!(error.contains("uint64 cash"), "{error}");
    }

    #[test]
    fn hire_cash_rejects_negative_multiplier_explicitly() {
        let error = validate_hire_cash(-1, [0, 0], [0, 0]).unwrap_err();
        assert!(error.contains("nonnegative"), "{error}");
    }
}
