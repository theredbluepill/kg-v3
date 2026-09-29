from pathlib import Path
source=(Path(__file__).parent/'run_mutations.py').read_text()
exec(source[:source.index('tracked=subprocess')])
OUT=OUT/'supplement';OUT.mkdir(exist_ok=True)
jobs=[]
add('corpus-bitwise-coherent-market',BASE+'observe.rs','out.market_int[product][channel] = integer;','let integer = integer + 1;\n            out.market_int[product][channel] = integer;','compare_observation_oracle')
add('recorded-redundancy-guard',BASE+'oracle_corpus.rs','require(recorded[offset].to_bits() == expected.to_bits(), format!(','require(true || recorded[offset].to_bits() == expected.to_bits(), format!(','reconstruction_recorded_constants_and_duplicates_are_checked_independently')
add('decoded-exact-eof',BASE+'oracle_corpus.rs','reader.fill_buf()?.is_empty(),','true || reader.fill_buf()?.is_empty(),','reconstruction_feature_streams_support_both_formats_and_exact_eof')
add('producer-policy-burst',BASE+'oracle_corpus.rs','const HIRE_BURST_HOURS: usize = 8;','const HIRE_BURST_HOURS: usize = 0;','producer_policy_v2_appends_hire_burst_within_order_limit')
add('action-frame-mask',BASE+'observe.rs','*present = frame < frames;','*present = frame + 1 < frames;','context_can_act_boundaries_and_terminal_snapshots_are_live_rows')
exec(source[source.index('tracked=subprocess'):])
