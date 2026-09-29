from pathlib import Path
import json
from run_core_mutations import Mutation, run_one
r = run_one(Mutation("timing-fixture-density", "src/kaggriculture/lifecycle_timing_tests.rs", "lifecycle_timing_fixtures_step_and_prepare", "let header = super::oracle_corpus::timing_header(dense).unwrap();", "let header = super::oracle_corpus::timing_header(false).unwrap();", "Replace the dense 241-actor diagnostic state with the early state."), 115)
print(json.dumps({k:r[k] for k in ("name", "classification", "restored_byte_exact")}))
