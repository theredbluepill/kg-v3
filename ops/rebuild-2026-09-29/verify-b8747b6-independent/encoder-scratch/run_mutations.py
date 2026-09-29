from pathlib import Path
import hashlib, json, os, subprocess, time
ROOT=Path(__file__).resolve().parents[4]
SCRATCH=ROOT/'.codex-tmp/verify-b8747b6-native'
OUT=Path(__file__).resolve().parent
TARGET=ROOT/'.codex-tmp/verify-b8747b6-native-target'
BASE='src/kaggriculture/'
jobs=[]
def add(name,file,old,new,test):jobs.append((name,file,old,new,test))
add('buffer-length',BASE+'buffers.rs','if actual != expected {','if false && actual != expected {','shape_every_field_rejects_short_and_long_without_writes')
add('buffer-zero-env',BASE+'buffers.rs','if n_envs == 0 {','if false && n_envs == 0 {','shape_zero_environments_is_an_error')
add('buffer-product',BASE+'buffers.rs','.checked_mul(2)','.checked_mul(0)','shape_product_overflow_is_an_error')
add('buffer-publication',BASE+'buffers.rs','if self.n_envs != out.n_envs {','if false && self.n_envs != out.n_envs {','shape_publish_mismatched_environments_leaves_all_fields_unchanged')
add('buffer-clear',BASE+'buffers.rs','self.actor_inventory_rank.fill([0; 12]);','// mutant: omitted rank clearing','shape_clear_overwrites_every_field_with_positive_zero_or_false')
add('config-envelope',BASE+'config.rs','if config.board_size != 10 {','if false && config.board_size != 10 {','config_rejects_envelope_violations_without_output_writes')
add('config-day-bound',BASE+'config.rs','.is_none_or(|product| product > 240)','.is_none_or(|product| product > 241)','config_day_order_product_has_exact_240_boundary_and_checked_overflow')
add('finite-f32',BASE+'observe.rs','if !value.is_finite() || !narrowed.is_finite() {','if false && (!value.is_finite() || !narrowed.is_finite()) {','config_weed_requires_finite_nonnegative_numeric_f32_without_unit_cap')
add('hire-own-rival',BASE+'observe.rs','out.player_features[role][10] = config.prepared_hire_cost(farm.hires_today);','out.player_features[role][10] = config.prepared_hire_cost(own.hires_today);','hire_cost_uses_each_public_count')
add('tile-coordinate',BASE+'observe.rs','let cell = y * 10 + x;','let cell = x * 10 + y;','tile_asymmetric_coordinates_and_every_channel')
add('actor-coordinate',BASE+'observe.rs','out.actor_cell[actor] = 10 * position[1] + position[0];','out.actor_cell[actor] = 10 * position[0] + position[1];','actors_public_positions_roles_and_all_private_channels')
add('seat-privacy',BASE+'observe.rs','&observation.snapshot.privates[seat.index()],','&observation.snapshot.privates[1 - seat.index()],','privacy_private_values_and_key_order_do_not_cross_seats')
add('inventory-rank',BASE+'observe.rs','out.actor_inventory_rank[actor][item] = rank as i64 + 1;','out.actor_inventory_rank[actor][item] = inventory.len() as i64 - rank as i64;','actors_rank_remove_reinsert_preserves_counts_and_replacement_preserves_rank')
add('storage-rank',BASE+'observe.rs','out.storage_rank[item] = rank as i64 + 1;','out.storage_rank[item] = own_private.shed.len() as i64 - rank as i64;','storage_rank_remove_reinsert_changes_order_not_counts')
add('row-diagnostic',BASE+'observe.rs','if condition {\n        Ok(())','if condition || true {\n        Ok(())','context_check_row_rejects_corruptions_in_every_domain')
add('one-snapshot',BASE+'observe.rs','let snapshot = self.acquire_snapshot();','let _redundant_snapshot = self.acquire_snapshot();\n        let snapshot = self.acquire_snapshot();','snapshot_both_seats_acquire_once_and_keep_staging_allocations')
add('relocated-root-grammar',BASE+'grammar.rs','let mut command = vec![Value::from(kind.as_str())];','let mut command = vec![Value::from("PASS")];','decoded_command_matrices_execute_both_seats')
add('reconstruction-scalar',BASE+'oracle_corpus.rs','row.banks[0] / 200000.0,','row.banks[0] / 100000.0,','reconstruction_hand_anchors_cover_every_range_and_boundary')
add('bitwise-signed-zero',BASE+'oracle_corpus.rs','if actual[offset].to_bits() != recorded[offset].to_bits() {','if actual[offset] != recorded[offset] {','reconstruction_comparator_names_record_seat_group_offset_and_signed_zero')
add('reconstruction-duplicate',BASE+'oracle_corpus.rs','!self.covered[offset],','true || !self.covered[offset],','reconstruction_coverage_rejects_duplicate_offset')
add('reconstruction-zero-hole',BASE+'oracle_corpus.rs','assert!(covered, "legacy offset {offset} never reconstructed");','assert!(true || covered, "legacy offset {offset} never reconstructed");','reconstruction_coverage_rejects_unwritten_zero_hole')
add('byteplane-order',BASE+'oracle_corpus.rs','interleaved[index * 4 + plane] = plane_bytes[index];','interleaved[index * 4 + (3-plane)] = plane_bytes[index];','reconstruction_byteplane_decode_preserves_bits_and_rejects_bad_lengths')
add('record-strict-header',BASE+'oracle_corpus.rs','if object.len() != keys.len() || keys.iter().any(|key| !object.contains_key(*key)) {','if false && (object.len() != keys.len() || keys.iter().any(|key| !object.contains_key(*key))) {','producer_strict_record_and_quota_guard_reject_bad_inputs')
add('legacy-domain',BASE+'oracle_corpus.rs','(0..=16_777_216).contains(&value),','true || (0..=16_777_216).contains(&value),','producer_legacy_domain_rejects_old_oracle_limits_without_narrowing_v3')
add('full-corpus-market',BASE+'observe.rs','out.market_int[product][channel] = integer;','out.market_int[product][channel] = integer + 1;','compare_observation_oracle')
add('l4-feature-unification','src/rules_engine/generation.rs','#[serde(deserialize_with = "fixture_float")]','// mutant: default f64 decoder','arbitrary_precision_uniform_fixture_float')
tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
tracked=[p for p in tracked if p and (ROOT/p).is_file()]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
before={p:digest(ROOT/p) for p in tracked}
(OUT/'tracked-before.json').write_text(json.dumps(before,indent=2)+'\n')
results=[]
env=dict(os.environ,CARGO_TARGET_DIR=str(TARGET))
for name,file,old,new,test in jobs:
    path=SCRATCH/file; original=path.read_bytes(); source=original.decode()
    count=source.count(old)
    if count<1:raise RuntimeError((name,'missing needle'))
    path.write_text(source.replace(old,new,1))
    start=time.monotonic()
    command=['cargo','test','--manifest-path',str(SCRATCH/'Cargo.toml'),'--locked','--offline',test,'--','--test-threads=1']
    try:
        with (OUT/(name+'.log')).open('w') as log:
            process=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=90)
        log=(OUT/(name+'.log')).read_text()
        killed=process.returncode==101 and 'test result: FAILED.' in log and 'error[E' not in log
        result={'name':name,'file':file,'test_filter':test,'argv':command,'returncode':process.returncode,'killed_by_test':killed,'seconds':round(time.monotonic()-start,3),'sha256_before':hashlib.sha256(original).hexdigest(),'sha256_mutant':digest(path)}
    finally:
        path.write_bytes(original)
        assert path.read_bytes()==original
    result['sha256_restored']=digest(path)
    results.append(result)
    (OUT/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(name,result['returncode'],result['killed_by_test'],flush=True)
after={p:digest(ROOT/p) for p in tracked}
assert before==after, 'tracked worktree changed'
(OUT/'tracked-after.json').write_text(json.dumps(after,indent=2)+'\n')
print('all tracked hashes identical',len(before),flush=True)
