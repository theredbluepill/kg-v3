"""Source-isolated verification; all mutations restored in finally blocks."""
import hashlib, json, os, shutil, subprocess, tempfile
from pathlib import Path
ROOT=Path.cwd()
OUT=ROOT/'ops/rebuild-2026-09-29/codex/verify-merge-teacher-r1/numerical-mutations'
OUT.mkdir(exist_ok=True)
SCRATCH=Path(tempfile.mkdtemp(prefix='kg-merge-teacher-numerical-',dir='/private/tmp'))
for name in ('python','tests','scripts','configs'):
    shutil.copytree(ROOT/name,SCRATCH/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
shutil.copy2(ROOT/'pyproject.toml',SCRATCH/'pyproject.toml')
ENV={**os.environ,'PYTHONPATH':str(SCRATCH/'python')+os.pathsep+str(SCRATCH),'OMP_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1'}
PY=ROOT/'.venv/bin/python'
TARGET='tests/kaggriculture/test_teacher.py'
(SCRATCH/'tests/kaggriculture/test_merge_teacher_guards.py').write_text('''import pytest
import torch._inductor.config as config
from tests.kaggriculture.helpers import _tiny, _base_case

@pytest.mark.parametrize("path", ["precompute", "cached", "combined_student", "combined_teacher"])
def test_teacher_paths_keep_integration_gemm_guard(monkeypatch, path):
    student, teacher = _tiny().eval(), _tiny(seed=6).eval()
    obs, actions = _base_case()
    targets = teacher.compute_teacher_distillation_targets(obs, actions)
    monkeypatch.setattr(config, "max_autotune_gemm_backends", "ATEN,TRITON")
    guarded = teacher if path in ("precompute", "combined_teacher") else student
    guarded.compiled_regions_require_gemm_backends = True
    with pytest.raises(RuntimeError, match="compiled Kaggriculture regions require"):
        if path == "precompute":
            teacher.compute_teacher_distillation_targets(obs, actions)
        elif path == "cached":
            student.evaluate_actions_with_cached_teacher(obs, actions, targets)
        else:
            student.evaluate_actions_with_teacher(obs, actions, teacher)
''')
M=[]
def add(name,file,old,new,tests,count=1): M.append((name,file,old,new,tests,count))
KA='python/owl/model/kaggriculture_actor.py'; KM='python/owl/model/kaggriculture.py'; KT='python/owl/model/kaggriculture_teacher.py'
COMMON='python/owl/model/actor/common.py'; PPO='python/owl/train/ppo.py'
def t(name):return TARGET+'::test_'+name
add('kl-liveness-removed',KA,'return kl.to(dtype=student_logits.dtype) * weight','return kl.to(dtype=student_logits.dtype)',[t('kl_is_zero_for_the_same_teacher_and_exactly_zero_off_support')])
add('student-logits-unmasked',KA,'teacher_logits[slot], floored, mask, unit_live','teacher_logits[slot], logits, mask, unit_live',[t('kl_matches_a_brute_force_oracle_over_the_admissible_values')])
add('cached-mask-negative-infinity',KA,'return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)','return logits.masked_fill(~mask, -torch.inf)',[t('collected_logits_reproduce_the_replay_density')])
add('fp64-demoted',COMMON,'teacher_logits.to(dtype)','teacher_logits.float()', [t('kl_gradient_matches_finite_differences_in_fp64')])
add('student-kl-detached',KA,'return kl.to(dtype=student_logits.dtype) * weight','return kl.detach().to(dtype=student_logits.dtype) * weight',[t('kl_gradients_reach_every_student_head_and_no_teacher_parameter')])
add('capture-kl-diverges',KA,'return kl.to(dtype=student_logits.dtype) * weight','return (kl + (1.0 if torch.compiler.is_compiling() else 0.0)).to(dtype=student_logits.dtype) * weight',[t('fullgraph_captured_core_matches_eager_with_teacher_logits')])
add('cache-indices-reversed',KT,'t[indices]','t[indices.flip(0)]',[t('index_then_concat_restores_the_targets')])
add('grammar-digest-ignores-values','python/owl/kaggriculture/gpu_grammar.py','digest.update(table.contiguous().numpy().tobytes())','digest.update(b"constant")',[t('grammar_signature_tracks_tables_and_hire_limit')])
add('grammar-signature-ignores-hire-limit',KM,'hire_limit=self.action_spec.hire_limit,','hire_limit=241,',[t('grammar_signature_tracks_tables_and_hire_limit')])
add('cache-nbytes-drops-winner',KT,'total += self.winner_probabilities.nbytes','total += 0',[t('nbytes_counts_every_cached_tensor')])
add('orbit-nbytes-drops-continuation','python/owl/model/stateless_transformer_v1.py','total += params.continue_logits.nbytes','total += 0',[t('nbytes_counts_every_cached_tensor')])
add('cache-row-estimate-drops-winner',KT,'for slot in POLICY_SLOTS) + kt.PLAYERS','for slot in POLICY_SLOTS) + 0',[t('cache_bytes_per_row_follow_the_contract_widths')])
add('head-chunk-teacher-slice-reuses-first',KM,'logits[rows] for slot, logits in teacher_logits.items()','logits[:rows.stop - rows.start] for slot, logits in teacher_logits.items()',[t('head_and_trunk_chunking_match_the_unchunked_kl')])
add('cache-precompute-batch-dependent',KM,'slot: logits.reshape(*lead, *logits.shape[1:])','slot: (logits + len(lead) * 0.125 * logits.shape[0]).reshape(*lead, *logits.shape[1:])',[t('chunked_precompute_equals_one_whole_batch_call')])
add('cached-winner-diverges-from-combined',KM,'teacher_winner = teacher_targets.winner_probabilities','teacher_winner = teacher_targets.winner_probabilities * 0.5',[t('cached_path_is_bit_for_bit_the_combined_path')])
add('ce-sums-live-seats',KM,'return (per_seat * live).sum(dim=-1) / live.sum(dim=-1).clamp_min(1.0)','return (per_seat * live).sum(dim=-1)',[t('value_cross_entropy_is_the_live_seat_mean'),t('a_copied_teacher_gives_zero_kl_and_the_student_winner_entropy')])
add('ce-includes-inactive-seats',KM,'live = value_mask.to(dtype=per_seat.dtype)','live = torch.ones_like(value_mask, dtype=per_seat.dtype)',[t('value_cross_entropy_is_the_live_seat_mean')])
add('ce-teacher-gradient-leak',KM,'-teacher_winner_probabilities.detach() * student_winner_log_probabilities','-teacher_winner_probabilities * student_winner_log_probabilities',[t('value_cross_entropy_is_the_live_seat_mean')])
add('precompute-no-grad-removed',KM,'with torch.no_grad():\n            encoded = self.encode_observations(obs)','with torch.enable_grad():\n            encoded = self.encode_observations(obs)',[t('targets_have_the_lead_layout_dtypes_and_optional_fields')])
add('targets-mix-private-seats',KM,'slot: logits.reshape(*lead, *logits.shape[1:])','slot: (logits + logits.roll(1, 0) * 0.125).reshape(*lead, *logits.shape[1:])',[t('targets_and_kl_are_seat_isolated_and_stateless')])
add('checkpoint-persists-grammar',KA,'table.clone(), persistent=False','table.clone(), persistent=True',[t('the_model_supports_both_cached_distillation_paths')])
add('teacher-cache-support-disabled',KM,'def supports_cached_teacher_distillation(self) -> bool:\n        return True','def supports_cached_teacher_distillation(self) -> bool:\n        return False',[t('the_model_supports_both_cached_distillation_paths')])
add('refresh-omits-weight-copy','scripts/run_ppo.py','    target_model.load_state_dict(source_model.state_dict())','    pass # mutation: no weight refresh',[t('last_best_refresh_keeps_the_tables_and_copies_the_student')])
for name,method in [('combined','teacher'),('cached','cached_teacher')]:
    old=('''    if hidden_state is None:
        return model.evaluate_actions_with_'''+method+'''(
            obs,
            actions,
            '''+('teacher,' if name=='combined' else 'teacher_targets,'))
    new=old+'\n            dones=dones,'
    add('ppo-'+name+'-passes-dones',PPO,old,new,[t('ppo_teacher_wrappers_dispatch_statelessly')])
add('integration-gemm-guard-removed',KM,'            require_compiled_gemm_backends()','            pass # mutation: bypass inherited guard',['tests/kaggriculture/test_merge_teacher_guards.py'])

def run(label, tests):
    log=OUT/(label+'.log')
    with log.open('w') as stream:
        proc=subprocess.run([str(PY),'-m','pytest',*tests,'-q'],cwd=SCRATCH,env=ENV,stdout=stream,stderr=subprocess.STDOUT,timeout=150)
    return proc.returncode,log.read_text()
code,txt=run('baseline',[TARGET,'tests/kaggriculture/test_merge_teacher_guards.py'])
if code: raise RuntimeError(txt[-6000:])
results=[]
for name,file,old,new,tests,count in M:
    path=SCRATCH/file; original=path.read_bytes(); text=original.decode()
    if text.count(old)!=count: raise AssertionError((name,text.count(old),count))
    try:
        path.write_text(text.replace(old,new))
        code,output=run(name,tests)
    finally:
        path.write_bytes(original)
    result={'name':name,'file':file,'tests':tests,'exit_code':code,'summary':output.splitlines()[-1], 'restored':path.read_bytes()==original,'sha256':hashlib.sha256(original).hexdigest()}
    results.append(result); print(json.dumps(result),flush=True)
    (OUT/'results.json').write_text(json.dumps(results,indent=2)+'\n')
# Independently reproduce the deferred seams without applying any production mutation.
path=SCRATCH/TARGET; original=path.read_bytes()
try:
    path.write_text(original.decode().replace('@pytest.mark.skip(reason=NEEDS_TRAINER_SEAM)\n','').replace('@pytest.mark.skip(reason=NEEDS_RUN_PPO_GAME_SEAM)\n',''))
    code,output=run('unskipped-integration-probe',[t('trainer_precomputes_once_and_logs_teacher_metrics'),t('trainer_checkpoint_after_a_teacher_iteration_holds_no_teacher_cache'),t('run_ppo_resume_restores_the_teacher_from_checkpoint_last_best'),t('run_ppo_fresh_launch_from_weights_activates_the_last_best_teacher')])
    print('unskipped probe',code,output.splitlines()[-1],flush=True)
finally:path.write_bytes(original)
code,txt=run('restored-baseline',[TARGET,'tests/kaggriculture/test_merge_teacher_guards.py'])
(OUT/'scratch-receipt.json').write_text(json.dumps({'scratch':str(SCRATCH),'final_baseline_exit_code':code,'final_baseline_summary':txt.splitlines()[-1], 'mutations':len(results),'all_restored':all(r['restored'] for r in results)},indent=2)+'\n')
print('restored baseline',code,txt.splitlines()[-1],flush=True)
