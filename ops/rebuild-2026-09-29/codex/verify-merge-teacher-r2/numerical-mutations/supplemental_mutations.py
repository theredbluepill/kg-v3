import hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path.cwd(); OUT=ROOT/'ops/rebuild-2026-09-29/codex/verify-merge-teacher-r2/numerical-mutations'
SCRATCH=Path(json.loads((OUT/'scratch-receipt.json').read_text())['scratch'])
PY=ROOT/'.venv/bin/python'; ENV={**os.environ,'PYTHONPATH':str(SCRATCH/'python')+os.pathsep+str(SCRATCH),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
TARGET='tests/kaggriculture/test_teacher.py'; COMMON='python/owl/model/actor/common.py'; KM='python/owl/model/kaggriculture.py'; KA='python/owl/model/kaggriculture_actor.py'
def t(name):return TARGET+'::test_'+name
(SCRATCH/'tests/kaggriculture/test_teacher_precision_probe.py').write_text('''import torch
from owl.model.actor.common import categorical_kl_from_logits

def test_fp64_teacher_probability_reference():
    teacher = torch.tensor([[1.00000001, 1.00000002]], dtype=torch.float64)
    student = torch.tensor([[0.3, 0.5]], dtype=torch.float64)
    mask = torch.ones_like(teacher, dtype=torch.bool)
    lp = teacher.log_softmax(-1)
    expected = (lp.exp() * (lp - student.log_softmax(-1))).sum(-1)
    actual = categorical_kl_from_logits(teacher, student, mask)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-14)
''')
def run(name,tests):
    with (OUT/(name+'.log')).open('w') as f:
        p=subprocess.run([str(PY),'-m','pytest',*tests,'-q'],cwd=SCRATCH,env=ENV,stdout=f,stderr=subprocess.STDOUT,timeout=150)
    return p.returncode,(OUT/(name+'.log')).read_text().splitlines()[-1]
probe='tests/kaggriculture/test_teacher_precision_probe.py'
assert run('precision-probe-baseline',[probe])[0]==0
mutants=[
('fp64-both-demoted',COMMON,[('teacher_logits.to(dtype)','teacher_logits.float()'),('student_logits.to(dtype)','student_logits.float()')],[t('kl_gradient_matches_finite_differences_in_fp64')]),
('fp64-teacher-demoted-reference',COMMON,[('teacher_logits.to(dtype)','teacher_logits.float()')],[probe]),
('plain-replay-exposes-teacher-logits',KA,[('slot_logits=collected if collect_logits else None,','slot_logits=collected,')],[t('forward_and_evaluate_actions_leave_the_teacher_fields_unset')]),
('value-cache-support-disabled',KM,[('''        # The critic is always the masked winner softmax.
        return True''','''        # mutation
        return False''')],[t('the_model_supports_both_cached_distillation_paths')]),
('teacher-targets-carry-between-call-state',KM,[('''        return KaggricultureTeacherTargets(
            slot_logits=slot_logits,''','''        self._verification_counter = self._verification_counter + 1 if hasattr(self, "_verification_counter") else 1
        if slot_logits is not None:
            slot_logits = {key: value + self._verification_counter * 0.1 for key, value in slot_logits.items()}
        return KaggricultureTeacherTargets(
            slot_logits=slot_logits,''')],[t('targets_and_kl_are_seat_isolated_and_stateless')]),
('orbit-value-ce-reduction-changed','python/owl/model/base.py',[(''').sum(dim=-1)

    def count_non_masked_tokens''',''').mean(dim=-1)

    def count_non_masked_tokens''')],[t('ppo_teacher_wrappers_leave_orbit_results_unchanged')]),
]
results=[]
for name,file,changes,tests in mutants:
    path=SCRATCH/file; original=path.read_bytes(); text=original.decode()
    for old,new in changes:
        assert text.count(old)==1,(name,text.count(old))
        text=text.replace(old,new)
    try:
        path.write_text(text); code,summary=run(name,tests)
    finally:path.write_bytes(original)
    r={'name':name,'file':file,'tests':tests,'exit_code':code,'summary':summary,'restored':path.read_bytes()==original,'sha256':hashlib.sha256(original).hexdigest()}
    results.append(r); print(json.dumps(r),flush=True)
    (OUT/'supplemental-results.json').write_text(json.dumps(results,indent=2)+'\n')
print('final',run('supplemental-restored',[TARGET,'tests/kaggriculture/test_merge_teacher_guards.py',probe]),flush=True)
