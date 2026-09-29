from pathlib import Path
exec(Path('ops/rebuild-2026-09-29/codex/verify-merge-teacher-r1/guard-mutations/run_mutations.py').read_text().split('mutants = [')[0])
EXTRA='tests/kaggriculture/test_verifier_guard_supplement.py'
(SCRATCH/EXTRA).write_text('''import dataclasses
import pytest
import torch
from owl.kaggriculture import types as kt
from owl.model import kaggriculture as km
from owl.model import kaggriculture_actor as ka
from owl.model import kaggriculture_teacher as kt_teacher
from tests.kaggriculture.helpers import _tiny, _base_case


def _no_encode(*args, **kwargs):
    raise AssertionError("invalid input reached encode_observations")


@pytest.mark.parametrize("side", ["student", "teacher"])
def test_value_ce_shape_guard(side):
    model = _tiny()
    live = torch.ones(1, 2, dtype=torch.bool)
    student = torch.zeros(1, 2, 2)
    teacher = torch.full((1, 2, 2), 0.5)
    if side == "student":
        student = student[0]
    else:
        teacher = teacher[0]
    with pytest.raises(ValueError, match="must have shape"):
        model.teacher_value_cross_entropy(student, teacher, value_mask=live)


@pytest.mark.parametrize("path", ["precompute", "cache", "combined"])
@pytest.mark.parametrize("malformed", ["tokens_shape", "lengths_shape", "dtype"])
def test_teacher_paths_validate_action_layout_before_encoding(monkeypatch, path, malformed):
    student, teacher = _tiny().eval(), _tiny(seed=6).eval()
    obs, actions = _base_case()
    targets = teacher.compute_teacher_distillation_targets(obs, actions)
    bad = kt.KaggricultureActions(tokens=actions.tokens.clone(), lengths=actions.lengths.clone())
    if malformed == "tokens_shape":
        bad.tokens = bad.tokens[..., :1]
    elif malformed == "lengths_shape":
        bad.lengths = bad.lengths[..., :1]
    else:
        bad.tokens = bad.tokens.float()
    monkeypatch.setattr(student, "encode_observations", _no_encode)
    with pytest.raises(ValueError, match="action (tokens|lengths)"):
        if path == "precompute":
            student.compute_teacher_distillation_targets(obs, bad)
        elif path == "cache":
            student.evaluate_actions_with_cached_teacher(obs, bad, targets)
        else:
            student.evaluate_actions_with_teacher(obs, bad, teacher)


@pytest.mark.parametrize("path", ["cache", "combined"])
@pytest.mark.parametrize("field", ["hidden_state", "dones"])
def test_teacher_paths_reject_temporal_inputs(monkeypatch, path, field):
    student, teacher = _tiny().eval(), _tiny(seed=6).eval()
    obs, actions = _base_case()
    targets = teacher.compute_teacher_distillation_targets(obs, actions)
    monkeypatch.setattr(student, "encode_observations", _no_encode)
    value = object() if field == "hidden_state" else torch.zeros_like(obs.still_playing)
    with pytest.raises(ValueError, match=field):
        if path == "cache":
            student.evaluate_actions_with_cached_teacher(obs, actions, targets, **{field:value})
        else:
            student.evaluate_actions_with_teacher(obs, actions, teacher, **{field:value})


def test_cached_path_admits_student_replay():
    student, teacher = _tiny().eval(), _tiny(seed=6).eval()
    obs, actions = _base_case()
    targets = teacher.compute_teacher_distillation_targets(obs, actions)
    bad = kt.KaggricultureActions(tokens=actions.tokens.clone(), lengths=actions.lengths.clone())
    bad.lengths[0, 0] += 1
    with pytest.raises(ka.GrammarReplayError, match="length"):
        student.evaluate_actions_with_cached_teacher(obs, bad, targets)


def test_combined_path_admits_teacher_before_student_replay(monkeypatch):
    student, teacher = _tiny().eval(), _tiny(seed=6).eval()
    obs, actions = _base_case()
    bad = kt.KaggricultureActions(tokens=actions.tokens.clone(), lengths=actions.lengths.clone())
    bad.lengths[0, 0] += 1
    def no_student_policy(*args, **kwargs):
        raise AssertionError("invalid teacher replay reached student policy")
    monkeypatch.setattr(student, "_policy", no_student_policy)
    with pytest.raises(ka.GrammarReplayError, match="length"):
        student.evaluate_actions_with_teacher(obs, bad, teacher)


def test_requested_policy_core_field_presence():
    with pytest.raises(RuntimeError, match="requested policy-core output"):
        km._require(None)


def test_validated_chunk_field_presence():
    with pytest.raises(ValueError, match="missing a field"):
        kt_teacher._present(None)
''')

def run_extra(label, selectors):
    result=subprocess.run([str(PYTHON),'-m','pytest','-q','-p','no:cacheprovider','--tb=short',*[EXTRA+'::'+s for s in selectors]],cwd=SCRATCH,env=ENV,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (OUT/(label+'.log')).write_text(result.stdout)
    return result.returncode,result.stdout

extra_tests=['test_value_ce_shape_guard','test_teacher_paths_validate_action_layout_before_encoding','test_teacher_paths_reject_temporal_inputs','test_cached_path_admits_student_replay','test_combined_path_admits_teacher_before_student_replay','test_requested_policy_core_field_presence','test_validated_chunk_field_presence']
code,output=run_extra('supplement_baseline',extra_tests)
print('Supplement baseline',code,output,flush=True)
assert code==0
supplement=[
('ce_shape',KM,'teacher_value_cross_entropy','if','tuple(tensor.shape) != expected',['test_value_ce_shape_guard']),
('precompute_layout',KM,'compute_teacher_distillation_targets','expr','_check_action_layout(actions',['test_teacher_paths_validate_action_layout_before_encoding[ tokens ]']),
('cached_layout',KM,'evaluate_actions_with_cached_teacher','expr','_check_action_layout(actions',['test_teacher_paths_validate_action_layout_before_encoding']),
('combined_layout',KM,'evaluate_actions_with_teacher','expr','_check_action_layout(actions',['test_teacher_paths_validate_action_layout_before_encoding']),
('cached_stateless_call',KM,'evaluate_actions_with_cached_teacher','expr','self._require_stateless_replay(hidden_state, dones)',['test_teacher_paths_reject_temporal_inputs']),
('combined_stateless_call',KM,'evaluate_actions_with_teacher','expr','self._require_stateless_replay(hidden_state, dones)',['test_teacher_paths_reject_temporal_inputs']),
('cached_student_replay',KM,'_teacher_evaluation','expr','check_replay_flags(result.valid)',['test_cached_path_admits_student_replay']),
('combined_teacher_replay',KM,'evaluate_actions_with_teacher','expr','check_replay_flags(teacher_result.valid)',['test_combined_path_admits_teacher_before_student_replay']),
('policy_core_field_presence',KM,'_require','if','value is None',['test_requested_policy_core_field_presence']),
('concat_internal_presence',KT,'_present','if','value is None',['test_validated_chunk_field_presence']),
]
supplement[1]=(*supplement[1][:-1],['test_teacher_paths_validate_action_layout_before_encoding'])
rows=[]
try:
    for label,path,func,kind,needle,selectors in supplement:
        old=change_node(path,func,kind,needle,'False' if kind=='if' else 'pass')
        try:
            code,output=run_extra(label,selectors)
            row={'mutant':label,'file':path,'function':func,'replaced':old,'replacement':'False' if kind=='if' else 'pass','tests':selectors,'exit_code':code,'result':'SURVIVED' if code==0 else 'KILLED' if code==1 and 'FAILED ' in output else 'SETUP_ERROR','tail':output[-2500:]}
            rows.append(row)
            print(label,row['result'],output.splitlines()[-1],flush=True)
            (OUT/'supplement_summary.json').write_text(json.dumps(rows,indent=2))
        finally:
            (SCRATCH/path).write_bytes(originals[path])
            assert (SCRATCH/path).read_bytes()==originals[path]
finally:
    for path,data in originals.items():
        (SCRATCH/path).write_bytes(data)
    hashes={path:{'source':hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),'restored_scratch':hashlib.sha256((SCRATCH/path).read_bytes()).hexdigest()} for path in originals}
    assert all(x['source']==x['restored_scratch'] for x in hashes.values())
    (OUT/'supplement_restoration.json').write_text(json.dumps(hashes,indent=2))
    shutil.copy2(SCRATCH/EXTRA,OUT/'supplement_tests.py')
result=subprocess.run([str(PYTHON),'-c','import owl.model.kaggriculture as km; import owl.model.kaggriculture_teacher as kt; import scripts.run_ppo as rp; print(km.__file__); print(kt.__file__); print(rp.__file__)'],cwd=SCRATCH,env=ENV,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
(OUT/'import_paths.log').write_text(result.stdout)
print('Supplement complete:',len(rows),'mutants, restored byte-for-byte',flush=True)
