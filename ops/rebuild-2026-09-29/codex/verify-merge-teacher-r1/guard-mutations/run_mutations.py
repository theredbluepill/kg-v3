from pathlib import Path
import ast
import hashlib
import json
import os
import shutil
import subprocess

ROOT = Path.cwd()
OUT = ROOT / 'ops/rebuild-2026-09-29/codex/verify-merge-teacher-r1/guard-mutations'
SCRATCH = OUT / 'scratch'
SCRATCH.mkdir(exist_ok=True)
for directory in ('python', 'scripts', 'tests'):
    shutil.copytree(ROOT / directory, SCRATCH / directory, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
shutil.copy2(ROOT / 'pyproject.toml', SCRATCH / 'pyproject.toml')
PYTHON = ROOT / '.venv/bin/python'
ENV = {**os.environ, 'PYTHONPATH': str(SCRATCH / 'python') + os.pathsep + str(SCRATCH), 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
TEST_FILE = 'tests/kaggriculture/test_teacher.py'
T = {
    'concat': 'test_concat_validates_every_chunk_symmetrically',
    'cache': 'test_cached_admission_rejects_bad_targets_before_any_kernel',
    'combined': 'test_combined_path_rejects_foreign_or_mismatched_teachers',
    'signature': 'test_a_grammar_mismatch_that_replay_admits_is_rejected_by_the_signature',
    'replay': 'test_teacher_targets_reject_programs_the_teacher_grammar_does_not_admit',
    'orbit': 'test_isaiah_cached_teacher_rejects_foreign_targets_before_any_kernel',
    'obs': 'test_teacher_obs_spec_dispatch_covers_kaggriculture',
    'wrapper': 'test_ppo_teacher_wrappers_do_not_pass_dones_to_stateless_models',
}
# Discover wrapper test's exact symbol, avoiding guessed selection.
src = (SCRATCH / TEST_FILE).read_text()
T['wrapper'] = next(n.name for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name.startswith('test_') and any(s in n.name for s in ('wrapper', 'wrappers')) and 'isaiah' not in n.name)

KM='python/owl/model/kaggriculture.py'
KT='python/owl/model/kaggriculture_teacher.py'
SV='python/owl/model/stateless_transformer_v1.py'
PPO='python/owl/train/ppo.py'
RUN='scripts/run_ppo.py'
originals = {p:(SCRATCH/p).read_bytes() for p in (KM,KT,SV,PPO,RUN)}

def run(label, tests):
    cmd = [str(PYTHON), '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '--tb=short', *[f'{TEST_FILE}::{T[t]}' for t in tests]]
    result = subprocess.run(cmd, cwd=SCRATCH, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (OUT / f'{label}.log').write_text(result.stdout)
    return result.returncode, result.stdout

def change_node(path, function, kind, needle, replacement='False'):
    source=originals[path].decode()
    funcs=[n for n in ast.walk(ast.parse(source)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name==function]
    assert len(funcs)==1,(function,len(funcs))
    matches=[]
    for n in ast.walk(funcs[0]):
        if kind=='if' and isinstance(n,ast.If):
            node=n.test
        elif kind=='expr' and isinstance(n,ast.Expr):
            node=n
        else:
            continue
        s=ast.get_source_segment(source,node)
        if needle in s:
            matches.append(node)
    exact=[n for n in matches if ast.get_source_segment(source,n)==needle]
    if exact:
        matches=exact
    assert len(matches)==1,(function,needle, len(matches))
    n=matches[0]
    lines=source.splitlines(keepends=True)
    start=sum(map(len,lines[:n.lineno-1]))+n.col_offset
    stop=sum(map(len,lines[:n.end_lineno-1]))+n.end_col_offset
    text=source[:start]+replacement+source[stop:]
    ast.parse(text)
    (SCRATCH/path).write_text(text)
    return ast.get_source_segment(source,n)

mutants = [
 ('concat_empty',KT,'concat','if','not chunks',['concat']),
 ('concat_slot_presence',KT,'concat','if','(chunk.slot_logits is None)',['concat']),
 ('concat_slot_keys',KT,'concat','if','set(chunk.slot_logits)',['concat']),
 ('concat_winner_presence',KT,'concat','if','chunk.winner_probabilities is None',['concat']),
 ('concat_grammar',KT,'concat','if','chunk.grammar != first.grammar',['concat']),
 ('cache_type',KM,'evaluate_actions_with_cached_teacher','if','not isinstance(teacher_targets',['cache']),
 ('cache_value_presence',KM,'evaluate_actions_with_cached_teacher','if','teacher_winner is None',['cache']),
 ('cache_value_shape',KM,'evaluate_actions_with_cached_teacher','if','tuple(teacher_winner.shape)',['cache']),
 ('cache_slot_presence',KM,'_admit_cached_slot_logits','if','slot_logits is None',['cache']),
 ('cache_grammar',KM,'_admit_cached_slot_logits','if','targets.grammar != self.grammar_signature()',['cache','signature']),
 ('cache_slot_keys',KM,'_admit_cached_slot_logits','if','set(slot_logits) != set(POLICY_SLOTS)',['cache']),
 ('cache_slot_dtype',KM,'_admit_cached_slot_logits','if','logits.dtype not in',['cache']),
 ('cache_slot_shape',KM,'_admit_cached_slot_logits','if','tuple(logits.shape)',['cache']),
 ('stateless_hidden_state',KM,'_require_stateless','if','hidden_state is not None',['cache']),
 ('stateless_dones',KM,'_require_stateless_replay','if','dones is not None',['cache']),
 ('combined_type',KM,'evaluate_actions_with_teacher','if','not isinstance(teacher,',['combined']),
 ('combined_action_spec',KM,'evaluate_actions_with_teacher','if','teacher.action_spec != self.action_spec',['combined']),
 ('combined_grammar_signature',KM,'evaluate_actions_with_teacher','if','teacher.grammar_signature() != self.grammar_signature()',['combined']),
 ('combined_tables',KM,'evaluate_actions_with_teacher','if','not torch.equal(teacher_tables[name], table)',['combined']),
 ('teacher_replay',KM,'compute_teacher_distillation_targets','expr','check_replay_flags(result.valid)',['replay']),
 ('orbit_foreign_targets',SV,'evaluate_actions_with_cached_teacher','if','not isinstance(teacher_targets',['orbit']),
 ('obs_kaggriculture_equality',RUN,'_teacher_obs_spec_for_student','if','teacher_obs_spec != student_obs_spec',['obs']),
 ('obs_cross_game',RUN,'_teacher_obs_spec_for_student','if','not isinstance(teacher_obs_spec, EntityBasedBaseConfig)',['obs']),
 ('wrapper_cached_stateless',PPO,'_model_evaluate_actions_with_cached_teacher','if','hidden_state is None',['wrapper']),
 ('wrapper_combined_stateless',PPO,'_model_evaluate_actions_with_teacher','if','hidden_state is None',['wrapper']),
]
code,output=run('baseline',list(T))
print('baseline:',code, output[-1200:],flush=True)
assert code==0,'Baseline failed; mutations not run'
summary=json.loads((OUT/'summary.json').read_text()) if (OUT/'summary.json').exists() else []
try:
    for label,path,func,kind,needle,tests in mutants:
        if label in {r['mutant'] for r in summary}:
            continue
        old=change_node(path,func,kind,needle,'False' if kind=='if' else 'pass')
        try:
            code,output=run(label,tests)
            row={'mutant':label,'file':path,'function':func,'replaced':old,'replacement':'False' if kind=='if' else 'pass','tests':[T[t] for t in tests],'exit_code':code,'result':'SURVIVED' if code==0 else 'KILLED' if code==1 and 'FAILED ' in output else 'SETUP_ERROR','tail':output[-1800:]}
            summary.append(row)
            print(label,row['result'],output.splitlines()[-1],flush=True)
            (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
        finally:
            (SCRATCH/path).write_bytes(originals[path])
            assert (SCRATCH/path).read_bytes()==originals[path]
finally:
    for path,data in originals.items():
        (SCRATCH/path).write_bytes(data)
    hashes={path:{'source':hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),'restored_scratch':hashlib.sha256((SCRATCH/path).read_bytes()).hexdigest()} for path in originals}
    assert all(x['source']==x['restored_scratch'] for x in hashes.values())
    (OUT/'restoration.json').write_text(json.dumps(hashes,indent=2))
print('Completed',len(summary),'mutations; source/restored hashes equal',flush=True)
