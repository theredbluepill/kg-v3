"""Attempt downstream integration mutations; classify preserved first failures as blocked."""
import ast, hashlib, json, os, subprocess
from pathlib import Path
ROOT=Path.cwd()
OUT=ROOT/'ops/rebuild-2026-09-29/codex/verify-merge-teacher-r2/numerical-mutations'
SCRATCH=Path(json.loads((OUT/'scratch-receipt.json').read_text())['scratch'])
PY=ROOT/'.venv/bin/python'
ENV={**os.environ,'PYTHONPATH':str(SCRATCH/'python')+os.pathsep+str(SCRATCH),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
TARGET='tests/kaggriculture/test_teacher.py'
PPO='python/owl/train/ppo.py'; RUN='scripts/run_ppo.py'
M=[
('blocked-trainer-cache-metric-zero',PPO,'float(teacher_targets.nbytes()) if teacher_targets is not None else 0.0','0.0','test_trainer_precomputes_once_and_logs_teacher_metrics','unsupported type KaggricultureActionMask'),
('blocked-checkpoint-includes-teacher-cache',PPO,'"model": checkpoint_model.state_dict(),','"model": checkpoint_model.state_dict(),\n            "teacher_cache": torch.zeros(1),','test_trainer_checkpoint_after_a_teacher_iteration_holds_no_teacher_cache','unsupported type KaggricultureActionMask'),
('blocked-resume-activates-student',RUN,'if cfg.rl.teacher_mode == "last_best":\n                trainer.set_teacher_model(last_best_model, active=True)','if cfg.rl.teacher_mode == "last_best":\n                trainer.set_teacher_model(model, active=True)','test_run_ppo_resume_restores_the_teacher_from_checkpoint_last_best','run_ppo cannot run Kaggriculture yet'),
('blocked-fresh-launch-disables-teacher',RUN,'):\n            trainer.set_teacher_model(last_best_model, active=True)','):\n            trainer.set_teacher_model(last_best_model, active=False)','test_run_ppo_fresh_launch_from_weights_activates_the_last_best_teacher','run_ppo cannot run Kaggriculture yet'),
]
results=[]
test_path=SCRATCH/TARGET; test_original=test_path.read_bytes()
try:
    unskipped=test_original.decode().replace('@pytest.mark.skip(reason=NEEDS_TRAINER_SEAM)\n','').replace('@pytest.mark.skip(reason=NEEDS_RUN_PPO_GAME_SEAM)\n','')
    assert unskipped!=test_original.decode()
    test_path.write_text(unskipped)
    for name,file,old,new,test_name,blocker in M:
        path=SCRATCH/file; original=path.read_bytes(); body=original.decode()
        assert body.count(old)==1,(name,body.count(old))
        mutated=body.replace(old,new); ast.parse(mutated)
        try:
            path.write_text(mutated)
            with (OUT/(name+'.log')).open('w') as f:
                p=subprocess.run([str(PY),'-m','pytest',TARGET+'::'+test_name,'-q'],cwd=SCRATCH,env=ENV,stdout=f,stderr=subprocess.STDOUT,timeout=150)
            output=(OUT/(name+'.log')).read_text()
        finally:
            path.write_bytes(original)
        r={'name':name,'file':file,'test':TARGET+'::'+test_name,'exit_code':p.returncode,'classification':'BLOCKED' if p.returncode==1 and blocker in output else 'UNEXPECTED','expected_first_blocker':blocker,'summary':output.splitlines()[-1],'restored':path.read_bytes()==original,'sha256':hashlib.sha256(original).hexdigest()}
        results.append(r); print(json.dumps(r),flush=True)
        (OUT/'blocked-oracle-results.json').write_text(json.dumps(results,indent=2)+'\n')
finally:
    test_path.write_bytes(test_original)
copied=json.loads((OUT/'copied-before.json').read_text())
after={rel:hashlib.sha256((SCRATCH/rel).read_bytes()).hexdigest() for rel in copied}
(OUT/'final-full-copy-restoration.json').write_text(json.dumps({'copied_files':len(copied),'sha256_before':copied,'sha256_after':after,'mismatched':[k for k in copied if copied[k]!=after[k]],'test_skips_restored':test_path.read_bytes()==test_original},indent=2)+'\n')
assert all(r['classification']=='BLOCKED' and r['restored'] for r in results)
assert copied==after
