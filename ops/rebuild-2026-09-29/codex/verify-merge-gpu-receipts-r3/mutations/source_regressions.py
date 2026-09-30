"""Fresh scratch source mutations; no tracked or GPU changes."""
import ast, hashlib, importlib.util, json, os, shutil, sys
from pathlib import Path
ROOT=Path('/Users/poonszesen/kg-v3-m-gpu-receipts')
SCRATCH=Path('/private/tmp/gpu-r3-mut-z43e6hsa')
B=SCRATCH/'regression-bundle'
S=B/'scripts'
E=ROOT/'ops/rebuild-2026-09-29/codex/verify-merge-gpu-receipts-r3/mutations'
sys.path.insert(0,str(S))
os.environ['GPUCHK_DRYRUN']='1'
os.environ['GPUCHK_ROOT']=str(ROOT)
originals={p:p.read_bytes() for p in B.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
results=[]
def load(name,p):
    shutil.rmtree(S/'__pycache__',ignore_errors=True)
    spec=importlib.util.spec_from_file_location(name,p)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

def record(name,killed,detail):
    r={'mutation':name,'killed':bool(killed),'detail':detail}
    results.append(r);print(json.dumps(r),flush=True)

try:
    p=S/'c1_fp32_ref.py';old=p.read_bytes();text=old.decode()
    assert text.count('if metric != "max_abs":')==1
    p.write_text(text.replace('if metric != "max_abs":','if med and metric != "max_abs":'))
    G=load('guards_c1',S/'test_driver_guards.py')
    r=G.check_c1_verdict();record('restore_C1_zero_median_bypass',not r['pass'],r)
    p.write_bytes(old)

    p=S/'driver.py';old=p.read_bytes();text=old.decode()
    start=text.index('    vals += [(k, r[k]["student_values"])')
    end=text.index('    for k, v in vals:',start)
    replacement='    vals += [(k, r[k]["student_values"]) for k in ("teacher_self_combined", "teacher_perturbed_cached")]\n'
    p.write_text(text[:start]+replacement+text[end:])
    G=load('guards_c3',S/'test_driver_guards.py')
    r=G.check_c3_values();record('restore_C3_two_path_omission',not r['pass'],r)
    p.write_bytes(old)

    module=ast.parse(text)
    klass=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='Driver')
    stream=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='stream')
    lines=text.splitlines(keepends=True)
    replacement='    def stream(self, stages: list[Stage], gpu: int) -> None:\n        for stage in stages:\n            self.run_stage(stage, gpu)\n'
    p.write_text(''.join(lines[:stream.lineno-1])+replacement+''.join(lines[stream.end_lineno:]))
    # The fake-stage harness avoids hanging a deliberately regressed driver.
    harness=(E/'driver_mutations.py').read_text().replace(str(B.parent/'bundle/scripts/driver.py'),str(p))
    harness=harness.replace("Path('/private/tmp/gpu-r3-mut-z43e6hsa/driver-mutations.json')", "Path('/private/tmp/gpu-r3-mut-z43e6hsa/regressed-driver-mutations.json')")
    namespace={'__name__':'mutation_driver_harness'}
    shutil.rmtree(S/'__pycache__',ignore_errors=True)
    exec(compile(harness,'mutation_driver_harness','exec'),namespace)
    faults=[x for x in namespace['results'] if x['probe']!='valid_baseline']
    record('remove_driver_stream_exception_boundary',all(not r['guard_rejected'] and r['rc']==0 and 'c4_after' in r['launched'] for r in faults),faults)
    p.write_bytes(old)

    assert text.count('self.defer_signals = True')==1
    p.write_text(text.replace('self.defer_signals = True','self.defer_signals = False'))
    C=load('cleanup_mutant',S/'test_driver_cleanup.py')
    def alive(pid):
        try:os.kill(pid,0);return True
        except ProcessLookupError:return False
    C.alive=alive
    C.survivors=lambda token: []
    for scenario in ('new_spawn_after_popen','new_spawn_in_popen'):
        r=C.run_scenario(scenario,p)
        record('disable_spawn_signal_deferral:'+scenario,bool(r['stage_pids_alive']) and not r['boundary']['registered_at_signal'],r)
    p.write_bytes(old)
finally:
    for p,content in originals.items():
        p.write_bytes(content)
    mismatches=[str(p) for p,content in originals.items() if p.read_bytes()!=content]
    restoration={'files':len(originals),'sha256':{str(p.relative_to(B)):hashlib.sha256(p.read_bytes()).hexdigest() for p in originals},'mismatches':mismatches}
    (E/'source-regression-restoration.json').write_text(json.dumps(restoration,indent=2)+'\n')
    (E/'source-regressions.json').write_text(json.dumps(results,indent=2)+'\n')
    print('RESTORED',len(originals),'byte mismatches',len(mismatches),flush=True)
assert all(r['killed'] for r in results)
assert not mismatches
