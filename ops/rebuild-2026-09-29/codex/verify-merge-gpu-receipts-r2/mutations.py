import ast, copy, hashlib, importlib.util, json, math, os, subprocess, sys, types
from pathlib import Path
from unittest.mock import patch

ROOT=Path('/Users/poonszesen/kg-v3-m-gpu-receipts')
SRC=ROOT/'ops/rebuild-2026-09-29/gpu-checks-2026-09-29'
B=Path('/private/tmp/kg-gpu-verify-r2-qmsanrkg/bundle')
S=B/'scripts'
os.environ['GPUCHK_ROOT']=str(ROOT)
os.environ['GPUCHK_DRYRUN']='1'
sys.path.insert(0,str(S))
def mod(name,path):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
D=mod('driver',S/'driver.py'); C=mod('common',S/'common.py'); sys.modules['common']=C
P1=mod('c1',S/'c1_fp32_ref.py');P2=mod('c2',S/'c2_trunk_bwd.py');P3=mod('c3',S/'c3_smoke.py');T=mod('templates',S/'check_templates.py');SUM=mod('summarize',B/'summarize.py')
import torch
results=[]
def add(name,passed,detail=None):
    results.append({'probe':name,'passed':bool(passed),'detail':detail}); print(json.dumps(results[-1]),flush=True)
def ev(case): return D.events(B/'pod/attempt2'/f'{case}.jsonl')
def record(case,event='result'): return [r for r in ev(case) if r.get('event')==event][0]
def mutate_case(case,judge,path,value):
    data=copy.deepcopy(ev(case)); rec=next(r for r in data if r.get('event')==('timing' if case.startswith('c4') else 'result'))
    dest=rec
    for k in path[:-1]: dest=dest[k]
    dest[path[-1]]=value
    out=judge(data);add(case+':'+'.'.join(map(str,path)),bool(out),out)
# Actual inputs are good, fixed receipt corruption should be rejected.
for case,judge in [('c1_aten',D.judge_c1),('c1_default',D.judge_c1),('c2_aten_bwd',D.judge_c2),('c3_mid_aten',D.judge_c3),('c3_dense_aten',D.judge_c3),('c3_mid_default',D.judge_c3),('c3_dense_default',D.judge_c3),('c4_mid_aten',D.judge_c4),('c4_dense_aten',D.judge_c4)]:
    add('baseline:'+case,not judge(ev(case)),judge(ev(case)))
for key in ('c1','c2','c3','c4'):
    add(key+':empty',bool(D.JUDGES[key]([])))
for path,val in [(('ref_nonfinite',),1)]+[(("vs_fp32",p,"nonfinite"),1) for p in ('compiled','eager','padded')]+[((p+'_pack_calls',),[]) for p in ('compiled','eager')]+[((p+'_use_flash',),[False]) for p in ('compiled','eager')]: mutate_case('c1_aten',D.judge_c1,path,val)
for path,val in [(('compiled','status'),'error'),(('eager','status'),'error'),(('compiled','trunk_call_M'),[]),(('out_compiled_vs_eager','wrong_tokens'),1),(('out_compiled_vs_eager','nonfinite'),1)]+[(('dx_compiled_vs_eager',k),v) for k,v in [('nonfinite',1),('tokens_rel_gt_0.5',1),('masked_nonzero',1),('rel_max',None),('rel_max',0.051)]]+[(('params_compiled_vs_eager','_missing'),['lost'])]+[(('params_compiled_vs_eager','blocks.0.attn.k.bias',k),v) for k,v in [('nonfinite',1),('max_abs_diff',1e8),('out_max_abs',1e8),('ref_max_abs',1e8)]]+[(('params_compiled_vs_eager','blocks.0.attn.q.bias',k),v) for k,v in [('nonfinite',1),('rel_max',0.051)]]:mutate_case('c2_aten_bwd',D.judge_c2,path,val)
for path,val in [(('error',),'Injected error'),(('steps',),{}),(('sample','log_probs','nonfinite'),1)]+[((p,'logratio_per_row',k),v) for p in ('replay_256','replay_1024') for k,v in [('nonfinite',1),('mean',0.051),('mean',-0.051)]]+[((p,'new_logp_nonfinite'),1) for p in ('replay_256','replay_1024')]+[((p,'kl_nonfinite'),1) for p in ('teacher_self_combined','teacher_self_cached','teacher_perturbed_combined','teacher_perturbed_cached')]+[((p,'kl_per_row',k),v) for p in ('teacher_self_combined','teacher_self_cached') for k,v in [('mean',0.0011),('max',0.0101)]]+[((p,'kl_per_row','mean'),0) for p in ('teacher_perturbed_combined','teacher_perturbed_cached')]+[((p,'kl_event_min'),-0.00011) for p in ('teacher_self_combined','teacher_self_cached','teacher_perturbed_combined','teacher_perturbed_cached')]+[(('loss_backward_1024','grad_nonfinite'),1),(('loss_backward_1024','loss'),float('nan')),(('loss_backward_1024','loss'),float('inf')),(('use_flash_attn_all_true',),False),(('compiled_trunk_calls',),0)]:mutate_case('c3_mid_aten',D.judge_c3,path,val)
for p,sub in [('sample','values'),('replay_256','values'),('replay_1024','values'),('compute_value_256','values'),('teacher_self_combined','student_values'),('teacher_self_cached','student_values'),('teacher_perturbed_combined','student_values'),('teacher_perturbed_cached','student_values')]:
    for k,v in [('nonfinite',1),('min',-1.01),('max',1.01)]:mutate_case('c3_mid_aten',D.judge_c3,(p,sub,k),v)
for path,val in [(('use_flash_attn_all_true',),False),(('pack_calls_per_iter',),999)]:mutate_case('c4_mid_aten',D.judge_c4,path,val)
xs=ev('c4_mid_aten');next(r for r in xs if '_B_' in r.get('workload',''))['last_scalar_finite']=False;add('c4:B_loss',bool(D.judge_c4(xs)))
xs=[r for r in ev('c4_mid_aten') if r.get('event')!='derived'];add('c4:no_derived',bool(D.judge_c4(xs)))
# New numerical oracles, run on tiny CPU tensors.
r=torch.zeros(2,4,8);m=torch.tensor([[1,1,0,0],[1,0,0,0]],dtype=torch.bool)
x=r.clone();x[0,0,0]=0.1;z=P1.compare(x,r,m,retain=True);add('c1:comparison_outlier',z['outside_tol']==1 and z['elements']==24 and z['_outlier_coords']==[[0,0,0]])
x=r.clone();x[0,2,0]=float('nan');z=P1.compare(x,r,m,retain=False);add('c1:masked_nan_ignored',z['nonfinite']==0 and z['outside_tol']==0)
x=r.clone();x[0,0,0]=float('nan');z=P1.compare(x,r,m,retain=False);add('c1:valid_nan_seen',z['nonfinite']==1)
for metric in ('mean_abs','outside_tol_frac'):
    paths={p:{'mean_abs':1.,'outside_tol_frac':1.,'max_abs':1.} for p in ('compiled','eager','padded')};paths['compiled'][metric]=2.001
    add('c1:2x_median_'+metric,P1.verdict(paths)['result']=='path_specific')
add('c1:jaccard',P1.jaccard([[1,2,3]],[[1,2,3],[4,5,6]])['jaccard']==0.5)
x=r.clone();x[0,0,0]=0.26;z=P2.cmp_out(x,r,m);add('c2:output_wrong_token',z['wrong_tokens']==1 and z['first_bad_row']==0)
x=r.clone();x[0,0,0]=float('nan');z=P2.cmp_out(x,r,m);add('c2:output_nan',z['wrong_tokens']==1 and z['nonfinite']==1)
r2=torch.ones_like(r);r2[~m]=0;x=r2.clone();x[0,0]=3;z=P2.cmp_dx(x,r2,m);add('c2:dx_token_relative',z['tokens_rel_gt_0.5']==1 and z['rel_max']>0.05)
x=r2.clone();x[0,2,0]=1;z=P2.cmp_dx(x,r2,m);add('c2:dx_masked_nonzero',z['masked_nonzero']==1)
x=r2.clone();x[0,0,0]=float('nan');z=P2.cmp_dx(x,r2,m);add('c2:dx_nonfinite',z['nonfinite']==1)
z=P2.cmp_params({'a':torch.tensor([1.1])},{'a':torch.tensor([1.])});add('c2:param_relative',z['a']['rel_max']>0.05)
z=P2.cmp_params({'a':torch.tensor([float('nan')])},{'a':torch.tensor([1.])});add('c2:param_nonfinite',z['a']['nonfinite']==1)
z=P2.cmp_params({'a':torch.tensor([1.]),'extra':torch.tensor([1.])},{'a':torch.tensor([1.])});add('c2:param_missing',z['_missing']==['extra'])
try:P2.make_mask('bad:1',8);add('c2:invalid_mask',False)
except ValueError:add('c2:invalid_mask',True)
add('c2:packed_mask',P2.make_mask('packed:9',8).tolist()==[[True]*8,[True]+[False]*7])
add('c3:stats_nonfinite',P3.stats(torch.tensor([1.,float('nan')]))['nonfinite']==1)
# Template census: mutate previously empty generated wrapper to contain each marker.
cache=B/'scratch-cache';cache.mkdir(exist_ok=True);f=cache/'synthetic.py';f.write_text('def call():\n    pass\n');base=T.census(cache)
f.write_text('def call():\n    extern_kernels.addmm(a,b)\n    triton_tem_probe.run()\ndef triton_tem_probe():\n    pass\n# decompose_k persistent_tma contiguous_mm triton_mm_1\n')
z=T.census(cache);add('templates:extern',z['extern_mm_calls']=={'addmm':1});add('templates:template_defs_launches',z['triton_tem_defs']==z['triton_tem_launches']==1)
for k in T.MARKERS:add('templates:marker:'+k,len(z['marker_files'][k])==1)
f.unlink();cache.rmdir()
# Summary must exactly regenerate frozen artifact; changed format is rejected, changed input propagates.
original=(B/'summary.json').read_bytes();SUM.main();add('summary:byte_exact_regeneration',(B/'summary.json').read_bytes()==original);(B/'summary.json').write_bytes(original)
f=B/'pod/attempt2/c1_aten.outliers.json';old=f.read_bytes();data=json.loads(old);data['format']='invalid';f.write_text(json.dumps(data))
try:SUM.c1_channels();add('summary:format_guard',False)
except AssertionError:add('summary:format_guard',True)
finally:f.write_bytes(old)
f=B/'pod/attempt2/c4_mid_aten.derived.json';old=f.read_bytes();before=SUM.c4_scaling();data=json.loads(old);data['splits']['8']['global_sps']*=0.5;f.write_text(json.dumps(data));after=SUM.c4_scaling();add('summary:scaling_efficiency',after['mid/8']['efficiency']==before['mid/8']['efficiency']*0.5);f.write_bytes(old)
# Common guards against CPU/wrong config and compile results without any actual compilation.
C.DRYRUN=False
with patch.object(torch.cuda,'is_available',return_value=False):
    try:C.device();add('common:CUDA_required',False)
    except AssertionError:add('common:CUDA_required',True)
import owl.model.kaggriculture as km
import owl.model.attn as attn
import owl.train.utils as U
fake_model=types.SimpleNamespace(_compiled_transformer_trunk=None)
for count,compiled in [(0,object()),(1,None)]:
    fake_model._compiled_transformer_trunk=compiled
    with patch.object(U,'configure_model_compile',return_value=count):
        try:C.compile_trunk(fake_model);add('common:compile:'+str(count)+':'+str(compiled is not None),False)
        except RuntimeError:add('common:compile:'+str(count)+':'+str(compiled is not None),True)
with patch.object(km.KaggricultureTransformerConfig,'from_file',return_value=types.SimpleNamespace(force_flash_attn=False)):
    try:C.build_model(torch.device('cpu'));add('common:force_flash',False)
    except AssertionError:add('common:force_flash',True)
mods={'flash_attn':types.SimpleNamespace(__version__='stub'),'triton':types.SimpleNamespace(__version__='stub')}
for issue in ('wrong_tree','wrong_head','missing_flash'):
    with patch.dict(sys.modules,mods),patch.object(torch.cuda,'get_device_name',return_value='CPU STUB'),patch.object(km,'__file__','/wrong/tree/kaggriculture.py' if issue=='wrong_tree' else str(ROOT/'python/owl/model/kaggriculture.py')),patch.object(C,'git_head',return_value={'head':'bad' if issue=='wrong_head' else C.EXPECTED_HEAD,'tree':'stub','porcelain':''}),patch.object(attn,'flash_attn_available',return_value=issue!='missing_flash'):
        try:C.process_record('mutation');add('common:'+issue,False)
        except RuntimeError as e:add('common:'+issue,True,str(e))
# Wrapper refuses environmental contamination and malformed invocation (script is harmless).
wrap=S/'gemm_backend_wrap.py';dummy=B/'noop.py';dummy.write_text('pass\n')
for name,args,env in [('environment',['--backends','ATEN','--record',str(B/'backend-test.json'),'--',str(dummy)],{**os.environ,'TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS':'TRITON'}),('separator',[],os.environ),('backend_choice',['--backends','INVALID','--record',str(B/'backend-test.json'),'--',str(dummy)],os.environ)]:
    p=subprocess.run([sys.executable,str(wrap),*args],env=env,capture_output=True,text=True);add('wrapper:'+name,p.returncode!=0,p.stderr[-300:])
dummy.unlink()
# Recheck prior findings: all three must now be rejected.
gaps=[]
for p in ('teacher_self_cached','teacher_perturbed_combined'):
    xs=ev('c3_mid_aten');next(r for r in xs if r.get('event')=='result')[p]['student_values']['nonfinite']=1
    gaps.append({'probe':'c3_unchecked_'+p+'_values','survived':not bool(D.judge_c3(xs))})
paths={p:{'mean_abs':0.,'outside_tol_frac':0.,'max_abs':0.} for p in ('compiled','eager','padded')};paths['compiled']['mean_abs']=1.;gaps.append({'probe':'c1_zero_median','survived':P1.verdict(paths)['result']=='similar'})
print('GAPS',json.dumps(gaps),flush=True)
for g in gaps: add('r1_resolved:'+g['probe'],not g['survived'])
# No tracked byte modified; scratch files restored byte-for-byte.
source_files=[p for p in SRC.rglob('*') if '__pycache__' not in p.parts];bad=[]
for source in source_files:
    if source.is_file():
        other=B/source.relative_to(SRC)
        if source.read_bytes()!=other.read_bytes():bad.append(str(other))
add('scratch:restored_byte_exact',not bad,bad)
Path('/private/tmp/kg-gpu-verify-r2-qmsanrkg/mutations.json').write_text(json.dumps({'results':results,'gaps':gaps},indent=2))
print('SUMMARY',len(results),'passed',sum(r['passed'] for r in results),'failed',sum(not r['passed'] for r in results),flush=True)

assert all(r['passed'] for r in results), 'mutation failure'
