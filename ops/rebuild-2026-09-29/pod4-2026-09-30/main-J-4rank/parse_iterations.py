import json,sys,math,collections
dec=json.JSONDecoder(); s=open(sys.argv[1],errors='replace').read(); T='[nt-probe] '
recs=[]; i=s.find(T)
while i>=0:
    try: o,e=dec.raw_decode(s,i+len(T)); recs.append(o)
    except Exception: pass
    i=s.find(T,i+1)
its=collections.defaultdict(dict)
for r in recs:
    if r['kind']=='iteration': its[r['iteration']][r['rank']]=r
print('ranks complete per iteration up to', max(n for n in its if len(its[n])==4))
keys=sys.argv[2].split(',') if len(sys.argv)>2 else []
for n in sorted(its):
    if 0 not in its[n]: continue
    r=its[n][0]; m=r['metrics']
    st=[its[n][k]['native_step_seconds']/max(its[n][k]['native_step_calls'],1)*1000 if k in its[n] else float('nan') for k in range(4)]
    line=f"{n:3d} w{r['wall_seconds']:.2f} os{m.get('optimizer/steps',0):.0f} lr{m.get('optimizer/learning_rate',0):.2e} R{m['time/rollout_seconds']:.2f} T{m['time/teacher_seconds']:.2f} U{m['time/update_seconds']:.2f} sps{m['perf/steps_per_second']:.0f} ms"+'/'.join(f'{x:.1f}' for x in st)
    for k in keys: 
        v=m.get(k); line+=f" {k.split('/')[-1]}={v:.4g}" if isinstance(v,(int,float)) else f" {k.split('/')[-1]}=-"
    print(line)
