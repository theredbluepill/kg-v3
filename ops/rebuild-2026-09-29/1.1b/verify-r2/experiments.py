import copy
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT = Path.cwd()
OUT = ROOT / '.codex-tmp/verify-1.1b-r2'
def load(name):
    spec = importlib.util.spec_from_file_location('r2_' + name, ROOT / ('scripts/kaggriculture_parity/' + name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
g = load('generate_traces')
s = load('sweep')
def replay(directory, label):
    report = OUT / (label + '-report.json')
    env = {**os.environ, 'TMPDIR': str(ROOT / '.codex-tmp'), 'UV_OFFLINE':'1', 'KAGG_PARITY_TRACES': str(directory), 'KAGG_PARITY_REPORT':str(report)}
    cmd = ['cargo', 'test', '--manifest-path','engine_rs/Cargo.toml','--locked','--offline','--test','replay_parity','env_directory_traces','--','--exact','--nocapture']
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    (OUT / (label + '.log')).write_text(r.stdout+r.stderr)
    return r.returncode, json.loads(report.read_text())
# A copy preserves every tracked byte while testing the exact same replay function.
original = ROOT / 'engine_rs/fixtures/generated/gen-edge-vs-edge.jsonl.gz'
original_bytes = original.read_bytes()
rows = [json.loads(x) for x in gzip.decompress(original_bytes).decode().splitlines()]
index = next(i for i,r in enumerate(rows) if r.get('type') == 'transition' and r.get('from_step') == 100)
before = rows[index]['expected']['farms'][0]['money']
rows[index]['expected']['farms'][0]['money'] = before + 1.0
pdir = OUT / 'perturb'
pdir.mkdir(exist_ok=True)
fixture = pdir / original.name
fixture.write_bytes(g.encode_trace(rows))
try:
    rc_bad, report_bad = replay(pdir,'perturbed')
    assert rc_bad == 101 and report_bad[0]['divergence']['field'] == 'public.farms[0].money'
finally:
    fixture.write_bytes(original_bytes)
rc_good, report_good = replay(pdir,'restored')
assert rc_good == 0 and report_good[0]['ok']
assert fixture.read_bytes() == original.read_bytes() == original_bytes
summary = {'fixture': str(original), 'line':index, 'from_step':100, 'before':before, 'after':before+1, 'failure_exit':rc_bad, 'failure_report':report_bad, 'restored_exit':rc_good, 'restored_sha256':hashlib.sha256(original_bytes).hexdigest(), 'restored_report':report_good}
(OUT/'perturb-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print('PERTURB',json.dumps(summary))
# Reproduce every minimal committed divergence through the ordinary external path.
ddir = OUT / 'divergences'
ddir.mkdir(exist_ok=True)
for path in (ROOT/'engine_rs/fixtures/generated').glob('divergence-*.jsonl.gz'):
    (ddir/path.name).write_bytes(path.read_bytes())
rc_div, report_div = replay(ddir,'divergences')
assert rc_div == 101 and len(report_div)==7
assert all(not r['ok'] and r['divergence']['line']==1 and r['divergence']['from_step']==0 for r in report_div)
print('DIVERGENCES',json.dumps(report_div))
# Repeat r1's unrelated state corruption on the actual D1 market fixture.
cdir = OUT/'classification-corrupt'
cdir.mkdir(exist_ok=True)
name='divergence-d1-unicode-digit-market-quantity.jsonl.gz'
rows=[json.loads(x) for x in gzip.decompress((ddir/name).read_bytes()).decode().splitlines()]
rows[1]['expected']['day'] += 1
(cdir/name).write_bytes(g.encode_trace(rows))
rc_corrupt, report_corrupt = replay(cdir,'classification-corrupt')
adir = OUT/'classification-ascii'
assert s.write_d1_rechecks(report_corrupt,cdir,adir) == [name]
rc_ascii, report_ascii = replay(adir,'classification-ascii')
entry=next(e for e in json.loads((ROOT/'engine_rs/fixtures/generated/MANIFEST.json').read_text())['traces'] if e['path']==name)
classified=s.summarize([entry],report_corrupt,cdir,report_ascii)
assert classified['divergences_by_class']=={'unclassified':1} and classified['new_divergences']==1
assert rc_corrupt==rc_ascii==101
(OUT/'classification-summary.json').write_text(json.dumps(classified,indent=2)+'\n')
print('CLASSIFICATION',json.dumps(classified))
# Probe labels must equal the submitted action (including JSON null).
probes=sorted((OUT/'sweep-traces').glob('probe-*.jsonl.gz'))
for path in probes:
    rows=[json.loads(x) for x in gzip.decompress(path.read_bytes()).decode().splitlines()]
    submitted=next(r for r in rows[1:] if r['from_step']==2)['actions'][0]
    assert json.loads(rows[0]['source']['probe'])==submitted, path
null_path=OUT/'sweep-traces/probe-296-action.jsonl.gz'
rows=[json.loads(x) for x in gzip.decompress(null_path.read_bytes()).decode().splitlines()]
assert next(r for r in rows[1:] if r['from_step']==2)['actions'][0] is None
print('PROBE LABELS',len(probes),'matched; probe-296 submitted null')
