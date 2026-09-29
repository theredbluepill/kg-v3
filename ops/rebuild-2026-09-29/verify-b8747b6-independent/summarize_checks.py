from pathlib import Path
from collections import Counter
import hashlib,json,subprocess,xml.etree.ElementTree as ET
root=Path.cwd();out=root/'ops/rebuild-2026-09-29/verify-b8747b6-independent'
results=json.loads((out/'shard-results.json').read_text())
selected=[r for r in results if not r['stop_reason']]
assert all(r['exit_status']==0 for r in selected)
nodes={};skips=[];counts=Counter()
for r in selected:
 for c in ET.parse(out/(r['name']+'.xml')).getroot().iter('testcase'):
  key=(c.attrib['classname'],c.attrib['name'])
  assert key not in nodes,key
  nodes[key]=r['name']
  category='failed' if c.find('failure') is not None else 'error' if c.find('error') is not None else 'skipped' if c.find('skipped') is not None else 'passed'
  counts[category]+=1
  if category=='skipped':skips.append({'class':key[0],'name':key[1],'reason':c.find('skipped').attrib['message']})
summary={'head':subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),'counts':dict(counts),'unique_cases':len(nodes),'source_test_files':45,'successful_invocations':len(selected),'guarded_attempts':[{'name':r['name'],'reason':r['stop_reason'],'rss':r['sampled_process_group_peak_rss_bytes']} for r in results if r['stop_reason']],'skips':skips,'unique_case_ids':['::'.join(k) for k in nodes]}
(out/'python-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
before=json.loads((out/'tracked-before.json').read_text())
changed=[p for p,h in before.items() if not (root/p).is_file() or hashlib.sha256((root/p).read_bytes()).hexdigest()!=h]
status=subprocess.check_output(['git','status','--short','--untracked-files=no']).decode()
restoration={'head':summary['head'],'tracked_file_count':len(before),'changed_tracked_files':changed,'tracked_status':status}
(out/'tracked-after.json').write_text(json.dumps(restoration,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='unique_case_ids'},indent=2))
print(json.dumps(restoration,indent=2))
assert not changed and not status
