import hashlib
import json
from pathlib import Path
import subprocess
import sys

root=Path.cwd()
out=root/'ops/rebuild-2026-09-29/1.3/independent-7b5eacd'
source=root/'src/kaggriculture/observe.rs'
original=source.read_bytes()
marker=b'fn write_shops_and_market('
start=original.index(marker)
old=b'[&public.market.inventory, &public.market.prices]'
new=b'[&public.market.prices, &public.market.inventory]'
prefix,body=original[:start],original[start:]
assert body.count(old)==1
mutated=prefix+body.replace(old,new,1)
receipt={'source':str(source.relative_to(root)),'mutation':'Swap market inventory and price sources in production writer only','before_sha256':hashlib.sha256(original).hexdigest(),'mutant_sha256':hashlib.sha256(mutated).hexdigest()}
try:
    source.write_bytes(mutated)
    cmd=[sys.executable,'ops/rebuild-2026-09-29/1.3/bounded.py','--name','independent-7b5eacd/mutant','--seconds','120','--','cargo','test','--locked','kaggriculture::oracle_corpus::compare_observation_oracle','--','--exact','--nocapture']
    result=subprocess.run(cmd)
    receipt['mutation_run_returncode']=result.returncode
finally:
    source.write_bytes(original)
    receipt['restored_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    receipt['restored_byte_for_byte']=source.read_bytes()==original
    (out/'mutation.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
assert receipt['restored_byte_for_byte']
assert receipt['mutation_run_returncode']==101
assert 'offset=889' in (out/'mutant.log').read_text()
