from pathlib import Path
import copy, hashlib, importlib.util, json, shutil
import numpy as np
from owl import rs

root=Path.cwd()
out=root/'ops/rebuild-2026-09-29/verify-e197528-independent'
scratch=root/'.codex-tmp/verify-e197528-boundary'
scratch.mkdir(exist_ok=True)
source=root/'tests/kaggriculture/test_observe.py'
target=scratch/'test_observe.py'
shutil.copyfile(source,target)
original=target.read_bytes()
spec=importlib.util.spec_from_file_location('boundary_helpers',target)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
results=[]
for name in ('shape','dtype','endian','stride','readonly','alignment','alias','late_header'):
 batch=mod._allocate(2)
 arrays=mod._arrays(batch)
 headers=[mod._header(),mod._header()]
 payload=json.dumps(headers)
 mod._write(payload,arrays)
 batch.check_contract()
 valid_arrays=arrays.copy()
 before={key:value.tobytes() for key,value in arrays.items()}
 if name=='shape': arrays['actor_inventory']=arrays['actor_inventory'].reshape(-1)
 elif name=='dtype': arrays['actor_slot']=arrays['actor_slot'].astype(np.uint64)
 elif name=='endian': arrays['player_features']=arrays['player_features'].astype(arrays['player_features'].dtype.newbyteorder('S'))
 elif name=='stride':
  value=arrays['storage_counts']; backing=np.empty((*value.shape[:-1],value.shape[-1]*2),dtype=value.dtype)
  arrays['storage_counts']=backing[...,::2]; arrays['storage_counts'][...]=value
 elif name=='readonly': arrays['order_limits'].flags.writeable=False
 elif name=='alignment':
  value=arrays['banks']; backing=np.zeros(value.nbytes+1,dtype=np.uint8)
  arrays['banks']=np.ndarray(value.shape,dtype=value.dtype,buffer=backing,offset=1); arrays['banks'][...]=value
 elif name=='alias': arrays['storage_rank']=batch.actor_inventory.reshape(-1)[:48].reshape(2,2,12).numpy()
 elif name=='late_header':
  bad=copy.deepcopy(headers); bad[1]['initial']['privates'][1]['inventories']=[]; payload=json.dumps(bad)
 mutated_before={key:value.tobytes() for key,value in arrays.items()}
 pointers={key:value.ctypes.data for key,value in arrays.items()}
 try: mod._write(payload,arrays)
 except ValueError as error: caught=str(error)
 else: raise AssertionError(f'{name}: guard did not reject')
 assert mutated_before=={key:value.tobytes() for key,value in arrays.items()},name
 assert pointers=={key:value.ctypes.data for key,value in arrays.items()},name
 # A fresh NumPy view restores writability of the original Torch storage.
 arrays=mod._arrays(batch)
 assert before=={key:value.tobytes() for key,value in arrays.items()},name
 mod._write(json.dumps(headers),arrays); batch.check_contract()
 assert before=={key:value.tobytes() for key,value in arrays.items()},name
 results.append({'mutation':name,'rejected':caught,'all_bytes_and_pointers_unchanged':True,'restored_baseline_matches':True})
assert target.read_bytes()==original==source.read_bytes()
report={'binding':rs.__file__,'copied_test_sha256':hashlib.sha256(original).hexdigest(),'scratch_bytes_restored':True,'probes':results}
(out/'boundary-probes.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
