from pathlib import Path
import difflib,json,os,subprocess
OUT=Path('/private/tmp/kg-merge84-contracts')
SCRATCH=OUT/'scratch'
grammar=SCRATCH/'src/kaggriculture/grammar.rs'
original=grammar.read_bytes()
before='token != 1 || self.shape.actors + u16::from(self.hires) < self.shape.hire_limit'
assert original.decode().count(before)==1
mutated=original.decode().replace(before,'true')
command=['cargo','test','--manifest-path',str(SCRATCH/'engine_rs/Cargo.toml'),'--locked','--offline','--test','grammar_kernel']
results=[]
try:
 for label,source in [('baseline_grammar',original.decode()),('task12_hire_capacity',mutated)]:
  grammar.write_text(source)
  if label!='baseline_grammar':
   (OUT/f'{label}.diff').write_text(''.join(difflib.unified_diff(original.decode().splitlines(True),source.splitlines(True),fromfile='merged',tofile=label)))
  proc=subprocess.run(command,cwd=SCRATCH,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
  (OUT/f'{label}.log').write_text(proc.stdout)
  lines=[line for line in proc.stdout.splitlines() if 'FAILED' in line or line.startswith('test result') or line.startswith('error')]
  print(label,'exit',proc.returncode,*lines,sep='\n',flush=True)
  results.append({'mutation':label,'exit':proc.returncode,'summary':lines})
  assert proc.returncode==(0 if label=='baseline_grammar' else 101),(label,proc.returncode)
finally:
 grammar.write_bytes(original)
 assert grammar.read_bytes()==original
 (OUT/'grammar-results.json').write_text(json.dumps(results,indent=2)+'\n')
