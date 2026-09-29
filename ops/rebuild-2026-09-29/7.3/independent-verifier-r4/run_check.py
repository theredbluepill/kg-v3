import json,os,pathlib,signal,subprocess,sys,time,resource
out=pathlib.Path(__file__).resolve().parent
name=sys.argv[1]; command=sys.argv[2:]
env=os.environ.copy(); env.update(UV_OFFLINE='1',CARGO_BUILD_JOBS='2',OMP_NUM_THREADS='2',WANDB_MODE='offline',PYTHONDONTWRITEBYTECODE='1')
guard=out/'runtime_guard'
env['PYTHONPATH']=str(guard)+os.pathsep+env.get('PYTHONPATH','')
start=time.monotonic(); peak=0; stopped=None
with (out/f'{name}.log').open('w') as log:
 p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
 try: code=p.wait(timeout=300)
 except subprocess.TimeoutExpired:
  stopped='300 second time limit'; os.killpg(p.pid,signal.SIGTERM)
  try: code=p.wait(timeout=10)
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGKILL); code=p.wait()
peak=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
if code == 86: stopped='Python RSS safety limit'
receipt={'command':command,'cwd':os.getcwd(),'environment':{k:env[k] for k in ('UV_OFFLINE','CARGO_BUILD_JOBS','OMP_NUM_THREADS','WANDB_MODE','PYTEST_ADDOPTS') if k in env},'exit_code':code,'wall_seconds':round(time.monotonic()-start,3),'peak_child_rss_bytes':peak,'memory_guard':'Python process peak RSS every 50 ms, stop at 950 MB; ps process-tree monitor denied by sandbox','stopped':stopped}
(out/f'{name}.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2)); print((out/f'{name}.log').read_text()[-8000:]); sys.exit(code if code >=0 else 1)
