from pathlib import Path
import hashlib,importlib.util,json,os,tempfile
ROOT=Path('/Users/poonszesen/kg-v3-env'); SCRATCH=Path('/private/tmp/kg-verify-env-oracle-20260929'); OUT=ROOT/'ops/rebuild-2026-09-29/codex/verify-env-independent/oracle'; SOURCE=SCRATCH/'scripts/record_kaggriculture_env_reference.py';BASE=SOURCE.read_bytes();text=BASE.decode()
def load(name):
    spec=importlib.util.spec_from_file_location(name,SOURCE); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
results=[]
try:
    for changed in (False,True):
        code=text.replace('(args.reference, args.games, args.first_seed, args.max_live_envs)\n        == (REFERENCE, GAMES, 17000, 1)','True') if changed else text
        SOURCE.write_text(code); m=load(f'cli_{changed}')
        original_argv=m.sys.argv
        m.sys.argv=['recorder','--games','2']
        m.supervise=lambda *_: (_ for _ in ()).throw(RuntimeError('reached supervised work'))
        try: m.main()
        except Exception as exc: outcome=f'{type(exc).__name__}: {exc}'
        finally: m.sys.argv=original_argv
        assert ('RuntimeError' if changed else 'ValueError') in outcome
        results.append(dict(guard='fixed-recipe',mutant=changed,outcome=outcome))
    for changed in (False,True):
        code=text.replace('group_rss(process.pid) + resident_bytes(os.getpid())','group_rss(process.pid)') if changed else text
        SOURCE.write_text(code);m=load(f'rss_{changed}')
        class Process:
            pid=12345
            stopped=False
            def poll(self): return -9 if self.stopped else None
            def wait(self): self.stopped=True;return -9
        process=Process();old_popen=m.subprocess.Popen;old_monotonic=m.time.monotonic;old_killpg=m.os.killpg
        m.subprocess.Popen=lambda *_a,**_k:process
        m.group_rss=lambda _:700_000;m.resident_bytes=lambda _:400_000
        ticks=iter([0.0,1.0,2.0]);m.time.monotonic=lambda:next(ticks)
        m.os.killpg=lambda *_:None
        with tempfile.TemporaryDirectory(dir=SCRATCH) as folder:
            m.OPS=Path(folder)
            try: m.supervise(['never-executed'],0.1,1)
            except SystemExit as exc: outcome=str(exc)
            finally: m.subprocess.Popen=old_popen;m.time.monotonic=old_monotonic;m.os.killpg=old_killpg
            receipt=json.loads((m.OPS/'reference-recording-attempt.json').read_text())
        assert receipt['stop_reason']==('wall_time' if changed else 'memory')
        results.append(dict(guard='supervisor-and-children-rss',mutant=changed,outcome=outcome,receipt=receipt))
finally:
    SOURCE.write_bytes(BASE);assert SOURCE.read_bytes()==BASE
(OUT/'supervision-probes.json').write_text(json.dumps({'baseline_sha256':hashlib.sha256(BASE).hexdigest(),'restored_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'probes':results},indent=2)+'\n')
print('2 guards: baseline rejects intended fault; both omissions independently detected')
