"""Reproduce the extra rollback-error finding on two real native games."""
import hashlib,json,os,stat,tempfile
from pathlib import Path
from unittest.mock import patch
from owl.kaggriculture import replay_export
from tests.kaggriculture.test_replay_export_integration import _run,_custody

ROOT=Path(__file__).resolve().parent
OUT=Path(tempfile.mkdtemp(prefix='kg-r4-double-fault-',dir='/tmp'))
original=OSError('r4 directory fsync failed')
cleanup=PermissionError('r4 rollback custody unlink failed')
real_fsync=os.fsync; real_unlink=Path.unlink
state={'fsync_failed':False,'cleanup_failed':False}
def fsync(fd):
    if stat.S_ISDIR(os.fstat(fd).st_mode) and not state['fsync_failed']:
        state['fsync_failed']=True
        raise original
    return real_fsync(fd)
def unlink(path,*args,**kwargs):
    if path==OUT/'game_000000.custody.json' and state['fsync_failed'] and not state['cleanup_failed']:
        state['cleanup_failed']=True
        raise cleanup
    return real_unlink(path,*args,**kwargs)
try:
    with patch.object(replay_export.os,'fsync',fsync),patch.object(Path,'unlink',unlink):
        _run(OUT,n_envs=2,n_games=2,count=2,configuration={'episodeSteps':3})
except Exception as error:
    assert error is cleanup
    assert error.__context__ is original
    error_text=repr(error)
    notes=error.__notes__
else:
    raise AssertionError('expected double failure')
records=_custody(OUT)
payload=(OUT/'game_000000.json').read_bytes()
assert records[0]['status']=='complete'
assert records[0]['episode_sha256']==hashlib.sha256(payload).hexdigest()
assert records[1]['status']=='error'
verification=json.loads(replay_export.rs.verify_kaggriculture_episode(payload.decode(),None))
assert verification['ok'] is True
receipt={'output':str(OUT),'original':repr(original),'observed':error_text,'original_preserved_as_primary':False,'original_preserved_as_context':True,'notes':notes,'files':sorted(p.name for p in OUT.iterdir()),'statuses':{i:r['status'] for i,r in records.items()},'game_0_episode_hash_matches':True,'game_0_episode_byte_verified':True,'injection_state':state,'source_sha256':hashlib.sha256(Path(replay_export.__file__).read_bytes()).hexdigest()}
(ROOT/'double-fault-live-result.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
