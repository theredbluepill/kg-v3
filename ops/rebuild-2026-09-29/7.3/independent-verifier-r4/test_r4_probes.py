"""Independent fault probes against unchanged r4 publication source."""
import hashlib
import json
from pathlib import Path
import pytest
from owl.kaggriculture import replay_export as subject
from tests.kaggriculture import test_replay_export as unit
from tests.kaggriculture import test_replay_export_integration as live

OUT = Path(__file__).resolve().parent

class FaultStream:
    def __init__(self, stream, stage, fault):
        self.stream, self.stage, self.fault = stream, stage, fault
    def __enter__(self):
        self.stream.__enter__()
        return self
    def __exit__(self, *args):
        self.stream.__exit__(*args)
        if self.stage == 'close' and args[0] is None:
            raise self.fault
    def write(self, data):
        if self.stage == 'partial':
            self.stream.write(data[:20])
            self.stream.flush()
            raise self.fault
        return self.stream.write(data)
    def flush(self):
        return self.stream.flush()
    def fileno(self):
        return self.stream.fileno()

@pytest.mark.parametrize('games', [1, 2])
@pytest.mark.parametrize('target,stage', [('episode','open'), ('episode','partial'), ('custody','partial'), ('custody','close')])
def test_live_publication_fault(tmp_path, monkeypatch, games, target, stage):
    """Adapt r3 probes to dot-prefixed staged writes; retain real native paths."""
    output=tmp_path/'replays'
    fault=PermissionError('r4 episode open denied') if stage=='open' else OSError(f'r4 {target} {stage}')
    original=Path.open; injected=[]; recorders=[]
    recorder_factory=live._recorder
    def capture(*args, **kwargs):
        result=recorder_factory(*args, **kwargs); recorders.append(result); return result
    monkeypatch.setattr(live, '_recorder', capture)
    name='game_000000.json' if target=='episode' else 'game_000000.custody.json'
    def open_(path, mode='r', *args, **kwargs):
        matched=path.parent==output and path.name.startswith('.'+name+'.') and mode=='xb' and not injected
        if matched:
            injected.append(str(path))
            if stage=='open': raise fault
        stream=original(path,mode,*args,**kwargs)
        return FaultStream(stream,stage,fault) if matched else stream
    monkeypatch.setattr(Path,'open',open_)
    with pytest.raises(type(fault)) as raised:
        live._run(output,n_envs=games,n_games=games,count=games,configuration={'episodeSteps':3})
    assert raised.value is fault
    assert len(injected)==1
    assert not hasattr(raised.value,'__notes__')
    names=sorted(p.name for p in output.iterdir())
    assert names==[f'game_{i:06d}.custody.json' for i in range(games)]
    records=live._custody(output)
    for ordinal,record in records.items():
        assert record['status']=='error'
        assert record['complete'] is False
        assert record['action_tape']['complete'] is False
        assert 'episode_sha256' not in record
        assert 'verification' not in record
        assert ('replay publication failed' if ordinal==0 else 'evaluation aborted') in record['error']
    assert recorders[0].active_games==frozenset()
    receipt={'games':games,'target':target,'stage':stage,'injection_path':injected[0],'exception_identity_preserved':True,'files':names,'records':records,'source_sha256':hashlib.sha256(Path(subject.__file__).read_bytes()).hexdigest()}
    (OUT/f'probe-{target}-{stage}-{games}.json').write_text(json.dumps(receipt,indent=2)+'\n')


def test_dot_staging_precedes_visible_file(tmp_path, monkeypatch):
    target=tmp_path/'game_000000.json'
    original=Path.open; writes=[]
    def open_(path,mode='r',*args,**kwargs):
        if mode=='xb':
            assert path.name.startswith('.'), path
            assert path.name.endswith('.tmp'), path
            assert path != target
            assert not target.exists()
            writes.append(str(path))
        return original(path,mode,*args,**kwargs)
    monkeypatch.setattr(Path,'open',open_)
    published=[]
    subject._publish_new_file(target,b'payload',published)
    assert len(writes)==1
    assert published==[target]
    assert target.read_bytes()==b'payload'
    assert list(tmp_path.iterdir())==[target]

@pytest.mark.parametrize('fail_call',[1,2,3])
def test_fsync_failure_rolls_back_and_records_error(tmp_path, monkeypatch, fail_call):
    # Calls 1/2 sync staged files; call 3 syncs final directory entries.
    native=unit.native_calls.__wrapped__(monkeypatch)
    recorder=unit._recorder(tmp_path)
    original=subject.os.fsync; calls=0; fault=OSError(f'r4 fsync {fail_call}')
    def fsync(fd):
        nonlocal calls
        calls+=1
        if calls==fail_call: raise fault
        return original(fd)
    monkeypatch.setattr(subject.os,'fsync',fsync)
    with pytest.raises(OSError) as raised: unit._finish_selected_game(recorder)
    assert raised.value is fault
    assert len(native)==1
    assert sorted(p.name for p in tmp_path.iterdir())==['game_000000.custody.json']
    record=json.loads((tmp_path/'game_000000.custody.json').read_text())
    assert record['status']=='error' and record['complete'] is False
    assert 'episode_sha256' not in record and 'verification' not in record
    assert recorder.active_games==frozenset()
