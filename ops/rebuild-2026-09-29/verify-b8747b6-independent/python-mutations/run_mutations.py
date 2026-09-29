from pathlib import Path
import hashlib, json, os, shutil, subprocess, time
ROOT=Path.cwd()
OUT=ROOT/'ops/rebuild-2026-09-29/verify-b8747b6-independent/python-mutations'
SCRATCH=OUT/'scratch'
PYTHON=ROOT/'.venv/bin/python'
files=[
 'tests/tools/test_observation_oracle_custody.py','tests/tools/test_check_engine_trim.py',
 'tests/kaggriculture/test_observe.py','scripts/kaggriculture_observation_oracle/regenerate.py',
 'scripts/check_engine_trim.py','engine_rs/fixtures/episode-95324500.jsonl.gz',
 'src/kaggriculture/grammar_kernel_tests.rs',
]
for f in files:
 p=SCRATCH/f; p.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ROOT/f,p)
(SCRATCH/'pytest.ini').write_text('[pytest]\n')
# Observe the actual skip expression under mocked availability, without importing
# torch or touching any allocator. Both values are checked independently.
(SCRATCH/'test_pinned_guard.py').write_text('''import ast\nfrom pathlib import Path\nfrom types import SimpleNamespace\nimport pytest\n\n@pytest.mark.parametrize("cuda", [False, True])\ndef test_pinned_guard_never_enables_allocator_without_cuda(cuda):\n    path=Path(__file__).parent / "tests/kaggriculture/test_observe.py"\n    tree=ast.parse(path.read_text())\n    node=next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id=="_CUDA_PINNED" for t in n.targets))\n    torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda))\n    marker=eval(compile(ast.Expression(node),str(path),"eval"), {"pytest":pytest,"torch":torch})\n    assert marker.values == (True,)\n    skip,=marker.marks\n    assert skip.name == "skipif"\n    assert skip.args[0] is (not cuda)\n''')
(SCRATCH/'test_snapshot_inventory.py').write_text('''from pathlib import Path
import importlib.util
import pytest
ROOT=Path(__file__).parent
spec=importlib.util.spec_from_file_location("oracle_independent",ROOT/"scripts/kaggriculture_observation_oracle/regenerate.py")
assert spec and spec.loader
oracle=importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)

def _engine(root):
    for path in ("engine_rs/src/lib.rs", "engine_rs/src/py_random.rs", "engine_rs/src/econ_attrib.rs"):
        p=root/path; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(b"source")

def test_capture_declares_sources_before_executing_any_command(tmp_path,monkeypatch):
    _engine(tmp_path)
    (tmp_path/"engine_rs/src/unlisted.rs").write_bytes(b"source")
    monkeypatch.setattr(oracle,"ROOT",tmp_path)
    def forbidden_run(*args, **kwargs):
        pytest.fail("undeclared source reached command execution")
    monkeypatch.setattr(oracle,"run",forbidden_run)
    with pytest.raises(ValueError,match="undeclared engine source"):
        oracle.source_snapshot()

def test_recheck_declares_new_sources(tmp_path,monkeypatch):
    _engine(tmp_path)
    monkeypatch.setattr(oracle,"ROOT",tmp_path)
    monkeypatch.setattr(oracle,"run",lambda *args,**kwargs: b"a"*40)
    oracle.check_source_snapshot(("a"*40, ()))
    (tmp_path/"engine_rs/src/unlisted.rs").write_bytes(b"source")
    with pytest.raises(ValueError,match="undeclared engine source"):
        oracle.check_source_snapshot(("a"*40, ()))
''')
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
original={f:sha(SCRATCH/f) for f in files}
repo_original={f:sha(ROOT/f) for f in files}
base='scripts/kaggriculture_observation_oracle/regenerate.py'
oracle_test='tests/tools/test_observation_oracle_custody.py'
trim_test='tests/tools/test_check_engine_trim.py'
# Each tuple: name, source, exact old/new, pytest target selector.
cases=[
 ('seat_bytes',base,'and sha(block)\n                        == digest(meta["seat_sha256"][seat], "seat hash"),','and True,',oracle_test+'::test_custody_rejects_corruption_even_with_rehashed_files[byte]'),
 ('header_order',base,'digest(meta["header_sha256"], "header hash")\n                    == sha(header_bytes(raw)),','True,',oracle_test+'::test_custody_rejects_corruption_even_with_rehashed_files[header_order]'),
 ('schema_version',base,'manifest["schema_version"] == 1','manifest["schema_version"] in (1, 2)',oracle_test+'::test_custody_rejects_corruption_even_with_rehashed_files[version]'),
 ('duplicate_record',base,'row["record_id"] not in found_ids, "duplicate record id"','True, "duplicate record id"',oracle_test+'::test_custody_rejects_corruption_even_with_rehashed_files[duplicate]'),
 ('source_header_step',base,'row["source"]["step"] == row["header"]["initial"]["public"]["step"],','True,',oracle_test+'::test_record_source_step_cannot_disagree_with_header'),
 ('quota',base,'counts = coverage["non_synthetic"]','counts = coverage["dense"]',oracle_test+'::test_final_quota_cannot_be_satisfied_by_dense_states'),
 ('byteplanes',base,'raw[plane::4] = src.read(length)','raw[(plane + 1) % 4::4] = src.read(length)',oracle_test+'::test_byteplane_round_trip_preserves_float_bits'),
 ('source_snapshot',base,'def check_source_snapshot(snapshot: SourceSnapshot) -> None:\n','def check_source_snapshot(snapshot: SourceSnapshot) -> None:\n    return\n',oracle_test+'::test_regeneration_source_identity_rejects_changes_between_phases'),
 ('archive_custody',base,'hash_file(directory / path) == expected,','True,',oracle_test+'::test_reference_export_checks_archive_bytes_against_git_show[True]'),
 ('recorder_copy',base,'hash_file(example) == recorder_sha,','True,',oracle_test+'::test_exported_recorder_must_match_preexecution_source[copy]'),
 ('caller_baseline',base,'caller_baseline = group_rss(os.getpgrp())','caller_baseline = 0',oracle_test+'::test_preexisting_caller_memory_is_not_charged_to_the_child'),
 ('caller_growth',base,'caller_growth = max(0, group_rss(os.getpgrp()) - caller_baseline)','caller_growth = 0',oracle_test+'::test_caller_growth_after_launch_still_counts'),
 ('shared_deadline',base,'or now >= RUN_DEADLINE:',':',oracle_test+'::test_shared_deadline_stops_child_group_before_outer_limit'),
 ('retired_authored_set','scripts/check_engine_trim.py','["engine_rs/tests/replay_parity.rs", GENERATED_MANIFEST],','["engine_rs/tests/replay_parity.rs", GENERATED_MANIFEST, "engine_rs/tests/grammar_kernel.rs"],',trim_test+'::test_task_authored_inventory_requires_exact_authored_set'),
 ('bridge_retirement','engine_rs/tests/grammar_kernel.rs',None,'// Independent verifier mutation: bridge reintroduced\n',trim_test+'::test_grammar_bridge_is_retired_to_root_integration'),
 ('pinned_availability','tests/kaggriculture/test_observe.py','not torch.cuda.is_available(), reason="pinned host memory requires CUDA"','False, reason="pinned host memory requires CUDA"','test_pinned_guard.py'),
]
cases.extend([
 ('engine_cargo_source_omitted',base,'    *ENGINE_SOURCE_PATHS,\n)','    *(p for p in ENGINE_SOURCE_PATHS if p != "engine_rs/Cargo.toml"),\n)',oracle_test+'::test_regeneration_source_identity_rejects_changes_between_phases[admission-engine_rs/Cargo.toml]'),
 ('engine_random_source_omitted',base,'    *ENGINE_SOURCE_PATHS,\n)','    *(p for p in ENGINE_SOURCE_PATHS if p != "engine_rs/src/py_random.rs"),\n)',oracle_test+'::test_regeneration_source_identity_rejects_changes_between_phases[producer-engine_rs/src/py_random.rs]'),
 ('engine_econ_source_omitted',base,'    *ENGINE_SOURCE_PATHS,\n)','    *(p for p in ENGINE_SOURCE_PATHS if p != "engine_rs/src/econ_attrib.rs"),\n)',oracle_test+'::test_regeneration_source_identity_rejects_changes_between_phases[export-engine_rs/src/econ_attrib.rs]'),
 ('engine_declaration_capture',base,'    require_declared_engine_sources(ROOT)\n    commit = run','    commit = run','test_snapshot_inventory.py::test_capture_declares_sources_before_executing_any_command'),
 ('engine_declaration_recheck',base,'    commit, files = snapshot\n    require_declared_engine_sources(ROOT)','    commit, files = snapshot','test_snapshot_inventory.py::test_recheck_declares_new_sources'),
])
records=[]
def run(name,phase,target):
 cmd=[str(PYTHON),'-B','-m','pytest','-c',str(SCRATCH/'pytest.ini'),'--confcutdir',str(SCRATCH),'-q',str(SCRATCH/target)]
 started=time.monotonic()
 proc=subprocess.run(cmd,cwd=SCRATCH,env=os.environ|{'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=50)
 (OUT/f'{name}-{phase}.log').write_bytes(proc.stdout)
 return {'exit':proc.returncode,'wall_seconds':round(time.monotonic()-started,3),'log':f'{name}-{phase}.log','argv':cmd}
for name,path,old,new,target in cases:
 file=SCRATCH/path
 before=file.read_bytes() if file.exists() else None
 entry={'name':name,'path':path,'before_sha256':sha(file),'mutation':{'old':old,'new':new}}
 try:
  entry['baseline']=run(name,'baseline',target)
  assert entry['baseline']['exit']==0, (name,'baseline failed')
  if before is None: file.parent.mkdir(parents=True,exist_ok=True); file.write_text(new)
  else:
   content=before.decode(); assert old in content,(name,'mutation anchor missing')
   # Recorder-copy mutation intentionally disables both copy and postexecution
   # comparisons, ensuring the test reaches the missing-guard assertion.
   file.write_text(content.replace(old,new))
  entry['mutant_sha256']=sha(file)
  entry['mutant']=run(name,'mutant',target)
  assert entry['mutant']['exit']==1, (name,'mutation not killed')
 finally:
  if before is None: file.unlink(missing_ok=True)
  else: file.write_bytes(before)
  entry['restored_sha256']=sha(file)
  entry['restoration_exact']=entry['restored_sha256']==entry['before_sha256']
  entry['restored']=run(name,'restored',target)
  records.append(entry)
  (OUT/'results.json').write_text(json.dumps(records,indent=2)+'\n')
 print(name,entry['baseline']['exit'],entry['mutant']['exit'],entry['restored']['exit'],entry['restoration_exact'],flush=True)
 assert entry['restored']['exit']==0
assert all(sha(SCRATCH/f)==original[f] for f in files)
assert all(sha(ROOT/f)==repo_original[f] for f in files)
(OUT/'restoration.json').write_text(json.dumps({'scratch':original,'repository':repo_original,'all_source_bytes_exact':True},indent=2)+'\n')
