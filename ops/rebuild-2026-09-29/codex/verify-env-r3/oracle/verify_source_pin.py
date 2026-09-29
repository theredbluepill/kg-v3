from pathlib import Path
import hashlib, json, subprocess, zipfile
import numpy as np
root=Path('/Users/poonszesen/kg-v3-env');out=root/'ops/rebuild-2026-09-29/codex/verify-env-r3/oracle'
fixture=root/'tests/fixtures/kaggriculture_env_reference_v1.npz';manifest=json.loads(fixture.with_suffix('.json').read_text());source=manifest['sources'];ref='65f0eac5bb00b18a9d3acce319c2a231cbd5dff0'
def digest(x):return hashlib.sha256(x).hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=root)
assert source['reference']==ref
paths=git('ls-tree','-r','--name-only',ref,'--','engine_rs','python/owl/kaggriculture/actor_codec.py','rust-toolchain.toml').decode().splitlines()
assert set(paths)==set(source['reference_sha256'])
for path in paths: assert digest(git('show',f'{ref}:{path}'))==source['reference_sha256'][path],path
local={'recorder_sha256':'scripts/record_kaggriculture_env_reference.py','rust_recorder_sha256':'ops/rebuild-2026-09-29/1.4/reference_recorder.rs','policy_sha256':'scripts/kaggriculture_env_reference_policy.py','grammar_fixture_sha256':'tests/fixtures/kaggriculture/grammar-v4-reference.jsonl.gz','grammar_manifest_sha256':'tests/fixtures/kaggriculture/grammar-v4-reference.manifest.json'}
for key,path in local.items():assert digest((root/path).read_bytes())==source[key],key
assert digest(git('archive',ref,'engine_rs','python/owl/kaggriculture/actor_codec.py','rust-toolchain.toml'))==source['archive_sha256']
assert digest(fixture.read_bytes())==manifest['fixture_sha256']
assert fixture.stat().st_size==manifest['compressed_bytes']
expanded=hashlib.sha256();size=0
with zipfile.ZipFile(fixture) as z:
 for name in z.namelist():
  data=z.read(name);expanded.update(data);size+=len(data)
assert expanded.hexdigest()==manifest['expanded_sha256'];assert size==manifest['expanded_bytes']
with np.load(fixture,allow_pickle=False) as a:
 for name in a.files:
  assert manifest['arrays'][name]==dict(dtype=a[name].dtype.str,shape=list(a[name].shape),sha256=digest(a[name].tobytes(order='C')))
 report=dict(reference=ref,pinned_git_source_files=len(paths),local_source_files=len(local),source_bytes_match=True,fixture_sha256=manifest['fixture_sha256'],compressed_bytes=fixture.stat().st_size,expanded_bytes=size,games=manifest['games'],steps_per_game=manifest['steps_per_game'],transitions=int(a['transition_indices'].size),seat_programs=int(a['lengths'].size),active_frames=len(a['tokens']),length_range=[int(a['lengths'].min()),int(a['lengths'].max())],coverage=manifest['coverage'],all_required_coverage_positive=all(all(v>0 for v in game.values())for game in manifest['coverage']),reference_training_source_sha256=source['reference_sha256']['engine_rs/src/training.rs'])
(out/'source-pin-corpus.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='coverage'}))
