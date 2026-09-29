"""Supplemental core probes; all source changes only in private scratch."""
import json, os, subprocess, sys, hashlib
from pathlib import Path
from run_core_mutations import HERE, SCRATCH, ROOT, Mutation, run_one
m=Mutation("timing-fixture-density", "src/kaggriculture/lifecycle_timing_tests.rs", "lifecycle_timing_fixtures_step_and_prepare", "timing_header(dense)", "timing_header(false)", "Replace dense fixture with early state; the density oracle must reject.")
r=run_one(m,115)
print(json.dumps({"timing":r["classification"]}),flush=True)
env=os.environ | {"CARGO_BUILD_JOBS":"1", "CARGO_NET_OFFLINE":"true", "UV_OFFLINE":"true", "RUST_TEST_THREADS":"1"}
def run(name,args):
    prefix=HERE/name
    cmd=[sys.executable,str(HERE/"bounded.py"),"--seconds","115","--name",str(prefix),"--",*args]
    with prefix.with_suffix(".runner.log").open("w") as f:
        p=subprocess.run(cmd,cwd=SCRATCH,env=env,stdout=f,stderr=subprocess.STDOUT)
    receipt=json.loads(prefix.with_suffix(".json").read_text())
    print(json.dumps({"name":name,"exit":receipt["exit_status"],"stop":receipt["stop_reason"]}),flush=True)
    return receipt
base=["cargo","test","--release","--locked","--offline"]
tail=["--lib","release_dependency_overflow_is_caught","--","--test-threads=1"]
release=run("release-canonical",base+tail)
flags=[] if release["exit_status"]==0 else ["--config","profile.release.lto=false","--config","profile.release.codegen-units=16"]
if flags:
    run("release-no-lto",base+flags+tail)
run("release-overflow-mutant",base+flags+["--config","profile.release.package.kaggriculture-engine.overflow-checks=false"]+tail)
run("release-restored",base+flags+tail)
run("restored-core",["cargo","test","--locked","--offline","--lib","kaggriculture","--","--test-threads=1"])
custody=json.loads((HERE/"source-custody.json").read_text())
custody["restoration"]={rel:{"primary_sha256":hashlib.sha256((ROOT/rel).read_bytes()).hexdigest(),"scratch_sha256":hashlib.sha256((SCRATCH/rel).read_bytes()).hexdigest(),"byte_exact":(ROOT/rel).read_bytes()==(SCRATCH/rel).read_bytes()} for rel in custody["files"]}
(HERE/"restoration.json").write_text(json.dumps(custody,indent=2)+"\n")
