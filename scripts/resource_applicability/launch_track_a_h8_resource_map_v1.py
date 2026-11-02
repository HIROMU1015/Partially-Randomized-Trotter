#!/usr/bin/env python3
"""Own the coordinator Popen and record native wait/reap; never restart it."""
import argparse,json,os,resource,subprocess,sys,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability.h8_resource_map_contract_v1 import RUNNER,verify
from trottertracks.resource_applicability.h6_server_resources_v1 import process_identity
from trottertracks.resource_applicability.ax2b_h4_contract_v5 import file_hash

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--grant',type=Path,required=True)
    p.add_argument('--control-directory',type=Path,required=True)
    a=p.parse_args();m=json.loads(a.manifest.read_text());g=json.loads(a.grant.read_text());verify(ROOT,m,g)
    d=a.control_directory.resolve()
    if not d.is_relative_to('/home/AbeHiromu'):raise ValueError('LAUNCH_CONTROL_HOME')
    d.mkdir(exist_ok=False)
    argv=[sys.executable,'-I','-P','-B',str(ROOT/RUNNER),'--manifest',str(a.manifest.resolve()),
        '--grant',str(a.grant.resolve()),'--manifest-sha256',file_hash(a.manifest),
        '--grant-sha256',file_hash(a.grant),'--execute']
    resource.setrlimit(resource.RLIMIT_AS,(256*2**20,32*2**30))
    resource.setrlimit(resource.RLIMIT_FSIZE,(64*2**20,32*2**30));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    os.sched_setaffinity(0,{23})
    with (d/'coordinator.log').open('xb') as log:
        child=subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,set(m['proposed_resources']['assigned_cpus'])))
        identity=process_identity(child.pid)
        (d/'launch.json').write_text(json.dumps(dict(identity=identity,command=argv,
            SOURCE=m['source_commit'],output=m['output'],manifest_sha256=file_hash(a.manifest),
            grant_sha256=file_hash(a.grant),launcher=process_identity(os.getpid())),indent=2)+'\n')
        code=child.wait()
    (d/'native_exit.json').write_text(json.dumps(dict(identity=identity,exit_code=code,wait_reaped=True,
        exit_observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),exact_exit_timestamp=None,
        automatic_retry=False),indent=2)+'\n')
    return code

if __name__=='__main__':raise SystemExit(main())
