"""External invocation/status receipt only; invokes the frozen runner once."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import resource
import subprocess
import time

ROOT=Path('/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-df-correction-comparison-one-shot-execution-20261012')
OUT=Path('/tmp/prt_df_comparison_one_shot_20261012')
PY='/home/abe/Project/Partially Randomized Trotter/.venv311/bin/python'
SOURCE='41349b73157c7781e0b534cf0fe61e06bf439954'
AUTH='a1f537b68f0f22419dff0202c553b7511653caa3'
MARKER=ROOT/'artifacts/track_b_df_correction_comparison_result/2026-10-11/v2/one_shot_marker.json'

def main():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==AUTH
    assert not subprocess.check_output(['git','status','--porcelain=v1'],cwd=ROOT,text=True).strip()
    assert not MARKER.exists()
    command=[PY,'-B','scripts/tracks/algorithm_codesign/run_df_correction_comparison_v2.py']
    with (OUT/'external_invocation_marker.json').open('x') as f:
        json.dump({'source':SOURCE,'authorization':AUTH,'command':command,'runs':1,'retries':0,
                   'date_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},f,indent=2)
        f.write('\n')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT/'src'),QISKIT_PARALLEL='FALSE')
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS'):
        env[name]='1'
    start=time.monotonic();started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (OUT/'runner.stdout.log').open('xb') as stdout:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=subprocess.STDOUT,start_new_session=True)
        with (OUT/'external_process_start.json').open('x') as f:
            json.dump({'pid':process.pid,'source':SOURCE,'execution_HEAD':AUTH,'started_at_utc':started,'runner_invocations':1,'retries':0},f,indent=2)
        status=process.wait()
    usage=resource.getrusage(resource.RUSAGE_CHILDREN)
    log=(OUT/'runner.stdout.log').read_bytes()
    receipt={'kind':'OUTER_PROCESS_STATUS_RECEIPT','source':SOURCE,'execution_HEAD':AUTH,
       'command':command,'pid':process.pid,'exit_status':status,'started_at_utc':started,
       'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'outer_wall_seconds_including_preflight':time.monotonic()-start,
       'outer_waited_child_CPU_seconds':usage.ru_utime+usage.ru_stime,
       'outer_max_child_RSS_MiB_not_aggregate':usage.ru_maxrss/1024,
       'stdout_bytes':len(log),'stdout_sha256':hashlib.sha256(log).hexdigest(),
       'one_shot_marker_exists':MARKER.exists(),
       'one_shot_marker_sha256':hashlib.sha256(MARKER.read_bytes()).hexdigest() if MARKER.exists() else None,
       'runner_invocations':1,'retries':0,'mandatory_STOP':True}
    with (OUT/'outer_process_receipt.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(json.dumps(receipt),flush=True)
    print(log.decode('utf-8',errors='replace'),flush=True)

if __name__=='__main__':main()
