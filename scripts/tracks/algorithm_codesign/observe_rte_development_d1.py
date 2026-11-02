#!/usr/bin/env python3
"""One launch and wait4 receipt. Never retries; preserves externally failed runs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign.rte_development.launch import verify_launch
from trottertracks.algorithm_codesign.rte_development.evidence import atomic_json,digest
CONTRACT=ROOT/'artifacts/track_b_rte_development_source/2026-10-13/v1/contract_v1.json'


def execute(source):
    c,auth,head=verify_launch(ROOT,CONTRACT,source)
    logdir=Path(tempfile.mkdtemp(prefix='track-b-d1-process-'))
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
             NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    start=time.monotonic()
    with (logdir/'stdout.log').open('xb') as out,(logdir/'stderr.log').open('xb') as err:
        child=subprocess.Popen([c['runtime_executable'],'-B',
            'scripts/tracks/algorithm_codesign/run_rte_development_d1.py','--source',source],
            cwd=ROOT,env=env,stdout=out,stderr=err,start_new_session=True)
        stopped_for_logs=False
        while True:
            waited,status,usage=os.wait4(child.pid,os.WNOHANG)
            if waited:break
            if any(p.stat().st_size>c['caps']['per_key_log_bytes'] for p in logdir.glob('*.log')):
                os.killpg(child.pid,signal.SIGKILL);stopped_for_logs=True
            time.sleep(.1)
        child.returncode=os.waitstatus_to_exitcode(status)
    directory=ROOT/c['result_directory'];tp=directory/'terminal.json'
    terminal=json.loads(tp.read_text()) if tp.exists() else None
    complete=bool(not stopped_for_logs and child.returncode==0 and terminal and
                  terminal.get('status')=='D1_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW')
    receipt={'exit_code':child.returncode,'signal':-child.returncode if child.returncode<0 else None,
        'wall_seconds':time.monotonic()-start,'CPU_seconds_including_children':usage.ru_utime+usage.ru_stime,
        'parent_peak_RSS_KiB':usage.ru_maxrss,'execution_HEAD':head,'source_commit':source,
        'authorization_sha256':digest(ROOT/c['authorization_path']),'contract_sha256':digest(CONTRACT),
        'terminal_sha256':digest(tp) if tp.exists() else None,'terminal_status':terminal.get('status') if terminal else None,
        'complete_status_consistent':complete,'effective_status':'D1_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW' if complete else 'D1_TECHNICAL_INCONCLUSIVE',
        'marker_present':(directory/'one_shot_consumed.json').exists(),
        'external_log_cap_exceeded':stopped_for_logs,
        'external_logs':[{'path':str(p),'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(logdir.iterdir())],
        'runs':1,'retries':0,'mandatory_STOP':True,'next_science_authorized':False}
    atomic_json(logdir/'process_receipt.json',receipt)
    if directory.exists():atomic_json(directory/'outer_process_receipt.json',receipt)
    print(json.dumps(receipt));return 0 if complete else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);a=p.parse_args()
    sys.exit(execute(a.source))
