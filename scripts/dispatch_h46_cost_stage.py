"""Own-child native witness and resource trace; never retries or alters limits."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--grant',required=True)
    a=p.parse_args();plan=json.loads(Path(a.plan).read_bytes());base=Path(a.plan).parent
    log=base/'native_stdout.log';err=base/'native_stderr.log';identity=None
    if log.exists() or err.exists() or Path(plan['output']).exists():
        raise ValueError('UNUSED_ONE_SHOT_ONLY_NO_RETRY')
    env=dict(os.environ,**plan['process_environment'])
    for name in ('TMPDIR','MPLCONFIGDIR','NUMBA_CACHE_DIR','XDG_CACHE_HOME'):
        path=Path(env[name]).resolve()
        if not path.is_relative_to('/home/AbeHiromu'):raise ValueError('OWN_HOME_CACHE_REQUIRED')
        path.mkdir(parents=True,exist_ok=True)
    command=[plan['python'],'-P','-B',str(Path(__file__).with_name('h46_cost_execution.py')),
             '--plan',a.plan,'--grant',a.grant]
    started=time.monotonic()
    with log.open('xb') as out,err.open('xb') as errors,(base/'resource_trace.jsonl').open('x') as trace:
        child=subprocess.Popen(command,stdout=out,stderr=errors,env=env)
        stat=Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()
        identity=dict(pid=child.pid,starttime=stat[19],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),uid=os.getuid())
        (base/'native_identity.json').write_text(json.dumps(identity,indent=2)+'\n')
        while child.poll() is None:
            record=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed=time.monotonic()-started)
            try:record['own_status']=Path(f'/proc/{child.pid}/status').read_text()
            except FileNotFoundError:record['own_status']='EXIT_RACE_NATIVE_WAIT_FOLLOWS'
            record['host_memory_pressure']=Path('/proc/pressure/memory').read_text()
            trace.write(json.dumps(record)+'\n');trace.flush();time.sleep(1)
        code=child.wait()
    receipt=dict(identity=identity,returncode=code,wait_reap=True,wall_seconds=time.monotonic()-started,
                 own_process_remaining=Path(f'/proc/{child.pid}').exists(),automatic_retry=False)
    with (base/'native_exit.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(json.dumps(receipt),flush=True)
    raise SystemExit(0 if code==0 else 1)


if __name__=='__main__':main()
