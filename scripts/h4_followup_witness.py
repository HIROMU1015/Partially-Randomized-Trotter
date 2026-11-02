"""Same-parent wait4 receipt for a detached, separately authorized H4 stage."""
import argparse
import os
from pathlib import Path
import subprocess
import time
import h4_followup_contract as c


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['R','B','T','F','FP'],required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--grant',type=Path,required=True)
    a=p.parse_args();config=c.read(a.config);grant=c.read(a.grant);base=Path(grant['output']).parent
    command=[config['python'],'-B',str(c.ROOT/'scripts/run_h4_followup.py'),'--stage',a.stage,
             '--config',str(a.config),'--grant',str(a.grant)]
    start=time.monotonic()
    with (base/'stdout.log').open('xb') as out,(base/'stderr.log').open('xb') as err:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=out,stderr=err)
        c.save(base/'native_started.json',dict(pid=child.pid,command=command,SOURCE=grant['SOURCE'],
               config_digest=c.digest(config),grant_digest=c.digest(grant)))
        pid,status,usage=os.wait4(child.pid,0);code=os.waitstatus_to_exitcode(status);child.returncode=code
    c.save(base/'native_exit.json',dict(pid=pid,exit_code=code,same_parent_reaped=True,wait4_complete=True,
        SOURCE=grant['SOURCE'],wall_seconds=time.monotonic()-start,native_cpu_seconds=usage.ru_utime+usage.ru_stime,
        peak_RSS_bytes=usage.ru_maxrss*1024,next_stage_authorized=False))
    return code


if __name__=='__main__':raise SystemExit(main())
