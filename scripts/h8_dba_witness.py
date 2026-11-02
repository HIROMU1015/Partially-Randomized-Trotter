"""Detached native wait witness for one H8 case, with no science dispatch beyond it."""
import argparse
import os
from pathlib import Path
import subprocess
import time
import h8_dba_contract as c


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--grant',type=Path,required=True)
    a=p.parse_args();config=c.read(a.config);grant=c.read(a.grant);base=Path(grant['output']).parent
    command=[config['python'],'-B',str(c.ROOT/'scripts/run_h8_dba_direct.py'),'--config',str(a.config),'--grant',str(a.grant)]
    start=time.monotonic()
    with (base/'stdout.log').open('xb') as out,(base/'stderr.log').open('xb') as err:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=out,stderr=err)
        c.save(base/'native_started.json',dict(pid=child.pid,SOURCE=grant['SOURCE'],case_id=grant['case_id'],
            command=command,config_digest=c.digest(config),grant_digest=c.digest(grant)))
        pid,status,usage=os.wait4(child.pid,0);code=os.waitstatus_to_exitcode(status);child.returncode=code
    c.save(base/'native_exit.json',dict(pid=pid,exit_code=code,wait4_complete=True,same_parent_reaped=True,
        SOURCE=grant['SOURCE'],case_id=grant['case_id'],wall_seconds=time.monotonic()-start,
        native_cpu_seconds=usage.ru_utime+usage.ru_stime,peak_RSS_bytes=usage.ru_maxrss*1024,
        next_stage_authorized=False))
    return code


if __name__=='__main__':raise SystemExit(main())
