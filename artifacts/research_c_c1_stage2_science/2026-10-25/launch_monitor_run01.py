"""Resource-only wrapper for the fixed one-shot command; never retries it."""
from pathlib import Path
import datetime
import json
import os
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
ENV_ROOT=Path('/tmp/prt-c1-stage2-preparation-20261025')


def descendants(parent):
    found={parent}
    todo=[parent]
    while todo:
        pid=todo.pop()
        try:
            children=Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
        except (FileNotFoundError,ProcessLookupError):
            continue
        for child in children:
            value=int(child)
            if value not in found:
                found.add(value);todo.append(value)
    return found


def sample(parent):
    rows=[]
    for pid in sorted(descendants(parent)):
        try:
            fields=dict(line.split(':',1) for line in Path(f'/proc/{pid}/status').read_text().splitlines() if ':' in line)
            rows.append(dict(pid=pid,name=fields['Name'].strip(),
                 rss_kib=int(fields.get('VmRSS','0 kB').split()[0]),
                 peak_rss_kib=int(fields.get('VmHWM','0 kB').split()[0]),
                 threads=int(fields.get('Threads','0').strip())))
        except (FileNotFoundError,ProcessLookupError):
            continue
    memory=Path('/proc/meminfo').read_text().splitlines()
    return dict(time_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                processes=rows,total_sampled_rss_kib=sum(x['rss_kib'] for x in rows),
                total_threads=sum(x['threads'] for x in rows),loadavg=os.getloadavg(),
                memory_available=next(x for x in memory if x.startswith('MemAvailable:')),
                memory_pressure=Path('/proc/pressure/memory').read_text())


def main():
    assert not (HERE/'run01').exists(), 'Never reuse an evaluation attempt'
    command=[str(ENV_ROOT/'.venv/bin/python'),
        str(ROOT/'scripts/run_research_c_c1_stage2_native.py'),
        '--contract',str(ROOT/'docs/research/research_c_c1_stage2_native_contract_unapproved_20261025.json'),
        '--input','/tmp/prt-c1-stage1-compatibility-20261025/.external/P-H4.npz',
        '--output',str(HERE/'run01'),
        '--execute']
    environment=dict(os.environ,XDG_CACHE_HOME=str(ROOT/'.cache'),
        PYTHONPYCACHEPREFIX=str(ROOT/'.cache/pycache'),MPLCONFIGDIR=str(ROOT/'.cache/matplotlib'),TMPDIR=str(ROOT/'.cache/tmp'),NUMBA_CACHE_DIR=str(ROOT/'.cache/numba'))
    started=time.monotonic();count=0;rss=0;threads=0
    with (HERE/'launch_run01.json').open('x') as f:
        json.dump(dict(command=command,source_commit='a977a4390668bc107360c8ff05f43d851dc81dc9',
            resource_caps=None,scientific_retry=0,started_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()),f,indent=2)
        f.write('\n')
    with (HERE/'runner_run01.txt').open('x') as log, (HERE/'resource_samples_run01.jsonl').open('x') as resources:
        child=subprocess.Popen(['/usr/bin/time','-v','-o',str(HERE/'resource_time_run01.txt'),*command],
            cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
        while child.poll() is None:
            row=sample(child.pid);count+=1;rss=max(rss,row['total_sampled_rss_kib']);threads=max(threads,row['total_threads'])
            resources.write(json.dumps(row)+'\n');resources.flush()
            time.sleep(1)
        exitcode=child.wait()
    result=dict(exitcode=exitcode,elapsed_seconds=time.monotonic()-started,
        sampling_interval_seconds=1,samples=count,peak_sampled_aggregate_rss_kib=rss,
        peak_sampled_total_threads=threads,scope='GNU time and sampled fixed-runner descendants, not a scientific metric',
        scientific_retries=0,automatic_resume=False)
    with (HERE/'resource_usage_run01.json').open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
