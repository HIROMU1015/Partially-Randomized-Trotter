#!/usr/bin/env python3
"""Known C namespace/process observations only; never alter worker/run files."""
from datetime import datetime,timezone
import json
from pathlib import Path
import os
import sys
META=Path(__file__).resolve().parent
NS=META.parent/'run01'

def read(path):return json.loads(path.read_text())

def process(identity):
    pid=identity['pid']
    try:
        fields=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
        if int(fields[19])!=identity['start_ticks']:return dict(pid=pid,state='IDENTITY_REUSED')
        return dict(pid=pid,state=fields[0],cpu_ticks=int(fields[11])+int(fields[12]),rss_pages=int(fields[21]),start_ticks=int(fields[19]))
    except (FileNotFoundError,ProcessLookupError):return dict(pid=pid,state='ABSENT')

def observe():
    out=dict(UTC=datetime.now(timezone.utc).isoformat(),run_files=sorted(x.name for x in NS.iterdir()),processes={},native={},stderr={})
    for name in ['call_ledger.json','STOP.json','pauli_jw_partial.json','df_native_partial.json']:
        p=NS/name
        if p.exists():out[name]=read(p)
    folders=[META/'supervisor_native',NS/'pauli_jw_native',NS/'df_native_native',NS/'pauli_jw_bridge_native',NS/'df_native_bridge_native']
    for folder in folders:
        if (folder/'worker_identity.json').exists():
            out['processes'][folder.name]=process(read(folder/'worker_identity.json'))
        if (folder/'worker_native_exit.json').exists():out['native'][folder.name]=read(folder/'worker_native_exit.json')
        p=folder/'stderr.log'
        if p.exists() and p.stat().st_size:out['stderr'][folder.name]=p.read_text()[-6000:]
    out['resources']=dict(loadavg=Path('/proc/loadavg').read_text().strip(),available_memory=next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')),cpu_pressure=Path('/proc/pressure/cpu').read_text().strip())
    return out

if __name__=='__main__':
    out=observe()
    if len(sys.argv)>1:
        with (META/sys.argv[1]).open('x') as f:json.dump(out,f,indent=2,sort_keys=True);f.write('\n')
    summary=dict(UTC=out['UTC'],processes=out['processes'],used=out.get('call_ledger.json',{}).get('used'),STOP=out.get('STOP.json'),partials={k:v for k,v in out.items() if k.endswith('_partial.json')},native_exits={k:v['exit_code'] for k,v in out['native'].items()},stderr=out['stderr'])
    print(json.dumps(summary,sort_keys=True))
