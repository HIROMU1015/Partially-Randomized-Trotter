"""External read-only progress snapshots; does not inspect operator values."""
from pathlib import Path
import json,time,os
p=Path.cwd()/'artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v2'
def snapshot():
 job=json.loads((p/'ledger_job.started.json').read_text());pid=job['pid'];cpu=None;rss=None;hwm=None;threads=None
 status=Path('/proc')/str(pid)/'status';stat=Path('/proc')/str(pid)/'stat'
 if stat.exists():
  s=stat.read_text();a=s[s.rfind(')')+2:].split();cpu=(int(a[11])+int(a[12]))/os.sysconf('SC_CLK_TCK')
 if status.exists():
  fields={x.split(':')[0]:x.split(':')[1].strip() for x in status.read_text().splitlines() if ':' in x}
  rss=fields.get('VmRSS');hwm=fields.get('VmHWM');threads=fields.get('Threads')
 files=[f for f in p.rglob('*') if f.is_file()];rows=sorted((p/'rows').glob('*.json'));last=None
 if rows:
  row=json.loads(rows[-1].read_text());last={k:row[k] for k in ('id','arm','candidate','proposal')}
 return dict(observed_UTC_epoch=time.time(),ledger_PID=pid,ledger_CPU_seconds=cpu,ledger_RSS=rss,ledger_HWM=hwm,ledger_threads=threads,completed_event_configs=len(list((p/'events').glob('config_*.json'))),saved_laws=len(list((p/'laws').glob('*.json'))),saved_rows=len(rows),latest_row_metadata=last,saved_bytes=sum(f.stat().st_size for f in files),partial_outputs=[dict(path=str(f.relative_to(p)),bytes=f.stat().st_size) for f in files if f.name.endswith('.partial')],terminal_present=(p/'terminal.json').exists())
latest=None
for i in range(5):
 latest=snapshot()
 with Path('/tmp/parent-allocation-v2-observation.jsonl').open('a') as out:out.write(json.dumps(latest)+'\n')
 if latest['terminal_present']:break
 if i<4:time.sleep(10)
print(json.dumps(latest))
