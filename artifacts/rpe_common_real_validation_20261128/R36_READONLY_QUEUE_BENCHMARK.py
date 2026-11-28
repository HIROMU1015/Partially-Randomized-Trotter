from contextlib import contextmanager
from pathlib import Path
import sqlite3,sys,time,statistics,json
root=Path('/home/abe/Project/Partially Randomized Trotter')
sys.path.insert(0,str(root/'.worktrees/rpe-common-review-fixes-20261128/src'))
from trottertracks.rpe_q3_store import JobStore
db_path=root/'.rpe-central/state/runs/search-78ee25e2f10dcaccee5c/distributed/QUEUE.sqlite3'
class ReadOnly:
    @contextmanager
    def _connection(self):
        conn=sqlite3.connect(db_path.as_uri()+'?mode=ro',uri=True)
        try:yield conn
        finally:conn.close()
    capacity_history=JobStore.capacity_history
    capacity_history_batch=JobStore.capacity_history_batch
s=ReadOnly()
with s._connection() as c:
    hosts=[h for h, in c.execute('SELECT DISTINCT host_id FROM jobs WHERE host_id IS NOT NULL ORDER BY host_id')]
    totals={j:max(0.0,a-e) for j,e,a in c.execute('SELECT job_id,enqueued_epoch,assigned_epoch FROM jobs WHERE assigned_epoch IS NOT NULL')}
old=lambda:{h:s.capacity_history(h,0,totals) for h in hosts}
batch=lambda:s.capacity_history_batch(hosts,0,totals)
a,b=old(),batch()
assert all(v[:2]==b[h][:2] and (v[2]==b[h][2] or abs(v[2]-b[h][2])<1e-6) for h,v in a.items())
if len(sys.argv)>1:
    which=sys.argv[1];fn=old if which=='old' else batch;until=time.monotonic()+12
    n=0;start=time.perf_counter();cpu=time.process_time()
    while time.monotonic()<until:fn();n+=1
    print(json.dumps(dict(mode=which,iterations=n,wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,hosts=hosts,wait_rows=len(totals))),flush=True)
else:
    times={'old':[],'batch':[]}
    for _ in range(15):
        for name,fn in [('old',old),('batch',batch)]:
            t=time.perf_counter();fn();times[name].append(time.perf_counter()-t)
    print(json.dumps(dict(hosts=hosts,wait_rows=len(totals),results=a,median_seconds={k:statistics.median(v) for k,v in times.items()})),flush=True)
