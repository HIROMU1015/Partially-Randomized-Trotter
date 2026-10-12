"""External status/resource receipt only; exactly one registered runner spawn."""
import datetime,hashlib,json,os,resource,subprocess,time
from pathlib import Path
root=Path('/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-df-correction-et-v4-one-shot-execution-20261012')
out=Path('/tmp/track-b-et-v4-one-shot-clcgpqv4')
command=['/home/abe/Project/Partially Randomized Trotter/.venv311/bin/python','-B','scripts/tracks/algorithm_codesign/run_df_correction_et_resumption_v4.py']
env=dict(os.environ,PYTHONPATH=str(root/'src'),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',RAYON_NUM_THREADS='1',QISKIT_PARALLEL='FALSE')
def exclusive(path,value):
 with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
def snapshot():
 rows={};ticks=os.sysconf('SC_CLK_TCK');page=os.sysconf('SC_PAGE_SIZE')
 for path in Path('/proc').iterdir():
  if not path.name.isdigit():continue
  try:
   values=(path/'stat').read_text().rsplit(') ',1)[1].split()
   rows[int(path.name)]={'parent':int(values[1]),'cpu':(int(values[11])+int(values[12]))/ticks,'rss':int(values[21])*page/1024**2,'state':values[0]}
  except (FileNotFoundError,ProcessLookupError,PermissionError):pass
 return rows
start=time.monotonic();utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
exclusive(out/'runner_invocation_claim.json',{'command':command,'cwd':str(root),'run_invocations':1,'retries':0,'mandatory_STOP':True,'created_UTC':utc,'source':'016e02b486cd4b182b654af18743ef1ef50fe7e6','authorization':'111199c45a9c37ac81821512ea8a44e6093f50cc'})
before=resource.getrusage(resource.RUSAGE_CHILDREN);peak=0.;peak_cpu=0.;samples=0
with (out/'outer_runner_stdout.log').open('xb') as stream:
 p=subprocess.Popen(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 while p.poll() is None:
  rows=snapshot();selected={p.pid}
  while True:
   more={pid for pid,row in rows.items() if row['parent'] in selected}
   if more<=selected:break
   selected|=more
  peak=max(peak,sum(rows[i]['rss'] for i in selected if i in rows and rows[i]['state']!='Z'))
  peak_cpu=max(peak_cpu,sum(rows[i]['cpu'] for i in selected if i in rows and rows[i]['state']!='Z'))
  samples+=1;time.sleep(.05)
 code=p.wait()
after=resource.getrusage(resource.RUSAGE_CHILDREN);raw=(out/'outer_runner_stdout.log').read_bytes()
value={'command':command,'cwd':str(root),'runner_PID':p.pid,'exit_status':code,'wall_seconds':time.monotonic()-start,'CPU_seconds_reaped_runner_tree':after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,'peak_sampled_runner_tree_RSS_MiB':peak,'peak_sampled_live_tree_CPU_seconds':peak_cpu,'sample_count':samples,'RSS_sample_seconds':.05,'RSS_scope':'runner plus observed descendants; sampled, not instantaneous ceiling','stdout_bytes':len(raw),'stdout_sha256':hashlib.sha256(raw).hexdigest(),'run_invocations':1,'retries':0,'mandatory_STOP':True,'source':'016e02b486cd4b182b654af18743ef1ef50fe7e6','authorization':'111199c45a9c37ac81821512ea8a44e6093f50cc'}
exclusive(out/'outer_process_receipt.json',value)
print(json.dumps(value,indent=2),flush=True)
print(raw.decode(errors='replace')[-5000:],flush=True)
