"""Durable launch receipt owner; one fixed invocation only, no retry/resume."""
import os,sys,json,datetime,subprocess,hashlib
from pathlib import Path
ROOT=Path('/home/abe/Project/Partially Randomized Trotter/.worktrees/track-a-ax2b-h4-post-review-20261010')
META=ROOT/'artifacts/resource_applicability/track_a_h6_matched_execution_v3/2026-10-11'
PREP=ROOT/'artifacts/resource_applicability/track_a_h6_matched_preparation_v3/2026-10-11'
OUT=ROOT/'artifacts/resource_applicability/track_a_h6_matched_supplement_v3/2026-10-11/launch_v1'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(name,obj):
 with (META/name).open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def proc_start(pid):return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])
def owner():
 sys.path.insert(0,str(ROOT/'src'))
 from trottertracks.resource_applicability.h6_matched_contract_v3 import validate_launch
 m=json.loads((PREP/'sealed_manifest_v3.json').read_text());g=json.loads((META/'authorization_v3.json').read_text())
 grant_sha=hashlib.sha256((META/'authorization_v3.json').read_bytes()).hexdigest()
 assert grant_sha=='b467d053d93b104223ff1ac3c99a14b647f831fcd59d9d41087995cbf10eadda'
 proof=json.loads((META/'remote_prelaunch_verification_v3.json').read_text())
 assert proof['status']=='INDEPENDENT_REMOTE_PRELAUNCH_BYTES_PASS' and proof['grant_sha256']==grant_sha
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==proof['publication_commit']
 validate_launch(ROOT,m,g,OUT,requested=True)
 command=[sys.executable,str(ROOT/'scripts/resource_applicability/run_track_a_h6_matched_v3.py'),'--execute','--output',str(OUT),'--manifest',str(PREP/'sealed_manifest_v3.json'),'--authorization',str(META/'authorization_v3.json'),'--authorization-sha256',grant_sha]
 save('launch_once_v3.json',{'recorded_at_utc':now(),'command':command,'scientific_runner_invocations':1,'retry':False,'resume':False,'source_commit':m['source_commit'],'grant_commit':proof['grant_commit'],'grant_sha256':grant_sha,'owner_pid':os.getpid(),'phase_A_index':447,'total_wall_cap':None,'phase_wall_cap':None,'next_stage_authorized':False})
 env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(ROOT/'src')}
 with (META/'runner_stdout_v3.log').open('xb') as log:
  process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  save('process_start_observation_v3.json',{'recorded_at_utc':now(),'pid':process.pid,'proc_start_ticks':proc_start(process.pid),'owner_pid':os.getpid(),'command':command,'scientific_runner_invocations':1})
  code=process.wait()
 save('runner_exit_v3.json',{'recorded_at_utc':now(),'pid':process.pid,'exit_code':code,'scientific_runner_invocations':1,'retry':False,'resume':False,'mandatory_stop':True,'next_stage_authorized':False,'terminal_exists':(OUT/'execution_terminal.json').exists()})
if __name__=='__main__':
 if sys.argv[1:]==['--owner']:owner()
 else:
  assert not (META/'launch_once_v3.json').exists() and not (META/'owner_launch_v3.json').exists() and not OUT.exists()
  with (META/'launch_owner_v3.log').open('xb') as log:
   p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--owner'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
  save('owner_launch_v3.json',{'recorded_at_utc':now(),'owner_pid':p.pid,'proc_start_ticks':proc_start(p.pid),'owner_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'one_owner_launched':True})
  print(json.dumps({'owner_pid':p.pid,'status':'DURABLE_ONE_SHOT_OWNER_STARTED'}))
