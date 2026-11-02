"""Foreground durable receipt owner; one frozen science invocation, no retry."""
import sys,os,json,subprocess,datetime,hashlib,shutil
from pathlib import Path
ROOT=Path('/home/abe/Project/Partially Randomized Trotter/.worktrees/track-a-ax2b-h4-post-review-20261010')
META=ROOT/'artifacts/resource_applicability/track_a_h6_matched_execution_launch_v2/2026-10-11'
PREP=ROOT/'artifacts/resource_applicability/track_a_h6_matched_launch_preparation_v4/2026-10-11'
OUT=ROOT/'artifacts/resource_applicability/track_a_h6_matched_supplement_v3/2026-10-11/launch_v2'
GRANTSHA='3564a78601474d18aceaa269d292bb8650b291d42042353fb5dd901331b50a57'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(name,obj):
 with (META/name).open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def start_ticks(pid):return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])
assert not (META/'launch_once_launch_v2.json').exists() and not OUT.exists()
proof=json.loads((META/'remote_prelaunch_verification_launch_v2.json').read_text())
assert proof['status']=='INDEPENDENT_REMOTE_PRELAUNCH_BYTES_PASS' and proof['grant_sha256']==GRANTSHA
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==proof['publication_commit']
env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(ROOT/'src')}
preflight=[sys.executable,str(ROOT/'scripts/resource_applicability/h6_matched_launch_preflight_v4.py'),'--manifest',str(PREP/'sealed_manifest_launch_v4.json'),'--manifest-sha256','c7e4dcc354515dd9cdd1bf656aa6cea0cf323fefd204585fc801a1e3f5adb617','--authorization',str(META/'authorization_launch_v2.json'),'--authorization-sha256',GRANTSHA,'--receipt',str(META/'final_prelaunch_receipt_launch_v2.json')]
subprocess.run(preflight,cwd=ROOT,env=env,check=True)
check=json.loads((META/'final_prelaunch_receipt_launch_v2.json').read_text())
assert check['fresh_grant_validated'] and check['output_parent']['actual_write_probe_passed'] and not check['output_parent']['final_leaf_exists']
with (META/'foreground_owner_source_launch_v2.py').open('xb') as f:f.write(Path(__file__).read_bytes())
command=[sys.executable,str(ROOT/'scripts/resource_applicability/run_track_a_h6_matched_v3.py'),'--execute','--output',str(OUT),'--manifest',str(PREP/'sealed_manifest_launch_v4.json'),'--authorization',str(META/'authorization_launch_v2.json'),'--authorization-sha256',GRANTSHA]
save('launch_once_launch_v2.json',dict(recorded_at_utc=now(),command=command,scientific_runner_invocations=1,retry=False,resume=False,source_commit='5a903aea2492c2d211c1e15e7737df0d50694c12',grant_commit=proof['grant_commit'],grant_sha256=GRANTSHA,owner_pid=os.getpid(),owner_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),transport='foreground maintained execution session',phase_A_index=447,phase_wall_cap=None,total_wall_cap=None,next_stage_authorized=False))
with (META/'runner_stdout_launch_v2.log').open('xb') as log:
 process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 save('process_start_observation_launch_v2.json',dict(recorded_at_utc=now(),pid=process.pid,proc_start_ticks=start_ticks(process.pid),owner_pid=os.getpid(),command=command,scientific_runner_invocations=1))
 print(json.dumps(dict(status='ONE_SHOT_H6_RUNNER_STARTED',pid=process.pid,owner_pid=os.getpid(),grant_commit=proof['grant_commit'])),flush=True)
 code=process.wait()
save('runner_exit_launch_v2.json',dict(recorded_at_utc=now(),pid=process.pid,exit_code=code,scientific_runner_invocations=1,retry=False,resume=False,mandatory_stop=True,next_stage_authorized=False,terminal_exists=(OUT/'execution_terminal.json').exists()))
print(json.dumps(dict(status='ONE_SHOT_H6_RUNNER_EXITED',exit_code=code,terminal_exists=(OUT/'execution_terminal.json').exists())),flush=True)
