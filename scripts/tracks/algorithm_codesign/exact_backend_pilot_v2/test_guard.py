"""Acceptance specification, frozen before execution. One run per case, no retries."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from guard import output_bytes

HERE=Path(__file__).resolve().parent
PRIVATE=Path('/tmp/ra-d0-v4-exact-backend-pilot-v2-20261009')
ROOT=PRIVATE/'phase_a'
START=json.loads((PRIVATE/'start.json').read_text())
DEADLINE=START['start_epoch']+3600
RESULTS=[]

def check(name,ok,details=None):
    RESULTS.append({'test':name,'PASS':bool(ok),'details':details})
    if not ok:raise AssertionError(name)

def case(name,mode,wall=3,rss=256*1024**2,output=8*1024**2,remaining=None):
    assert time.time()<DEADLINE
    assert output_bytes([PRIVATE])<64*1024**2
    root=ROOT/name;root.mkdir(parents=True,exist_ok=False)
    spec={'id':name,'ledger':str(root/'ledger'),'stop_path':str(root/'STOP.json'),
          'command':[sys.executable,'-B',str(HERE/'process_fixture.py'),mode,str(root/'generated')],
          'cwd':str(root),'stdout':str(root/'stdout.txt'),'stderr':str(root/'stderr.txt'),
          'TMPDIR':str(root/'tmp'),'pilot_deadline_epoch':DEADLINE if remaining is None else time.time()+remaining,
          'wall_seconds':wall,'RSS_bytes':rss,'address_space_bytes':1536*1024**2,
          'output_bytes':output,'output_roots':[str(root),str(root/'generated')],'sample_seconds':.02}
    path=root/'spec.json';path.write_text(json.dumps(spec,indent=2)+'\n')
    proc=subprocess.run([sys.executable,'-B',str(HERE/'guard.py'),str(path)],capture_output=True,text=True,timeout=12)
    result=json.loads((root/'ledger'/f'{name}.result.json').read_text())
    result['supervisor_exit_code']=proc.returncode
    for sample in result['samples']:
        ids=[v['pid'] for v in sample['parts']]
        check(f'{name}:unique_PID_sum',len(ids)==len(set(ids)) and sample['RSS_sum_bytes']==sum(v['RSS_bytes'] for v in sample['parts']))
    check(f'{name}:no_remaining_process',result['residual_processes']==0 and not result.get('termination_failed',False))
    for pid in result['known_target_PIDs']:
        check(f'{name}:PID_{pid}_reaped',not Path(f'/proc/{pid}').exists())
    return root,result

def execute():
    root,r=case('single','normal')
    check('single_child_normal_exit',r['failure'] is None and r['returncode']==0 and len(r['known_target_PIDs'])==1)
    check('direct_child_CPU_accounted',r['CPU_seconds']>=.10)
    root,r=case('grandchild','grandchild')
    check('child_and_grandchild_observed',r['failure'] is None and len(r['known_target_PIDs'])==2)
    check('reaped_grandchild_CPU_accounted',r['CPU_seconds']>=.10)
    root,r=case('multiple','multiple')
    check('multiple_children_observed',r['failure'] is None and len(r['known_target_PIDs'])==3)
    check('multiple_CPU_accounted',r['CPU_seconds']>=.20)
    root,r=case('orphan','orphan')
    check('orphan_grandchild_adopted_and_reaped',r['failure'] is None and len(r['wait4_records'])==2)
    check('orphan_CPU_accounted',r['CPU_seconds']>=.10)
    root,r=case('wall','sleep',wall=.35)
    check('wall_timeout',r['failure']=='WALL_CAP' and r['wall_seconds']<2)
    root,r=case('rss','memory',rss=48*1024**2)
    check('RSS_cap_trigger',r['failure']=='RSS_CAP' and r['peak_RSS_sum_bytes']>=48*1024**2)
    root,r=case('output','output',output=512*1024)
    check('aggregate_output_cap_trigger',r['failure']=='OUTPUT_CAP' and r['peak_output_bytes']>=512*1024)
    root,r=case('kill_tree','kill_tree',wall=.5)
    check('SIGKILL_entire_tree',r['failure']=='WALL_CAP' and len(r['wait4_records'])==2)
    check('killed_grandchild_CPU_accounted',r['CPU_seconds']>=.15)
    # Attempted relaunch is refused before fork; the original ledger remains byte-identical.
    before=(root/'ledger/kill_tree.result.json').read_bytes()
    refused=subprocess.run([sys.executable,'-B',str(HERE/'guard.py'),str(root/'spec.json')],capture_output=True,text=True,timeout=3)
    payload=json.loads(refused.stdout)
    check('retry_after_STOP_refused',not payload['launched'] and payload['failure']=='RETRY_OR_AFTER_STOP_REFUSED')
    check('original_execution_ledger_unchanged',before==(root/'ledger/kill_tree.result.json').read_bytes())
    root,r=case('remaining_wall','sleep',wall=3,remaining=.35)
    check('total_pilot_wall_inherited',r['failure']=='WALL_CAP' and 0<r['effective_wall_seconds']<=.35 and r['wall_seconds']<2)
    root,r=case('expired','normal',remaining=-.01)
    check('expired_pilot_refuses_launch',not r['launched'] and r['failure']=='PILOT_WALL_CAP')
    # Decoy is a sibling of the guard supervisor, intentionally outside the target.
    decoy=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(4)'])
    try:
        root,r=case('outside_process','normal')
        check('unrelated_process_not_counted',all(decoy.pid not in [v['pid'] for v in s['parts']] for s in r['samples']))
        check('unrelated_process_not_killed',decoy.poll() is None)
    finally:
        decoy.terminate();decoy.wait()
    directory=ROOT/'inode_dedup';directory.mkdir()
    file=directory/'a';file.write_bytes(b'X'*1234);os.link(file,directory/'b')
    check('overlapping_roots_and_hardlinks_counted_once',output_bytes([directory,file,directory])==1234)
    check('global_caps_preserved',time.time()<DEADLINE and output_bytes([PRIVATE])<64*1024**2)

if __name__=='__main__':
    started=time.time();failure=None
    try:execute()
    except Exception as e:failure=f'{type(e).__name__}: {e}'
    result={'classification':'GUARD_V2_PASS' if failure is None else 'GUARD_V2_REVISION_REQUIRED',
            'PASS':sum(r['PASS'] for r in RESULTS),'FAIL':sum(not r['PASS'] for r in RESULTS),
            'checks':RESULTS,'failure':failure,'wall_seconds':time.time()-started,
            'test_runs':1,'retries':0,'backend_calls':0,
            'CPU_guarantee':'wait4 reports terminated child CPU including already waited descendants; adopted orphans are separately reaped',
            'RSS_guarantee':'UNIQUE_PID_RSS_SUM_CONSERVATIVE sampled at 20 ms; shared pages and between-sample peaks remain limitations'}
    (PRIVATE/'phase_a_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}))
    sys.exit(0 if failure is None else 1)
