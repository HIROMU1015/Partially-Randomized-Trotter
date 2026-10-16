#!/usr/bin/env python3
"""Single-use supervisor for the adopted frozen worker. Never retry or resume."""
import argparse,hashlib,json,os,resource,subprocess,sys,time,traceback
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3]
C='eea84105cb28f1aebdf5381cfd292d9e4e7eb478'
CONTRACT='artifacts/track_b_m1s_finite_signal_v2_completion/2026-10-16/attempt_01/execution_contract_adopted.json'
AUTH='artifacts/track_b_m1s_finite_signal_v2_execution_authorization/2026-10-16/completion_attempt_01/execution_authorization.json'
PREFIX=Path('/tmp')/('m1s_finite_signal_v2_completion_once_'+C)

def write_new(path,value):
    raw=json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False).encode()+b'\n'
    with path.open('xb') as out:out.write(raw);out.flush();os.fsync(out.fileno())

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--authorization-commit',required=True);args=parser.parse_args()
    if len(args.authorization_commit)!=40 or any(c not in '0123456789abcdef' for c in args.authorization_commit):raise ValueError('full authorization SHA required')
    A=args.authorization_commit
    # Stable C-keyed exclusive reservation protects against a different A being
    # used to retry this same adopted contract. Reserve before any numerical work.
    marker=Path(str(PREFIX)+'.launch.json')
    write_new(marker,{'contract_commit':C,'authorization_commit':A,'authorization_path':AUTH,
      'attempt':1,'one_shot_consumed':True,'automatic_retry':False,'pid':os.getpid(),'started_unix':time.time(),'mandatory_STOP':True})
    completion={'contract_commit':C,'authorization_commit':A,'one_shot_consumed':True,'run_attempts':1,'automatic_retry':False,'mandatory_STOP':True}
    started=time.monotonic();child=None
    try:
        cfg=json.loads((ROOT/CONTRACT).read_bytes())
        for key,value in cfg['resources']['thread_environment'].items():os.environ[key]=value
        sys.path.insert(0,str(ROOT/'src'))
        from trottertracks.algorithm_codesign.m1s_finite_signal_v2.control import authorize
        permit=authorize(ROOT,CONTRACT,A,AUTH)
        approval=ROOT/json.loads((ROOT/AUTH).read_bytes())['user_approval_record']
        if json.loads(approval.read_bytes()).get('explicit_user_scientific_execution_approval') is not True:raise PermissionError('explicit separate approval record')
        cpu=min(os.sched_getaffinity(0));limits=cfg['resources']
        command=[sys.executable,'-B',str(ROOT/'scripts/tracks/algorithm_codesign/run_m1s_finite_signal_v2.py'),'run',
          '--contract',CONTRACT,'--authorization-commit',A,'--authorization-path',AUTH]
        write_new(Path(str(PREFIX)+'.preflight.json'),{'status':'PASS_METADATA_AUTHORIZATION_PREFLIGHT',
          'permit_audit':permit.audit,'numerical_modules_loaded':any(x.startswith('numpy') for x in sys.modules),
          'CPU_affinity_before':sorted(os.sched_getaffinity(0)),'selected_CPU':cpu,'thread_environment':limits['thread_environment'],
          'memory_cap_bytes':limits['memory_bytes'],'output_log_cap_bytes':limits['result_bytes'],'GPU':False,
          'wall_time_cap':None,'child_command':command,'source_and_contract_changed':False,'scientific_calculations_before_child':0})
        def child_limits():
            os.sched_setaffinity(0,{cpu})
            for kind,cap in [(resource.RLIMIT_AS,limits['memory_bytes']),(resource.RLIMIT_FSIZE,limits['result_bytes'])]:
                soft,hard=resource.getrlimit(kind);value=min([cap]+[v for v in (soft,hard) if v!=resource.RLIM_INFINITY]);resource.setrlimit(kind,(value,hard))
        console_path=Path(str(PREFIX)+'.console.log')
        with console_path.open('xb') as console:
            child=subprocess.Popen(command,cwd=ROOT,stdout=console,stderr=subprocess.STDOUT,env=os.environ.copy(),preexec_fn=child_limits)
            completion['child_pid']=child.pid
            print(json.dumps({'stage':'ONE_SHOT_CHILD_STARTED','pid':child.pid,'marker':str(marker),'selected_CPU':cpu}),flush=True)
            while child.poll() is None:
                time.sleep(1)
            console.flush();os.fsync(console.fileno())
        completion.update(child_returncode=child.returncode,child_started=True,supervisor_status='CHILD_FINISHED_NO_RETRY')
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.terminate();child.wait()
        completion.update(supervisor_status='TECHNICAL_STOP_NO_RETRY',failure={'type':type(exc).__name__,'reason':str(exc),'traceback':traceback.format_exc()},child_started=child is not None)
    finally:
        completion['elapsed_seconds']=time.monotonic()-started
        write_new(Path(str(PREFIX)+'.completion.json'),completion)
        print(json.dumps(completion,ensure_ascii=False),flush=True)
    return 0 if completion.get('child_returncode')==0 else 2
if __name__=='__main__':raise SystemExit(main())
