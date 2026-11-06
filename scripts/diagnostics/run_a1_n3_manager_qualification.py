"""Own synthetic jobs only. No real N3, engine, RNG or compiler entry point."""
from pathlib import Path
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign import a1_n3_cron_manager as cron
from trottertracks.algorithm_codesign import a1_n3_managed_runtime as managed
from trottertracks.algorithm_codesign import a1_n3_lifecycle_observer as old


def toy_task(case,out):
    out=Path(out);journal=out/f'worker_{os.getpid()}.jsonl'
    old.event(journal,'WORKER_STARTED',ticket='toy-0',synthetic=True)
    with old.traced_call(managed._WORKER_LOG,'toy-0','mock_measurement'):
        old.event(journal,'BIT_SAVED',ticket='toy-0',record={'bit':0},synthetic=True)
        managed.durable_new(out/'BIT_READY',{'synthetic':True})
    if case=='worker_failure':os._exit(23)
    if case in ('stop','parent_failure','manager_failure'):
        while not managed._WORKER_STOP.is_set():time.sleep(.01)
        return {'reason':'synthetic STOP after retained bit'}
    with old.traced_call(managed._WORKER_LOG,'toy-0','mock_price'):
        old.event(journal,'PRICE_SAVED',ticket='toy-0',record={'bit':0},mock_price=11,synthetic=True)
    return {'synthetic':True}


def parent(case,out):
    out=Path(out);runtime=managed.Runtime(out);pool=None;code=0
    try:
        with runtime.signals():
            pool=managed.ManagedPool(runtime,max_workers=1,mp_context=mp.get_context('fork'))
            future=pool.submit(toy_task,case,str(out))
            if case in ('stop','parent_failure','manager_failure'):
                while not (out/'BIT_READY').exists():time.sleep(.01)
                if case=='parent_failure':os.kill(os.getpid(),signal.SIGKILL)
                if case=='stop':runtime.request_stop('synthetic explicit STOP')
            result=future.result()
            if 'reason' in result:code=17
    except BaseException as exc:
        runtime.request_stop(type(exc).__name__);code=17
    finally:
        if pool is not None:pool.shutdown()
        runtime.classify()
    return code


def quiescent(out):
    for _ in range(500):
        live=[]
        for identity in managed.worker_identities(out):
            p=Path('/proc',str(identity['pid']),'stat')
            if not p.exists():continue
            try:state=p.read_text().rsplit(')',1)[1].split()[0]
            except FileNotFoundError:continue
            if state!='Z':live.append(identity['pid'])
        if not live:return True
        time.sleep(.01)
    return False


def suite(specpath):
    spec=json.loads(Path(specpath).read_text());out=Path(spec['output'])
    report={'kind':'artificial_qualification','science_calls':0,'cases':{}}
    identity=old.process_identity(os.getpid())
    assert '/system.slice/cron.service' in identity['cgroup']
    previous=spec['launcher_identity'];now=old.process_identity(previous['pid'])
    assert now.get('start_ticks')!=previous.get('start_ticks') or now.get('boot_id')!=previous.get('boot_id')
    report['submission_client_exited_before_job']=True
    report['cron_identity']=identity
    for case in ('normal','worker_failure','parent_failure','client_disconnect','stop','manager_failure'):
        path=out/case
        if case=='client_disconnect':
            fake=subprocess.Popen([spec['python'],'-c','pass'],start_new_session=True)
            assert fake.wait()==0
        argv=[spec['python'],'-B',str(Path(__file__).resolve()),'--manager',case,str(path)]
        status=cron.record_manager(argv,path,ROOT)
        assert quiescent(path),'owned workers did not stop'
        attempts=json.loads((path/'ATTEMPTS.json').read_text())['trials']
        parent_status=json.loads((path/'PARENT_EXIT.json').read_text()) if (path/'PARENT_EXIT.json').exists() else {'status':'UNKNOWN'}
        worker_codes=json.loads((path/'WORKER_EXIT_CODES.json').read_text()) if (path/'WORKER_EXIT_CODES.json').exists() else {'status':'UNKNOWN: parent wait owner lost'}
        assert attempts.get('toy-0')==('BIT_AND_PRICE_DURABLE' if case in ('normal','client_disconnect') else 'BIT_DURABLE_PRICE_MISSING')
        if case in ('normal','client_disconnect'):
            assert parent_status['exit_code']==0
            assert all(v['exit_code']==0 for v in worker_codes.values())
        elif case=='parent_failure':assert parent_status.get('signal')==signal.SIGKILL
        elif case=='manager_failure':assert status.get('signal')==signal.SIGKILL
        elif case=='worker_failure':assert any(v.get('exit_code')==23 for v in worker_codes.values())
        if case not in ('normal','client_disconnect'):
            assert (path/'STOP_REQUESTED.json').exists()
        if (path/'WORKER_EXIT_CODES.json').exists() and (path/'STOP_REQUESTED.json').exists():
            rows=[json.loads(line) for line in (path/'OBS_parent.jsonl').read_text().splitlines()]
            stop=json.loads((path/'STOP_REQUESTED.json').read_text())['monotonic_ns']
            assert all(stop<r['monotonic_ns'] for r in rows if r['event']=='POOL_DRAIN_START')
        report['cases'][case]={'status':'PASS','manager_wait':status,'parent_wait':parent_status,
            'worker_wait':worker_codes,'attempts':attempts,'owned_workers_quiescent':True}
    duplicate=subprocess.run([spec['python'],'-B',str(Path(__file__).resolve()),'--dispatch',str(specpath)],capture_output=True,text=True)
    assert duplicate.returncode==72
    assert cron.own_crontab() is None
    report['duplicate_dispatch_exit']=72
    report['temporary_crontab_removed_before_application_start']=True
    report['restart_policy']='no; consumed receipt retained, trigger removed'
    report['status']='PASS'
    old.atomic_write_json(out/'QUALIFICATION_RESULT.json',report)
    print(json.dumps(report,indent=2))
    return 0


def main():
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--submit',type=Path);group.add_argument('--dispatch',type=Path)
    group.add_argument('--suite',type=Path);group.add_argument('--manager',nargs=2)
    group.add_argument('--parent',nargs=2)
    args=parser.parse_args()
    if args.submit:
        path=cron.submit_qualification(ROOT,args.submit)
        print(json.dumps({'spec':str(path),'science_authorized':False}));return 0
    if args.dispatch:return cron.dispatch(args.dispatch)
    if args.suite:return suite(args.suite)
    if args.manager:
        case,out=args.manager
        managed.run_parent([sys.executable,'-B',str(Path(__file__).resolve()),'--parent',case,out],
            out,ROOT,simulate_manager_loss=case=='manager_failure')
        return 0
    if args.parent:return parent(*args.parent)


if __name__=='__main__':
    raise SystemExit(main())
