"""Actual-host artificial preparation/Q3 acceptance for the RomuTail controller.

The ordinary schedule entry, durable queue, stage worker, CAS, Q3 adapter and
SSH mailboxes are reused. Scientific leaves are labelled synthetic subprocesses.
No molecular input, real compiler, scientific Q3 or production queue is opened.
"""
import argparse
import base64
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
import check_rpe_distributed_q3 as F
import check_rpe_distributed_preparation as P
import check_pr_width_search as W
from trottertracks.rpe_q3_coordinator import atomic_json as put,digest
from trottertracks import rpe_q3_coordinator as Q
from trottertracks.rpe_q3_store import JobStore
from trottertracks.rpe_preparation_queue import normalize_preparation_request
from trottertracks.rpe_preparation_worker import current_preparation_fingerprint

spec=importlib.util.spec_from_file_location('_romutail_direct_relay',ROOT/'scripts/run_rpe_q3_relay.py')
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
HOSTS=['local','server2','local0-wsl']


def read(path):return json.loads(Path(path).read_text())


def prepare(directory,deployment):
    directory=Path(directory).resolve()
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if deployment['source_commit']!=source:raise ValueError('Artificial deployment SOURCE differs from controller HEAD')
    port=F.free_port()
    F.prepare(directory,port=port,total_slots=3,local_slots=1,sleep_seconds=.5)
    problem=read(directory/'PROBLEM.json')
    problem['problem']=dict(system='ARTIFICIAL_ROMUTAIL_THREE_HOSTS',hamiltonian_data=dict(expected=dict(fragment_count=10)))
    for method in problem['methods'].values():method['family']='PR'
    put(directory/'PROBLEM.json',problem)
    profiles={}
    for host in HOSTS:
        location=deployment['hosts'][host]
        profile=read(location['profile_template'])if location.get('profile_template')else{}
        profile.update(host_id=host,root=location['root'],interpreter=location['python'],slots=1,
            work_dir=location['directory'].rstrip('/')+'/spool',
            artifact_cache_directory=location['directory'].rstrip('/')+'/artifacts',
            prefix_cache_directory=location['directory'].rstrip('/')+'/prefix',
            slot_policy=dict(mode='auto',initial_slots=1,max_slots=1))
        profile.setdefault('dp_workers',1)
        profile.setdefault('route_views',{})
        profiles[host]=profile
        put(directory/('PROFILE_'+host+'.json'),profile)
    round=read(directory/'ROUND.json')
    round.update(round_id='ARTIFICIAL_ROMUTAIL_REAL_THREE_HOSTS',
        artificial_preparation=dict(compile_host='server2',audit_host='local0-wsl'))
    round['search_overrides']['requested_widths']={'PR_PAULI':[.1],'PR_DF':[.1]}
    round['search_overrides']['lookahead']=1
    round['executor'].pop('max_concurrent_jobs',None)
    distributed=round['executor']['distributed']
    distributed.update(local_worker=profiles['local'],affinity_wait_seconds=0,
        auto_allocation=dict(discovery_grace_seconds=0,priority_groups=[['local','server2','server1'],['local0-wsl']]),
        preparation=dict(budgets=dict(signal=100,compile=100,audit=100)),
        host_admission=str(directory/'HOST_ADMISSION.json'))
    distributed.pop('required_hosts',None)
    if deployment.get('computation_source'):
        distributed.update(computation_source=deployment['computation_source'],
            computation_root=deployment['hosts']['local']['root'],
            worker_script=str(Path(deployment['hosts']['local'].get('worker_root',deployment['hosts']['local']['root']))/'scripts/run_rpe_q3_worker.py'))
    put(directory/'ROUND.json',round)
    put(directory/'HOST_ADMISSION.json',dict(hosts={'server1':dict(accepting=False,optional=True),
        'local0-wsl':dict(accepting=False,optional=True)}))
    workers=[]
    for host in ['server2','local0-wsl']:
        location=deployment['hosts'][host]
        worker={key:deepcopy(value)for key,value in location.items()if key not in('directory','python','profile_template','worker_root')}
        worker.update(host_id=host,python=location['python'],root=location['root'],
            profile=location['directory'].rstrip('/')+'/PROFILE.json',priority_group=1 if host=='local0-wsl'else 0)
        worker['root']=location.get('worker_root',location['root'])
        workers.append(worker)
    workers.append(dict(host_id='server1',enabled=False,disabled_reason='User reports server1 stopped; no connection attempted'))
    put(directory/'RELAY.json',dict(state_directory=str(directory/'relay-state'),external_tunnel=True,
        controller=dict(local_port=port),workers=workers,poll_seconds=.05,request_timeout_seconds=30,
        retry_backoff_max_seconds=.3))
    put(directory/'DEPLOYMENT.json',deployment)
    return dict(status='PREPARED',directory=str(directory),profile_files=[str(directory/('PROFILE_'+h+'.json'))for h in HOSTS],
        scientific_H4_calls=0,server1_contacted=False)


def fixture_driver(directory):
    directory=Path(directory)
    relay=R.Relay(directory/'RELAY.json')
    workers={worker['host_id']:worker for worker in relay.config['workers'] if worker.get('enabled',True)}
    deployment=read(directory/'DEPLOYMENT.json')
    cache={};cache_lock=threading.Lock()

    def fingerprint(host,assignment):
        key=(host,assignment['job_id'])
        with cache_lock:
            if key in cache:return cache[key]
        if host=='local':
            value=current_preparation_fingerprint(assignment,read(directory/'PROFILE_local.json'))
        else:
            response,_=relay.remote(workers[host],'readiness',dict(challenges=[assignment]))
            if response.get('readiness_errors'):raise ValueError('Actual-host artificial metadata check failed: '+json.dumps(response['readiness_errors']))
            value=response['host_fingerprints'][assignment['job_id']]
        with cache_lock:cache[key]=value
        return value

    class Completion(W.ArtificialCompletion):
        def __init__(self,root):super().__init__(root);self.lock=threading.RLock()
        def bind_stage_executor(self,runtime):self.runtime=runtime;return self
        def save(self):
            with self.lock:super().save()
        def ingest_price_results(self):return None
        def complete(self,req):
            cid=W.configuration(req['method'],req['design']['delta_pf'],req['design']['kappa'],req['design']['L_D'])['configuration_id']
            def submit(stage,name,hosts,deps=(),depinputs=None):
                command=['@PYTHON@','-B','@SOURCE@/scripts/artificial_preparation_leaf.py','--stage',stage,
                    '--output','@WORK@/products','--sleep',str(45 if stage=='signal'else 3)]
                if stage=='audit':command+=['--inputs','@INPUTS@']
                request=normalize_preparation_request(dict(route=dict(kind='preparation'),method='ARTIFICIAL',
                    preparation=dict(stage=stage,provider='ARTIFICIAL_'+stage,artificial=True,
                        command=command,outputs=['products'],source=(Q.source_manifest(
                            Path(deployment['preparation_root']),deployment['preparation_source'])
                            if deployment.get('preparation_source')else self.runtime.executor.source_manifest),
                        dependencies=list(deps),dependency_inputs=depinputs or{},identity=name)))
                assignment=dict(request=request,request_sha256=digest(request),job_id=request['request_id'],source_manifest=request['preparation']['source'])
                facts={host:fingerprint(host,assignment)for host in hosts}
                eligibility=dict(request_sha256=digest(request),eligible_host_ids=hosts,
                    host_fingerprints={host:digest(value)for host,value in facts.items()})
                return self.runtime.executor.submit_preparation(request,approved=True,budget=100,
                    eligibility=eligibility,contract={'ARTIFICIAL':True})
            signal_job=submit('signal',cid,['local','server2'])
            compiled=submit('compile','ONE_SHARED_CALIBRATION',['server2'])
            audit=submit('audit','ONE_SHARED_AUDIT',['local0-wsl'],deps=[compiled],depinputs={compiled:{
                'products/circuit.txt':'compiled/circuit.txt','products/receipt.json':'compiled/receipt.json'}})
            for job in [signal_job,compiled,audit]:
                result=self.runtime.executor.preparation_result(job,wait=True)
                if result['status']!='PREPARATION_COMPLETE':raise AssertionError(result)
            with self.lock:
                self.state['price_received']=True
                return super().complete(req)
        def install_snapshot(self,context,problem,keys):
            with self.lock:records=deepcopy([row for row in self.state['records'].values()if row.get('key')in keys])
            routes=W.ArtificialCompletion.install(self,context,problem,records)
            for candidate in context['candidates']:candidate['caps']=[1]*12+[2]
            for values in routes.values():
                for rule in values:rule['artificial']['sleep_seconds']=1.
            return routes,records
    class Driver(W.ArtificialDriver):
        def completion_api(self):
            if self.completion is None:self.completion=Completion(self.run)
            return self.completion
    return Driver


def controller(directory,*,plan_only=False,control_run=None,batch=32,max_cycles=3):
    import run_rpe_schedule_search as E
    directory=Path(directory).resolve()
    if read(directory/'ROUND.json')['artificial_fixture']['only']is not True:raise ValueError('Only explicit artificial fixtures allowed')
    inputs=Path(control_run).resolve()if control_run else directory
    round_path=inputs/('ROUND.requested.json'if plan_only and control_run else'ROUND.json')
    if not str(read(inputs/'PROBLEM.json').get('problem',{}).get('system','')).startswith('ARTIFICIAL'):
        raise ValueError('Only artificial Gateway problem inputs are accepted')
    if read(round_path).get('artificial_fixture',{}).get('only')is not True:
        raise ValueError('Gateway round must preserve artificial-only declaration')
    run=(inputs/'plan-only')if plan_only else(directory/'run')
    if not plan_only:
        put(directory/'CONTROL_INPUTS.json',dict(directory=str(inputs),round=str(round_path)))
    clear_mirrored_stop(directory,inputs,run,explicit_control=control_run is not None,plan_only=plan_only)
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in()).throw(KeyboardInterrupt()))
    watching=threading.Event()
    def copy_stop():
        while not watching.wait(.1):
            if control_run and(inputs/'STOP_REQUESTED.json').exists():
                run.mkdir(parents=True,exist_ok=True)
                put(run/'STOP_REQUESTED.json',read(inputs/'STOP_REQUESTED.json'))
                return
    monitor=threading.Thread(target=copy_stop,daemon=True)
    if not plan_only:monitor.start()
    command=['--problem',str(inputs/'PROBLEM.json'),'--round',str(round_path),
        '--run-dir',str(run),'--max-cycles',str(max_cycles),'--batch',str(batch)]
    if plan_only:command.append('--plan-only')
    fixture_spec=read(round_path)
    if fixture_spec.get('artificial_pbs_acceptance'):
        import check_rpe_pbs as PBS
        driver=PBS.driver_type(F,preparation_source=fixture_spec.get('artificial_preparation_source'),
                               pf_fixture=fixture_spec.get('artificial_pf_preparation'),
                               all_methods=fixture_spec.get('artificial_pbs_all_methods', False))
    else:
        driver=(F.fixture_driver()if fixture_spec.get('artificial_mode')=='Q3_ONLY'else fixture_driver(directory))
    try:E.main(command,driver_factory=driver)
    finally:
        watching.set()
        if not plan_only:monitor.join(timeout=1)
    if not plan_only:put(directory/'CONTROLLER_COMPLETE.json',dict(status='COMPLETE',scientific_H4_calls=0))


def clear_mirrored_stop(directory,control,run,*,explicit_control,plan_only=False):
    """An explicit Gateway resume may clear only its copied artificial stop."""
    marker=run/'STOP_REQUESTED.json'
    if plan_only or not explicit_control or(control/'STOP_REQUESTED.json').exists()or not marker.exists():return
    if read(directory/'ROUND.json').get('artificial_fixture',{}).get('only')is not True:
        raise ValueError('Only the artificial mirrored user stop may be cleared')
    value=read(marker)
    if value.get('reason')!='USER_REQUESTED_STOP':return
    archive=run/'STOP_REQUESTED_BEFORE_GATEWAY_RESUME.json'
    if archive.exists():archive=run/('STOP_REQUESTED_BEFORE_GATEWAY_RESUME_'+str(time.time_ns())+'.json')
    marker.replace(archive)


class RecoveryRelay(R.Relay):
    def __init__(self,directory):
        self.fixture=Path(directory)
        super().__init__(self.fixture/'RELAY.json')
    def remote(self,worker,action,value=None):
        if worker['host_id']=='local0-wsl'and(self.fixture/'OUTAGE_STARTED.json').exists()and not(self.fixture/'ALLOW_RECOVERY').exists():
            raise ConnectionError('ARTIFICIAL_LOCAL0_CONNECTION_LOST; actual WSL supervisor keeps running')
        result=super().remote(worker,action,value)
        if(worker['host_id']=='local0-wsl'and action=='submit'and
                (value.get('request',{}).get('preparation')or{}).get('stage')=='audit'and
                not(self.fixture/'OUTAGE_STARTED.json').exists()):
            put(self.fixture/'OUTAGE_STARTED.json',dict(job_id=value['job_id'],attempt_id=value['attempt_id'],host_id='local0-wsl',epoch=time.time()))
        return result


def relay_run(directory):RecoveryRelay(Path(directory).resolve()).run()


def wait_for(predicate,reason,seconds=300):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        result=predicate()
        if result:return result
        time.sleep(.1)
    raise TimeoutError(reason)


def synthetic_participation(worker,enabled):
    path=worker.get('participation_wsl_file')
    if not path or '.romutail-controller-acceptance/'not in path:
        raise ValueError('Only the dedicated synthetic WSL participation file may be changed')
    # The dedicated profile directory exists before acceptance starts. Passing
    # JSON to tee avoids Windows native argv rewriting a quoted Python -c body.
    remote=R.ssh_wsl_command(worker,['tee',path],'fixture-participation')
    command=worker.get('ssh_command',['ssh'])+['-o','BatchMode=yes','-o','ConnectTimeout=10',worker['ssh_alias'],remote]
    result=subprocess.run(command,input=json.dumps(dict(artificial_only=True,accepting_new_jobs=enabled))+chr(10),
        text=True,encoding='utf-8',errors='replace',capture_output=True,timeout=30,check=True)
    return json.loads(result.stdout)


def snapshot(directory):
    store=JobStore(directory/'run/distributed/QUEUE.sqlite3')
    rows=store.rows();events=store.events()
    return dict(jobs=[dict(job_id=row['job_id'],attempt_id=row['attempt_id'],completion_sha256=row['completion_sha256'])for row in rows],
        event_counts={name:sum(event['event']==name for event in events)for name in ['ASSIGN','ACCEPT','ADOPT']},
        launch_markers=F.launch_markers(directory))


def synthetic_admission(directory,enabled):
    """Update the synthetic run's actual policy, including Gateway's copy."""
    directory=Path(directory).resolve()
    policy=dict(hosts={'server1':dict(accepting=False,optional=True),
        'local0-wsl':dict(accepting=enabled,optional=True)})
    paths={directory/'HOST_ADMISSION.json'}
    locator=directory/'CONTROL_INPUTS.json'
    round_path=Path(read(locator)['round']).resolve()if locator.exists()else directory/'run/ROUND.json'
    if round_path.exists():
        if not round_path.is_relative_to(directory):
            raise ValueError('Synthetic control inputs must stay in their owned fixture directory')
        current=read(round_path)['executor']['distributed'].get('host_admission')
        if isinstance(current,str):
            current=Path(current).resolve()
            if not current.is_relative_to(directory):
                raise ValueError('Synthetic host admission must stay in its owned fixture directory')
            paths.add(current)
    for path in paths:put(path,policy)


def exercise(directory):
    """Drive actual SSH faults while the Windows client owns controller start.

    This process starts only the synthetic run's relay. It does not launch the
    controller or change the real local0-wsl SETTINGS/Pause file.
    """
    directory=Path(directory).resolve()
    plain=R.Relay(directory/'RELAY.json')
    wsl=next(worker for worker in plain.config['workers']if worker['host_id']=='local0-wsl')
    synthetic_participation(wsl,False)
    worker=None
    observation_stop=threading.Event()
    observation_thread=None
    with(directory/'relay-exercise.log').open('ab')as log:
        try:
            worker=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--relay',str(directory)],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            database=directory/'run/distributed/QUEUE.sqlite3'
            wait_for(lambda:database.exists(),'Windows-started synthetic controller did not create queue')
            store=JobStore(database)
            def observe_parallel():
                while not observation_stop.is_set():
                    try:
                        running=[row for row in store.rows()if row['state']=='RUNNING']
                        signals=[row for row in running if row['method']=='PREPARATION:signal']
                        compiles=[row for row in running if row['method']=='PREPARATION:compile']
                        if signals and compiles and not(directory/'PARALLEL_RUNNING.json').exists():
                            put(directory/'PARALLEL_RUNNING.json',dict(controller_observed_epoch=time.time(),
                                source='SAME_CONTROLLER_CLOCK_RUNNING_STATE_SNAPSHOT',
                                jobs=[dict(job_id=row['job_id'],host_id=row['host_id'],method=row['method'])for row in signals+compiles]))
                    except Exception:pass
                    observation_stop.wait(.05)
            observation_thread=threading.Thread(target=observe_parallel,daemon=True)
            observation_thread.start()
            def initial_started():
                rows=store.rows()
                return all(any(row['host_id']==host and row['started_epoch']is not None for row in rows)for host in ['local','server2'])
            wait_for(initial_started,'Primary actual hosts did not start artificial work')
            synthetic_participation(wsl,True)
            synthetic_admission(directory,True)
            wait_for(lambda:(directory/'OUTAGE_STARTED.json').exists(),'WSL audit was not delivered')
            outage=read(directory/'OUTAGE_STARTED.json')
            before=sum(row['host_id']in('local','server2')and row['state']=='ACCEPTED'for row in store.rows())
            def completed_untransferred():
                status,_=R.Relay.remote(plain,wsl,'status')
                return next((attempt for attempt in status['attempts']if attempt['job_id']==outage['job_id']and attempt['state']=='COMPLETED_PENDING_TRANSFER'),None)
            preserved=wait_for(completed_untransferred,'Actual WSL did not complete and retain its artifact after disconnect')
            assert store.row(outage['job_id'])['state']!='ACCEPTED'
            digest_before=hashlib.sha256(preserved['files']['OUTCOME.json'].encode()).hexdigest()
            wait_for(lambda:sum(row['host_id']in('local','server2')and row['state']=='ACCEPTED'for row in store.rows())>before,
                'Primary actual hosts made no progress during local0 connection outage')
            synthetic_participation(wsl,False)
            put(directory/'ALLOW_RECOVERY',dict(artificial_only=True))
            wait_for(lambda:store.row(outage['job_id'])['state']=='ACCEPTED','Saved WSL result did not recover with same attempt')
            paused,_=R.Relay.remote(plain,wsl,'status')
            assert paused.get('accepting_new_jobs')is False and paused['slots']==0
            # Paused host's completion is recovered; it receives no new work.
            assigned=sum(row['host_id']=='local0-wsl'for row in store.rows())
            time.sleep(.3)
            assert assigned==sum(row['host_id']=='local0-wsl'for row in store.rows())
            synthetic_participation(wsl,True)
            wait_for(lambda:(directory/'CONTROLLER_COMPLETE.json').exists(),'Windows-started ordinary synthetic search did not complete')
            row=store.row(outage['job_id'])
            central=directory/'run/jobs'/row['method']/row['job_id']/'OUTCOME.json'
            digest_after=hashlib.sha256(base64.b64encode(central.read_bytes())).hexdigest()
            assert digest_after==digest_before
            put(directory/'FAULT_CHECKS.json',dict(remote_job_completed_while_local0_disconnected=True,
                primary_hosts_progressed_during_local0_disconnect=True,saved_outcome_recovered_unchanged=True,
                recovered_attempt_unchanged=row['attempt_id']==outage['attempt_id'],
                paused_wsl_offers_zero_and_still_transfers=True,synthetic_pause_resumed=True,
                signal_compile_parallel_observed_by_controller=(directory/'PARALLEL_RUNNING.json').exists()))
            snap=snapshot(directory)
            put(directory/'BEFORE_RESUME.json',{key:value for key,value in snap.items()if key!='launch_markers'})
            put(directory/'BEFORE_RESUME_LAUNCH_SHA256.json',{key:hashlib.sha256(value).hexdigest()for key,value in snap['launch_markers'].items()})
            return check(directory)
        finally:
            observation_stop.set()
            if observation_thread is not None:observation_thread.join(timeout=1)
            synthetic_participation(wsl,False)
            (plain.directory/'STOP').touch()
            if worker is not None:
                try:worker.wait(timeout=15)
                except subprocess.TimeoutExpired:worker.terminate();worker.wait(timeout=10)


def verify_resume(directory):
    directory=Path(directory).resolve();before=read(directory/'BEFORE_RESUME.json');after=snapshot(directory)
    marker_hashes={key:hashlib.sha256(value).hexdigest()for key,value in after['launch_markers'].items()}
    checks=dict(completed_resume_new_launches_zero=marker_hashes==read(directory/'BEFORE_RESUME_LAUNCH_SHA256.json'),
        completed_resume_assignment_acceptance_adoption_unchanged=after['event_counts']==before['event_counts'],
        completed_resume_attempts_and_artifacts_unchanged=after['jobs']==before['jobs'])
    if not all(checks.values()):raise AssertionError(checks)
    old=read(directory/'FAULT_CHECKS.json');old.update(checks);put(directory/'FAULT_CHECKS.json',old)
    return check(directory)


def check(directory):
    directory=Path(directory).resolve();store=JobStore(directory/'run/distributed/QUEUE.sqlite3')
    rows=store.rows();events=store.events();prep=[row for row in rows if row['method'].startswith('PREPARATION:')]
    counts={name:Counter(event['job_id']for event in events if event['event']==name)for name in ['ASSIGN','ACCEPT','ADOPT']}
    hosts={host:dict(assigned=sum(row['host_id']==host for row in rows),
        accepted=sum(row['host_id']==host and row['state']=='ACCEPTED'for row in rows),
        adopted=sum(row['host_id']==host and row['adopted_epoch']is not None for row in rows))for host in HOSTS}
    interval=[]
    for path in (directory/'run/jobs').glob('PREPARATION*/*/JOB_STAGES.jsonl'):
        stages=[json.loads(line)for line in path.read_text().splitlines()]
        starts=[s for s in stages if s.get('event')=='PREPARATION_BODY_START'];ends=[s for s in stages if s.get('event')=='PREPARATION_BODY_END']
        assert len(starts)==len(ends)==1,'Synthetic body started more than once'
        interval.append(dict(stage=starts[0]['stage'],host=read(path.parent/'LAUNCH.json')['host_id'],
            start_epoch=starts[0]['epoch'],end_epoch=ends[0]['epoch'],clock_id=starts[0]['clock_id']))
    # Only same-clock intervals are comparable. Different-machine timestamps
    # alone are not used to claim overlap or a one-way transfer duration.
    explicit=read(directory/'FAULT_CHECKS.json')if(directory/'FAULT_CHECKS.json').exists()else{}
    checks=dict(all_three_actual_hosts_adopted=all(v['adopted']>0 for v in hosts.values()),
        all_jobs_accepted_and_adopted=bool(rows)and all(row['state']=='ACCEPTED'and row['adopted_epoch']is not None for row in rows),
        shared_compile_once=sum(row['method']=='PREPARATION:compile'for row in prep)==1,
        shared_audit_once=sum(row['method']=='PREPARATION:audit'for row in prep)==1,
        different_compile_audit_hosts={row['host_id']for row in prep if row['method']=='PREPARATION:compile'}!={row['host_id']for row in prep if row['method']=='PREPARATION:audit'},
        no_duplicate_assignment_acceptance_adoption=all(all(n==1 for n in count.values())for count in counts.values()),
        server1_never_assigned=not any(row['host_id']=='server1'for row in rows),**explicit)
    value=dict(status='PASS'if all(checks.values())else'FAIL',actual_OS_host_count=3,host_names_are_simulated=False,
        scientific_H4_calls=0,source_commit=read(directory/'DEPLOYMENT.json')['source_commit'],checks=checks,
        by_host=hosts,preparation_count=len(prep),q3_count=len(rows)-len(prep),preparation_intervals=interval,
        tasks=[dict(job_id=row['job_id'],method=row['method'],host_id=row['host_id'],attempt_id=row['attempt_id'])for row in rows])
    put(directory/'ACCEPTANCE.json',value);return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    commands=p.add_mutually_exclusive_group(required=True)
    commands.add_argument('--prepare',type=Path);commands.add_argument('--controller',type=Path)
    commands.add_argument('--relay',type=Path);commands.add_argument('--check',type=Path)
    commands.add_argument('--exercise',type=Path);commands.add_argument('--verify-resume',type=Path)
    p.add_argument('--deployment',type=Path)
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--control-run',type=Path)
    p.add_argument('--batch',type=int,default=32);p.add_argument('--max-cycles',type=int,default=3)
    a=p.parse_args()
    if a.prepare:
        if not a.deployment:p.error('--deployment is required with --prepare')
        result=prepare(a.prepare,read(a.deployment))
    elif a.controller:return controller(a.controller,plan_only=a.plan_only,control_run=a.control_run,batch=a.batch,max_cycles=a.max_cycles)
    elif a.relay:return relay_run(a.relay)
    elif a.exercise:result=exercise(a.exercise)
    elif a.verify_resume:result=verify_resume(a.verify_resume)
    else:result=check(a.check)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
