"""Normal-entry preparation -> Q3 acceptance with real artificial subprocesses.

Four names are simulated hosts on one OS. Only the unavailable scientific leaf
and its supplied metadata are fixtures; the ordinary entry/queue/worker/CAS and
Q3 adapter are used. This never opens real H4 inputs or invokes a compiler.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
import check_rpe_distributed_q3 as F
import check_pr_width_search as W
from trottertracks.rpe_q3_coordinator import atomic_json as put,digest
from trottertracks.rpe_preparation_queue import normalize_preparation_request
from trottertracks.rpe_preparation_worker import current_preparation_fingerprint
from trottertracks.rpe_q3_store import JobStore
HOSTS=['server2','server1','local','local0-wsl']


def read(p):return json.loads(Path(p).read_text())


def q3_outcomes(directory):
    values={}
    for file in sorted((Path(directory)/'run').glob('jobs/*/*/OUTCOME.json')):
        request=read(file.parent/'REQUEST.json');result=read(file)
        if request['route']['kind']=='preparation':continue
        assert request['route']['kind']=='artificial'and result.get('artificial')
        key=json.dumps([request['method'],request['configuration_id'],request['N_X'],request['N_Y']],separators=(',',':'))
        values[key]=[result['status'],result.get('MSE_lower'),result.get('MSE_upper')]
    assert values,'No Q3 result'
    return values


def summarize(directory):
    results={name:read(directory/name/'ACCEPTANCE.json')for name in ['first','swapped']}
    assert q3_outcomes(directory/'first')==q3_outcomes(directory/'swapped'),'Host placement changed exact MSE bounds'
    for name,summary in results.items():
        root=directory/name/'run';intervals=[]
        for path in root.glob('jobs/PREPARATION*/*/JOB_STAGES.jsonl'):
            rows=[json.loads(s)for s in path.read_text().splitlines()]
            starts=[r for r in rows if r['event']=='PREPARATION_BODY_START'];ends=[r for r in rows if r['event']=='PREPARATION_BODY_END']
            assert len(starts)==len(ends)==1,'Completed scientific fixture body relaunched'
            assert starts[0]['clock_id']==ends[0]['clock_id']
            launch=read(path.parent/'LAUNCH.json')
            intervals.append(dict(stage=starts[0]['stage'],host=launch['host_id'],clock=starts[0]['clock_id'],
                start=starts[0]['monotonic'],end=ends[0]['monotonic'],end_epoch=ends[0]['epoch']))
        overlap=any(a['stage']=='compile'and b['stage']=='signal'and a['host']!=b['host']and a['clock']==b['clock']
            and min(a['end'],b['end'])>max(a['start'],b['start'])for a in intervals for b in intervals)
        assert overlap,'No independent signal/compile subprocess overlap'
        launches=[read(p)for p in root.glob('jobs/PR_*/*/LAUNCH.json')]
        assert min(r['started']for r in launches)<max(r['end_epoch']for r in intervals),'Q3 waited for every candidate preparation'
        summary.update(signal_compile_parallel=True,ready_candidate_q3_before_all_preparation=True,
            completed_preparation_body_start_count=len(intervals),preparation_intervals=intervals)
        put(directory/name/'ACCEPTANCE.json',summary)
    out=dict(status='PASS',cases=results,host_swaps_exact_bounds_equal=True,scientific_H4_calls=0)
    put(directory/'ACCEPTANCE.json',out);return out


def fixture_driver():
    # Use the published fixture completion directly; production orchestration
    # is in WidthDriver, and this adapter substitutes its scientific leaves.
    class Completion(W.ArtificialCompletion):
        def __init__(self,root):super().__init__(root);self.lock=threading.RLock()
        def bind_stage_executor(self,runtime):self.runtime=runtime;return self
        def save(self):
            with self.lock:super().save()
        def ingest_price_results(self):return None
        def complete(self,req):
            spec=read(self.root.parent/'ROUND.json');case=spec['artificial_preparation']
            if not hasattr(self,'runtime'):raise AssertionError('Normal entry did not bind shared preparation runtime')
            cid=W.configuration(req['method'],req['design']['delta_pf'],req['design']['kappa'],req['design']['L_D'])['configuration_id']
            def submit(stage,name,hosts,deps=(),depinputs=None):
                slow=(stage=='signal'and req['method']=='PR_PAULI'and req['design']['L_D']==2)
                if slow:hosts=['local0-wsl']
                command=['@PYTHON@','-B','@SOURCE@/scripts/artificial_preparation_leaf.py','--stage',stage,
                    '--output','@WORK@/products','--sleep',str(4 if slow else .35 if stage in('signal','compile')else .1)]
                if stage=='audit':command+=['--inputs','@INPUTS@']
                request=dict(route=dict(kind='preparation'),method='ARTIFICIAL',preparation=dict(stage=stage,
                    provider='ARTIFICIAL_'+stage,artificial=True,command=command,outputs=['products'],
                    source=self.runtime.executor.source_manifest,dependencies=list(deps),
                    dependency_inputs=depinputs or{},identity=name))
                request=normalize_preparation_request(request)
                assignment=dict(request=request,request_sha256=digest(request),job_id=request['request_id'],
                    source_manifest=request['preparation']['source'])
                fingerprints={h:current_preparation_fingerprint(assignment,read(self.root.parent/('PROFILE_'+h+'.json')))for h in hosts}
                eligibility=dict(request_sha256=digest(request),eligible_host_ids=hosts,
                    host_fingerprints={h:digest(v)for h,v in fingerprints.items()})
                return self.runtime.executor.submit_preparation(request,approved=True,budget=100,
                    eligibility=eligibility,contract={'ARTIFICIAL':True})
            signal=submit('signal',cid,[h for h in HOSTS if h!=case['compile_host']])
            compiled=submit('compile','ONE_SHARED_CALIBRATION',[case['compile_host']])
            audit=submit('audit','ONE_SHARED_AUDIT',[case['audit_host']],deps=[compiled],depinputs={compiled:{
                'products/circuit.txt':'compiled/circuit.txt','products/receipt.json':'compiled/receipt.json'}})
            for job in [signal,compiled,audit]:
                result=self.runtime.executor.preparation_result(job,wait=True)
                if result['status']!='PREPARATION_COMPLETE':raise AssertionError(result)
            with self.lock:
                # This fixture grants every requested width the same saved
                # synthetic price after its own signal and shared audit.
                self.state['price_received']=True
                return super().complete(req)
        def install_snapshot(self,context,problem,keys):
            with self.lock:records=deepcopy([r for r in self.state['records'].values()if r.get('key')in keys])
            routes=W.ArtificialCompletion.install(self,context,problem,records)
            for c in context['candidates']:c['caps']=[1]*12+[2]
            for values in routes.values():
                for rule in values:rule['artificial']['sleep_seconds']=.1
            return routes,records
    class Driver(W.ArtificialDriver):
        def completion_api(self):
            if self.completion is None:self.completion=Completion(self.run)
            return self.completion
    return Driver


def prepare(directory,swapped=False):
    directory=Path(directory).resolve();F.prepare(directory,port=F.free_port(),total_slots=3,local_slots=1,sleep_seconds=.1)
    spec=read(directory/'ROUND.json');spec['round_id']='ARTIFICIAL_PREPARATION_PIPELINE'
    spec['search_overrides']['requested_widths']={'PR_PAULI':[.1],'PR_DF':[.1]}
    spec['search_overrides']['lookahead']=1
    spec['executor'].pop('max_concurrent_jobs',None)
    dist=spec['executor']['distributed'];dist.update(affinity_wait_seconds=0,
        preparation={'budgets':{'signal':100,'compile':100,'audit':100}},
        host_admission=str(directory/'HOST_ADMISSION.json'))
    # External simulated workers permit a deliberate completed-before-transfer
    # disconnect. The controller continues to own all scientific decisions.
    dist.pop('local_worker',None)
    case=dict(compile_host='server1'if swapped else'local',audit_host='local'if swapped else'server1')
    spec['artificial_preparation']=case
    put(directory/'ROUND.json',spec)
    put(directory/'HOST_ADMISSION.json',dict(hosts={'local0-wsl':dict(accepting=False,optional=True)}))
    for host in HOSTS:put(directory/('PROFILE_'+host+'.json'),F.profile(directory,host))
    return dist['port']


def controller(directory):
    import run_rpe_schedule_search as E
    E.main(['--problem',str(directory/'PROBLEM.json'),'--round',str(directory/'ROUND.json'),
        '--run-dir',str(directory/'run'),'--max-cycles','3','--batch','32'],driver_factory=fixture_driver())


def worker(directory,host):
    from trottertracks.rpe_q3_worker import Worker
    profile=read(directory/('PROFILE_'+host+'.json'));port=read(directory/'ROUND.json')['executor']['distributed']['port']
    instance=Worker(profile,'http://127.0.0.1:'+str(port),poll_seconds=.03)
    original=instance._http
    def transport(endpoint,payload):
        if endpoint=='/v1/complete' and str(payload.get('job_id','')).startswith('prep-'):
            attempts=list((directory/'workers'/host).glob('attempts/*/*/ASSIGNMENT.json'))
            # Payload outcome contains the typed stage; inspect metadata only.
            outcome=json.loads(base64.b64decode(payload['files']['OUTCOME.json']))
            if outcome.get('stage')=='compile'and not(directory/'ALLOW_TRANSFER').exists():
                put(directory/'COMPLETED_BEFORE_TRANSFER.json',dict(host_id=host,job_id=payload['job_id'],attempt_id=payload['attempt_id']))
                raise ConnectionError('ARTIFICIAL_DISCONNECT_AFTER_COMPUTE_BEFORE_TRANSFER')
        return original(endpoint,payload)
    instance.transport=transport
    try:instance.run()
    finally:instance.close()


def run_case(directory,swapped=False):
    directory=Path(directory).resolve();port=prepare(directory,swapped)
    children=[];logs=[];start=time.monotonic();actions=[]
    def spawn(args,name):
        log=(directory/(name+'.log')).open('wb');logs.append(log)
        child=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        children.append(child);return child
    try:
        ctrl=spawn(['--controller',str(directory)],'controller')
        for host in HOSTS:spawn(['--worker',str(directory),'--host',host],host)
        while ctrl.poll() is None:
            if time.monotonic()-start>120:raise TimeoutError('Artificial pipeline controller stalled: '+str(directory))
            marker=directory/'COMPLETED_BEFORE_TRANSFER.json'
            if marker.exists() and not(directory/'ALLOW_TRANSFER').exists():
                record=read(marker);store=JobStore(directory/'run/distributed/QUEUE.sqlite3');row=store.row(record['job_id'])
                assert row['state']!='ACCEPTED'
                actions.append(dict(event='COMPLETED_PENDING_TRANSFER',**record))
                put(directory/'ALLOW_TRANSFER',{'ARTIFICIAL':True})
                put(directory/'HOST_ADMISSION.json',dict(hosts={'local0-wsl':dict(accepting=True,optional=True)}))
                actions.append(dict(event='WSL_JOIN_AFTER_PAUSE'))
            time.sleep(.04)
        if ctrl.returncode:raise AssertionError('Controller failed: '+str(directory/'controller.log'))
        store=JobStore(directory/'run/distributed/QUEUE.sqlite3');rows=store.rows();events=store.events()
        prep=[r for r in rows if r['method'].startswith('PREPARATION:')]
        assert sum(r['method']=='PREPARATION:compile'for r in prep)==1
        assert sum(r['method']=='PREPARATION:audit'for r in prep)==1
        assert {r['host_id']for r in prep if r['method']=='PREPARATION:compile'}!={r['host_id']for r in prep if r['method']=='PREPARATION:audit'}
        assert all(r['state']=='ACCEPTED'and r['adopted_epoch']is not None for r in rows)
        counts={name:Counter(e['job_id']for e in events if e['event']==name)for name in ['ASSIGN','ACCEPT','ADOPT']}
        assert all(all(n==1 for n in c.values())for c in counts.values())
        snapshot=[(r['job_id'],r['attempt_id'],r['completion_sha256'])for r in rows]
        with(directory/'resume.log').open('wb')as log:
            resumed=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--controller',str(directory)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=60)
        assert resumed.returncode==0,'Resume failed'
        assert snapshot==[(r['job_id'],r['attempt_id'],r['completion_sha256'])for r in store.rows()]
        after=store.events()
        assert all(sum(e['event']==k for e in after)==sum(e['event']==k for e in events)for k in ['ASSIGN','ACCEPT','ADOPT'])
        result=dict(status='PASS',actual_OS_host_count=1,host_names_are_simulated=True,
            scientific_H4_calls=0,preparation_count=len(prep),q3_count=len(rows)-len(prep),
            shared_compile_tasks=1,shared_audit_tasks=1,distinct_compile_audit_hosts=True,
            resume_new_launches=0,duplicate_adoptions=0,actions=actions,wall_seconds=time.monotonic()-start,
            tasks=[dict(job_id=r['job_id'],stage=r['method'],host_id=r['host_id'])for r in rows])
        put(directory/'ACCEPTANCE.json',result);return result
    finally:
        put(directory/'HOST_ADMISSION.json',dict(hosts={'local0-wsl':dict(accepting=False,optional=True)}))
        for child in children:
            if child.poll()is None:child.terminate()
        for child in children:
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        for log in logs:log.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--controller',type=Path);p.add_argument('--worker',type=Path)
    p.add_argument('--host');p.add_argument('--output',type=Path);p.add_argument('--summarize',type=Path);a=p.parse_args()
    if a.controller:return controller(a.controller.resolve())
    if a.worker:return worker(a.worker.resolve(),a.host)
    if a.summarize:return summarize(a.summarize.resolve())
    if not a.output:p.error('--output required')
    a.output.mkdir(parents=True,exist_ok=False)
    results={name:run_case(a.output/name,swapped)for name,swapped in [('first',False),('swapped',True)]}
    summarize(a.output)
    print(json.dumps({'status':'PASS','cases':list(results),'scientific_H4_calls':0}),flush=True)


if __name__=='__main__':main()
