#!/usr/bin/env python3
"""Evaluate one unchanged saved Q3 REQUEST through the fixed RomuTail/PBS engine.

This is an external technical caller of schedule.main(driver_factory=...), not
the unmodified Windows/Gateway Start CLI. It neither reads an old OUTCOME to
answer the first call nor selects another scientific candidate. The fixed
executor, task budget, source checks, worker and adoption path remain in use.
"""
import argparse
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


CONTROLLER_SOURCE = '925985b4ea76a15f8a07c507179d4fa6c492eabd'
COMPUTATION_SOURCE = '28d976b99cdf8329adcc1b79327e329e947ef774'
COMPLETE = 'SINGLE_SAVED_Q3_PORTABILITY_COMPLETE'


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),default=str).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,default=str)+'\n');temporary.replace(path)


def immutable(path,value):
    path=Path(path)
    if path.exists():
        if read(path)!=value:raise ValueError('Existing portability input differs: '+str(path))
    else:put(path,value)


def load_fixed(root):
    root=Path(root).resolve()
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    changed=subprocess.check_output(['git','diff','HEAD','--name-only','--','src','scripts'],cwd=root,text=True).strip()
    if head!=CONTROLLER_SOURCE or changed:
        raise ValueError('Fixed controller/worker SOURCE925 differs or has modified code')
    sys.path[:0]=[str(root/'src'),str(root/'scripts')]
    entry=importlib.import_module('run_rpe_schedule_search')
    if Path(entry.__file__).resolve()!=root/'scripts/run_rpe_schedule_search.py':
        raise ValueError('Normal entry was imported from a different SOURCE')
    from trottertracks import rpe_q3_coordinator as coordinator
    from trottertracks.rpe_q3_store import JobStore
    from trottertracks.rpe_q3_timing import stamp
    return SimpleNamespace(root=root,entry=entry,cli=entry.K.CLI,optimizer=entry.O,
                           coordinator=coordinator,JobStore=JobStore,stamp=stamp)


def validate(service,request,problem,spec,output,fixed_root):
    """Inspect fixed configuration and finite task scope; run no scientific leaf."""
    root=Path(fixed_root).resolve();output=Path(output).resolve()
    if Path(service['code_root']).resolve()!=root or service['code_source']!=CONTROLLER_SOURCE:
        raise ValueError('Service must select the exact fixed controller SOURCE925')
    if service['computation_source']!=COMPUTATION_SOURCE:
        raise ValueError('Service scientific computation SOURCE must remain28d')
    state=Path(service['state_directory']).resolve()
    if output==state or not output.is_relative_to(state):
        raise ValueError('Use a dedicated output directory below the central state root')
    method=request['method'];cid=request['configuration_id']
    if request['route']['kind'] in ('artificial','preparation'):
        raise ValueError('This caller requires an existing real scientific Q3 route')
    if not cid or not request['request_id'] or not request.get('expected'):
        raise ValueError('Saved REQUEST identity and expected binding are required')
    if spec.get('methods')!=[method] or spec.get('max_new_q3_per_method')!={method:1}:
        raise ValueError('Exactly one selected method with a one-Q3 budget required')
    if spec.get('science_approved') is not True:
        raise ValueError('Use the adopted one-Q3 round configuration')
    for shots in (request['N_X'],request['N_Y']):
        if len(shots)!=problem['rpe']['M']+1 or any(type(n)is not int or n<1 for n in shots):
            raise ValueError('Saved shot allocation differs from the full round horizon')
    options=spec['executor'];distributed=options['distributed'];pbs=distributed['pbs']
    if options.get('scheduler')!='concurrent' or options.get('max_concurrent_jobs')!=1:
        raise ValueError('The ordinary distributed executor must have one global task slot')
    if options.get('prior_run_dirs') or spec.get('continues'):
        raise ValueError('Do not import old evaluation results into this fresh reevaluation')
    if distributed.get('local_worker') or distributed.get('relay') or distributed.get('required_hosts'):
        raise ValueError('No local worker, existing-host relay or required host is allowed')
    if distributed.get('computation_source')!=COMPUTATION_SOURCE:
        raise ValueError('Distributed scientific computation SOURCE must remain28d')
    if not distributed.get('computation_root'):
        raise ValueError('Explicit frozen scientific computation root required')
    budgets=(distributed.get('preparation')or{}).get('budgets',{})
    if any(budgets.get(stage)!=0 for stage in ('signal','compile','audit')):
        raise ValueError('Signal/compile/audit budgets must all be explicitly zero')
    if spec.get('distributed_preparation') or spec.get('candidate_completion'):
        raise ValueError('Do not attach candidate preparation or another task graph')
    price=spec.get('price_validation')or{}
    if price.get('compile_approved') is not False:
        raise ValueError('Optional price compile acquisition must be disabled')
    store=Path(price.get('store','')).resolve()
    if not store.is_relative_to(output) or store==output:
        raise ValueError('Use a private metadata store below this output directory')
    workers=pbs.get('workers',[])
    if len(workers)!=1:raise ValueError('Exactly one PBS worker profile required')
    host=workers[0]['host_id']
    if host in ('local','server1','server2','local0-wsl'):
        raise ValueError('The sole selected host must be the dedicated PBS worker')
    if pbs.get('transport_source')!=CONTROLLER_SOURCE:
        raise ValueError('PBS transport/worker must use fixed SOURCE925')
    resources=pbs.get('resources')or{}
    if any(name not in resources for name in ('ncpus','mem','walltime')):
        raise ValueError('Explicit PBS CPU/memory/walltime resources required')
    if type(resources['ncpus'])is not int or resources['ncpus']<1:
        raise ValueError('Positive allocated PBS CPU request required')
    estimate=(pbs.get('task_seconds')or{}).get('q3')
    if not isinstance(estimate,(int,float))or estimate<=0:
        raise ValueError('An explicit Q3 duration estimate must fit PBS admission')
    dp=options.get('dp_workers')
    if type(dp)is not int or dp<1 or dp>resources['ncpus']:
        raise ValueError('Declared Q3 DP workers must fit the PBS CPU request')
    policy=distributed.get('host_eligibility')
    if isinstance(policy,(str,Path)):
        path=Path(policy);policy=read(path if path.is_absolute() else root/path)
    if not isinstance(policy,dict) or policy.get('schema')!='RPE_Q3_HOST_ELIGIBILITY_V1':
        raise ValueError('Existing request-specific host eligibility policy required')
    if set(policy.get('jobs',{}))!={request['request_id']}:
        raise ValueError('Host policy must contain only this exact saved REQUEST')
    decision=policy['jobs'][request['request_id']]
    if decision.get('eligible_host_ids')!=[host] or decision.get('request_sha256')!=digest(request):
        raise ValueError('Only the current exact-REQUEST PBS PASS may receive the task')
    if decision.get('decision_sha256')!=digest({k:v for k,v in decision.items()if k!='decision_sha256'}):
        raise ValueError('Existing host eligibility decision content differs')
    if decision.get('source_manifest',{}).get('commit')!=COMPUTATION_SOURCE:
        raise ValueError('Host policy scientific SOURCE must remain28d')
    if (decision.get('hosts',{}).get(host)or{}).get('status')!='PASS':
        raise ValueError('PBS request/runtime/input readiness is not PASS')
    return dict(host_id=host,method=method,request_sha256=digest(request),
                source_manifest_sha256=decision.get('source_manifest_sha256'))


def driver_type(core,service,request,problem,spec,bindings):
    """Use the ordinary entry's supported driver hook for exactly one adopted task."""
    class SingleSavedRequestDriver:
        def __init__(self,args):
            self.a=args;self.run=Path(args.run_dir)
            path=self.run/'PORTABILITY_STATE.json'
            self.state=read(path)if path.exists()else dict(stop={})

        def save(self):put(self.run/'PORTABILITY_STATE.json',self.state)

        def log(self,**fields):
            with(self.run/'DRIVER_PROGRESS.jsonl').open('a')as stream:
                stream.write(json.dumps(dict(core.stamp(),**fields),default=str)+'\n')

        def loop(self):
            if self.a.plan_only:
                raise ValueError('Use the saved metadata inspection record; this caller only executes the adopted one task')
            if self.state.get('stop',{}).get('reason')==COMPLETE:
                self.log(event='PORTABILITY_COMPLETED_RUN_REUSED',new_scientific_calls=0,new_pbs_allocations=0)
                return
            database=self.run/'distributed/QUEUE.sqlite3'
            if database.exists():
                rows=core.JobStore(database).rows()
                if any(row['job_id']!=request['request_id'] or row['request_sha256']!=bindings['request_sha256'] for row in rows):
                    raise ValueError('Existing portability queue contains a different task')
                if any(row['state']in('UNKNOWN','CANCELLED')for row in rows):
                    raise ValueError('Existing scientific STOP/UNKNOWN requires reconciliation; no new attempt')
            routes={request['method']:[deepcopy(request['route'])]}
            expected={request['configuration_id']:deepcopy(request['expected'])}
            executor,_=core.cli.make_executor(core.root,self.run,routes,expected,problem,spec)
            try:
                view=executor.for_method(request['method'])
                proposal=dict(configuration_id=request['configuration_id'],
                    vector=core.optimizer.interleave(request['N_X'],request['N_Y']),
                    cost=core.optimizer.frac(request['cost']),origin=deepcopy(request['origin']))
                if view.job_dir(proposal).name!=request['request_id']:
                    raise ValueError('Original problem/route/shot contract does not reproduce the saved job ID')
                row=executor.store.row(request['request_id'])
                if row is None:
                    decision=view._eligibility_for(request,request['request_id'])
                    if not executor.store.eligible(request,decision):
                        raise ValueError('Current fixed SOURCE/input/PBS readiness rejects the saved REQUEST')
                    reserved=executor.store.reserve(request['method'],request['request_id'],deepcopy(request),1,True,
                        contract=dict(evaluation=view.contract,prefix_key=core.coordinator.prefix_key(request)),
                        eligibility=decision)
                    if not reserved:raise ValueError('Existing one-task reservation declined')
                    view.log(event='PORTABILITY_EXACT_REQUEST_RESERVED',job=request['request_id'],
                             configuration=request['configuration_id'],request_sha256=bindings['request_sha256'])
                answer=view.answer(proposal)
                row=executor.store.row(request['request_id'])
                if row is None or row['state']!='ACCEPTED' or row.get('adopted_epoch') is None:
                    raise ValueError('Single Q3 did not reach ordinary accepted/adopted state; preserve its existing attempt')
                if row['host_id']!=bindings['host_id'] or row['request']!=request:
                    raise ValueError('Accepted task host or original scientific REQUEST differs')
                # This OUTCOME is read only after the new worker attempt has
                # been accepted; no old scientific result answers the first call.
                job=self.run/'jobs'/request['method']/request['request_id']
                outcome=read(job/'OUTCOME.json')
                if outcome.get('scientific_evaluation_calls')!=1:
                    raise ValueError('Expected exactly one scientific Q3 evaluation in the new attempt')
                rows=executor.store.rows()
                if len(rows)!=1 or executor.store.counts(request['method'],started_only=True)!=1:
                    raise ValueError('One-task scientific budget invariant differs')
                result=dict(schema='RPE_PBS_SAVED_Q3_PORTABILITY_V1',
                    entry='scripts/run_rpe_schedule_search.py:main(driver_factory)',
                    gateway_cli_start_used=False,controller_source=CONTROLLER_SOURCE,
                    computation_source=COMPUTATION_SOURCE,request_sha256=bindings['request_sha256'],
                    job_id=request['request_id'],attempt_id=row['attempt_id'],host_id=row['host_id'],
                    method=request['method'],configuration_id=request['configuration_id'],
                    N_X=request['N_X'],N_Y=request['N_Y'],cost=request['cost'],answer=answer,
                    scientific_evaluation_calls=1,signal_calls=0,compile_calls=0,price_fit_calls=0,
                    adoption_count=sum(event['event']=='ADOPT'for event in executor.store.events()),
                    executor_metrics=executor.metrics())
                put(self.run/'PORTABILITY_RESULT.json',core.optimizer.jsonable(result))
                self.state['stop']=dict(reason=COMPLETE,job_id=request['request_id'],attempt_id=row['attempt_id'])
                self.save()
            finally:
                executor.finish()
    return SingleSavedRequestDriver


def execute(service_path,request_path,output,fixed_root):
    if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)
    service_path=Path(service_path).resolve();request_path=Path(request_path).resolve()
    output=Path(output).resolve();fixed_root=Path(fixed_root).resolve()
    service=read(service_path);raw=request_path.read_bytes();request=json.loads(raw)
    problem=read(service['problem']);spec=read(service['round'])
    bindings=validate(service,request,problem,spec,output,fixed_root)
    core=load_fixed(fixed_root)
    if output.exists()and not(output/'PORTABILITY_INPUTS.json').exists():
        allowed={'PROBLEM.json','ROUND.json','SERVICE.json'}
        if any(path.name not in allowed for path in output.iterdir()):
            raise ValueError('A fresh reevaluation output cannot contain previous jobs or outcomes')
    output.mkdir(parents=True,exist_ok=True)
    immutable(output/'PROBLEM.json',problem);immutable(output/'ROUND.json',spec)
    immutable(output/'SERVICE.json',service)
    immutable(output/'PORTABILITY_INPUTS.json',dict(controller_source=CONTROLLER_SOURCE,
        computation_source=COMPUTATION_SOURCE,service_path=str(service_path),
        request_path=str(request_path),request_raw_sha256=hashlib.sha256(raw).hexdigest(),
        request_canonical_sha256=bindings['request_sha256'],problem_sha256=digest(problem),
        round_sha256=digest(spec),service_sha256=digest(service),
        caller_path=str(Path(__file__).resolve()),caller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        execution_scope='ONE_EXACT_SAVED_Q3_REQUEST; NO_SIGNAL_COMPILE_FIT',
        normal_engine_programmatic_hook=True,gateway_cli_start_used=False))
    original=output/'ORIGINAL_REQUEST.json'
    if original.exists()and original.read_bytes()!=raw:
        raise ValueError('Saved original REQUEST bytes differ on resume')
    if not original.exists():original.write_bytes(raw)
    core.entry.main(['--problem',str(output/'PROBLEM.json'),'--round',str(output/'ROUND.json'),
                     '--run-dir',str(output),'--max-cycles','1','--batch','1'],
        driver_factory=driver_type(core,service,request,problem,spec,bindings))
    result_path=output/'PORTABILITY_RESULT.json'
    if not result_path.exists():raise ValueError('Single Q3 ended without an adopted result')
    return read(result_path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--service',type=Path,required=True)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fixed-code-root',type=Path,required=True)
    args=parser.parse_args()
    result=execute(args.service,args.request,args.output,args.fixed_code_root)
    print(json.dumps({key:result[key]for key in('job_id','attempt_id','host_id','scientific_evaluation_calls')},default=str))


if __name__=='__main__':main()
