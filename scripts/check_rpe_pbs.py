#!/usr/bin/env python3
"""Artificial acceptance through the normal schedule entry and real PBS worker.

Only artificial rational MSE and text 'circuit' leaves execute. Real saved-input
metadata checks are a separate path; this harness never opens real H4 data.
"""
import argparse
from collections import Counter
from copy import deepcopy
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import types


def read(path): return json.loads(Path(path).read_text())
def put(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def modules(source):
    sys.path[:0] = [str(source/'scripts'), str(source/'src'), str(source/'tests')]
    import check_rpe_distributed_q3 as fixture
    return fixture


def q3_outcomes(directory):
    values = {}
    for path in sorted((directory/'run').glob('jobs/*/*/OUTCOME.json')):
        request = read(path.parent/'REQUEST.json')
        if request['route']['kind'] == 'preparation':
            continue
        result = read(path)
        if request['route']['kind'] != 'artificial' or not result.get('artificial'):
            raise AssertionError('Non-artificial Q3 result in acceptance fixture')
        key = json.dumps([request['method'],request['configuration_id'],
                          request['N_X'],request['N_Y']],separators=(',',':'))
        values[key] = [result['status'],result.get('MSE_lower'),result.get('MSE_upper')]
    if not values:
        raise AssertionError('No artificial Q3 outcomes')
    return values


def prepare(directory, source, settings=None, *, exercise_faults=False):
    fixture = modules(source)
    fixture.prepare(directory, port=fixture.free_port() if exercise_faults else 0,
                    total_slots='auto', sleep_seconds=2.)
    problem=read(directory/'PROBLEM.json')
    problem['problem']=dict(system='ARTIFICIAL_PBS_NORMAL_ENTRY_ACCEPTANCE',
        hamiltonian_data=dict(expected=dict(fragment_count=10)))
    for method in problem['methods'].values():method['family']='PR'
    put(directory/'PROBLEM.json',problem)
    spec = read(directory/'ROUND.json')
    spec['search_overrides']['requested_widths'] = dict(PR_PAULI=[.1], PR_DF=[.1])
    spec['search_overrides']['lookahead'] = 1
    distributed = spec['executor']['distributed']
    local = fixture.profile(directory, 'local', 1)
    distributed.update(local_worker=local, affinity_wait_seconds=0,
        preparation=dict(budgets=dict(signal=100,compile=100,audit=100)),
        auto_allocation=dict(priority_groups=[['local','pbs-0']], discovery_grace_seconds=0, offer_ttl_seconds=5))
    if settings:
        distributed['pbs'] = settings
        distributed['host_stage_capabilities'] = dict(local=['q3','signal','audit'], **{'pbs-0':['q3','compile']})
    spec['artificial_pbs_acceptance'] = True
    if exercise_faults:
        spec['artificial_pbs_fault_exercise'] = True
    put(directory/'ROUND.json', spec)
    put(directory/'SOURCE.json', dict(root=str(source), commit=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=source,text=True).strip()))


def driver_type(fixture, *, preparation_source=None):
    from trottertracks.rpe_preparation_queue import normalize_preparation_request
    Base = fixture.fixture_driver()

    class Driver(Base):
        def completion_api(self):
            completion = super().completion_api()
            if getattr(completion, '_pbs_wrapped', False): return completion
            completion._pbs_wrapped = True
            original = completion.complete
            saved = completion.save
            save_lock = threading.RLock()
            completion.lock = save_lock
            def save(obj):
                with save_lock: return saved()
            completion.save = types.MethodType(save,completion)

            def bind(obj, runtime): obj.runtime = runtime; return obj
            def complete(obj, request):
                runtime = obj.runtime
                cid = json.dumps(request,sort_keys=True)
                def task(stage, identity, dependencies=(), inputs=None):
                    faults = read(obj.root.parent/'ROUND.json').get('artificial_pbs_fault_exercise', False)
                    command=['@PYTHON@','-B','@SOURCE@/scripts/artificial_preparation_leaf.py',
                        '--stage',stage,'--output','@WORK@/products','--sleep',
                        '3' if faults and stage == 'signal' else '.2']
                    if stage == 'audit': command += ['--inputs','@INPUTS@']
                    value=normalize_preparation_request(dict(method='ARTIFICIAL',route=dict(kind='preparation'),
                        preparation=dict(stage=stage,provider='ARTIFICIAL_'+stage,artificial=True,
                            identity=identity,command=command,outputs=['products'],
                            source=deepcopy(preparation_source or runtime.executor.source_manifest),dependencies=list(dependencies),
                            dependency_inputs=inputs or {})))
                    return runtime.executor.submit_preparation(value, approved=True,budget=100,
                        contract=dict(artificial=True))
                signal=task('signal',cid)
                compile=task('compile','ONE_SHARED_ARTIFICIAL_CIRCUIT')
                audit=task('audit','SHARED_CROSS_HOST_AUDIT',(compile,), {compile:{
                    'products/circuit.txt':'compiled/circuit.txt',
                    'products/receipt.json':'compiled/receipt.json'}})
                for job in (signal,compile,audit):
                    result=runtime.executor.preparation_result(job,wait=True)
                    if result['status'] != 'PREPARATION_COMPLETE': raise AssertionError(result)
                with save_lock:
                    return original(request)
            def install_snapshot(obj, context, problem, keys):
                with save_lock:
                    records = deepcopy([record for record in obj.state['records'].values()
                                        if record.get('key') in keys])
                return obj.install(context, problem, records), records
            completion.bind_stage_executor=types.MethodType(bind,completion)
            completion.complete=types.MethodType(complete,completion)
            completion.install_snapshot=types.MethodType(install_snapshot,completion)
            return completion
    return Driver


@contextmanager
def artificial_transport_faults(directory, spec):
    """Fault only this labelled fixture process; production transport is untouched."""
    if not spec.get('artificial_pbs_fault_exercise'):
        yield
        return
    if not spec.get('artificial_pbs_acceptance'):
        raise ValueError('Transport fault exercise requires labelled artificial inputs')
    from trottertracks.rpe_pbs import Gateway
    from trottertracks.rpe_q3_timing import stamp
    original = Gateway.call

    def call(gateway, payload):
        result = original(gateway, payload)
        lost = directory/'LOST_SUBMIT_REPLY.json'
        if payload['action'] == 'ensure' and not lost.exists():
            token = payload['spec']['token']
            deadline = time.monotonic() + 5
            while True:
                snapshot = original(gateway, dict(action='snapshot'))
                observed = next((row for row in snapshot['allocations']
                                 if row['token'] == token and row.get('job_id')), None)
                if observed is not None:
                    put(lost, dict(stamp(), artificial_only=True, token=token,
                                   pbs_job_id=observed['job_id'],
                                   event='ARTIFICIAL_LOST_REPLY_AFTER_QSUB_ACCEPTED',
                                   scheduler_status=observed['status']))
                    gateway.close()
                    raise OSError('ARTIFICIAL_LOST_SUBMIT_REPLY_AFTER_QSUB_ACCEPTED')
                if time.monotonic() >= deadline:
                    raise TimeoutError('Artificial fault exercise could not confirm accepted PBS job within 5s')
                time.sleep(.05)
        if payload['action'] == 'exchange' and not (directory/'ALLOW_TRANSFER').exists():
            for token, requests in result.get('requests', {}).items():
                completed = next((request for request in requests
                                  if request['endpoint'] == '/v1/complete'), None)
                if completed is not None:
                    marker = directory/'COMPLETED_BEFORE_TRANSFER.json'
                    if not marker.exists():
                        allocation = next((row for row in result['allocations'] if row['token'] == token), {})
                        put(marker, dict(stamp(), artificial_only=True, token=token,
                            pbs_job_id=allocation.get('job_id'),
                            event='ARTIFICIAL_DISCONNECT_BEFORE_CENTRAL_RESULT_DISPATCH',
                            request_id=completed['request_id'],
                            job_id=completed['payload']['job_id'],
                            attempt_id=completed['payload']['attempt_id']))
                    gateway.close()
                    raise OSError('ARTIFICIAL_DISCONNECT_AFTER_COMPUTE_BEFORE_TRANSFER')
        return result

    Gateway.call = call
    try:
        yield
    finally:
        Gateway.call = original


def execute(directory, source):
    fixture = modules(source)
    import run_rpe_schedule_search as entry
    from trottertracks.rpe_q3_store import JobStore
    spec=read(directory/'ROUND.json')
    if not spec.get('artificial_pbs_acceptance'): raise ValueError('Labelled artificial fixture required')
    if read(directory/'SOURCE.json')['commit'] != subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip():
        raise ValueError('Fixed acceptance SOURCE changed')
    started=time.monotonic()
    with artificial_transport_faults(directory, spec):
        entry.main(['--problem',str(directory/'PROBLEM.json'),'--round',str(directory/'ROUND.json'),
                    '--run-dir',str(directory/'run'),'--max-cycles','5','--batch','32'],driver_factory=driver_type(fixture))
    rows=JobStore(directory/'run/distributed/QUEUE.sqlite3').rows()
    result=dict(artificial_only=True,scientific_H4_calls=0,source=read(directory/'SOURCE.json'),
        controller_wall_seconds=time.monotonic()-started,
        outcomes=q3_outcomes(directory), queue_states=dict(Counter(row['state'] for row in rows)),
        hosts=dict(Counter(row['host_id'] for row in rows)),
        stages={stage:[dict(job_id=row['job_id'],host_id=row['host_id'],state=row['state']) for row in rows
                       if row['method']=='PREPARATION:'+stage] for stage in ('signal','compile','audit')})
    put(directory/'ACCEPTANCE.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--settings',type=Path)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--exercise-faults',action='store_true',
                        help='Prepare labelled artificial lost-reply/transfer/restart checks')
    args=parser.parse_args(); directory=args.directory.resolve(); source=args.source_root.resolve()
    if args.prepare:
        prepare(directory,source,read(args.settings) if args.settings else None,
                exercise_faults=args.exercise_faults)
        print(json.dumps(dict(prepared=str(directory),artificial_only=True)));return
    result=execute(directory,source)
    print(json.dumps({key:value for key,value in result.items() if key!='outcomes'},ensure_ascii=False))


if __name__=='__main__': main()
