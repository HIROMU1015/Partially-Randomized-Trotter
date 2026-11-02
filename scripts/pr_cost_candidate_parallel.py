"""Candidate-local control gates and native-receipt-only H8 task continuation.

No scientific work at import. A live origin cannot be accepted for continuation.
Scientific lowering and the originally sealed trajectory seed registry are reused.
"""
from copy import deepcopy
import importlib.util
import inspect
import math
import os
from pathlib import Path
import time

import pr_cost_batch as b
c=b.old
POLICY='H8_CANDIDATE_CONTROL_CONTINUATION_V1'
# Load the explicitly sealed pool, even after native provider import paths change.
_loader=importlib.util.spec_from_file_location('pr_candidate_owned_pool',b.ROOT/'scripts/h8_key_compile_parallel.py')
pool=importlib.util.module_from_spec(_loader);_loader.loader.exec_module(pool)


def file_row(path):
    path=Path(path)
    c.require(path.is_file() and not path.is_symlink(),'REGULAR_RECEIPT_FILE')
    return dict(path=str(path),bytes=path.stat().st_size,sha256=c.sha(path))


def task_index(tasks):
    result={t['id']:t for t in tasks}
    c.require(len(result)==len(tasks),'UNIQUE_FROZEN_TASKS')
    pairs=set()
    for task in tasks:
        cid,rep=task['cell']['id'],task['replica']
        c.require(type(rep) is int and 0<=rep<8 and task['id']==cid+'_'+str(rep),'FROZEN_TASK_ID')
        c.require((cid,rep) not in pairs,'UNIQUE_CELL_REPLICA');pairs.add((cid,rep))
    return result


def validate_completed(task,directory,binding,*,synthetic=False):
    """Mechanical acceptance of saved cost/control evidence; no recomputation."""
    directory=Path(directory);result=c.read(directory/'task_result.json');trace=c.read(directory/'trajectory.json')
    c.require(result['cell_id']==task['cell']['id'] and result['replica']==task['replica']
              and result['seed']==task['seed'] and result['binding']==binding,'FROZEN_TASK_RESULT_BINDING')
    c.require(result['cohort']=='EXPLORATION_N8' and result['synthetic'] is synthetic
              and result['normalization_unchanged'] is True,'UNCHANGED_TRAJECTORY_COHORT')
    event=result['event_digest']
    c.require(trace['cell']==task['cell'] and trace['seed']==task['seed']
              and trace['cohort']=='EXPLORATION_N8' and trace['conditional'] is False
              and trace['quantum_weights_changed'] is False and trace['event_digest']==event
              and c.digest(trace['events'])==event,'SAVED_EVENT_BINDING')
    expected=dict(trajectory=1,occurrence=task['cell']['q'],RTE_event=task['cell']['R'],
                  native_evolution=1,wrapper_compile=2)
    if task['replica']==0:expected.update(control_probe=4,independent_event_oracle=1)
    c.require(result['calls']==expected,'EXACT_NATIVE_TASK_CALLS')
    paths=[directory/'task_result.json',directory/'trajectory.json',directory/'started.json']
    c.require(set(result['axes'])=={'cosine','sine'},'BOTH_NATIVE_AXES')
    for axis in ('cosine','sine'):
        path=directory/('measured_'+axis+'.json');item=result['axes'][axis]
        c.require(c.read(path)==item and item['seed']==task['seed'] and item['event_digest']==event
                  and item['compiler_hash']==binding['compiler'],'SAVED_COMPILE_BINDING')
        c.require(all(type(item[k]) is int and item[k]>=0 for k in
                      ('RZ','CX','depth','RZ_depth','CX_depth','size')) and item['RZ']>0,'INTEGER_NATIVE_METRICS')
        c.require(isinstance(item['fingerprint'],str) and len(item['fingerprint'])==64,'NATIVE_FINGERPRINT')
        paths.append(path)
    if task['replica']==0:
        control=c.read(directory/'control.json');tolerance=task.get('control_tolerance',1e-8)
        c.require(control['status']=='PASS' and control['tolerance']==tolerance
                  and math.isfinite(control['max_error']) and 0<=control['max_error']<tolerance
                  and control['control_probes']==4 and control['independent_event_oracles']==1
                  and control['full_Fock_scalar_once'] is True and control['reference_rotated'] is False
                  and control['ordinary_ancilla_branch'] is True,'NATIVE_CONTROL_REQUIRED')
        paths.append(directory/'control.json')
    for kind,total in expected.items():
        reservations=sorted(directory.glob(kind+'_*_reserved.json'))
        values=[c.read(p) for p in reservations]
        c.require(values and max(x['total'] for x in values)==total
                  and sum(x['count'] for x in values)==total
                  and all(x['kind']==kind and x['nonrefundable'] is True for x in values),
                  'BEFORE_CALL_RESERVATIONS')
        paths.extend(reservations)
    return result,[file_row(p) for p in paths]


class CandidateQueue:
    def __init__(self,tasks,accepted=()):
        self.tasks=task_index(tasks);self.completed=set(accepted);self.passed=set()
        c.require(self.completed<=set(self.tasks),'ACCEPTED_REGISTERED_TASKS_ONLY')
        for key in self.completed:
            task=self.tasks[key]
            if task['replica']==0:self.passed.add(task['cell']['id'])
        c.require(all(t['replica']==0 or t['cell']['id'] in self.passed
                      for k,t in self.tasks.items() if k in self.completed),'REUSED_DEPENDENTS_REQUIRE_CONTROL')
    def pending(self):
        # Start every missing representative as slots become available, while
        # finished candidates release replicas without a global barrier.
        return sorted((t for k,t in self.tasks.items() if k not in self.completed),
                      key=lambda t:(t['replica']!=0,-t['cell']['R'],-t['cell']['q'],t['id']))
    def ready(self,task):return task['replica']==0 or task['cell']['id'] in self.passed
    def finish(self,task):
        c.require(task['id'] not in self.completed and self.ready(task),'ONE_COMPLETION_AFTER_LOCAL_CONTROL')
        self.completed.add(task['id'])
        if task['replica']==0:self.passed.add(task['cell']['id'])


def quiescent(identities,identity_fn=None):
    identity_fn=identity_fn or pool.identity
    for expected in identities:
        observed=identity_fn(expected[0])
        c.require(observed is None or list(observed)!=list(expected),'ORIGIN_STILL_RUNNING')


def verify_origin(origin,tasks,binding):
    c.require(c.sha(origin['origin_spec'])==origin['origin_spec_sha256'],'ORIGIN_SPEC_BYTES')
    spec=c.read(origin['origin_spec'])
    c.require(spec['SOURCE']==origin['origin_SOURCE'] and spec['scope']=='H8_A_COST'
              and spec['binding']==binding and spec['output']==origin['origin_output'],'ORIGIN_SOURCE_SCOPE')
    c.verify_source(Path(origin['origin_source_root']),spec['SOURCE'],spec['source_hashes'])
    c.verify_byte_rows(spec['inputs'])
    payload=c.read(spec['payload'])
    c.require(payload['new_seed_registry']['tasks']==tasks,'FROZEN_ORIGINAL_SEED_REGISTRY')
    c.require(c.sha(b.ROOT/'scripts/pr_cost_native.py')==spec['source_hashes']['scripts/pr_cost_native.py'],
              'UNCHANGED_ORIGINAL_NATIVE_KERNEL')


def collect_receipts(origin,tasks,binding,receipts,*,handoff=None,synthetic=False,identity_fn=None):
    """Allow exit0 workers of an intentional handoff STOP; keep that STOP intact."""
    quiescent([origin['origin_parent_identity']],identity_fn)
    by=task_index(tasks);accepted={};files=[];identities=[];attempts=[];seen=set()
    if handoff:
        c.require(handoff.get('receipt'),'HASHED_HANDOFF_AUTHORIZATION_RECEIPT')
        files.append(handoff['receipt'])
    for receipt_path in receipts:
        receipt_path=Path(receipt_path);receipt=c.read(receipt_path)
        c.require(receipt_path.name=='parallel_receipt.json'
                  and receipt_path.parent.parent.resolve()==Path(origin['origin_output']).resolve()
                  and receipt_path.parent.name in ('first_control','remaining'),'ORIGIN_MAP_RECEIPT_PATH')
        c.require(receipt['all_reaped'] is True and not receipt['cleanup_errors'],'ORIGIN_NATIVE_CLEANUP')
        if receipt['status']!='PASS':
            c.require(handoff and handoff.get('handoff_authorized') is True
                      and handoff.get('origin_SOURCE')==origin['origin_SOURCE']
                      and receipt['status']=='STOP' and receipt['first_error']=='OWN_RUN_SIGNAL_15',
                      'ONLY_EXPLICIT_HANDOFF_STOP_ACCEPTED')
        files.append(file_row(receipt_path))
        for worker in receipt['workers']:
            key=worker['task_id'];c.require(key in by,'ORIGIN_UNREGISTERED_TASK')
            c.require(key not in seen,'DUPLICATE_ORIGIN_TASK_ATTEMPT');seen.add(key)
            directory=Path(worker['directory'])
            c.require(directory.resolve().is_relative_to(Path(origin['origin_output']).resolve()) and not directory.is_symlink(),
                      'ORIGIN_EXCLUSIVE_TASK_PATH')
            c.require(directory.name==key and directory.parent.resolve()==receipt_path.parent.resolve()
                      and (by[key]['replica']==0)==(receipt_path.parent.name=='first_control'),'ORIGIN_TASK_PHASE')
            c.require(worker['reaped'] is True and worker['pid']==worker['identity'][0]
                      and worker['identity'][2]==origin['origin_parent_identity'][0]
                      and worker['identity'][3]==os.getuid(), 'ORIGIN_WORKER_PARENT')
            identities.append(worker['identity'])
            reservations=[file_row(p) for p in sorted(directory.glob('*_reserved.json'))]
            attempts.append(dict(task_id=key,exit_code=worker['exit_code'],reservations=reservations,
                                 failed_or_unfinished_charge_refunded=False))
            if worker['exit_code']!=0 or not worker.get('result_sha256'):
                attempts[-1]['reuse_status']='NOT_ACCEPTED_INCOMPLETE_NATIVE_PROOF'
                continue
            c.require(key not in accepted,'DUPLICATE_ACCEPTED_TASK')
            c.require(worker.get('result_sha256')==c.sha(directory/'task_result.json'),'NATIVE_RESULT_SHA')
            started=c.read(directory/'started.json')
            c.require(started['identity']==worker['identity'] and started['task']==key,'START_NATIVE_IDENTITY')
            result,required=validate_completed(by[key],directory,binding,synthetic=synthetic)
            accepted[key]=dict(directory=str(directory),result_sha256=worker['result_sha256'],result=result)
            files.extend(required)
    quiescent(identities,identity_fn)
    queue=CandidateQueue(tasks,accepted)
    unique={row['path']:row for row in files+[f for a in attempts for f in a['reservations']]}
    return dict(accepted=accepted,files=list(unique.values()),old_attempts=attempts,
                remaining_task_ids=[t['id'] for t in queue.pending()],retired_identities=identities)


def verify_manifest(manifest,tasks,binding,origin,*,synthetic=False,identity_fn=None):
    value=dict(manifest);saved=value.pop('manifest_digest')
    c.require(c.digest(value)==saved,'RESUME_MANIFEST_DIGEST')
    c.require(manifest['schema']==POLICY and manifest['origin']==origin
              and manifest['task_registry_sha256']==c.digest(tasks) and manifest['binding']==binding,
              'RESUME_SOURCE_INPUT_TASK_BINDING')
    c.require(manifest['origin_quiescent_observations']>=2 and manifest['synthetic'] is synthetic,
              'TWO_QUIESCENCE_OBSERVATIONS')
    c.verify_byte_rows([dict(row,role=str(i)) for i,row in enumerate(manifest['files'])])
    observed=collect_receipts(origin,tasks,binding,manifest['receipt_paths'],
        handoff=manifest.get('handoff'),synthetic=synthetic,identity_fn=identity_fn)
    for key in ('accepted','files','remaining_task_ids','old_attempts','retired_identities'):
        c.require(observed[key]==manifest[key],'RECHECKED_NATIVE_ACCEPTANCE:'+key)
    return observed


def build_manifest(origin,tasks,binding,receipts,*,handoff=None,synthetic=False,identity_fn=None):
    first=collect_receipts(origin,tasks,binding,receipts,handoff=handoff,synthetic=synthetic,identity_fn=identity_fn)
    time.sleep(.02)
    second=collect_receipts(origin,tasks,binding,receipts,handoff=handoff,synthetic=synthetic,identity_fn=identity_fn)
    c.require(first==second,'STABLE_ORIGIN_RECEIPTS')
    manifest=dict(schema=POLICY,origin=deepcopy(origin),task_registry_sha256=c.digest(tasks),binding=binding,
        receipt_paths=[str(Path(p)) for p in receipts],handoff=handoff,synthetic=synthetic,
        origin_quiescent_observations=2,**second)
    manifest['manifest_digest']=c.digest(manifest)
    return manifest


def run(spec,grant,data,output):
    """Execution entry: called only through the approved batch contract gate."""
    import pr_cost_native as native
    from pr_cost_batch_science import native_context
    tasks=data['new_seed_registry']['tasks'];origin=spec['continuation_origin']
    verify_origin(origin,tasks,spec['binding'])
    c.require(len(tasks)==128 and len({t['cell']['id'] for t in tasks})==16
              and all({t['replica'] for t in tasks if t['cell']['id']==cid}==set(range(8))
                      for cid in {t['cell']['id'] for t in tasks}),'REGISTERED_16_CANDIDATES_N8')
    records=c.read(data['A_signal'])['signals']
    c.require({v['cell']['id']:c.digest(v) for v in records}==data['signal_record_hashes'],'PINNED_A_SIGNAL_RECORDS')
    manifest=c.read(spec['resume_manifest']);verified=verify_manifest(manifest,tasks,spec['binding'],origin)
    # New SOURCE changes scheduling/acceptance only; retain the frozen native
    # kernel and original SOURCE-bound seed registry of the original 128 tasks.
    c.require(c.sha(inspect.getsourcefile(native.compile_task))==spec['native_kernel_sha256'], 'UNCHANGED_NATIVE_KERNEL')
    queue=CandidateQueue(tasks,verified['accepted']);results={k:v['result'] for k,v in verified['accepted'].items()}
    provenance={key:dict(SOURCE=origin['origin_SOURCE'],directory=value['directory'],
                       result_sha256=value['result_sha256'],reused=True)
                for key,value in verified['accepted'].items()}
    pending=queue.pending()
    if pending:
        cells={t['cell']['id']:t['cell'] for t in tasks}
        context=native_context(spec,data,list(cells.values()),output)
        signals={v['cell']['id']:v for v in c.read(data['A_signal'])['signals']}
        enriched={}
        for task in pending:
            record=signals[task['cell']['id']]
            enriched[task['id']]=dict(task,preparation_hash=context.preparations[task['cell']['prefix']].preparation_hash,
                saved_log_B=record['log_B'],log_B_tolerance=record['allowance']['log_B_margin'],control_tolerance=1e-8)
        def compile_one(task,directory):return native.compile_task(context,enriched[task['id']],directory)
        def accept(task,handle):
            c.require(handle['reaped'] and handle['exit_code']==0,'OWN_NATIVE_COMPLETION')
            value,_=validate_completed(enriched[task['id']],handle['directory'],spec['binding'])
            results[task['id']]=value;queue.finish(task)
            provenance[task['id']]=dict(SOURCE=spec['SOURCE'],directory=handle['directory'],
                                      result_sha256=handle['result_sha256'],reused=False)
        budget=c.Budget(output,spec['continuation_call_budget'])
        def reserve(task):
            for kind,count in dict(trajectory=1,native_evolution=1,wrapper_compile=2,
                    occurrence=task['cell']['q'],RTE_event=task['cell']['R'],
                    control_probe=4 if task['replica']==0 else 0,
                    independent_event_oracle=1 if task['replica']==0 else 0).items():
                if count:budget.take(kind,count)
        receipt=pool.map_owned(pending,compile_one,cpus=grant['allowed_cpus'],
            limits=dict(max_workers=len(grant['allowed_cpus'])),directory=output/'candidate_workers',
            deadline=math.inf,font_seed=data['native_adapter']['font_seed'],ready=queue.ready,on_complete=accept,
            reserve=reserve)
        c.save(output/'candidate_native_receipt.json',receipt)
    c.require(len(results)==spec['calls']['trajectory'] and len(queue.passed)==16,'COMPLETE_REGISTERED_128_AND_16_CONTROLS')
    c.save(output/'continuation_accounting.json',dict(reused_tasks=len(verified['accepted']),
        executed_tasks=len(pending),original_task_count=len(tasks),old_attempts=verified['old_attempts'],
        task_provenance=provenance,seed_derivation_SOURCE=origin['origin_SOURCE'],
        reused_compile_calls_not_recharged=True,failed_or_unfinished_charge_refunded=False))
    return dict(status='COMPLETE_MANDATORY_STOP',paired_costs=[results[t['id']] for t in tasks],
        scope=spec['scope'],scheduling=POLICY,original_confirmation_n32_unchanged=True,
        new_signal=0,new_ranking=0,new_quantum_shots=0,next_stage_authorized=False,automatic_retry=False)
