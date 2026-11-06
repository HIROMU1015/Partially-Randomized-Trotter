"""Owned persistent evaluation workers; reuse JSON RPC and Calls journals."""
from pathlib import Path
import pickle
import threading

from . import c1_stage1_execution as io
from .c_integrated_execution_rpc import RpcProcess


class Actor:
    def __init__(self,root,out):
        self.root,self.out=Path(root),Path(out);self.backend=None;self.states={}

    def request(self,operation,**r):
        from .c_h6_runtime import save
        try:
            if operation=='initialize':
                from .c1_stage1 import legacy_df_module
                from .c1_stage3_integrated import Calls
                from .c_h6_downstream import Backend
                if self.backend is not None:raise ValueError('ACTOR_ALREADY_INITIALIZED')
                legacy_df_module()
                with (self.root/r['common_pickle']).open('rb') as stream:common=pickle.load(stream)
                if common['input_identity']!=r['contract']['physical_generation']['input_sha256']:raise ValueError('ACTOR_INPUT_IDENTITY')
                self.calls=Calls(self.out,r['contract']['workload_ceiling']['stage_units'],218350)
                self.backend=Backend(self.root,r['contract'],self.out/'candidates',self.calls,common,
                    threads=r['threads'],batch=r['batch'])
                self.backend.out.mkdir(exist_ok=False)
                return dict(status='READY')
            if self.backend is None:raise ValueError('ACTOR_NOT_INITIALIZED')
            b=self.backend;key=r['id']
            if operation=='prepare':
                if key in self.states:raise ValueError('CANDIDATE_ALREADY_PREPARED')
                a=b.prepare(r['task'],r['checkpoint']);self.states[key]=a
                return None if a is None else dict(candidate=a['candidate'])
            a=self.states[key]
            if operation=='references':
                p=b.references(a,r['r']);return dict(r=r['r'],eligible=p['eligible'])
            if operation=='native':
                return b.native(a,a['policies'][r['r']],r['role'],r['count'])
            if operation=='account':
                return b.account(a,a['policies'][r['r']],r['rows'],r['labels'])
            if operation=='D':return b.deterministic(a)
            if operation=='park':
                # Keep immutable small PF/exact/policy data, drop multi-GiB
                # candidate-local atoms/events between active cohorts.
                a['atoms']=None;a['reference_context']=None
                return dict(status='LARGE_CACHES_RELEASED')
            if operation=='release':
                del self.states[key];return dict(status='RELEASED')
            raise ValueError('ACTOR_OPERATION')
        except BaseException as error:
            if hasattr(self,'calls'):self.calls.stop.set()
            save(self.out/'failure.json',dict(error=repr(error)));raise
        finally:
            if hasattr(self,'calls'):
                save(self.out/'counts.json',dict(self.calls.counts))


class PoolBackend:
    """Same Backend interface; only compact metadata crosses process pipes."""
    def __init__(self,run,common,candidates,plan):
        from .c_h6_runtime import ENTRY,save
        from .c_h6_downstream import Backend
        self.run,self.common,self.out=run,common,run.out/'downstream'
        self.out.mkdir(exist_ok=False)
        bootstrap=Backend(run.root,run.c,self.out,run.call,common)
        common['h6_shells']=bootstrap.ensure_shells()
        path=run.out/'shared_context.pkl'
        with path.open('xb') as stream:pickle.dump(common,stream,protocol=5)
        self.slots=[];self.locks=[];self.seen=[]
        self.assignment={task['id']:i%plan['workers'] for i,(task,_) in enumerate(candidates)}
        for i in range(plan['workers']):
            affinity=run.cpus[i::plan['workers']] or run.cpus
            slot=RpcProcess(run.root,run.c['environment']['evaluation_python'],'evaluation',
                run.out/f'evaluation_workers/worker{i}',affinity,script_path=run.root/ENTRY,
                config=dict(threads=plan['threads_per_worker']))
            run.services.append(slot);self.slots.append(slot);self.locks.append(threading.RLock());self.seen.append({})
            slot.request('initialize',common_pickle=str(path.relative_to(run.root)),contract=run.c,
                threads=plan['threads_per_worker'],batch=run.c.get('runtime_batch_columns',16))
        self.workers=plan['workers']

    def _request(self,id,operation,**values):
        i=self.assignment[id];slot=self.slots[i]
        with self.locks[i]:
            try:return slot.request(operation,id=id,**values)
            except BaseException as error:
                self.run.note_failure(error,candidate_id=id,operation=operation)
                (self.run.out/'DEPENDENCY_STOP').touch(exist_ok=True)
                raise
            finally:
                path=slot.out/'counts.json'
                if path.exists():
                    counts=io.read(path)
                    for stage,count in counts.items():
                        delta=count-self.seen[i].get(stage,0)
                        if delta:self.run.call.record_external(stage,f'worker{i}:{operation}',delta)
                    self.seen[i]=counts

    def ensure_shells(self):return self.common['h6_shells']

    def prepare(self,task,checkpoint):
        result=self._request(task['id'],'prepare',task=task,checkpoint=checkpoint)
        return None if result is None else dict(task=task,candidate=result['candidate'],policies={})

    def references(self,a,r):
        result=self._request(a['task']['id'],'references',r=r);a['policies'][r]=result;return result

    def native(self,a,policy,role,count):
        result=self._request(a['task']['id'],'native',r=policy['r'],role=role,count=count)
        if role=='selection':
            slot=self.slots[self.assignment[a['task']['id']]]
            folder=slot.out/'candidates'/a['task']['id']/f"r{policy['r']}_{role}"
            measured=io.read(folder/'timing.json');first=io.read(folder/'first_registered_batch_timing.json')
            with self.run.lock:
                announce=not getattr(self.run,'native_timing_seen',False)
                self.run.native_timing_seen=True
            if announce:
                unit=max(0.,(measured['total_seconds']-first['wall_seconds'])/(count-1))
                serial=25*first['wall_seconds']+1035*unit
                self.run.progress('first_registered_native_batch',samples=count,candidate=a['candidate'],r=policy['r'],
                    observed=measured,checked_sample0_seconds=first['wall_seconds'],
                    unchecked_sample_mean_seconds=unit,remaining_native_seconds_serial_equivalent_model=serial,
                    remaining_native_seconds_parallel_model=serial/self.workers,
                    assumptions='maximum remaining counts; same candidate/r timings generalized; excludes reference/fit work; not a confidence bound')
        return result

    def account(self,a,policy,rows,labels):
        return self._request(a['task']['id'],'account',r=policy['r'],rows=rows,labels=labels)

    def deterministic(self,a):return self._request(a['task']['id'],'D')

    def park(self,a):return self._request(a['task']['id'],'park')
