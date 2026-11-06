"""H6 physical RPC and fixed-scope parent. No science on import/preflight."""
from dataclasses import asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import pickle
import threading
import time

from . import c1_stage1_execution as io
from . import c_h6_validation_preparation as design
from .c_integrated_execution_rpc import PhysicalWorker, RpcProcess

ENTRY='scripts/run_research_c_h6.py'
FAMILY='artifacts/research_c_h6_independent_validation/2026-11-06-ready'
CONTRACT='docs/research/research_c_h6_conditionfix_contract_20261106.json'


def save(path,value):
    from .c_integrated_fit_worker import _plain
    import tempfile
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    # Only the newly-created run owns these mutable summaries (e.g. counts).
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,delete=False,encoding='utf-8') as stream:
        temporary=Path(stream.name)
        json.dump(_plain(value),stream,ensure_ascii=False,allow_nan=False,indent=2)
        stream.write('\n')
    os.replace(temporary,path)


def prepare_codec(task):
    from . import c_integrated_fit_worker as fit
    from .c1_real_orthogonal_codec import roundtrip
    root=Path(task['root']);out=root/task['out'];out.mkdir(parents=True,exist_ok=False)
    g,cold=fit._inputs(task,fit._load)
    j,u,x,report,_=roundtrip(cold['J'],cold['U'])
    # The existing codec checks tensor/density fidelity itself. The local
    # recentered optimizer uses the original J/U, not the provenance x0 chart.
    path=out/'C0.npz';fit._arrays(path,dict(cold,x0=x))
    value=dict(path=str(path.relative_to(root)),sha256=io.sha(path),codec=report,
        original_JU_preserved=True,x0_role='faithful gauge-equivalent chart provenance only',
        array_identity=fit._identity(cold),new_fit=0)
    save(out/'result.json',value);return value


class Evaluation:
    def __init__(self,root,system,input_path,out):
        from .c_near_transfer import PhysicalContext,System
        from .c_pr_aware_df_stage_b_runner import load
        self.root,self.out=Path(root),Path(out);self.out.mkdir(parents=True,exist_ok=True)
        self.system=System(**system)
        self.packet=load(self.root/input_path,('h','g','constant'))
        self.engine=PhysicalContext(self.packet,self.system,observed_input_sha256=io.sha(self.root/input_path))
        self.ctx,self.physical=self.engine.context,self.engine.physical
        self.cache={}
        from .c_pr_integrated_evaluation import arrays
        arrays(self.out/'physical_reference.npz',H=self.physical,sector_basis_indices=self.ctx.sector.basis_indices)

    def assess(self,label,path):
        import numpy as np
        from .c_pr_aware_df_stage_b_runner import load
        from .c_pr_integrated_evaluation import arrays
        values=load(path,('J','U'))
        row,G=self.engine.assess(**values,candidate_sha256=io.sha(path),budget_Ha=.00019875)
        directory=self.out/'physical';directory.mkdir(exist_ok=True)
        arrays(directory/(label+'.npz'),**values,generators=np.asarray(G),H=sum(G))
        save(directory/(label+'.json'),row)
        self.cache[label]=dict(d=values)
        from .c_integrated_fit_worker import _plain
        return _plain(row)


class H6PhysicalWorker(PhysicalWorker):
    def __init__(self,root,out,stop_path=None):
        super().__init__(root,out);self.stop_path=stop_path

    def request(self,operation,**request):
        if self.stop_path and (self.root/self.stop_path).exists():raise RuntimeError('DEPENDENT_WORK_STOPPED')
        if operation not in ('initialize','assess'):raise ValueError('H6_PHYSICAL_OPERATION')
        if operation!='initialize':return super().request(operation,**request)
        if self.evaluation is not None:raise ValueError('PHYSICAL_ALREADY_INITIALIZED')
        self._dependencies()
        system=request['system']
        if (system['norb'],system['rank'],system['ld'])!=(6,14,4):raise ValueError('FIXED_H6_SYSTEM')
        if io.sha(self.root/request['input_rel'])!=system['input_sha256']:raise ValueError('PHYSICAL_INPUT_IDENTITY')
        self.evaluation=Evaluation(self.root,system,request['input_rel'],self.out)
        ev=self.evaluation
        common=dict(p=ev.packet,ctx=ev.ctx,physical_reference=ev.physical,
            input_identity=system['input_sha256'],system=system,
            basis=self.stage1.bit_reverse_indices(2*system['norb'])[ev.ctx.sector.basis_indices])
        path=self.out/'common_context.pkl'
        with path.open('xb') as stream:pickle.dump(common,stream,protocol=5)
        self.input_identity=system['input_sha256']
        return dict(status='READY',common_pickle=str(path.relative_to(self.root)),input_sha256=self.input_identity)


class Run:
    """One fresh experiment, with injectable boundaries for artificial E2E."""
    def __init__(self,root,c,*,fit_python,eval_python,backend_factory=None,physical_factory=None):
        self.root,self.c=Path(root).resolve(),dict(c)
        self.c['environment']=dict(c['environment'],fit_python=str(fit_python),evaluation_python=str(eval_python))
        self.family=c['fresh_output'];self.out=self.root/self.family
        self.c['_dependency_stop']=self.family+'/DEPENDENCY_STOP'
        self.entry=ENTRY;self.cpus=sorted(os.sched_getaffinity(0))
        self.lock=threading.Lock();self.processes=set();self.services=[]
        self.first_failure=None
        self.backend_factory,self.physical_factory=backend_factory,physical_factory
        self.tick=time.monotonic();self.partial=dict(optimization={},selection=[],final={},D={},comparisons={})

    def fit(self,task,physical=None,affinity=None):
        from .c_integrated_execution import Run as OldRun
        return OldRun.fit(self,task,physical,affinity)

    def progress(self,phase,**fields):
        row=dict(phase=phase,elapsed_seconds=time.monotonic()-self.tick,**fields)
        with self.lock:
            with (self.out/'progress.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')

    def note_failure(self,error,**context):
        import traceback
        with self.lock:
            if self.first_failure is None:
                self.first_failure=dict(error_type=type(error).__name__,error=str(error),
                    traceback=traceback.format_exc(),**context)

    def base_task(self,**fields):
        return dict(root=str(self.root),system=self.system,input_path=self.c['physical_generation']['input_path'],
            input_sha256=self.system['input_sha256'],cold_path=self.cold['path'],cold_sha256=self.cold['sha256'],
            geometry='H6_100',start='C0',source_sha256=self.c['source_sha256'],
            numerical_backend=self.c['numerical_repair']['backend'],**fields)

    def optimize(self,physical):
        from .c_near_transfer import System
        self.system=asdict(System(6,14,4,1.,self.c['physical_generation']['input_sha256']))
        self.cold=dict(path=self.c['neutral']['path'],sha256=self.c['neutral']['sha256'])
        unit=lambda label,fn:self.call('input_neutral',label,fn)
        context=unit('physical_context',lambda:physical.request('initialize',system=self.system,
            input_rel=self.c['physical_generation']['input_path']))
        original=self.base_task(operation='prepare_codec',job_id='C0_codec',out=self.family+'/initialization/codec')
        self.cold=unit('codec',lambda:self.fit(original))
        row=unit('common_C0_DF',lambda:physical.request('assess',label='C0',path=str(self.root/self.cold['path'])))
        if not row['budget_PASS']:
            self.partial['status']='UNRESOLVED_INITIAL_DF';return context,[]
        metrics=unit('tensor_scale',lambda:self.fit(self.base_task(operation='tensor_scale',job_id='C0_scale')))
        scale=metrics['objective_from_tensor'];jobs=[]
        for strength in design.STRENGTHS:
            specs=unit('specs:'+str(strength),lambda strength=strength:self.fit(self.base_task(
                operation='specs',strength=strength,loss_scale=scale,job_id='spec_'+str(strength))))
            for arm in ('PR','ENTRY'):
                tag=f'H6_100_C0_{arm}_g{str(strength).replace(".","")}'
                task=self.base_task(operation='cell',arm=arm,strength=strength,loss_scale=scale,
                    objective_spec=specs[arm],job_id=tag,out=self.family+'/optimization/'+tag)
                jobs.append(task)
        # Existing fit cells use independent processes; physical RPC is shared
        # read-only context plus serialized assessments. No system setting changes.
        workers=min(len(jobs),len(self.cpus))
        def run_cell(item):
            i,task=item;result=None
            try:
                result=self.fit(task,physical,self.cpus[i::workers] or self.cpus)
                self.call.record_external('optimization',task['job_id'],result['scientific_units'])
                return task,result
            except BaseException as error:
                self.note_failure(error,trajectory=task['job_id'])
                (self.out/'DEPENDENCY_STOP').touch(exist_ok=True)
                stop=self.root/task['out']/'STOP.json'
                if result is None and stop.exists():
                    count=io.read(stop).get('scientific_units',0)
                    if count:self.call.record_external('optimization',task['job_id']+':partial',count)
                self.call.stop.set();raise
        candidates=[]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in as_completed([pool.submit(run_cell,item) for item in enumerate(jobs)]):
                task,result=future.result();self.partial['optimization'][task['job_id']]=result
                checkpoint=result['checkpoints'].get('64')
                meta=dict(id=task['job_id']+'_b64',arm=task['arm'],strength=task['strength'],
                    start='C0',budget=64,geometry='H6_100')
                candidates.append((meta,checkpoint))
                self.progress('trajectory_complete',id=task['job_id'],checkpoints=list(result['checkpoints']))
        return context,sorted(candidates,key=lambda x:x[0]['id'])

    def downstream(self,backend,candidates):
        from .c_h6_downstream import QUALIFICATION
        from . import c_near_r32_statistics as stats
        states={}
        def selection(item):
            task,checkpoint=item
            a=None if checkpoint is None else backend.prepare(task,checkpoint)
            if a is not None:
                with self.lock:states[task['id']]=a
            for r in design.R_VALUES:
                policy=None if a is None else backend.references(a,r)
                eligible=bool(policy is not None and policy['eligible'])
                row=dict(task,id=task['id'],r=r,input_sha256=backend.common['input_identity'],
                    candidate_sha256=(None if checkpoint is None else checkpoint['candidate_sha256']) if a is None else a['candidate'],cohort_role='selection',samples=32,
                    eligible=eligible,correctness='PASS',total_RZ=None,
                    numerical_equivalence='NOT_EXECUTED',
                    correctness_scope='required sample0 and structural policy checks only; never all-sample numerical equivalence',
                    qualification=QUALIFICATION,
                    eligibility_reason='NO_QUALIFIED_ENDPOINT_OR_GROUND' if a is None else 'REFERENCE_COMPLETE')
                if eligible:
                    rows,labels=backend.native(a,policy,'selection',32)
                    row['numerical_equivalence']='REGISTERED_SAMPLE0_PASS_OTHERS_NOT_CHECKED'
                    cost=self.call('native_build_compile_and_reports',task['id']+f':r{r}:account',lambda:backend.account(a,policy,rows,labels))
                    row['total_RZ']=cost['result']['lanes']['ideal_exact_native']['conditional_totals']['totals']['rz_count']
                with self.lock:self.partial['selection'].append(row)
            if a is not None and hasattr(backend,'park'):backend.park(a)
        with ThreadPoolExecutor(max_workers=getattr(backend,'workers',1)) as pool:
            for future in as_completed([pool.submit(selection,item) for item in candidates]):future.result()
        self.partial['selection'].sort(key=lambda row:(row['id'],row['r']))
        selected={arm:design.select_training([row for row in self.partial['selection'] if row['arm']==arm]) for arm in ('PR','ENTRY')}
        self.partial['selected']=selected
        save(self.out/'selected.json',selected)  # frozen BEFORE independent final data
        final={};d_cache={};wanted={}
        for arm in ('PR','ENTRY'):
            for regime in ('common_r32','fair_retuned'):
                row=selected[arm][regime]
                if row is None:continue
                key=(arm,row['candidate_sha256'],row['r'])
                wanted.setdefault(key,row)
        d_locks={row['candidate_sha256']:threading.Lock() for row in wanted.values()}
        def final_task(item):
                key,row=item;a=states[row['id']];dkey=(backend.common['input_identity'],a['candidate'])
                with d_locks[a['candidate']]:
                    if dkey not in d_cache:
                        d_cache[dkey]=backend.deterministic(a)
                        with self.lock:self.partial['D'][a['candidate']]=d_cache[dkey]
                rows,labels=backend.native(a,a['policies'][row['r']],'final',128)
                cost=self.call('native_build_compile_and_reports',row['id']+':final_account',lambda:backend.account(a,a['policies'][row['r']],rows,labels))
                secondary_policy=dict(stats.POLICY,samples=128,confidence=.95)
                secondary=self.call('statistics_and_exports',row['id']+':secondary',lambda:stats.arm_summary(
                    rows,cost['bindings'],d_cache[dkey],calibration=cost['calibration'],expected_n=128,statistics=secondary_policy))
                value=dict(cost,statistics=secondary,qualification=QUALIFICATION)
                with self.lock:
                    final[key]=value;self.partial['final']['|'.join(map(str,key))]=value
                if hasattr(backend,'park'):backend.park(a)
        with ThreadPoolExecutor(max_workers=min(len(wanted),getattr(backend,'workers',1)) or 1) as pool:
            for future in as_completed([pool.submit(final_task,item) for item in wanted.items()]):future.result()
        for regime in ('common_r32','fair_retuned'):
            group=[]
            for arm in ('PR','ENTRY'):
                row=selected[arm][regime]
                group.append(None if row is None else final[(arm,row['candidate_sha256'],row['r'])])
            if any(x is None for x in group):comparison=dict(status='UNRESOLVED_MISSING_FINAL_PAIR',qualification=QUALIFICATION)
            else:
                pr,entry=group;policy=dict(stats.POLICY,samples=128,confidence=.975)
                comparison=self.call('statistics_and_exports',regime,lambda:stats.primary(pr['rows'],entry['rows'],
                    pr['bindings']['ideal_exact_native'],entry['bindings']['ideal_exact_native'],
                    calibrations=(pr['calibration'],entry['calibration']),expected_n=128,statistics=policy))
                comparison['qualification']=QUALIFICATION
            self.partial['comparisons'][regime]=comparison

    def execute(self):
        from .c1_stage3_integrated import Calls
        from .c_integrated_server import resources
        self.out.mkdir(parents=True,exist_ok=False)
        self.call=Calls(self.out,self.c['workload_ceiling']['stage_units'],218350)
        failure=None
        try:
            save(self.out/'contract.json',self.c)
            (self.out/'SOURCE.json').write_bytes((self.root/self.c['execution_SOURCE']).read_bytes())
            (self.out/'fixed_contract.json').write_bytes((self.root/CONTRACT).read_bytes())
            save(self.out/'resources_start.json',resources(self.root))
            if self.physical_factory is None:
                physical=RpcProcess(self.root,self.c['environment']['evaluation_python'],'physical',
                    self.out/'physical_service',self.cpus,script_path=self.root/ENTRY,
                    config=dict(dependency_stop=self.c['_dependency_stop']))
            else:physical=self.physical_factory(self)
            self.services.append(physical)
            context,candidates=self.optimize(physical)
            if candidates:
                # Pickles are owned, newly created evaluation-environment data.
                # Register the existing sector class before unpickling.
                from .c1_stage1 import legacy_df_module
                legacy_df_module()
                with (self.root/context['common_pickle']).open('rb') as stream:common=pickle.load(stream)
                if common['input_identity']!=self.system['input_sha256']:raise ValueError('COMMON_INPUT_IDENTITY')
                if self.backend_factory is None:
                    from .c_h6_parallel import PoolBackend
                    plan=resource_plan(self.root,len(candidates))
                    save(self.out/'resource_plan.json',plan)
                    backend=PoolBackend(self,common,candidates,plan)
                else:
                    backend=self.backend_factory(self.root,self.c,self.out/'downstream',self.call,common,
                        threads=self.c.get('runtime_threads',1),batch=self.c.get('runtime_batch_columns',16))
                    backend.out.mkdir(parents=True,exist_ok=False)
                backend.ensure_shells();self.downstream(backend,candidates)
                self.partial['status']='H6_REGISTERED_SCOPE_COMPLETE'
            return self.partial
        except BaseException as error:
            (self.out/'DEPENDENCY_STOP').touch(exist_ok=True)
            self.call.stop.set();self.note_failure(error)
            failure=self.first_failure
            save(self.out/'failure.json',failure);raise
        finally:
            close_error=None
            for service in reversed(self.services):
                try:service.close()
                except BaseException as error:
                    self.call.stop.set()
                    if failure is None:
                        self.note_failure(error,phase='service_close');failure=self.first_failure
                        close_error=error
            if failure is not None:save(self.out/'failure.json',failure)
            if failure is None and not self.call.stop.is_set():save(self.out/'result.json',self.partial)
            save(self.out/'partial.json',self.partial)
            save(self.out/'STOP.json',dict(failed=self.call.stop.is_set(),failure=failure,
                call_counts=dict(self.call.counts),scientific_units=sum(self.call.counts.values()),
                maximum_logical_units=218350,retry=0,resume=0,mandatory_stop=True))
            if close_error is not None:raise RuntimeError('SERVICE_CLOSE_FAILED') from close_error


def resource_plan(root,tasks):
    from .c_integrated_server import resources
    snapshot=resources(root);cpus=len(snapshot['affinity'])
    for group in snapshot['cgroup_hierarchy']:
        quota=group.get('cpu.max','max').split()
        if quota[0]!='max':cpus=min(cpus,max(1,int(quota[0])//int(quota[1])))
    # Observe affinity-local spare CPU time without touching other jobs/settings.
    def cpu_ticks():
        ticks={}
        for line in Path('/proc/stat').read_text().splitlines():
            fields=line.split()
            if fields and fields[0][3:].isdigit():
                cpu=int(fields[0][3:])
                if cpu in snapshot['affinity']:
                    values=list(map(int,fields[1:9]));ticks[cpu]=(sum(values),values[3]+values[4])
        return ticks
    before=cpu_ticks();time.sleep(.25);after=cpu_ticks()
    idle=sum((after[k][1]-v[1])/max(1,after[k][0]-v[0]) for k,v in before.items() if k in after)
    import math
    budget=min(cpus,max(1,math.ceil(idle)))
    memory=snapshot['available_memory_bytes']
    for group in snapshot['cgroup_hierarchy']:
        limit=group.get('memory.max','max')
        if limit!='max':memory=min(memory,max(0,int(limit)-int(group.get('memory.current','0'))))
    # Candidate-local atom+one-r event caches plus interpreter/BLAS workspace.
    estimate=2*design.memory_model()['dense_atom_cache_bytes']+2*2**30
    if memory<estimate:raise RuntimeError('INSUFFICIENT_AVAILABLE_CANDIDATE_MEMORY')
    workers=min(tasks,budget,max(1,memory//estimate))
    return dict(workers=workers,threads_per_worker=max(1,budget//workers),
        observed_idle_CPU_equivalents=idle,effective_CPU_budget=budget,
        estimated_bytes_per_worker=estimate,resources=snapshot,
        policy='affinity/quota/observed spare CPU/memory-aware candidate processes; coordinated BLAS/Aer threads; no nested round pool; observation is not a reservation')
