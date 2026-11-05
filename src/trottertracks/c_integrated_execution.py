"""One-shot orchestration of the already fixed integrated research-C scope.

Scientific engines and the proposal stay unchanged. A dedicated execution root
holds byte-identical SOURCE/inputs and all fresh outputs. No retry or resume.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from collections import Counter
import json
import os
import pickle
import resource
import subprocess
import sys
import threading
import time
import traceback

from . import c_integrated_plan as plan
from . import c1_stage1_execution as io
from .c1_stage3_integrated import Calls

CONTRACT = 'docs/research/research_c_integrated_contract_unapproved_20261105.json'
ENTRY = 'scripts/run_research_c_integrated_execution.py'
FAMILY = 'artifacts/research_c_integrated/2026-11-05'


def environment(root, label):
    env = dict(os.environ)
    for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[name] = '1'
    env['JAX_ENABLE_X64'] = '1'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    for name, leaf in [('TMPDIR','tmp'),('XDG_CACHE_HOME','cache'),('JAX_COMPILATION_CACHE_DIR','jax'),
                       ('MPLCONFIGDIR','mpl'),('NUMBA_CACHE_DIR','numba')]:
        p = Path(root)/'.cache/integrated_execution'/label/leaf
        p.mkdir(parents=True, exist_ok=True)
        env[name] = str(p)
    return env


def plain(value):
    from .c_integrated_fit_worker import _plain
    return _plain(value)


def fit_worker(root, task):
    """Fit environment entry. Physical work is delegated over its JSON pipe."""
    import contextlib
    from . import c_integrated_fit_worker as fit
    from . import c_pr_aware_df_stage_b_runner as loader
    from . import c_near_transfer as transfer
    reply = os.fdopen(os.dup(sys.stdout.fileno()),'w',buffering=1)
    os.dup2(sys.stderr.fileno(),sys.stdout.fileno())
    def send(value):
        reply.write(json.dumps(plain(value), allow_nan=False)+'\n');reply.flush()
    sequence = 0
    def physical(label, J, U):
        nonlocal sequence
        sequence += 1
        folder = Path(root)/task['out']/'rpc_arrays'
        folder.mkdir(exist_ok=True)
        path = folder/f'{sequence:06d}.npz'
        loader.save_arrays(path, J=J, U=U)
        send(dict(type='physical', label=label, path=str(path)))
        response = json.loads(sys.stdin.readline())
        if not response['ok']: raise RuntimeError(response['error'])
        return response['value']
    try:
        with contextlib.redirect_stdout(sys.stderr):
            op = task['operation']
            if op == 'restore': value = fit.run_restore(task, physical)
            elif op == 'cell': value = fit.run_cell(task, physical)
            elif op == 'specs': value = fit.make_objective_specs(task)
            elif op == 'load_development':
                packet = loader.load(Path(root)/task['input_path'], ('h','g','constant'))
                cold = loader.load(Path(root)/task['cold_path'], ('J','U','x0'))
                value = dict(input_sha256=io.sha(Path(root)/task['input_path']),
                    cold_sha256=io.sha(Path(root)/task['cold_path']),
                    input_shapes={k:list(v.shape) for k,v in packet.items()},
                    initial_shapes={k:list(v.shape) for k,v in cold.items()})
            elif op == 'neutral_identity':
                from . import c_near_practical as old
                c = dict(input_paths=task['input_roles'], saved_inputs=task['saved_inputs'])
                p = loader.load(Path(root)/task['input_path'], ('g',))
                cold = loader.load(Path(root)/task['cold_path'], ('J','U','x0'))
                value = old.initial_identity(p,cold,io.read(Path(root)/task['input_roles']['scales']),c)
            elif op == 'start_identity':
                g, cold = fit._inputs(task, loader.load)
                transfer.validate_arrays(g,cold['J'],cold['U'],transfer.System(**task['system']))
                value = dict(status='PAIRED_START_PASS', array_identity=fit._identity(cold),
                    input_sha256=task['system']['input_sha256'], cold_sha256=io.sha(Path(root)/task['cold_path']))
            elif op == 'tensor_scale':
                from .c1_h6_initialization import tensor_metrics
                g,cold = fit._inputs(task,loader.load)
                value = tensor_metrics(g,cold['J'],cold['U'])
                if not value['objective_from_tensor'] > 0: raise ValueError('POSITIVE_NEUTRAL_SCALE_REQUIRED')
            elif op == 'generate_input':
                from .c_integrated_transfer_input import generate
                out = Path(root)/task['out'];out.mkdir(parents=True,exist_ok=False)
                call = Calls(out,dict(scf=1,spatial_integrals=1),2)
                packet = generate(call)
                mo = packet.pop('mo_coeff')
                loader.save_arrays(out/'input.npz',**packet)
                loader.save_arrays(out/'mo_provenance.npz',mo_coeff=mo)
                value = dict(input_path=str((out/'input.npz').relative_to(root)),
                    input_sha256=io.sha(out/'input.npz'),scientific_units=sum(call.counts.values()),
                    call_counts=dict(call.counts))
                io.save(out/'result.json',value)
            else: raise ValueError('FIT_OPERATION')
        send(dict(type='result',value=value))
    except BaseException as exc:
        send(dict(type='error',error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        reply.close()


class Run:
    def __init__(self, root, source):
        self.root = Path(root).resolve()
        self.c = io.read(self.root/CONTRACT)
        self.source, self.digest = source, io.sha(source)
        self.cpus = sorted(os.sched_getaffinity(0))
        self.summary, self.services, self.stages = {}, [], {}
        self.processes, self.lock = set(), threading.Lock()
        self.tick = time.monotonic()
        self.progress_path = self.root/FAMILY/'execution_progress.jsonl'

    def progress(self, phase, **fields):
        row = dict(phase=phase,elapsed_seconds=time.monotonic()-self.tick,**fields)
        with self.lock:
            with self.progress_path.open('a') as stream:stream.write(json.dumps(plain(row),allow_nan=False)+'\n')
        print(json.dumps(plain(row),ensure_ascii=False,allow_nan=False),flush=True)

    def task(self, geometry, system, input_path, cold_path=None, **kwargs):
        return dict(root=str(self.root),geometry=geometry,system=system,input_path=input_path,
            input_sha256=system['input_sha256'],cold_path=cold_path,
            cold_sha256=None if cold_path is None else io.sha(self.root/cold_path),
            source_sha256=self.digest,**kwargs)

    def fit(self, task, physical=None, affinity=None):
        task = dict(task, affinity=self.cpus if affinity is None else affinity)
        tag = task['job_id'];folder=self.root/FAMILY/'execution_jobs'
        folder.mkdir(exist_ok=True)
        config=folder/(tag+'.json');io.save(config,task)
        log=(folder/(tag+'.log')).open('x')
        p=subprocess.Popen([self.c['environment']['fit_python'],'-B',str(self.root/ENTRY),
            '--root',str(self.root),'--worker','fit','--worker-config',str(config)],
            cwd=self.root,env=environment(self.root,tag),stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=log,text=True)
        with self.lock:self.processes.add(p)
        try:
            while True:
                line=p.stdout.readline()
                if not line:raise RuntimeError('FIT_WORKER_EXIT:'+tag)
                response=json.loads(line)
                if response['type']=='physical':
                    if physical is None:raise ValueError('PHYSICAL_SERVICE_REQUIRED')
                    try:
                        value=physical.request('assess',label=tag+'_'+response['label'],path=response['path'])
                        answer=dict(ok=True,value=value)
                    except BaseException as exc:answer=dict(ok=False,error=repr(exc))
                    p.stdin.write(json.dumps(answer,allow_nan=False)+'\n');p.stdin.flush()
                elif response['type']=='error':raise RuntimeError('FIT_WORKER_FAIL:'+tag+':'+response['error'])
                elif response['type']=='result':
                    p.stdin.close()
                    if p.wait()!=0:raise RuntimeError('FIT_WORKER_NONZERO:'+tag)
                    return response['value']
                else:raise ValueError('FIT_WORKER_PROTOCOL')
        finally:
            if p.poll() is None:
                p.terminate();p.wait()
            with self.lock:self.processes.discard(p)
            log.close()

    def stage(self, name, actual):
        limit=self.c['maximum_calls']['stage_totals'][name]
        if actual>limit:raise ValueError('STAGE_CALL_LIMIT:'+name)
        self.stages[name]=actual
        if sum(self.stages.values())>self.c['maximum_calls']['total']:raise ValueError('TOTAL_CALL_LIMIT')
        self.progress('stage_complete',stage=name,scientific_units=actual,maximum=limit)

    def initialize(self, geometry, output, strengths=None):
        from .c_integrated_execution_rpc import RpcProcess
        from .c_near_transfer import System
        output=self.root/output;output.mkdir(parents=True,exist_ok=False)
        io.save(output/'contract.json',self.c);io.save(output/'SOURCE.json',io.read(self.source))
        init_limits=dict(self.c['maximum_calls']['components']['development_initialization' if geometry=='H4_100' else 'transfer_initialization'])
        del init_limits['neutral_restore']
        if geometry=='H4_115':
            # The generated input worker owns the same two scientific units.
            del init_limits['scf'];del init_limits['spatial_integrals']
        shared=output/'initialization';shared.mkdir()
        call=Calls(shared,init_limits,sum(init_limits.values()))
        extra=0
        if geometry=='H4_100':
            input_path=self.c['input_roles']['input'];cold_path=self.c['input_roles']['initial']
            input_sha=self.c['saved_development_inputs'][input_path]
            scale=self.c['objective_policy']['development_tensor_scale']
        else:
            generated=self.fit(dict(operation='generate_input',root=str(self.root),job_id='H4_115_generate',
                out=str((output/'input').relative_to(self.root))))
            extra+=generated['scientific_units'];input_path=generated['input_path'];input_sha=generated['input_sha256']
            cold_path=None;scale=None
        system=asdict(System(4,8,2,1. if geometry=='H4_100' else 1.15,input_sha))
        base=self.task(geometry,system,input_path,cold_path,loss_scale=scale)
        if geometry=='H4_100':
            call('saved_input_load','fixed neutral h/g/J/U/x0',lambda:self.fit(dict(base,operation='load_development',job_id=geometry+'_load')))
            call('neutral_identity','saved faithful x0',lambda:self.fit(dict(base,operation='neutral_identity',job_id=geometry+'_identity',input_roles=self.c['input_roles'],saved_inputs=self.c['saved_development_inputs'])))
        physical=RpcProcess(self.root,self.c['environment']['evaluation_python'],'physical',output/'physical_service',self.cpus)
        self.services.append(physical)
        context=call('physical_context','original H sector36',lambda:physical.request('initialize',input_rel=input_path,system=system))
        starts={}
        if geometry=='H4_100':
            row=call('neutral_physical_assessment','saved neutral C0',lambda:physical.request('assess',label='C0_neutral',path=str(self.root/cold_path)))
            if row['budget_PASS']:starts['C0']=dict(path=cold_path,physical=row)
        else:
            task=self.task(geometry,system,input_path,None,operation='restore',job_id=geometry+'_C0_restore',
                start='C0',loss_scale=None,out=str((output/'starts/C0').relative_to(self.root)))
            restored=self.fit(task,physical);extra+=sum(io.read(output/'starts/C0'/'STOP.json')['actual_calls'].values())
            if restored['eligible']:
                cold_path=restored['candidate_path'];starts['C0']=dict(path=cold_path,physical=restored['physical'])
                base=self.task(geometry,system,input_path,cold_path,loss_scale=None)
                metrics=call('neutral_scale','restored C0 positive tensor scale',lambda:self.fit(dict(base,operation='tensor_scale',job_id=geometry+'_neutral_scale')))
                scale=metrics['objective_from_tensor'];base['loss_scale']=scale
        if 'C0' in starts:
            def restore(start):
                task=dict(base,operation='restore',start=start,job_id=geometry+'_'+start+'_restore',
                    out=str((output/'starts'/start).relative_to(self.root)))
                affinity=self.cpus[('P1','P2').index(start)::2] or self.cpus
                return start,self.fit(task,physical,affinity)
            with ThreadPoolExecutor(max_workers=2) as pool:
                for start,value in pool.map(restore,('P1','P2')):
                    extra+=sum(io.read(output/'starts'/start/'STOP.json')['actual_calls'].values())
                    if value['eligible']:starts[start]=dict(path=value['candidate_path'],physical=value['physical'])
        specs={}
        for start in plan.STARTS:
            if start not in starts:continue
            task=self.task(geometry,system,input_path,starts[start]['path'],loss_scale=scale,start=start)
            starts[start]['identity']=call('paired_start_identity',start,lambda task=task:self.fit(dict(task,operation='start_identity',job_id=geometry+'_'+start+'_paired')))
            if strengths is None:
                for strength in plan.STRENGTHS:
                    specs[start,strength]=call('objective_setup',f'{start}:{strength}',lambda task=task,strength=strength:self.fit(dict(task,operation='specs',strength=strength,job_id=geometry+'_'+start+'_spec_'+str(strength).replace('.',''))))
            else:
                def target_specs(task=task):
                    paired=self.fit(dict(task,operation='specs',strength=strengths['PR'],job_id=geometry+'_'+start+'_spec_transfer'))
                    # Initial mass scales are independent of strength. Preserve
                    # them and set ENTRY's already frozen relative contribution.
                    paired['ENTRY']['strength']=strengths['ENTRY']
                    return paired
                specs[start,'transfer']=call('objective_setup',start,target_specs)
        call('start_summary','fixed starts/no replacements',lambda:io.save(shared/'result.json',dict(starts=starts,scale=scale,system=system)))
        self.stage('development_initialization' if geometry=='H4_100' else 'transfer_initialization',sum(call.counts.values())+extra)
        return dict(geometry=geometry,output=output,system=system,input_path=input_path,input_sha256=input_sha,
            scale=scale,starts=starts,specs=specs,physical=physical,physical_context=context,strengths=strengths)

    def optimize(self, g):
        jobs=[]
        for start in plan.STARTS:
            if start not in g['starts']:continue
            for arm in plan.ARMS:
                for strength in plan.STRENGTHS if g['strengths'] is None else (g['strengths'][arm],):
                    tag=f"{g['geometry']}_{start}_{arm}_g{str(strength).replace('.','')}"
                    spec=g['specs'][start,strength][arm] if g['strengths'] is None else g['specs'][start,'transfer'][arm]
                    jobs.append(self.task(g['geometry'],g['system'],g['input_path'],g['starts'][start]['path'],
                        operation='cell',job_id=tag,arm=arm,start=start,strength=strength,loss_scale=g['scale'],
                        objective_spec=spec,out=str((g['output']/'optimization'/tag).relative_to(self.root))))
        results={};actual=0
        # Tiny per-cell contractions predominantly use one CPU; split the whole
        # available affinity across all independent finite trajectories.
        partitions=[self.cpus[i::len(jobs)] for i in range(len(jobs))] if jobs else []
        with ThreadPoolExecutor(max_workers=min(len(jobs),len(self.cpus)) or 1) as pool:
            fs={pool.submit(self.fit,t,g['physical'],partitions[i] or self.cpus):t for i,t in enumerate(jobs)}
            for f in as_completed(fs):
                t=fs[f];value=f.result();results[t['job_id']]=value;actual+=value['scientific_units']
                self.progress('optimization_cell_complete',geometry=g['geometry'],id=t['job_id'],status=value['summary']['status'],checkpoints=list(value['checkpoints']),completed=len(results),total=len(jobs))
        g['optimization']=results
        self.stage('development_optimization' if g['geometry']=='H4_100' else 'transfer_optimization',actual)
        return g

    def candidate_jobs(self,g):
        jobs=[];missing=[]
        indexed={(v['start'],v['arm'],v['strength']):v for v in g['optimization'].values()}
        for start in plan.STARTS:
            for arm in plan.ARMS:
                strengths=plan.STRENGTHS if g['strengths'] is None else (g['strengths'][arm],)
                for strength in strengths:
                    value=indexed.get((start,arm,strength))
                    for budget in plan.BUDGETS:
                        tag=f"{g['geometry']}_{start}_{arm}_g{str(strength).replace('.','')}_b{budget}"
                        task=dict(id=tag,geometry=g['geometry'],start=start,arm=arm,strength=strength,budget=budget)
                        checkpoint=None if value is None else value['checkpoints'].get(str(budget))
                        if checkpoint is None:
                            for r in plan.R_VALUES:
                                missing.append(dict(task,input_sha256=g['input_sha256'],candidate_sha256=None,r=r,
                                    cohort_role='selection',samples=32,actual_samples=0,correctness='PASS',
                                    correctness_scope='no detected upstream execution mismatch; downstream NOT_RUN',
                                    eligible=False,total_RZ=None,qualification='NO_QUALIFIED_DF_CHECKPOINT'))
                        else:
                            jobs.append(dict(task=task,candidate_path=checkpoint['candidate_path'],
                                reference_path=checkpoint['physical']['reference_path'],physical_record=checkpoint['physical']))
        return jobs,missing

    def seed_manifest(self,g,jobs,out):
        from .c_integrated_downstream_execution import cohort_key
        from .c_pr_aware_df_stage_b import connection
        rows=[]
        for job in jobs:
            candidate=job['physical_record']['candidate_sha256'];task=job['task']
            for r in plan.R_VALUES:
                for split in ('selection','final'):
                    for batch in range(1 if split=='selection' else 2):
                        cohort=f'integrated:{cohort_key(task,candidate)}:{split}:r{r}:batch{batch:02d}'
                        axis=f'INTEGRATED_{split.upper()}_PRICE_ONLY'
                        master=(2026110501,cohort,self.digest)
                        for i in range(32):
                            seed=connection.trajectory_seed(master,'proposed',candidate,-1,axis,i,0)
                            rows.append(dict(task=task['id'],r=r,split=split,batch=batch,sample=i,seed=seed))
        if len({x['seed'] for x in rows})!=len(rows):raise ValueError('REGISTERED_STREAM_SEED_COLLISION')
        io.save(out,dict(status='PASS',rows=rows,PRNG_draws=0,
            scope='all selection and all potential final cohorts; final draws only for selected pairs',
            qualification='finite-bit stream separation, not proof of ideal IID'))
        return dict(status='PASS',streams=len(rows))

    def selection(self,g):
        from .c_integrated_execution_rpc import RpcProcess
        from . import c_integrated_downstream_execution as backend
        from . import c1_stage1
        from .c1_stage3_accuracy import AccuracyTarget
        output=g['output']/'downstream';output.mkdir()
        commonout=output/'common';commonout.mkdir()
        limits=dict(evidence_load=1,input_load=1,physical_reference_load=1,sector_context=1,
            physical_ground=1,target_validation=1,seed_manifest=1,selection=12,main_comparison=12,result_summary=1)
        call=Calls(commonout,limits,32)
        call('evidence_load','execution source and original shell provenance',lambda:io.read(self.source))
        c1_stage1.legacy_df_module()
        call('input_load','same original h/g/scalar',lambda:io.sha(self.root/g['input_path']))
        common=call('physical_reference_load','reuse original-H context',lambda:pickle.load((self.root/g['physical_context']['common_pickle']).open('rb')))
        call('sector_context','reuse immutable sector CAR',lambda:common['ctx'])
        ground=call('physical_ground','original-H own geometry',lambda:g['physical'].request('ground',out=str((commonout/'physical_ground').relative_to(self.root))))
        common['target']=call('target_validation','fixed CA/RPE horizon',lambda:AccuracyTarget(**self.c['target']))
        old=io.read(self.root/'docs/research/research_c_near_practical_r32_downstream_contract_20261103.json')
        common['shells']=io.read(self.root/old['shell_reports'])
        ready=commonout/'common_ready.pkl'
        with ready.open('xb') as stream:pickle.dump(common,stream,protocol=5)
        runtime=backend.runtime_contract(self.c,self.digest,old['shells'])
        jobs,rows=self.candidate_jobs(g)
        call('seed_manifest','fixed streams before any new native draws',lambda:self.seed_manifest(g,jobs,commonout/'seed_manifest.json'))
        workers=[];owners={};selection_calls=0
        # Use independent compiler processes across the available CPU set.
        # The measured old peak RSS (2.74e6 KiB) supplies a workload estimate,
        # rather than an artificial worker/RAM cap. Each process's own affinity
        # drives the unchanged inner round/native resource_plan.
        available=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
        estimate=2740000*1024
        count=min(len(jobs),len(self.cpus),max(1,available//estimate)) if jobs else 0
        io.save(commonout/'parallel_plan.json',dict(candidate_processes=count,affinity_cpus=self.cpus,
            available_memory_bytes=available,estimated_peak_RSS_bytes_per_process=estimate,
            estimate_origin='old 96-native r-retuning run peak RSS; observed workload, not quota',
            speedup_claim=None))
        for i in range(count):
            affinity=self.cpus[i::count]
            w=RpcProcess(self.root,self.c['environment']['evaluation_python'],'evaluation',output/f'worker{i:02d}',affinity)
            self.services.append(w);w.request('initialize',common_pickle=str(ready.relative_to(self.root)),runtime=runtime)
            workers.append(w)
        if ground.get('status')=='PASS':
            buckets=[jobs[i::count] for i in range(count)] if count else []
            def bucket(worker,items):
                found=[]
                for job in items:
                    task=job['task'];folder=output/'selection'/task['id']
                    value=worker.request('selection',**job,out=str(folder.relative_to(self.root)),state_key=task['id'])
                    report=value['report'];actual=io.read(folder/'call_summary.json')['scientific_units']
                    found.append((task['id'],report,actual,worker))
                    self.progress('selection_candidate_complete',geometry=g['geometry'],id=task['id'],
                        eligible_r=[x['r'] for x in report['rows'] if x['eligible']],scientific_units=actual)
                return found
            with ThreadPoolExecutor(max_workers=count or 1) as pool:
                fs=[pool.submit(bucket,w,b) for w,b in zip(workers,buckets)]
                for f in as_completed(fs):
                    for key,report,actual,w in f.result():
                        rows.extend(report['rows']);owners[key]=w;selection_calls+=actual
        else:
            for job in jobs:
                for r in plan.R_VALUES:
                    rows.append(dict(job['task'],input_sha256=g['input_sha256'],candidate_sha256=job['physical_record']['candidate_sha256'],
                        r=r,cohort_role='selection',samples=32,actual_samples=0,correctness='PASS',eligible=False,total_RZ=None,
                        qualification='ORIGINAL_PHYSICAL_GROUND_ALIAS_UNQUALIFIED'))
        selected={}
        for start in plan.STARTS:
            for budget in plan.BUDGETS:
                for arm in plan.ARMS:
                    block=[x for x in rows if (x['start'],x['budget'],x['arm'])==(start,budget,arm)]
                    allowed=plan.STRENGTHS if g['strengths'] is None else (g['strengths'][arm],)
                    selected[start,budget,arm]=call('selection',f'{start}:{budget}:{arm}',lambda block=block,allowed=allowed:plan.select_training(block,strengths=allowed))
        io.save(output/'selection_rows.json',rows)
        io.save(output/'selected.json',{'/'.join(map(str,k)):v for k,v in selected.items()})
        g.update(downstream_output=output,downstream_common_call=call,rows=rows,selected=selected,owners=owners,
            evalworkers=workers,selection_calls=selection_calls,runtime=runtime)
        return g

    def finals(self,g):
        from . import c_integrated_downstream_execution as backend
        chosen={}
        for key,regimes in g['selected'].items():
            for regime,row in regimes.items():
                if row is not None:chosen[key+(regime,)]=row
        d_registry={};finals={};actual=g['selection_calls']
        # One state is owned by one persistent worker. Assign its complete D and
        # final work to that worker; different workers can progress independently.
        by_worker={w:[] for w in g['evalworkers']}
        for claim,row in chosen.items():by_worker[g['owners'][row['id']]].append((claim,row))
        registry_lock=threading.Lock()
        def bucket(worker,items):
            local={};units=0;ds={};pairs={}
            for claim,row in items:
                state=row['id'];candidate=row['candidate_sha256'];r=row['r']
                if state not in ds:
                    key=(row['input_sha256'],candidate)
                    with registry_lock:
                        if key not in d_registry:
                            out=g['downstream_output']/'deterministic'/state
                            response=worker.request('D',state_key=state,out=str(out.relative_to(self.root)))
                            value=response['result'];d_registry[key]=(value,str(out))
                            units+=io.read(out/'call_summary.json')['scientific_units']
                        else:
                            value,path=d_registry[key]
                            worker.request('attach_D',state_key=state,D_result=value,D_output=path,
                                candidate_sha256=candidate,input_sha256=row['input_sha256'])
                    ds[state]=True
                pair=(state,r)
                if pair not in pairs:
                    out=g['downstream_output']/'final'/state/f'r{r}'
                    response=worker.request('final',state_key=state,r=r,out=str(out.relative_to(self.root)))
                    pairs[pair]=response['final'];units+=io.read(out/'call_summary.json')['scientific_units']
                    self.progress('final_pair_complete',geometry=g['geometry'],id=state,r=r,samples=64)
                local[claim]=pairs[pair]
            return local,units
        with ThreadPoolExecutor(max_workers=len(by_worker) or 1) as pool:
            for f in as_completed([pool.submit(bucket,w,v) for w,v in by_worker.items()]):
                values,units=f.result();finals.update(values);actual+=units
        comparisons={};call=g['downstream_common_call']
        for start in plan.STARTS:
            for budget in plan.BUDGETS:
                for regime in ('common_r32','fair_retuned'):
                    key=(start,budget,regime)
                    destination=g['downstream_output']/f'comparison_{start}_b{budget}_{regime}.json'
                    comparisons['/'.join(map(str,key))]=backend.compare_final(
                        finals.get((start,budget,'PR',regime)),finals.get((start,budget,'ENTRY',regime)),
                        destination,call,primary=start=='C0' and budget==64)
        report=dict(status='INTEGRATED_GEOMETRY_COMPLETE',geometry=g['geometry'],comparisons=comparisons,
            actual_CA_achieved=None,scientific_qualification='conditional finite-bit price proxy and nominal intervals',retry=0,resume=0)
        call('result_summary','registered geometry summary',lambda:io.save(g['output']/'result.json',report))
        actual+=sum(call.counts.values())
        self.stage('development_downstream' if g['geometry']=='H4_100' else 'transfer_downstream',actual)
        for w in g['evalworkers']:w.close()
        g['physical'].close()
        self.summary[g['geometry']]=report
        return report

    def execute(self):
        failure=None
        try:
            self.progress('starting',source_sha256=self.digest,root=str(self.root),affinity=self.cpus)
            development=self.selection(self.optimize(self.initialize('H4_100',self.c['fresh_outputs'][0])))
            freezeout=self.root/FAMILY/'recipe_freeze';freezeout.mkdir()
            freeze=Calls(freezeout,dict(training_only_strength_freeze=1),1)
            def choose():
                try:return dict(status='TRANSFER_RECIPE_FROZEN',strengths=plan.freeze_transfer_strength([x for x in development['rows'] if x['budget']==64]))
                except ValueError as exc:
                    if str(exc)!='NO_COMMON_COMPLETE_STRENGTH_FOR_TRANSFER':raise
                    return dict(status='NO_QUALIFIED_TRANSFER_RECIPE',strengths=None)
            recipe=freeze('training_only_strength_freeze','development selection only',choose)
            io.save(freezeout/'recipe.json',recipe);self.stage('freeze_transfer_recipe',1)
            self.finals(development)
            if recipe['strengths'] is not None:
                target=self.selection(self.optimize(self.initialize('H4_115',self.c['fresh_outputs'][1],recipe['strengths'])))
                self.finals(target)
            else:self.summary['H4_115']=dict(status='SKIPPED_NO_QUALIFIED_FROZEN_RECIPE')
            end=self.root/FAMILY/'integrated_result';end.mkdir()
            call=Calls(end,dict(summary=1),1)
            call('summary','fixed scope completed without formal research GO',lambda:io.save(end/'result.json',self.summary))
            self.stage('integrated_summary',1)
        except BaseException as exc:
            failure=dict(error=repr(exc),traceback=traceback.format_exc())
            self.progress('failure',**failure)
            raise
        finally:
            for service in self.services:
                try:service.close()
                except BaseException as exc:
                    if failure is None:failure=dict(error=repr(exc),phase='worker_cleanup')
            stop=dict(status='FAILED_MANDATORY_STOP' if failure else 'COMPLETE_MANDATORY_STOP',
                failed=failure is not None,failure=failure,stage_counts=self.stages,
                completed_stage_units=sum(self.stages.values()),maximum_logical_units=172835,
                partial_task_journals='preserved in fresh output; completed_stage_units excludes an interrupted stage',
                retry=0,resume=0,mandatory_stop=True,wall_seconds=time.monotonic()-self.tick,
                parent_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            io.save(self.root/FAMILY/'INTEGRATED_STOP.json',stop)
            self.progress('stopped',status=stop['status'],completed_stage_units=stop['completed_stage_units'])


def preflight(root, execution_source):
    import hashlib
    root=Path(root).resolve();source=io.read(execution_source)
    result=plan.preflight(root,root/CONTRACT)
    for name,digest in source['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('EXECUTION_SOURCE:'+name)
    c=io.read(root/CONTRACT)
    for role,python in [('fit',c['environment']['fit_python']),('evaluation',c['environment']['evaluation_python'])]:
        pins=c['environment'][role+'_packages']
        code='import importlib.metadata,json; print(json.dumps({k:importlib.metadata.version(k) for k in '+repr(pins)+'}))'
        actual=json.loads(subprocess.check_output([python,'-B','-c',code],text=True))
        if actual!=pins:raise ValueError('PACKAGE_METADATA:'+role)
    result.update(execution_SOURCE_sha256=io.sha(execution_source),storage_root=str(root))
    return result
