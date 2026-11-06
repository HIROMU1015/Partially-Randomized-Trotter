"""Single fixed N3 campaign driver. No scientific work on import.

Uses the frozen N3 adapter/kernel. Parent records independent path/word tapes;
fork workers inherit one decoded bundle and expose only bits and price metrics.
No probability/state cache or new numerical kernel is implemented here.
"""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from collections import Counter
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, is_dataclass
from fractions import Fraction as F
from pathlib import Path
from types import SimpleNamespace as NS
import fcntl
import hashlib
import json
import math
import multiprocessing as mp
import os
import resource
import subprocess
import time
from unittest.mock import patch

from . import a1_control_variates as cv
from . import a1_engine_connection as kernel
from . import a1_n3_connection as n3
from .shared_cpu import module

BASE = 'a167effd4a1192d9806ea0bde8e0f7af71a1ef55'
DOC = Path('docs/research/o2_a1_n3_science_20261106')
OUTPUT = Path('artifacts/o2_a1_new_target/2026-11-06/n3_campaign_run01')
EPSILON, ALPHA = F(1,8), F(1,20)
TRAIN, CAL, CAP, SEED = 64, 512, 8192, 20261106
_SERVICES = None
_WORKER = None
_STOP_EVENT = None


def encode(value):
    if isinstance(value,F):return str(value)
    if is_dataclass(value):return asdict(value)
    if isinstance(value,Path):return str(value)
    raise TypeError(type(value).__name__)


def dumps(value):return json.dumps(value,default=encode,ensure_ascii=False,sort_keys=True,allow_nan=False)


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:f.write(dumps(value)+'\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def append(path,value):
    with Path(path).open('a') as f:f.write(dumps(value)+'\n');f.flush();os.fsync(f.fileno())


def profile(root):
    result=kernel.technical_profile(root)
    for name in ('scripts/tracks/algorithm_codesign/run_a1_n3_campaign.py',
                 'tests/tracks/algorithm_codesign/test_a1_n3_campaign.py'):
        result['source_sha256'][name]=hashlib.sha256((Path(root)/name).read_bytes()).hexdigest()
    return result


@contextmanager
def forbidden_science():
    dep=kernel.n2.n1.dependencies()
    def stop(*args,**kwargs):raise RuntimeError('N3 forbids preparation/CDF/state/spectral/engine requalification')
    with ExitStack() as stack:
        for owner,name in ((dep.split,'split_df_hamiltonian_by_ld'),(dep.partial,'prepare_df_partial_s2'),
            (dep.hamiltonian,'solve_df_ground_state'),(kernel.n2,'spectral_enclosure'),
            (kernel.n2,'state_certificate'),(kernel.n2,'run'),(kernel.n2.n1,'run'),
            (kernel,'qualify_fixed_bundle'),(kernel,'compose_error'),(kernel,'certify_gate'),
            (kernel.SavedEngine,'qualify'),(kernel.SavedEngine,'uniform_inventory')):
            stack.enter_context(patch.object(owner,name,stop))
        yield


def domains():
    cells={'B0':('all',),'B1-O':('0','2'),'B1-OL':('0','1','2','3','4'),'A1':('all',)}
    result=[]
    for method in n3.METHODS:
        for axis in ('X','Y'):
            if method!='B0':
                train_cells=cells['B1-OL'] if method=='A1' else cells[method]
                result.extend(n3.Domain(method,axis,'train',cell) for cell in train_cells)
            for stage in ('cal','prod'):result.extend(n3.Domain(method,axis,stage,cell) for cell in cells[method])
    return tuple(sorted(result,key=lambda d:d.token))


class PinnedSampler:
    """Replay an already journaled path without drawing the path stream twice."""
    def __init__(self,sampler,path,method,cell):
        self.bank,self.original,self.path,self.method,self.cell=sampler.bank,sampler,path,method,cell
    def draw(self,method,cell,source):
        if (method,cell)!=(self.method,self.cell) or self.original.cell_of(method,self.path)!=cell:
            raise ValueError('retained path replay mismatch')
        return self.path


def prepare_trial(sampler,streams,domain,trial,journal,word_journal):
    ticket=hashlib.sha256((domain.token+':'+str(trial)).encode()).hexdigest()
    append(journal,dict(event='TRIAL_PLANNED',ticket=ticket,domain=domain,trial=trial))
    started=time.perf_counter();path_word,bit_word=streams.claim(domain,trial);words=[]
    def path_source():
        word=path_word();words.append(word);return word
    method='B1-OL' if domain.method=='A1' and domain.stage=='train' else domain.method
    bit=None
    try:path=sampler.draw(method,domain.cell,path_source);bit=bit_word()
    finally:append(word_journal,dict(ticket=ticket,path_words=words,measurement_word=bit))
    job=dict(ticket=ticket,domain=domain,trial=trial,path=path,word=bit,sampling_method=method,
        generation_seconds=time.perf_counter()-started)
    append(journal,dict(event='PATH_FIXED',ticket=ticket,domain=domain,trial=trial,path=path,
        path_word_count=len(words),bit_word_count=1,generation_seconds=job['generation_seconds']))
    return job


def worker_initialize(out):
    global _WORKER
    # Only our own workers get lower scheduling priority; no quota/affinity cap,
    # no other-user process or global environment setting is changed.
    os.nice(10)
    out=Path(out);compiler_module=module('rte_compiled_cost')
    compiler=module('rte').CompilerSettings(('rz','sx','x','cx'),None,None,2,None,None,17,'1.3.0')
    cache=compiler_module.TranspiledCircuitCostCache(maximum_entries=76288,
        persistent_path=out/'cache'/'metrics.sqlite',retain_transpiled_circuits=False)
    _WORKER=(out,out/f'worker_{os.getpid()}.jsonl',compiler,cache)


def worker_trial(job):
    """One exact probability call, bit fsync, then full price. No oracle return."""
    facade,prices,sampler,scopes,_certificate=_SERVICES
    out,log,compiler,cache=_WORKER;d=job['domain'];ticket=job['ticket'];counts={}
    def counted(owner,name,label):
        original=getattr(owner,name)
        def call(*args,**kwargs):
            counts[label]=counts.get(label,0)+1
            append(log,dict(event='CALL',ticket=ticket,name=label))
            return original(*args,**kwargs)
        return patch.object(owner,name,call)
    append(log,dict(event='WORKER_STARTED',ticket=ticket,pid=os.getpid(),domain=d,trial=job['trial']))
    started=time.perf_counter();engine_seconds=None;record=None;compile_seconds=0.
    original_compile=module('rte_compiled_cost').transpile_and_measure_cost
    def timed_compile(*args,**kwargs):
        nonlocal compile_seconds
        if _STOP_EVENT is not None and _STOP_EVENT.is_set():raise RuntimeError('campaign STOP before compile')
        counts['compile']=counts.get('compile',0)+1
        append(log,dict(event='CALL',ticket=ticket,name='compile'))
        t=time.perf_counter()
        try:return original_compile(*args,**kwargs)
        finally:compile_seconds+=time.perf_counter()-t
    try:
        if _STOP_EVENT is not None and _STOP_EVENT.is_set():raise RuntimeError('campaign STOP before engine')
        original_probability=facade._MeasurementFacade__probability
        def observed_probability(*args):
            counts['exact_engine']=counts.get('exact_engine',0)+1
            append(log,dict(event='CALL',ticket=ticket,name='exact_engine'))
            return original_probability(*args)
        facade=n3.MeasurementFacade(observed_probability,scopes)
        original_apply=kernel.apply_exact
        def guarded_apply(*args):
            if _STOP_EVENT is not None and _STOP_EVENT.is_set():raise RuntimeError('campaign STOP during exact engine')
            return original_apply(*args)
        with ExitStack() as stack:
            stack.enter_context(patch.object(kernel,'apply_exact',guarded_apply))
            stack.enter_context(counted(kernel.SavedEngine,'selected_circuit','selected_full_build'))
            stack.enter_context(counted(kernel,'born_probability','Born'))
            stack.enter_context(patch.object(module('rte_compiled_cost'),'transpile_and_measure_cost',timed_compile))
            streams=n3.Streams(64)
            def no_redraw():raise AssertionError('path was already sampled and journaled')
            streams.bind(d,no_redraw,lambda:job['word'])
            pinned=PinnedSampler(sampler,job['path'],job['sampling_method'],d.cell)
            record=facade.acquire(pinned,streams,d,job['trial'],sampling_method=job['sampling_method'] if d.method=='A1' and d.stage=='train' else None)
            engine_seconds=time.perf_counter()-started
            # This durable event precedes even the price lock/cache lookup.
            append(log,dict(event='BIT_SAVED',ticket=ticket,domain=d,record=record,engine_seconds=engine_seconds))
            if _STOP_EVENT is not None and _STOP_EVENT.is_set():raise RuntimeError('campaign STOP after bit preservation')
            t=time.perf_counter();lock_key=hashlib.sha256(dumps((job['path'],d.axis)).encode()).hexdigest()
            (out/'cache'/'locks').mkdir(parents=True,exist_ok=True)
            with (out/'cache'/'locks'/lock_key).open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                cost,hit=prices.price(job['path'],d.axis,compiler,cache=cache)
            price_seconds=time.perf_counter()-t
            if type(cost.rz_count) is not int or cost.rz_count<0:raise ValueError('invalid full RZ price')
            metrics={k:getattr(cost,k) for k in ('rz_count','rz_depth','cx_count','cx_depth','total_depth','circuit_size','qubit_count','compiler_settings_hash','actual_circuit_fingerprint')}
            result=dict(ticket=ticket,record=record,metrics=metrics,cache_hit=hit,counts=counts,
                engine_seconds=engine_seconds,price_seconds=price_seconds,compile_seconds=compile_seconds,
                worker_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
            append(log,dict(event='PRICE_SAVED',domain=d,**result));return result
    except Exception as exc:
        result=dict(ticket=ticket,error_type=type(exc).__name__,reason=str(exc),counts=counts,
            bit_saved=record is not None,engine_seconds=engine_seconds,
            compile_seconds=compile_seconds,elapsed_worker_seconds=time.perf_counter()-started,
            worker_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        append(log,dict(event='WORKER_STOP',domain=d,**result));return result


def available_workers():
    def read():return list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:]))
    before=read();time.sleep(.25);after=read();delta=[b-a for a,b in zip(before,after)]
    idle=delta[3]+delta[4];total=sum(delta[:8])
    available=min(len(os.sched_getaffinity(0)),math.floor(len(os.sched_getaffinity(0))*idle/total))
    if available<1:raise RuntimeError('no observed idle CPU for noninterfering execution')
    return available


class Campaign:
    def __init__(self,out,services,streams,pool,workers):
        self.out=Path(out);self.services=services;self.sampler=services[2];self.scopes=services[3]
        self.streams,self.pool,self.workers=streams,pool,workers;self.results={};self.costs=[];self.trials=[]
        self.journal=self.out/'JOURNAL.jsonl';self.words=self.out/'engine_internal_words.jsonl'
        self.classical=dict(generation=0.,fit=0.,correction=0.,compile=0.,cache=0.,engine_simulation=0.,setup=0.)

    def group(self,method,axis,stage,quotas):
        waiting=iter((n3.Domain(method,axis,stage,key),trial) for key,count in quotas for trial in range(count))
        completed={key:[] for key,_ in quotas};metrics={key:[] for key,_ in quotas};pending={};last=time.monotonic()
        def submit():
            try:d,trial=next(waiting)
            except StopIteration:return False
            job=prepare_trial(self.sampler,self.streams,d,trial,self.journal,self.words)
            self.classical['generation']+=job['generation_seconds']
            pending[self.pool.submit(worker_trial,job)]=(d,trial);return True
        for _ in range(self.workers):
            if not submit():break
        while pending:
            done,_=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
            for future in sorted(done,key=lambda f:(pending[f][0].token,pending[f][1])):
                d,trial=pending.pop(future);result=future.result()
                if 'reason' in result:
                    if _STOP_EVENT is not None:_STOP_EVENT.set()
                    append(self.journal,dict(event='CAMPAIGN_STOP',**result));raise RuntimeError(result['reason'])
                completed[d.cell].append((trial,result['record']));metrics[d.cell].append(result['metrics']['rz_count'])
                self.trials.append(dict(domain=d,trial=trial,**result))
                self.classical['engine_simulation']+=result['engine_seconds'];self.classical['compile']+=result['compile_seconds']
                self.classical['cache']+=result['price_seconds']-result['compile_seconds']
                record=result['record'];cost=result['metrics']
                self.costs.append(cv.TrialCost(self.scopes[axis],record.stream_id,record.trial_id,stage,cv.FULL_SCOPE,
                    (None,F(cost['cx_count']),F(cost['rz_count']),None,F(cost['total_depth'])),F(0),F(0),result['worker_peak_RSS_bytes']))
                append(self.journal,dict(event='TRIAL_COMPLETE',ticket=result['ticket'],domain=d,trial=trial,rz_count=cost['rz_count']))
            for _ in range(len(done)):
                if not submit():break
            if time.monotonic()-last>=15:
                append(self.journal,dict(event='RESOURCE',load=os.getloadavg(),memory=[x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith(('MemAvailable:','MemTotal:'))]))
                print(dumps(dict(event='PROGRESS',method=method,axis=axis,stage=stage,completed=sum(len(v) for v in completed.values()),pending=len(pending))),flush=True);last=time.monotonic()
        rows=[]
        for key,count in quotas:
            batch=cv.Batch(self.scopes[axis],stage,tuple(r for _,r in sorted(completed[key])))
            if len(batch.records)!=count:raise ValueError('quota not complete')
            rows.append((key,batch))
        save(self.out/f'{method}_{axis}_{stage}.json',dict(batches=rows,RZ=metrics))
        return rows,tuple(F(sum(metrics[key]),len(metrics[key])) for key,_ in quotas)

    def learn(self,axis):
        rows,_=self.group('A1',axis,'train',[(k,TRAIN) for k,_ in self.sampler.cells('B1-OL')])
        training=n3.merge_training([b for _,b in rows]);t=time.perf_counter()
        # Same fixed OLS/rank/range checks as n3.freeze, separated only so the
        # actual coefficients/rank are saved even when the range guard fails.
        import numpy as np
        model=cv.fit_and_freeze(self.sampler.bank,training)
        rank=int(np.linalg.matrix_rank([[1,*self.sampler.bank.values(r.events)] for r in training.records]))
        width=model.output_range.hi-model.output_range.lo
        self.classical['fit']+=time.perf_counter()-t
        save(self.out/f'A1_{axis}_FROZEN.json',dict(coefficients=model.coefficients,rank=rank,range=model.output_range,width=width,training_count=len(training.records),guard=4,fit_seconds=time.perf_counter()-t))
        if rank!=4:raise ValueError('rank deficient fixed controls; STOP')
        if width>4:raise ValueError('CV range guard exceeded; STOP')
        return model

    def execute(self):
        models={axis:self.learn(axis) for axis in ('X','Y')}
        pilot={}
        for method in ('B1-O','B1-OL'):
            for axis in ('X','Y'):
                pilot[method,axis]=self.group(method,axis,'train',[(k,TRAIN) for k,_ in self.sampler.cells(method)])
        error=F(self.services[4]['total_engine_coordinate_upper_exact'])
        for method in n3.METHODS:
            for axis in ('X','Y'):
                rows,_=self.group(method,axis,'cal',[(k,CAL) for k,_ in self.sampler.cells(method)])
                pilot_rows,prices=pilot.get((method,axis),([], (F(1),)))
                t=time.perf_counter()
                plan=n3.plan_production(self.sampler,method,rows,epsilon=EPSILON,engine_error=error,alpha=ALPHA,cap=CAP,
                    pilot_prices=prices,units=8,model=models[axis] if method=='A1' else None,pilot_batches=tuple(b for _,b in pilot_rows))
                self.classical['correction']+=time.perf_counter()-t
                save(self.out/f'{method}_{axis}_PLAN.json',dict(cells=plan.cells,calibrations=plan.calibrations,epsilon=plan.epsilon,engine_error=plan.engine_error,alpha=plan.alpha,allocation_seconds=time.perf_counter()-t))
                production,_=self.group(method,axis,'prod',[(k,n) for k,p,n,e in plan.cells])
                t=time.perf_counter();mean=n3.production_mean(plan,self.sampler,production);self.classical['correction']+=time.perf_counter()-t
                result=dict(status='SUFFICIENT_CONFIDENCE_COMPLETED',mean_exact=str(mean),mean_preview=float(mean),
                    epsilon=str(EPSILON),engine_error=str(error),statistical_budget=str(sum(p*e for k,p,n,e in plan.cells)),
                    alpha_method_axis=str(ALPHA/8),production_shots=sum(n for k,p,n,e in plan.cells),
                    target='N1 actual Q / selected full / saved DF ground',truth_error_measured=False,correction_seconds=time.perf_counter()-t)
                self.results[method,axis]=result;save(self.out/f'{method}_{axis}_RESULT.json',result)

    def summary(self):
        by={}
        for method in n3.METHODS:
            stages={};costs=[c for c in self.costs if json.loads(c.stream_id)[0]==method]
            for axis in ('X','Y'):
                for stage in ('train','cal','prod'):
                    cells=[c for c in costs if c.scope.axis==axis and c.stage==stage]
                    stages[axis+':'+stage]=dict(shots=len(cells),RZ=sum(int(c.quantum[2]) for c in cells))
            by[method]=dict(stages=stages,total_shots=len(costs),realized_RZ=sum(int(c.quantum[2]) for c in costs),
                results={axis:self.results.get((method,axis),dict(status='INCOMPLETE_NO_PRECISION_CLAIM')) for axis in ('X','Y')})
        save(self.out/'CAMPAIGN_RESULT.json',dict(methods=by,classical_worker_seconds=self.classical,
            parent_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            max_single_worker_peak_RSS_bytes=max((r['worker_peak_RSS_bytes'] for r in self.trials),default=None),
            cache_hits=sum(r['cache_hit'] for r in self.trials),cache_misses=sum(not r['cache_hit'] for r in self.trials),
            expected_cost_guarantee='UNKNOWN; realized campaign only',CV_improvement='UNDETERMINED unless all comparator confidence results complete'))


def reconcile(out):
    """Reconcile ALL durable worker records, including interrupted group tails.

    Metadata/cost accounting only. No probabilities, state actions, RNG,
    calibration or N are reevaluated here.
    """
    out=Path(out);bits={};priced={};calls=Counter();started=set();peaks={};stages={};failures=[]
    for path in sorted(out.glob('worker_*.jsonl')):
        for line in path.read_text().splitlines():
            row=json.loads(line);event=row['event'];ticket=row['ticket']
            if event=='CALL':calls[row['name']]+=1
            elif event=='WORKER_STARTED':started.add(ticket)
            elif event=='BIT_SAVED':
                if ticket in bits:raise ValueError('duplicate persisted bit')
                bits[ticket]=row
            elif event=='PRICE_SAVED':
                if ticket not in bits or ticket in priced:raise ValueError('price before bit or duplicate price')
                if bits[ticket]['record']!=row['record']:raise ValueError('price/measurement record mismatch')
                priced[ticket]=row
            elif event=='WORKER_STOP':failures.append(row)
            if 'worker_peak_RSS_bytes' in row:peaks[path.name]=max(peaks.get(path.name,0),row['worker_peak_RSS_bytes'])
    def cell(domain):
        key=domain['method']+':'+domain['axis']+':'+domain['stage']+':'+domain['cell']
        if key not in stages:stages[key]=dict(bits_saved=0,priced_shots=0,RZ_priced=0,cache_hits=0,cache_misses=0,
            engine_worker_seconds=0.,compile_worker_seconds=0.,price_cache_build_worker_seconds=0.,generation_parent_seconds=0.)
        return stages[key]
    for row in bits.values():
        s=cell(row['domain']);s['bits_saved']+=1;s['engine_worker_seconds']+=row['engine_seconds']
    for row in priced.values():
        s=cell(row['domain']);s['priced_shots']+=1;s['RZ_priced']+=row['metrics']['rz_count']
        s['cache_hits' if row['cache_hit'] else 'cache_misses']+=1
        s['compile_worker_seconds']+=row['compile_seconds'];s['price_cache_build_worker_seconds']+=row['price_seconds']-row['compile_seconds']
    planned=paths=0
    if (out/'JOURNAL.jsonl').exists():
        for line in (out/'JOURNAL.jsonl').read_text().splitlines():
            row=json.loads(line)
            if row['event']=='TRIAL_PLANNED':planned+=1
            if row['event']=='PATH_FIXED':paths+=1;cell(row['domain'])['generation_parent_seconds']+=row['generation_seconds']
    total_words=0
    if (out/'engine_internal_words.jsonl').exists():
        for line in (out/'engine_internal_words.jsonl').read_text().splitlines():
            row=json.loads(line);total_words+=len(row['path_words'])+(row['measurement_word'] is not None)
    methods={}
    for method in n3.METHODS:
        relevant={k:v for k,v in stages.items() if k.startswith(method+':')}
        nb=sum(v['bits_saved'] for v in relevant.values());np=sum(v['priced_shots'] for v in relevant.values())
        result={}
        for axis in ('X','Y'):
            file=out/f'{method}_{axis}_RESULT.json'
            result[axis]=json.loads(file.read_text()) if file.exists() else dict(status='INCOMPLETE_NO_PRECISION_CLAIM')
        methods[method]=dict(stages=relevant,bits_saved=nb,priced_shots=np,realized_RZ_priced=sum(v['RZ_priced'] for v in relevant.values()),
            realized_RZ_complete=(nb==np),unpriced_bits=nb-np,results=result)
    save(out/'ACCOUNTING.json',dict(methods=methods,all_worker_tails_included=True,
        failed_trial_classical_records=failures,
        parent_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        per_worker_peak_RSS_bytes=peaks,sum_worker_RSS_highwaters_bytes=sum(peaks.values()),
        RSS_note='sum of per-process highwaters is conservative and double-counts shared pages; not simultaneous physical memory',
        expected_cost_guarantee='UNKNOWN; realized prices only'))
    save(out/'CALL_COUNTS.json',dict(campaigns=1,trial_planned=planned,path_fixed=paths,workers_started=len(started),
        bits_saved=len(bits),priced_trials=len(priced),uniform53_words=total_words,actual_worker_calls=dict(calls),
        preparation=0,CDF_recompute=0,state_solve=0,spectral_recompute=0,engine_requalification=0,retry=0))
    return methods


def run(root,out):
    global _SERVICES,_STOP_EVENT
    root,out=Path(root),Path(out);source=kernel.n2.git(root,'rev-parse','HEAD')
    dirty=kernel.n2.git(root,'status','--porcelain=v1','--untracked-files=all')
    if dirty:raise ValueError('scientific SOURCE worktree not clean')
    lock=kernel.read(root/DOC/'PROFILE_LOCK.json')
    if kernel.canonical(profile(root))!=kernel.canonical(lock):raise ValueError('execution SOURCE/environment lock mismatch')
    binding=n3.inspect_certified_bundle(root)  # static only, before numerical decoding
    out.mkdir(parents=True,exist_ok=False)
    marker=Path('/tmp/a1-n3-science-invocation-20261106.json')
    with marker.open('x') as f:f.write(dumps(dict(source=source,output=str(out),campaign_invocations=1))+'\n')
    save(out/'SOURCE.json',dict(source_commit=source,contract_commit=BASE,binding=binding,conditions=dict(epsilon='1/8',alpha='1/20',train_per_cell=64,cal_per_cell=512,production_cap=8192,J=8,master_seed=SEED,path_words=64),
        campaigns=1,prohibited_calls={k:0 for k in ('preparation','CDF','state_solve','spectral','engine_qualification','other_round','native_synthesis')},C1='frozen'))
    campaign=None;pool=None;started=time.monotonic();status='STOPPED';reason=None
    try:
        with forbidden_science():
            t=time.perf_counter();_SERVICES=n3.load_certified_services(root);setup=time.perf_counter()-t
            save(out/'SETUP.json',dict(saved_bundle_decode=1,setup_seconds=setup,
                b_engine_exact=_SERVICES[4]['total_engine_coordinate_upper_exact'],state_solve=0,engine_requalification=0))
            for method,count in (('B0',1),('B1-O',2),('B1-OL',5),('A1',1)):
                if len(_SERVICES[2].cells(method))!=count:raise ValueError('fixed positive stratum count mismatch')
            streams=n3.pcg64_streams(SEED,domains(),path_word_cap=64)
            save(out/'STREAMS.json',dict(master_seed=SEED,model='independent uniform53bit assumption; PCG64 not an independence proof',domains=[dict(domain=d,path_spawn_key=[2*i],bit_spawn_key=[2*i+1]) for i,d in enumerate(domains())]))
            workers=available_workers();save(out/'RESOURCE_START.json',dict(workers=workers,affinity_CPUs=len(os.sched_getaffinity(0)),load=os.getloadavg(),memory=Path('/proc/meminfo').read_text(),policy='actual idle cores; own nice+10 workers; no global settings or resource cap'))
            _STOP_EVENT=mp.get_context('fork').Event()
            pool=ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('fork'),initializer=worker_initialize,initargs=(out,))
            campaign=Campaign(out,_SERVICES,streams,pool,workers);campaign.classical['setup']=setup
            campaign.execute();status='COMPLETED'
    except BaseException as exc:
        if _STOP_EVENT is not None:_STOP_EVENT.set()
        reason=dict(type=type(exc).__name__,message=str(exc));print(dumps(dict(event='STOP',reason=reason)),flush=True)
    finally:
        if pool is not None:
            # Finish already-started fixed trials and retain their journals.
            # No next stage/new trial is submitted after the first failure.
            pool.shutdown(wait=True,cancel_futures=True)
        if campaign is not None:campaign.summary()
        reconcile(out)
        save(out/'STOP.json',dict(status='MANDATORY_STOP',campaign_status=status,reason=reason,no_retry=True,additional_campaign_authorized=False,elapsed_wall_seconds=time.monotonic()-started))
    return status
