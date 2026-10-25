"""Fixed B1 IID256 pilot. Import/metadata inspection performs no science."""
from collections import Counter,OrderedDict,defaultdict
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import json,math,os,time

from . import b0_native as b0
from .followup_io import FollowupLedger,registered_scope,preflight
from .launcher import Evidence,_encode,rejection
from .scope import STOP,Stage1Failure

SAMPLES=256
SEED_POLICY='sha256-b1-v1/master41/candidate/T/sample/occurrence/step/PCG64'
OUTPUTS=('binding.json','inputs.json','graph.json','events.jsonl','trajectories.jsonl',
         'samples.jsonl','failure.json','partial_results.json','result.json','status.json','STOP.json')
LIMITS=dict(total_bytes=128*1024**2,json_document_bytes=8*1024**2,event_record_bytes=64*1024,
            files=len(OUTPUTS),stop_reserve_bytes=512*1024)

def sample_id(row,index):return b0.case_key(row)+':'+str(index)

def subseed(row,index,occurrence,step):
    text=json.dumps([SEED_POLICY,41,row['candidate_id'],str(row['case']['total_time']),index,occurrence,step],separators=(',',':'))
    return int.from_bytes(sha256(text.encode()).digest()[:16],'big')

def resource_plan():
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    workers=len(os.sched_getaffinity(0))
    # A B0 run retained96 compiled copies in1.51GB. B1 keeps only a small LRU
    # of full circuits per worker, derived from current RAM, not a science cap.
    budget=int(available*.70/workers)
    if budget<640*1024**2:workers=max(1,int(available*.70/(640*1024**2)));budget=int(available*.70/workers)
    memo=max(1,(budget-512*1024**2)//(64*1024**2))
    return dict(workers=workers,memo_entries=memo,available_bytes=available,
        worker_memory_budget=budget,stop_available_bytes=int(available*.10),
        child_threads=1,policy='affinity_CPUs_RAM_70percent_512MiB_base_64MiB_per_cached_trajectory')

def scope(workers):
    rows=b0.cases();calls=[];steps=0
    for row in rows:
        stages=b0.trajectory(row,0)
        steps+=SAMPLES*sum(spec.rte_steps for spec,word in stages)
        for i in range(SAMPLES):calls.extend((('b1.sample',sample_id(row,i)),('b1.receive',sample_id(row,i))))
        calls.extend(('b1.statistics',b0.case_key(row)+':'+axis) for axis in ('cosine','sine'))
    calls.extend(('b1.worker',str(i)) for i in range(workers))
    result=registered_scope('b1',calls,cases=12,samples_per_case=SAMPLES,trajectories=3072,
        wrapper_build_requests=6144,compile_requests=6144,operator_gates=18432,expectations=12288,
        cost_requests=6144,sampled_short_steps=steps,workers=workers,compiler=b0.PROFILE,
        master_seed=41,seed_policy=SEED_POLICY,precision='complex128',atol=1e-12,rtol=0,
        phase_fit=False,statistics='CompiledMetricAccumulator(weighted=False);canonical_sample_order',
        correctness_policy='all gates per trajectory before its costs;final expectations only after all samples PASS',
        primitive_cache_policy='bounded runtime expansion inside B0 renderer/compiler closures;actual worker counters saved')
    result['max_event_records']+=workers+1
    result['parent_native_boundaries']=dict(fixture=1,split=1,prepare=1,full_inputs=1,input_gate=1,config=76,event=steps)
    return result

class Cancelled(Exception):pass

class NativeLedger:
    """Record B0 closure calls locally; no old B0 representative scope reused."""
    allowed=frozenset('fixture split prepare full_inputs input_gate config event event_operator exponential block_build rte_build'.split())
    def __init__(self,stop=None,bounds=None):self.stop=stop;self.rows=[];self.counts=Counter();self.bounds=bounds
    def call(self,name,function,*,identity=None,summary=None):
        if self.stop is not None and self.stop.is_set():raise Cancelled()
        kind=name.removeprefix('b0.')
        if kind not in self.allowed:raise Stage1Failure('UNREGISTERED_B1_NATIVE_CLOSURE')
        if self.bounds is not None and self.counts[kind]>=self.bounds.get(kind,0):raise Stage1Failure('B1_NATIVE_CALL_SCOPE_EXCEEDED')
        self.counts[kind]+=1
        return function()

class ParentLedger(FollowupLedger):
    def partial_rows(self):
        return [dict(completed_samples=len(self.rows),authoritative_stream='samples.jsonl',
                     **getattr(self,'progress',{}))]

def draw(native,row,index,iterator=None,stages=None):
    if iterator is None:
        from trotterlib.rte import iter_sample_rte_events
        iterator=iter_sample_rte_events
    components=native.prep.rte_preparation.symbolic_tail.components
    indices={c.component_id:i for i,c in enumerate(components)}
    words=[];details=[]
    for occurrence,(spec,unused) in enumerate(b0.trajectory(row,0) if stages is None else stages):
        config,law=native.config(spec);word=[];events=[]
        for step in range(spec.rte_steps):
            seed=subseed(row,index,occurrence,step)
            event=next(iterator(components,law,sample_count=1,seed=seed))
            descriptor=(event.taylor_order,tuple(indices[c] for c in (event.rotation_component_id,*event.product_component_ids)))
            if native.event(spec,descriptor)!=event:raise Stage1Failure('B1_SAMPLED_EVENT_LAW_MISMATCH')
            word.append(descriptor)
            events.append(dict(seed=seed,order=event.taylor_order,indices=descriptor[1],
                probability=event.event_probability,rotation_angle=event.rotation_angle,
                phase=event.phase,product_sign=event.product_sign_phase))
        words.append((spec,tuple(word)))
        details.append(dict(occurrence=occurrence,spec=asdict(spec),events=events,
            orders=law.orders,order_probabilities=law.order_probabilities,
            exact_finite_distribution=law.exact_finite_distribution,
            dimensionless_step_time=law.dimensionless_step_time))
    key=b0.fingerprint((row['case'],row['rte_parameters'],[(asdict(s),w) for s,w in words],
        b0.PROFILE,native.run['inputs']['input_data_sha256'],native.run['source'],
        'full-controlled/ancilla2/mode0-prepared/paired-X-Y'))
    return dict(id=sample_id(row,index),row=row,index=index,words=words,details=details,key=key)

def allocate(jobs,workers):
    groups=defaultdict(list)
    for job in jobs:groups[job['key']].append(job)
    bins=[[] for _ in range(workers)];loads=[0.]*workers
    weights={}
    for job in jobs:
        key=b0.case_key(job['row'])
        if key not in weights:
            plan=b0.outer_plan(job['row']);weights[key]=sum(len(s.append_operations()) for s in plan.body if isinstance(s,b0.DStage))
    def weight(group):
        return weights[b0.case_key(group[0]['row'])]*(1+.02*len(group))
    for group in sorted(groups.values(),key=lambda g:(-weight(g),g[0]['key'])):
        slot=min(range(workers),key=lambda i:(loads[i],i));bins[slot].extend(group);loads[slot]+=weight(group)
    return bins

def circuit_blob(circuit):
    import io,gzip,base64
    from qiskit import qpy
    stream=io.BytesIO();qpy.dump(circuit,stream)
    return dict(encoding='gzip+base64+qpy',data=base64.b64encode(gzip.compress(stream.getvalue(),mtime=0)).decode())

def evaluate(native,job,memo,limit,cache_stats,stop):
    """Cache retains actual QCs and operators, never metrics alone."""
    key=job['key'];circuits={};entry=None;tick=time.monotonic();cpu=time.process_time()
    try:
        hit=key in memo
        if hit:
            entry=memo.pop(key);memo[key]=entry;cache_stats['hits']+=2
        else:
            cache_stats['misses']+=2
            row=job['row'];words=job['words']
            circuits['system']=native.render(row,None,False,sampled_words=words)
            circuits['controlled']=native.render(row,None,True,sampled_words=words)
            reference=native.reference(row,None,sampled_words=words)
            system=native.Operator(circuits['system']).data;controlled=native.Operator(circuits['controlled']).data
            entry=dict(circuits=circuits,reference=reference,system=system,controlled=controlled,axes={})
            for axis in ('cosine','sine'):
                if stop.is_set():raise Cancelled()
                raw=native.wrapper(circuits['controlled'],axis);circuits['raw_'+axis]=raw
                ct=time.monotonic();cc=time.process_time();compiled=native.compile(raw)
                cache_stats['compiler_wall_seconds']+=time.monotonic()-ct;cache_stats['compiler_cpu_seconds']+=time.process_time()-cc
                circuits['compiled_'+axis]=compiled
                if stop.is_set():raise Cancelled()
                entry['axes'][axis]=dict(raw=raw,compiled=compiled,
                    reference=native.wrapper_reference(reference,axis),
                    raw_operator=native.Operator(raw.remove_final_measurements(inplace=False)).data,
                    compiled_operator=native.Operator(compiled.remove_final_measurements(inplace=False)).data)
            memo[key]=entry
            if len(memo)>limit:memo.popitem(last=False);cache_stats['evictions']+=1
        circuits=entry['circuits'];reference=entry['reference']
        errors=dict(system=native.check(entry['system'],reference),
                    controlled=native.check(entry['controlled'],native.controlled(reference)))
        axes={}
        # Both axes must pass all numerical gates before any cost for this sample.
        for axis,a in entry['axes'].items():
            if stop.is_set():raise Cancelled()
            axes[axis]=dict(operator_errors=dict(**errors,
                raw_wrapper=native.check(a['raw_operator'],a['reference']),
                compiled_wrapper=native.check(a['compiled_operator'],a['reference'])),
                raw_signal=native.expectation(a['raw'],a['reference'],operator=a['raw_operator']),
                compiled_signal=native.expectation(a['compiled'],a['reference'],operator=a['compiled_operator']))
        for axis,a in entry['axes'].items():
            if stop.is_set():raise Cancelled()
            if 'metrics' not in a:a['metrics']=native.measure(a['compiled']);cache_stats['actual_metric_queries']+=1
            axes[axis]['metrics']=a['metrics']
        return dict(id=job['id'],candidate=job['row']['candidate_id'],T=str(job['row']['case']['total_time']),
            sample=job['index'],physical_key=key,cache_hit=hit,axes=axes,
            correctness='PASS_EMPIRICAL_NO_PHASE_FIT',wall_seconds=time.monotonic()-tick,
            cpu_seconds=time.process_time()-cpu)
    except Cancelled:raise
    except BaseException as error:
        def matrix(value):return [[dict(real=float(z.real) if math.isfinite(z.real) else None,
            imag=float(z.imag) if math.isfinite(z.imag) else None) for z in row] for row in value]
        operators={}
        if entry is not None:
            for name in ('reference','system','controlled'):operators[name]=matrix(entry[name])
            for axis,a in entry['axes'].items():
                for name in ('reference','raw_operator','compiled_operator'):operators[axis+'_'+name]=matrix(a[name])
        def saved(qc):
            try:return circuit_blob(qc)
            except BaseException as save_error:return dict(serialization_error=type(save_error).__name__,message=str(save_error))
        error.b1_context=dict(id=job['id'],physical_key=key,diagnostics=getattr(error,'diagnostics',{}),
            operator_matrices=operators,
            circuits={k:saved(v) for k,v in circuits.items()})
        raise
    finally:
        native.compile_cache.clear()
        for k in list(native.cache):
            if k[0] not in ('config','block_build','exponential'):native.cache.pop(k)

def worker(slot,jobs,run,resources,queue,stop):
    # Process-local thread settings coordinate all available CPU processes.
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS'):
        os.environ[name]='1'
    import resource,traceback
    steps=sum(sum(spec.rte_steps for spec,word in job['words']) for job in jobs)
    stages=sum(len(job['words']) for job in jobs)
    bounds=dict(fixture=1,split=1,prepare=1,full_inputs=1,input_gate=1,config=76,
        event=steps,event_operator=steps,exponential=206,block_build=372,rte_build=2*stages)
    ledger=NativeLedger(stop,bounds);cache_stats=Counter();tick=time.monotonic();cpu=time.process_time()
    try:
        native=b0.Native(run,ledger);memo=OrderedDict()
        queue.put(dict(type='ready',worker=slot,input_diagnostics=native.input_diagnostics))
        for job in jobs:
            if stop.is_set():raise Cancelled()
            record=evaluate(native,job,memo,resources['memo_entries'],cache_stats,stop)
            record['worker']=slot;queue.put(dict(type='sample',record=record))
            cache_stats['completed_samples']+=1
    except Cancelled:pass
    except BaseException as error:
        stop.set();queue.put(dict(type='failed',worker=slot,error_type=type(error).__name__,
            reason=getattr(error,'code','B1_TECHNICAL_EXCEPTION'),message=str(error),
            traceback=traceback.format_exc(),context=getattr(error,'b1_context',{}),native_rows=ledger.rows))
    finally:
        queue.put(dict(type='done',worker=slot,closure_calls=dict(ledger.counts),cache=dict(cache_stats),
            closure_bounds=bounds,
            wall_seconds=time.monotonic()-tick,cpu_seconds=time.process_time()-cpu,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

def statistics(records):
    from trotterlib.rte_compiled_cost import CompiledMetricAccumulator
    groups={};overlap=[]
    for row in b0.cases():
        key=b0.case_key(row);selected=sorted((r for r in records if r['candidate']==row['candidate_id'] and r['T']==str(row['case']['total_time'])),key=lambda r:r['sample'])
        if len(selected)!=SAMPLES:raise Stage1Failure('B1_INCOMPLETE_IID_SAMPLE_SET')
        for axis in ('cosine','sine'):
            acc=CompiledMetricAccumulator(weighted=False)
            for record in selected:
                metrics=record['axes'][axis]['metrics']
                if metrics['measurement_count']!=1:raise Stage1Failure('B1_FULL_WRAPPER_MEASUREMENT_MISMATCH')
                acc.update(SimpleNamespace(**metrics))
            metrics={name:dict(asdict(s),sample_count=acc.count,
                relative_SE=s.standard_error/abs(s.mean) if s.mean else None,
                relative_SE_2pct_target_met=s.standard_error/abs(s.mean)<=.02 if s.mean else None,
                normal95_reference=[s.mean-1.96*s.standard_error,s.mean+1.96*s.standard_error]) for name,s in acc.finalize()}
            metrics['measurement_count']=dict(mean=1.,unbiased_sample_variance=0.,standard_error=0.,minimum=1.,maximum=1.,sample_count=SAMPLES,relative_SE=0.,normal95_reference=[1.,1.])
            groups[key+':'+axis]=metrics
    from itertools import combinations
    for T in ('9/50','-9/50'):
        for axis in ('cosine','sine'):
            for left,right in combinations(b0.IDS,2):
                for metric in ('rz_count','cx_count'):
                    a=groups[left+':'+T+':'+axis][metric]['normal95_reference'];b=groups[right+':'+T+':'+axis][metric]['normal95_reference']
                    overlap.append(dict(T=T,axis=axis,metric=metric,left=left,right=right,
                        reference_intervals_overlap=max(a[0],b[0])<=min(a[1],b[1])))
    return dict(groups=groups,interval_comparisons=overlap,policy='IID unweighted Welford;normal95 is diagnostic,not rigorous coverage')

def coverage(jobs):
    groups={}
    for job in jobs:
        key=b0.case_key(job['row']);group=groups.setdefault(key,dict(K=job['row']['rte_parameters']['K'],occurrences={}))
        for item in job['details']:
            entry=group['occurrences'].setdefault(str(item['occurrence']),dict(spec=item['spec'],orders=item['orders'],theoretical_order_probabilities=item['order_probabilities'],observed=Counter(),components=Counter()))
            for event in item['events']:
                entry['observed'][str(event['order'])]+=1
                entry['components'].update(str(i) for i in event['indices'])
    for group in groups.values():
        group['max_order_observed']=any(e['observed'][str(group['K'])]>0 for e in group['occurrences'].values())
        group['status']='OBSERVED' if group['max_order_observed'] else 'INCONCLUSIVE_RARE_EVENT_COVERAGE'
    return groups

def run_once(run,resources,*,native_factory=b0.Native,draw_function=draw,worker_function=worker,statistics_function=statistics,mock_only=False):
    import multiprocessing as mp,queue as queue_module
    out=run['output'];out.parent.mkdir(parents=True,exist_ok=True);out.mkdir(mode=0o700)
    ledger=ParentLedger(run['scope']);evidence=Evidence(out,outputs=OUTPUTS,limits=LIMITS,mock_only=mock_only)
    evidence.precision_status='EMPIRICAL_COMPLEX128_NO_GAMMA_CERTIFICATE';ledger.sink=evidence.event
    records=[];done={};ready={};failure=None;processes=[];streams=[];stop=None;tick=time.monotonic()
    try:
        evidence.write('binding.json',{k:v for k,v in run.items() if k not in ('root','output','inputs')})
        evidence.write('inputs.json',run['inputs']);evidence.write('graph.json',dict(cases=b0.cases(),scope=run['scope'],resources=resources))
        trajectories=(out/'trajectories.jsonl').open('x');samples=(out/'samples.jsonl').open('x');streams=[trajectories,samples]
        def append(stream,value):
            text=_encode(value)+'\n'
            if evidence.total_bytes+len(text.encode())>LIMITS['total_bytes']-LIMITS['stop_reserve_bytes']:raise Stage1Failure('B1_OUTPUT_SCOPE_EXCEEDED')
            stream.write(text);stream.flush();os.fsync(stream.fileno());evidence.total_bytes+=len(text.encode())
        parent_ledger=NativeLedger(bounds=run['scope']['parent_native_boundaries']);native=native_factory(run,parent_ledger);jobs=[]
        ledger.emit(dict(event='parent_input',diagnostics=native.input_diagnostics))
        for row in b0.cases():
            stages=b0.trajectory(row,0)
            for i in range(SAMPLES):
                job=ledger.call('b1.sample',lambda row=row,i=i:draw_function(native,row,i,stages=stages),identity=sample_id(row,i))
                append(trajectories,{k:v for k,v in job.items() if k not in ('row','words')});jobs.append(job)
        rare=coverage(jobs);components=[dict(id=c.component_id,probability=c.probability,sign=c.coefficient_sign,support=c.diagonal_pauli_support) for c in native.prep.rte_preparation.symbolic_tail.components]
        del native
        batches=allocate(jobs,resources['workers']);ctx=mp.get_context('spawn');stop=ctx.Event();queue=ctx.Queue(maxsize=2*resources['workers'])
        for slot,batch in enumerate(batches):
            process=ctx.Process(target=worker_function,args=(slot,batch,run,resources,queue,stop));process.start();processes.append(process)
        while len(done)<len(processes):
            try:message=queue.get(timeout=1.)
            except queue_module.Empty:
                dead=[i for i,p in enumerate(processes) if p.exitcode is not None and i not in done]
                if dead:raise RuntimeError('B1 worker exited unexpectedly: '+str(dead))
                available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
                if available<resources['stop_available_bytes']:raise RuntimeError('B1 shared-memory pressure;stopping own job')
                continue
            if message['type']=='sample':
                record=message['record'];ledger.call('b1.receive',lambda:None,identity=record['id']);append(samples,record);records.append(record);ledger.rows.append(record['id'])
            elif message['type']=='ready':
                ready[message['worker']]=message['input_diagnostics'];ledger.emit(dict(event='worker_input',**message))
            elif message['type']=='done':
                done[message['worker']]=message
                ledger.call('b1.worker',lambda:None,identity=str(message['worker']))
            elif message['type']=='failed':
                if failure is None:failure=message;evidence.write('failure.json',message)
                stop.set()
        for process in processes:process.join()
        if failure is not None:raise Stage1Failure('B1_WORKER_FAILURE')
        if len(records)!=3072:raise Stage1Failure('B1_INCOMPLETE_SAMPLE_RESULTS')
        if len(ready)!=resources['workers']:raise Stage1Failure('B1_INCOMPLETE_INPUT_CHECKS')
        stats=statistics_function(records)
        for key in stats['groups']:ledger.call('b1.statistics',lambda:None,identity=key)
        ledger.finish()
        inconclusive=any(not v['max_order_observed'] for k,v in rare.items() if v['K']==4)
        result=dict(study='B1',source=run['source'],sampling_policy=SEED_POLICY,master_seed=41,
            statistics=stats,rare_event_coverage=rare,components=components,parent_closure_calls=dict(parent_ledger.counts),
            worker_input_diagnostics=ready,
            workers=[done[i] for i in sorted(done)],resources=resources,wall_seconds=time.monotonic()-tick,
            scientific_status='INCONCLUSIVE_RARE_EVENT_COVERAGE' if inconclusive else 'COMPLETE',
            fixed_sample_set_complete=True,gamma=None,shots=None,u_num=None,total_resource_G=None,
            raw_samples='samples.jsonl',trajectories='trajectories.jsonl',stop=STOP)
        return evidence.finish(dict(status='COMPLETED_DESCRIPTIVE_MANDATORY_STOP'),ledger,result=result)
    except BaseException as error:
        if stop is not None:stop.set()
        if failure is None:
            import traceback
            try:evidence.write('failure.json',dict(error_type=type(error).__name__,reason=getattr(error,'code','B1_TECHNICAL_EXCEPTION'),message=str(error),traceback=traceback.format_exc(),diagnostics=getattr(error,'diagnostics',{})))
            except BaseException:pass
        # Never retry/resume or start another output after science has begun.
        # Drain the multiprocessing pipe while stopping; joining without this
        # can deadlock a child's queue feeder and lose already-complete records.
        while any(process.is_alive() for process in processes):
            try:
                message=queue.get(timeout=1.)
                if message['type']=='sample':
                    record=message['record']
                    if record['id'] not in ledger.rows:
                        ledger.call('b1.receive',lambda:None,identity=record['id']);append(samples,record);ledger.rows.append(record['id'])
                elif message['type']=='done':done[message['worker']]=message
            except queue_module.Empty:pass
            except BaseException:pass
        for process in processes:process.join()
        if stop is not None:
            while True:
                try:
                    message=queue.get_nowait()
                    if message['type']=='sample' and message['record']['id'] not in ledger.rows:
                        record=message['record'];ledger.call('b1.receive',lambda:None,identity=record['id']);append(samples,record);ledger.rows.append(record['id'])
                    elif message['type']=='done':done[message['worker']]=message
                except queue_module.Empty:break
                except BaseException:break
        ledger.progress=dict(workers=done,worker_inputs=ready,resources=resources,
            elapsed_wall_seconds=time.monotonic()-tick)
        return evidence.finish(dict(status='FAILED_MANDATORY_STOP'),ledger,error=error)
    finally:
        for stream in streams:stream.close()

def main():
    import argparse
    parser=argparse.ArgumentParser(description='Fixed B1 IID256,one attempt;no automatic retry/resume')
    mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--inspect',action='store_true');mode.add_argument('--execute',action='store_true')
    parser.add_argument('--source-commit');parser.add_argument('--output');args=parser.parse_args()
    resources=resource_plan();s=scope(resources['workers'])
    if args.inspect:print(_encode(dict(scope=s,resources=resources,status='METADATA_ONLY_NO_SCIENCE')));return 0
    if not args.source_commit or not args.output:parser.error('--execute requires --source-commit and --output')
    root=Path(__file__).resolve().parents[3]
    try:
        run=preflight(root,args.output,args.source_commit,'b1',s)
        if run['versions']['qiskit']!='1.3.0':raise ValueError('B1 compiler requires Qiskit1.3.0')
        from .input_package import read_fixture_definition
        read_fixture_definition(root)
        import platform,sys
        run['environment']=dict(python=sys.version,executable=sys.executable,
            platform=platform.platform(),cpu_affinity=sorted(os.sched_getaffinity(0)))
        run['resources']=resources;status=run_once(run,resources)
    except BaseException as error:status=rejection(error)
    print(_encode(status));return 0 if status['status']=='COMPLETED_DESCRIPTIVE_MANDATORY_STOP' else 2
