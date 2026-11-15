"""C2 m9 finite driver: pinned legacy science, separate fixed m9 binding.

No old validator changes. Metadata -> exact predictions -> published SHA ->
two independent one-shot path subprocesses -> public receipts -> comparison.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = 'scripts/run_pauli_ld2_m9.py'
PIN = '13ab1acfaebe5f33958e1db67862f5c45725ab91'
PIN_ROOT = ROOT.parent / 'c2-pauli-m9-source-13ab'
BASE = 'docs/research/d_pauli_ld2_m9_longpoint_results_20261115'
PRIVATE = Path('/home/AbeHiromu/projects/pauli-ld2-m9-longpoint-20261115-private')
PACKET = dict(commit='714247619afb4714fd663eb15c0470ef32bdc7af', path='docs/research/df_head_pauli_tail_all_round_results_20261114/MEAN_HANDOFF.json')
MODEL = dict(commit='b74e4e536392c2bbe94cd371436615fc56c0564f',path='docs/research/d_pauli_ld2_b1_cal_results_20261115/c2_freeze/run01/FROZEN_MODELS.json')
META = dict(commit='043c7567451e8f502a1cf0f425d32efe0a50f5dc',path='docs/research/d_pauli_b1_caller_preparation_20261114/CAL_AFTER_W_METADATA.json')
COHORT = 'D_PAULI_LD2_M9_LONG_NORMAL_N2_V1'
CID = 'RMSE:978923356c4e83c897d03442fb488997d32047237444f95757658f61321b2729'
AXES = {'X':'cosine','Y':'sine'}


def blob(commit,path):
    return subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)


def read(ref):
    data=json.loads(blob(ref['commit'],ref['path']))
    for key in ref.get('selector',[]): data=data[key]
    return data


def enc(x):
    x=Fraction(x);return dict(numerator=x.numerator,denominator=x.denominator)


def exact(x): return Fraction(x['numerator'],x['denominator'])


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f: json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def same(a,b,label):
    if a!=b: raise ValueError(label)


def legacy():
    same(subprocess.check_output(['git','rev-parse','HEAD'],cwd=PIN_ROOT,text=True).strip(),PIN,'Legacy checkout HEAD differs')
    if str(PIN_ROOT/'scripts') not in sys.path:sys.path.insert(0,str(PIN_ROOT/'scripts'))
    import run_pauli_b1_prices as B1
    for p in (*B1.REPLAY_SOURCES,B1.A_PATH):same((PIN_ROOT/p).read_bytes(),blob(PIN,p),'Pinned SOURCE differs: '+p)
    return B1


def metadata():
    B1=legacy();P=B1.B0.connection();I,DC,G=B1.B0._meta()
    req,packet,dictionary=B1.canonical(2)
    physical=P.round_request(packet,9,dictionary)
    same((physical['candidate_id'],physical['prefix'],physical['q'],physical['R'],physical['K']),
         (CID,2,512,526,2),'m9 physical ID/schedule differs')
    context=I.price_context(req,9)
    same(context['physical_candidate']['id'],CID,'C canonical context differs')
    same(context['split']['tail_dictionary'],packet['dictionary_identity'],'Canonical dictionary differs')
    same(physical['T'],Fraction(3602879701896397,70368744177664),'Exact T differs')
    physical['T']=float(physical['T'])
    seeds=[[P.seed(CID,COHORT,i,outer,0) for outer in range(512)] for i in range(2)]
    old=read(META)['dictionaries']['LD2']['cells']
    old_seeds={s for c in old for row in c['step_seeds'] for s in row}
    wiring=read(dict(commit='6c4709f85d0f28ce7c6957e9a70c85e378d8cad0',path='docs/research/d_pauli_ld2_b1_w_results_20261115/JOB.json'))
    old_seeds.update(s for row in wiring['step_seeds'] for s in row)
    flat=[s for row in seeds for s in row]
    if len(set(flat))!=1024 or set(flat)&old_seeds:raise ValueError('New or old cohort seed collision')
    model=read(MODEL)
    plan=read(dict(commit='5dac74959ed9ea49465d393fc792ef5b4727edba',path='docs/research/d_pauli_ld2_b1_cal_results_20261115/JOBS/S66.json'))['plan']
    gate=B1.frozen_model_gate(plan,MODEL)  # Original CAL plan, not a modified m9 plan.
    same(model['features'],['one','pf_step_count','event_count'],'Feature order differs')
    return dict(schema='C2_LD2_M9_BINDING_V1',m=9,prefix=2,cell_id='M9_ROUND',cohort=COHORT,
        path_indices=[0,1],axes=['X','Y'],n_paths=2,compile_maximum=4,request=physical,
        T_exact=enc(Fraction(3602879701896397,70368744177664)),
        features=dict(one=enc(1),pf_step_count=enc(512),event_count=enc(269312)),
        step_seeds=seeds,dictionary_identity=packet['dictionary_identity'],
        configuration_id=req['configuration']['id'],model_source=MODEL,packet_source=PACKET,
        seed_checks=dict(new_step_seeds=1024,unique_new=True,old_cohort_overlap=False,
                         old_source=META,wiring_source='6c4709f85d0f28ce7c6957e9a70c85e378d8cad0'),
        frozen_model_gate=gate,domain={a:model['models'][a]['domain'] for a in AXES},
        legacy_source=PIN,legacy_checkout=str(PIN_ROOT),driver=SCRIPT)


def check_binding(actual,expected):
    same(actual,expected,'m9 binding/source/cohort/seed differs')


def freeze():
    m=metadata();source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    same((ROOT/SCRIPT).read_bytes(),blob(source,SCRIPT),'Commit driver SOURCE before prediction')
    model=read(MODEL);M=legacy().design_api().c_model(PIN_ROOT)
    predictions=[]
    for a in AXES:
        P=M.predict(model['models'][a],m['features'])
        predictions.append(dict(prediction_id='LD2:m9:'+a,axis=a,m=9,canonical_physical_id=CID,
            cohort=COHORT,n_paths=2,features=m['features'],predicted_RZ=enc(P),model_source=dict(MODEL,selector=['models',a])))
    result=dict(schema='C2_LD2_M9_PREOBSERVATION_PREDICTIONS_V1',binding=m,predictions=predictions,
        execution_source=dict(commit=source,path=SCRIPT,legacy_source=PIN),fit_calls=0,prediction_calls=2,
        sampling=0,compiles=0,observations_opened=0,tolerance=None,qualified_for_candidate=False,
        created_UTC=datetime.now(timezone.utc).isoformat(),status='PREDICTIONS_FIXED_BEFORE_SAMPLING')
    write(ROOT/BASE/'PREDICTIONS.json',result)
    print(json.dumps(dict(predictions=predictions,execution_source=result['execution_source'])))


class Progress:
    def __init__(self,path):self.path=Path(path);self.started=time.monotonic()
    def __call__(self,stage,state,**kw):
        value=dict(UTC=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),elapsed=time.monotonic()-self.started,
                   stage=stage,state=state,**kw)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.open('a') as f:f.write(json.dumps(value)+'\n');f.flush()
    @contextmanager
    def stage(self,name,**kw):
        start=time.monotonic();self(name,'START',**kw)
        try:yield
        except BaseException as exc:self(name,'STOP',error=type(exc).__name__+': '+str(exc));raise
        else:self(name,'DONE',seconds=time.monotonic()-start,**kw)


@contextmanager
def accelerated_observed(P,log):
    """Same functions/ordering; identical-byte hash writer + full event-matrix cache.

    Cache only pure event_matrix outputs under the complete original event JSON,
    fixed label map and n. No RNG, emitter, simulator, compiler or gate change.
    """
    from h46_b3_fast_fingerprint import use_fast_fingerprint
    from trottertracks.resource_applicability import h6_hash_reuse_v1 as H
    import trotterlib.df_trotter.circuit as circuit
    saved={};event_cache={};stats=dict(event_cache_hits=0,event_cache_misses=0)
    def patch(obj,key,value):saved[obj,key]=getattr(obj,key);setattr(obj,key,value)
    original_matrix=P.event_matrix
    def matrix(event,labels,n):
        key=(n,tuple(sorted(labels.items())),json.dumps(event.to_dict(),sort_keys=True))
        if key not in event_cache:
            event_cache[key]=original_matrix(event,labels,n);stats['event_cache_misses']+=1
        else:stats['event_cache_hits']+=1
        return event_cache[key]
    patch(P,'event_matrix',matrix)
    # pilot_core enters this context itself; wrapper-internal hash aliases stay inside it.
    patch(H,'use_fingerprint_reuse',use_fast_fingerprint)
    for name in ('bit_order_checks','sample_steps','full_space_expected','compose','index0_checks'):
        fn=getattr(P,name)
        def observed(*args,_name=name,_fn=fn,**kwargs):
            with log.stage(_name):return _fn(*args,**kwargs)
        patch(P,name,observed)
    sim=circuit.simulate_statevector;counter=[0]
    def simulate(*args,**kwargs):
        counter[0]+=1
        with log.stage('simulate_'+str(counter[0]),instructions=len(args[0].data)):
            return sim(*args,**kwargs)
    patch(circuit,'simulate_statevector',simulate)
    tail=P.controlled_tail;counts=[0]
    def controlled(*args,**kwargs):
        counts[0]+=1
        result=tail(*args,**kwargs)
        if counts[0]%16==0:log('compose_steps','PROGRESS',completed=counts[0],total=512)
        return result
    patch(P,'controlled_tail',controlled)
    try:yield stats
    finally:
        for (obj,key),value in reversed(list(saved.items())):setattr(obj,key,value)


def check_result_path(path,binding,receipts=None,source=None):
    i=path['index']
    if i not in (0,1):raise ValueError('Wrong path index')
    B1=legacy();B1.receipts().check_path(path,request=binding['request'],dictionary=binding['dictionary_identity'],
        cohort=COHORT,index=i,seeds=binding['step_seeds'][i])
    if path['result']['sampling']!='ORIGINAL_NORMAL_SAMPLER':raise ValueError('Independent normal sampler required')
    if receipts is not None:
        for a,key in AXES.items():
            receipt=receipts['p%d%s'%(i,a)];r=path['result']
            same((receipt['axis'],receipt['index'],receipt['cohort'],receipt['request'],receipt['seeds'],receipt['dictionary_identity']),
                 (a,i,COHORT,binding['request'],binding['step_seeds'][i],binding['dictionary_identity']),'Individual receipt binding differs')
            same(receipt['execution_source'],source,'Receipt execution SOURCE differs')
            same(receipt['record'],r['compile_receipts'][key],'Original compiler record differs')
            same(receipt['record']['fingerprint'],r['fingerprints'][key]['fingerprint'],'Fingerprint differs')
            same(receipt['record']['metrics'],r['prices'][key],'Receipt metrics differ')
            same(receipt['record']['compiler'],binding['domain'][a]['profile']['compiler'],'Compiler differs')


def worker(job_path):
    job=json.loads(Path(job_path).read_text());i=job['index'];folder=Path(job['private'])
    log=Progress(folder/'PROGRESS.jsonl')
    try:
        frozen=read(job['prediction_reference'])
        check_binding(job['binding'],frozen['binding'])
        same(job['driver_commit'],frozen['execution_source']['commit'],'Frozen driver SOURCE differs')
        B1=legacy()
        # Provider-first process, exactly as the existing child.
        import h46_long_time_signal as signal
        with log.stage('install_provider'):
            signal.install();B1.B0.provider_checks();B1.B0.register_pauli_modules()
            import qiskit
            same(qiskit.__version__,'1.3.0','Qiskit differs')
            P=B1.B0.connection();dictionary=P.load_dictionary(PIN_ROOT,job['binding']['dictionary_identity'])
            import h46_long_time_native as native
            native.PUBLIC,native.PRIVATE=folder/'native_public',folder
            ctx=native.native_context('H4',(2,))
            same(ctx.ham.constant,B1.B0.HAMILTONIAN_CONSTANT,'Hamiltonian constant differs')
            operators,blocks,_=B1.B0.p0_arrays(2)
        same(job['execution_source']['commit'],job['driver_commit'],'Worker SOURCE differs')
        same(blob(job['driver_commit'],SCRIPT),(ROOT/SCRIPT).read_bytes(),'Running driver bytes differ')
        captured={};compile_calls=[]
        def compiler(wrapper,fp,representative):
            axis='X' if not compile_calls else 'Y'
            if len(compile_calls)>=2:raise ValueError('Two compile maximum per path')
            compile_calls.append(axis)
            with log.stage('compile_'+axis,instructions=len(wrapper.data),fingerprint=fp):
                result=native.compile_cached(wrapper,fp,representative)
            return result
        with log.stage('acquire_path',index=i),accelerated_observed(P,log) as stats:
            r=B1.acquire_path(native,ctx,dictionary,job['binding']['request'],cohort=COHORT,index=i,
                seeds=job['binding']['step_seeds'][i],operators=operators,blocks=blocks,compile_cached=compiler)
        path=dict(index=i,cohort=COHORT,request=job['binding']['request'],dictionary_identity=job['binding']['dictionary_identity'],result=r,reused=False)
        check_result_path(path,job['binding'])
        with log.stage('receipt_export'):
            receipts=B1.receipts().compile_receipts([path],job['execution_source'],job['binding']['dictionary_identity'],folder)
            check_result_path(path,job['binding'],receipts,job['execution_source'])
            write(folder/'COMPILE_RECEIPTS.json',receipts)
        # Compact price/gate summary; event data remains lossless in the raw path.
        summary={k:v for k,v in r.items() if k not in ('events','event_metadata')}
        summary.update(acceleration=stats,compile_calls=compile_calls)
        with log.stage('raw_save'):
            raw=folder/'PATH.json.gz'
            with gzip.open(raw,'wt',compresslevel=1) as f:json.dump(path,f,separators=(',',':'),allow_nan=False)
            write(folder/'SUMMARY.json',dict(status='PASS',index=i,result=summary,resources=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        log('worker','DONE',index=i);return 0
    except BaseException as exc:
        log('worker','STOP',error=repr(exc))
        write(folder/'FAILURE.json',dict(status='STOP',index=i,error=traceback.format_exc(),
              peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,automatic_retry=False))
        return 2


def observe(processes,log,poll_seconds=5):
    states={}
    while len(states)<len(processes):
        for i,p,stream in processes:
            if i not in states and p.poll() is not None:
                states[i]=dict(index=i,exit_code=p.returncode,signal=-p.returncode if p.returncode<0 else None)
                log('worker_exit','DONE' if p.returncode==0 else 'STOP',**states[i]);stream.close()
        if len(states)<len(processes):time.sleep(poll_seconds)
    return states


def execute(prediction_commit):
    reference=dict(commit=prediction_commit,path=BASE+'/PREDICTIONS.json')
    pre=read(reference);m=metadata();check_binding(pre['binding'],m)
    source=pre['execution_source'];same((ROOT/SCRIPT).read_bytes(),blob(source['commit'],SCRIPT),'Frozen SOURCE changed')
    # Commit is on remote before any sampling; no approval-proof artifact required.
    remote=subprocess.check_output(['git','ls-remote','--heads','origin','codex/pauli-ld2-price-freeze-20261115'],cwd=ROOT,text=True).split()[0]
    subprocess.check_call(['git','merge-base','--is-ancestor',prediction_commit,remote],cwd=ROOT)
    same(blob(remote,reference['path']),blob(prediction_commit,reference['path']),'Published prediction blob differs')
    out=ROOT/BASE/'run01'
    if out.exists() or PRIVATE.exists():raise FileExistsError('Fresh scientific output required; no automatic retry')
    out.mkdir();PRIVATE.mkdir()
    log=Progress(out/'PROGRESS.jsonl')
    execution_source=dict(commit=source['commit'],caller=SCRIPT,prefix=2,kind='M9_LONG_NORMAL',cell='M9_ROUND',
        cohort=COHORT,step_seeds=m['step_seeds'],result_path=BASE+'/run01/RESULT.json',legacy_source=PIN)
    write(out/'EXECUTION.json',dict(predictions=reference,execution_source=execution_source,binding=m,python=sys.version,
        started_UTC=datetime.now(timezone.utc).isoformat(),compile_maximum=4,automatic_retry=False))
    env=os.environ.copy()
    # Tiny dense 256x256 repeated matvecs: one BLAS thread per independent process avoids thread-launch overhead.
    env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
    processes=[]
    for i in (0,1):
        folder=PRIVATE/('path'+str(i));folder.mkdir()
        job=dict(index=i,private=str(folder),binding=m,execution_source=execution_source,driver_commit=source['commit'],prediction_reference=reference)
        write(folder/'JOB.json',job)
        stream=(folder/'CONSOLE.txt').open('x')
        command=[sys.executable,'-B',str(ROOT/SCRIPT),'--worker',str(folder/'JOB.json')]
        p=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=env)
        processes.append((i,p,stream));log('worker','START',index=i,pid_child=p.pid)
    states=observe(processes,log)
    write(out/'PROCESS_EXITS.json',list(states.values()))
    ok=all(s['exit_code']==0 for s in states.values())
    summaries=[];receipts={};manifests=[]
    for i in (0,1):
        folder=PRIVATE/('path'+str(i))
        for name in ('PROGRESS.jsonl','SUMMARY.json','FAILURE.json','CONSOLE.txt'):
            if (folder/name).exists():
                (out/('path%d_'%i+name)).write_bytes((folder/name).read_bytes())
        if (folder/'SUMMARY.json').exists():summaries.append(json.loads((folder/'SUMMARY.json').read_text()))
        if (folder/'COMPILE_RECEIPTS.json').exists():receipts.update(json.loads((folder/'COMPILE_RECEIPTS.json').read_text()))
        if (folder/'PATH.json.gz').exists():
            raw=folder/'PATH.json.gz';data=raw.read_bytes();name='PATH%d.json.gz'%i
            (out/name).write_bytes(data)
            manifests.append(dict(index=i,path=BASE+'/run01/'+name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),
                encoding='GZIP_JSON',selector=[]))
    write(out/'RAW_PATHS.json',manifests);write(out/'COMPILE_RECEIPTS.json',receipts)
    write(out/'RESULT.json',dict(status='FINITE_M9_PATHS_COMPLETE' if ok else 'STOP_M9',source=execution_source,
        summaries=summaries,predictions=reference,raw_paths=manifests,automatic_retry=False,mandatory_stop=True))
    log('acquisition','DONE' if ok else 'STOP',valid_paths=len(summaries))
    return 0 if ok else 2


def compare(pre,units):
    rows=[]
    for prediction in pre['predictions']:
        axis=prediction['axis'];unit=units[axis];valid=len(unit)==2 and all(x['valid'] for x in unit)
        P=exact(prediction['predicted_RZ']);O=sum((Fraction(x['compiled_RZ']) for x in unit),Fraction())/2 if valid else None
        d=P-O if valid else None
        rows.append(dict(m=9,axis=axis,P=enc(P),O=enc(O) if valid else None,
            signed_error=enc(d) if valid else None,absolute_error=enc(abs(d)) if valid else None,
            signed_relative_error=enc(d/O) if valid and O else None,absolute_relative_error=enc(abs(d)/O) if valid and O else None,
            reason='INCOMPLETE_OR_INVALID' if not valid else 'ZERO_OBSERVED_PRICE' if O==0 else None))
    valid=all(r['O'] is not None for r in rows)
    return dict(schema='C2_LD2_M9_DESCRIPTIVE_COMPARISON_V1',rows=rows,
        campaign='INCOMPLETE_OR_INVALID' if not valid else 'DESCRIPTIVE_COMPLETE_RELATIVE_UNDEFINED' if any(r['reason'] for r in rows) else 'DESCRIPTIVE_COMPLETE',
        model_verdict='NOT_ASSESSED_NO_ADOPTED_TOLERANCE',tolerance=None,fit=False,E_C=None,CI=None,N=None,G=None,mandatory_stop=True)


def publish(raw_commit):
    """Resolve saved raw paths/receipts, run canonical guard, then exact comparison."""
    B1=legacy();I,DC,G=B1.B0._meta();P=B1.B0.connection();m=metadata()
    data=read(dict(commit=raw_commit,path=BASE+'/run01/RESULT.json'));source=data['source']
    same(blob(source['commit'],SCRIPT),(ROOT/SCRIPT).read_bytes(),'Publication driver differs from execution SOURCE')
    same(source['step_seeds'],m['step_seeds'],'Result seed SOURCE differs')
    pre=read(data['predictions']);check_binding(pre['binding'],m)
    receipts=read(dict(commit=raw_commit,path=BASE+'/run01/COMPILE_RECEIPTS.json'))
    paths=[]
    for manifest in data['raw_paths']:
        raw=blob(raw_commit,manifest['path']);same(hashlib.sha256(raw).hexdigest(),manifest['sha256'],'Raw path hash differs')
        path=json.loads(gzip.decompress(raw));check_result_path(path,m,receipts,source);paths.append(path)
    paths.sort(key=lambda p:p['index']);same([p['index'] for p in paths],[0,1],'Incomplete m9 paths')
    req,packet,dictionary=B1.canonical(2);config=req['configuration'];B=I.providers(str(PIN_ROOT)).B
    metrics={a:{k:float(sum(Fraction(p['result']['prices'][key][k]) for p in paths)/2) for k in paths[0]['result']['prices'][key]}
             for a,key in (('cosine','cosine'),('sine','sine'))}
    proofs_path=BASE+'/run01/IMPLEMENTATION_EVIDENCE.json'
    prices,sides,proofs=P.price_records([r for r in DC.pauli_requests(PIN_ROOT,req) if r['m']==9],config,
        trial_prices=metrics,scalar=paths[0]['result']['scalar'],native_source=dict(commit=PIN,path=P.MODULE,driver_source=source),
        cohort=COHORT,price_source=source,evidence_commit=None,evidence_path=proofs_path,measurement=B.p.measurement)
    for price in prices:price['n_paths']=2;price['diagnostic'].update(pilot_role='M9_LONG_NORMAL',law_expectation=False)
    for name,value in (('PRICES.json',prices),('SIDECARS_UNBOUND.json',sides),('IMPLEMENTATION_EVIDENCE.json',proofs)):
        write(ROOT/BASE/'run01'/name,value)
    write(ROOT/BASE/'RAW_VALIDATION.json',dict(status='FOUR_RECEIPTS_CHECKED',raw_source=raw_commit,paths=2,
        receipts=4,predictions=data['predictions'],gate_pass=True,model_qualified=False))
    print('Raw receipt checks complete; commit PRICES/SIDECARS/EVIDENCE before --admit')


def admit(publication_commit):
    B1=legacy();I,DC,G=B1.B0._meta();P=B1.B0.connection();m=metadata();req,packet,dictionary=B1.canonical(2)
    ref=lambda name:dict(commit=publication_commit,path=BASE+'/run01/'+name)
    prices=read(ref('PRICES.json'));sides=read(ref('SIDECARS_UNBOUND.json'))
    for s in sides:s['implementation_evidence']['commit']=publication_commit
    config=req['configuration'];cohorts={(CID,a):COHORT for a in AXES}
    policy=dict(id='B1_TECHNICAL_COMPATIBILITY_NOT_ADOPTED',source=dict(commit=B1.A_SOURCE,path=B1.A_DESIGNS),
        profile=config['spec']['profile'],by_method={P.ROUTE:dict(native_policy=P.NATIVE_POLICY,rte_rotation_lowering=P.LOWERING_ID,scalar_quantity=P.SCALAR_QUANTITY)})
    guard=G.check(PIN_ROOT,config,prices,cohorts=cohorts,bindings=sides,policy=policy)
    write(ROOT/BASE/'run01/C_GUARD_CHECK.json',guard);write(ROOT/BASE/'run01/SIDECARS_PUBLISHED.json',sides)
    data=read(ref('RESULT.json'));receipts=read(ref('COMPILE_RECEIPTS.json'));units={a:[] for a in AXES}
    for a in AXES:
        for i in (0,1):
            r=receipts['p%d%s'%(i,a)]
            units[a].append(dict(index=i,axis=a,valid=True,compiled_RZ=r['record']['metrics']['rz_count'],
                fingerprint=r['record']['fingerprint'],receipt=dict(ref('COMPILE_RECEIPTS.json'),selector=['p%d%s'%(i,a)]),
                path=dict(ref('PATH%d.json.gz'%i),encoding='GZIP_JSON'),execution_source=r['execution_source']))
    comparison=compare(read(data['predictions']),units)
    comparison['prediction_source']=data['predictions'];comparison['receipt_units']=units
    write(ROOT/BASE/'COMPARISON.json',comparison)
    print(json.dumps(dict(campaign=comparison['campaign'],rows=comparison['rows'])))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--metadata',action='store_true');ap.add_argument('--freeze',action='store_true')
    ap.add_argument('--execute');ap.add_argument('--worker');ap.add_argument('--publish');ap.add_argument('--admit')
    a=ap.parse_args()
    if a.worker:return worker(a.worker)
    if a.freeze:freeze();return 0
    if a.execute:return execute(a.execute)
    if a.publish:publish(a.publish);return 0
    if a.admit:admit(a.admit);return 0
    print(json.dumps(metadata(),indent=2));return 0


if __name__=='__main__':sys.exit(main())
