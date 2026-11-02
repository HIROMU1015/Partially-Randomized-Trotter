"""One sealed H4 job, three fixed methods, forked batches, no retry or measurement."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction as F
import gzip
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback
from support import (HERE, ROOT, BASE, MODEL, require, digest, sha, canonical, git, save,
                     gzip_bytes, verify_rows, resource_snapshot, Timers)
from diagnostic import Sampler, load_model, generate_path, METHODS
from trottertracks.algorithm_codesign.g6_f3_contract import Evidence, fingerprint

G = None

def read_json(path):
    return json.loads(Path(path).read_text())

def verify_frozen(all_files=True):
    auth=read_json(HERE/'authorization_receipt.json')
    for name in ('source_manifest.json','input_manifest.json','environment_manifest.json'):
        require(digest(HERE/name)==auth['manifest_SHA256'][name],'MANIFEST_CHANGED:'+name)
        data=read_json(HERE/name)
        if all_files or name!='environment_manifest.json':
            verify_rows(data['files'] if all_files else data.get('online_guard_files',data['files']))

def batch(index):
    start=time.perf_counter(),time.process_time()
    sampler,method_dir,contract=G
    timers=Timers()
    records=[]
    last_trial=None
    try:
        verify_frozen(False)
        hits,misses=sampler.angle_hits,sampler.angle_misses
        ehits,emisses=sampler.event_cache_hits,sampler.event_cache_misses
        before_calls=sampler.calls.copy()
        begin=index*contract['batch_trials']
        for trial in range(begin,min(begin+contract['batch_trials'],contract['trials_per_method'])):
            last_trial=trial
            row=generate_path(sampler,trial,timers)
            records.append(row)
            if (trial-begin)%8==7:
                save(method_dir/'progress'/('worker_'+str(os.getpid())+'.json'),
                     dict(pid=os.getpid(),batch=index,last_completed_trial=trial))
        with timers.stage('batch_serialization_and_compressed_storage'):
            raw=b''.join(canonical(row)+b'\n' for row in records)
            compressed=gzip_bytes(raw)
            target=method_dir/'batches'/('batch_'+str(index).zfill(3)+'.jsonl.gz')
            target.parent.mkdir(exist_ok=True)
            with target.open('xb') as stream:
                stream.write(compressed)
        verify_frozen(False)
        receipt=dict(batch=index,pid=os.getpid(),first_trial=begin,completed=len(records),
            last_trial=last_trial,wall_seconds=time.perf_counter()-start[0],CPU_seconds=time.process_time()-start[1],
            stages=timers.rows,angle_cache_hits=sampler.angle_hits-hits,angle_cache_misses=sampler.angle_misses-misses,
            event_cache_hits=sampler.event_cache_hits-ehits,event_cache_misses=sampler.event_cache_misses-emisses,
            SOURCE_derived_online_query_counts=dict(sampler.calls-before_calls),
            max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            path=str(target.relative_to(HERE)),bytes=len(compressed),uncompressed_bytes=len(raw),sha256=sha(compressed))
        save(method_dir/'receipts'/('batch_'+str(index).zfill(3)+'.json'),receipt)
        return receipt
    except Exception as exc:
        if hasattr(sampler,'abort'):
            sampler.abort.set()
        # Preserve every completed path in this failed batch; never certify its prefix.
        method_dir.joinpath('failures').mkdir(exist_ok=True)
        prefix=method_dir/'failures'/('batch_'+str(index)+'.prefix.jsonl.gz')
        prefix.write_bytes(gzip_bytes(b''.join(canonical(row)+b'\n' for row in records)))
        save(method_dir/'failures'/('batch_'+str(index)+'.json'),dict(batch=index,pid=os.getpid(),
            last_started_trial=last_trial,completed_prefix=len(records),exception=repr(exc),traceback=traceback.format_exc(),
            F3_last_failure=(sampler.session.last_failure.to_dict() if sampler.session and sampler.session.last_failure else None),
            technical_failure_not_zero=True,confidence_certified=False,mandatory_STOP=True))
        raise

def summarize(method_dir,receipts,contract):
    """Inside the single authorized job: summarize its own fixed trajectories only."""
    seen=set(); weights=[]; logical=Counter();zero_reasons=Counter();raw=Counter();reduced=Counter()
    by_step=[Counter() for _ in range(4)]; max_exact=F(0);accepted=zeros=0
    for receipt in sorted(receipts,key=lambda r:r['batch']):
        path=HERE/receipt['path'];require(digest(path)==receipt['sha256'],'BATCH_BYTES')
        for line in gzip.decompress(path.read_bytes()).splitlines():
            row=json.loads(line)
            rec=Evidence.create(__import__('trottertracks.algorithm_codesign.g6_f3_contract',fromlist=['_unpack'])._unpack(row['exact']))
            require(rec.digest==row['exact_digest'],'PATH_EVIDENCE_DIGEST')
            d=rec.read();trial=d['trial']
            require(trial not in seen and 0<=trial<1024 and len(d['microsteps'])==4,'TRIAL_PLAN')
            seen.add(trial);weights.append(row['observed_weight_square_binary64'])
            max_exact=max(max_exact,d['weight'])
            zero=d['kind']=='ZERO'
            require(zero==(row['logical'] is None) and d['measurement_performed'] is False,'ZERO_MEASUREMENT')
            zeros+=int(zero);accepted+=int(not zero)
            zero_reasons.update(row['zero_reasons'])
            raw.update(str(v) for v in row['raw_degrees'])
            reduced.update('ZERO' if v is None else str(v) for v in row['reduced_lengths'])
            for k,step in enumerate(d['microsteps']):
                by_step[k]['ZERO_'+step['header']['reason'] if step['event_id'] is None else 'EVENT']+=1
                require(all(b['width']==2048 for b in step['bits_draws']),'BIT_WIDTH')
            if not zero:
                for key in ('signed_Q','child_rotations','controlled_gate_instructions','logical_1Q','logical_2Q',
                            'logical_multi_qubit','instructions','depth'):
                    logical[key]+=row['logical'][key]
    require(seen==set(range(1024)),'INCOMPLETE_ALL_TRIALS')
    stages={}
    for receipt in receipts:
        for name,values in receipt['stages'].items():
            target=stages.setdefault(name,Counter());target.update(values)
    summary=dict(method=method_dir.name,scope='TAIL_ONLY_PLUS_SCALAR; UNCERTIFIED_EXPLORATORY',
        planned=1024,completed=len(seen),microsteps=4096,accepted_paths=accepted,valid_zero_paths=zeros,
        accepted_fraction_observed=accepted/1024,zero_reasons_microsteps=dict(zero_reasons),
        by_microstep=[dict(c) for c in by_step],raw_degree_counts=dict(raw),reduced_length_counts=dict(reduced),
        empirical_weight_second_moment=math.fsum(weights)/1024,maximum_observed_weight=float(max_exact),
        maximum_observed_weight_exact=Evidence.create(max_exact).to_dict(),
        moments_are_empirical_not_certified=True,logical_totals=dict(logical),
        logical_per_classical_trial={k:v/1024 for k,v in logical.items()},
        logical_per_accepted_path={k:v/accepted for k,v in logical.items()} if accepted else None,
        quantum_call_equivalent_circuits=accepted,zero_avoided_circuits=zeros,
        workspace_beyond_system=1,native_cost='UNKNOWN',deterministic_halves='UNKNOWN',
        stages={k:dict(v) for k,v in stages.items()},
        worker_batch_CPU_seconds=math.fsum(r['CPU_seconds'] for r in receipts),
        worker_batch_wall_seconds_sum=math.fsum(r['wall_seconds'] for r in receipts),
        max_worker_RSS_KiB=max(r['max_RSS_KiB'] for r in receipts),
        angle_cache_hits=sum(r['angle_cache_hits'] for r in receipts),
        angle_cache_misses=sum(r['angle_cache_misses'] for r in receipts),
        event_cache_hits=sum(r['event_cache_hits'] for r in receipts),
        event_cache_misses=sum(r['event_cache_misses'] for r in receipts),
        query_counter_scope='SOURCE-derived logical call counts, not every internal arithmetic operation',
        SOURCE_derived_online_query_counts=dict(sum((Counter(r['SOURCE_derived_online_query_counts']) for r in receipts),Counter())),
        raw_batch_bytes=sum(r['bytes'] for r in receipts),
        total_method_saved_bytes_before_summary=sum(p.stat().st_size for p in method_dir.rglob('*') if p.is_file()),
        technical_failure=None,complete=True,quantum_shots=0,coherent_signal_estimator=None,
        sufficient_N_computed=False,final_resource_winner_not_assessed=True)
    save(method_dir/'summary.json',summary)
    return summary

def coordinator(method):
    global G
    contract=read_json(HERE/'execution_contract.json');auth=read_json(HERE/'authorization_receipt.json')
    require(method in contract['methods'] and (HERE/'attempt_marker.json').exists(),'JOB_AUTHORITY')
    require(digest(HERE/'execution_contract.json')==auth['contract_sha256'],'CONTRACT_CHANGED')
    method_dir=HERE/'result'/method;method_dir.mkdir()
    t=Timers();start=time.perf_counter();initial=resource_snapshot()
    try:
        verify_frozen()
        with t.stage('R1_input_and_registry_setup'):
            model=load_model()
        with t.stage('sampler_normalizer_selector_setup'):
            sampler=Sampler(method,model,auth['contract_sha256'],method_dir)
        setup=dict(method=method,PID=os.getpid(),stages=t.rows,workers=contract['workers_per_method'],
            groups=len(getattr(sampler.g,'groups',[])),
            normalizer=Evidence.create(dict(B=(sampler.g.B.lo,sampler.g.B.hi),
                U=(sampler.g.U.lo,sampler.g.U.hi) if hasattr(sampler.g,'U') else None)).to_dict(),
            model_id=MODEL,labels_id=model['labels_id'],p_id=fingerprint(model['p']),
            x=Evidence.create(model['x']).to_dict(),precision=Evidence.create(sampler.precision).to_dict(),
            F3_preflight=sampler.session.preflight.to_dict() if sampler.session else 'UNCERTIFIED_EXPLORATORY',
            registry_local_matrix_validation='unchanged saved R1 constructors/hash checks; no new diagonalization',
            setup_max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        save(method_dir/'setup.json',setup)
        G=sampler,method_dir,contract
        total=(1024+contract['batch_trials']-1)//contract['batch_trials']
        online_start=time.perf_counter()
        # fork shares read-only setup; each worker owns its cache and fixed unique trial batches.
        sampler.abort=mp.get_context('fork').Event()
        pool=ProcessPoolExecutor(max_workers=contract['workers_per_method'],mp_context=mp.get_context('fork'))
        futures=[]
        try:
            futures=[pool.submit(batch,i) for i in range(total)]
            receipts=[]
            for future in as_completed(futures):
                receipts.append(future.result())
                save(method_dir/'progress.json',dict(completed_batches=len(receipts),planned_batches=total,
                    completed_trials=sum(r['completed'] for r in receipts),wall_seconds=time.perf_counter()-online_start))
        except BaseException:
            sampler.abort.set()
            for f in futures:
                f.cancel()
            for process in pool._processes.values():
                if process.is_alive():
                    process.terminate()
            pool.shutdown(wait=True,cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)
        online_wall=time.perf_counter()-online_start
        agg=Timers()
        with agg.stage('final_validation_summary'):
            summary=summarize(method_dir,receipts,contract)
        verify_frozen()
        own=resource.getrusage(resource.RUSAGE_SELF);children=resource.getrusage(resource.RUSAGE_CHILDREN)
        save(method_dir/'terminal.json',dict(status='C1_5A_METHOD_COMPLETE',complete=True,
            online_wall_seconds=online_wall,total_wall_seconds=time.perf_counter()-start,
            setup_stages=t.rows,final_stages=agg.rows,
            coordinator_user_CPU=own.ru_utime,coordinator_system_CPU=own.ru_stime,
            worker_user_CPU=children.ru_utime,worker_system_CPU=children.ru_stime,
            all_method_hierarchy_CPU=own.ru_utime+own.ru_stime+children.ru_utime+children.ru_stime,
            max_coordinator_RSS_KiB=own.ru_maxrss,max_reaped_child_RSS_KiB=children.ru_maxrss,
            initial_resources=initial,final_resources=resource_snapshot(),retry=0))
    except BaseException as exc:
        save(method_dir/'terminal.json',dict(status='C1_5A_TECHNICAL_FAILURE_STOP',complete=False,
            exception=repr(exc),traceback=traceback.format_exc(),mandatory_STOP=True,retry=0))
        raise

def main():
    require(sys.argv==[sys.argv[0]],'NO_SCIENTIFIC_OVERRIDES')
    auth=read_json(HERE/'authorization_receipt.json');contract=read_json(HERE/'execution_contract.json')
    require(git('rev-parse','HEAD^')==auth['source_commit'],'AUTHORIZATION_SOURCE_PARENT')
    require(git('diff','HEAD^','HEAD','--name-only')==str((HERE/'authorization_receipt.json').relative_to(ROOT)),
            'AUTHORIZATION_ONLY_CHILD')
    require(git('status','--porcelain')=='','CLEAN_BEFORE_ONE_SHOT')
    require(digest(HERE/'execution_contract.json')==auth['contract_sha256'],'CONTRACT_HASH')
    require(contract['methods']==list(METHODS) and contract['trials_per_method']==1024 and
            contract['microsteps']==4 and contract['scientific_attempts']==1 and contract['retry']==0,'FIXED_JOB')
    verify_frozen()
    marker=HERE/'attempt_marker.json'
    with marker.open('x') as stream:
        json.dump(dict(schema='C1_5A_CONSUMED_ONE_SHOT',authorization_HEAD=git('rev-parse','HEAD'),
            source=auth['source_commit'],contract_sha256=auth['contract_sha256'],pid=os.getpid(),
            scientific_attempt=1,retry=0,started_monotonic=time.monotonic()),stream,indent=2)
    result=HERE/'result';result.mkdir()
    children=[];start=time.perf_counter()
    try:
        for method in METHODS:
            out=(result/(method+'.stdout.log')).open('xb');err=(result/(method+'.stderr.log')).open('xb')
            child=subprocess.Popen([sys.executable,'-B',__file__,'--coordinator',method],
                cwd=ROOT,env=os.environ,start_new_session=True,stdout=out,stderr=err)
            children.append((method,child,out,err))
        with (result/'resource_observations.jsonl').open('x') as observations:
            while any(c.poll() is None for _,c,_,_ in children):
                observations.write(json.dumps(resource_snapshot())+'\n');observations.flush()
                for method,c,_,_ in children:
                    if c.poll() not in (None,0):
                        raise RuntimeError('METHOD_FAILED:'+method)
                time.sleep(5)
        for method,c,_,_ in children:
            require(c.wait()==0,'METHOD_EXIT:'+method)
        verify_frozen()
        rows=[read_json(result/m/'summary.json') for m in METHODS]
        require(all(r['completed']==1024 and r['complete'] for r in rows),'ALL_PLANNED_COMPLETE')
        require(len(set(read_json(result/m/'setup.json')['p_id'] for m in METHODS))==1,'COMMON_P')
        save(result/'terminal.json',dict(status='C1_5A_EXPLORATORY_COMPLETE',scientific_attempts=1,retry=0,
            completed_paths=3072,completed_microsteps=12288,wall_seconds=time.perf_counter()-start,
            coherent_signal_estimator=None,native_calls=0,shots=0,sufficient_N_computed=False,
            strict_resource_winner=False,mandatory_STOP=True))
    except BaseException as exc:
        for _,child,_,_ in children:
            if child.poll() is None:
                # Only process groups created by this job; never another user's processes.
                os.killpg(child.pid,signal.SIGTERM)
        for _,child,_,_ in children:
            child.wait()
        save(result/'terminal.json',dict(status='C1_5A_TECHNICAL_FAILURE_STOP',scientific_attempts=1,retry=0,
            exception=repr(exc),traceback=traceback.format_exc(),partial_results_only=True,mandatory_STOP=True))
        raise
    finally:
        for _,_,out,err in children:
            out.close();err.close()

if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--coordinator':
        coordinator(sys.argv[2])
    else:
        main()
