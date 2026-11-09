"""Reuse H8 full-wrapper checks and operator-derived compiler phase accounting."""
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import gc
import gzip
import io
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import resource
import subprocess
import time

import h8_full_h_pure_rte as s

CTX=None


def rss(pid):
    try:
        return 1024*int(next(x.split()[1] for x in Path('/proc',str(pid),'status').read_text().splitlines() if x.startswith('VmRSS:')))
    except (FileNotFoundError,StopIteration):return 0


def price_job(job):
    import h46_long_time_native as native
    import h8_long_time_runtime as runtime
    import h8_compiler_phase_bookkeeping as phase
    from qiskit import qpy
    runtime.CONTEXT=CTX
    original=native.compile_cached
    original_compose=native.compose
    from trotterlib.df_trotter import circuit
    original_simulate=circuit.simulate_statevector
    old_key=phase.gate_key;keys={};small_gates={};accounting={}

    def cached_key(op):
        # Exactly the fields in the unchanged H8 helper's gate_key.
        sig=(op.name,op.base_class.__module__,op.base_class.__name__,op.num_qubits,
            getattr(op,'ctrl_state',None),tuple(float(p).hex() for p in op.params))
        if sig not in keys:keys[sig]=old_key(op)
        return keys[sig]

    phase.gate_key=cached_key
    directory=Path(job[-1]);started=time.monotonic()
    representative=job[3]==0;action_index=0

    def stage(name,**extra):
        path=s.PUBLIC/'price_stage.json' if representative else directory/'action_progress.json'
        s.save(path,dict(stage=name,candidate=job[0]['id'],index=job[3],pid=os.getpid(),
            elapsed=time.monotonic()-started,**extra))

    def compose(*args,**kwargs):
        stage('NATIVE_BUILD')
        value=original_compose(*args,**kwargs)
        stage('NATIVE_BUILT',native_instructions=len(value.circuit.data))
        return value

    def independent(ctx,row,requests):
        stage('INDEPENDENT_ALGEBRAIC_EVENT_VECTOR')
        return ctx.oracle.trajectory(ctx,row,requests)

    def simulate(qc,initial):
        nonlocal action_index
        labels=['LOW_CONTROL','HIGH_CONTROL','NATIVE_COSINE','COMPILED_COSINE','NATIVE_SINE','COMPILED_SINE']
        stage(labels[action_index]);action_index+=1
        return original_simulate(qc,initial)

    native.compose=compose;runtime.independent_trajectory=independent
    circuit.simulate_statevector=simulate

    def compile_with_phase(wrapper,fingerprint,representative):
        stage('COMPILE_FULL_WRAPPER',fingerprint=fingerprint)
        record,qc,hit=original(wrapper,fingerprint,representative)
        if representative:
            stage('OPERATOR_DERIVED_COMPILER_PHASE',fingerprint=fingerprint)
            s.require(qc is not None,'REPRESENTATIVE_COMPILED_VECTOR_REQUIRED')
            from pf_common_run_contract import resources
            obs=resources()
            proof=phase.certify(wrapper,small_gates,workers=max(1,int(obs['idle_cpu'])))
            qc,receipt=phase.correct_phase(wrapper,qc,proof)
            s.require(qc.count_ops().get('rz',0)==record['metrics']['rz_count'],'PHASE_RZ_COUNT_UNCHANGED')
            folder=directory/'phase';folder.mkdir(exist_ok=True)
            s.old.save(folder/(fingerprint+'_operators_PRIVATE.json'),proof)
            stream=io.BytesIO();qpy.dump(qc,stream)
            with (folder/(fingerprint+'_corrected.qpy.gz')).open('xb') as f:
                f.write(gzip.compress(stream.getvalue(),compresslevel=1,mtime=0))
            accounting[fingerprint]=receipt
        return record,qc,hit

    native.compile_cached=compile_with_phase
    try:
        value=runtime.price_job(job)
        value.update(compiler_phase_source='18ce41942051065fa097644a8a937bd11adba708',
            compiler_phase_accounting=accounting,wall_seconds=time.monotonic()-started,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            simulation='Aer CPU double exact gate fusion; input circuit unchanged',
            independent_oracle='saved-basis explicit event algebra with vectorized <=4-qubit local actions',
            python_pid=os.getpid())
        # Original result.json remains untouched; extension metadata is separate.
        s.old.save(directory/'supplement.json',value)
        stage('PASS')
        return value
    except BaseException as exc:
        if directory.exists():s.old.save(directory/'STOP.json',dict(reason=str(exc),automatic_retry=False))
        raise
    finally:
        native.compile_cached=original;native.compose=original_compose
        circuit.simulate_statevector=original_simulate;phase.gate_key=old_key


def main():
    global CTX
    s.require(s.read(s.PUBLIC/'validation.json')['status']=='PASS','PHASE_A_PASS')
    s.require(s.read(s.PUBLIC/'signal_vector_validation.json')['status']=='PASS','SPARSE_FINITE_MEAN_PASS')
    selection=s.read(s.PUBLIC/'selection.json')
    s.require(selection['status']=='FIXED_BEFORE_PRICES' and len(selection['candidates'])<=2
        and selection['n']==9 and selection['cohort']==s.COHORT,'FIXED_SMALL_PRICE_SELECTION')
    native=s.install();CTX=s.context('price')
    native.PRIVATE=s.PRIVATE
    from pf_common_run_contract import resources
    observations=[];accepted=[];errors=[]
    for row,endpoint in zip(selection['candidates'],selection['endpoints']):
        base=s.PRIVATE/'cost'/s.COHORT/row['id']
        s.require(not base.exists(),'NO_AUTOMATIC_PRICE_RETRY')
        obs=resources();observations.append(dict(phase='representative_before',observation=obs))
        prior=json.loads(gzip.decompress(subprocess.check_output(['git','show',
            '7a8bc7f132e166390c943e6b32c92cc6abe1baf4:artifacts/resource_applicability/df_full_h_pure_rte_t20_opt/2026-11-09/price_trials.json.gz'],cwd=s.ROOT)))
        anchor=next(t for t in prior if t['candidate']['system']=='H6' and t['candidate']['s']==16384 and t['index']==0)
        forecast=1.25*anchor['peak_RSS_bytes']*(row['s']/16384)*(16*15)/(12*11)
        observations[-1].update(first_peak_forecast=forecast,
            forecast_source='saved H6 s16384 representative peak; scale by event count and Gaussian orbital-pair count, margin25%; empirical admission only')
        s.save(s.PRIVATE/'price_resources.json',observations)
        if obs['available_ram']<forecast:
            s.save(s.PUBLIC/'price_unacquired.json',dict(candidate=row,indices=list(range(9)),
                reason='Current RAM below empirical representative-peak forecast',observation=obs,
                forecast=forecast,forecast_is_rigorous_bound=False))
            s.save(s.PUBLIC/'prices.json',dict(status='RESOURCE_DEFERRED',summaries=[],accepted=0,
                planned=9,errors=[],all_workers_joined=True,cohort=s.COHORT,pooled=False))
            return
        # Phase C's first real trajectory is validated before releasing the rest.
        first=price_job((row,endpoint,s.COHORT,0,str(base/'0')))
        accepted.append(first)
        s.save(s.PUBLIC/'price_progress.json',dict(complete=len(accepted),planned=9,errors=[]))
        gc.collect()
        queue=list(range(1,9));active={};fresh=[]
        # Eight is the remaining scientific sample count, not a resource cap.
        with ProcessPoolExecutor(max_workers=len(queue),mp_context=mp.get_context('fork')) as pool:
            while queue or active:
                estimate=1.25*(max(v['peak_RSS_bytes'] for v in fresh) if fresh else first['peak_RSS_bytes'])
                obs=resources()
                live=sum(rss(p.pid) for p in pool._processes.values())
                reserve=max(0,len(active)*estimate-live)
                headroom=max(0,obs['available_ram']-reserve)
                slots=max(0,min(8-len(active),int(obs['idle_cpu'])))
                while queue and slots>0 and headroom>=estimate and not errors:
                    index=queue.pop(0)
                    f=pool.submit(price_job,(row,endpoint,s.COHORT,index,str(base/str(index))))
                    active[f]=index;slots-=1;headroom-=estimate
                observations.append(dict(phase='remaining',observation=obs,active=len(active),
                    pending=len(queue),memory_admission_estimate=estimate,future_growth_reserve=reserve))
                s.save(s.PRIVATE/'price_resources.json',observations)
                if not active:
                    # No task is stopped; report unavailable fixed samples explicitly.
                    if queue:
                        s.save(s.PUBLIC/'price_unacquired.json',dict(candidate=row,indices=queue,
                            reason='Current idle CPU/RAM cannot admit another trial at the measured peak estimate',
                            observation=obs,estimate=estimate))
                    break
                done,_=wait(active,timeout=30,return_when=FIRST_COMPLETED)
                for f in done:
                    index=active.pop(f)
                    try:
                        value=f.result();accepted.append(value);fresh.append(value)
                    except BaseException as exc:
                        errors.append(dict(index=index,reason=str(exc)))
                if errors:queue=[]
                s.save(s.PUBLIC/'price_progress.json',dict(complete=len(accepted),planned=9,errors=errors))
    s.save(s.PUBLIC/'price_trials.json.gz',accepted)
    summaries=[]
    for row,ep in zip(selection['candidates'],selection['endpoints']):
        trials=[t for t in accepted if t['candidate_id']==row['id']]
        if not trials:continue
        out=s.old.paired_G(trials,ep,s.COHORT)
        observed={str(k):sum(t['orders'].get(str(k),0) for t in trials) for k in range(0,row['K']+1,2)}
        rarity=native.rarity(trials[0]['laws'],CTX.preps[0],row['K'],len(trials),observed)
        from fractions import Fraction
        out.update(candidate=row,endpoint=ep,rarity=rarity,
            mean_C_cos_exact=str(Fraction(sum(t['prices']['cosine']['rz_count'] for t in trials),len(trials))),
            mean_C_sin_exact=str(Fraction(sum(t['prices']['sine']['rz_count'] for t in trials),len(trials))),
            mean_CX_cos_exact=str(Fraction(sum(t['prices']['cosine']['cx_count'] for t in trials),len(trials))),
            mean_CX_sin_exact=str(Fraction(sum(t['prices']['sine']['cx_count'] for t in trials),len(trials))))
        one_ids={c.component_id for c in CTX.preps[0].rte_preparation.symbolic_tail.components if c.df_fragment_id=='full-h-one-body'}
        observed_one=0
        for t in trials:
            events=s.read(s.PRIVATE/'cost'/s.COHORT/row['id']/str(t['index'])/'events_PRIVATE.json')['events']
            observed_one+=sum(app['component_id'] in one_ids for batch in events for event in batch for app in event['application_sequence'])
        s.require(observed_one>0,'ONE_BODY_ACTUALLY_SAMPLED')
        out['observed_one_body_applications']=observed_one;summaries.append(out)
    s.save(s.PUBLIC/'prices.json',dict(status='PASS' if not errors else 'STOP',summaries=summaries,
        accepted=len(accepted),planned=9*len(selection['candidates']),errors=errors,
        all_workers_joined=True,cohort=s.COHORT,pooled=False,automatic_top_up=False))
    print(dict(accepted=len(accepted),errors=errors))


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        s.save(s.PUBLIC/'price_STOP.json',dict(reason=str(exc),type=type(exc).__name__))
        raise
