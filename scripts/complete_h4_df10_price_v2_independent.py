"""Prepared continuation only: replay saved PR33; requires explicit STOP release.

Never run this automatically after run01. No RNG, PF compile, fit, or new M0.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import importlib.util
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_h4_df10_price_v2_independent as original

OLD=original.PRIVATE
NEW=OLD.parent/'completion01'
PUBLIC=original.PUBLIC/'completion01'
RUNNER=original


def replay_requests(prep, raw):
    """Decode exact applications, reconstruct config without sampling."""
    from h46_same_event import decode_event
    from h46_pr_pilot_repair import canonical_tau
    from trotterlib import rte
    from trotterlib.df_rte_circuit import DFRTEEventSequenceCircuitRequest
    from trotterlib.df_partial_s2 import DFPartialS2StepRequest
    from trotterlib.pr2_matched_accuracy_m1_execution import _explicit_cutoff_tolerance
    row=raw['candidate'];requests=[];laws=[]
    if row['q']!=16 or row['T']!=1.6 or row['R_outer']!=4 or row['K']!=2:
        raise ValueError('REPLAY_FIXED_SCOPE')
    for batch,seed in zip(raw['events'],raw['seeds'],strict=True):
        events=tuple(decode_event(rte,e) for e in batch)
        if json.dumps([e.to_dict() for e in events],sort_keys=True)!=json.dumps(batch,sort_keys=True):
            raise ValueError('EVENT_ROUNDTRIP')
        for event in events:prep.rte_preparation.resolve_event_basis_definitions(event)
        tau=canonical_tau(prep,.1,4)
        config,dist=rte.make_rte_config(prep.rte_preparation.symbolic_tail,evolution_time=.1,
            rte_steps=4,finite_taylor_order=2,truncation_tolerance=_explicit_cutoff_tolerance(tau,2),seed=seed)
        occurrence=DFRTEEventSequenceCircuitRequest(events=events,
            component_specs=prep.rte_preparation.component_specs,controlled=True,
            ancilla_qubit=prep.num_system_qubits,tail_id=config.tail_id,tail_hash=config.tail_hash,
            occurrence_rte_steps=4)
        requests.append(DFPartialS2StepRequest(preparation=prep,step_time=.1,rte_config=config,
            rte_distribution=dist,rte_occurrence=occurrence,controlled=True,
            ancilla_qubit=prep.num_system_qubits,seed=seed))
        laws.append(dict(steps=4,orders=list(dist.orders),probabilities=list(dist.order_probabilities)))
    if len(requests)!=16 or laws!=raw['laws']:
        raise ValueError('REPLAY_LAW_COUNT_BINDING')
    return requests,raw['seeds'],laws


def saved_requests(row,prep,cohort,index):
    return RUNNER.SAMPLED[row['id'],index]


def task(job):
    return RUNNER.trial(job)


def main():
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    own='scripts/complete_h4_df10_price_v2_independent.py'
    if (ROOT/own).read_bytes()!=original.blob(source,own):raise ValueError('COMMIT_CONTINUATION_FIRST')
    stop=json.loads((OLD/'STOP.json').read_text())
    old_source=stop['source']
    for p in original.SOURCE_FILES+original.ORIGINAL_FILES+['scripts/h46_same_event.py']:
        if (ROOT/p).read_bytes()!=original.blob(old_source,p):raise ValueError('FIXED_SOURCE_CHANGED:'+p)
    import h46_long_time_native as native
    cfg=native.study.install()
    import trottertracks
    trottertracks.__path__.insert(0,str(Path(cfg['provider_root'])/'src/trottertracks'))
    predictions=json.loads((original.PUBLIC/'PREDICTIONS.json').read_bytes())
    provenance=json.loads((original.PUBLIC/'PROVENANCE.json').read_bytes())
    if old_source!=predictions['source'] or old_source!=provenance['source']:
        raise ValueError('ORIGINAL_EXECUTION_SOURCE_BINDING')
    for p in ['src/trottertracks/resource_applicability/ax2b_native_df_v5.py',
              'src/trottertracks/resource_applicability/h6_hash_reuse_v1.py',
              'src/trotterlib/df_rte_qiskit.py','src/trotterlib/df_partial_s2.py',
              'src/trotterlib/rte_compiled_cost.py','src/trotterlib/rte.py']:
        if (Path(cfg['provider_root'])/p).read_bytes()!=original.blob('c31ef424175c02379eafebe8fc6c7548c1d9aabe',p):
            raise ValueError('PROVIDER_SOURCE_CHANGED:'+p)
    if original.digest((original.PUBLIC/'PREDICTIONS.json').read_bytes())!=provenance['frozen_predictions_sha256']:
        raise ValueError('FROZEN_PREDICTIONS_CHANGED')
    raws=native.study.read(OLD/'COHORT_PRIVATE.json.gz')
    pf_raw=next(r for r in raws if r['candidate']['method']=='B0')
    pf_path=OLD/'cost'/original.model.COHORT/pf_raw['candidate']['id']/'0/result.json'
    pf=json.loads(pf_path.read_bytes())
    pf_id=next(p['candidate_id'] for p in predictions['predictions'] if p['method']=='PF')
    if pf['status']!='PASS' or pf['candidate']['id']!=pf_id:raise ValueError('PF_NOT_COMPLETE_OR_BOUND')
    NEW.mkdir(parents=True,exist_ok=False);PUBLIC.mkdir(parents=True,exist_ok=False)
    native.PRIVATE=NEW;original.PRIVATE=NEW;original.NATIVE=native
    original.ORIGINAL_COMPILE=native.compile_cached
    try:
        for name,meta in cfg['systems']['H4']['inputs'].items():
            if original.digest(Path(meta['path']).read_bytes())!=provenance['input_sha256'][name]:
                raise ValueError('SAVED_INPUT_CHANGED')
        native.CONTEXT['H4']=native.native_context('H4',[1])
        import run_h4_df10_price_v1 as v1
        prep=native.CONTEXT['H4'].preparations[1]
        if prep.preparation_hash!=v1.PREPARATION:raise ValueError('PREPARATION_CHANGED')
        jobs=[]
        for raw in raws:
            if raw['candidate']['method']!='PR':continue
            row,index=raw['candidate'],raw['index']
            event_hash=original.digest(json.dumps(raw['events'],sort_keys=True,separators=(',',':')).encode())
            expected=[p['events_sha256'] for p in predictions['predictions']
                      if p['method']=='PR' and p['candidate_id']==row['id'] and p['path_index']==index]
            if expected!=[event_hash,event_hash]:raise ValueError('FROZEN_EVENT_PREDICTION_BINDING')
            seeds=[native.trial_seed(row,original.model.COHORT,index,o,0) for o in range(16)]
            if seeds!=raw['seeds']:raise ValueError('FROZEN_SEED_RULE')
            original.SAMPLED[row['id'],index]=replay_requests(prep,raw)
            original.EVENT_HASHES[row['id'],index]=event_hash
            jobs.append((row,original.model.COHORT,index))
        if [j[2] for j in jobs]!=list(range(33)):raise ValueError('SAVED_PR33_SCOPE')
        native.sample_requests=saved_requests
        current=[''];old_trial=native.cost_trial
        def compiled(wrapper,fingerprint,representative):
            record,qc,hit=original.cached(wrapper,fingerprint,representative)
            path=Path(current[0])/'bindings.json'
            entries=json.loads(path.read_bytes()) if path.exists() else []
            entries.append(dict(axis=('cosine','sine')[len(entries)],fingerprint=fingerprint,
                compiler=original.model.COMPILER,cache_hit=hit,
                old_cache_hit=(original.OLD_CACHE/(fingerprint+'.json')).exists()))
            native.save(path,entries)
            return record,qc,hit
        def local_trial(job):
            current[0]=str(NEW/'cost'/job[1]/job[0]['id']/str(job[2]))
            return old_trial(job)
        native.compile_cached=compiled;native.cost_trial=local_trial
        results=[pf];errors=[]
        with ProcessPoolExecutor(max_workers=min(33,len(os.sched_getaffinity(0))),mp_context=mp.get_context('fork')) as pool:
            pending={pool.submit(task,jobs[0]):jobs[0]}
            while pending:
                future=next(as_completed(pending));job=pending.pop(future)
                try:
                    results.append(future.result())
                    print(json.dumps(dict(PR_complete=len(results)-1,index=job[2])),flush=True)
                    if job[2]==0:
                        pending.update({pool.submit(task,j):j for j in jobs[1:]})
                except BaseException as error:errors.append(dict(index=job[2],reason=str(error)))
                native.save(NEW/'progress.json',dict(PR_complete=len(results)-1,errors=errors))
        if errors:raise RuntimeError('CONTINUATION_STOP:'+json.dumps(errors))
        residuals=original.model.residual_rows(predictions['predictions'],results)
        summaries=original.model.summarize(residuals)
        import numpy as np
        for s in summaries:
            group=[r for r in residuals if all(r[k]==s[k] for k in ('method','model','axis'))]
            metrics=s['metrics'];sd=metrics['residual_SD'];n=s['n_paths']
            metrics['residual_variance']=sd**2 if sd is not None else None
            metrics['descriptive_mean_error_SE_under_IID_given_fit']=sd/math.sqrt(n) if sd is not None else None
            metrics['observed_price_SD']=float(np.std([r['compiled_rz'] for r in group],ddof=1)) if n>1 else None
            metrics['observed_mean_SE_under_IID']=metrics['observed_price_SD']/math.sqrt(n) if n>1 else None
        new_compiles=sum(not b['cache_hit'] for t in results[1:] for b in t['wrapper_bindings'])
        if new_compiles>66:raise ValueError('PR_ONLY_COMPILE_SCOPE')
        native.save(PUBLIC/'RESULT.json',dict(status='INDEPENDENT_Q_COMPLETE_AFTER_TECHNICAL_STOP',
            source=source,original_source=old_source,frozen_prediction_source=predictions['source'],
            frozen_predictions_sha256=provenance['frozen_predictions_sha256'],
            reused_PF_result_sha256=original.digest(pf_path.read_bytes()),
            b=original.model.B,summaries=summaries,
            trials=[dict(candidate_id=t['candidate']['id'],index=t['index'],prices=t['prices'],
                wrapper_bindings=t['wrapper_bindings'],checks=t['checks'],orders=t['orders']) for t in results],
            new_PR_wrapper_compiles=new_compiles,new_PF_compiles=0,new_sampling=0,new_predictions=0,
            fit_calls=0,new_N_calls=0,N_times_C_calls=0,long_q_qualified=False,
            total_CI=None,original_STOP_preserved=True,mandatory_STOP=True))
        with (PUBLIC/'PATH_RESIDUALS.csv').open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(residuals[0]),lineterminator='\n')
            writer.writeheader();writer.writerows(residuals)
        native.save(NEW/'COMPLETE.json',dict(source=source,PR_complete=33,mandatory_STOP=True))
    except BaseException as error:
        native.save(NEW/'STOP.json',dict(source=source,reason=str(error),mandatory_STOP=True))
        raise


if __name__=='__main__':main()
