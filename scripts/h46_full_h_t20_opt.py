"""T=2.0 full-H RTE grid and adjacent-boundary optimization.

The full-H preparation, finite-mean, sampler, native builder, compiler and
statistics are reused. Pure-S2 means depend on s=q*R; prices remain measured.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import copy
import gzip
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import resource
import time
from types import SimpleNamespace
import traceback

import h46_full_h_pure_rte as full

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'artifacts/resource_applicability/df_full_h_pure_rte_t20_opt/2026-11-09'
PRIVATE=PUBLIC/'PRIVATE'
BASE=full.BASE
B3=full.B3
COHORT='FULL_H_T20_EXPLORATION'
read,save=full.read,full.save
CONTEXT={}


def setup():
    native=full.setup()
    full.PUBLIC,full.PRIVATE=PUBLIC,PRIVATE
    full.signal.PUBLIC,full.signal.PRIVATE=PUBLIC,PRIVATE
    native.PUBLIC,native.PRIVATE=PUBLIC,PRIVATE
    native.trial_seed=trial_seed
    # Only the action verifier changes implementation, never the priced circuit
    # or Qiskit transpiler. Its agreement with Statevector is tested separately.
    from trotterlib.df_trotter import circuit
    circuit.simulate_statevector=aer_statevector
    return native


def trial_seed(row,cohort,index,outer,piece):
    return int(full.digest(['FULL_H_T20_OPT_20261109',cohort,row['id'],index,outer,piece])[:16],16)


def row(system,q,r,k,stage):
    v=dict(system=system,T=2.0,q=q,R_outer=r,K=k,s=q*r,delta=2.0/q,
        formula='2nd',method='PR',family='FULL_H_PURE_RTE',prefix=0,L_D=0,grid_stage=stage)
    v['id']='FH20_'+full.digest([system,q,r,k])[:20]
    return v


def aer_statevector(qc,initial):
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    # Native bases use this one non-Aer gate. Expand its exact definition;
    # compiled circuits already consist only of rz/sx/x/cx.
    if any(i.operation.name=='xx_plus_yy' for i in qc.data):
        qc=qc.decompose(gates_to_decompose=['xx_plus_yy'])
    sim=QuantumCircuit(qc.num_qubits)
    sim.set_statevector(np.asarray(initial,dtype=complex))
    sim.compose(qc,inplace=True)
    sim.save_statevector()
    backend=AerSimulator(method='statevector',device='CPU',precision='double',
        max_parallel_threads=1,zero_threshold=0)
    result=backend.run(sim,shots=1).result()
    full.require(result.success,'AER_STATEVECTOR')
    return np.asarray(result.get_statevector(sim),dtype=complex)


def context(system):
    if system in CONTEXT:return CONTEXT[system]
    ctx=full.context(system)
    original=read(full.BASE_PRIVATE/(system+'_context.json'))
    z=original['reference']['2.0']
    ctx.refs={2.0:complex(z['real'],z['imag'])}
    full.require(not ctx.preparations[0].deterministic_blocks,'NO_DETERMINISTIC_BLOCKS')
    full.require(ctx.preparations[0].exact_rte_lambda_r==read(ROOT/'artifacts/resource_applicability/df_full_h_pure_rte_light/2026-11-09/input_and_validation_summary.json')[system]['lambda_R'],'UNCHANGED_FULL_H_PREPARATION')
    info=read(PRIVATE/(system+'_context.json'))
    info.update(evaluation_T=2.0,reference={'2.0':z},reference_source=str(full.BASE_PRIVATE/(system+'_context.json')),
        reused_previous_correctness='9b83b96b83f62142c9d87858dc24a94db7ac5d61',new_reference_calls=0)
    save(PRIVATE/(system+'_context.json'),info)
    CONTEXT[system]=ctx
    return ctx


def initial_rows():
    saved=read(B3/'signal_records.json.gz')
    return [row(r['system'],r['q'],r['R_outer'],r['K'],'INITIAL_B3_GRID')
        for rec in saved.values() if (r:=rec['candidate'])['T']==2.0 and r['formula']=='2nd']


def evaluate(rows,phase):
    from h46_long_time_report import write_csv
    setup()
    records=read(PUBLIC/'signal_records.json.gz') if (PUBLIC/'signal_records.json.gz').exists() else {}
    for system in sorted({r['system'] for r in rows}):
        ctx=context(system)
        pending=[r for r in rows if r['system']==system and r['id'] not in records]
        groups={}
        for r in pending:groups.setdefault((r['s'],r['K']),[]).append(r)
        # The exact pure-H polynomial and normalization depend only on s and K.
        canonical=[row(system,1,s,k,'SIGNAL_EQUIVALENCE') for s,k in groups]
        batch=full.signal.evolve_batch(ctx,canonical) if canonical else []
        for (s,k),rec in zip(groups,batch):
            for r in groups[s,k]:
                value=copy.deepcopy(rec);value['candidate']=r
                value['origin']='REUSED_EVOLVE_BATCH_PURE_S2_S_EQUIVALENCE'
                value['signal_equivalence']=dict(s=s,K=k,canonical_q=1,canonical_R=s,priced_equivalence_claim=False)
                records[r['id']]=value
        save(PUBLIC/'signal_records.json.gz',records)
    endpoints=[ep for rec in records.values() for ep in full.signal.qualify(rec,rec['candidate'])]
    previous=read(ROOT/'artifacts/resource_applicability/df_full_h_pure_rte_light/2026-11-09/price_trials.json')
    scale={s:sum(t['prices']['cosine']['rz_count']/t['candidate']['s'] for t in previous if t['candidate']['system']==s)/9 for s in ('H4','H6')}
    from trotterlib.rte import finite_rte_distribution
    for ep in endpoints:
        r=ep['candidate'];lam=context(r['system']).preparations[0].exact_rte_lambda_r
        dist=finite_rte_distribution(lam*(2.0/r['s']),r['K'])
        expected_order=math.fsum(k*p for k,p in zip(dist.orders,dist.order_probabilities))
        ep['proxy_C']=scale[r['system']]*r['s']*(1+expected_order)
        ep['proxy_G']=None if ep['status']!='ELIGIBLE' else ep['proxy_C']*(ep['N_cos']+ep['N_sin'])
        ep['proxy_is_compiled_price_or_bound']=False
    save(PUBLIC/'endpoints.json.gz',endpoints)
    inventory=[dict(r,signal_status='EVALUATED',price_status='ACQUIRED' if (PUBLIC/'costs.json.gz').exists() and r['id'] in read(PUBLIC/'costs.json.gz') else 'NOT_ACQUIRED') for r in (rec['candidate'] for rec in records.values())]
    save(PUBLIC/'candidate_inventory.json',inventory)
    summary=dict(phase=phase,total_candidates=len(records),new_candidates=len(rows),endpoints=len(endpoints),
        eligible=sum(ep['status']=='ELIGIBLE' for ep in endpoints),new_reference_calls=0,
        proxy='Previous compiled RZ/event times s times (1+expected event product order); screening only')
    save(PUBLIC/(phase+'_signal_summary.json'),summary)
    return summary


def validate_t20():
    import numpy as np
    setup();checks=[]
    for s in ('H4','H6'):
        ctx=context(s)
        # Compare collapsed and noncollapsed execution through the original API.
        a=row(s,8,64,2,'VALIDATION');b=row(s,1,512,2,'VALIDATION')
        x=full.signal.evolve_batch(ctx,[a])[0];y=full.signal.evolve_batch(ctx,[b])[0]
        error=max(abs(x['signal'][key][axis]-y['signal'][key][axis]) for key in ('corrected','raw') for axis in ('real','imag'))
        full.require(error<1e-9,'PURE_S_EQUIVALENCE')
        full.require(abs(x['signal']['log_B']-y['signal']['log_B'])<1e-10,'PURE_S_NORMALIZATION')
        # Reuse one-body/mixed event and actual-adjoint checks at the new time.
        result=full.validate(s,row(s,1,512,2,'T20_EVENT_CHECK'))
        checks.append(dict(system=s,signal_s_equivalence_error=error,
            log_B_equivalence_error=abs(x['signal']['log_B']-y['signal']['log_B']),
            T=2.0,conditional_event_checks=result['conditional_event_checks'],
            deterministic_blocks=0,previous_hamiltonian_state_preparation_preserved=True))
    save(PUBLIC/'T20_validation.json',dict(status='PASS',checks=checks,old_correctness_preserved=True))
    return checks


def worker(job):
    native=setup();r=job[0]
    native.CONTEXT[r['system']]=context(r['system'])
    return native.cost_trial(job)


def price(plan_path):
    native=setup();plan=read(plan_path);cohort=plan.get('cohort',COHORT)
    costs=read(PUBLIC/'costs.json.gz') if (PUBLIC/'costs.json.gz').exists() else {}
    jobs=[];groups={}
    n=plan.get('n',9)
    for r in plan['candidates']:
        if cohort==COHORT and r['id'] in costs:continue
        context(r['system']);native.CONTEXT[r['system']]=CONTEXT[r['system']]
        groups[r['id']]=[]
        for i in range(n):
            path=PRIVATE/'cost'/cohort/r['id']/str(i)
            full.require(not path.exists(),'TRIAL_DIRECTORY_EXISTS_NO_AUTOMATIC_RETRY:'+str(path))
            jobs.append((r,cohort,i))
    if not jobs:return dict(new_trials=0)
    from pf_common_run_contract import resources
    obs=resources()
    # Scale a measured previous peak with event count, including temporary
    # circuit/transpiler copies. This is an admission estimate, not an RSS cap.
    previous=read(ROOT/'artifacts/resource_applicability/df_full_h_pure_rte_light/2026-11-09/price_trials.json')
    proxy=max(max(t['peak_RSS_bytes']/t['candidate']['s'] for t in previous if t['candidate']['system']==r['system'])*r['s']*1.25 for r in plan['candidates'])
    workers=max(1,min(len(jobs),int(obs['idle_cpu']),int(obs['available_ram']//proxy)))
    save(PRIVATE/(plan['phase']+'_resources.json'),dict(observation=obs,workers=workers,memory_admission_proxy=proxy,arbitrary_caps=None,BLAS_threads_per_process=1,simulation_device='CPU'))
    errors=[];byid={r['id']:[j for j in jobs if j[0]['id']==r['id']] for r in plan['candidates']}
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('fork')) as pool:
        pending={pool.submit(worker,j):j for j in jobs if j[2]==0}
        while pending:
            future=next(as_completed(pending));job=pending.pop(future)
            try:
                value=future.result();groups[job[0]['id']].append(value)
                if job[2]==0 and not errors:
                    for other in byid[job[0]['id']]:
                        if other[2]>0:pending[pool.submit(worker,other)]=other
            except BaseException as exc:
                errors.append(dict(candidate=job[0]['id'],index=job[2],reason=str(exc)))
            save(PRIVATE/(plan['phase']+'_progress.json'),dict(complete=sum(map(len,groups.values())),total=len(jobs),errors=errors))
    if errors:
        save(PUBLIC/(plan['phase']+'_STOP.json'),errors)
        raise ValueError('SCIENTIFIC_TRIAL_STOP')
    result={}
    for r in plan['candidates']:
        if r['id'] not in groups:continue
        ts=sorted(groups[r['id']],key=lambda t:t['index']);full.require(len(ts)==n,'FIXED_COHORT_SIZE')
        expected=read(PUBLIC/'signal_records.json.gz')[r['id']]['signal']['log_B']
        full.require(max(abs(t['log_B']-expected) for t in ts)<1e-9,'MEAN_NATIVE_LOG_B')
        result[r['id']]=native.summarize_trials(r,cohort,ts)
    if cohort==COHORT:
        costs.update(result);save(PUBLIC/'costs.json.gz',costs)
    else:save(PUBLIC/(plan['phase']+'_costs.json.gz'),result)
    save(PUBLIC/(plan['phase']+'_completion.json'),dict(status='PASS',new_trials=len(jobs),groups=len(result),all_workers_joined=True,cohort=cohort))
    return dict(new_trials=len(jobs),groups=len(result),workers=workers)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['validate','initial','evaluate','price']);p.add_argument('--plan')
    args=p.parse_args();setup()
    if args.command=='validate':value=validate_t20()
    elif args.command=='initial':value=evaluate(initial_rows(),'initial')
    elif args.command=='evaluate':
        plan=read(args.plan);value=evaluate(plan['candidates'],plan['phase'])
    else:value=price(args.plan)
    print(json.dumps(value))


if __name__=='__main__':main()
