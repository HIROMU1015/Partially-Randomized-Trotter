"""Native full-wrapper costs for the new H46 finite-time survey.

Reuses saved preparation, exact signed schedules, RTE sampler, native lowerer,
fingerprints, compiler and paired statistics. Never calls old launchers/grants.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from functools import partial
import gzip
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import resource
import time
import traceback
from types import SimpleNamespace

import h46_long_time_signal as study

ROOT,PUBLIC,PRIVATE=study.ROOT,study.PUBLIC,study.PRIVATE
read,save,sha=study.read,study.save,study.sha
CONTEXT={}
METRICS=('rz_count','rz_depth','cx_count','cx_depth','total_depth','circuit_size')
COMPILER_HASH='1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b'


def trial_seed(row,cohort,index,outer,piece):
    data=['H46_LONG_TIME_20261025',cohort,row['id'],index,outer,piece]
    return int(hashlib.sha256(json.dumps(data).encode()).hexdigest()[:16],16)


def compose(row,prep,requests,scalar,fingerprint):
    from qiskit import QuantumCircuit
    from trottertracks.resource_applicability import ax2b_native_df_v5 as native
    from trotterlib.df_rte_qiskit import QiskitDFRTEEventCircuitBuilder
    from h46_m5_contract import core
    steps=core.schedule(row['formula'],row['q'],
        'ABS_TIME_PER_OUTER' if requests else 'PER_PIECE',row['R_outer'] if requests else 1,T=row['T'])
    if requests and len(steps)!=len(requests):
        raise ValueError('SIGNED_REQUEST_COUNT')
    n=prep.num_system_qubits;qc=QuantumCircuit(n+1);bound=1
    builder=QiskitDFRTEEventCircuitBuilder(basis_registry=prep.rte_preparation.basis_registry) if requests else None
    blocks=prep.deterministic_blocks
    base=sum(native.block_instruction_bound(b,mode) for b in blocks for mode in ('UNCONTROLLED','DIRECTIONAL'))
    for index,s in enumerate(steps):
        if requests:
            req=requests[index]
            if req.step_time!=s['signed_time'] or req.preparation.preparation_hash!=prep.preparation_hash:
                raise ValueError('SIGNED_NATIVE_IDENTITY')
            tail=builder.build_sequence(req.rte_occurrence,basis_plan=req.rte_basis_plan).circuit
            part=native.build_partial_native(blocks,num_system_qubits=n,T=s['signed_time'],q=1,
                scalar=0.,controlled_tails=[tail],control_policy='symmetric_directional',
                max_instructions=base+len(tail.data)+1)
        else:
            part=native.build_deterministic_native(blocks,num_system_qubits=n,T=s['signed_time'],q=1,
                formula='2nd',scalar=0.,control_policy='symmetric_directional',max_instructions=base+1)
        qc.compose(part.circuit,inplace=True);bound+=part.instruction_upper_bound
    if scalar*row['T']!=0:
        qc.p(-scalar*row['T'],n)
    return native.NativeEvolution(qc,n,row['T'],row['q'],row['formula'],
                                  'symmetric_directional',bound,fingerprint(qc))


def sample_requests(row,prep,cohort,index):
    from h46_m5_contract import core
    from h46_pr_pilot_repair import canonical_tau
    from trotterlib.df_partial_s2 import make_df_partial_s2_step_request
    from trotterlib.pr2_matched_accuracy_m1_execution import _explicit_cutoff_tolerance
    requests=[];seeds=[];laws=[]
    for s in core.schedule(row['formula'],row['q'],'ABS_TIME_PER_OUTER',row['R_outer'],T=row['T']):
        seed=trial_seed(row,cohort,index,s['outer'],s['piece'])
        tau=canonical_tau(prep,s['signed_time'],s['rte_steps'])
        req=make_df_partial_s2_step_request(prep,step_time=s['signed_time'],rte_steps=s['rte_steps'],
            truncation_tolerance=_explicit_cutoff_tolerance(tau,row['K']),finite_taylor_order=row['K'],
            seed=seed,controlled=True,ancilla_qubit=prep.num_system_qubits,pf_label='2nd')
        if len(req.rte_occurrence.events)!=s['rte_steps']:
            raise ValueError('RTE_ALLOCATION_COUNT')
        requests.append(req);seeds.append(seed)
        laws.append(dict(steps=s['rte_steps'],orders=list(req.rte_distribution.orders),
                         probabilities=list(req.rte_distribution.order_probabilities)))
    return requests,seeds,laws


def cdf_law(probabilities):
    """Actual float64 CDF thresholds of NumPy choice on a 53-bit U[0,1) grid.

    This describes rounding of the categorical map, not a proof that a fixed
    finite PRNG stream is an independent random sample.
    """
    import numpy as np
    values=np.asarray(probabilities,dtype=float);cdf=np.cumsum(values);cdf/=cdf[-1]
    grid=2**53;previous=0;effective=[]
    for x in cdf:
        v=Fraction.from_float(float(x))*grid
        upper=-(-v.numerator//v.denominator)
        effective.append(Fraction(upper-previous,grid));previous=upper
    if previous!=grid:
        raise ValueError('FINITE_BIT_CDF_MASS')
    ideal=[Fraction.from_float(float(x)) for x in values]
    total=sum(ideal);ideal=[x/total for x in ideal]
    return dict(probabilities=[float(x) for x in effective],
                L1_roundoff=float(sum(abs(a-b) for a,b in zip(effective,ideal))),
                map='NUMPY_FLOAT64_CUMSUM_NORMALIZED_SEARCHSORTED_RIGHT_53BIT_UNIFORM',
                PRNG_independence_proved=False)


def rarity(laws,prep,K,n,observed):
    # Generic extension of h46_pr_pilot_evaluation.mixture to K2/K6. Keep
    # arbitrary-precision probabilities so rare unobserved orders remain >0.
    from decimal import Decimal,localcontext
    with localcontext() as ctx:
        ctx.prec=80
        no={k:Decimal(1) for k in range(2,K+1,2)};roundoff=0.
        for law in laws:
            bit=cdf_law(law['probabilities']);roundoff+=law['steps']*bit['L1_roundoff']
            for order in no:
                chance=sum(Decimal.from_float(float(p)) for k,p in zip(law['orders'],bit['probabilities']) if k>=order)
                no[order]*=(1-chance)**law['steps']
        probabilities={str(k):float(1-v) for k,v in no.items()}
        missed={str(k):float(v**n) for k,v in no.items()}
    comps=prep.rte_preparation.symbolic_tail.components
    weights=[c.probability for c in comps];total=math.fsum(weights)
    comp=cdf_law([p/total for p in weights])
    draws=sum(l['steps'] for l in laws)
    return dict(probability_at_least_one_order_or_higher=probabilities,
         probability_unobserved_in_n=missed,observed_order_counts=observed,
         unknown_conditional_costs_for_unobserved_orders={str(k):None for k in range(2,K+1,2) if observed.get(str(k),0)==0},
         sampling_law='UNCHANGED_PCG64_NUMPY_CHOICE_FINITE_PAIRED_RTE',
         finite_bit_map=comp['map'],component_L1_roundoff=comp['L1_roundoff'],
         trajectory_order_L1_roundoff_sum=roundoff,
         trajectory_component_L1_roundoff_union_proxy=draws*(K+1)*comp['L1_roundoff'],
         population_tail_cost_bound=None,unobserved_cost_imputed=False,PRNG_independence_proved=False)


def native_context(system,prefixes):
    import numpy as np
    import qiskit
    from h46_m5_tol_native import restore
    from h46_new2_native import OccupationOracle,prepare
    cfg=study.origin_config();info=cfg['systems'][system]
    ham,sector,full,state,metadata=restore(dict(system=system,**info))
    if qiskit.__version__!='1.3.0' or np.__version__!='1.26.4':
        raise ValueError('COMMON_COMPILER_RUNTIME_VERSION')
    import trotterlib.rte_compiled_cost as costs
    costs.transpile=partial(qiskit.transpile,num_processes=1)
    oracle=OccupationOracle(ham,full,prefix=ham.n_blocks)
    qstate=np.zeros(1<<ham.n_qubits,complex);qstate[oracle.qindices]=oracle.initial
    preparations={p:prepare(ham,p) for p in prefixes}
    for p,prep in preparations.items():
        if tuple(prep.deterministic_fragment_indices)!=tuple(range(p)):
            raise ValueError('DF_GENERATION_ORDER')
    return SimpleNamespace(ham=ham,oracle=oracle,qstate=qstate,preparations=preparations)


def compile_cached(wrapper,fingerprint,representative):
    import fcntl
    from qiskit import qpy
    from trotterlib.rte import CompilerSettings
    from trotterlib.rte_compiled_cost import transpile_and_measure_cost
    cache=PRIVATE/'compiled_cache';cache.mkdir(exist_ok=True)
    path=cache/(fingerprint+'.json')
    with (cache/(fingerprint+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if path.exists():
            return read(path),None,True
        compiler=CompilerSettings(('rz','sx','x','cx'),None,None,1,None,None,17,'1.3.0')
        cost=transpile_and_measure_cost(wrapper,compiler,actual_circuit_fingerprint=fingerprint)
        if cost.compiler_settings_hash!=COMPILER_HASH:
            raise ValueError('COMMON_COMPILER_HASH')
        metrics={k:getattr(cost,k) for k in METRICS}
        result=dict(metrics=metrics,compiler=COMPILER_HASH,fingerprint=fingerprint)
        if representative:
            for name,qc in [('wrapper',wrapper),('compiled',cost.transpiled_circuit)]:
                stream=io.BytesIO();qpy.dump(qc,stream)
                (cache/(fingerprint+'_'+name+'.qpy.gz')).write_bytes(gzip.compress(stream.getvalue(),compresslevel=1,mtime=0))
        save(path,result)
        return result,cost.transpiled_circuit,False


def cost_trial(job):
    import numpy as np
    from qiskit import QuantumCircuit
    from trottertracks.resource_applicability import ax2b_native_df_v5 as native
    from trottertracks.resource_applicability.h6_hash_reuse_v1 import use_fingerprint_reuse
    from trotterlib.df_trotter.circuit import simulate_statevector
    from h46_m5_contract import core
    row,cohort,index=job;started=time.monotonic();ctx=CONTEXT[row['system']]
    directory=PRIVATE/'cost'/cohort/row['id']/str(index)
    directory.mkdir(parents=True,exist_ok=False)
    def gate(name,value,tolerance=1e-9):
        value=float(value)
        if not math.isfinite(value) or value>=tolerance:
            raise ValueError(name+':'+str(value))
        return dict(check=name,error=value,tolerance=tolerance,phase_alignment=False)
    try:
        prep=ctx.preparations[row['prefix']];pr=row['method']=='PR'
        requests,seeds,laws=sample_requests(row,prep,cohort,index) if pr else ([],[],[])
        events=[[e.to_dict() for e in r.rte_occurrence.events] for r in requests]
        save(directory/'events_PRIVATE.json',dict(events=events,seeds=seeds,conditional=False,cohort=cohort))
        scalar=ctx.ham.constant+(prep.extracted_identity_coefficient if row['method'] in ('PR','B0I') else 0.)
        checks=[];prices={};cache_hits=0
        with use_fingerprint_reuse() as fp:
            evolution=compose(row,prep,requests,scalar,fp.fingerprint)
            low=np.r_[ctx.qstate,np.zeros_like(ctx.qstate)]
            if index==0:
                if pr:
                    expected=ctx.oracle.trajectory(requests)
                else:
                    value=ctx.oracle.initial.copy()
                    for s in core.schedule(row['formula'],row['q'],'PER_PIECE',1,T=row['T']):
                        for spec in list(ctx.oracle.specs[:row['prefix']+1])+list(reversed(ctx.oracle.specs[:row['prefix']+1])):
                            value=spec[1]@(np.exp(-.5j*s['signed_time']*spec[0])*(spec[2]@value))
                    value*=np.exp(-1j*row['T']*scalar)
                    expected=np.zeros_like(ctx.qstate);expected[ctx.oracle.qindices]=value
                checks.append(gate('low_control',np.linalg.norm(simulate_statevector(evolution.circuit,low)-low)))
                hi=simulate_statevector(evolution.circuit,np.r_[np.zeros_like(ctx.qstate),ctx.qstate])
                checks.append(gate('high_control_leakage',np.linalg.norm(hi[:len(expected)])))
                checks.append(gate('independent_occupation_vector',np.linalg.norm(hi[len(expected):]-expected)))
                z=complex(np.vdot(ctx.qstate,expected))
            for axis in ('cosine','sine'):
                wrapper=native.build_native_hadamard_wrapper(evolution,axis=axis,include_measurement=True,
                                                           max_instructions=len(evolution.circuit.data)+4)
                fingerprint=fp.fingerprint(wrapper)
                compiled,qc,reused=compile_cached(wrapper,fingerprint,index==0)
                cache_hits+=reused;prices[axis]=compiled['metrics']
                if fp.fingerprint(wrapper)!=fingerprint:
                    raise ValueError('COMPILER_MUTATED_INPUT')
                if index==0:
                    before=simulate_statevector(wrapper.remove_final_measurements(inplace=False),low)
                    observed=float(np.vdot(before[:len(ctx.qstate)],before[:len(ctx.qstate)]).real
                                  -np.vdot(before[len(ctx.qstate):],before[len(ctx.qstate):]).real)
                    checks.append(gate('Hadamard_'+axis,abs(observed-(z.real if axis=='cosine' else z.imag))))
                    if qc is not None:
                        after=simulate_statevector(qc.remove_final_measurements(inplace=False),low)
                        checks.append(gate('compiled_vector_'+axis,np.linalg.norm(after-before),1e-7))
                del wrapper,qc
        orders={str(k):sum(e['taylor_order']==k for batch in events for e in batch) for k in range(0,(row['K'] or 0)+1,2)}
        result=dict(status='PASS',candidate=row,cohort=cohort,index=index,prices=prices,checks=checks,
             laws=laws,orders=orders,conditional=False,scalar=scalar,
             log_B=sum(len(r.rte_occurrence.events)*math.log(r.rte_distribution.exact_finite_distribution) for r in requests),
             preparation_hash=prep.preparation_hash,cache_hits=cache_hits,
             wall_seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        save(directory/'result.json',result);return result
    except BaseException as exc:
        save(directory/'STOP.json',dict(reason=str(exc),traceback=traceback.format_exc()))
        raise


def cost_proxy(row):
    """Screening work proxy ONLY, never a measured cost or a pruning bound."""
    from h46_m5_contract import core
    return row['q']*(len(core.weights(row['formula']))*(row['prefix']+1)
                     +(.25*row['R_outer'] if row['method']=='PR' else 0.))


def select():
    endpoints=read(PUBLIC/'signal_endpoints.json.gz');chosen={};reasons={}
    def add(row,reason):
        chosen[row['id']]=row;reasons.setdefault(row['id'],[]).append(reason)
    for s in study.PREFIX:
        for T in study.Q:
            for f in study.FORMULAS:
                for e in study.EPS:
                    for family in ('PF','PR'):
                        available=[v for v in endpoints if v['status']=='ELIGIBLE' and v['candidate']['system']==s
                            and v['candidate']['T']==T and v['candidate']['formula']==f and v['epsilon']==e
                            and (v['candidate']['method']=='PR')==(family=='PR')]
                        # Two nominally strongest configurations per cell;
                        # evaluated subset minima are explicitly reported.
                        available.sort(key=lambda v:(v['N_cos']+v['N_sin'])*cost_proxy(v['candidate']))
                        for value in available[:2]:
                            add(value['candidate'],'TOP2_SCREEN_PROXY_'+str(e))
                    full=[v for v in endpoints if v['status']=='ELIGIBLE' and v['epsilon']==e
                        and v['candidate']['system']==s and v['candidate']['T']==T
                        and v['candidate']['formula']==f and v['candidate']['method']=='FULL']
                    if full:
                        add(min(full,key=lambda v:(v['N_cos']+v['N_sin'])*cost_proxy(v['candidate']))['candidate'],
                            'FULL_PF_STRONG_CONTROL_'+str(e))
                # Retain a tuned S2 r4/K2 control at every T.
                if f=='2nd':
                    for e in study.EPS:
                        available=[v for v in endpoints if v['status']=='ELIGIBLE' and v['epsilon']==e
                            and v['candidate']['system']==s and v['candidate']['T']==T
                            and v['candidate']['formula']==f and v['candidate']['method']=='PR'
                            and v['candidate']['R_outer']==4 and v['candidate']['K']==2]
                        if available:
                            add(min(available,key=lambda v:(v['N_cos']+v['N_sin'])*cost_proxy(v['candidate']))['candidate'],'S2_R4_K2_STRONG_CONTROL')
    # Same q/prefix, common R32/K4 controls at the chosen PR configurations.
    lookup={r['id']:r for r in study.candidates()}
    eligible={v['candidate']['id'] for v in endpoints if v['status']=='ELIGIBLE'}
    for row in list(chosen.values()):
        if row['method']!='PR':
            continue
        for f in study.FORMULAS:
            control=dict(row,formula=f,R_outer=32,K=4);control['id']=study.cid(control)
            if control['id'] in eligible:
                add(lookup[control['id']],'SAME_Q_PREFIX_R32_K4_CONTROL')
    plan=dict(candidates=list(chosen.values()),reasons=reasons,screening_trajectories_per_new_PR=9,
        nominal_design='Reuse coarse n=ceil(2/.5^2)+1=9 variance-planning reference; not a tail guarantee',
        screening_proxy='q*(signed_pieces*(prefix+1)+0.25*R_outer); not compiled cost or a bound',
        selection='Exploratory cost subset; no full-grid cost optimality claim',
        registered_candidates=len(study.candidates()),unpriced_eligible_candidates=len(eligible-set(chosen)))
    save(PUBLIC/'native_selection.json',plan);return plan


def summarize_trials(row,cohort,trials):
    from h246_saved_cost_evaluation import paired
    pair=[(t['prices']['cosine']['rz_count'],t['prices']['sine']['rz_count']) for t in trials]
    result=dict(candidate=row,cohort=cohort,n=len(trials),unit_statistics=paired(pair),
        mean_metrics={a:{m:sum(t['prices'][a][m] for t in trials)/len(trials) for m in METRICS} for a in ('cosine','sine')},
        observed_maximum_native_error=max((v['error'] for t in trials for v in t['checks'] if not v['check'].startswith('compiled_')),default=0),
        observed_maximum_compiled_error=max((v['error'] for t in trials for v in t['checks'] if v['check'].startswith('compiled_')),default=0),
        cache_hits=sum(t['cache_hits'] for t in trials),trial_files=[str((PRIVATE/'cost'/cohort/row['id']/str(t['index'])/'result.json')) for t in trials],
        population_guarantee=False,conditional_sampling=False,paired_cosine_sine=True)
    if row['method']=='PR':
        counts={str(k):sum(t['orders'].get(str(k),0) for t in trials) for k in range(0,row['K']+1,2)}
        result['rarity']=rarity(trials[0]['laws'],CONTEXT[row['system']].preparations[row['prefix']],row['K'],len(trials),counts)
    return result


def run_cost(cohort='EXPLORATION',selection=None):
    study.install()
    from pf_common_run_contract import resources
    plan=read(PUBLIC/'native_selection.json') if selection is None else selection
    saved=read(PRIVATE/'saved_prices.json')
    groups={};jobs=[];results={};errors=[]
    for row in plan['candidates']:
        if cohort=='EXPLORATION' and row['id'] in saved:
            results[row['id']]={**saved[row['id']], 'candidate':row,
                'reuse_provenance':saved[row['id']].get('origin'), 'origin':'SAVED_COST_NO_BUILD_OR_COMPILE'}
            continue
        groups[row['id']]=[]
        n=(plan.get('n_by_candidate',{}).get(row['id'],plan.get('screening_trajectories_per_new_PR',9))
           if row['method']=='PR' else 1)
        jobs.extend((row,cohort,i) for i in range(n))
    # Reuse the explicit historical exclusion sets; do not search other jobs'
    # histories. Distinct cohort names enter every new seed derivation.
    def integers(value):
        if isinstance(value,int):
            yield value
        elif isinstance(value,dict):
            for item in value.values():
                yield from integers(item)
        elif isinstance(value,list):
            for item in value:
                yield from integers(item)
    known=set()
    for s in study.PREFIX:
        cfg=read(study.ORIGIN/(s+'_PR_EXPLORATION/config_APPROVED_PRIVATE.json'))
        known.update(read(cfg['excluded_seeds_path'])['values'])
        known.update(integers(read(cfg['registry_path'])))
    if cohort!='EXPLORATION' and (PRIVATE/'EXPLORATION_seeds.json.gz').exists():
        known.update(read(PRIVATE/'EXPLORATION_seeds.json.gz'))
    from h46_m5_contract import core
    new=[]
    for row,kind,index in jobs:
        if row['method']!='PR':
            continue
        for step in core.schedule(row['formula'],row['q'],'ABS_TIME_PER_OUTER',row['R_outer'],T=row['T']):
            seed=trial_seed(row,kind,index,step['outer'],step['piece'])
            if seed in known:
                raise ValueError('KNOWN_OR_NEW_SEED_COLLISION')
            known.add(seed);new.append(seed)
    save(PRIVATE/(cohort+'_seeds.json.gz'),new)
    save(PUBLIC/(cohort.lower()+'_seed_independence.json'),dict(new_seeds=len(new),known_seed_collisions=0,
        separate_cohort=True,complete_historical_coverage_claimed=False,seed_values_private=True))
    for s in study.PREFIX:
        prefixes=sorted({r['prefix'] for r,_,_ in jobs if r['system']==s})
        if prefixes:
            CONTEXT[s]=native_context(s,prefixes)
    obs=resources();workers=max(1,min(len(jobs) or 1,int(obs['idle_cpu']),obs['available_ram']//(2*2**30)))
    save(PRIVATE/(cohort+'_cost_resources.json'),dict(observation=obs,workers=workers,nested_compiler_processes=1))
    # Representative/control checks precede the remaining trials for each
    # candidate. A failed candidate is stopped, without discarding other work.
    progress=dict(total=len(jobs),complete=0,errors=[],reused_groups=len(results))
    save(PRIVATE/(cohort+'_cost_progress.json'),progress)
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('fork')) as pool:
        first={pool.submit(cost_trial,j):j for j in jobs if j[2]==0}
        pending=dict(first)
        while pending:
            f=next(as_completed(pending));job=pending.pop(f);row=job[0]
            try:
                result=f.result();groups[row['id']].append(result);progress['complete']+=1
                if job[2]==0:
                    for other in jobs:
                        if other[0]['id']==row['id'] and other[2]>0:
                            pending[pool.submit(cost_trial,other)]=other
            except BaseException as exc:
                error=dict(candidate=row,index=job[2],reason=str(exc));errors.append(error);progress['errors'].append(error)
            save(PRIVATE/(cohort+'_cost_progress.json'),progress)
    failed={e['candidate']['id'] for e in errors}
    for row in plan['candidates']:
        if row['id'] in results or row['id'] in failed:
            continue
        results[row['id']]=summarize_trials(row,cohort,groups[row['id']])
    save(PUBLIC/(cohort.lower()+'_costs.json.gz'),results)
    save(PUBLIC/(cohort.lower()+'_cost_summary.json'),dict(progress,evaluated_groups=len(results),
        all_workers_joined=True,failed_groups=len(failed),population_tail_guarantee=False))
    return dict(progress,evaluated_groups=len(results))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['select','cost']);args=parser.parse_args()
    study.install()
    print(json.dumps(select() if args.command=='select' else run_cost(),ensure_ascii=False))
