"""B3 extension of the completed H46 long-time benchmark; old runs are read-only.

Only study configuration, selection, and checkpoint orchestration are new.
Finite means, signed schedules, allocation, native circuits, compiler, shots,
and paired statistics come from the 155529e H46 implementation.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import time
from types import SimpleNamespace

import h46_long_time_signal as signal

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'artifacts/resource_applicability/track_a_h46_long_time_b3/2026-10-26'
PRIVATE = Path('/home/AbeHiromu/projects/h46-long-time-b3/20261026')
BASE_PUBLIC = ROOT/'artifacts/resource_applicability/track_a_h46_long_time/2026-10-25'
BASE_PRIVATE = Path('/home/AbeHiromu/projects/h46-long-time-pf-pr/20261025')
read, save = signal.read, signal.save
CONTEXT = {}


def setup():
    signal.install()
    signal.PUBLIC, signal.PRIVATE = PUBLIC, PRIVATE
    signal.PREFIX = {'H4': (0,), 'H6': (0,)}
    import h46_long_time_native as native
    native.PUBLIC, native.PRIVATE = PUBLIC, PRIVATE
    native.trial_seed = trial_seed
    return native


def trial_seed(row, cohort, index, outer, piece):
    data = ['H46_B3_20261026', cohort, row['id'], index, outer, piece]
    return int(hashlib.sha256(json.dumps(data).encode()).hexdigest()[:16], 16)


def row(system, T, q, formula, r, k, stage):
    value = dict(system=system, T=T, q=q, delta=T/q, formula=formula,
                 method='PR', family='B3', prefix=0, L_D=0, R_outer=r, K=k,
                 s=q*r, grid_stage=stage)
    value['id'] = signal.cid(value)
    return value


def context(system):
    import numpy as np
    from h46_m5_tol_native import restore
    from h46_new2_core import occupation_one_body, spectrum, matrix_only_tail_coefficients
    cfg = signal.origin_config()['systems'][system]
    for item in cfg['inputs'].values():
        if signal.sha(item['path']) != item['sha256']:
            raise ValueError('SAVED_INPUT_SHA')
    ham, sector, full, state, metadata = restore(dict(system=system, **cfg))
    if metadata['hamiltonian_hash'] != cfg['binding']['hamiltonian'] or metadata['state_hash'] != cfg['binding']['state']:
        raise ValueError('SAVED_INPUT_BINDING')
    if ham.n_blocks != signal.RANK[system]:
        raise ValueError('SAVED_RANK')
    old = read(BASE_PRIVATE/(system+'_context.json'))
    if old['binding'] != cfg['binding']:
        raise ValueError('REFERENCE_BINDING')
    refs = {float(t):complex(v['real'], v['imag']) for t,v in old['reference'].items()}
    one = occupation_one_body(ham.one_body, sector)
    tail = np.zeros_like(one)
    for weight, g in zip(ham.lambdas, ham.g_matrices):
        v = occupation_one_body(g, sector)
        tail += weight*(v@v)
    scalar, lam = matrix_only_tail_coefficients(ham, 0)
    out = SimpleNamespace(system=system, ham=ham, sector=sector, full=full, state=state,
        specs=(spectrum(one),), tails={0:dict(scalar=scalar, lam=lam,
        spec=spectrum(tail-scalar*np.eye(len(state))))}, refs=refs)
    save(PRIVATE/(system+'_context.json'), dict(binding=cfg['binding'],
         reference=old['reference'], reused_reference_times=list(refs), new_reference_calls=0,
         sector_dimension=len(state), prefixes={'0':dict(scalar=scalar, lambda_R=lam)},
         constant=ham.constant, all_two_body_fragments_random=True, one_body_retained=True))
    return out


def signal_job(rows):
    return signal.evolve_batch(CONTEXT[rows[0]['system']], rows)


def evaluate(rows, stage):
    from pf_common_run_contract import resources
    groups = {}; records = read(PUBLIC/'signal_records.json.gz') if (PUBLIC/'signal_records.json.gz').exists() else {}
    for r in rows:
        checkpoint=PRIVATE/'signals'/(r['id']+'.json')
        if r['id'] not in records and checkpoint.exists():
            saved=read(checkpoint)
            if saved['candidate']!=r: raise ValueError('SIGNAL_CHECKPOINT_BINDING')
            records[r['id']]=saved
        if r['id'] not in records:
            groups.setdefault((r['system'],r['T'],r['q'],r['formula']), []).append(r)
    for s in {r['system'] for rs in groups.values() for r in rs}:
        if s not in CONTEXT:
            CONTEXT[s] = context(s)
    obs = resources()
    workers = max(1,min(len(groups) or 1, int(obs['idle_cpu']),obs['available_ram']//(512*2**20)))
    save(PRIVATE/(stage+'_resources.json'),dict(observation=obs,workers=workers,BLAS_threads=1))
    errors=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('fork')) as pool:
        pending = {pool.submit(signal_job, rs):rs for rs in groups.values()}
        for f in as_completed(pending):
            rs=pending[f]
            try:
                for rec in f.result():
                    identity=rec['candidate']['id'];records[identity]=rec
                    save(PRIVATE/'signals'/(identity+'.json'),rec)
            except Exception as exc:
                errors.append(dict(candidates=[r['id'] for r in rs],reason=str(exc)))
            save(PRIVATE/'signal_progress.json',dict(stage=stage,complete=len(records),errors=errors))
    endpoints=[ep for rec in records.values() for ep in signal.qualify(rec,rec['candidate'])]
    save(PUBLIC/'signal_records.json.gz',records);save(PUBLIC/'signal_endpoints.json.gz',endpoints)
    if errors:
        save(PUBLIC/(stage+'_signal_STOP.json'),errors)
        raise ValueError('SIGNAL_TASK_STOP:'+str(errors))
    save(PUBLIC/'signal_summary.json',dict(candidates=len(records),endpoints=len(endpoints),errors=errors,
        new_full_H_references=0,input_regeneration=0,empirical_allowance_not_certificate=True))
    return records,endpoints


def run_signal():
    basic=[row(s,T,q,f,r,k,'BASIC') for s in signal.PREFIX for T,qs in signal.Q.items()
           for q in qs for f in signal.FORMULAS for r in signal.R[f] for k in signal.K]
    records,eps=evaluate(basic,'basic')
    # Extend only q/K slices whose coarse proxy winner reaches R64 or lacks
    # eligibility. Other slices remain explicitly outside the acquired domain.
    from h46_long_time_native import cost_proxy
    extra={};why={}
    for s in signal.PREFIX:
        for T in signal.Q:
            for f in signal.FORMULAS:
                for k in signal.K:
                    for e in signal.EPS:
                        domain=[v for v in eps if v['candidate']['system']==s and v['candidate']['T']==T
                            and v['candidate']['formula']==f and v['candidate']['K']==k and v['epsilon']==e]
                        byq={q:[v for v in domain if v['candidate']['q']==q and v['status']=='ELIGIBLE'] for q in signal.Q[T]}
                        for q,eligible in byq.items():
                            best=min(eligible,key=lambda v:(v['N_cos']+v['N_sin'])*cost_proxy(v['candidate'])) if eligible else None
                            if best is None or best['candidate']['R_outer']==64:
                                key=(s,T,q,f,k);why.setdefault(str(key),set()).add('NO_ELIGIBLE_BASIC' if best is None else 'BASIC_PROXY_WINNER_R64')
                                for r in (128,256,512):
                                    v=row(s,T,q,f,r,k,'R_EXTENSION');extra[v['id']]=v
    save(PUBLIC/'grid_extension.json',dict(reasons={k:sorted(v) for k,v in why.items()},
        candidates=list(extra.values()),R_max=512,q_extension='DECIDE_AFTER_SIGNAL',
        proxy_is_not_compiled_cost_or_lower_bound=True))
    records,eps=evaluate(list(extra.values()),'R_extension')
    # q extension is restricted to formula/system/time slices where the
    # tightest precision has no eligible row, or its proxy best is at q16.
    qextra={};qwhy=[]
    for qmax in (32,64):
        for s in signal.PREFIX:
            for T in signal.Q:
                for f in signal.FORMULAS:
                    domain=[v for v in eps if v['candidate']['system']==s and v['candidate']['T']==T
                        and v['candidate']['formula']==f and v['epsilon']==min(signal.EPS) and v['status']=='ELIGIBLE']
                    best=min(domain,key=lambda v:(v['N_cos']+v['N_sin'])*cost_proxy(v['candidate'])) if domain else None
                    boundary=qmax//2
                    if best is None or best['candidate']['q']==boundary:
                        qwhy.append(dict(system=s,T=T,formula=f,q=qmax,reason='NO_TIGHT_ELIGIBLE' if best is None else 'TIGHT_PROXY_BEST_AT_BOUNDARY'))
                        for r in (*signal.R[f],128,256,512):
                            for k in signal.K:
                                v=row(s,T,qmax,f,r,k,'Q_EXTENSION');qextra[v['id']]=v
        if qextra:
            records,eps=evaluate(list(qextra.values()),'q'+str(qmax))
    ext=read(PUBLIC/'grid_extension.json');ext.update(q_reasons=qwhy,q_candidates=list(qextra.values()))
    save(PUBLIC/'grid_extension.json',ext)
    save(PUBLIC/'candidate_inventory.json.gz',[v['candidate'] for v in records.values()])
    return dict(candidates=len(records),basic=len(basic),R_extension=len(extra),q_extension=len(qextra))


def select():
    native=setup();eps=read(PUBLIC/'signal_endpoints.json.gz');chosen={};reasons={}
    def add(v,reason):
        r=v['candidate'];chosen[r['id']]=r;reasons.setdefault(r['id'],set()).add(reason)
    for s in signal.PREFIX:
        for T in signal.Q:
            for f in signal.FORMULAS:
                for e in signal.EPS:
                    domain=[v for v in eps if v['status']=='ELIGIBLE' and v['candidate']['system']==s
                        and v['candidate']['T']==T and v['candidate']['formula']==f and v['epsilon']==e]
                    score=lambda v:(v['N_cos']+v['N_sin'])*native.cost_proxy(v['candidate'])
                    for v in sorted(domain,key=score)[:2]: add(v,'TOP2_PROXY_'+str(e))
                    # Keep alternatives in event budget, cutoff, and outer q;
                    # low normalization and low single-circuit work both matter.
                    for k in signal.K:
                        values=[v for v in domain if v['candidate']['K']==k]
                        if values: add(min(values,key=score),'BEST_PER_K_'+str(e))
                    if e==min(signal.EPS) and domain:
                        near=[v for v in domain if score(v)<=2*min(map(score,domain))]
                        add(min(near,key=lambda v:v['log_B']),'LOW_NORMALIZATION_WITHIN_2X_PROXY')
                        add(min(domain,key=lambda v:native.cost_proxy(v['candidate'])),'LOWEST_SINGLE_CIRCUIT_PROXY')
                    fixed=[v for v in domain if abs(v['candidate']['delta']-.2)<1e-12]
                    if fixed: add(min(fixed,key=score),'FIXED_DELTA_POINT_'+str(e))
    plan=dict(candidates=list(chosen.values()),reasons={k:sorted(v) for k,v in reasons.items()},
         screening_trajectories_per_new_PR=9,selection='EXPLORATORY_SUBSET_NOT_BOUND',
         nominal_design='Historical coarse n9; independent confirmation fixed separately')
    save(PUBLIC/'native_selection.json',plan)
    return dict(candidates=len(chosen),trajectories=9*len(chosen))


def run_cost(cohort,planfile):
    native=setup()
    from pf_common_run_contract import resources
    from h46_m5_contract import core
    plan=read(PUBLIC/planfile);jobs=[];groups={};known=set();new=[]
    # Include the completed benchmark's recorded cohorts and this study's
    # existing cohorts; exclusion is for scientific sample separation only.
    for folder in (BASE_PRIVATE,PRIVATE):
        for p in folder.glob('*_seeds.json.gz'):
            if p.name!=cohort+'_seeds.json.gz' or folder!=PRIVATE:
                known.update(read(p))
    for s in signal.PREFIX:
        cfg=read(signal.ORIGIN/(s+'_PR_EXPLORATION/config_APPROVED_PRIVATE.json'))
        known.update(read(cfg['excluded_seeds_path'])['values'])
    for r in plan['candidates']:
        n=plan.get('n_by_candidate',{}).get(r['id'],9);groups[r['id']]=[]
        for i in range(n):
            for st in core.schedule(r['formula'],r['q'],'ABS_TIME_PER_OUTER',r['R_outer'],T=r['T']):
                seed=trial_seed(r,cohort,i,st['outer'],st['piece'])
                if seed in known: raise ValueError('SEED_COLLISION')
                known.add(seed);new.append(seed)
            directory=PRIVATE/'cost'/cohort/r['id']/str(i)
            if (directory/'result.json').exists():
                t=read(directory/'result.json')
                if t['status']!='PASS' or t['candidate']!=r or t['cohort']!=cohort or t['index']!=i:
                    raise ValueError('SAVED_TRIAL_BINDING')
                groups[r['id']].append(t)
            elif directory.exists():
                # Scientific STOPs are never retried. Incomplete technical
                # output is retained, and only that unfinished trial restarts.
                if (directory/'STOP.json').exists(): raise ValueError('PRESERVED_TRIAL_STOP:'+str(directory))
                directory.rename(directory.with_name(directory.name+'_incomplete_'+str(time.time_ns())))
                jobs.append((r,cohort,i))
            else: jobs.append((r,cohort,i))
    save(PRIVATE/(cohort+'_seeds.json.gz'),new)
    save(PUBLIC/(cohort.lower()+'_seed_independence.json'),dict(seeds=len(new),collisions=0,
        distinct_namespace=True,historical_exclusion_complete=False))
    for s in signal.PREFIX:
        native.CONTEXT[s]=native.native_context(s,[0])
    obs=resources();workers=max(1,min(len(jobs) or 1,int(obs['idle_cpu']),obs['available_ram']//(2*2**30)))
    save(PRIVATE/(cohort+'_resources.json'),dict(observation=obs,workers=workers,compiler_processes=1))
    progress=dict(total=sum(len(v) for v in groups.values())+len(jobs),complete=sum(map(len,groups.values())),errors=[])
    save(PRIVATE/(cohort+'_progress.json'),progress)
    # Validate candidate's first trajectory before releasing its remaining jobs.
    byid={r['id']:[j for j in jobs if j[0]['id']==r['id']] for r in plan['candidates']}
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('fork')) as pool:
        pending={}
        for identity,js in byid.items():
            if not js: continue
            initial=[j for j in js if j[2]==0]
            for j in initial or js: pending[pool.submit(native.cost_trial,j)]=j
        while pending:
            f=next(as_completed(pending));j=pending.pop(f)
            try:
                groups[j[0]['id']].append(f.result());progress['complete']+=1
                if j[2]==0:
                    for other in byid[j[0]['id']]:
                        if other[2]>0: pending[pool.submit(native.cost_trial,other)]=other
            except Exception as exc:
                progress['errors'].append(dict(candidate=j[0],index=j[2],reason=str(exc)))
            save(PRIVATE/(cohort+'_progress.json'),progress)
    failed={v['candidate']['id'] for v in progress['errors']}
    results={r['id']:native.summarize_trials(r,cohort,groups[r['id']]) for r in plan['candidates'] if r['id'] not in failed}
    save(PUBLIC/(cohort.lower()+'_costs.json.gz'),results)
    save(PUBLIC/(cohort.lower()+'_cost_summary.json'),dict(progress,groups=len(results),all_workers_joined=True))
    if failed: raise ValueError('NATIVE_TASK_STOP')
    return dict(progress,groups=len(results))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['signal','select','cost','confirm'])
    args=parser.parse_args();setup()
    if args.command=='signal': out=run_signal()
    elif args.command=='select': out=select()
    elif args.command=='cost': out=run_cost('B3_EXPLORATION','native_selection.json')
    else: out=run_cost('B3_CONFIRMATION','confirmation_selection.json')
    print(json.dumps(out,ensure_ascii=False))


if __name__=='__main__': main()
