"""H4/H6 finite-time survey using saved inputs and existing PF/RTE kernels.

This is a new study. Historical SOURCE, grants, runs and STOPs are read-only.
The chat instruction authorizes this scope; no approval-proof mechanism is used.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from decimal import Decimal
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / 'artifacts/resource_applicability/track_a_h46_long_time/2026-10-25'
PRIVATE = Path('/home/AbeHiromu/projects/h46-long-time-pf-pr/20261025')
ORIGIN = Path('/home/AbeHiromu/projects/h46-pr-exploration-execution/20261021/private/authorization_run01')
FORMULAS = ('2nd', '4th', '4th(m5_best)')
Q = {.8: (1, 2, 4, 8, 16), 1.2: (1, 2, 4, 6, 8, 16),
     1.6: (1, 2, 4, 8, 16), 2.: (1, 2, 4, 8, 10, 16)}
PREFIX = {'H4': (1, 2, 3, 5, 7, 9), 'H6': (3, 4, 5, 7, 11, 15)}
RANK = {'H4': 10, 'H6': 19}
R = {'2nd': (4, 8, 16, 32, 64), '4th': (8, 16, 32, 64),
     '4th(m5_best)': (16, 32, 64)}
K = (2, 4, 6)
EPS = (.05, .01, .005, .001, .0005)
CONTEXT = {}


def read(path):
    path = Path(path)
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes())


def save(path, value):
    """Atomic checkpoint, confined to this study's own output."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial.' + str(os.getpid()))
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n').encode()
    temporary.write_bytes(gzip.compress(data, compresslevel=1, mtime=0) if path.suffix == '.gz' else data)
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cid(row):
    fields = (row['system'], row['method'], row['formula'], row['prefix'],
              row['T'], row['q'], row.get('R_outer'), row.get('K'))
    return hashlib.sha256(json.dumps(fields).encode()).hexdigest()[:20]


def candidates(system=None, extra=()):
    rows = []
    for s in ((system,) if system else PREFIX):
        for T, qs in Q.items():
            for q in (*qs, *(32 for x in extra if x == (s, T))):
                for f in FORMULAS:
                    base = dict(system=s, T=T, q=q, delta=T/q, formula=f)
                    rows.append(dict(base, method='FULL', prefix=RANK[s], R_outer=None, K=None))
                    for p in PREFIX[s]:
                        rows.extend(dict(base, method=m, prefix=p, R_outer=None, K=None) for m in ('B0', 'B0I'))
                        rows.extend(dict(base, method='PR', prefix=p, R_outer=r, K=k) for r in R[f] for k in K)
    for row in rows:
        row['id'] = cid(row)
    return rows


def origin_config():
    return read(ORIGIN / 'H4_PR_EXPLORATION/config_APPROVED_PRIVATE.json')


def install():
    cfg = origin_config()
    sys.path.insert(0, str(Path(cfg['provider_root']) / 'src'))
    return cfg


def saved_assets():
    """Read accepted numerical values; never launch their original contracts."""
    signals, endpoints, prices = {}, {}, {}
    for s in PREFIX:
        folder = ROOT / f'artifacts/resource_applicability/track_a_pf_common_{s}_results/2026-10-22'
        for identity, value in read(folder / 'signal_records.json.gz').items():
            tail = identity.split('_')
            row = dict(system=s, method=tail[1], prefix=int(tail[2]), q=int(tail[3]),
                       formula='_'.join(tail[4:]), T=.8, R_outer=None, K=None)
            value = dict(value, origin=str((folder/'signal_records.json.gz').relative_to(ROOT)) + '#' + identity)
            signals[cid(row)] = value
        for ep in read(folder/'PF_endpoint_costs.json.gz'):
            row = dict(ep['candidate'], T=.8, R_outer=None, K=None)
            endpoints[cid(row), ep['epsilon']] = ep
            if ep.get('price'):
                prices[cid(row)] = dict(ep['price'], origin=str((folder/'PF_endpoint_costs.json.gz').relative_to(ROOT)))
    # The complete saved m5/S2/Y4 finite-mean records, including R_outer32/K4.
    source = ROOT/'artifacts/resource_applicability/track_a_h46_m5_signal_run02_results/2026-10-20/signal_records.json'
    for item in read(source)['records']:
        row = dict(item['candidate'], T=.8, R_outer=32 if item['candidate']['method']=='PR' else None,
                   K=4 if item['candidate']['method']=='PR' else None)
        signals.setdefault(cid(row), dict(signal=item['signal'],
            old_u=item.get('empirical_eligibility', {}).get('u', 0.),
            old_margin=item.get('empirical_eligibility', {}).get('log_B_margin', 0.),
            origin=str(source.relative_to(ROOT))+'#'+row['id']))
    prior=ROOT/'artifacts/resource_applicability/track_a_h246_saved_cost_results/2026-10-21/evaluation.json'
    for group in read(prior)['groups']:
        c=group['candidate']
        if c['system'] not in PREFIX or c['method']!='PR':
            continue
        row=dict(c,T=.8,R_outer=32,K=4)
        prices[cid(row)]=dict(kind='SAVED_PR_EXPLORATION',unit_statistics=group['unit_statistics'],
                              strata=group['strata'],origin=str(prior.relative_to(ROOT))+'#'+c['id'])
        for value in group['rows']:
            if value['mode']!='USUAL_SIGNED':
                continue
            shot=value['shot']
            endpoints[cid(row),value['epsilon']]=dict(candidate=row,epsilon=value['epsilon'],
                status=shot['status'],axes=shot['axes'],u=shot['u'],log_B=shot['log_B'],
                log_B_margin=shot['log_B_margin'],N_cos=shot['N_cos'],N_sin=shot['N_sin'],
                empirical_not_certified=True,origin='SAVED_SAME_COMMON_POLICY_PR_ENDPOINT')
    # Fixed-delta H6 signals are scientifically reusable even though their
    # historical dense-Gaussian circuit prices belong to another cost series.
    prior=ROOT/'artifacts/resource_applicability/track_a_h6_fixed_delta_result_v1/2026-10-15/signal_summary_v1.json'
    for item in read(prior):
        if item['T'] not in Q:
            continue
        c=item['cell'];method={'B1':'FULL','B2':'PR'}.get(c['method'],c['method'])
        row=dict(system='H6',T=item['T'],method=method,prefix=c['prefix'],q=c['q'],formula=c['formula'],
                 R_outer=c['r'] if method=='PR' else None,K=c['K'] if method=='PR' else None)
        z=item['signals']['corrected'];ref=item['reference']
        signal=dict(reference=ref,corrected=z,raw=item['signals'].get('raw'),log_B=item['log_B'],
                    total_signed_bias={a:z[a]-ref[a] for a in ('real','imag')},
                    saved_error_decomposition=item['decomposition'],allowance_evidence=None)
        signals.setdefault(cid(row),dict(signal=signal,old_u=max(item['allowance'][a] for a in ('real','imag')),
             old_margin=item['allowance']['log_B_margin'],origin=str(prior.relative_to(ROOT))+'#'+c['id'],
             saved_allowance_only=True,raw_signal_missing=signal['raw'] is None))
    return signals, endpoints, prices


def inventory():
    signals, endpoints, prices = saved_assets()
    rows = candidates()
    records = [dict(r, signal='SAVED' if r['id'] in signals else 'NEW_REQUIRED',
                    saved_shot_endpoints=sum((r['id'], e) in endpoints for e in EPS),
                    compiled_RZ='SAVED' if r['id'] in prices else 'NOT_YET_ACQUIRED') for r in rows]
    cfg = origin_config()
    inputs = {}
    for s, info in cfg['systems'].items():
        inputs[s] = dict(binding=info['binding'], inputs=info['inputs'], reference_root=info['reference_root'])
        for item in info['inputs'].values():
            if sha(item['path']) != item['sha256']:
                raise ValueError('SAVED_INPUT_SHA:' + item['path'])
    save(PRIVATE/'inputs.json', dict(provider=cfg['provider_root'], python=cfg['runtime']['python'], systems=inputs))
    save(PUBLIC/'candidate_inventory.json.gz', records)
    summary = dict(registered=len(rows), saved_signal=sum(r['signal']=='SAVED' for r in records),
                   saved_compiled_RZ=sum(r['compiled_RZ']=='SAVED' for r in records),
                   saved_shot_endpoints=sum(r['saved_shot_endpoints'] for r in records),
                   K0='OPTIONAL_NOT_INCLUDED; main paired cutoffs 2/4/6',
                   scientific_input_regeneration=0, repeated_legacy_runs=0)
    save(PUBLIC/'inventory_summary.json', summary)
    return summary


def context(system):
    import numpy as np
    from h46_m5_tol_native import restore
    from h46_new2_core import occupation_one_body, spectrum, matrix_only_tail_coefficients
    info = origin_config()['systems'][system]
    ham, sector, full, state, metadata = restore(dict(system=system, **info))
    if metadata['hamiltonian_hash'] != info['binding']['hamiltonian'] or metadata['state_hash'] != info['binding']['state']:
        raise ValueError('SAVED_INPUT_IDENTITY')
    # Existing ladder-algebra routine rejects every nonzero transition outside
    # the saved spin sector; this is exact invariant-subspace acceleration.
    matrices = [occupation_one_body(ham.one_body, sector)]
    for weight, g in zip(ham.lambdas, ham.g_matrices):
        v = occupation_one_body(g, sector)
        matrices.append(weight*(v@v))
    specs = tuple(spectrum(m) for m in matrices)
    tails = {}
    for p in PREFIX[system]:
        scalar, lam = matrix_only_tail_coefficients(ham, p)
        tails[p] = dict(scalar=scalar, lam=lam,
            spec=spectrum(sum(matrices[p+1:], np.zeros_like(matrices[0]))-scalar*np.eye(len(state))))
    full_matrix = sum(matrices, np.zeros_like(matrices[0]))
    refspec = spectrum(full_matrix)
    refpath = Path(info['reference_root'])/'reference.npz'
    receipt = read(refpath.parent/'receipt.json')
    if sha(refpath) != receipt['reference_sha256']:
        raise ValueError('SAVED_REFERENCE_SHA')
    particles = int(sector.nelec_alpha + sector.nelec_beta)
    particle_basis = [i for i in range(1 << ham.n_qubits) if i.bit_count()==particles]
    locations = {b:i for i,b in enumerate(particle_basis)}
    selected = [locations[int(b)] for b in sector.basis_indices]
    with np.load(refpath, allow_pickle=False) as z:
        saved_reference = z['reference']
        # Ensure the larger historical number-sector vector has no support in
        # spin sectors omitted by the acceleration.
        other = np.ones(len(particle_basis), bool); other[selected] = False
        if np.linalg.norm(saved_reference[other]) > 1e-10:
            raise ValueError('REFERENCE_SPIN_SECTOR_LEAKAGE')
        refs = {.8: complex(np.vdot(state, saved_reference[selected]))}
    fixed=read(ROOT/'artifacts/resource_applicability/track_a_h6_fixed_delta_result_v1/2026-10-15/signal_summary_v1.json')
    if system=='H6':
        z=next(r['reference'] for r in fixed if r['T']==1.6)
        refs[1.6]=complex(z['real'],z['imag'])
    for T in Q:
        if T in refs:
            continue
        v = refspec[1]@(np.exp(-1j*T*refspec[0])*(refspec[2]@state))
        refs[T] = complex(np.vdot(state, v*np.exp(-1j*T*ham.constant)))
    # One independent full-H check for the new longest time, not a replay of
    # the old T=.8 validation. It checks this study's invariant-sector path.
    from scipy.sparse.linalg import expm_multiply
    T = 2.
    independent = expm_multiply(-1j*T*full_matrix, state)*np.exp(-1j*T*ham.constant)
    error = abs(complex(np.vdot(state, independent))-refs[T])
    if error > 1e-10:
        raise ValueError('FULL_H_REFERENCE_DISAGREEMENT')
    ctx = SimpleNamespace(system=system, ham=ham, sector=sector, full=full, state=state,
                          specs=specs, tails=tails, refs=refs, matrices=matrices)
    save(PRIVATE/(system+'_context.json'), dict(binding=info['binding'], sector_dimension=len(state),
         reference={str(T):complex_record(z) for T,z in refs.items()},
         new_reference_times=[T for T in Q if T not in ((.8,1.6) if system=='H6' else (.8,))],
         reused_reference_times=[.8,1.6] if system=='H6' else [.8],
         reference_expm_multiply_check=error,
         prefixes={str(p):dict(scalar=t['scalar'],lambda_R=t['lam']) for p,t in tails.items()},
         invariant_sector_check='NONZERO_LADDER_TRANSITION_OUTSIDE_SECTOR_REJECTED'))
    return ctx


def complex_record(z):
    return dict(real=float(z.real), imag=float(z.imag))


def evolve_batch(ctx, rows):
    """Batch multiple r/K columns through common deterministic spectral blocks."""
    import numpy as np
    from h46_new2_core import allocation, polynomial, spectral_apply
    from h46_m5_contract import core
    from trotterlib.rte import finite_rte_distribution
    first = rows[0]; p, q, T, f = (first[k] for k in ('prefix','q','T','formula'))
    pr = first['method']=='PR'
    coefficients = core.weights(f)
    # The exact-tail column is shared. Raw and corrected finite means are
    # propagated separately, preserving the existing normalization diagnostic.
    columns = 1+2*len(rows) if pr else 1
    v = np.repeat(ctx.state[:,None], columns, axis=1)
    maximum = np.ones(columns); logs = [0.]*len(rows)
    stages = []
    for piece,c in enumerate(coefficients):
        t = T*c/q
        factors = None
        if pr:
            tail = ctx.tails[p]
            factors = np.empty((len(ctx.state), columns), complex)
            factors[:,0] = np.exp(-1j*t*tail['spec'][0])
            for j,row in enumerate(rows):
                count = allocation(coefficients,row['R_outer'])[piece]
                b = finite_rte_distribution(tail['lam']*(t/count),row['K']).exact_finite_distribution
                value = polynomial(tail['spec'][0],t/count,row['K'])**count
                factors[:,1+j] = value
                factors[:,1+len(rows)+j] = value/(b**count)
                logs[j] += q*count*math.log(b)
        stages.append((t,factors))
    with np.errstate(over='raise',invalid='raise'):
        for _ in range(q):
            for t,factors in stages:
                for spec in ctx.specs[:p+1]:
                    v = spectral_apply(spec,np.exp(-.5j*t*spec[0]),v)
                    maximum = np.maximum(maximum,np.linalg.norm(v,axis=0))
                if pr:
                    spec=ctx.tails[p]['spec'];v=spec[1]@(factors*(spec[2]@v))
                    maximum=np.maximum(maximum,np.linalg.norm(v,axis=0))
                for spec in reversed(ctx.specs[:p+1]):
                    v=spectral_apply(spec,np.exp(-.5j*t*spec[0]),v)
                    maximum=np.maximum(maximum,np.linalg.norm(v,axis=0))
    scalar=ctx.ham.constant+(ctx.tails[p]['scalar'] if pr else 0.)
    v*=np.exp(-1j*T*scalar)
    reference=ctx.refs[T]; records=[]
    if abs(np.linalg.norm(v[:,0])-1)>1e-9:
        raise ValueError('EXACT_PF_NORM')
    exact=complex(np.vdot(ctx.state,v[:,0]))
    for j,row in enumerate(rows):
        if pr:
            corrected=complex(np.vdot(ctx.state,v[:,1+j]));raw=complex(np.vdot(ctx.state,v[:,1+len(rows)+j]))
            closure=float(np.linalg.norm(v[:,1+len(rows)+j]-math.exp(-logs[j])*v[:,1+j]))
            if closure>1e-9:
                raise ValueError('FINITE_MEAN_NORMALIZATION_CLOSURE')
            norm=float(max(maximum[0],maximum[1+j],maximum[1+len(rows)+j]))
            finite=corrected-exact
        else:
            phase=np.exp(-1j*T*ctx.tails[p]['scalar']) if row['method']=='B0I' else 1.
            corrected=exact*phase;raw=corrected;closure=0.;norm=float(maximum[0]);finite=0j
        signal=dict(reference=complex_record(reference),corrected=complex_record(corrected),raw=complex_record(raw),
                    exact_tail=complex_record(exact if pr else corrected),
                    total_signed_bias=complex_record(corrected-reference),
                    outer_PF_or_discard_bias=complex_record((exact if pr else corrected)-reference),
                    finite_RTE_bias=complex_record(finite),log_B=logs[j] if pr else 0.,
                    allowance_evidence=dict(maximum_intermediate_norm=norm,raw_corrected_vector_closure=closure))
        records.append(dict(candidate=row,signal=signal,old_u=0.,old_margin=0.,
                            origin='NEW_SAVED_INPUT_FINITE_MEAN',phase_alignment=False))
    return records


def qualify(record, row):
    if record.get('saved_allowance_only'):
        from d0_frozen_shot_rule_v1 import u_aware_shots
        results=[];u=max(1e-9,record['old_u']);margin=max(1e-12,record['old_margin'])
        for e in EPS:
            sig=record['signal']
            value=u_aware_shots({a:abs(sig['total_signed_bias'][a]) for a in ('real','imag')},
                dict(real=u,imag=u),epsilon=e,log_B_upper=sig['log_B']+margin,
                evidence_kind='EMPIRICAL',evidence_ref=record['origin'])
            for axis in value['axes'].values():
                if axis['status']=='ELIGIBLE' and (axis['shots'] is None or axis['u_over_headroom']>.01):
                    axis.update(status='UNDETERMINED',shots=None)
            statuses={a['status'] for a in value['axes'].values()}
            status='INELIGIBLE' if 'INELIGIBLE' in statuses else 'ELIGIBLE' if statuses=={'ELIGIBLE'} else 'UNDETERMINED'
            results.append(dict(candidate=row,epsilon=e,status=status,axes=value['axes'],u=u,
                log_B=sig['log_B'],log_B_margin=margin,empirical_not_certified=True,
                N_cos=value['axes']['real']['shots'] if status=='ELIGIBLE' else None,
                N_sin=value['axes']['imag']['shots'] if status=='ELIGIBLE' else None,
                origin='SAVED_SIGNAL_AND_ALLOWANCE_WITH_COMMON_FLOOR_NO_REEVOLUTION'))
        return results
    from run_pf_common_run import qualify_record
    results=[]
    for e in EPS:
        ep=qualify_record(record,row,e)
        ep['candidate']=row
        results.append(ep)
    return results


def signal_job(job):
    system,rows=job
    started=time.monotonic();records=evolve_batch(CONTEXT[system],rows)
    return dict(records=records,wall_seconds=time.monotonic()-started)


def run_signal():
    install()
    import numpy as np
    from pf_common_run_contract import resources
    inventory()
    old_signals,old_endpoints,old_prices=saved_assets()
    rows=candidates();groups={};records={};endpoints=[]
    for row in rows:
        if row['id'] in old_signals:
            rec=dict(old_signals[row['id']],candidate=row)
            records[row['id']]=rec
            missing_eps=[e for e in EPS if (row['id'],e) not in old_endpoints]
            new_eps={x['epsilon']:x for x in qualify(rec,row)} if missing_eps else {}
            for e in EPS:
                old=old_endpoints.get((row['id'],e))
                if old:
                    endpoints.append(dict(old,candidate=row,origin='SAVED_ENDPOINT_NO_RECOMPUTATION'))
                else:
                    endpoints.append(new_eps[e])
            continue
        # B0/B0I share one base propagation, scalar applied to final signal.
        key=(row['system'],row['T'],row['q'],row['formula'],row['prefix'],row['method']=='PR')
        groups.setdefault(key,[]).append(row)
    # Two systems prepare once in the primary process, then fork immutable
    # spectra. The numerical work, not AI-agent delegation, is parallelized.
    for s in PREFIX:
        CONTEXT[s]=context(s)
    compatibility=[]
    for s in PREFIX:
        anchor=3 if s=='H4' else 5
        selected=[next(r for r in rows if r['system']==s and r['T']==.8 and r['q']==1
                    and r['method']=='PR' and r['prefix']==anchor and r['formula']=='4th(m5_best)'
                    and r['R_outer']==32 and r['K']==4),
                  next(r for r in rows if r['system']==s and r['T']==.8 and r['q']==1
                    and r['method']=='FULL' and r['formula']=='2nd')]
        for row in selected:
            actual=evolve_batch(CONTEXT[s],[row])[0]['signal'];expected=old_signals[row['id']]['signal']
            err=max(abs(actual['corrected'][a]-expected['corrected'][a]) for a in ('real','imag'))
            if err>1e-10:
                raise ValueError('NEW_ADAPTER_OLD_REPRESENTATIVE_DISAGREEMENT:'+str(err))
            compatibility.append(dict(candidate=row,error=err,
                reason='NEW_TIME_VARIABLE_INVARIANT_SECTOR_ADAPTER_CHECK; historical scientific run not replayed'))
    save(PUBLIC/'new_adapter_compatibility.json',compatibility)
    obs=resources();worker_count=max(1,min(len(groups),int(obs['idle_cpu']),obs['available_ram']//(512*2**20)))
    save(PRIVATE/'signal_resources.json',dict(observation=obs,workers=worker_count,BLAS_threads_per_worker=1))
    save(PRIVATE/'saved_prices.json',old_prices)
    # Largest jobs first; results remain per-candidate, so completed work is
    # retained if a genuinely inconsistent candidate must stop.
    jobs=[(key[0],value) for key,value in groups.items()]
    jobs.sort(key=lambda j:len(CONTEXT[j[0]].state)**2*j[1][0]['q']*len(j[1])*
              (11 if 'm5' in j[1][0]['formula'] else 3 if j[1][0]['formula']=='4th' else 1),reverse=True)
    progress=dict(total_groups=len(jobs),completed_groups=0,reused_signals=len(records),new_signals=0,errors=[])
    save(PRIVATE/'signal_progress.json',progress)
    with ProcessPoolExecutor(max_workers=worker_count,mp_context=mp.get_context('fork')) as pool:
        futures={pool.submit(signal_job,j):j for j in jobs}
        for future in as_completed(futures):
            job=futures[future]
            try:
                result=future.result()
                for rec in result['records']:
                    row=rec['candidate'];identity=row['id'];records[identity]=rec
                    save(PRIVATE/'signals'/(identity+'.json'),rec)
                    endpoints.extend(qualify(rec,row))
                progress['completed_groups']+=1;progress['new_signals']+=len(result['records'])
            except BaseException as exc:
                progress['errors'].append(dict(rows=[r['id'] for r in job[1]],reason=str(exc),traceback=traceback.format_exc()))
                save(PRIVATE/'signal_errors'/('group_'+job[1][0]['id']+'.json'),progress['errors'][-1])
            save(PRIVATE/'signal_progress.json',progress)
    save(PUBLIC/'signal_records.json.gz',records)
    save(PUBLIC/'signal_endpoints.json.gz',endpoints)
    summary=dict(progress,registered_candidates=len(rows),evaluated_candidates=len(records),endpoint_count=len(endpoints),
                 empirical_allowance_not_certificate=True,independent_scientific_review=False)
    save(PUBLIC/'signal_summary.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['inventory','signal'])
    a=parser.parse_args()
    print(json.dumps(inventory() if a.command=='inventory' else run_signal(),ensure_ascii=False))


if __name__=='__main__':
    main()
