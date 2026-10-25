"""Authorized q32 S2 extension at the observed q16 boundary; base run unchanged."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing as mp
from types import SimpleNamespace
import h46_long_time_signal as study


def context(system):
    import numpy as np
    from h46_m5_tol_native import restore
    from h46_new2_core import occupation_one_body,spectrum
    info=study.origin_config()['systems'][system]
    ham,sector,full,state,md=restore(dict(system=system,**info))
    matrices=[occupation_one_body(ham.one_body,sector)]
    for weight,g in zip(ham.lambdas,ham.g_matrices):
        v=occupation_one_body(g,sector);matrices.append(weight*(v@v))
    saved=study.read(study.PRIVATE/(system+'_context.json'))
    # References and canonical full-Fock tail coefficients are saved scientific
    # results, reused here without repeating those calculations.
    refs={float(T):complex(z['real'],z['imag']) for T,z in saved['reference'].items()}
    tails={}
    for p in study.PREFIX[system]:
        item=saved['prefixes'][str(p)];scalar=item['scalar']
        tails[p]=dict(scalar=scalar,lam=item['lambda_R'],
            spec=spectrum(sum(matrices[p+1:],np.zeros_like(matrices[0]))-scalar*np.eye(len(state))))
    return SimpleNamespace(system=system,ham=ham,state=state,specs=tuple(spectrum(m) for m in matrices),tails=tails,refs=refs)


def run_signal():
    study.install()
    from pf_common_run_contract import resources
    extra=tuple((s,T) for s in study.PREFIX for T in (1.6,2.))
    rows=[r for r in study.candidates(extra=extra) if r['q']==32 and r['formula']=='2nd']
    groups={}
    for r in rows:
        key=(r['system'],r['T'],r['prefix'],r['method']=='PR')
        groups.setdefault(key,[]).append(r)
    for s in study.PREFIX:
        study.CONTEXT[s]=context(s)
    obs=resources();workers=max(1,min(len(groups),int(obs['idle_cpu']),obs['available_ram']//(512*2**20)))
    study.save(study.PRIVATE/'extension_resources.json',dict(observation=obs,workers=workers))
    records={};endpoints=[];errors=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('fork')) as pool:
        pending={pool.submit(study.signal_job,(key[0],value)):value for key,value in groups.items()}
        for f in as_completed(pending):
            try:
                result=f.result()
                for rec in result['records']:
                    row=rec['candidate'];records[row['id']]=rec;endpoints.extend(study.qualify(rec,row))
            except BaseException as exc:
                errors.append(dict(candidates=[r['id'] for r in pending[f]],reason=str(exc)))
    study.save(study.PUBLIC/'extension_signal_records.json.gz',records)
    study.save(study.PUBLIC/'extension_signal_endpoints.json.gz',endpoints)
    summary=dict(candidates=len(rows),evaluated=len(records),errors=errors,formula='2nd',q=32,
        conditions=[dict(system=s,T=T) for s,T in extra],
        reason='S2 strict-precision eligibility/selected q reached q16; H6 T2 eps=.0005 has no eligible PF in base grid',
        common_base_grid_unchanged=True,other_formulas_q32_not_evaluated=True)
    study.save(study.PUBLIC/'extension_summary.json',summary)
    return summary


def select():
    import h46_long_time_native as native
    ep=study.read(study.PUBLIC/'extension_signal_endpoints.json.gz');chosen={};reasons={}
    for s in study.PREFIX:
        for T in (1.6,2.):
            for e in study.EPS:
                for family in ('PF','PR','FULL'):
                    values=[x for x in ep if x['status']=='ELIGIBLE' and x['epsilon']==e
                        and x['candidate']['system']==s and x['candidate']['T']==T
                        and ((x['candidate']['method']=='PR') if family=='PR' else
                             (x['candidate']['method']=='FULL') if family=='FULL' else (x['candidate']['method']!='PR'))]
                    values.sort(key=lambda x:(x['N_cos']+x['N_sin'])*native.cost_proxy(x['candidate']))
                    for x in values[:2 if family!='FULL' else 1]:
                        r=x['candidate'];chosen[r['id']]=r;reasons.setdefault(r['id'],[]).append(family+'_'+str(e))
    result=dict(candidates=list(chosen.values()),reasons=reasons,screening_trajectories_per_new_PR=9,
                selection='q32 S2 extension only; subset minima, no exhaustive compiled-cost claim')
    study.save(study.PUBLIC/'extension_native_selection.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['signal','select','cost']);args=parser.parse_args()
    if args.command=='signal':
        result=run_signal()
    elif args.command=='select':
        study.install();result=select()
    else:
        import h46_long_time_native as native
        result=native.run_cost('EXTENSION_EXPLORATION',study.read(study.PUBLIC/'extension_native_selection.json'))
    print(json.dumps(result,ensure_ascii=False))
