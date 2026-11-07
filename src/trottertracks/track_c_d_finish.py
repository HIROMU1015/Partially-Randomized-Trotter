"""Nonresonant control, finite-step holdouts, and final native-qualified costs."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from .track_c_d_internal import ROOT,OUT,kernel,common,save_json
from . import track_c_d_qualify as qual
from . import track_c_compiler_a_18round as ca
from .track_c_d_resources import r_key,SAMPLES

def spectral(task):
    case,h=task;u=kernel(case,h,1)[0];eig,vec=np.linalg.eig(u);p=abs(vec.conj().T@common()['state'])**2;i=int(np.argmax(p))
    delta=-float(np.angle(eig[i]*np.exp(1j*common()['energy']*h)))/h
    leakage=math.asin(min(1.,(1-p[i])/p[i])) if p[i]>.5 else None
    return dict(case=case,h=h,energy_bias=delta,overlap=float(p[i]),leakage_phase=leakage)

def robust():
    if (OUT/'NONRESONANT.json').exists():raise RuntimeError('Do not overwrite nonresonant check')
    data=json.loads((OUT/'G_RESULTS.json').read_text());policies=[x for x in data['policies'] if not x['name'].startswith(('PR_','Full_'))]
    allrows=[r for p in policies for r in p['selected']];tasks={(json.dumps(r['signal']['case'],sort_keys=True),r['signal']['h']):(r['signal']['case'],r['signal']['h']) for r in allrows};spectra={}
    with ProcessPoolExecutor(max_workers=min(len(tasks),len(os.sched_getaffinity(0)))) as pool:
        for f in as_completed([pool.submit(spectral,t) for t in tasks.values()]):
            s=f.result();spectra[(json.dumps(s['case'],sort_keys=True),s['h'])]=s
    def allowed(record):
        r=record['signal'];s=spectra[(json.dumps(r['case'],sort_keys=True),r['h'])]
        # A declared additional sufficient test on the existing physical model,
        # applied to both arms, rather than silently replacing old raw results.
        return s['leakage_phase'] is not None and abs(s['energy_bias']*r['t'])+s['leakage_phase']<.4
    result=[]
    for family,full in [('PR',False),('Full',True)]:
        chosen=[min((r for r in allrows if r['signal']['m']==m and r['signal']['case']['kind'].startswith('full_')==full and allowed(r)),key=lambda r:r['round_G'][0]) for m in range(18)]
        result.append(dict(name=family+'_nonresonant_RZ',selected=chosen,G=np.sum([r['round_G'] for r in chosen],axis=0).tolist()))
    # h perturbations are independent of the BCH coefficient fit. Physical T is
    # always the same fixed m17 time; only the diagnostic integrator mesh changes.
    holdouts=[];t=.02*(1<<17)
    for name in ('tuned_0123','tuned_0321'):
        base=next(p for p in policies if p['name']==name)['selected'][-1]['signal']
        for factor in (.9,1.,1.1):
            n=math.ceil(t/(base['h']*factor));h=t/n;u,f,lb,_,recipe=kernel(base['case'],h,32)
            z=np.vdot(common()['state'],np.linalg.matrix_power(f,n)@common()['state']);phase=abs(float(np.angle(z*np.exp(1j*common()['energy']*t))))
            holdouts.append(dict(name=name,factor=factor,actual_h=h,steps=n,finite_PF_phase=phase,log_normalizer=n*lb,finite_radius=abs(z),RTE_events=n*sum(r['r'] for r in recipe)))
    save_json(OUT/'NONRESONANT.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),policies=result,spectra=list(spectra.values()),h_holdouts=holdouts,
        interpretation='Separate sufficient dominant-bias-plus-leakage screen. Wrapped endpoint/source guarantees remain preserved. This adds no physical Hamiltonian or RPE time condition; no hidden exclusion of resonant raw results.'))
    print('NONRESONANT_COMPLETE',flush=True)

def close():
    if (OUT/'QUALIFIED_G.json').exists():raise RuntimeError('Do not overwrite qualified G')
    oldq=json.loads((OUT/'QUALIFICATION.json').read_text());local=json.loads((OUT/'LOCAL_INTERVALS.json').read_text());data=json.loads((OUT/'G_RESULTS.json').read_text())
    policies=[p for p in data['policies'] if p['name'] in (*qual.POLICIES,'full_S2','full_Y4')]+json.loads((OUT/'NONRESONANT.json').read_text())['policies']
    def identity(x):return x['name'],x['m'],x['h'],x['rate']
    records={identity(x):x for x in oldq['rows']};new={identity(r['signal']):r for p in policies for r in p['selected'] if identity(r['signal']) not in records}
    cat=qual.compiler_catalogue(new.values());workers=min(len(os.sched_getaffinity(0)),max(1,len(new)));bounds=local['primitive_bounds'].copy();checked_before={r['lowering_sha256'] for r in local['checked']}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        contexts=list(pool.map(ca.context_compile,[(mask,cat) for mask in (11,15)]));unique={r['lowering_sha256']:r for c in contexts for r in c['records'] if r['lowering_sha256'] not in checked_before}
        checked=list(pool.map(ca.interval_check,unique.values()))
        for r in checked:bounds[r['source_key']]=max(bounds.get(r['source_key'],0.),r['operator_error_upper'])
        for f in as_completed([pool.submit(qual.qualify,(r['signal'],bounds)) for r in new.values()]):
            r=f.result();records[identity(r)]=r
    save_json(OUT/'QUALIFICATION_EXTENSION.json',dict(contexts=contexts,checked=checked,rows=[records[k] for k in new],source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()))
    samples={}
    for path in (OUT/'rte_price').glob('*.json'):
        r=json.loads(path.read_text());samples.setdefault(r_key(r['t'],r['n']),{}).setdefault(str(r['rare']),[]).append(r)
    for strata in samples.values():
        for rs in strata.values():rs.sort(key=lambda r:r['j'])
    results=[]
    for p in policies:
        counts=np.zeros(4);reps={str(rare):np.zeros((SAMPLES,4)) for rare in (False,True)};used=[]
        for record in p['selected']:
            x=record['signal'];q=records[identity(x)];assert q['closed'];N=q['shots_per_axis']
            counts+=N*np.sum(record['price_mean'],axis=0)
            for recipe in x['recipe']:
                strata=samples[r_key(recipe['time'],recipe['r'])]
                for rare,rs in strata.items():reps[rare]+=2*N*x['steps']*rs[0]['probability']*np.array([r['metrics'] for r in rs])
            used.append(dict(m=x['m'],name=x['name'],case=x['case'],h=x['h'],rate=x['rate'],shots_per_axis=N,PF_phase_upper=q['PF_phase_upper'],rho_lower=q['rho_lower'],dominant_accumulated_bias_abs=q['dominant_accumulated_bias_abs']))
        covariance=sum((np.cov(r,rowvar=False,ddof=1)/SAMPLES for r in reps.values()),np.zeros((4,4)))
        results.append(dict(name=p['name'],G=counts.tolist(),MC_SE=np.sqrt(np.maximum(0,np.diag(covariance))).tolist(),covariance_MC=covariance.tolist(),reps={k:v.tolist() for k,v in reps.items()},selected=used,total_shots=sum(2*r['shots_per_axis'] for r in used),all18_closed=True))
    original=next(r for r in results if r['name']=='original')
    for r in results:
        r['ratio_to_original']=(np.array(r['G'])/np.array(original['G'])).tolist()
        r['paired_delta_MC_SE']=np.sqrt(sum(np.var(np.array(r['reps'][s])-np.array(original['reps'][s]),axis=0,ddof=1)/SAMPLES for s in ('False','True'))).tolist()
    save_json(OUT/'QUALIFIED_G.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),results=results,all_selected_closed=True,
        success='At least .95 for fixed saved Hermitian-H matrix, exact ground input, ideal native gates, fresh independent bits, delta0 .02 and beta .4. Original continuum-H inclusion remains unqualified.',
        price='Actual modular A counts, MC SE includes shared-cache covariance. Serial depth is an upper bound. No T synthesis. Nonresonant comparison is separately labelled; wrapped endpoint policies preserved.',
        selection='Finite menus and surrogate-selected h/r. Adaptive finite-MC cost selection has not been certified globally optimal or corrected for selection bias.'))
    print('QUALIFIED_G_COMPLETE',len(results),flush=True)

if __name__=='__main__':robust() if sys.argv[1:]==['robust'] else close()
