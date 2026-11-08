"""Paired directed signal disks for cancellation-independent P2 qualification.

P1 must finish first. Candidate domains and shortlist rules are saved before
new P2 signal evaluation. All fixed-study inputs are read-only dependencies.
"""
from fractions import Fraction as F
from functools import lru_cache
from pathlib import Path
import copy
import hashlib
import json
import math
import pickle
import numpy as np
import mpmath as mp
from scipy.linalg import eigh
from . import track_c_df_cancellation_validation as v

d=v.d;s=v.s;OUT=v.OUT;REF=v.REF;SAVE=v.SAVE;SOURCE=v.SOURCE


def directed_phase(z,error):
    if error>=s.lower(abs(s.C(z.real,z.imag))):return None
    return s.phase_upper(z,error)


def pair_phase(zp,ep,zk,ek):
    """Enclose zp*conj(zk) without assuming cancellation of its errors."""
    mp.iv.prec=192;a=s.C(zp.real,zp.imag);b=s.C(zk.real,-zk.imag);prod=a*b;c=s.centre(prod)
    rounding=s.up(abs(prod-s.C(c.real,c.imag)))
    radius=s.add(s.up(abs(a)*s.I(ek)),s.up(abs(b)*s.I(ep)),s.mul(ep,ek),rounding)
    return dict(centre=[c.real,c.imag],radius=radius,radius_lower=s.lower(abs(s.C(c.real,c.imag))-s.I(radius)),phase_upper=directed_phase(c,radius))


def key(n,row,m):return (n,row['candidate']['name'],row['formula'],row['h'],m)


@lru_cache(None)
def original_wrapper(n,m):
    point=v.original_policy(n)['selected'][m];r=point['row'];g=r['groups'];steps=point['signal']['steps']
    D=np.array(g['first'])+np.array(g['last'])+(steps-1)*np.array(g['middle'])
    return np.array(point['mean'])-D


def prices(n,row,m):
    steps=round(.02*2**m/row['h']);g=row['groups'];D=np.array(g['first'])+np.array(g['last'])+(steps-1)*np.array(g['middle'])
    return D+original_wrapper(n,m)


def pool_plan():
    assert v.read_json(OUT/'P1/FINAL.json')['P2_may_proceed']
    pool={};saved={}
    for n in(4,6):
        v.prepare_model(n)
        for path in d.scan_paths(n):
            rel=str(path.relative_to(REF));r=v.read_reference(rel)
            if r['candidate']['family']!='Truncated':continue
            for m in r['viable_rounds']:pool[key(n,r,m)]=dict(n=n,row=r,m=m,reference_row_path=rel)
        for path in sorted((REF/f'H{n}/qualified_repriced').glob('*.json')):
            x=v.read_reference(str(path.relative_to(REF)));r=x['row'];m=x['m']
            if r['candidate']['family']=='Truncated':
                k=key(n,r,m);pool[k]=dict(n=n,row=r,m=m,reference_row_path=str(path.relative_to(REF)));saved[k]=dict(q=x['qualification'],supplement_needed=True)
    for x in v.read_reference('ordering/QUALIFIED_COMPLETE.json')['points']:
        if x['n']in(4,6)and x['family']=='Truncated':
            r=x['row'];m=x['signal']['m'];k=key(x['n'],r,m);pool[k]=dict(n=x['n'],row=r,m=m,reference_row_path='ordering/QUALIFIED_COMPLETE.json');saved[k]=dict(q=x['qualification'],supplement_needed=False)
    chosen=set(saved)
    # Four B and two A screen points at every round. These numerical surrogates
    # select a finite cohort only; all final B bounds are independent disks.
    screens={}
    for k,item in pool.items():
        n,r,m=item['n'],item['row'],item['m'];p=v.read_reference(f'H{n}/pilot/{r["candidate"]["name"]}.json');t=.02*2**m
        leakT=math.asin(min(1.,(1-p['dominant_overlap'])/p['dominant_overlap']))
        B=abs(p['dominant_energy_bias'])*t+abs(r['dominant_energy_bias']-p['dominant_energy_bias'])*t+2*leakT+r['nonresonant_leak_bound']
        A=abs(r['dominant_energy_bias'])*t+r['nonresonant_leak_bound'];rho=max(1e-8,2*r['dominant_overlap']-1);mean=prices(n,r,m)
        scores={}
        for method,bias in [('A',A),('B',B)]:
            beta=.4-bias-.005
            scores[method]=math.inf if beta<=0 else math.ceil(4*math.log(1440)/(rho*rho*math.sin(beta)**2))*float(mean[:,0].sum())
            screens.setdefault((n,m,method),[]).append((scores[method],k))
        item['screen']=dict(combined_phase_proxy=A,conservative_phase_proxy=B,scores={k:(z if math.isfinite(z)else None)for k,z in scores.items()})
    for (n,m,method),rs in screens.items():
        for score,k in sorted(rs)[:4 if method=='B' else 2]:
            if math.isfinite(score):chosen.add(k)
    records=[]
    for k in sorted(chosen):
        item=pool[k];item['saved_combined']=saved.get(k);item['mean']=prices(item['n'],item['row'],item['m']).tolist();item['id']=len(records);records.append(item)
    SAVE(OUT/'P2/PLAN.json',dict(source=SOURCE,fixed_reference=v.BASE,pool_points=len(pool),cohort_points=len(records),cohort=records,rule='all saved Truncated qualifications plus top4 conservative-proxy and top2 combined-proxy per molecule/round from existing viable scans; no P1 new h, no new retention/order/formula',selection='proxy only; directed individual disks required; freeze before independent final checks',evaluation='B: upper(abs phase Truncation)+upper(abs phase PF/native), not abs of signed point estimates; actual circuit unchanged',scope='fixed H4/H6, same original H/state/precision/failure/18 rounds/compiler; finite shortlist, not global optimum'))
    print('P2_PLAN',len(pool),len(records),flush=True)


def keep_cache_task(task):
    n,candidate=task;c=d.model(n);bb=d.kept(c,candidate);size=len(c['H'])
    # Reference is the sum of the numerical Hermitian generators already used
    # by the PF qualifier. Enclose the additional summation/averaging error.
    H=sum((s.hermitian(b['matrix'])for b in bb),c['constant']*np.eye(size));H=s.hermitian(H)
    rounding=s.up(s.gamma(2*len(bb)+12)*(sum((s.I(d.fast_fro(b['matrix']))for b in bb),s.I(0))+abs(s.I(c['constant']))*mp.iv.sqrt(s.I(size))))
    e,V=eigh(H);cert=s.spectral_certificate(H,e,V);path=OUT/f'P2/keep/H{n}_{candidate["name"]}'
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(Path(str(path)+'.npz'),H=H,e=e,V=V)
    SAVE(Path(str(path)+'.json'),dict(source=SOURCE,n=n,candidate=candidate,formation_rounding_upper=rounding,spectral_certificate=cert,scope='numerical Hkeep and eigenbasis; source generator transfer is separately charged from the matching PF certificate'))
    return n,candidate['name']


def prepare_keep():
    plan=v.read_json(OUT/'P2/PLAN.json');tasks={}
    for item in plan['cohort']:tasks[(item['n'],item['row']['candidate']['name'])]=(item['n'],item['row']['candidate'])
    v.parallel(keep_cache_task,list(tasks.values()))


@lru_cache(None)
def keeper(n,name):
    path=OUT/f'P2/keep/H{n}_{name}'
    with np.load(Path(str(path)+'.npz'))as z:arrays={k:z[k]for k in z.files}
    return arrays,v.read_json(Path(str(path)+'.json'))


def keep_signal(c,row,m,q):
    t=.02*2**m;steps=round(t/row['h']);bb=d.kept(c,row['candidate']);times={b['index']:F(0)for b in bb}
    for _,i,tt in d.schedule(row['case'],row['h']):times[i]+=abs(F.from_float(tt))*steps
    # For S2/Y4, every block's absolute PF time dominates the positive Hkeep
    # time. Thus the saved SOURCE generator disk also bounds Hkeep evolution.
    assert all(z>=F.from_float(t)for z in times.values())
    arrays,meta=keeper(c['n'],row['candidate']['name']);u,eu=s.spectral_unitary(arrays['H'],arrays['e'],arrays['V'],t,meta['spectral_certificate'])
    psi=c['state'];normdef,normupper=d.state_norm_bounds(psi);y,ey=s.product(u,psi[:,None]);zz,ez=s.product(psi.conj()[None,:],y);z=complex(zz[0,0])
    error=s.add(s.mul(eu,normupper,normupper),s.mul(ey,normupper),ez,s.mul(2,q['original_state_distance_upper'],max(1.,normupper)),normdef)
    source=s.add(q['source_generator_disk'],q.get('complete_number_sector_diagonal_supplement',0),s.mul(t,meta['formation_rounding_upper']))
    time=s.up(abs(s.I(t)-s.I('.02')*(1<<m))*(s.I(d.fast_fro(arrays['H']))+s.I(d.fast_fro(c['H']))))
    error=s.add(error,source,time)
    return z,error,dict(numerical_evolution_disk=eu,source_and_formation_disk=source,time_disk=time,positive_time_dominated_by_saved_PF_absolute_times=True,minimum_PF_absolute_block_time_exact=str(min(times.values())))


def qualify_point(item):
    n,r,m=item['n'],item['row'],item['m'];c=d.model(n);saved=item['saved_combined']
    if saved is None:q=d.qualification_supplement(c,r,m,d.det_qualification(c,r,m));origin='new directed combined point'
    elif saved['supplement_needed']:q=d.qualification_supplement(c,r,m,saved['q']);origin='saved combined + required original supplement, no matrix reevaluation'
    else:q=saved['q'];origin='saved complete combined point'
    zk,ek,kmeta=keep_signal(c,r,m,q);zp=complex(*q['signal'])
    ep=s.add(q['numerical_disk'],q['coordinate_disk'],q.get('complete_number_sector_diagonal_supplement',0))
    rot=s.C(zk.real,zk.imag)*s.expi(s.I(.02*2**m)*s.I(c['energy']));zc=s.centre(rot);erot=s.up(abs(rot-s.C(zc.real,zc.imag)))
    trunc=directed_phase(zc,s.add(ek,erot,s.mul(.02*2**m,q['ground_energy_error_upper'])))
    pair=pair_phase(zp,ep,zk,ek);pf=pair['phase_upper'];upper=s.add(trunc,pf) if trunc is not None and pf is not None else None
    beta=s.lower(s.I('.4')-s.I(upper))if upper is not None else None
    bq={**q,'closed':bool(q['alias'] and q['rho_lower']>0 and beta is not None and beta>0),'beta_stat_lower':beta,'conservative_phase_upper':upper,'truncation_phase_upper':trunc,'PF_phase_upper':pf,'method':'sum of two directed phase-disk upper bounds; actual PF circuit unchanged'}
    result=dict(id=item['id'],n=n,key=[r['candidate']['name'],r['formula'],r['h'],m],row=r,signal=dict(m=m,h=r['h'],steps=round(.02*2**m/r['h']),name=r['formula']),mean=item['mean'],qualification=bq,combined_qualification=q,combined_origin=origin,
        individual_disks=dict(keep_signal=[zk.real,zk.imag],keep_radius=ek,PF_signal_radius=ep,PF_pair=pair,keep_details=kmeta),observed=dict(truncation_phase=float(np.angle(zk*np.exp(1j*c['energy']*.02*2**m))),PF_phase=float(np.angle(zp*np.conj(zk))),combined_phase=float(np.angle(zp*np.exp(1j*c['energy']*.02*2**m)))),source=SOURCE)
    SAVE(OUT/f'P2/qualified/{item["id"]:04d}.json',result);return item['id'],n,bq['closed']


def qualify():v.parallel(qualify_point,v.read_json(OUT/'P2/PLAN.json')['cohort'])


def freeze():
    plan=v.read_json(OUT/'P2/PLAN.json');points=[v.read_json(p)for p in sorted((OUT/'P2/qualified').glob('*.json'))];assert len(points)==plan['cohort_points'];policies=[]
    for n in(4,6):
        for method in('A_combined','B_conservative'):
            ps=[{**p,'qualification':p['combined_qualification'] if method=='A_combined'else p['qualification'],'mean':np.array(p['mean'])}for p in points if p['n']==n]
            best=d.choose(ps)
            if best is None:policies.append(dict(n=n,method=method,status='NO_QUALIFIED_18ROUND_POLICY'));continue
            cost,selected,alpha,ns=best;policies.append(dict(n=n,method=method,status='FROZEN',selected=[{**p,'mean':p['mean'].tolist()}for p in selected],alpha=alpha.tolist(),trials=ns.tolist(),trials_total=int(ns.sum()),totals=np.sum([p['mean']*nn[:,None]for p,nn in zip(selected,ns)],axis=(0,1)).tolist(),MC_SE=[0.]*4))
    # A on B's *fixed circuits*: the evaluation rule alone changes, with an
    # independently sufficient integer allocation and no circuit reselection.
    for p in list(policies):
        if p['method']=='B_conservative' and p['status']=='FROZEN':
            ps=[{**x,'qualification':x['combined_qualification'],'mean':np.array(x['mean'])}for x in p['selected']];alpha,ns=d.discrete_allocate(ps)
            policies.append(dict(n=p['n'],method='A_on_B_fixed_circuits',status='FROZEN',selected=[{**x,'mean':x['mean'].tolist()}for x in ps],alpha=alpha.tolist(),trials=ns.tolist(),trials_total=int(ns.sum()),totals=np.sum([x['mean']*nn[:,None]for x,nn in zip(ps,ns)],axis=(0,1)).tolist(),MC_SE=[0.]*4))
    SAVE(OUT/'P2/POLICY.json',dict(source=SOURCE,policies=policies,selection='directed qualified cohort; deterministic actual saved native prices; alpha/point selection before independent checks; no new price MC'))
    print('P2_FROZEN',[(p['n'],p['method'],p.get('totals',[None])[0])for p in policies],flush=True)


def independent():
    policy=v.read_json(OUT/'P2/POLICY.json');tasks={};reused=[]
    for pol in policy['policies']:
        if pol['status']!='FROZEN':continue
        for point in pol['selected']:
            n=pol['n'];r=point['row'];path=f'H{n}/independent/{r["candidate"]["name"]}_{r["formula"]}_{r["h"]}_{point["signal"]["m"]}.json'
            if path in v.reference_index():
                check=v.read_reference(path);assert check['key']==point['key'] and check['price_recompile']==[r['groups']['first'],r['groups']['last']];reused.append(check)
            else:tasks[(n,tuple(point['key']))]=(n,{**point,'qualification':point['combined_qualification']})
    new=v.parallel(v.independent,list(tasks.values()));SAVE(OUT/'P2/INDEPENDENT.json',dict(source=SOURCE,reused=reused,new=new,policy_sha256=hashlib.sha256((OUT/'P2/POLICY.json').read_bytes()).hexdigest()))


def finish():
    path=OUT/'P2/POLICY.json';digest=hashlib.sha256(path.read_bytes()).hexdigest();policy=v.read_json(path);checks=v.read_json(OUT/'P2/INDEPENDENT.json');assert checks['policy_sha256']==digest
    lookup={(x['n'],tuple(x['key'])):x for x in checks['reused']+checks['new']}
    for pol in policy['policies']:
        if pol['status']!='FROZEN':continue
        phases=[float(np.angle(complex(*lookup[(pol['n'],tuple(p['key']))]['independent_signal'])))for p in pol['selected']]
        decoded=d.reconstruct_rpe_phase(phases,[1<<m for m in range(18)]);energy=-decoded['final_phase_estimate']/.02;error=abs(energy-d.model(pol['n'])['energy']);assert error<1.5936e-4
        assert sum(F.from_float(a)for row in pol['alpha']for a in row)<=F(1,20)
        for point,aa,nn in zip(pol['selected'],pol['alpha'],pol['trials']):
            assert point['qualification']['closed']
            for a,N in zip(aa,nn):assert d.rigorous_shots(point['qualification']['rho_lower'],point['qualification']['beta_stat_lower'],a)<=N
        pol.update(status='QUALIFIED_INDEPENDENT_COMPLETE',mean_decoder=dict(energy=energy,error_validation_only=error,no_E0_argument=True))
    bases=[x for x in v.read_reference('ordering/RESULTS.json')['methods']if x['n']in(4,6) and x['method']in('Full','PR')];comparisons=[]
    for n in(4,6):
        B=next(p for p in policy['policies']if p['n']==n and p['method']=='B_conservative');bench=[p for p in bases if p['n']==n]
        outcome='A'if B['status']=='QUALIFIED_INDEPENDENT_COMPLETE'and all(B['totals'][0]<x['RZ']for x in bench)else'B'
        comparisons.append(dict(n=n,outcome=outcome,meaning='finite qualified cohort and chosen conservative bound only; not a no-go theorem',baselines=bench,conservative_total_RZ=B.get('totals',[None])[0]))
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    SAVE(OUT/'P2/FINAL.json',dict(source=SOURCE,policy_sha256=digest,policies=policy['policies'],comparisons=comparisons,independent_new=len(checks['new']),independent_reused=len(checks['reused']),price_MC_new=0,phase_budget=.4,energy_tolerance=1.5936e-4,failure_upper_per_method=.05,scope='fixed saved original H and exact original ground; ideal native; independent measurement outcomes; state preparation excluded'))
    print('P2_FINAL',comparisons,flush=True)


if __name__=='__main__':
    import sys
    if (OUT/'STUDY_STOP.json').exists():raise RuntimeError('Study has stopped')
    {'plan':pool_plan,'keep':prepare_keep,'qualify':qualify,'freeze':freeze,'independent':independent,'finish':finish}[sys.argv[1]]()
