"""Fixed-input P1/P2 adapters; old studies and their STOPs are read-only."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction as F
from functools import lru_cache
from pathlib import Path
import copy
import hashlib
import json
import math
import os
import pickle
import subprocess
import numpy as np
import mpmath as mp
from scipy.linalg import eigh
from . import track_c_df_truncation as d

ROOT=d.ROOT
OUT=Path(os.environ.get('TRACK_C_CANCELLATION_OUT',str(ROOT/'artifacts/track_c_df_cancellation_validation/2026-11-08')))
REF=Path(os.environ.get('TRACK_C_CANCELLATION_REFERENCE','/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/track-c-df-truncated-pf-20261108/artifacts/track_c_df_truncation/2026-11-08'))
CACHE=ROOT/'.cache/df_cancellation_validation'
SAVE=d.save_json
d.OUT=REF
d.CACHE=CACHE
d.sg.fro=d.fast_fro
s=d.sg
SOURCE=d.source()
BASE='da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13'


def redirected_save(path,value):
    """The one reused independent checker writes only into this new study."""
    SAVE(OUT/'checks'/Path(path).relative_to(REF),value)
d.save_json=redirected_save


@lru_cache(None)
def reference_index():return {x['path']:x for x in json.loads((REF/'RAW_INDEX.json').read_text())['entries']}


@lru_cache(None)
def read_reference(rel):
    raw=(REF/rel).read_bytes();index=reference_index()
    if rel in index:assert hashlib.sha256(raw).hexdigest()==index[rel]['sha256'],rel
    return json.loads(raw)


def rational(v):return s.I(v.numerator)/v.denominator


def parallel(function,tasks):
    if not tasks:return []
    results=[]
    with ProcessPoolExecutor(max_workers=d.workers(len(tasks))) as pool:
        for f in as_completed(pool.submit(function,t) for t in tasks):
            result=f.result();results.append(result);print(function.__name__,len(results),len(tasks),flush=True)
    return results


def prepare_model(n):
    CACHE.mkdir(parents=True,exist_ok=True);path=CACHE/f'H{n}.pkl'
    if path.exists():return
    for name in ('INPUT.json','INPUT_ARRAYS.npz'):
        rel=f'H{n}/{name}';assert hashlib.sha256((REF/rel).read_bytes()).hexdigest()==reference_index()[rel]['sha256']
    old=d.ROOT
    try:
        d.ROOT=REF.parents[2];c=d.replay_model(n)
    finally:d.ROOT=old
    with path.open('xb') as f:pickle.dump(c,f)


def original_policy(n):return copy.deepcopy(next(p for p in read_reference('ordering/FINAL.json')['policies'] if p['n']==n and p['family']=='Truncated'))


def native_row(n,old,h,label):
    c=d.model(n);row=copy.deepcopy(old);row.update(h=h,formula=label,source=SOURCE,groups={},angle_errors={},variants={})
    seq=d.schedule(row['case'],h);blocks=d.kept(c,row['candidate'])
    pieces=dict(first=seq[:1],last=seq[1:],middle=seq[1:-1]+[(seq[0][0],seq[0][1],2*seq[0][2])])
    for k,part in pieces.items():row['groups'][k],row['angle_errors'][k],row['variants'][k],_=d.price_piece(c,blocks,part)
    return row


def scalar_time_transfer(q,n,h,t,constant):
    """Kernel uses n*h scalar, implemented wrapper uses the fixed total t.

    The original qualifier already charges the target decimal/binary time and
    the wrapper's binary64 angle. Add the new n*h versus t discrepancy. A disk
    transfer plus an extra absolute phase charge is conservative (not tight).
    """
    mp.iv.prec=192;delta=F(n)*F.from_float(h)-F.from_float(t)
    error=s.up(abs(rational(delta)*s.I(constant)));disk=s.up(2*mp.iv.sqrt(s.I(2))*s.I(error))
    rho=s.lower(s.I(q['rho_lower'])-s.I(disk));beta=None
    if q['beta_stat_lower'] is not None and disk<q['rho_lower']:
        ratio=s.I(disk)/s.I(q['rho_lower']);pe=mp.iv.atan2(ratio,mp.iv.sqrt(1-ratio**2))
        beta=s.lower(s.I(q['beta_stat_lower'])-pe-s.I(error))
    return {**q,'closed':bool(q['closed'] and rho>0 and beta is not None and beta>0),
        'rho_lower':rho,'beta_stat_lower':beta,'reference_combined_phase_upper':q['total_truncation_PF_phase_upper'],
        'total_truncation_PF_phase_upper':s.add(q['total_truncation_PF_phase_upper'],error),
        'fixed_total_scalar_time_error_upper':error,'fixed_total_scalar_coordinate_disk':disk,
        'n_h_minus_fixed_t_exact':str(delta),'time_definition':'t=(1/50)*2^m; h=round_binary64(t/n); fixed-total scalar wrapper'}


def physical_signal(z,n,h,t,constant):
    delta=(F(n)*F.from_float(h)-F.from_float(t))*F.from_float(constant)
    phase=s.centre(s.expi(rational(delta)));return complex(*z)*phase


def h_rounding_supplement(q,c,row,n,m):
    """Also enclose the ideal rational h=t/n versus binary64 D times.

    The direct fixed-target certificate already tests the implemented operator.
    This additional disk makes the logical time definition explicit, using
    exact Pauli L1 and unitary Gaussian conjugation on the full number sector.
    """
    drop={(i,tuple(support))for i,support in row['candidate']['drop']}
    mass=sum((abs(value)for i,b in enumerate(c['blocks'])for support,value in b['coeff'].items()if(i,support)not in drop),F(0))
    delta=F(n)*F.from_float(row['h'])-F(1,50)*(1<<m);error=s.up(abs(rational(delta))*rational(mass));disk=s.up(2*mp.iv.sqrt(s.I(2))*s.I(error))
    rho=s.lower(s.I(q['rho_lower'])-s.I(disk));beta=None
    if q['beta_stat_lower']is not None and disk<q['rho_lower']:
        ratio=s.I(disk)/s.I(q['rho_lower']);angle=mp.iv.atan2(ratio,mp.iv.sqrt(1-ratio**2));beta=s.lower(s.I(q['beta_stat_lower'])-angle-s.I(error))
    return {**q,'closed':bool(q['closed']and rho>0 and beta is not None and beta>0),'rho_lower':rho,'beta_stat_lower':beta,'total_truncation_PF_phase_upper':s.add(q['total_truncation_PF_phase_upper'],error),'logical_h_rounding_source_disk':error,'logical_h_rounding_coordinate_disk':disk,'n_h_minus_decimal_target_exact':str(delta),'retained_Pauli_L1_exact':str(mass)}


def p1_point(nsteps):
    m=17;c=d.model(6);old=original_policy(6)['selected'][m];t=F(1,50)*(1<<m);h=float(t/nsteps)
    assert round(float(t)/h)==nsteps
    row=native_row(6,old['row'],h,f'S2_P1_n{nsteps}')
    q=d.qualification_supplement(c,row,m,d.det_qualification(c,row,m));q=scalar_time_transfer(q,nsteps,h,float(t),c['constant'])
    g=row['groups'];v=np.array(g['first'])+np.array(g['last'])+(nsteps-1)*np.array(g['middle'])
    mean=np.array([v+d.wrapper(c,float(t),a)for a in('X','Y')],float)
    point=dict(key=[row['candidate']['name'],row['formula'],h,m],row=row,signal=dict(m=m,h=h,steps=nsteps,name=row['formula']),qualification=q,mean=mean.tolist(),n=6,source=SOURCE,h_exact=str(t/nsteps))
    z=physical_signal(q['signal'],nsteps,h,float(t),c['constant']);pilot=read_reference('H6/pilot/prefix13_I0.json');zk=complex(*pilot['exact_signal'][m]);ideal=np.exp(1j*c['energy']*float(t))
    point['observed']=dict(truncation_phase=pilot['exact_signal_phase'][m],PF_phase=float(np.angle(z*np.conj(zk))),combined_phase=float(np.angle(z*ideal)),signal=[z.real,z.imag])
    SAVE(OUT/f'P1/qualified/n{nsteps}.json',point);return point


def p1_plan():
    OUT.mkdir(parents=True,exist_ok=True);prepare_model(6)
    SAVE(OUT/'P1/PLAN.json',dict(source=SOURCE,reference=BASE,molecule=6,retention='prefix13_I0',formula='S2',order=[2,0,1,3,*range(4,14)],m=17,t_exact=str(F(1,50)*(1<<17)),steps=[32752,32767,32768,32769,32784],reuse_steps=[32768],new_steps=[32752,32767,32769,32784],independent_evaluation='all four new points after qualification; other 17 rounds and alpha fixed to original policy',P2='not executed until P1 finish confirms interpretation'))


def p1_qualify():parallel(p1_point,read_json(OUT/'P1/PLAN.json')['new_steps'])


def read_json(path):return json.loads(Path(path).read_text())


def independent(task):return d.independent_one(task)


def p1_independent():
    points=[read_json(p)for p in sorted((OUT/'P1/qualified').glob('*.json'))]
    parallel(independent,[(6,p)for p in points])


def saved_check(n,p):
    r=p['row'];return read_reference(f'H{n}/independent/{r["candidate"]["name"]}_{r["formula"]}_{r["h"]}_{p["signal"]["m"]}.json')


def p1_finish():
    base=original_policy(6);c=d.model(6);rows=[]
    for nsteps in read_json(OUT/'P1/PLAN.json')['steps']:
        if nsteps==32768:
            p=base['selected'][17];observed=next(x for x in read_reference('ordering/FINAL.json')['checks'] if x['n']==6 and x['key']==p['key']);reused=True
            mech=__import__('csv').DictReader((REF/'ordering/TRUNCATION_MECHANISMS.csv').open());r=next(x for x in mech if x['molecule']=='H6' and x['m']=='17')
            p={**p,'observed':dict(truncation_phase=float(r['truncation_phase_signed_observed']),PF_phase=float(r['PF_phase_signed_observed']),combined_phase=float(r['combined_phase_signed_observed']))}
        else:
            p=read_json(OUT/f'P1/qualified/n{nsteps}.json');r=p['row'];observed=read_json(OUT/f'checks/H6/independent/{r["candidate"]["name"]}_{r["formula"]}_{r["h"]}_17.json');reused=False
            p={**p,'qualification':h_rounding_supplement(p['qualification'],c,r,nsteps,17)}
        q=p['qualification'];ns=np.array([d.rigorous_shots(q['rho_lower'],q['beta_stat_lower'],a) for a in base['alpha'][17]]) if q['closed'] else None
        signals=[saved_check(6,x)['independent_signal']for x in base['selected'][:17]]
        z=physical_signal(observed['independent_signal'],nsteps,p['row']['h'],.02*2**17,c['constant']);phases=[float(np.angle(complex(*v)))for v in signals]+[float(np.angle(z))]
        decoded=d.reconstruct_rpe_phase(phases,[1<<m for m in range(18)]);energy=-decoded['final_phase_estimate']/.02;error=abs(energy-c['energy'])
        assert error<1.5936e-4
        totals=np.array(base['totals'])-(np.array(base['selected'][17]['mean'])*np.array(base['trials'][17])[:,None]).sum(axis=0)+(np.array(p['mean'])*ns[:,None]).sum(axis=0) if ns is not None else None
        rows.append(dict(steps=nsteps,h=p['row']['h'],h_exact=str(F(1,50)*(1<<17)/nsteps),reused=reused,qualification=q,observed=p['observed'],trials_fixed_alpha=ns.tolist() if ns is not None else None,whole_18round_totals_fixed_other_rounds=totals.tolist() if totals is not None else None,mean_decoder=dict(energy=energy,error_validation_only=error,no_E0_argument=True),mean=p['mean']))
    supported=all(x['qualification']['closed']for x in rows)
    SAVE(OUT/'P1/FINAL.json',dict(source=SOURCE,classification='安定性を支持（指定した有限5点）' if supported else '局所的な資格悪化または未確定',rows=rows,P2_may_proceed=supported,continuous_h_robustness_proved=False,reused_original_other_17_rounds=True,all_frozen_alpha_budget_from_original=True))
    print('P1_FINAL',supported,flush=True)


def run(stage):
    if (OUT/'STUDY_STOP.json').exists():raise RuntimeError('Study has stopped')
    {'p1-plan':p1_plan,'p1-qualify':p1_qualify,'p1-independent':p1_independent,'p1-finish':p1_finish}[stage]()


if __name__=='__main__':
    import sys
    run(sys.argv[1])
