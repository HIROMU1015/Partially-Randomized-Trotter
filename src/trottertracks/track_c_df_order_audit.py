"""One bounded leading-factor ordering audit of the frozen comparison.

The fixed Hamiltonians, PF families, P3 family, compiler and precision do not
change. Only late-round orders and adjacent dyadic h values are considered.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import lru_cache
from fractions import Fraction as F
import copy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import numpy as np
from scipy.linalg import eig
from . import track_c_df_truncation as d
from . import track_c_df_truncation_pr as pr

ROOT=d.OUT
OUT=ROOT/'ordering'


def anchors():
    det=json.loads((ROOT/'DETERMINISTIC_FINAL.json').read_text())['policies'];prs=json.loads((ROOT/'PR_FINAL.json').read_text())['policies'];out=[]
    for p in det+prs:
        # All deterministic methods receive the same opportunity. H4 PR is the
        # already optimized 24-order/Closed-P3 public result, kept unchanged.
        seen=set()
        for m in (17,16,15):
            point=p['selected'][m];row=point['row'];candidate=row.get('candidate',dict(name='PR',drop=[]))
            signature=(candidate['name'],row.get('ld'))
            if signature in seen:continue
            seen.add(signature);out.append(dict(n=p['n'],family=p['family'],point=point,base_m=m,base_trials=p['trials'][m]))
    for i,a in enumerate(out):a['id']=i
    return out


@lru_cache(None)
def blocks(n,candidate_json):return d.kept(d.model(n),json.loads(candidate_json))


@lru_cache(None)
def factor(n,candidate_json,index,t):
    b=next(b for b in blocks(n,candidate_json) if b['index']==index)
    return (b['B']*np.exp(-1j*t*b['diagonal']))@b['B'].conj().T


def pilot_one(task):
    anchor,kind,order,h=task;n=anchor['n'];family=anchor['family'];point=anchor['point'];old=point['row'];c=d.model(n);size=len(c['H']);case=dict(kind=kind,order=list(order));u=np.eye(size,dtype=complex)
    if family=='PR':
        partition=pr.partition(n,old['ld']);candidate_json=json.dumps(dict(drop=[]),sort_keys=True);scalar=float(partition['scalar']);e,v=partition['spectrum']
    else:
        candidate_json=json.dumps(old['candidate'],sort_keys=True);scalar=c['constant']
    seq=d.schedule(case,h)
    for role,index,t in seq:
        a=(v*np.exp(-1j*t*e))@v.conj().T if role=='R' else factor(n,candidate_json,index,t)
        u=a@u
    u*=np.exp(-1j*h*scalar);es,vs=eig(u);ps=abs(vs.conj().T@c['state'])**2;k=int(np.argmax(ps));pk=float(ps[k]);bias=-float(np.angle(es[k]*np.exp(1j*c['energy']*h)))/h
    leak=math.asin(min(1.,(1-pk)/pk)) if pk>.5 else math.pi
    old_steps=point['signal']['steps'];oldseq=d.schedule(old['case'],old['h']);oldDcount=sum(x[0]=='D' for x in oldseq);newDcount=sum(x[0]=='D' for x in seq)
    if family=='PR':baseD=np.array(point['D']);baseR=np.array(point['R']);baseW=np.array(point['overhead']);r=max(1,round(point['signal']['r']*h/old['h']))
    else:
        g=old['groups'];baseD=np.array(g['first'])+np.array(g['last'])+(old_steps-1)*np.array(g['middle']);baseR=np.zeros(4);baseW=np.array(point['mean'])-baseD;r=0
    rows=[]
    for m in (15,16,17):
        t=.02*2**m;steps=round(t/h);phase=abs(bias)*t+leak
        if phase>=.33 or pk<=.5:continue
        multiplier=steps/old_steps;Dproxy=baseD*multiplier*newDcount/oldDcount
        Rproxy=baseR*(steps*r/(old_steps*point['signal']['r'])) if family=='PR' else baseR
        logrange=point['qualification'].get('log_range',0.)*(steps*r/(old_steps*point['signal'].get('r',1))) if family=='PR' else 0.
        rho=(2*pk-1)*math.exp(-logrange);beta=.4-phase-(.03 if family=='PR' else .002)
        if beta<=0:continue
        N=math.ceil(4*math.log(1440)/(rho*rho*math.sin(beta)**2));proxy=N*float((2*(Dproxy+Rproxy)+baseW.sum(axis=0))[0])
        rows.append(dict(m=m,steps=steps,r=r,phase_nonresonant_proxy=phase,rho_proxy=rho,beta_proxy=beta,RZ_proxy=proxy))
    label=''.join(map(str,order[:min(4,len(order))]))
    result=dict(anchor=anchor['id'],n=n,family=family,kind=kind,order=list(order),h=h,order_label=label,dominant_overlap=pk,dominant_energy_bias=bias,nonresonant_leak_bound=leak,rounds=rows,source=d.source())
    d.save_json(OUT/f'pilot/a{anchor["id"]}_{kind}_{label}_{h}.json',result);return result


def pilot():
    OUT.mkdir(parents=True,exist_ok=True);aa=anchors();d.save_json(OUT/'ANCHORS.json',dict(source=d.source(),anchors=aa,scope='one bounded audit, late m15..17, leading up to four factors, h/2,h,2h inside original menu'))
    tasks=[]
    for a in aa:
        row=a['point']['row'];n=a['n']
        indices=list(range(row['ld']+1)) if a['family']=='PR' else [b['index'] for b in d.kept(d.model(n),row['candidate'])]
        leading=indices[:4];rest=indices[4:]
        for perm in itertools.permutations(leading):
            for kind in (('ordered','D4') if a['family']=='PR' else ('full_S2','full_Y4')):
                for h in sorted({row['h']/2,row['h'],row['h']*2}):
                    if .0025<=h<=2.56:tasks.append((a,kind,tuple(perm)+tuple(rest),h))
    with ProcessPoolExecutor(max_workers=d.workers(len(tasks))) as pool:
        for j,f in enumerate(as_completed(pool.submit(pilot_one,t) for t in tasks)):
            r=f.result()
            if j%64==0:print('ORDER_PILOT',j+1,len(tasks),r['n'],r['family'],flush=True)
    candidates={}
    for path in sorted((OUT/'pilot').glob('*.json')):
        row=json.loads(path.read_text())
        for q in row['rounds']:candidates.setdefault((row['n'],row['family'],q['m']),[]).append((q['RZ_proxy'],row,q))
    shortlist=[]
    for key,rows in sorted(candidates.items()):
        seen=set()
        for _,row,q in sorted(rows,key=lambda x:(x[0],x[1]['order_label'])):
            identity=(row['anchor'],row['kind'],tuple(row['order']),row['h'])
            if identity in seen:continue
            seen.add(identity);shortlist.append(dict(pilot=row,round=q))
            if len(seen)==2:break
    d.save_json(OUT/'SHORTLIST.json',dict(source=d.source(),tasks=len(tasks),points=shortlist,screen='proxy only; top two per molecule/method/late round. All final prices and qualifications must be direct. Original policies remain candidates.'))
    print('ORDER_SHORTLIST',len(shortlist),flush=True)


def qualify_one(item):
    p=item['pilot'];late=item['round'];n=p['n'];m=late['m'];c=d.model(n)
    anchor=json.loads((OUT/'ANCHORS.json').read_text())['anchors'][p['anchor']]
    old=anchor['point']['row'];case=dict(kind=p['kind'],order=p['order']);seq=d.schedule(case,p['h'])
    label=p['kind'].removeprefix('full_')+'_order'+p['order_label']
    row=dict(n=n,h=p['h'],case=case,source=d.source(),dominant_overlap=p['dominant_overlap'],dominant_energy_bias=p['dominant_energy_bias'],nonresonant_leak_bound=p['nonresonant_leak_bound'],groups={},angle_errors={},variants={})
    if p['family']=='PR':
        row.update(ld=old['ld'],kind=label);bb=pr.partition(n,old['ld'])['D'];gg=[[]]
        for x in seq:
            if x[0]=='R':gg.append([])
            else:gg[-1].append(x)
        from .track_c_mixed_formula import compact
        pieces=dict(first=gg[0],last=gg[-1],junction=compact(gg[-1]+gg[0]))
    else:
        row.update(candidate=old['candidate'],formula=label);bb=d.kept(c,old['candidate'])
        pieces=dict(first=seq[:1],last=seq[1:],middle=seq[1:-1]+[(seq[0][0],seq[0][1],2*seq[0][2])])
    for k,part in pieces.items():row['groups'][k],row['angle_errors'][k],row['variants'][k],_=d.price_piece(c,bb,part)
    signal=dict(m=m,name=label,h=p['h'],steps=late['steps'])
    if p['family']=='PR':
        signal['r']=late['r'];point=dict(key=[row['ld'],label,p['h'],late['r'],m],signal=signal,mean=np.zeros((2,4)),row=row)
        record=pr.qualify_point(point,n)
    else:
        d.sg.fro=d.fast_fro
        q=d.qualification_supplement(c,row,m,d.det_qualification(c,row,m))
        g=row['groups'];v=np.array(g['first'])+np.array(g['last'])+(late['steps']-1)*np.array(g['middle'])
        mean=np.array([v+d.wrapper(c,.02*2**m,a) for a in ('X','Y')],float)
        record=dict(key=[row['candidate']['name'],label,p['h'],m],signal=signal,qualification=q,mean=mean.tolist(),row=row)
    record.update(n=n,family=p['family'],audit_pilot_anchor=p['anchor'])
    partition_label='ld'+str(row['ld']) if p['family']=='PR' else row['candidate']['name']
    name=f'H{n}_{p["family"]}_{partition_label}_{label}_{p["h"]}_{m}.json'
    d.save_json(OUT/'qualified'/name,record);return n,p['family'],record['key'],record['qualification']['closed']


def qualify():
    points=json.loads((OUT/'SHORTLIST.json').read_text())['points']
    with ProcessPoolExecutor(max_workers=d.workers(len(points))) as pool:
        for r in pool.map(qualify_one,points):print('ORDER_QUALIFIED',r,flush=True)


def training_data():
    data={}
    for phase in ('training','holdout','ordering_training'):
        for cell,strata in pr.samples(phase).items():
            for rare,rs in strata.items():data.setdefault(cell,{}).setdefault(rare,[]).extend(rs)
    return data


def train():
    aa=json.loads((OUT/'ANCHORS.json').read_text())['anchors'];cells=set()
    for x in json.loads((OUT/'SHORTLIST.json').read_text())['points']:
        p=x['pilot'];q=x['round']
        if p['family']=='PR':cells.add((p['n'],aa[p['anchor']]['point']['row']['ld'],p['h'],q['r']))
    existing=training_data();needed=sorted(cells-set(existing));tasks=[(*cell,rare,j,'ordering_training') for cell in needed for rare in (False,True) for j in range(32)]
    if tasks:
        with ProcessPoolExecutor(max_workers=d.workers(len(tasks))) as pool:
            for j,_ in enumerate(pool.map(pr.price_one,tasks)):
                if j%64==0:print('ORDER_TRAIN',j+1,len(tasks),flush=True)
    d.save_json(OUT/'TRAINING.json',dict(source=d.source(),new_cells=needed,new_trajectories=len(tasks),reused_cells=sorted(cells&set(existing)),selection_data='Initial training AND initial holdout are selection data for this subsequent audit. A new independent ordering_holdout is mandatory for the final frozen PR policies.'))


def repaired_qualifications():
    anchors=json.loads((OUT/'ANCHORS.json').read_text())['anchors'];records=[]
    for x in json.loads((OUT/'SHORTLIST.json').read_text())['points']:
        p=x['pilot'];m=x['round']['m'];n=p['n'];label=p['kind'].removeprefix('full_')+'_order'+p['order_label']
        if p['family']=='PR':
            ld=anchors[p['anchor']]['point']['row']['ld'];path=ROOT/f'H{n}/pr_qualified/{ld}_{label}_{p["h"]}_{x["round"]["r"]}_{m}.json'
            record=json.loads(path.read_text());record.update(n=n,family='PR',audit_pilot_anchor=p['anchor'])
        else:
            path=OUT/f'qualified/H{n}_{p["family"]}_{label}_{p["h"]}_{m}.json';record=json.loads(path.read_text())
            assert record['row']['candidate']==anchors[p['anchor']]['point']['row']['candidate']
        assert record['row']['case']['order']==p['order'] and record['qualification']['closed']
        records.append({**record,'qualification_raw_path':str(path.relative_to(ROOT))})
    assert len({(x['n'],x['family'],tuple(x['key'])) for x in records})==48
    d.save_json(OUT/'QUALIFIED_COMPLETE.json',dict(source=d.source(),points=records,repair='Original audit filename omitted LD; two H6 PR m16 rows shared an audit filename. All 48 computations were closed and their distinct root pr_qualified records remain intact. This complete view reads those original records without recomputation. Original audit files, frozen policy, and holdout remain unchanged.'))


def freeze(check_complete=False):
    initial=json.loads((ROOT/'DETERMINISTIC_FINAL.json').read_text())['policies']+json.loads((ROOT/'PR_FINAL.json').read_text())['policies'];new=json.loads((OUT/'QUALIFIED_COMPLETE.json').read_text())['points'] if (OUT/'QUALIFIED_COMPLETE.json').exists() else [json.loads(p.read_text()) for p in sorted((OUT/'qualified').glob('*.json'))];rs=training_data();policies=[]
    assert len(new)==len(json.loads((OUT/'SHORTLIST.json').read_text())['points'])
    for old in initial:
        n=old['n'];family=old['family'];ps=copy.deepcopy(old['selected'])+[x for x in new if x['n']==n and x['family']==family and x['qualification']['closed']]
        if family=='PR':ps=[pr.priced(p,rs) for p in ps]
        else:ps=[{**p,'mean':np.array(p['mean'])} for p in ps]
        cost,selected,alpha,ns=d.choose(ps)
        baseline=ps[:18];baseN=np.array(old['trials']);basecost=sum((p['mean']*nn[:,None]).sum(axis=0)[0] for p,nn in zip(baseline,baseN))
        if basecost<cost:cost,selected,alpha,ns=basecost,baseline,np.array(old['alpha']),baseN
        selected=[{**p,'mean':p['mean'].tolist()} for p in selected]
        pol=dict(n=n,family=family,status='FROZEN',selected=selected,alpha=alpha.tolist(),trials=ns.tolist(),training_RZ=float(cost),trials_total=int(ns.sum()),selection='original frozen points plus top-two late-round bounded ordering audit; two-start integer allocation; initial policies remain whole-policy alternatives; finite-menu result')
        if family!='PR':pol.update(totals=np.sum([np.array(p['mean'])*nn[:,None] for p,nn in zip(selected,ns)],axis=(0,1)).tolist(),MC_SE=[0.]*4)
        policies.append(pol)
    if check_complete:
        old=json.loads((OUT/'POLICY.json').read_text())['policies'];same=[]
        for p,q in zip(policies,old):same.append(dict(n=p['n'],family=p['family'],same_keys=[x['key'] for x in p['selected']]==[x['key'] for x in q['selected']],same_alpha=p['alpha']==q['alpha'],same_trials=p['trials']==q['trials'],same_training_RZ=p['training_RZ']==q['training_RZ']))
        d.save_json(OUT/'SHORTLIST_COMPLETENESS_CHECK.json',dict(source=d.source(),candidate_points=len(new),comparisons=same,selection_data='unchanged original training, original holdout and ordering_training ONLY; final ordering_holdout not read',new_policy=policies))
        assert all(all(x[k] for k in ('same_keys','same_alpha','same_trials','same_training_RZ')) for x in same)
    else:d.save_json(OUT/'POLICY.json',dict(source=d.source(),policies=policies,price_training='initial training + initial holdout + new ordering_training; fresh final holdout required'))
    print('ORDER_FROZEN',[(p['n'],p['family'],p['training_RZ']) for p in policies],flush=True)


def holdout():
    policy=json.loads((OUT/'POLICY.json').read_text());cells=sorted({tuple(p['cell']) for pol in policy['policies'] if pol['family']=='PR' for p in pol['selected']});tasks=[(*cell,rare,j,'ordering_holdout') for cell in cells for rare in (False,True) for j in range(64)]
    with ProcessPoolExecutor(max_workers=d.workers(len(tasks))) as pool:
        for j,_ in enumerate(pool.map(pr.price_one,tasks)):
            if j%64==0:print('ORDER_HOLDOUT',j+1,len(tasks),flush=True)
    d.save_json(OUT/'HOLDOUT.json',dict(source=d.source(),cells=cells,trajectories=len(tasks),samples_per_stratum=64,independence='new SHA256 seed namespace ordering_holdout, frozen policy unchanged'))


def check_task(task):
    n,family,p=task
    if family=='PR':
        r=p['row'];target=ROOT/f'H{n}/pr_independent/{r["ld"]}_{r["kind"]}_{r["h"]}_{p["signal"]["r"]}_{p["signal"]["m"]}.json';fun=pr.independent_one
    else:
        r=p['row'];target=ROOT/f'H{n}/independent/{r["candidate"]["name"]}_{r["formula"]}_{r["h"]}_{p["signal"]["m"]}.json';fun=d.independent_one
    if target.exists():
        row=json.loads(target.read_text());assert row['key']==p['key'];return row
    return fun((n,p))


def finish():
    path=OUT/'POLICY.json';digest=hashlib.sha256(path.read_bytes()).hexdigest();policy=json.loads(path.read_text());rs=pr.samples('ordering_holdout');tasks={}
    for pol in policy['policies']:
        for p in pol['selected']:tasks[(pol['n'],pol['family'],tuple(p['key']))]=(pol['n'],pol['family'],p)
    with ProcessPoolExecutor(max_workers=d.workers(len(tasks))) as pool:checks=list(pool.map(check_task,tasks.values()))
    lookup={(x['n'],tuple(x['key'])):x for x in checks}
    for pol in policy['policies']:
        phases=[];total=np.zeros(4);det=np.zeros(4);coeff={};ispr=pol['family']=='PR'
        for p,ns in zip(pol['selected'],np.array(pol['trials'])):
            result=lookup[(pol['n'],tuple(p['key']))];z=complex(*result['finite_mean_signal' if ispr else 'independent_signal']);phases.append(float(np.angle(z)))
            if ispr:
                point=pr.priced(p,rs);total+=(point['mean']*ns[:,None]).sum(axis=0);det+=np.array(point['D'])*ns.sum()+(np.array(point['overhead'])*ns[:,None]).sum(axis=0)
                cell=tuple(p['cell']);coeff[cell]=coeff.get(cell,0)+int(ns.sum())*p['signal']['steps']
        if ispr:
            cov=np.zeros((4,4))
            for cell,k in coeff.items():
                for sub in rs[cell].values():
                    values=np.array([r['metrics'] for r in sub]);cov+=k*k*sub[0]['probability']**2*np.cov(values,rowvar=False,ddof=1)/len(values)
            pol.update(totals=total.tolist(),D_and_wrapper_totals=det.tolist(),R_totals=(total-det).tolist(),MC_covariance=cov.tolist(),MC_SE=np.sqrt(np.maximum(0,np.diag(cov))).tolist())
        decoded=d.reconstruct_rpe_phase(phases,[1<<m for m in range(18)]);energy=-decoded['final_phase_estimate']/.02;err=abs(energy-d.model(pol['n'])['energy']);assert err<1.5936e-4
        pol.update(status='QUALIFIED_INDEPENDENT_CHECK_COMPLETE',mean_decoder=dict(decoded=decoded,energy=energy,absolute_error_validation_only=err))
    seeds={x['seed'] for strata in training_data().values() for sub in strata.values() for x in sub};assert not seeds&{x['seed'] for strata in rs.values() for sub in strata.values() for x in sub}
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    d.save_json(OUT/'FINAL.json',dict(source=d.source(),policy_sha256=digest,policies=policy['policies'],checks=checks,fresh_holdout_seed_overlap=0))
    print('ORDER_FINAL',[(p['n'],p['family'],p['totals'][0]) for p in policy['policies']],flush=True)


if __name__=='__main__':
    import sys
    if (ROOT/'STUDY_STOP.json').exists():raise RuntimeError('Study has stopped')
    if sys.argv[1]=='pilot':pilot()
    elif sys.argv[1]=='qualify':qualify()
    elif sys.argv[1]=='train':train()
    elif sys.argv[1]=='freeze':freeze()
    elif sys.argv[1]=='holdout':holdout()
    elif sys.argv[1]=='finish':finish()
    elif sys.argv[1]=='repair-qualification-view':repaired_qualifications()
    elif sys.argv[1]=='confirm-complete':freeze(check_complete=True)
    else:raise ValueError(sys.argv[1])
