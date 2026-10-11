#!/usr/bin/env python3
"""Post-STOP independent replay of saved N3/N2 diagnostics, not a scientific batch.

No project/Qiskit imports, candidate search, circuit compilation or LP solve.
Uses exactly the saved coefficient fixtures and accepted/rejected certificate records.
"""
from fractions import Fraction as F
import argparse
import itertools as it
import json
from pathlib import Path
import numpy as np


def ff(x):return F(*x)
def fj(x):return [x.numerator,x.denominator]


def sector_matrix(h,density):
    states=[s for s in range(64) if s.bit_count()==2];loc={s:i for i,s in enumerate(states)};H=np.zeros((15,15))
    for column,s in enumerate(states):
        for p,q in it.product(range(6),repeat=2):
            if not(s>>q&1):continue
            removed=s^(1<<q)
            if removed>>p&1:continue
            sign=(-1)**((s&((1<<q)-1)).bit_count()+(removed&((1<<p)-1)).bit_count())
            H[loc[removed|(1<<p)],column]+=sign*float(h[p][q])
        H[column,column]+=sum(float(v) for (i,j),v in density.items() if s>>i&1 and s>>j&1)
    return states,H


def solve_exact(a,b):
    n=len(b);m=[row.copy()+[b[i]] for i,row in enumerate(a)]
    for i in range(n):
        pivot=next(j for j in range(i,n) if m[j][i]);m[i],m[pivot]=m[pivot],m[i]
        scale=m[i][i];m[i]=[x/scale for x in m[i]]
        for j in range(n):
            if j!=i:
                scale=m[j][i];m[j]=[x-scale*y for x,y in zip(m[j],m[i])]
    return [row[-1] for row in m]


def audit(out):
    count=0;spectra=0;accepted=0;directions=0;lpchecks=[]
    def req(ok,msg):
        nonlocal count
        if not ok:raise ValueError(msg)
        count+=1
    n3=json.loads((out/'n3.json').read_bytes())
    for item,di in zip(n3['selector'],n3['diagnostics'],strict=True):
        req(item['input']==di['input'],'case identity');h=[[ff(x) for x in row] for row in item['h']];density={(i,j):ff(v) for i,j,v in item['density']}
        states,H=sector_matrix(h,density);req(np.linalg.norm(H-H.T)==0,'real Hermitian saved model')
        E=float(np.linalg.eigvalsh(H)[0]);spectra+=1
        diagnostic={(x['method'],x['stage']):x for x in di['diagnostics']};cache={}
        for method in item['methods']:
            previous=None
            for record in method['records']:
                active=record['active'];external=record['external'];key=tuple(active)
                if key not in cache:
                    ids=[j for j,s in enumerate(states) if all(not(s>>e&1) for e in external)];qids=[j for j in range(15) if j not in ids]
                    A=H[np.ix_(ids,ids)];C=H[np.ix_(qids,qids)];B=H[np.ix_(ids,qids)]
                    a=float(np.linalg.eigvalsh(A)[0]);c=float(np.linalg.eigvalsh(C)[0]) if qids else None;spectra+=1+bool(qids);cache[key]=(ids,qids,A,C,B,a,c)
                ids,qids,A,C,B,a,c=cache[key];saved=diagnostic[method['method'],record['stage']]
                req(abs(a-saved['active_ground_diagnostic'])<=3e-12 and abs(E-saved['full_ground_diagnostic'])<=3e-12,'independent explicit JW spectra')
                req(abs(a-E-saved['width_diagnostic'])<=3e-12,'independent width')
                req((c is None and saved['discarded_ground_diagnostic'] is None) or (c is not None and abs(c-saved['discarded_ground_diagnostic'])<=3e-12),'independent discarded spectrum')
                if previous is not None:
                    prior=previous['active'];ext=previous['external'];promoted=max(ext,key=lambda j:(sum((abs(h[i][j]) for i in prior),F(0)),-j))
                    req(active==sorted(prior+[promoted]),'coefficient-only promotion rule')
                previous=record
                if record['accepted']:
                    accepted+=1;req(a-E<=.004+3e-12,'accepted reduction soundness')
                if 'classes' in record:
                    U=ff(record['U']);req(a<=float(U)+3e-12,'legacy trial upper')
                    off=ff(record['off_bound'])
                    classes=[]
                    for pos,z in enumerate(record['classes']):
                        orb=z['orbital'];js=[j for j,s in enumerate(states) if s>>orb&1 and all(not(s>>e&1) for e in external[:pos])]
                        block=H[np.ix_(js,js)];small=float(np.linalg.eigvalsh(block)[0]);spectra+=1
                        req(small+3e-12>=float(ff(z['class_lower'])),'legacy class lower')
                        req(np.linalg.norm(H[np.ix_(ids,js)],2)<=float(ff(z['beta']))+3e-12,'legacy active-class coupling bound');classes.append(js)
                    for i,j in it.combinations(range(len(classes)),2):req(np.linalg.norm(H[np.ix_(classes[i],classes[j])],2)<=float(off)+3e-12,'legacy class off bound')
                if 'lambda_candidates' in record:
                    req(c+3e-12>=float(ff(record['c'])) and a<=float(ff(record['U']))+3e-12,'improved U/c enclosure')
                    for trial in record['trials']:
                        if trial['kind']=='determinant':
                            index=states.index(sum(1<<i for i in trial['occupation']));value=H[index,index]
                        else:
                            left,right=trial['occupations'];i,j=(states.index(sum(1<<k for k in occ)) for occ in (left,right));value=(H[i,i]+H[j,j])/2-abs(H[i,j])
                            req(abs(H[i,j]-float(ff(trial['fermionic_matrix_element'])))<=3e-12,'independent trial CAR sign')
                        req(abs(value-float(ff(trial['value'])))<=3e-12,'independent trial Rayleigh expectation')
                    req(abs(min(float(ff(x['value'])) for x in record['trials'])-float(ff(record['U'])))<=3e-12,'finite trial upper selection')
                    v=np.array(list(map(lambda x:float(ff(x)),record['v'])));rank_h=np.zeros((6,6))
                    rank_h[np.ix_(active,active)]=np.outer(v,v);_,rank=sector_matrix([[F(float(x)) for x in row] for row in rank_h],{})
                    req(np.linalg.norm(B@B.T-rank[np.ix_(ids,ids)])<=3e-12,'independent CAR BBt direction')
                    for z in record['lambda_candidates']:
                        if not z['spd']:continue
                        lam=ff(z['lambda']);kappa=ff(z['kappa']);margin=float(np.linalg.eigvalsh(float(kappa)*(A-2*float(lam)*np.eye(len(A)))-B@B.T)[0]);spectra+=1;directions+=1
                        req(margin>=-3e-12,'independent lifted matrix dominance')
    n2=json.loads((out/'n2.json').read_bytes());D=[[F(v) for v in row] for row in n2['design']];w=n2['weights']
    normal=[[sum(w[k]*D[k][i]*D[k][j] for k in range(len(w))) for j in range(3)] for i in range(3)]
    for fixture in n2['fixtures']:
        J=list(map(ff,fixture['J_upper']));rhs=[sum(w[k]*D[k][i]*J[k] for k in range(len(w))) for i in range(3)];K=solve_exact(normal,rhs)
        l1=sum((w[k]*abs(J[k]-sum(D[k][j]*K[j] for j in range(3))) for k in range(len(w))),F(0))
        req(abs(float(l1)-fixture['Frobenius_fit_L1_diagnostic'])<=3e-12,'rational weighted-Frobenius solution residual')
        req(l1>F(21,10000) and ff(fixture['primal_L1'])<=F(21,10000),'L1 eligibility recovered, LS ineligible')
        lpchecks.append({'interpretation':fixture['interpretation'],'Frobenius_K_exact':[fj(x) for x in K],'Frobenius_L1_exact':fj(l1)})
    return {'status':'PASS','checks':count,'N3_accepted_records':accepted,'N3_direction_matrix_checks':directions,'saved_diagnostic_spectral_checks':spectra,
       'N2_exact_LS_audit':lpchecks,'scientific_batch_runs':0,'scientific_candidate_searches':0,'circuit_compilations':0,'LP_solves':0,
       'scope':'Independent replay of saved coefficient fixtures and closed decisions only; no new scientific conditions or compiler/LP execution.','mandatory_stop':True}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--report',required=True,type=Path);a=p.parse_args();result=audit(a.output);data=json.dumps(result,indent=2)+'\n';a.report.write_text(data);print(data,end='')


if __name__=='__main__':main()
