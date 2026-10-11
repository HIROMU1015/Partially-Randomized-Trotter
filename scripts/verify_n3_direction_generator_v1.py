#!/usr/bin/env python3
"""Independent saved rational audit. No science module, solver or eigenproblem."""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import itertools
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DELTA=F(1,250)

def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def fs(x):
    if not isinstance(x,list) or len(x)!=2 or any(type(v)!=int for v in x) or x[1]<=0:raise ValueError('rational schema')
    f=F(*x)
    if max(f.numerator.bit_length(),f.denominator.bit_length())>4096:raise ValueError('rational bits')
    return f

class Audit:
    def __init__(self):self.checks=0;self.max_bits=0
    def req(self,ok,msg):
        if not ok:raise ValueError(msg)
        self.checks+=1
    def f(self,x):
        x=F(x);b=max(x.numerator.bit_length(),x.denominator.bit_length());self.max_bits=max(self.max_bits,b)
        self.req(b<=4096,'rational intermediate cap');return x
    def proof(self,M,p):
        # Independent LDL recurrence on the original matrix; producer uses Schur updates.
        n=len(M);L=[[F(i==j) for j in range(n)] for i in range(n)];ds=[];status='PSD'
        for i in range(n):
            d=self.f(M[i][i]-sum((self.f(L[i][k]*L[i][k]*ds[k]) for k in range(i)),F(0)));ds.append(d)
            if d<0:status='NEGATIVE_PIVOT';break
            residual=[self.f(M[j][i]-sum((self.f(L[j][k]*L[i][k]*ds[k]) for k in range(i)),F(0))) for j in range(i+1,n)]
            if d==0:
                if p['strict']:status='ZERO_PIVOT_NOT_SPD';break
                if any(residual):status='ZERO_PIVOT_NONZERO_ROW';break
            else:
                for j,x in zip(range(i+1,n),residual):L[j][i]=self.f(x/d)
        self.req(p['status']==status and p['accepted']==(status=='PSD'),'independent PSD status')
        self.req(list(map(fs,p['pivots']))==ds and [[fs(x) for x in row] for row in p['L']]==L,'saved LDL evidence')
        if status=='PSD':
            for i in range(n):
                for j in range(n):self.req(sum((self.f(L[i][k]*ds[k]*L[j][k]) for k in range(n)),F(0))==M[i][j],'exact PSD reconstruction')
        return status=='PSD'
    def witness(self,A,v,nu,U,c,w):
        k,mu,D=fs(w['kappa']),fs(w['mu']),c-U+DELTA
        self.req(fs(w['D'])==D,'D reconstruction');matrix=[[self.f(k*A[i][j]-(mu if i==j else 0)-v[i]*v[j]) for j in range(len(A))] for i in range(len(A))]
        proofok=self.proof(matrix,w['PSD']);margin=DELTA*D-(k*U-nu*mu)
        self.req(fs(w['affine_margin'])==margin and fs(w['lower_box_margin'])==k and fs(w['upper_box_margin'])==D-k,'original three inequality margins')
        ok=c>U and 0<=k<=D and margin>=0 and proofok;self.req(w['accepted']==ok,'original witness acceptance');return ok
    def case(self,case,record):
        self.req(record['input_id']==case['id'] and record['coefficient_input_sha256']==hashlib.sha256(canon(case)).hexdigest(),'coefficient identity')
        h=[[fs(x) for x in row] for row in case['h']];active=case['active'];ext=case['external'];nu=case['particles'];density={}
        self.req(set(active+[ext])==set(range(len(h))) and len(set(active+[ext]))==len(h),'one external scope')
        for d in case['density']:
            key=tuple(sorted((d['i'],d['j'])));self.req(key[0]!=key[1] and key not in density and fs(d['coefficient'])>=0,'unordered nonnegative density');density[key]=fs(d['coefficient'])
        A=[[h[i][j] for j in active] for i in active];v=[h[i][ext] for i in active];b=record['bounds'];dets=list(itertools.combinations(sorted(active),nu));expected=[]
        def diag(occ):return sum((h[i][i] for i in occ),F(0))+sum((value for (i,j),value in density.items() if i in occ and j in occ),F(0))
        for occ in dets:expected.append(('determinant',[list(occ)],diag(occ),1))
        for a,bb in itertools.combinations(dets,2):
            left=set(a)-set(bb);right=set(bb)-set(a)
            if len(left)==len(right)==1:
                p,q=next(iter(left)),next(iter(right));hop=h[p][q]*(-1)**(bb.index(q)+a.index(p))
                if hop:expected.append(('connected_equal_pair',[list(a),list(bb)],(diag(a)+diag(bb))/2-abs(hop),-1 if hop>0 else 1))
        trials=b['trial_states'];self.req(len(trials)==len(expected)<=128 and len(dets)<=35,'trial inventory')
        for t,(kind,occs,value,sign) in zip(trials,expected):
            self.req(t['kind']==kind and t['occupations']==occs and fs(t['value'])==value and t['relative_sign']==sign,'normalized coefficient trial')
            self.req(t['normalization']==('1' if kind=='determinant' else '1/sqrt(2)'),'trial normalization')
        U=min(x[2] for x in expected);self.req(fs(b['U'])==U and fs(trials[b['selected_trial']]['value'])==U,'trial U')
        r=min(A[i][i]-sum((abs(A[i][j]) for j in range(len(A)) if j!=i),F(0)) for i in range(len(A)));s=max(A[i][i] for i in range(len(A)))-min(A[i][i] for i in range(len(A)))
        self.req(fs(b['r'])==r and fs(b['s'])==s,'Gershgorin and span');lows=[r];oldlows=[r];self.req(len(b['old_lambda'])==5,'old lambda cap')
        for x,rho in zip(b['old_lambda'],(F(0),F(1,20),F(1,10),F(3,20),F(1,5))):
            lam=r+rho*s;self.req(fs(x['lambda'])==lam and x['proof']['strict'],'frozen lambda');M=[[A[i][j]-(lam if i==j else 0) for j in range(len(A))] for i in range(len(A))]
            if self.proof(M,x['proof']):
                L=[[fs(z) for z in row] for row in x['proof']['L']];ds=list(map(fs,x['proof']['pivots']));y=[]
                for i in range(len(v)):y.append(self.f(v[i]-sum((L[i][j]*y[j] for j in range(i)),F(0))))
                self.req(fs(x['kappa'])==sum((y[i]**2/ds[i] for i in range(len(v))),F(0)),'inverse quadratic');lows.append(lam);oldlows.append(lam)
        refined=b['refined_lower'];self.req(len(refined)<=6 and b['one_body_proposal_calls']==1,'refinement cap')
        if refined:
            E=max([F(1)]+[abs(x) for row in A for x in row]);step=E/(1<<24);floor=(F(b['one_body_proposal'])/step).__floor__()*step
            self.req(len(refined)==6,'refinement inventory')
            for x,offset in zip(refined,(1,2,4,8,16,32)):
                lam=floor-offset*step;self.req(fs(x['lambda'])==lam and x['offset']==offset and not x['proof']['strict'],'refinement recipe')
                if self.proof([[A[i][j]-(lam if i==j else 0) for j in range(len(A))] for i in range(len(A))],x['proof']):lows.append(lam)
        ell=max(lows);c=h[ext][ext]+(nu-1)*ell;oldc=h[ext][ext]+(nu-1)*max(oldlows);self.req(fs(b['ell'])==ell and fs(b['c'])==c and fs(b['old_ell'])==max(oldlows) and fs(b['legacy_c_for_five_point'])==oldc,'coefficient c')
        beta1=sum(map(abs,v),F(0));beta2=sum((z*z for z in v),F(0));D=c-U+DELTA;names=['legacy_scalar','common_norm_only','strengthened_scalar','five_point_common','direct_one_variable','five_point_legacy_c'];self.req([m['method'] for m in record['methods']]==names,'method inventory')
        for m in record['methods']:
            name=m['method'];self.req(fs(m['common_U'])==U and fs(m['common_c'])==c and fs(m['beta1'])==beta1 and fs(m['beta2'])==beta2,'common comparison bounds')
            if name=='legacy_scalar':
                lu=diag(tuple(sorted(active)[:nu]));vnorm=sum((abs(h[i][j]) for i in range(len(h)) for j in range(i+1,len(h))),F(0))+sum(density.values(),F(0));lc=h[ext][ext]+sum(sorted(h[i][i] for i in active)[:nu-1],F(0))-vnorm;margin=DELTA*(lc-lu+DELTA)-beta1**2;ok=lc>lu and margin>=0
                self.req(fs(m['legacy_U'])==lu and fs(m['legacy_c'])==lc and fs(m['legacy_vnorm'])==vnorm and fs(m['margin'])==margin,'original legacy definition')
            elif name in ('common_norm_only','strengthened_scalar'):
                margin=DELTA*D-(beta2 if name=='common_norm_only' else beta1**2);self.req(fs(m['margin'])==margin,'scalar margin');ok=c>U and margin>=0
            elif name.startswith('five_point'):
                mc=c if name=='five_point_common' else oldc;oks=[];self.req(len(m['attempts'])==5,'five point attempts')
                for attempt,x in zip(m['attempts'],b['old_lambda']):
                    self.req(attempt['lambda']==x['lambda'] and attempt['spd']==x['proof']['accepted'],'cached lambda evidence')
                    if attempt['spd']:
                        w=attempt['witness'];self.req(fs(w['kappa'])==fs(x['kappa']) and fs(w['mu'])==fs(x['kappa'])*fs(x['lambda']),'five point witness recipe');oks.append(self.witness(A,v,nu,U,mc,w))
                ok=any(oks)
                if m['witness']:self.req(m['witness']['accepted'] and m['witness'] in [x.get('witness') for x in m['attempts']],'chosen five witness')
            else:
                attempts=m['attempts'];self.req(len(attempts)<=10 and not m.get('old_grid_fallback',False),'direct cap/no fallback')
                if m['search']:
                    search=m['search'];self.req(search['eigenvalue_calls']<=128 and 0<=search['x']<=1,'search limits');xr=F(search['x']);expected_x={F(0),F(1)}
                    for bits in (16,24,32,40):expected_x.update((F((xr*(1<<bits)).__floor__(),1<<bits),F((xr*(1<<bits)).__ceil__(),1<<bits)))
                    self.req([fs(w['x']) for w in attempts]==sorted(expected_x),'direct dyadic recipe')
                oks=[]
                for w in attempts:
                    k=fs(w['x'])*D;self.req(fs(w['kappa'])==k and fs(w['mu'])==(k*U-DELTA*D)/nu and fs(w['affine_margin'])==0,'mu0 and exact zero affine margin');oks.append(self.witness(A,v,nu,U,c,w))
                ok=any(oks)
                if m['witness']:self.req(m['witness']['accepted'] and m['witness'] in attempts,'chosen direct witness')
            self.req(m['accepted']==ok and m['selected_modes']==(len(active) if ok else len(h)) and m['used_full_space_fallback']==(not ok),'method acceptance/fallback')
        return {m['method']:m['accepted'] for m in record['methods']}

def audit(root,out):
    a=Audit();receipt=json.loads((out/'launch_receipt.json').read_bytes());run=json.loads((out/'run_audit.json').read_bytes())
    a.req(receipt['scientific_attempt']==1 and receipt['scientific_retries']==0,'one launch')
    for p,s in receipt['source_input_sha256'].items():a.req(hashlib.sha256((root/p).read_bytes()).hexdigest()==s,'frozen identity '+p)
    for p,entry in run['outputs'].items():
        raw=(out/p).read_bytes();a.req(hashlib.sha256(raw).hexdigest()==entry['sha256'] and len(raw)==entry['bytes'],'output hash '+p)
    ledgerraw=(out/'certificate_ledger.json').read_bytes();seal=json.loads((out/'ledger_seal.json').read_bytes());ledger=json.loads(ledgerraw)
    a.req(hashlib.sha256(ledgerraw).hexdigest()==seal['sha256'] and seal['before_many_body_diagnostics'],'pretruth ledger seal')
    cases=json.loads((root/'artifacts/n3_generator_n1_recovery/2026-10-11/preparation/n3_transfer_inputs.json').read_bytes())['cases'];a.req(len(cases)==len(ledger)==12,'12 frozen cases');accepts=[]
    for case,record in zip(cases,ledger):accepts.append(a.case(case,record));a.req(record['resources']['rational_max_bits_observed']<=4096 and record['resources']['matrix_dimension']<=7,'arithmetic/matrix scope')
    diags=json.loads((out/'diagnostics.json').read_bytes());a.req(diags['certificate_ledger_sha256']==seal['sha256'] and diags['ground_eigenvalue_calls']==24,'truth access order/call cap')
    for case,record,d in zip(cases,ledger,diags['cases']):
        a.req(d['input_id']==case['id'] and abs(d['width']-(d['active_ground']-d['full_ground']))<=1e-14,'diagnostic inventory/width')
        if any(m['accepted'] for m in record['methods']):a.req(d['width']<=.004+1e-10,'saved diagnostic containment')
    groups=sorted({case['base'] for case,x in zip(cases,accepts) if x['direct_one_variable'] and not x['common_norm_only']});gains=[case['id'] for case,x in zip(cases,accepts) if x['direct_one_variable'] and not x['five_point_common']]
    status='GENERATED_TRANSFER_GAIN' if len(groups)>=2 and gains else 'TRANSFER_CERTIFIED_NO_NEW_GRID_GAIN' if len(groups)>=2 else 'NO_TRANSFER_ADVANTAGE_IN_FROZEN_SCOPE'
    result=json.loads((out/'result.json').read_bytes());a.req(result['status']==status and result['mandatory_stop'] and not result['next_scientific_stage_authorized'] and result['central_theme'] is None,'terminal decision/STOP')
    return {'status':'PASS','checks':a.checks,'method_records':72,'rational_max_bits_observed':a.max_bits,'scientific_status':status,'direct_vs_norm_base_groups':groups,'direct_vs_common_five_point_cases':gains,'new_scientific_runs':0,'eigenvalue_solves':0,'scope':'independent saved rational verification, not immutable CI or external scientific reproduction','mandatory_stop':True}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path,required=True);p.add_argument('--report',type=Path);args=p.parse_args();result=audit(args.root.resolve(),args.output.resolve());data=json.dumps(result,indent=2)+'\n'
    if args.report:args.report.write_text(data)
    print(data,end='')
if __name__=='__main__':main()
