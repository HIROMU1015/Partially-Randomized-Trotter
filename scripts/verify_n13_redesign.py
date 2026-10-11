#!/usr/bin/env python3
"""Independent saved audit: stdlib + NumPy only, no science imports/compile/solve.

Replays native gates and rational certificate arithmetic on saved records.
This is saved evidence verification, not external physical reproduction or CI.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x):return hashlib.sha256(canon(x)).hexdigest()
def fs(x):return F(*x)
def mat(x):return np.asarray(x['real'])+1j*np.asarray(x['imag'])


@lru_cache(None)
def symplectic(label):
    x=z=0
    for i,p in enumerate(reversed(label)):
        if p in 'XY':x|=1<<i
        if p in 'ZY':z|=1<<i
    return x,z


@lru_cache(None)
def product(a,b):
    xa,za=symplectic(a);xb,zb=symplectic(b);x,z=xa^xb,za^zb
    exponent=((xa&za).bit_count()+(xb&zb).bit_count()-(x&z).bit_count()+2*(za&xb).bit_count())%4
    chars=[]
    for i in reversed(range(len(a))):chars.append('Y' if (x>>i&1 and z>>i&1) else 'X' if x>>i&1 else 'Z' if z>>i&1 else 'I')
    return ''.join(chars),exponent


def bracket(a,b):
    out={}
    for ka,(ar,ai) in a.items():
        for kb,(br,bi) in b.items():
            k,e=product(ka,kb)
            if not e%2:continue
            real,imag=ar*br-ai*bi,ar*bi+ai*br;sign=2 if e==1 else -2
            v=(-sign*imag,sign*real);old=out.get(k,(0,0));out[k]=(old[0]+v[0],old[1]+v[1])
    return {k:v for k,v in out.items() if v!=(0,0)}


def independently_pf(certs):
    scale=1<<128;centres=[];radii=[]
    for c in certs:
        centres.append({k:((lo+hi)//2,0) for k,(lo,hi) in c.items() if k.strip('I') and (lo+hi)//2})
        radii.append(sum((F(max((lo+hi)//2-lo,hi-(lo+hi)//2),scale) for k,(lo,hi) in c.items() if k.strip('I')),F(0)))
    def norm(c,degree=1):return F(sum(abs(re)+abs(im) for re,im in c.values()),scale**degree)
    value=F(0)
    for j,h in enumerate(centres[:-1]):
        r={}
        for c in centres[j+1:]:
            for k,v in c.items():old=r.get(k,(0,0));r[k]=(old[0]+v[0],old[1]+v[1])
        rn,hn=norm(r),norm(h);dr,dh=sum(radii[j+1:],F(0)),radii[j]
        en1=4*((rn+dr)**2*(hn+dh)-rn**2*hn);en2=4*((hn+dh)**2*(rn+dr)-hn**2*rn)
        value+=(norm(bracket(r,bracket(r,h)),3)+en1)/12+(norm(bracket(h,bracket(h,r)),3)+en2)/24
    return value


def native(ir):
    n=ir['qubits'];u=np.eye(1<<n,dtype=complex)
    # Independent column-by-column scalar index loop, different from optimized kernel.
    for op in ir['operations']:
        name,qs,params=op['name'],op['qubits'],op['parameters'];bit=1<<qs[-1]
        for low in range(1<<n):
            if low&bit:continue
            high=low|bit
            if name=='rz':u[low]*=np.exp(-.5j*params[0]);u[high]*=np.exp(.5j*params[0])
            elif name=='sx':
                a,b=u[low].copy(),u[high].copy();u[low]=((1+1j)*a+(1-1j)*b)/2;u[high]=((1-1j)*a+(1+1j)*b)/2
            elif name=='x' or (name=='cx' and low&(1<<qs[0])):u[[low,high]]=u[[high,low]]
            elif name!='cx':raise ValueError('unknown native operation')
    return np.exp(1j*ir['global_phase'])*u


def audit(root,out):
    checks=0;counts={'IR':0,'N1_rows':0,'N3_records':0,'N2_fixtures':0};worst=0.
    def req(ok,msg):
        nonlocal checks
        if not ok:raise ValueError(msg)
        checks+=1
    a=json.loads((out/'run_audit.json').read_bytes());req(a['scientific_retries']==0,'no retry');req(a['mandatory_stop'] and not a['next_scientific_stage_authorized'] and a['central_theme'] is None,'STOP')
    for p,s in a['input_sha256'].items():req(hashlib.sha256((root/p).read_bytes()).hexdigest()==s,'frozen source/input '+p)
    for p,entry in a['outputs'].items():
        raw=(out/p).read_bytes();req(len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256'],'saved output '+p)
    for p in out.rglob('ir_*.json.gz'):
        ir=json.loads(gzip.decompress(p.read_bytes()));req(ir['qubits_initially_zero'] is False and ir['qubits']<=5,'arbitrary-input semantics')
        got=native(ir);target=mat(ir['logical_reference']);err=float(np.linalg.norm(got-target,2));worst=max(worst,err);req(err<=3e-9,'absolute all-column native action '+p.name)
        req(np.linalg.norm(mat(ir['emitted_reference'])-target,2)<=3e-9,'builder reference');counts['IR']+=1
    if (out/'result.json').exists():
        result=json.loads((out/'result.json').read_bytes());req(result['wrapper_requests']<=3072 and result['transpile_calls']<=3116,'compile cap')
    for p in out.glob('*/result.json'):
        r=json.loads(p.read_bytes());closed=r['selection'];rows=r['rows'];req(closed['rows_sha256']==digest(rows),'closed N1 rows');req(r['diagnostics']['closed_selection_sha256']==digest(closed),'diagnostic order seal')
        byid={x['id']:x for x in rows};truth={x['id']:x for x in r['diagnostics']['rows']}
        pack=json.loads((root/'artifacts/n13_redesign/2026-10-11/preparation/n1_candidate_input.json').read_bytes())
        ctx=next(c for c in pack['contexts'] if c['context']==r['context']);hcert={}
        for i in range(4):
            label=['I']*4;label[3-i]='Z';v=-F(ctx['one_body']['real'][i][i])/2;hcert[''.join(label)]=[int(v*(1<<128))]*2
        pf_cache={}
        for x in rows:
            counts['N1_rows']+=1;b=fs(x['bias_bound']);req(fs(x['pf_bound'])==F(.6)**3*fs(x['pf_certificate']['alpha2'])/x['q']**2,'PF accounting')
            key=(*x['factor_indices'],x['gauge'])
            if key not in pf_cache:
                aidx,bidx,g=key;pf_cache[key]=independently_pf([hcert,r['factors'][0][aidx][g]['pauli_coefficients'],r['factors'][1][bidx][g]['pauli_coefficients']])
            req(pf_cache[key]==fs(x['pf_certificate']['alpha2']),'independent exact Pauli commutators')
            req(b==F(.6)*fs(x['model_bound'])+fs(x['raw_input_growth_extra'])+fs(x['pf_bound'])+fs(x['angle_bound']),'bias accounting')
            for metric in ('rz','cx','sx','x','size','depth'):
                xy=sum(x['costs'][axis][metric] for axis in ('X','Y'));req(xy==x['xy_cost'][metric],'XY cost')
                req(x['objective'][metric]==(x['shots']*xy if x['shots'] else None),'shot work')
            if x['shots']:
                # Verify the saved integer upper shot is sufficient; not a new selector.
                margin=.04/math.sqrt(2)-float(b);req(margin>0 and x['shots']>=math.ceil(2*math.log(80)/margin**2)-1,'shot lower check')
            t=truth[x['id']];req(t['model_error']<=float(fs(x['model_bound']))+3e-11 and t['pf_error']<=float(fs(x['pf_bound']))+3e-11 and t['total_error']<=float(b)+3e-11,'diagnostic containment')
        families={'exact':{'exact'},'N1':{'exact','cluster'},'shifted':{'exact','shifted_cutoff'},'zero':{'exact','zero_cutoff'},'all_mixed':{'exact','cluster','shifted_cutoff','zero_cutoff'}}
        for family,sels in closed['selections'].items():
            eligible=[x for x in rows if all(f in families[family] for f in x['families']) and x['shots'] and fs(x['model_bound'])<=F(.02)]
            for metric,id_ in sels.items():
                secondary='cx' if metric=='rz' else 'rz';chosen=min(eligible,key=lambda x:(x['objective'][metric],x['objective'][secondary],fs(x['model_bound']),x['q'],x['id'])) if eligible else None
                req(id_==(chosen['id'] if chosen else None),'saved selector argmin')
        for ls in r['factors']:
            for vs in ls:
                for x in vs.values():
                    d0,d1=2*fs(x['r0']),2*fs(x['r1']);b=fs(x['commuting_bound'])+abs(fs(x['weight']))*(d0*(2*fs(x['s0'])+d0)+d1*(2*fs(x['s1'])+d1))
                    req(b==fs(x['model_bound']),'coefficient residual accounting')
        for x in r['whole_pauli']:
            certs=[{k:[int(fs(v)*(1<<128))]*2} for k,v in x['pauli_coefficients'].items()]
            req(independently_pf(certs)==fs(x['pf_certificate']['alpha2']),'whole-Pauli commutator bound')
            t=x['diagnostic'];req(t['model_error']<=float(fs(x['model_bound']))+3e-11 and t['pf_error']<=float(F(.6)**3*fs(x['pf_certificate']['alpha2'])/x['q']**2)+3e-11 and t['total_error']<=float(fs(x['bias_bound']))+3e-11,'whole-Pauli containment')
    if (out/'n3.json').exists():
        n3=json.loads((out/'n3.json').read_bytes());req(n3['closed_selection_sha256']==digest(n3['selector']),'N3 order seal')
        diagnostics={(item['input'],x['method'],x['stage']):x for item in n3['diagnostics'] for x in item['diagnostics']}
        for item in n3['selector']:
            h=[[fs(x) for x in row] for row in item['h']]
            for method in item['methods']:
                for x in method['records']:
                    counts['N3_records']+=1;diag=diagnostics[item['input'],method['method'],x['stage']]
                    if x['accepted']:req(diag['width_diagnostic']<=1/250+3e-12,'accepted width')
                    if 'lambda_candidates' in x:
                        active=x['active'];v=list(map(fs,x['v']));U,c=fs(x['U']),fs(x['c']);req(diag['active_ground_diagnostic']<=float(U)+3e-12 and diag['discarded_ground_diagnostic']+3e-12>=float(c),'N3 U/c')
                        req(fs(x['beta2'])==sum((z*z for z in v),F(0)),'CAR norm')
                        for z in x['lambda_candidates']:
                            if not z['spd']:continue
                            l=[[fs(a) for a in row] for row in z['LDL_L']];d=list(map(fs,z['LDL_D']));lam=fs(z['lambda']);req(all(a>0 for a in d),'positive LDL pivots')
                            for i in range(len(active)):
                                for j in range(len(active)):
                                    req(sum((l[i][k]*d[k]*l[j][k] for k in range(len(active))),F(0))==h[active[i]][active[j]]-int(i==j)*lam,'exact LDL identity')
                            y=[]
                            for i in range(len(v)):y.append(v[i]-sum((l[i][j]*y[j] for j in range(i)),F(0)))
                            req(sum((y[i]**2/d[i] for i in range(len(v))),F(0))==fs(z['kappa']),'exact quadratic solve')
                            if z.get('accepted'):
                                t=fs(z['kappa'])/(c-U+F(1,250));req(c>U and t<=1 and F(1,250)>=t*(U-2*lam),'rational direction acceptance')
                        if x['method']=='norm_only' and x['accepted'] and fs(x['beta2']):req(c>U and F(1,250)>=fs(x['beta2'])/(c-U+F(1,250)),'norm-only acceptance')
                    elif 'classes' in x and x['accepted']:
                        U=fs(x['U']);rhs=sum((fs(z['beta'])**2/(fs(z['discarded_lower'])-U+F(1,250)) for z in x['classes']),F(0));req(all(fs(z['gap'])>0 for z in x['classes']) and rhs<=F(1,250),'legacy rational acceptance')
        req(counts['N3_records']<=36,'N3 record cap')
    if (out/'n2.json').exists():
        n2=json.loads((out/'n2.json').read_bytes());req(n2['LP_calls']==1 and n2['new_S_scan']==0 and n2['circuit_compilations']==0,'one fixed-S LP')
        design=[[F(v) for v in row] for row in n2['design']];weights=n2['weights'];n=len(weights)
        for x in n2['fixtures']:
            js=list(map(fs,x['J_upper']));ks=list(map(fs,x['K_primal']));dual=list(map(fs,x['dual']));res=[js[i]-sum((design[i][j]*ks[j] for j in range(3)),F(0)) for i in range(n)]
            primal=sum((weights[i]*abs(res[i]) for i in range(n)),F(0));lower=sum(((dual[i]-dual[n+i])*js[i] for i in range(n)),F(0))
            req(primal==fs(x['primal_L1']) and lower==fs(x['dual_lower']) and primal>=lower,'LP sandwich')
            req(all(z<=0 for z in dual),'dual signs')
            for j in range(3):req(sum(((dual[i]-dual[n+i])*design[i][j] for i in range(n)),F(0))==0,'dual coefficient equality')
            req(all(-dual[i]-dual[n+i]<=weights[i] for i in range(n)),'dual slack feasibility');counts['N2_fixtures']+=1
    return {'status':'PASS','checks':checks,'counts':counts,'worst_absolute_native_residual':worst,'scientific_runs':0,'compilations':0,'LP_solves':0,
            'scope':'independent saved arithmetic/native replay; no new science, no external reproduction or immutable CI','mandatory_stop':True}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path,required=True);p.add_argument('--report',type=Path);a=p.parse_args();result=audit(a.root.resolve(),a.output.resolve())
    data=json.dumps(result,indent=2)+'\n'
    if a.report:a.report.write_text(data)
    print(data,end='')


if __name__=='__main__':main()
