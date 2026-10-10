#!/usr/bin/env python3
"""Saved evidence only: explicit JW, tensor-native action, no Qiskit/project imports."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from scipy.linalg import expm

ROOT=Path(__file__).resolve().parents[1];TOL=4e-9


def mat(x):return np.asarray(x['real'])+1j*np.asarray(x['imag'])
def require(x,message):
    if not x:raise ValueError(message)
def close(x,y,message):
    e=float(np.linalg.norm(np.asarray(x)-np.asarray(y),2));require(e<TOL,message+str(e));return e
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def annihilate(n,i):
    a=np.zeros((1<<n,1<<n),complex)
    for s in range(1<<n):
        if s>>i&1:a[s^(1<<i),s]=(-1)**((s&((1<<i)-1)).bit_count())
    return a


def orbital(g):
    n=len(g);a=[annihilate(n,i) for i in range(n)]
    return sum((g[i,j]*a[i].conj().T@a[j] for i,j in itertools.product(range(n),repeat=2)),np.zeros((1<<n,1<<n),complex))


def pauli(label):
    ops={'I':np.eye(2),'X':np.array([[0,1],[1,0]]),'Y':np.array([[0,-1j],[1j,0]]),'Z':np.diag([1,-1])}
    out=np.ones((1,1),complex)
    for x in label:out=np.kron(out,ops[x])
    return out


def occupation_norm(x,nu):
    vals=[abs(sum(x[list(i)])) for i in itertools.combinations(range(len(x)),nu)]
    return float(max(vals))


def wrapped(op,axis):
    n=int(math.log2(len(op)));eye=np.eye(len(op));h=np.array([[1,1],[1,-1]])/math.sqrt(2)
    prep=np.eye(len(op))[np.arange(len(op))^3]
    return np.kron(h,eye)@np.kron(np.diag([1,-1j]) if axis=='Y' else np.eye(2),eye)@np.block([[eye,np.zeros_like(eye)],[np.zeros_like(eye),op]])@np.kron(h,prep)


def native(ir):
    require(digest({k:v for k,v in ir.items() if k!='sha256'})==ir['sha256'],'IR hash')
    n=ir['qubits'];ns=ir['system'];nw=ir['workspace'];anc=int(ir['controlled']);logical=mat(ir['logical_reference'])
    require(n==ns+nw+anc and 1<=n<=13,'IR dimensions')
    require(logical.shape==(1<<(ns+anc),)*2,'reference dimensions')
    out=np.zeros((1<<n,len(logical)),complex);target=out.copy()
    for a in range(1<<anc):
        for s in range(1<<ns):
            row=s+(a<<(ns+nw));col=s+(a<<ns);out[row,col]=1;target[row]=logical[col]
    counts=Counter();layers=[0]*n
    for item in ir['operations']:
        name,qs,p=item['name'],item['qubits'],item['parameters'];counts[name]+=1
        require(name in ('rz','sx','x','cx'),'gate vocabulary')
        require(len(qs)==(2 if name=='cx' else 1) and len(set(qs))==len(qs) and all(0<=q<n for q in qs),'gate qubits')
        require(len(p)==int(name=='rz') and all(math.isfinite(v) for v in p),'parameters')
        layer=max(layers[q] for q in qs)+1
        for q in qs:layers[q]=layer
        if name=='cx':
            inds=np.arange(1<<n);perm=inds^(((inds>>qs[0])&1)<<qs[1]);out=out[perm]
        else:
            u=np.diag(np.exp(np.array([-1j,1j])*p[0]/2)) if name=='rz' else np.array([[0,1],[1,0]]) if name=='x' else np.array([[1+1j,1-1j],[1-1j,1+1j]])/2
            shape=out.reshape([2]*n+[out.shape[1]])
            axis=n-1-qs[0];moved=np.moveaxis(shape,axis,0)
            moved=np.einsum('ab,b...->a...',u,moved)
            out=np.moveaxis(moved,0,axis).reshape(out.shape)
    out*=np.exp(1j*ir['global_phase'])
    error=close(out,target,'native absolute clean-workspace action:')
    return error,{**{m:counts[m] for m in ('rz','cx','sx','x')},'size':sum(counts.values()),'depth':max(layers)}


def validate(result,irs,audit,check_source=True):
    require(result['next_stage_authorized'] is False and result['central_hypothesis_adopted'] is None and result['mandatory_stop'],'STOP')
    if check_source:
        for path,expected in audit['source_sha256'].items():
            data=subprocess.check_output(['git','show',audit['source_commit']+':'+path],cwd=ROOT)
            require(hashlib.sha256(data).hexdigest()==expected,'source blob '+path)
    costs={};maxerr=0.;logical={}
    for ir in irs:
        require(ir['label'] not in costs,'unique circuit label')
        error,cost=native(ir);maxerr=max(maxerr,error);costs[ir['label']]=cost;logical[ir['label']]=mat(ir['logical_reference'])
    rows=0
    for context in result['contexts']:
        if context['track']=='N1':
            n,nu=context['modes'],context['particles'];idx=[s for s in range(1<<n) if s.bit_count()==nu]
            h1=orbital(mat(context['one_body']));scalar=context['scalar'];h=h1+scalar*np.eye(1<<n)
            for g,w in zip(context['input_factors'],context['weights']):
                f=orbital(mat(g));h+=w*f@f
            close(h,mat(context['hamiltonian']),'input H')
            selected={c['name']:c for c in context['selected']}
            for candidates in context['generator_candidates']:
                for c in candidates:
                    eta=np.asarray(c['eta']);et=np.asarray(c['approx_eta']);frame=mat(c['frame'])
                    close(frame.conj().T@frame,np.eye(n),'frame unitary')
                    close(frame@np.diag(et)@frame.conj().T,mat(c['g']),'factor frame')
                    require(c['gauge_residual']<1e-10,'gauge residual')
            for row in context['rows']:
                c=selected[row['candidate']];blocks=[h1];hs=h1+scalar*np.eye(1<<n)
                for g,w in zip(c['factors'],c['weights']):
                    f=orbital(mat(g));blocks.append(w*f@f);hs+=w*f@f
                if c['paulis'] is not None:
                    blocks=[v*pauli(p) for p,v in c['paulis'].items()]
                    hs=sum(blocks,np.zeros_like(h));scalar_pf=0.
                else:scalar_pf=scalar
                op=np.eye(1<<n,dtype=complex);dt=context['time']/row['q']
                for _ in range(row['q']):
                    for block in blocks+list(reversed(blocks)):op=expm(-.5j*dt*block)@op
                op*=np.exp(-1j*context['time']*scalar_pf)
                close(op,mat(row['operator']),'PF operator')
                actual_model=float(np.linalg.norm((h-hs)[np.ix_(idx,idx)],2))
                require(actual_model<=row['model_bound']+1e-10,'model bound')
                pf=float(np.linalg.norm((op-expm(-1j*context['time']*hs))[np.ix_(idx,idx)],2))
                require(abs(pf-row['pf_sector_bias'])<1e-10,'PF bias')
                require(abs(context['time']*row['model_bound']+pf-row['accounted_original_bias'])<1e-10,'original H accounting')
                verify_row(row,op,costs,logical,context['track']);rows+=1
        elif context['track']=='N2':
            j=np.asarray(context['J']);n=len(j);bits=np.array([[s>>i&1 for i in range(n)] for s in range(1<<n)])
            con=context['construction']
            if 'S' in con:
                s,k=np.asarray(con['S']),np.asarray(con['K']);e=j-s@k@s.T
                close(e,np.asarray(con['E']),'charge residual')
                require(sum(abs(e.ravel()))<=context['model_budget']+1e-12,'charge budget')
            for row in context['rows']:
                model=np.asarray(row['model_J']);energies=np.einsum('bi,ij,bj->b',bits,model,bits)
                op=np.diag(np.exp(-1j*context['time']*energies));close(op,mat(row['operator']),'N2 phase')
                require(abs(sum(abs((j-model).ravel()))-row['model_bound'])<1e-12,'N2 coefficient bound')
                require(abs(context['time']*row['model_bound']-row['resources']['bias_budget_used'])<1e-12,'N2 original H accounting')
                verify_row(row,op,costs,logical,context['track']);rows+=1
        else:
            for row in context['rows']:
                n=row['modes'];g=np.diag(row['energies']);a=[annihilate(n,i) for i in range(n)]
                for i,j,v in row['hoppings']:g[i,j]=g[j,i]=v
                h=orbital(g)
                for i,j,v in row['density']:h+=v*(a[i].conj().T@a[i])@(a[j].conj().T@a[j])
                idx=[s for s in range(1<<n) if s.bit_count()==row['particles']]
                ground=float(np.linalg.eigvalsh(h[np.ix_(idx,idx)])[0]);active=row['selected_active']
                p=[s for s in idx if all(not(s>>i&1) for i in range(n) if i not in active)]
                aground=float(np.linalg.eigvalsh(h[np.ix_(p,p)])[0])
                require(abs(ground-row['ground_truth_post_selection'])<1e-12,'N3 evaluator truth')
                require(abs(aground-row['active_ground_post_selection'])<1e-12,'N3 active truth')
                for hist in row['history']:verify_certificate(row,hist)
                require(aground-ground<=row['certificate']['delta']+1e-12,'N3 shift bound');rows+=1
    require(len(irs)==result['compiled_circuits'],'compile count')
    return {'status':'SAVED_EVIDENCE_PASS','source_blobs':len(audit['source_sha256']),'native_ir':len(irs),
            'semantic_rows':rows,'maximum_native_action_error':maxerr,'new_science_runs':0}


def verify_row(row,op,costs,logical,track):
    for axis,c in zip(('X','Y'),row['costs'],strict=True):
        close(logical[c['label']],wrapped(op,axis),'wrapper semantics')
        for m,v in costs[c['label']].items():require(c[m]==v,'native metric')
    r=row['resources'];bias=r['bias_budget_used'];margin=r['epsilon_complex']/math.sqrt(2)-bias
    shots=math.ceil(2*math.log(80)/margin**2) if margin>0 else None
    require(shots==r['shots_per_axis'],'shots')
    for m,v in r['xy_cost'].items():
        require(v==sum(c[m] for c in row['costs']),'XY sum')
        if shots is not None:require(r['work'][m]==shots*v,'work')


def verify_certificate(row,hist):
    cert=hist['certificate'];active=hist['active'];external=[i for i in range(row['modes']) if i not in active]
    trial=set(sorted(active)[:row['particles']]);u=sum(row['energies'][i] for i in trial)+sum(v for i,j,v in row['density'] if i in trial and j in trial)
    require(abs(u-cert['U'])<1e-12,'variational coefficient U')
    vn=sum(abs(v) for i,j,v in row['hoppings']+row['density']);off=sum(abs(v) for i,j,v in row['hoppings'] if i in external or j in external)
    for j,c in enumerate(cert['classes']):
        orb=external[j];free=[i for i in range(row['modes']) if i not in external[:j+1]]
        # independent occupation enumeration for the small evaluation check
        ref=min(row['energies'][orb]+sum(row['energies'][i] for i in subset) for subset in itertools.combinations(free,row['particles']-1))
        beta=sum(abs(v) for i,k,v in row['hoppings'] if orb in (i,k) and (k if i==orb else i) in active)
        require(abs(c['gap_lower']-(ref-vn-(len(external)-1)*off-u))<1e-12,'coefficient class lower bound')
        require(abs(c['beta']-beta)<1e-12,'coefficient coupling bound')
    if cert['status']=='UNRESOLVED_LOWER_BOUND':require(cert['delta'] is None,'no fabricated certificate')
    elif cert['classes']:
        d=cert['delta'];rhs=sum(c['beta']**2/(c['gap_lower']+d) for c in cert['classes'])
        require(all(c['gap_lower']>0 for c in cert['classes']) and d>=0 and abs(d-rhs)<1e-12,'class fixed point')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():raise SystemExit('no overwrite')
    result=json.loads((args.run/'result.json').read_text());irs=json.loads((args.run/'native_ir.json').read_text());audit=json.loads((args.run/'run_audit.json').read_text())
    report=validate(result,irs,audit)
    # Semantic/hash mutations: absolute phase, workspace dimensions, resource shots.
    rejected=[]
    for mutation in ('phase','workspace','shots'):
        rr=json.loads(json.dumps(result));ii=json.loads(json.dumps(irs))
        if mutation=='phase':ii[0]['global_phase']+=.01;ii[0]['sha256']=digest({k:v for k,v in ii[0].items() if k!='sha256'})
        elif mutation=='workspace':ii[0]['workspace']+=1;ii[0]['sha256']=digest({k:v for k,v in ii[0].items() if k!='sha256'})
        else:rr['contexts'][0]['rows'][0]['resources']['shots_per_axis']=1
        try:validate(rr,ii,audit,False)
        except ValueError:rejected.append(mutation)
        else:raise ValueError('mutation accepted '+mutation)
    report['mutations_rejected']=rejected
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
