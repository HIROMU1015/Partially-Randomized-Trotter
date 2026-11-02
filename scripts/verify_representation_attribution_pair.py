#!/usr/bin/env python3
"""Read-only saved-data audit; no Qiskit, project imports, sampling or compilation.

Uses exterior-power minors for Gaussian Fock matrices and direct native row
updates for complete absolute-phase wrappers. This is a local independent
implementation check, not externally reproduced science or interval arithmetic.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import time

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
from scipy.linalg import expm

ROOT=Path(__file__).resolve().parents[1]
ATOL=3e-10
METRICS=('rz','cx','sx','x','size','depth')
P={'I':np.eye(2,dtype=complex),'X':np.array([[0,1],[1,0]],complex),
   'Y':np.array([[0,-1j],[1j,0]]),'Z':np.diag([1.,-1.])}
H=np.array([[1,1],[1,-1]],complex)/math.sqrt(2)

def require(condition,message):
    if not condition: raise ValueError(message)

def sha(data): return hashlib.sha256(data).hexdigest()

def load(path): return json.loads(path.read_text())

def matrix(value): return np.array(value['real'])+1j*np.array(value['imag'])

def close(a,b,label):
    error=float(np.linalg.norm(np.asarray(a)-np.asarray(b),ord=2)) if np.ndim(a)==2 else abs(a-b)
    require(error<ATOL,label+': '+str(error))
    return error

def pauli(label):
    out=np.ones((1,1),complex)
    for ch in label: out=np.kron(out,P[ch])
    return out

def single(n,q,ch): return pauli(''.join(ch if i==q else 'I' for i in reversed(range(n))))

def fock(v):
    n=len(v);occ=[[i for i in range(n) if (k>>i)&1] for k in range(2**n)]
    out=np.zeros((2**n,2**n),complex)
    for i,a in enumerate(occ):
        for j,b in enumerate(occ):
            if len(a)==len(b): out[i,j]=np.linalg.det(v[np.ix_(a,b)]) if a else 1
    return out

def native(ir):
    require(set(ir)=={'id','label','qubits','global_phase','operations','sha256'},'IR fields')
    raw={k:v for k,v in ir.items() if k!='sha256'}
    require(sha(json.dumps(raw,sort_keys=True,separators=(',',':')).encode())==ir['sha256'],'IR digest')
    n=ir['qubits'];require(1<=n<=5,'IR qubit cap');indices=np.arange(2**n)
    out=np.eye(2**n,dtype=complex);counts=Counter();depth=[0]*n
    require(len(ir['operations'])<=10000,'IR gate cap')
    for op in ir['operations']:
        name=op['name'];qs=op['qubits'];params=op['parameters'];counts[name]+=1
        require(name in ('rz','sx','x','cx'),'native gate vocabulary')
        require(len(qs)==(2 if name=='cx' else 1) and len(set(qs))==len(qs)
                and all(0<=q<n for q in qs),'native qubits')
        require(len(params)==(1 if name=='rz' else 0),'native parameters')
        require(all(math.isfinite(p) for p in params),'finite parameters')
        layer=max(depth[q] for q in qs)+1
        for q in qs: depth[q]=layer
        q=qs[-1];lo=indices[(indices&(1<<q))==0];hi=lo|(1<<q)
        if name=='rz':
            out[lo]*=np.exp(-.5j*params[0]);out[hi]*=np.exp(.5j*params[0])
        elif name=='sx':
            a=out[lo].copy();b=out[hi].copy()
            out[lo]=((1+1j)*a+(1-1j)*b)/2
            out[hi]=((1-1j)*a+(1+1j)*b)/2
        else:
            if name=='cx': lo=lo[(lo&(1<<qs[0]))!=0];hi=lo|(1<<q)
            a=out[lo].copy();out[lo]=out[hi];out[hi]=a
    out*=np.exp(1j*ir['global_phase'])
    return out,{**{m:counts[m] for m in METRICS[:4]},'size':sum(counts.values()),'depth':max(depth)}

def wrapped(u,axis,prep):
    d=len(u);eye=np.eye(d);had=np.kron(H,eye)
    ctrl=np.block([[eye,np.zeros_like(eye)],[np.zeros_like(eye),u]])
    return had@np.kron(np.diag([1,-1j]) if axis=='Y' else np.eye(2),eye)@ctrl@had@np.kron(np.eye(2),prep)

def event_operator(event,primitive,n):
    eye=np.eye(2**n);out=eye.copy();apps=event['application_sequence']
    require(len(apps)==event['taylor_order']+1,'event length')
    require([a['component_id'] for a in apps]==event['selected_component_ids'],'event order')
    require([a['component_id'] for a in apps[:-1]]==event['product_component_ids'],'event products')
    require(apps[-1]['component_id']==event['rotation_component_id'],'event rotation')
    for app in apps:
        require(app['coefficient_sign'] in (-1,1),'component sign')
        op=app['coefficient_sign']*primitive(app['component_id'])
        if app['role']=='rotation':
            angle=event['rotation_angle'];op=math.cos(angle)*eye-1j*math.sin(angle)*op
        else: require(app['role']=='product','event application role')
        out=op@out
    phase=complex(event['phase']['real'],event['phase']['imag'])
    close(abs(phase),1.,'event phase modulus')
    return phase*out

def scalar_close(a,b,label):
    require(math.isclose(a,b,rel_tol=2e-14,abs_tol=1e-8),label)

def annihilators(n):
    aa=[]
    for j in range(n):
        a=np.zeros((2**n,2**n),complex)
        for k in range(2**n):
            if (k>>j)&1:a[k^(1<<j),k]=(-1)**((k&((1<<j)-1)).bit_count())
        aa.append(a)
    return aa

def gamma(lam,time,q):
    tau=lam*time/q
    return (math.sqrt(1+tau*tau)+tau*tau/2*math.sqrt(1+(tau/3)**2))**q

def budget(g,b,e):
    margin=e/math.sqrt(2)-b
    return None if margin<=0 else math.ceil(2*g*g/margin**2*math.log(80))

def primitives(con):
    out={}
    for p in con['primitives']:
        v=fock(matrix(p['orbital_frame']))
        if p['kind']=='pair_Q':
            z=np.ones(2**con['n']);z[(np.arange(len(z))&3)==3]=-1;op=np.diag(z)
        else:op=pauli(p['label'])
        op=v@op@v.conj().T
        if p['reflected_aux'] is not None:
            z=single(con['n'],p['reflected_aux'],'Z');op=z@op@z
        out[p['id']]=op
    return out

def trajectory(con,event,time,ops):
    n=con['n'];out=np.eye(2**n,dtype=complex)
    blocks=[expm(-.5j*time*matrix(b['matrix'])) for b in con['deterministic_blocks']]
    for b in blocks:out=b@out
    if event is not None:out=event_operator(event,lambda cid:ops[cid],n)@out
    for b in reversed(blocks):out=b@out
    v=fock(matrix(con['outer_frame']))
    return np.exp(-1j*time*con['identity'])*v@out@v.conj().T

def stats(row,metric,axis):
    mean=0.;variance=0.
    for s in row['strata']:
        values=np.array([d['costs'][axis][metric] if axis!='XY' else d['costs']['X'][metric]+d['costs']['Y'][metric]
                         for d in s['draws']])
        mean+=s['probability']*np.dot([d['weight'] for d in s['draws']],values)
        if not s['exact']:variance+=s['probability']**2*np.var(values,ddof=1)/len(values)
    return float(mean),math.sqrt(variance)

def work_vector(row,metric,epsilon):
    n=next(d['shots_per_axis'] for d in row['precision_resources'] if d['epsilon_complex']==epsilon)
    if n is None:return None
    result=np.zeros(96)
    for s in row['strata']:
        vals=np.array([sum(d['costs'][a][metric] for a in ('X','Y')) for d in s['draws']])
        if s['exact']:result+=s['probability']*np.dot([d['weight'] for d in s['draws']],vals)
        else:result+=s['probability']*vals
    return n*result

def verify(run,mutation_checks=False):
    start=time.monotonic();r=load(run/'result.json');irs=load(run/'native_ir.json');audit=load(run/'run_audit.json')
    for name,info in audit['outputs'].items():
        raw=(run/name).read_bytes();require(len(raw)==info['bytes'] and sha(raw)==info['sha256'],'output identity '+name)
    source=audit['source_commit'];require(r['source_commit']==source,'source identity')
    for path,digest in audit['source_sha256'].items():
        require(sha(subprocess.check_output(['git','show',source+':'+path],cwd=ROOT))==digest,'source blob '+path)
    require(r['mandatory_stop'] and r['next_stage_authorized'] is False and r['central_hypothesis_adopted'] is None,'STOP')
    for key in ('quantum_shots_executed','molecular_loads','gpu_calls','ground_state_solves'):require(r[key]==0,key)
    require(r['C']['new_runs']==0 and r['conditional_draws']==96,'scope')
    uniforms=np.array(r['common_uniforms']);require(uniforms.shape==(96,3),'uniform shape')
    # Replay saved generator state only; no sampling of scientific trajectories.
    require(np.array_equal(uniforms,np.random.default_rng(r['uniform_seed']).random((96,3))),'uniform identity')
    aa=annihilators(3);factors=[matrix(g) for g in r['A']['input_factors']]
    lift=lambda g:sum(g[i,j]*aa[i].conj().T@aa[j] for i,j in itertools.product(range(3),repeat=2))
    ha=sum(lift(g)@lift(g) for g in factors);close(ha,matrix(r['A']['hamiltonian']),'independent A input')
    bm=r['B']['metadata'];v=matrix(bm['unitary_completion']);bf=fock(v);u=matrix(bm['orbital_isometry'])
    close(u@u.conj().T,np.eye(3),'isometry')
    enlarged=bf@sum(c*pauli(p) for p,c in bm['diagonal_paulis'].items())@bf.conj().T
    hp=enlarged[:8,:8];close(hp,matrix(r['B']['hamiltonian']),'B vacuum projection')
    quartic=np.zeros((8,8),complex)
    for i,j,w in bm['diagonal_density_edges']:
        cx=sum(u[k,i].conjugate()*aa[k] for k in range(3));cy=sum(u[k,j].conjugate()*aa[k] for k in range(3))
        quartic+=w*cx.conj().T@cy.conj().T@cy@cx
    close(quartic,hp,'independent B quartic')
    bindings={};max_mean=0.;max_reconstruction=0.;mean_rows=0;cost_draws=0
    structural=('basis_calls','basis_operations','reflection_z_actions')
    for group,expected,h in [('A',6,ha),('B',4,hp)]:
        study=r[group];cons={c['name']:c for c in study['candidates']};rows={x['candidate']:x for x in study['rows']}
        require(len(cons)==len(rows)==expected,'candidate coverage')
        for name,con in cons.items():
            row=rows[name];ops=primitives(con);n=con['n'];eye=np.eye(2**n)
            blocks=[fock(matrix(b['orbital_frame']))@sum(c*pauli(p) for p,c in b['terms'].items())@
                    fock(matrix(b['orbital_frame'])).conj().T for b in con['deterministic_blocks']]
            for built,b in zip(blocks,con['deterministic_blocks']):close(built,matrix(b['matrix']),'block algebra')
            tail=sum((p['coefficient']*ops[p['id']] for p in con['primitives']),np.zeros_like(eye,dtype=complex))
            close(tail,matrix(con['tail_matrix']),'tail dictionary')
            outer=fock(matrix(con['outer_frame']));whole=outer@(sum(blocks,np.zeros_like(tail))+tail+con['identity']*eye)@outer.conj().T
            close(whole,matrix(con['whole_hamiltonian']),'whole dictionary')
            max_reconstruction=max(max_reconstruction,close(whole[:8,:8],h,'physical reconstruction'))
            lam=sum(abs(p['coefficient']) for p in con['primitives']);close(lam,con['tail_lambda'],'tail l1')
            if name=='complete_occupation_core':close(blocks[0],np.diag(np.diag(ha)),'complete diagonal')
            for pair in con['metadata'].get('pairs',[]):
                x=matrix(pair['x']);y=matrix(pair['y']);cx=sum(x[k]*aa[k] for k in range(3));cy=sum(y[k]*aa[k] for k in range(3))
                delta=float((np.vdot(x,x)*np.vdot(y,y)-abs(np.vdot(x,y))**2).real);close(delta,pair['delta'],'Gram determinant')
                basis=fock(matrix(pair['orbital_frame']));proj=np.diag(((np.arange(8)&3)==3).astype(float))
                close(cx.conj().T@cy.conj().T@cy@cx,delta*basis@proj@basis.conj().T,'pair identity')
            require([d['q'] for d in row['all_q_diagnostics']]==[1,2,4] and row['q']==1,'q coverage')
            for d in row['all_q_diagnostics']:
                q=d['q'];dt=study['time']/q;z=-1j*dt*tail;poly=eye+z+z@z/2+z@z@z/6
                ds=[expm(-.5j*dt*b) for b in blocks];left=eye.copy();right=eye.copy()
                for b in ds:left=b@left
                for b in reversed(ds):right=b@right
                mean=np.exp(-1j*study['time']*con['identity'])*outer@np.linalg.matrix_power(right@poly@left,q)@outer.conj().T
                physical=mean[:8,:8];max_mean=max(max_mean,close(physical,matrix(d['corrected_mean_physical_block']),'finite mean'))
                bias=float(np.linalg.norm(physical-expm(-1j*study['time']*h),2));close(bias,d['operator_bias'],'bias')
                g=gamma(lam,study['time'],q);close(g,d['normalization'],'Gamma')
                require(d['shots_per_axis']==[budget(g,bias,e) for e in (.05,.02)],'diagnostic shots');mean_rows+=1
            probs=np.array([abs(p['coefficient'])/lam for p in con['primitives']]) if lam else np.array([])
            if lam:probs/=probs.sum()
            tau=lam*study['time'];weights=[math.sqrt(1+tau*tau),tau*tau/2*math.sqrt(1+(tau/3)**2)]
            prep=np.eye(2**n,dtype=complex)
            for k,gate in study['prepared_gates']:prep=single(n,k,gate.upper())@prep
            for k,s in enumerate(row['strata']):
                order=s['order'];count=1 if not lam else len(probs)**(order+1)
                require(s['exact']==(count<=96),'stratum exact flag')
                require(len(s['draws'])==(count if s['exact'] else 96),'stratum coverage')
                close(s['probability'],1. if not lam else weights[k]/sum(weights),'order weight')
                exact_indices=list(itertools.product(range(len(probs)),repeat=order+1)) if lam and s['exact'] else None
                sampled_indices=np.searchsorted(np.cumsum(probs),uniforms,side='right') if lam and not s['exact'] else None
                for j,d in enumerate(s['draws']):
                    indices=d['indices_rotation_first'];event=d['event'];cost_draws+=1
                    require(d['draw']==j,'draw index')
                    if lam:
                        require(indices==(list(exact_indices[j]) if s['exact'] else sampled_indices[j].tolist()),'component draw coupling')
                        prob=float(np.prod(probs[indices]));close(d['weight'],prob if s['exact'] else 1/96,'conditional weight')
                        ids=[con['primitives'][i]['id'] for i in indices]
                        require(event['rotation_component_id']==ids[0] and event['product_component_ids']==ids[1:],'event selected law')
                        require(event['taylor_order']==order,'event order');close(event['rotation_angle'],math.atan(tau/(order+1)),'angle')
                        close(event['event_probability'],s['probability']*prob,'event probability')
                        close(event['event_coefficient'],weights[k]*prob,'event coefficient')
                        close(event['event_normalization'],sum(weights),'event normalization')
                        close(complex(event['phase']['real'],event['phase']['imag']),(-1)**(order//2),'order phase')
                        for app in event['application_sequence']:
                            coeff=next(p['coefficient'] for p in con['primitives'] if p['id']==app['component_id'])
                            require(app['coefficient_sign']==(1 if coeff>0 else -1),'component sign binding')
                    else:require(event is None and indices==[] and d['weight']==1,'deterministic design')
                    unitary=trajectory(con,event,study['time'],ops)
                    for axis in ('X','Y'):
                        cost=d['costs'][axis];ideal=wrapped(unitary,axis,prep);iid=cost['ir_id']
                        if iid in bindings:close(bindings[iid][1],ideal,'cache consistency')
                        else:bindings[iid]=(cost,ideal)
            for axis in ('X','Y','XY'):
                for metric in (*METRICS,*structural):
                    mean,se=stats(row,metric,axis)
                    scalar_close(mean,row['stratified_cost'][axis][metric]['mean'],'stratified mean')
                    scalar_close(se,row['stratified_cost'][axis][metric]['se'],'stratified SE')
            for resource in row['precision_resources']:
                diag=row['all_q_diagnostics'][0];shots=budget(diag['normalization'],diag['operator_bias'],resource['epsilon_complex'])
                require(resource['shots_per_axis']==shots,'resource shots')
                if shots is None:require(resource['expected_work'] is None and resource['expected_work_se'] is None,'excluded resources')
                else:
                    for metric in METRICS:
                        mean,se=stats(row,metric,'XY');scalar_close(shots*mean,resource['expected_work'][metric],'work')
                        scalar_close(shots*se,resource['expected_work_se'][metric],'work paired XY SE')
        for d in study['paired_differences']:
            left=work_vector(rows[d['left']],d['metric'],d['epsilon_complex']);right=work_vector(rows[d['right']],d['metric'],d['epsilon_complex'])
            if left is None or right is None:require(d['mean_left_minus_right'] is None and d['paired_se'] is None,'excluded difference')
            else:
                scalar_close(float(np.mean(left-right)),d['mean_left_minus_right'],'candidate difference')
                scalar_close(float(np.std(left-right,ddof=1)/math.sqrt(96)),d['paired_se'],'coupled candidate SE')
    require(len(irs)==r['compiled_circuits']<=2048 and set(bindings)==set(range(len(irs))),'complete IR binding')
    maximum=0.;gate_total=0;max_gate=0
    for index,ir in enumerate(irs):
        require(ir['id']==index,'IR ordering');actual,counts=native(ir);cost,ideal=bindings[index]
        require(cost['ir_sha256']==ir['sha256'] and all(cost[m]==counts[m] for m in METRICS),'native resource binding')
        require(cost['state_preparation_included'] and cost['measurement_count']==1,'wrapper accounting')
        maximum=max(maximum,close(actual,ideal,'absolute wrapper '+ir['label']))
        gate_total+=counts['size'];max_gate=max(max_gate,counts['size'])
    mutations=[]
    if mutation_checks:
        edited=copy.deepcopy(irs[0]);edited['global_phase']+=.03
        try:native(edited)
        except ValueError:mutations.append('unsigned phase edit rejected by digest')
        else:raise ValueError('phase digest edit accepted')
        edited['sha256']=sha(json.dumps({k:v for k,v in edited.items() if k!='sha256'},sort_keys=True,separators=(',',':')).encode())
        try:close(native(edited)[0],bindings[0][1],'resigned phase')
        except ValueError:mutations.append('resigned phase edit rejected by absolute operator')
        else:raise ValueError('resigned phase accepted')
        d=r['A']['rows'][0]['all_q_diagnostics'][0]
        require(budget(d['normalization'],d['operator_bias'],.05)!=d['shots_per_axis'][0]+1,'shot tamper rejection')
        mutations.append('shot increment rejected by independent Hoeffding accounting')
    return {'schema_version':1,'status':'PASS','source_commit':source,'source_blobs_checked':len(audit['source_sha256']),
            'native_ir_checked':len(irs),'conditional_cost_draws_checked':cost_draws,'mean_rows_checked':mean_rows,
            'max_absolute_operator_residual':maximum,'max_saved_mean_residual':max_mean,
            'max_hamiltonian_reconstruction_residual':max_reconstruction,'native_gate_total':gate_total,
            'max_gates_per_circuit':max_gate,'mutations_rejected':mutations,'wall_seconds':time.monotonic()-start,
            'new_scientific_conditions':0,'new_scientific_trajectory_sampling':0,'new_compilation':0,
            'saved_uniform_replay':True,'implementation':'NumPy/SciPy row updates, explicit JW and exterior-power minors; no Qiskit/project imports',
            'limitation':'local binary64 saved-data audit; not interval certificate or external scientific replication',
            'verifier_sha256':sha(Path(__file__).read_bytes())}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    require(not args.output.exists(),'Refusing existing audit output');receipt=verify(args.run,mutation_checks=True)
    args.output.write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))

