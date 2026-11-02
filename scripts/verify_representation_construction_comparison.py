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

def a_trajectory(sample,con,factors,time):
    n=len(factors[0]);eye=np.eye(2**n);v=fock(matrix(con['metadata']['orbital_frame'])) if 'orbital_frame' in con['metadata'] else eye
    def primitive(cid):
        if not cid.startswith('df'):return pauli(cid)
        prefix,label=cid.split(':');_,vec=np.linalg.eigh(factors[int(prefix[2:])])
        val=np.linalg.eigvalsh(factors[int(prefix[2:])]);vec=vec[:,np.argsort(np.abs(val))[::-1]]
        basis=fock(vec);return basis@pauli(label)@basis.conj().T
    dt=time/sample['q'];blocks=[expm(-.5j*dt*matrix(b['matrix'])) for b in con['deterministic_blocks']]
    out=eye.copy()
    for k in range(sample['q']):
        for b in blocks:out=b@out
        if sample['events']:out=event_operator(sample['events'][k],primitive,n)@out
        for b in reversed(blocks):out=b@out
    return np.exp(-1j*time*con['identity'])*v@out@v.conj().T

def c_unitary(row):
    n=len(row['fields']);a=sum(c*single(n,i,'Z') for i,c in enumerate(row['fields']))
    for i,j,c in row['edges']:a+=c*single(n,i,'Z')@single(n,j,'Z')
    xs=[row['alpha']*single(n,i,'X') for i in range(n)];dt=row['delta']
    if row['method']=='ordinary_first':short=expm(-1j*dt*a)@expm(-1j*dt*sum(xs))
    elif row['method']=='symmetric_s2':short=expm(-.5j*dt*a)@expm(-1j*dt*sum(xs))@expm(-.5j*dt*a)
    else:
        require(row['method']=='thrift','C method');short=expm(-1j*dt*(a+xs[0]))
        for x in xs[1:]:short=short@expm(1j*dt*a)@expm(-1j*dt*(a+x))
    return np.linalg.matrix_power(short,row['q']),a+sum(xs)

def budget(gamma,bias,eps):
    margin=eps/math.sqrt(2)-bias
    return None if margin<=0 else math.ceil(2*gamma**2/margin**2*math.log(4/.05))

def resources(values,samples,gamma,bias):
    axes={a:[s['cost'] for s in samples if s['axis']==a] for a in ('X','Y')}
    require(len(axes['X'])==len(axes['Y'])>0,'paired sample coverage')
    require([v['epsilon_complex'] for v in values]==[.05,.02],'precision coverage')
    def se(v):return float(np.std(v,ddof=1)/math.sqrt(len(v))) if len(v)>1 else 0.
    for row in values:
        close(row['normalization'],gamma,'resource Gamma');close(row['bias'],bias,'resource bias')
        eps=row['epsilon_complex'];shots=budget(gamma,bias,eps)
        require(row['shots_per_axis']==shots,'shot budget')
        for m in METRICS:
            arrays={a:[x[m] for x in axes[a]] for a in axes}
            for a,v in arrays.items():
                close(row['per_axis_cost'][a][m]['mean'],float(np.mean(v)),'axis mean')
                close(row['per_axis_cost'][a][m]['se'],se(v),'axis SE')
            if shots is None:require(row['expected_work'] is None and row['expected_work_se'] is None,'excluded work')
            else:
                close(row['expected_work'][m],shots*sum(float(np.mean(v)) for v in arrays.values()),'total work')
                close(row['expected_work_se'][m],shots*se(np.array(arrays['X'])+arrays['Y']),'paired work SE')

def finite_mean(h,identity,time,q):
    x=-1j*time/q*(h-identity*np.eye(len(h)))
    return np.exp(-1j*time*identity)*np.linalg.matrix_power(np.eye(len(h))+x+x@x/2+x@x@x/6,q)

def gamma(lam,time,q):
    tau=lam*time/q
    return (math.sqrt(1+tau*tau)+tau*tau/2*math.sqrt(1+(tau/3)**2))**q

def verify(run,mutation_checks=False):
    start=time.monotonic();r=load(run/'result.json');irs=load(run/'native_ir.json');audit=load(run/'run_audit.json')
    for name,info in audit['outputs'].items():
        raw=(run/name).read_bytes();require(len(raw)==info['bytes'] and sha(raw)==info['sha256'],'output identity '+name)
    source=audit['source_commit'];require(r['source_commit']==source,'source identity')
    for path,digest in audit['source_sha256'].items():
        blob=subprocess.check_output(['git','show',source+':'+path],cwd=ROOT)
        require(sha(blob)==digest,'source blob '+path)
    require(r['mandatory_stop'] and r['next_stage_authorized'] is False and r['central_hypothesis_adopted'] is None,'STOP')
    for key in ('quantum_shots_executed','molecular_loads','gpu_calls','ground_state_solves'):require(r[key]==0,key)
    a=r['A'];factors=[matrix(g) for g in a['input_factors']];cons={c['name']:c for c in a['candidates']}
    require(len(cons)==6 and len(a['rows'])==18 and len(r['C']['rows'])==48 and len(r['B']['rows'])==2,'row coverage')
    bindings={};max_mean_error=0.;max_bias_error=0.
    for row in a['rows']:
        con=cons[row['candidate']];dt=row['delta'];require(dt==a['time']/row['q'],'A delta')
        tail=matrix(con['tail_matrix']);x=-1j*dt*tail;eye=np.eye(len(tail))
        short=eye+x+x@x/2+x@x@x/6
        blocks=[expm(-.5j*dt*matrix(b['matrix'])) for b in con['deterministic_blocks']]
        left=eye.copy();right=eye.copy()
        for b in blocks:left=b@left
        for b in reversed(blocks):right=b@right
        v=fock(matrix(con['metadata']['orbital_frame'])) if 'orbital_frame' in con['metadata'] else eye
        mean=np.exp(-1j*a['time']*con['identity'])*v@np.linalg.matrix_power(right@short@left,row['q'])@v.conj().T
        max_mean_error=max(max_mean_error,close(mean,matrix(row['corrected_mean']),'A finite mean'))
        b=float(np.linalg.norm(mean-expm(-1j*a['time']*matrix(a['hamiltonian'])),2))
        max_bias_error=max(max_bias_error,close(b,row['operator_bias'],'A bias'))
        g=gamma(con['tail_lambda'],a['time'],row['q']);close(g,row['normalization'],'A normalization')
        samples=[s for s in a['sampled_wrappers'] if s['candidate']==row['candidate'] and s['q']==row['q']]
        require(len(samples)==(2 if con['tail_components']==0 else 16),'A sample coverage')
        resources(row['precision_resources'],samples,row['normalization'],row['operator_bias'])
        prep=single(a['modes'],0,'X')
        for sample in samples:
            u=a_trajectory(sample,con,factors,a['time']);bindings[sample['cost']['ir_id']]=(sample['cost'],wrapped(u,sample['axis'],prep))
    for row in r['C']['rows']:
        u,h=c_unitary(row);b=float(np.linalg.norm(u-expm(-1j*row['time']*h),2))
        max_bias_error=max(max_bias_error,close(b,row['operator_bias'],'C bias'))
        resources(row['precision_resources'],row['costs'],1.,row['operator_bias'])
        prep=np.ones((1,1),complex)
        for _ in row['fields']:prep=np.kron(prep,H)
        for sample in row['costs']:bindings[sample['cost']['ir_id']]=(sample['cost'],wrapped(u,sample['axis'],prep))
    bm=r['B']['metadata'];bf=fock(matrix(bm['unitary_completion']));btime=r['B']['time']
    hp=matrix(bm['physical_hamiltonian']);s=np.diag([1.]*8+[-1.]*8)
    close(matrix(bm['orbital_isometry'])@matrix(bm['orbital_isometry']).conj().T,np.eye(3),'B isometry')
    enlarged=bf@sum(c*pauli(p) for p,c in bm['diagonal_paulis'].items())@bf.conj().T
    close(enlarged,matrix(bm['enlarged_hamiltonian']),'B enlarged encoding');close(enlarged[:8,:8],hp,'B projection')
    close((enlarged+s@enlarged@s)/2,matrix(bm['symmetric_generator']),'B symmetrization')
    for row in r['B']['rows']:
        reflected=row['candidate']=='enlarged_generator_reflection';h=matrix(bm['symmetric_generator']) if reflected else hp;n=4 if reflected else 3
        def primitive(cid):
            if not reflected:return pauli(cid)
            label,bit=cid.split(':');op=bf@pauli(label)@bf.conj().T
            return s@op@s if bit=='1' else op
        require([d['q'] for d in row['all_q_diagnostics']]==[1,2,4],'B q coverage')
        for d in row['all_q_diagnostics']:
            mean=finite_mean(h,row['identity'],btime,d['q'])[:8,:8]
            max_mean_error=max(max_mean_error,close(mean,matrix(d['corrected_mean_physical_block']),'B finite mean'))
            b=float(np.linalg.norm(mean-expm(-1j*btime*hp),2))
            max_bias_error=max(max_bias_error,close(b,d['physical_operator_bias'],'B bias'))
            close(gamma(row['tail_lambda'],btime,d['q']),d['normalization'],'B normalization')
            require([budget(d['normalization'],d['physical_operator_bias'],eps) for eps in (.05,.02)]==d['shots_per_axis'],'B diagnostic shots')
        samples=[x for x in r['B']['sampled_wrappers'] if x['candidate']==row['candidate']]
        d=next(x for x in row['all_q_diagnostics'] if x['q']==row['cost_q'])
        resources(row['precision_resources'],samples,d['normalization'],d['physical_operator_bias'])
        prep=single(n,0,'X')@single(n,1,'X')
        for sample in samples:
            u=np.eye(2**n,dtype=complex)
            for e in sample['events']:u=event_operator(e,primitive,n)@u
            u*=np.exp(-1j*btime*row['identity'])
            bindings[sample['cost']['ir_id']]=(sample['cost'],wrapped(u,sample['axis'],prep))
    require(len(irs)==r['compiled_circuits']==374 and set(bindings)==set(range(374)),'complete IR coverage')
    maximum=0.;gate_total=0;max_gate=0
    for index,ir in enumerate(irs):
        require(ir['id']==index,'IR ordering');u,counts=native(ir);cost,ideal=bindings[index]
        require(cost['ir_sha256']==ir['sha256'] and cost['qubits']==ir['qubits'],'IR binding')
        require(all(cost[m]==counts[m] for m in METRICS),'saved native count/depth')
        require(cost['measurement_count']==1 and cost['state_preparation_included'],'wrapper accounting')
        maximum=max(maximum,close(u,ideal,'absolute wrapper '+ir['label']))
        gate_total+=counts['size'];max_gate=max(max_gate,counts['size'])
    mutations=[]
    if mutation_checks:
        changed=copy.deepcopy(irs[0]);changed['global_phase']+=.03
        try:native(changed)
        except ValueError:mutations.append('unsigned phase edit rejected by digest')
        else:raise ValueError('digest mutation accepted')
        changed['sha256']=sha(json.dumps({k:v for k,v in changed.items() if k!='sha256'},sort_keys=True,separators=(',',':')).encode())
        try:close(native(changed)[0],bindings[0][1],'resigned phase mutation')
        except ValueError:mutations.append('resigned phase edit rejected by absolute operator')
        else:raise ValueError('phase mutation accepted')
        values=copy.deepcopy(a['rows'][0]['precision_resources']);values[0]['shots_per_axis']+=1
        samples=[x for x in a['sampled_wrappers'] if x['candidate']==a['rows'][0]['candidate'] and x['q']==1]
        try:resources(values,samples,a['rows'][0]['normalization'],a['rows'][0]['operator_bias'])
        except ValueError:mutations.append('shot edit rejected by independent accounting')
        else:raise ValueError('shot mutation accepted')
    return {'schema_version':1,'status':'PASS','source_commit':source,'source_blobs_checked':len(audit['source_sha256']),
            'output_identity_checked':audit['outputs'],'a_rows':18,'c_rows':48,'b_mean_rows':6,'native_ir_checked':374,
            'max_absolute_operator_residual':maximum,'max_saved_mean_residual':max_mean_error,
            'max_saved_bias_difference':max_bias_error,'native_gate_total':gate_total,'max_gates_per_circuit':max_gate,
            'mutations_rejected':mutations,'wall_seconds':time.monotonic()-start,'new_sampling':0,'new_compilation':0,
            'new_scientific_conditions':0,'implementation':'NumPy/SciPy row updates and exterior-power minors; no Qiskit/project imports',
            'limitation':'local saved-data verification; binary64 diagnostics, not interval certificate or external scientific replication',
            'verifier_sha256':sha(Path(__file__).read_bytes())}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();require(not args.output.exists(),'Refusing existing audit output')
    receipt=verify(args.run,mutation_checks=True)
    args.output.write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))
