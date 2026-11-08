"""DF truncation study, original-H target, H2/H4/H6, separate from old runs.

Reuse saved chemistry, the Gaussian builder, compiler A, finite PF schedules,
directed matrix arithmetic and RPE/shot tools. No fitting or SCF is performed.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import time

import mpmath as mp
import numpy as np
from scipy.linalg import eigh, eig, expm, schur
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from .track_c_h4 import ROOT, save_json, weights
from .track_c_closed_p3_inputs import load_model
from .track_c_compiler_a import compile_reference, PROFILE
from . import track_c_compiler_a_18round as ca
from . import track_c_source_guarantee as sg
from .track_c_d_internal import schedule
from trotterlib.df_trotter.ops import U_to_qiskit_ops_jw
from trotterlib.df_hamiltonian import PhysicalSector
from .c1_stage1 import physical_sector, corrected_kappa
from trotterlib.h4_d2_pauli_control import _rotation
from .track_c_bottleneck import rigorous_shots
from .track_c_closed_p3_joint_opt_alpha import discrete_allocate
from trotterlib.rpe_four_round_phase_validation import reconstruct_rpe_phase
from qiskit.synthesis import synth_cnot_phase_aam

OUT=Path(os.environ.get('TRACK_C_DF_TRUNCATION_OUTPUT',str(ROOT/'artifacts/track_c_df_truncation/2026-11-08')))
CACHE=Path(os.environ.get('TRACK_C_DF_CACHE',str(ROOT/'.cache/df_truncation')))
INPUT_REF='5385c6ef01879ee8a70734bfc0d5ae8999dd6186'
BASE='c5a3c102199fe97f4d5c21837c059ae2684f5ef2'
METRICS=['RZ','CX','serial_depth_upper','size']
EXECUTION_SOURCE=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()


def source():
    return EXECUTION_SOURCE


def workers(count):
    # Optional per-stage coordination when another authorized stage already
    # occupies the available CPUs. Default uses the actual process affinity.
    requested=int(os.environ.get('TRACK_C_DF_WORKERS',len(os.sched_getaffinity(0))))
    if requested<1:raise ValueError('workers must be positive')
    return min(count,requested,len(os.sched_getaffinity(0)))


def git_array(path):
    n=int(path.split('/')[-2].split('-')[0][1:]);assert n in (2,6)
    local='ORIGINAL_INPUT.npz' if path.endswith('/input.npz') else 'SAVED_FULL_DF.npz'
    key='input_sha256' if local=='ORIGINAL_INPUT.npz' else 'df_sha256'
    directory=ROOT/f'artifacts/track_c_df_truncation/2026-11-08/H{n}'
    if (directory/local).exists():
        raw=(directory/local).read_bytes();expected=json.loads((directory/'INPUT.json').read_text())['provenance'][key]
        assert hashlib.sha256(raw).hexdigest()==expected
    else:raw=subprocess.check_output(['git','show',INPUT_REF+':'+path],cwd=ROOT)
    with np.load(io.BytesIO(raw),allow_pickle=False) as z: data={k:z[k] for k in z.files}
    return data,hashlib.sha256(raw).hexdigest()


def coefficients(block):
    """Exact rational polynomial of the original binary64 diagonal operands."""
    if 'one' in block:
        d=list(map(F.from_float,block['one']))
        return {():sum(d)/2,**{(i,):-v/2 for i,v in enumerate(d) if v}}
    K=block['K'];n=len(K);K=[[F.from_float(float(v)) for v in row] for row in K]
    return {(): (sum(map(sum,K))+sum(K[i][i] for i in range(n)))/4,
        **{(i,):-sum(K[i])/2 for i in range(n) if sum(K[i])},
        **{(i,j):K[i][j]/2 for i in range(n) for j in range(i+1,n) if K[i][j]}}


def h4_coefficients(b):
    if b.block_kind=='one_body':return coefficients({'one':b.diagonal_eigenvalues})
    eta=list(map(F.from_float,b.diagonal_eta));lam=F.from_float(b.lam)
    K=[[lam*a*d for d in eta] for a in eta];n=len(eta)
    return {(): (sum(map(sum,K))+sum(K[i][i] for i in range(n)))/4,
        **{(i,):-sum(K[i])/2 for i in range(n) if sum(K[i])},
        **{(i,j):K[i][j]/2 for i in range(n) for j in range(i+1,n) if K[i][j]}}


def one_particle(ops,n):
    """Gaussian single-particle action after removing its vacuum phase.

    Adjacent XX+YY is a fermionic Givens gate. No intermediate projection on a
    spin sector occurs. Every operation is applied in the full one-particle space.
    """
    a=np.eye(n,dtype=complex)
    for g,qs in ops:
        if g.name=='rz':a[qs[0]]*=np.exp(1j*float(g.params[0]))
        else:
            assert g.name=='xx_plus_yy' and abs(qs[0]-qs[1])==1
            q,r=qs;m=Operator(g).data
            # Local computational order |01>,|10> corresponds to q,r.
            a[[q,r]]=m[np.ix_([1,2],[1,2])]@a[[q,r]]
    return a


def exterior(a,occupations):
    """Number-sector Gaussian action; determinants have at most six rows."""
    return np.array([[np.linalg.det(a[np.ix_(r,c)]) for c in occupations] for r in occupations])


def build_model(n):
    if n==4:
        c=load_model(ROOT/'artifacts/track_c_closed_p3/2026-11-08')
        sector=c['sector'];bs=c['preps'][12].deterministic_blocks
        blocks=[dict(ops=b.runtime_basis_operations,coeff=h4_coefficients(b)) for b in bs]
        H=c['physical_H'];psi=c['state'];energy=c['energy'];constant=c['preps'][12].constant_coefficient
        provenance=dict(model='saved H4 DF12 original generation order',identity='acc2262de2ae74a1fc8a163b493764617ddf4e2715653b3b966801c6165bde88')
    else:
        prefix=f'artifacts/research_c_c1_h2_h6_science/2026-10-25/H{n}-run01/'
        p,sha=git_array(prefix+'input.npz');df,dfsha=git_array(prefix+'XDF_full.npz')
        sector=PhysicalSector.spin_sector(n_qubits=2*n,nelec_alpha=n//2,nelec_beta=n//2)
        # physical_sector expects Qiskit little-endian indices; DF sector stores OF.
        rev=np.array([int(f'{int(v):0{2*n}b}'[::-1],2) for v in sector.basis_indices])
        H,_=physical_sector(p['h'],p['g'],float(p['constant']),rev)
        ev,v=eigh(H);energy=float(ev[0]);psi=v[:,0]
        e,u=eigh(corrected_kappa(p['h'],p['g']));one=np.repeat(e,2);one_u=np.kron(u,np.eye(2))
        blocks=[dict(ops=tuple(U_to_qiskit_ops_jw(one_u)),coeff=coefficients({'one':one}))]
        for J,U in zip(df['J'],df['U']):
            blocks.append(dict(ops=tuple(U_to_qiskit_ops_jw(np.kron(U,np.eye(2)))),coeff=coefficients({'K':.5*np.kron(J,np.ones((2,2)))})))
        constant=float(p['constant'])
        provenance=dict(model=f'saved full XDF{len(df["J"])}; no refit',input_ref=INPUT_REF,input_path=prefix+'input.npz',input_sha256=sha,df_path=prefix+'XDF_full.npz',df_sha256=dfsha,
            new_SCF=0,new_DF_fit=0,new_original_ground_solve=1,ground_method='scipy.linalg.eigh, complete spin sector')
    modes=2*n;little=[int(f'{int(v):0{modes}b}'[::-1],2) for v in sector.basis_indices]
    occupations=[tuple(q for q in range(modes) if b>>q&1) for b in little]
    signs={s:np.array([math.prod(1-2*((b>>q)&1) for q in s) for b in little]) for b in blocks for s in b['coeff']}
    for b in blocks:
        sp=one_particle(b['ops'],modes);B=exterior(sp,occupations)
        diag=sum((float(v)*signs[s] for s,v in b['coeff'].items()),np.zeros(len(H)))
        b.update(B=B,sp=sp,diagonal=diag,matrix=(B*diag)@B.conj().T)
    model=sum((b['matrix'] for b in blocks),constant*np.eye(len(H)))
    discrepancy=float(np.linalg.norm(model-H,2))
    assert discrepancy<1e-6,(n,discrepancy)
    c=dict(n=n,modes=modes,sector=sector,occupations=occupations,signs=signs,H=H,state=psi,energy=energy,constant=constant,blocks=blocks,provenance=provenance,
        original_gap=float(eigh(H,subset_by_index=[1,1],eigvals_only=True)[0]-energy),model_residual_norm=discrepancy)
    CACHE.mkdir(parents=True,exist_ok=True)
    with (CACHE/f'H{n}.pkl').open('wb') as f:pickle.dump(c,f)
    # Reproducible public numeric/IR input, excluding private original objects.
    arrays=dict(H=H,state=psi,sector_indices=sector.basis_indices)
    manifest=[]
    for i,b in enumerate(blocks):
        arrays[f'B{i}']=b['B'];arrays[f'sp{i}']=b['sp']
        manifest.append(dict(index=i,coefficients=[dict(support=list(s),exact=str(v)) for s,v in b['coeff'].items()],operations=[dict(name=g.name,params=list(map(float,g.params)),qubits=list(qs)) for g,qs in b['ops']]))
    d=OUT/f'H{n}';d.mkdir(parents=True,exist_ok=True);np.savez_compressed(d/'INPUT_ARRAYS.npz',**arrays)
    save_json(d/'INPUT.json',dict(source=source(),n=n,modes=modes,sector_dimension=len(H),constant=constant,energy_validation_only=energy,original_gap=c['original_gap'],provenance=provenance,model_residual_norm=discrepancy,blocks=manifest))
    return c


@lru_cache(None)
def model(n):
    with (CACHE/f'H{n}.pkl').open('rb') as f:c=pickle.load(f)
    raw=c['H'];H=(raw+raw.conj().T)/2
    c={**c,'H':H,'raw_to_hermitian_fro':float(np.linalg.norm(H-raw,'fro'))}
    return c


def replay_model(n):
    """Reconstruct the public numeric/IR snapshot; no SCF, fit or ground solve."""
    directory=ROOT/f'artifacts/track_c_df_truncation/2026-11-08/H{n}'
    metadata=json.loads((directory/'INPUT.json').read_text())
    with np.load(directory/'INPUT_ARRAYS.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    modes=metadata['modes'];sector=PhysicalSector(n_qubits=modes,n_electrons=n,basis_indices=arrays['sector_indices'])
    ids=[int(f'{int(v):0{modes}b}'[::-1],2) for v in sector.basis_indices];occupations=[tuple(q for q in range(modes) if b>>q&1) for b in ids]
    blocks=[]
    for record in metadata['blocks']:
        co={tuple(v['support']):F(v['exact']) for v in record['coefficients']}
        ops=tuple((ca.gate(ca.spec(v['name'],v['params'])),tuple(v['qubits'])) for v in record['operations'])
        i=record['index'];blocks.append(dict(coeff=co,ops=ops,B=arrays[f'B{i}'],sp=arrays[f'sp{i}']))
    signs={s:np.array([math.prod(1-2*((b>>q)&1) for q in s) for b in ids]) for block in blocks for s in block['coeff']}
    for b in blocks:
        diag=sum((float(v)*signs[s] for s,v in b['coeff'].items()),np.zeros(len(ids)))
        b.update(diagonal=diag,matrix=(b['B']*diag)@b['B'].conj().T)
    return dict(n=n,modes=modes,sector=sector,occupations=occupations,signs=signs,H=arrays['H'],state=arrays['state'],energy=metadata['energy_validation_only'],constant=metadata['constant'],blocks=blocks,provenance=metadata['provenance'],original_gap=metadata['original_gap'],model_residual_norm=metadata['model_residual_norm'])


def replay_inputs():
    CACHE.mkdir(parents=True,exist_ok=True)
    for n in (2,4,6):
        target=CACHE/f'H{n}.pkl'
        if target.exists():raise FileExistsError('Use a fresh replay cache')
        c=replay_model(n)
        with target.open('xb') as f:pickle.dump(c,f)
    save_json(OUT/'INPUT_REPLAY.json',dict(source=source(),molecules=[2,4,6],new_SCF=0,new_DF_fit=0,new_ground_solve=0,source_arrays='public INPUT_ARRAYS.npz and INPUT.json, restored from the raw archive'))


def candidates(c):
    rank=len(c['blocks'])-1;out=[dict(name='full',family='Full',drop=[])]
    # Original generation-order prefix. Identity-retaining and whole-fragment
    # removal are distinct operators, with no energy-informed scalar correction.
    for k in range(rank):
        for keep_identity in (False,True):
            drop=[(i,s) for i,b in enumerate(c['blocks']) if i>k for s in b['coeff'] if not keep_identity or s]
            out.append(dict(name=f'prefix{k}_I{int(keep_identity)}',family='Truncated',drop=drop,retained_fragments=k,identity_retained=keep_identity))
    small=sorted((abs(v),i,s) for i,b in enumerate(c['blocks']) for s,v in b['coeff'].items() if s and v)
    # Nested L1 budgets tied to the specified energy tolerance; no subset fit.
    for f in (.01,.03,.1,.25,.5,1,2,4,8,16,64):
        budget=F.from_float(1.5936e-4*f);drop=[];mass=F(0)
        for a,i,s in small:
            if mass+a>budget:break
            mass+=a;drop.append((i,s))
        if drop and not any(x['drop']==drop for x in out):out.append(dict(name=f'coeff_L1_{f}',family='Truncated',drop=drop,coefficient_L1_exact=str(mass),identity_retained=True))
    return out


def kept(c,candidate):
    drop={(i,tuple(s)) for i,s in candidate['drop']};out=[]
    for i,b in enumerate(c['blocks']):
        co={s:v for s,v in b['coeff'].items() if (i,s) not in drop}
        if not co:continue
        d=sum((float(v)*c['signs'][s] for s,v in co.items()),np.zeros(len(c['H'])))
        out.append(dict(index=i,coeff=co,diagonal=d,B=b['B'],matrix=(b['B']*d)@b['B'].conj().T,ops=b['ops']))
    return out


def pilot_one(task):
    n,candidate=task;c=model(n);blocks=kept(c,candidate);H=sum((b['matrix'] for b in blocks),c['constant']*np.eye(len(c['H'])))
    e,v=eigh(H);p=abs(v.conj().T@c['state'])**2;k=int(np.argmax(p));bias=float(e[k]-c['energy'])
    z=(p[:,None]*np.exp(-1j*e[:,None]*(.02*2.**np.arange(18)))).sum(axis=0)
    phase=np.angle(z*np.exp(1j*c['energy']*.02*2.**np.arange(18)))
    residual=c['H']-H
    row=dict(candidate=candidate,source=source(),drop_operator_norm=float(np.linalg.norm(residual,2)),ground_energy_difference=float(e[0]-c['energy']),dominant_energy_bias=bias,dominant_overlap=float(p[k]),minimum_exact_radius=float(min(abs(z))),exact_signal_phase=phase.tolist(),exact_signal=[[float(a.real),float(a.imag)] for a in z],
        nonresonant_bound=abs(bias)*.02*2**17+math.asin(min(1.,(1-p[k])/p[k])) if p[k]>.5 else None,
        retained_blocks=len(blocks),retained_nonidentity_terms=sum(bool(s) for b in blocks for s in b['coeff']),Hermiticity_residual=float(np.linalg.norm(H-H.conj().T,'fro')))
    row['pilot_eligible']=row['nonresonant_bound'] is not None and row['nonresonant_bound']<.3
    save_json(OUT/f'H{n}/pilot/{candidate["name"]}.json',row)
    return row


@lru_cache(None)
def parity_network(masks,modes):
    """Same identity-ending GraySynth topology as the existing fusion emitter."""
    active=sorted({q for mask in masks for q in range(modes+1) if mask>>q&1});n=len(active)
    local=[sum(((mask>>q)&1)<<j for j,q in enumerate(active)) for mask in masks]
    q=synth_cnot_phase_aam([[(s>>i)&1 for s in local] for i in range(n)],['t']*len(local),section_size=1)
    state=[1<<i for i in range(n)];out=[];seen=[]
    for inst in q.data:
        wires=[q.find_bit(x).index for x in inst.qubits]
        if inst.operation.name=='cx':
            a,b=wires;state[b]^=state[a];out.append(('cx',active[a],active[b]))
        else:
            assert inst.operation.name=='t';mask=sum(((state[wires[0]]>>j)&1)<<x for j,x in enumerate(active))
            out.append(('rz',active[wires[0]],mask));seen.append(mask)
    assert sorted(seen)==sorted(masks) and state==[1<<i for i in range(n)]
    return out


def diagonal(q,co,t,modes,mode):
    """Exact rational angle formation, one rounding; ordinary controlled phase."""
    poly={};error=F(0);tf=F.from_float(t)
    if mode=='network':
        for s,v in co.items():
            if not s:continue
            mask=sum(1<<i for i in s);a=tf*v
            poly[mask]=poly.get(mask,F(0))+a;ctrl=mask|(1<<modes);poly[ctrl]=poly.get(ctrl,F(0))-a
        poly={s:v for s,v in poly.items() if v}
        rounded={s:float(v) for s,v in poly.items()};error+=sum((abs(v-F.from_float(rounded[s]))/2 for s,v in poly.items()),F(0))
        if poly:
            for op,a,b in parity_network(tuple(sorted(poly)),modes):
                if op=='cx':q.cx(a,b)
                else:q.rz(rounded[b],a)
    else:
        for s,v in co.items():
            if not s:continue
            a=2*tf*v;f=float(a);error+=abs(a-F.from_float(f))/2
            # Same known-Pauli central control; decompose CRZ explicitly.
            temp=QuantumCircuit(modes+1);_rotation(temp,'Z'*len(s),f,s,modes)
            q.compose(temp.decompose(gates_to_decompose=['crz']),inplace=True)
    phase=-tf*co.get((),F(0));f=float(phase);error+=abs(phase-F.from_float(f))
    if f:q.p(f,modes)
    return math.nextafter(float(error),math.inf) if error else 0.


def circuit_piece(c,blocks,seq,mode):
    q=QuantumCircuit(c['modes']+1);error=0.;lookup={b['index']:b for b in blocks}
    for _,i,t in seq:
        b=lookup[i];ops=b['ops']
        if not any(s and v for s,v in b['coeff'].items()):
            error+=diagonal(q,b['coeff'],t,c['modes'],mode)
            continue
        for g,qs in reversed(ops):q.append(g.inverse(),qs)
        error+=diagonal(q,b['coeff'],t,c['modes'],mode)
        for g,qs in ops:q.append(g,qs)
    return q,error


def metric(q):
    counts=q.count_ops();return np.array([counts.get('rz',0),counts.get('cx',0),q.depth(),q.size()],dtype=np.int64)


def price_piece(c,blocks,seq):
    variants={};circuits={};errors={}
    for mode in ('term_parity','network'):
        q,e=circuit_piece(c,blocks,seq,mode);native=compile_reference(q)
        variants[mode]=metric(native).tolist();circuits[mode]=native;errors[mode]=e
    chosen=min(variants,key=lambda k:tuple(variants[k]));return variants[chosen],errors[chosen],dict(selected=chosen,variants=variants),circuits[chosen]


def wrapper(c,t,axis):
    return wrapper_metrics(c['modes'],float(c['constant']),t,axis)


@lru_cache(None)
def wrapper_metrics(modes,constant,t,axis):
    q=QuantumCircuit(modes+1,1);a=modes;q.p(-t*constant,a)
    if axis=='Y':q.sdg(a)
    q.h(a);q.measure(a,0)
    head=QuantumCircuit(modes+1);head.h(a)
    return metric(compile_reference(head))+metric(compile_reference(q))


def det_scan(task):
    n,candidate,formula,h=task;c=model(n);blocks=kept(c,candidate)
    case=dict(kind='full_'+formula,order=[b['index'] for b in blocks]);seq=schedule(case,h)
    lookup={b['index']:b for b in blocks};d=len(c['H']);u=np.eye(d,dtype=complex)
    for _,i,t in seq:
        b=lookup[i];a=(b['B']*np.exp(-1j*t*b['diagonal']))@b['B'].conj().T;u=a@u
    u*=np.exp(-1j*h*c['constant'])
    e,v=eig(u);p=abs(v.conj().T@c['state'])**2;k=int(np.argmax(p));bias=-float(np.angle(e[k]*np.exp(1j*c['energy']*h)))/h
    pmin=float(p[k]);leak=math.asin(min(1.,(1-pmin)/pmin)) if pmin>.5 else math.pi
    viable=[m for m in range(18) if h<=.02*2**m+1e-15 and abs(bias)*.02*2**m+leak<.33]
    row=dict(n=n,candidate=candidate,formula=formula,h=h,case=case,dominant_overlap=pmin,dominant_energy_bias=bias,nonresonant_leak_bound=leak,viable_rounds=viable,source=source())
    if viable:
        first=seq[:1];last=seq[1:];middle=seq[1:-1]+[(seq[0][0],seq[0][1],2*seq[0][2])]
        groups={};errs={};vs={}
        for name,s in [('first',first),('last',last),('middle',middle)]:
            groups[name],errs[name],vs[name],_=price_piece(c,blocks,s)
        row.update(groups=groups,angle_errors=errs,variants=vs)
    save_json(OUT/f'H{n}/scan/{candidate["name"]}_{formula}_{h}.json',row)
    return row


def scan_all():
    tasks=[]
    for n in (2,4,6):
        pilot=[json.loads(p.read_text()) for p in sorted((OUT/f'H{n}/pilot').glob('*.json'))]
        selected=[r['candidate'] for r in pilot if r['pilot_eligible']]
        for c in selected:
            for formula in ('S2','Y4'):
                for h in (.0025,.005,.01,.02,.04,.08,.16,.32,.64,1.28,2.56):tasks.append((n,c,formula,h))
    with ProcessPoolExecutor(max_workers=min(len(tasks),len(os.sched_getaffinity(0)))) as pool:
        for j,f in enumerate(as_completed(pool.submit(det_scan,t) for t in tasks)):
            r=f.result()
            if j%20==0:print('SCAN',j+1,len(tasks),r['n'],r['candidate']['name'],flush=True)
    save_json(OUT/'SCAN_STATUS.json',dict(source=source(),tasks=len(tasks),h_menu=[.0025*2**j for j in range(11)],formulas=['S2','Y4'],selection='same finite h/order menu for all deterministic candidates; original order, no coefficient refit'))


def fast_fro(a):
    """Upward binary64 norm bound using an exact directed sum error bound."""
    vals=np.asarray(a).reshape(-1);total=float(np.sum(vals.real**2+vals.imag**2))
    if not math.isfinite(total):raise ArithmeticError('nonfinite norm reduction')
    mp.iv.prec=192
    underflow=(4*len(vals)+4)*sg.I(2)**-1074
    return sg.up(mp.iv.sqrt((sg.I(total)+underflow)/(1-sg.gamma(4*len(vals)+4))))


def state_norm_bounds(psi):
    mp.iv.prec=192
    square=sum((abs(sg.C(z.real,z.imag))**2 for z in psi),sg.I(0))
    return sg.up(abs(square-1)),sg.up(mp.iv.sqrt(square))


def basis_error(c,b):
    """Enclose SOURCE one-particle action, then its exterior power.

    The determinant comparison uses an explicit Leibniz polynomial as a second
    evaluator; no stability assumption about numpy.linalg.det is needed.
    """
    mp.iv.prec=192;n=c['modes'];a=[[sg.C(int(i==j)) for j in range(n)] for i in range(n)]
    for g,qs in b['ops']:
        if g.name=='rz':a[qs[0]]=[z*sg.expi(sg.I(float(g.params[0]))) for z in a[qs[0]]]
        else:
            assert g.name=='xx_plus_yy' and abs(qs[0]-qs[1])==1
            q,r=qs;m=ca.source_matrix(ca.spec(g.name,g.params));x,y=a[q],a[r]
            a[q]=[m[1][1]*v+m[1][2]*w for v,w in zip(x,y)]
            a[r]=[m[2][1]*v+m[2][2]*w for v,w in zip(x,y)]
    sp=b['sp'].copy();de=sg.up(mp.iv.sqrt(sum((abs(a[i][j]-sg.C(sp[i,j].real,sp[i,j].imag))**2 for i in range(n) for j in range(n)),sg.I(0))))
    cross=0.
    if c['n'] in (2,6):
        off=sp.copy();off[0::2,0::2]=0;off[1::2,1::2]=0;cross=fast_fro(off)
        sp[0::2,1::2]=0;sp[1::2,0::2]=0
    occ=np.array(c['occupations']);ne=c['n'];d=len(occ);leib=np.zeros((d,d),complex);absolute=np.zeros((d,d))
    # Batched six-by-six determinants are reduced to two spin determinants for
    # H6's spin-preserving target, but the generic Leibniz bound covers leakage.
    for perm in itertools.permutations(range(ne)):
        parity=(-1)**sum(perm[i]>perm[j] for i in range(ne) for j in range(i+1,ne));v=np.full((d,d),complex(parity))
        for j,k in enumerate(perm):v*=sp[occ[:,j,None],occ[None,:,k]]
        leib+=v;absolute+=abs(v)
    polynomial_round=sg.up(sg.gamma(10*ne+2*math.factorial(ne))*sg.I(fast_fro(absolute))/(1-sg.gamma(10*ne+2*math.factorial(ne))))
    # Exterior powers are restrictions of tensor powers. The telescoping norm
    # inequality includes spin leakage, without multiplying by a sector size.
    full=math.comb(n,ne);delta=sg.add(de,cross)
    lift=sg.up(ne*sg.I(delta)*(1+sg.I(delta))**(ne-1))
    exterior_error=sg.add(fast_fro(b['B']-leib),polynomial_round,lift)
    return dict(one_particle_error=de,exterior_error=exterior_error,Leibniz_rounding_upper=polynomial_round,number_sector_dimension=full,method='192-bit one-particle interval action and explicit exterior polynomial; leakage included')


def source_cert_task(task):
    n,index=task;c=model(n);b=c['blocks'][index];be=basis_error(c,b)
    save_json(OUT/f'H{n}/certificates/basis_{index}.json',be);return n,index,be


def compiler_certificates():
    cat={}
    for n in (2,4,6):
        for b in model(n)['blocks']:
            for g,_ in b['ops']:
                if g.name=='xx_plus_yy':
                    for x in (g,g.inverse()):s=ca.spec(x.name,x.params);cat[s['key']]=s
    # All Gaussian pieces have only XX+YY and native RZ/CX, optionally P.
    for family in ('h','sdg'):
        s=ca.spec(family);cat[s['key']]=s
    contexts=[ca.context_compile((mask,cat)) for mask in (1,9,56)]
    unique={r['lowering_sha256']:r for ct in contexts for r in ct['records']}
    with ProcessPoolExecutor(max_workers=min(len(unique),len(os.sched_getaffinity(0)))) as pool:checked=list(pool.map(ca.interval_check,unique.values()))
    bounds={}
    for x in checked:bounds[x['source_key']]=max(bounds.get(x['source_key'],0.),x['operator_error_upper'])
    for family in ('h','sdg'):
        s=ca.spec(family);q=QuantumCircuit(1);q.append(ca.gate(s),[0]);p=compile_reference(q)
        rec=dict(source=s,n=1,lowering=[[i.operation.name,[p.find_bit(x).index for x in i.qubits],list(map(float,i.operation.params))] for i in p.data],lowering_sha256=family)
        result=ca.interval_check(rec);checked.append(result);bounds[s['key']]=result['operator_error_upper']
    save_json(OUT/'COMPILER_INTERVALS.json',dict(source=source(),profile=PROFILE,contexts=contexts,checked=checked,bounds=bounds))
    print('COMPILER_CERTIFICATES',len(checked),flush=True)


def certificates():
    tasks=[(n,i) for n in (2,4,6) for i in range(len(model(n)['blocks']))]
    with ProcessPoolExecutor(max_workers=min(len(tasks),len(os.sched_getaffinity(0)))) as pool:
        for n,i,b in pool.map(source_cert_task,tasks):print('BASIS_CERT',n,i,b['exterior_error'],flush=True)
    compiler_certificates()


@lru_cache(None)
def ground_bounds(n):
    c=model(n);ec,vc=eigh(c['H']);hc=sg.spectral_certificate(c['H'],ec,vc)
    error=sg.add(hc['generator_error_upper'],abs(float(ec[0])-c['energy']))
    gap=sg.lower(sg.I(ec[1])-sg.I(ec[0])-2*sg.I(error));assert gap>0
    return error,gap


@lru_cache(None)
def prepared_blocks(n,candidate_json):
    return {b['index']:b for b in kept(model(n),json.loads(candidate_json))}


@lru_cache(None)
def block_spectrum(n,candidate_json,i):
    c=model(n);b=prepared_blocks(n,candidate_json)[i]
    H=sg.hermitian(b['matrix']);e,v=eigh(H)
    return H,e,v,sg.spectral_certificate(H,e,v)


@lru_cache(None)
def coefficient_diagonal_error(n,candidate_json,i):
    c=model(n);b=prepared_blocks(n,candidate_json)[i];errors=[]
    for j in range(len(c['H'])):
        exact=sum((sg.I(v.numerator)/v.denominator*int(c['signs'][s][j]) for s,v in b['coeff'].items()),sg.I(0))
        errors.append(sg.up(abs(exact-sg.I(b['diagonal'][j]))))
    return max(errors)


@lru_cache(None)
def block_unitary(n,candidate_json,i,t):
    H,e,v,sc=block_spectrum(n,candidate_json,i)
    return sg.spectral_unitary(H,e,v,t,sc)


def det_qualification(c,row,m):
    """Directed endpoint and SOURCE/native disks, original-H state and energy."""
    mp.iv.prec=192;candidate_json=json.dumps(row['candidate'],sort_keys=True);lookup=prepared_blocks(c['n'],candidate_json);seq=schedule(row['case'],row['h']);d=len(c['H']);steps=round(.02*2**m/row['h']);t=.02*2**m
    u=np.eye(d,dtype=complex);eu=0.;src=0.;native=0.;angle=0.
    bounds=json.loads((OUT/'COMPILER_INTERVALS.json').read_text())['bounds']
    for _,i,time_ in seq:
        b=lookup[i];cert=json.loads((OUT/f'H{c["n"]}/certificates/basis_{i}.json').read_text())
        # Spectral representation of a Hermitian numerical matrix; the SOURCE
        # comparison also covers exact coefficient formation and basis leakage.
        a,ea=block_unitary(c['n'],candidate_json,i,time_)
        u,eu=sg.compose(a,ea,u,eu)
        diagerr=coefficient_diagonal_error(c['n'],candidate_json,i)
        eb=cert['exterior_error'];normB=sg.up(1+sg.I(eb));maxd=max(abs(b['diagonal']))
        # B diag(d) B^dagger arithmetic and SOURCE-basis perturbation.
        gen=sg.add(sg.mul(maxd,eb,sg.add(2,eb)),sg.mul(diagerr,normB,normB),sg.up(sg.gamma(8*d+6)*d*sg.I(maxd)*sg.I(normB)**2))
        src=sg.add(src,sg.mul(abs(time_),gen))
        count=0.
        for g,_ in b['ops']:
            if g.name=='xx_plus_yy':count=sg.add(count,bounds[ca.key(g.name,g.params)],bounds[ca.key(g.inverse().name,g.inverse().params)])
        native=sg.add(native,count)
    phase=sg.expi(-sg.I(row['h'])*sg.I(c['constant']));ph=sg.centre(phase)
    u,eu=sg.compose(ph*np.eye(d),sg.up(abs(phase-sg.C(ph.real,ph.imag))),u,eu)
    power=np.eye(d,dtype=complex);ep=0.;q=steps
    while q:
        if q&1:power,ep=sg.compose(u,eu,power,ep)
        q//=2
        if q:u,eu=sg.compose(u,eu,u,eu)
    psi=c['state'];norm=sg.I(fast_fro(psi));state_norm=sg.up(abs(sg.I(float(np.vdot(psi,psi).real))-1)+sg.gamma(8*d+4)*sg.I(fast_fro(psi))**2)
    hv=c['H']@psi;res=sg.add(fast_fro(hv-c['energy']*psi),sg.up(sg.gamma(8*d+6)*(sg.I(fast_fro(c['H']))+abs(sg.I(c['energy'])))*sg.I(fast_fro(psi))))
    # Full spectral residual bounds certify the ground interval, using the
    # separately computed first gap as a locator and verified eigenbasis.
    grounderr,gap=ground_bounds(c['n'])
    stateerr=sg.up(mp.iv.sqrt(sg.I(2))*sg.I(res)/sg.I(gap))
    _,norm_upper=state_norm_bounds(psi)
    z=complex(np.vdot(psi,power@psi));evalerr=sg.add(sg.mul(ep,sg.add(1,state_norm)),sg.up(sg.gamma(16*d+8)*sg.I(d)*(1+sg.I(ep))),sg.mul(2,stateerr,max(1.,norm_upper)),state_norm)
    rotated=sg.C(z.real,z.imag)*sg.expi(sg.I(t)*sg.I(c['energy']));centre=sg.centre(rotated)
    bias=sg.phase_upper(centre,sg.add(evalerr,sg.mul(t,grounderr)))
    errors=row['angle_errors'];angle=sg.add(errors['first'],errors['last'],sg.mul(steps-1,errors['middle']))
    # Boundary-fused implementation has fewer basis pairs than this conservative
    # one-kernel bound; no negative error subtraction is needed.
    native=sg.add(sg.mul(steps,native),sg.mul(2,bounds[ca.key('h')]),bounds[ca.key('sdg')])
    extra=sg.add(sg.mul(steps,src),native,angle,sg.up(abs(sg.I(float(-t*c['constant']))+sg.I(t)*sg.I(c['constant']))))
    disk=sg.mul(2*math.sqrt(2),extra);rho=sg.lower(abs(sg.C(z.real,z.imag))-sg.I(evalerr)-sg.I(disk))
    ratio=sg.I(disk)/(abs(sg.C(z.real,z.imag))-sg.I(evalerr)) if disk<abs(z)-evalerr else None
    phaseextra=sg.up(mp.iv.atan2(ratio,mp.iv.sqrt(1-ratio**2))) if ratio is not None else None
    beta=sg.lower(sg.I('.4')-sg.I(bias)-sg.I(phaseextra)) if bias is not None and phaseextra is not None else None
    hbound=sg.up(sg.I(fast_fro(c['H'])))
    alias=bool((sg.I('.02')*sg.I(hbound)+sg.I('.4')).b<mp.iv.pi.a)
    closed=bool(alias and beta is not None and beta>0 and rho>0)
    return dict(closed=closed,rho_lower=rho,beta_stat_lower=beta,total_truncation_PF_phase_upper=bias,numerical_disk=evalerr,source_generator_disk=sg.mul(steps,src),native_disk=native,angle_disk=angle,coordinate_disk=disk,
        ground_energy_error_upper=grounderr,original_state_distance_upper=stateerr,alias=alias,signal=[z.real,z.imag],truncation_exact_phase=json.loads((OUT/f'H{c["n"]}/pilot/{row["candidate"]["name"]}.json').read_text())['exact_signal_phase'][m],
        explicit_Hermitian_target=True,
        explicit_state_norm_factor=True,
        qualification='fixed saved original Hermitian H; exact original ground input; ideal native gates; fresh independent measurements')


def screen_points(n,family):
    c=model(n);points=[]
    for path in scan_paths(n):
        r=json.loads(path.read_text())
        if r['candidate']['family']!=family or not r['viable_rounds']:continue
        for m in r['viable_rounds']:
            steps=round(.02*2**m/r['h']);g=r['groups'];v=np.array(g['first'])+np.array(g['last'])+(steps-1)*np.array(g['middle'])
            price=np.array([v+wrapper(c,.02*2**m,axis) for axis in ('X','Y')],float)
            phase=abs(r['dominant_energy_bias'])*.02*2**m+r['nonresonant_leak_bound'];rho=max(1e-6,2*r['dominant_overlap']-1)
            points.append(dict(key=[r['candidate']['name'],r['formula'],r['h'],m],signal=dict(m=m,name=r['candidate']['name'],h=r['h'],steps=steps),qualification=dict(closed=True,rho_lower=rho,beta_stat_lower=.4-phase),mean=price,row=r))
    return points


def scan_paths(n):
    original={p.name:p for p in (OUT/f'H{n}/scan').glob('*.json')}
    original.update({p.name:p for p in (OUT/f'H{n}/scan_repriced').glob('*.json')})
    return [original[k] for k in sorted(original)]


def reprice_one(task):
    n,path=task;c=model(n);row=json.loads(Path(path).read_text());blocks=kept(c,row['candidate']);seq=schedule(row['case'],row['h']);first=seq[:1];last=seq[1:];middle=seq[1:-1]+[(seq[0][0],seq[0][1],2*seq[0][2])]
    row['prior_price_source']=row['source'];row['source']=source();row['price_correction']='omit B inverse/B around identity-only diagonal blocks, same operator'
    for name,s in [('first',first),('last',last),('middle',middle)]:row['groups'][name],row['angle_errors'][name],row['variants'][name],_=price_piece(c,blocks,s)
    save_json(OUT/f'H{n}/scan_repriced/{Path(path).name}',row);return n,Path(path).name


def reprice():
    tasks=[];reused=0
    for n in (2,4,6):
        for path in (OUT/f'H{n}/scan').glob('*.json'):
            row=json.loads(path.read_text())
            dropped={(i,tuple(s)) for i,s in row['candidate']['drop']}
            identity_only=any(not any(s and v and (i,s) not in dropped for s,v in b['coeff'].items()) and any((i,s) not in dropped for s in b['coeff']) for i,b in enumerate(model(n)['blocks']))
            if row['viable_rounds'] and identity_only:
                if (OUT/f'H{n}/scan_repriced/{path.name}').exists():reused+=1
                else:tasks.append((n,str(path)))
    with ProcessPoolExecutor(max_workers=workers(len(tasks))) as pool:
        for n,name in pool.map(reprice_one,tasks):print('REPRICE',n,name,flush=True)
    save_json(OUT/'REPRICING.json',dict(source=source(),cells=len(tasks),reused=reused,change='identity-only fragments use the existing scalar implementation; old prices are retained; same chance for Full and Truncated'))


def choose(points):
    grouped=[[p for p in points if p['signal']['m']==m and p['qualification']['closed']] for m in range(18)]
    if not all(grouped):return None
    best=None
    for start in (1/720,.01):
        alpha=np.full((18,2),start);seen=set()
        while True:
            selected=[min(ps,key=lambda p:(sum(rigorous_shots(p['qualification']['rho_lower'],p['qualification']['beta_stat_lower'],a)*p['mean'][b,0] for b,a in enumerate(alpha[m])),p['key'])) for m,ps in enumerate(grouped)]
            keys=tuple(tuple(p['key']) for p in selected)
            if keys in seen:break
            seen.add(keys);alpha,ns=discrete_allocate(selected);cost=float(sum((p['mean']*nn[:,None]).sum(axis=0)[0] for p,nn in zip(selected,ns)))
            if best is None or cost<best[0]:best=(cost,selected,alpha.copy(),ns.copy())
    return best


def qualify_one(task):
    n,r,m=task;q=det_qualification(model(n),r,m);name=f'{r["candidate"]["name"]}_{r["formula"]}_{r["h"]}_{m}'
    save_json(OUT/f'H{n}/qualified/{name}.json',dict(row=r,m=m,qualification=q));return n,name,q


def qualify_shortlist(version='original'):
    tasks={}
    for n in (2,4,6):
        for family in ('Full','Truncated'):
            ps=screen_points(n,family);selected=choose(ps)
            if selected is None:continue
            # At each round take the best two cheap screening points plus the
            # joint-alpha nominee. This is a declared finite shortlist.
            for m in range(18):
                local=sorted((p for p in ps if p['signal']['m']==m),key=lambda p:sum(rigorous_shots(p['qualification']['rho_lower'],p['qualification']['beta_stat_lower'],1/720)*p['mean'][b,0] for b in (0,1)))[:2]
                for p in local+[selected[1][m]]:tasks[(n,tuple(p['key']))]=(n,p['row'],m)
    filename='SHORTLIST.json' if version=='original' else 'SHORTLIST_REPRICED.json'
    save_json(OUT/filename,dict(source=source(),keys=[dict(n=n,key=list(k)) for n,k in tasks],selection='screening only, before rigorous qualification and independent evaluation'))
    with ProcessPoolExecutor(max_workers=workers(len(tasks))) as pool:
        function=qualify_one if version=='original' else qualify_fast_one
        for j,(n,name,q) in enumerate(pool.map(function,tasks.values())):print('QUALIFY',version,j+1,len(tasks),n,name,q['closed'],flush=True)


def qualify_fast_one(task):
    n,r,m=task;name=f'{r["candidate"]["name"]}_{r["formula"]}_{r["h"]}_{m}';old=OUT/f'H{n}/qualified/{name}.json'
    if old.exists():
        item=json.loads(old.read_text());q=item['qualification'];origin='reused unchanged scientific point from original qualification'
    else:
        # Same directed formulas. Only the Frobenius upper-bound evaluator is
        # vectorized in this new worker process; historical SOURCE is untouched.
        sg.fro=fast_fro;q=det_qualification(model(n),r,m);origin='new point, same formulas with cached operands and outward vectorized norm'
    save_json(OUT/f'H{n}/qualified_repriced/{name}.json',dict(row=r,m=m,qualification=q,origin=origin,source=source()))
    return n,name,q


def freeze_deterministic():
    final_version=(OUT/'SHORTLIST_REPRICED.json').exists();shortlist='SHORTLIST_REPRICED.json' if final_version else 'SHORTLIST.json';directory='qualified_repriced' if final_version else 'qualified'
    expected=json.loads((OUT/shortlist).read_text())['keys']
    assert sum(len(list((OUT/f'H{n}/{directory}').glob('*.json'))) for n in (2,4,6))==len(expected)
    policies=[]
    for n in (2,4,6):
        ps=[];c=model(n)
        for path in sorted((OUT/f'H{n}/{directory}').glob('*.json')):
            item=json.loads(path.read_text());r=item['row'];m=item['m'];q=qualification_supplement(c,r,m,item['qualification'])
            if not q['closed']:continue
            steps=round(.02*2**m/r['h']);g=r['groups'];v=np.array(g['first'])+np.array(g['last'])+(steps-1)*np.array(g['middle'])
            price=np.array([v+wrapper(c,.02*2**m,axis) for axis in ('X','Y')],float)
            ps.append(dict(key=[r['candidate']['name'],r['formula'],r['h'],m],signal=dict(m=m,name=r['candidate']['name'],h=r['h'],steps=steps),qualification=q,mean=price,row=r))
        for family in ('Full','Truncated'):
            best=choose([p for p in ps if p['row']['candidate']['family']==family])
            if best is None:
                policies.append(dict(n=n,family=family,status='UNRESOLVED',reason='no qualified point at every round'));continue
            cost,points,alpha,ns=best
            selected=[]
            for p in points:
                p={**p,'mean':p['mean'].tolist()};selected.append(p)
            policies.append(dict(n=n,family=family,status='FROZEN',selected=selected,alpha=alpha.tolist(),trials=ns.tolist(),totals=np.sum([np.array(p['mean'])*nn[:,None] for p,nn in zip(selected,ns)],axis=(0,1)).tolist(),trials_total=int(ns.sum()),MC_SE=[0.]*4,
                selection='per-round Hamiltonian retention and h, two-start integer alpha search over qualified shortlist; finite-menu result, not a global optimum'))
    save_json(OUT/'DETERMINISTIC_POLICY.json',dict(source=source(),policies=policies,independent_evaluation='policy and alpha frozen before direct source/native state action and Schur signal check'))
    print('DETERMINISTIC_FROZEN',[(p['n'],p['family'],p.get('totals',[None])[0]) for p in policies],flush=True)


def qualification_supplement(c,row,m,q):
    """Retain raw qualification and cover diagonal norms outside a spin sector.

    Tiny Gaussian spin leakage is bounded in the complete number sector.
    Its multiplication by a diagonal operator therefore uses full coefficient
    L1, rather than the maximum diagonal seen only in the balanced-spin sector.
    """
    mp.iv.prec=192;extra=0.;blocks={b['index']:b for b in kept(c,row['candidate'])}
    if c['n'] in (2,6):
        for _,i,t in schedule(row['case'],row['h']):
            b=blocks[i];eb=json.loads((OUT/f'H{c["n"]}/certificates/basis_{i}.json').read_text())['exterior_error']
            l1=sg.up(sum((sg.I(abs(v.numerator))/v.denominator for v in b['coeff'].values()),sg.I(0)))
            difference=max(0.,sg.up(sg.I(l1)-sg.I(float(max(abs(b['diagonal']))))))
            extra=sg.add(extra,sg.mul(abs(t),difference,eb,sg.add(2,eb)))
        extra=sg.mul(round(.02*2**m/row['h']),extra)
    t=.02*2**m;steps=round(t/row['h']);angles=row['angle_errors']
    new_angle=sg.add(angles['first'],angles['last'],sg.mul(steps-1,angles['middle']))
    extra=sg.add(extra,max(0.,sg.up(sg.I(new_angle)-sg.I(q['angle_disk']))))
    timing=sg.up(abs(sg.I(t)-sg.I('.02')*(1<<m))*sg.I(fast_fro(c['H'])))
    target=0.
    if not q.get('explicit_Hermitian_target',False):
        # Historical raw qualification is retained. Transfer its approximate
        # eigenbasis/residual enclosure to the explicitly Hermitian target.
        delta=sg.up(sg.I(c['raw_to_hermitian_fro'])+sg.gamma(4)*sg.I(fast_fro(c['H'])))
        gap=sg.lower(sg.I(c['original_gap'])-4*sg.I(q['ground_energy_error_upper'])-4*sg.I(delta));assert gap>0
        target=sg.up(sg.I(t)*sg.I(delta)+8*sg.I(delta)/sg.I(gap))
    norm_transfer=0.
    if not q.get('explicit_state_norm_factor',False):
        _,norm_upper=state_norm_bounds(c['state'])
        norm_transfer=sg.mul(2,q['original_state_distance_upper'],max(0.,sg.up(sg.I(norm_upper)-1)))
        target=sg.add(target,norm_transfer)
    disk=sg.up(2*mp.iv.sqrt(sg.I(2))*(sg.I(extra)+sg.I(target))+sg.I(timing));rho=sg.lower(sg.I(q['rho_lower'])-sg.I(disk));beta=None
    if q['beta_stat_lower'] is not None and disk<q['rho_lower']:
        ratio=sg.I(disk)/sg.I(q['rho_lower']);angle=mp.iv.atan2(ratio,mp.iv.sqrt(1-ratio**2));beta=sg.lower(sg.I(q['beta_stat_lower'])-angle)
    return {**q,'closed':bool(q['closed'] and rho>0 and beta is not None and beta>0),'rho_lower':rho,'beta_stat_lower':beta,'complete_number_sector_diagonal_supplement':disk,'Hermitian_target_transfer_disk':target,'state_norm_transfer_disk':norm_transfer,'explicit_Hermitian_target':True,'explicit_state_norm_factor':True,'supplement_source':source()}


def independent_one(task):
    n,point=task;c=model(n);r=point['row'];blocks=kept(c,r['candidate']);seq=schedule(r['case'],r['h']);d=len(c['H']);lookup={b['index']:b for b in blocks}
    # Independent expm of numerical generators, rather than diagonal B phases.
    u=np.eye(d,dtype=complex)
    for _,i,t in seq:u=expm(-1j*t*sg.hermitian(lookup[i]['matrix']))@u
    u*=np.exp(-1j*r['h']*c['constant'])
    T,Z=schur(u,output='complex');steps=point['signal']['steps'];power=Z@np.linalg.matrix_power(T,steps)@Z.conj().T
    signal=complex(np.vdot(c['state'],power@c['state']));reference=complex(*point['qualification']['signal']);difference=abs(signal-reference)
    # Numerical independent check only; the success guarantee remains the
    # directed qualification, including this implementation's arithmetic disk.
    assert difference<max(1e-8,point['qualification']['numerical_disk'])
    native_error=[];source_error=[];price_checks=[]
    # One full step with both readouts. No giant late-round circuit is generated.
    initial=np.zeros(1<<(c['modes']+1),complex);ids=[sum(1<<q for q in occ) for occ in c['occupations']];initial[ids]=c['state']
    source_circuit,_=circuit_piece(c,blocks,seq,'term_parity');source_circuit.p(-r['h']*c['constant'],c['modes'])
    first=seq[:1];last=seq[1:];qc=QuantumCircuit(c['modes']+1)
    for role,part in [('first',first),('last',last)]:
        selected=r['variants'][role]['selected'];src,_=circuit_piece(c,blocks,part,selected);nat=compile_reference(src);qc.compose(nat,inplace=True)
        assert metric(nat).tolist()==r['groups'][role];price_checks.append(metric(nat).tolist())
    qc.p(-r['h']*c['constant'],c['modes'])
    for axis in ('X','Y'):
        states=[]
        for core in (source_circuit,qc):
            full=QuantumCircuit(c['modes']+1);a=c['modes'];full.h(a);full.compose(core,inplace=True)
            if axis=='Y':full.sdg(a)
            full.h(a);states.append(Statevector(initial).evolve(full).data)
        read=lambda x:float(np.vdot(x[:1<<c['modes']],x[:1<<c['modes']]).real-np.vdot(x[1<<c['modes']:],x[1<<c['modes']:]).real)
        native_error.append(abs(read(states[0])-read(states[1])))
        zstep=complex(np.vdot(c['state'],u@c['state']))
        source_error.append(abs(read(states[0])-(zstep.real if axis=='X' else zstep.imag)))
    assert max(native_error)<1e-8
    assert max(source_error)<1e-8
    row=dict(n=n,key=point['key'],source=source(),independent_signal=[signal.real,signal.imag],directed_centre_difference=difference,native_XY_difference=native_error,source_XY_sector_difference=source_error,price_recompile=price_checks,
        exact_original_energy_not_passed_to_decoder=True,independent_method='expm + complex Schur; native full statevector action, no intermediate sector projection')
    save_json(OUT/f'H{n}/independent/{r["candidate"]["name"]}_{r["formula"]}_{r["h"]}_{point["signal"]["m"]}.json',row);return row


def evaluate_deterministic():
    policy_path=OUT/'DETERMINISTIC_POLICY.json';before=hashlib.sha256(policy_path.read_bytes()).hexdigest();policy=json.loads(policy_path.read_text());tasks={}
    for p in policy['policies']:
        if p['status']!='FROZEN':continue
        for point in p['selected']:tasks[(p['n'],tuple(point['key']))]=(p['n'],point)
    checks=[]
    with ProcessPoolExecutor(max_workers=min(len(tasks),len(os.sched_getaffinity(0)))) as pool:
        for row in pool.map(independent_one,tasks.values()):checks.append(row);print('INDEPENDENT',row['n'],row['key'],flush=True)
    lookup={(x['n'],tuple(x['key'])):x for x in checks}
    for p in policy['policies']:
        if p['status']!='FROZEN':continue
        phases=[float(np.angle(complex(*lookup[(p['n'],tuple(x['key']))]['independent_signal']))) for x in p['selected']]
        decoded=reconstruct_rpe_phase(phases,[1<<m for m in range(18)])
        energy=-decoded['final_phase_estimate']/.02
        p['mean_decoder']=dict(decoded=decoded,energy=energy,absolute_error_validation_only=abs(energy-model(p['n'])['energy']))
        assert p['mean_decoder']['absolute_error_validation_only']<1.5936e-4
        p['status']='QUALIFIED_INDEPENDENT_CHECK_COMPLETE'
    assert hashlib.sha256(policy_path.read_bytes()).hexdigest()==before
    save_json(OUT/'DETERMINISTIC_FINAL.json',dict(source=source(),policy_sha256=before,policies=policy['policies'],checks=len(checks),metrics=METRICS,energy_tolerance=1.5936e-4,failure_upper=.05,phase_budget=.4,energy_bound=.4/(.02*2**17),scope='fixed saved original H, exact original ground, ideal native gates, independent measurement outcomes; no hardware shots'))
    print('DETERMINISTIC_FINAL',flush=True)


def run(stage):
    if (OUT/'STUDY_STOP.json').exists():raise RuntimeError('Study stopped; use an explicitly authorized fresh output')
    if stage.startswith('pr-'):
        from . import track_c_df_truncation_pr as pr
        return {'bch':pr.bch,'probes':pr.probes,'isometries':pr.build_isometries,'qualify':pr.shortlist,'training':lambda:pr.price_cells('training'),'freeze':pr.freeze,'holdout':lambda:pr.price_cells('holdout'),'finish':pr.finish}[stage[3:]]()
    if stage=='inputs':
        OUT.mkdir(parents=True,exist_ok=True)
        with ProcessPoolExecutor(max_workers=3) as pool:
            for c in pool.map(build_model,(2,4,6)):print('INPUT',c['n'],c['model_residual_norm'],flush=True)
    elif stage=='inputs-replay':replay_inputs()
    elif stage=='scan':scan_all()
    elif stage=='certificates':certificates()
    elif stage=='qualify':qualify_shortlist()
    elif stage=='reprice':reprice()
    elif stage=='qualify-repriced':qualify_shortlist('repriced')
    elif stage=='freeze':freeze_deterministic()
    elif stage=='independent':evaluate_deterministic()
    elif stage=='pilot':
        tasks=[(n,x) for n in (2,4,6) for x in candidates(model(n))];rows=[]
        for n in (2,4,6):save_json(OUT/f'H{n}/MENU.json',candidates(model(n)))
        with ProcessPoolExecutor(max_workers=min(len(tasks),len(os.sched_getaffinity(0)))) as pool:
            for f in as_completed(pool.submit(pilot_one,t) for t in tasks):rows.append(f.result())
        save_json(OUT/'PILOT.json',dict(source=source(),rows=rows,selection='original-H state direct signals; nonresonant diagnostic is screening, not final native qualification'))
        print('PILOT_COMPLETE',len(rows),flush=True)
    else:raise ValueError(stage)


if __name__=='__main__':
    import sys
    run(sys.argv[1])
