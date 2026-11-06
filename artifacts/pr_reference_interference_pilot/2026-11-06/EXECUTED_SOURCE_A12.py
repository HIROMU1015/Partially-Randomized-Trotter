"""Authorized finite reference-interference pilot; old runners are never called.

Only the gadget and Simon--Love lift are new. DF primitives, finite events,
conditional sampling, source inputs and compiler A are reused unchanged.
"""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from importlib.metadata import version
import hashlib
import io
import json
import lzma
import math
import os
from pathlib import Path
import pickle
import subprocess
import tarfile
import time

import numpy as np
from qiskit import QuantumCircuit, qpy
from qiskit.circuit.library import XXPlusYYGate
from qiskit.quantum_info import Operator, Statevector

from .track_c_h4 import ROOT, common, save_json, complex_json, git_npz, DFREF, DFPATH, BASE
from .track_c_compiler_a import compile_reference, PROFILE
from .track_c_h4_price import conditional_flags, d_part
from trotterlib.df_partial_s2 import QiskitDFPartialS2CircuitBuilder
from trotterlib.df_trotter.ops import one_body_diagonal_primitives, df_squared_diagonal_primitives
from trotterlib.df_rte_circuit import DFRTEEventSequenceCircuitRequest
from trotterlib.df_rte_qiskit import QiskitDFRTEEventCircuitBuilder
from trotterlib.rte import _make_event, finite_rte_distribution

FIXED = '602ca4b67b1037e221e910a2733444630c5add62'
OUT = ROOT/'artifacts/pr_reference_interference_pilot/2026-11-06'
CACHE = ROOT.parent/'track-c-h4-parallel-pilots-20261106/artifacts/track_c_h4_pilots/2026-11-06/run01/common.pkl'
TOL = 1e-9
METRICS = ('rz', 'cx', 'depth', 'size')
PRICE_Q = (1, 2, 8)
PRICE_N = 8


def stored(path):
    return subprocess.check_output(['git','show',FIXED+':'+path],cwd=ROOT)


def init_inputs():
    """Bind cached preparation to immutable published arrays, no new solve."""
    OUT.mkdir(parents=True,exist_ok=True)
    saved=OUT/'common.pkl.xz'
    raw=lzma.decompress(saved.read_bytes()) if saved.exists() else CACHE.read_bytes()
    if not saved.exists():saved.write_bytes(lzma.compress(raw))
    # Old code's common() is a read-only loader. Point it at our own snapshot.
    snapshot=OUT/'common.pkl'
    if not snapshot.exists():snapshot.write_bytes(raw)
    os.environ['TRACK_C_H4_COMMON_CACHE']=str(snapshot)
    c=common()
    archive=stored('artifacts/track_c_h4_pilots/2026-11-06/run01/common_numerical_arrays.tar.xz')
    with tarfile.open(fileobj=io.BytesIO(archive),mode='r:xz') as t:
        data=t.extractfile('common_arrays.npz').read()
    with np.load(io.BytesIO(data),allow_pickle=False) as z:
        tests={'state':c['state'],'basis_indices':c['sector'].basis_indices,
               'original_H':c['physical_H'],'blocks':np.stack(c['block_matrices']),
               'tail':c['tails'][3]}
        assert all(np.array_equal(value,z[key]) for key,value in tests.items())
    df,dfhash=git_npz(DFREF,DFPATH)
    assert np.array_equal(c['ham'].one_body,df['one_body'])
    assert np.array_equal(c['ham'].lambdas,df['lambdas'])
    assert np.array_equal(np.stack(c['ham'].g_matrices),df['g_matrices'])
    assert c['ham'].constant==df['constant'].item()
    ref,refhash=git_npz(BASE,'artifacts/track_c_h4_input_check/2026-11-06/run02/ground_reference.npz')
    assert np.array_equal(c['state'],ref['state'])
    sourcefiles=[Path(__file__),ROOT/'src/trottertracks/track_c_compiler_a.py',
        ROOT/'src/trottertracks/track_c_h4.py',ROOT/'src/trottertracks/track_c_h4_price.py',
        ROOT/'src/trotterlib/rte.py',ROOT/'src/trotterlib/df_rte_qiskit.py',
        ROOT/'src/trotterlib/df_partial_s2.py',ROOT/'src/trotterlib/df_trotter/ops.py']
    save_json(OUT/'INPUTS.json',dict(scientific_source=FIXED,
        common_pickle_sha256=hashlib.sha256(raw).hexdigest(),
        published_array_archive_sha256=hashlib.sha256(archive).hexdigest(),
        cached_vs_published_arrays_equal=list(tests),df_sha256=dfhash,ground_reference_sha256=refhash,
        new_SCF=0,new_DF_fit=0,new_ground_solve=0,
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sourcefiles}))
    save_json(OUT/'ENVIRONMENT.json',dict(host=os.uname().nodename,
        packages={p:version(p) for p in ('qiskit','numpy','scipy')},
        affinity=len(os.sched_getaffinity(0)),profile=PROFILE,
        memory_available_observation=Path('/proc/meminfo').read_text(),
        cpu_pressure=Path('/proc/pressure/cpu').read_text(),
        memory_pressure=Path('/proc/pressure/memory').read_text(),
        runtime=os.sys.executable,shared_packages_modified=False))


def vacuum_phase(qc):
    """Analytic all-zero action of this physical SOURCE family, not E0."""
    phase=float(qc.global_phase)
    for inst in qc.data:
        op=inst.operation
        if op.name in ('rz','rzz'):phase-=float(op.params[0])/2
        elif op.name in ('xx_plus_yy','z','p','id'):pass
        else:raise ValueError('Unqualified vacuum factor: '+op.name)
    return phase


def vacuum_check(qc):
    phase=vacuum_phase(qc)
    v=Statevector.from_int(0,2**qc.num_qubits).evolve(qc).data
    ideal=np.zeros_like(v);ideal[0]=np.exp(1j*phase)
    return dict(phase=phase,leakage_norm=float(np.linalg.norm(v[1:])),
                phase_action_error=float(np.linalg.norm(v-ideal)))


def concatenate(parts,n):
    q=QuantumCircuit(n)
    for p in parts:q.compose(p,qubits=range(n),inplace=True)
    return q


def reference_core(f,g,phase):
    """Fig.1: TWO N-CSWAP cascades, F^adjoint and G in separate lanes."""
    n=f.num_qubits
    if g.num_qubits!=n:raise ValueError('lane widths')
    q=QuantumCircuit(2*n+1);a=2*n
    for j in range(n):q.cswap(a,j,n+j)
    q.compose(f.inverse(),qubits=range(n),inplace=True)
    q.compose(g,qubits=range(n,2*n),inplace=True)
    q.p(phase,a)
    for j in range(n):q.cswap(a,j,n+j)
    return q


def ordinary_core(v):
    n=v.num_qubits;q=QuantumCircuit(n+1)
    q.append(v.to_gate().control(),[n,*range(n)])
    return q


def wrapper(core,axis):
    n=core.num_qubits;a=n-1;q=QuantumCircuit(n,1)
    q.h(a);q.compose(core,inplace=True)
    if axis=='Y':q.sdg(a)
    q.h(a);q.measure(a,0)
    return q


def coherence(core,psi):
    """Ancilla last, target first; all other registers start at zero."""
    half=2**(core.num_qubits-1)
    v=np.zeros(2*half,complex);v[:len(psi)]=psi/math.sqrt(2);v[half:half+len(psi)]=psi/math.sqrt(2)
    out=Statevector(v).evolve(core).data
    return 2*np.vdot(out[:half],out[half:])


def axis_signals(core,psi):
    out={}
    for axis in ('X','Y'):
        q=wrapper(core,axis).remove_final_measurements(inplace=False)
        v=np.zeros(2**core.num_qubits,complex);v[:len(psi)]=psi
        v=Statevector(v).evolve(q).data;half=len(v)//2
        out[axis]=float(np.vdot(v[:half],v[:half]).real-np.vdot(v[half:],v[half:]).real)
    return complex(out['X'],out['Y'])


@lru_cache(None)
def deterministic(time,reverse=False,controlled=False):
    return d_part(3,time,reverse,False,controlled)


@lru_cache(None)
def directional_right(time):
    """Simon--Love half lift: control0 L^adjoint, control1 reversed L."""
    p=common()['preps'][3];builder=QiskitDFPartialS2CircuitBuilder()
    q=QuantumCircuit(9);a=8
    for block in reversed(p.deterministic_blocks):
        builder._append_basis(q,block,inverse=True)
        prim=(one_body_diagonal_primitives(np.asarray(block.diagonal_eigenvalues),time)
              if block.block_kind=='one_body' else
              df_squared_diagonal_primitives(np.asarray(block.diagonal_eta),block.lam,time))
        for left,right,angle in prim.rzz:
            q.cx(a,left);q.rzz(-float(angle),left,right);q.cx(a,left)
        for j,angle in prim.rz:
            q.cx(a,j);q.rz(-float(angle),j);q.cx(a,j)
        phi=float(prim.global_phase)
        q.global_phase-=phi
        if phi:q.p(2*phi,a)
        builder._append_basis(q,block,inverse=False)
    return q


def trajectory(q,rare,j):
    """Unchanged canonical conditional law, new fixed cohort; no price calls."""
    p=common()['preps'][3];n=8;dist=finite_rte_distribution(p.exact_rte_lambda_r*.02,2)
    seed=int.from_bytes(hashlib.sha256(f'newA-ref-pilot:20261106:{q}:{rare}:{j}'.encode()).digest()[:8],'big')
    rng=np.random.default_rng(seed)
    flags=conditional_flags([dist.order_probabilities[1]]*q,rare,rng)
    components=p.rte_preparation.symbolic_tail.components
    probs=np.array([x.probability for x in components]);probs/=probs.sum()
    factory=QiskitDFRTEEventCircuitBuilder(basis_registry=p.rte_preparation.basis_registry)
    scalar=-.02*(p.constant_coefficient+p.extracted_identity_coefficient)
    full=[];direct=[];sl=[];events=[]
    for flag in flags:
        idx=rng.choice(len(components),size=3 if flag else 1,p=probs)
        event=_make_event(idx,components,dist,1 if flag else 0);events.append(event.to_dict())
        tail={}
        for ctrl in (False,True):
            req=DFRTEEventSequenceCircuitRequest(events=(event,),
                component_specs=p.rte_preparation.component_specs,controlled=ctrl,
                ancilla_qubit=8 if ctrl else None,tail_id=p.rte_preparation.symbolic_tail.tail_id,
                tail_hash=p.rte_preparation.symbolic_tail.tail_hash,occurrence_rte_steps=1)
            tail[ctrl]=factory.build_sequence(req).circuit
        l=deterministic(.01);rr=deterministic(.01,True)
        u=l.copy();u.global_phase+=scalar
        full.extend((u,tail[False],rr))
        d=deterministic(.01,controlled=True).copy();d.p(scalar,8)
        direct.extend((d,tail[True],deterministic(.01,True,True)))
        lift=QuantumCircuit(9);lift.compose(l,qubits=range(8),inplace=True);lift.p(scalar,8)
        sl.extend((lift,tail[True],directional_right(.01)))
    # Fixed temporal split, never optimized after observing prices.
    boundary=2 if q==1 else 3*(q//2)
    g=concatenate(full[:boundary],8);f=concatenate(full[boundary:],8)
    probability=-math.expm1(q*math.log1p(-dist.order_probabilities[1]))
    return dict(F=f,G=g,V=g.compose(f),direct=concatenate(direct,9),
        symmetric=concatenate(sl,9),phase=vacuum_phase(g)+vacuum_phase(f),
        events=events,seed=seed,probability=probability if rare else 1-probability)


def stage_a1():
    c=common();p=c['preps'][3];rows=[];families=set()
    for k,block in enumerate(p.deterministic_blocks):
        q=QuantumCircuit(8);QiskitDFPartialS2CircuitBuilder._append_basis(q,block,inverse=False)
        families.update(x.operation.name for x in q.data)
        rows.append(dict(kind='D_basis',index=k,**vacuum_check(q)))
    reg=p.rte_preparation.basis_registry
    for bid,definition in reg._definitions.items():
        q=QuantumCircuit(8)
        for gate,wires in definition.runtime_operations:q.append(gate,list(wires))
        families.update(x.operation.name for x in q.data)
        rows.append(dict(kind='R_basis',basis_id=bid,**vacuum_check(q)))
    for t in (.01,-.01):
        for reverse in (False,True):
            rows.append(dict(kind='D_signed',time=t,reverse=reverse,**vacuum_check(deterministic(t,reverse))))
    for rare in (False,True):
        tr=trajectory(1,rare,0)
        rows.append(dict(kind='complete_trajectory',rare=rare,seed=tr['seed'],
                         event=tr['events'],**vacuum_check(tr['V'])))
    result=dict(status='PASS',basis_families=sorted(families),checks=rows,
        analytic_scope='RZ and XXPlusYY basis; signed conjugated Z/ZZ and their rotations; scalar phases; products/inverses',
        independent_branch_rule='Primary fixed Track C input uses diag(I,V). No two-branch law is silently substituted.',
        phase_uses_E0_gap_or_ground_vector=False,
        max_phase_action_error=max(x['phase_action_error'] for x in rows),
        all_passed=all(x['phase_action_error']<TOL for x in rows))
    save_json(OUT/'A1_REFERENCE_PHASE.json',result)
    if not result['all_passed']:raise ArithmeticError('A1 reference condition')
    return result


def stage_a2_synthetic():
    rows=[];psi=np.array([1,2+1j,-1+.3j,.4j],complex);psi/=np.linalg.norm(psi)
    for sign in (1,-1):
        for scalar in (.173,math.pi):
            f=QuantumCircuit(2,global_phase=sign*scalar);f.rz(sign*.31,0);f.append(XXPlusYYGate(sign*.47,.12),[0,1])
            g=QuantumCircuit(2,global_phase=-.23);g.rz(-sign*.27,1);g.z(0);g.append(XXPlusYYGate(-.19,-.33),[0,1])
            v=g.compose(f);phi=vacuum_phase(v);core=reference_core(f,g,phi)
            a=Operator(f.inverse()).data;b=Operator(g).data
            expected=np.zeros((32,32),complex);expected[:16,:16]=np.kron(b,a)
            expected[16:,16:]=np.exp(1j*phi)*np.kron(a,b)
            u=Operator(core).data;native=compile_reference(core);un=Operator(native).data
            phase=np.vdot(u,un);phase/=abs(phase)
            z=np.vdot(psi,Operator(v).data@psi)
            row=dict(sign=sign,scalar=scalar,phi=phi,
                noncommutator_norm=float(np.linalg.norm(Operator(f).data@b-b@Operator(f).data)),
                target_eigen_residual=float(np.linalg.norm(Operator(f).data@psi-np.vdot(psi,Operator(f).data@psi)*psi)),
                full_operator_error=float(np.linalg.norm(u-expected)),
                compiled_full_operator_error_one_common_phase=float(np.linalg.norm(u-un/phase)),
                source_xy_error=float(abs(axis_signals(core,psi)-z)),
                native_xy_error=float(abs(axis_signals(native,psi)-z)),
                omitted_phase_signal_error=float(abs(coherence(reference_core(f,g,0.),psi)-z)),
                actual_inverse_error=float(np.linalg.norm(Operator(f.inverse()).data-Operator(f).data.conj().T)))
            rows.append(row)
    # Independent random branches stay independent; F=V0^dagger,G=V1.
    r=np.random.default_rng(20261106)
    branches=[]
    for _ in range(2):
        c=QuantumCircuit(2,global_phase=float(r.uniform(-1,1)))
        c.rz(float(r.uniform(-1,1)),0);c.append(XXPlusYYGate(float(r.uniform(-1,1)),.2),[0,1]);branches.append(c)
    v0,v1=branches;v=v1.compose(v0.inverse())
    z=np.vdot(psi,Operator(v0).data.conj().T@Operator(v1).data@psi)
    core=reference_core(v0.inverse(),v1,vacuum_phase(v1)-vacuum_phase(v0))
    joint=dict(branches_different=branches[0]!=branches[1],same_joint_realization=True,
               phase_difference=vacuum_phase(v1)-vacuum_phase(v0),
               source_xy_error=float(abs(axis_signals(core,psi)-z)),
               native_xy_error=float(abs(axis_signals(compile_reference(core),psi)-z)))
    f=QuantumCircuit(1);g=QuantumCircuit(1);g.rx(.7,0)
    core=reference_core(f,g,0.);one=np.array([1.,0.]);z=np.cos(.35)
    counter=dict(reference_overlap_abs=float(abs(np.cos(.35))),
                 source_signal=complex_json(coherence(core,one)),target_signal=float(z),
                 phase_only_mismatch=float(abs(coherence(core,one)-z)),guard_rejected=False)
    try:vacuum_phase(g)
    except ValueError:counter['guard_rejected']=True
    keys=('full_operator_error','compiled_full_operator_error_one_common_phase','source_xy_error','native_xy_error','actual_inverse_error')
    ok=all(all(x[k]<1e-11 for k in keys) for x in rows)
    ok=ok and joint['source_xy_error']<1e-11 and joint['native_xy_error']<1e-11 and counter['guard_rejected'] and counter['phase_only_mismatch']>1e-3
    result=dict(all_passed=bool(ok),cases=rows,independent_joint=joint,counterexample=counter,
                qiskit_bit_order='target low bits, reference next, ancilla last; Y=Sdg then H',
                source_CSWAP_count=4,tolerance=1e-11)
    save_json(OUT/'A2_SYNTHETIC.json',result)
    if not ok:raise ArithmeticError('A2 synthetic mismatch')
    return result


def ground():
    c=common();indices=np.array([int(f'{int(x):08b}'[::-1],2) for x in c['sector'].basis_indices])
    psi=np.zeros(256,complex);psi[indices]=c['state'];return psi


def stage_a2_h4(rare):
    tr=trajectory(1,rare,0);psi=ground()
    z=coherence(tr['direct'],psi)
    ref=reference_core(tr['F'],tr['G'],tr['phase'])
    result=dict(rare=rare,seed=tr['seed'],reference_phase=tr['phase'],expected=complex_json(z),errors={})
    for name,core in [('existing_direct',tr['direct']),('Simon_Love',tr['symmetric']),('reference',ref)]:
        native=compile_reference(core)
        result['errors'][name]=dict(source=float(abs(axis_signals(core,psi)-z)),
            native=float(abs(axis_signals(native,psi)-z)),n_qubits=core.num_qubits)
        if name!='reference':
            a=Operator(tr['direct']).data;b=Operator(core).data;c=Operator(native).data
            phase=np.vdot(a,c);phase/=abs(phase)
            result['errors'][name]['full_512_operator_source_error']=float(np.linalg.norm(a-b))
            result['errors'][name]['full_512_operator_native_error_one_common_phase']=float(np.linalg.norm(a-c/phase))
    result['all_passed']=all(v<TOL for row in result['errors'].values() for k,v in row.items() if k!='n_qubits')
    save_json(OUT/f'A2_H4_rare{int(rare)}.json',result)
    if not result['all_passed']:raise ArithmeticError('A2 H4 mismatch')
    return result


def metric(q):
    ops=q.count_ops()
    return [int(ops.get('rz',0)),int(ops.get('cx',0)),int(q.depth()),int(q.size())]


def price_sample(request):
    q,rare,j=request;tr=trajectory(q,rare,j)
    ref=reference_core(tr['F'],tr['G'],tr['phase'])
    values={};circuits={}
    for name,core in [('existing_direct',tr['direct']),('Simon_Love',tr['symmetric']),('reference',ref)]:
        values[name]=[]
        for axis in ('X','Y'):
            source=wrapper(core,axis);native=compile_reference(source)
            # Extra fresh reference preparation is explicit, not excluded.
            if name=='reference':
                full=QuantumCircuit(17,1);full.reset(range(8,16));full.compose(native,inplace=True);native=full
            values[name].append(metric(native))
            if j==0 and q in (1,8) and axis=='X':
                file=OUT/'circuits'/f'q{q}_rare{int(rare)}_{name}.qpy';file.parent.mkdir(exist_ok=True)
                with file.open('wb') as stream:qpy.dump(native,stream)
                circuits[name]=hashlib.sha256(file.read_bytes()).hexdigest()
    result=dict(q=q,rare=rare,j=j,seed=tr['seed'],probability=tr['probability'],
                event_records=tr['events'],reference_phase=tr['phase'],values=values,
                representative_native_qpy_sha256=circuits,metrics=list(METRICS),
                T=None,T_status='not synthesized; no conversion from RZ',
                extra_reference_resets=8,source_CSWAPs=16,extra_qubits=8,
                routing='same all-to-all compiler-A profile; no topology-specific routing claim')
    save_json(OUT/'prices'/f'q{q}_rare{int(rare)}_{j:02d}.json',result)
    return result


def parallel(function,requests):
    workers=min(len(requests),len(os.sched_getaffinity(0)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(function,r) for r in requests]
        return [f.result() for f in as_completed(futures)]


def stage_a3():
    requests=[(q,rare,j) for q in PRICE_Q for rare in (False,True) for j in range(PRICE_N)]
    rows=parallel(price_sample,requests);summary=[]
    for q in PRICE_Q:
        means={name:np.zeros((2,4)) for name in ('existing_direct','Simon_Love','reference')}
        difference=np.zeros((2,4));var=np.zeros((2,4))
        for rare in (False,True):
            cohort=[r for r in rows if r['q']==q and r['rare']==rare];p=cohort[0]['probability']
            for name in means:means[name]+=p*np.mean([r['values'][name] for r in cohort],axis=0)
            delta=np.array([np.array(r['values']['reference'])-r['values']['Simon_Love'] for r in cohort])
            difference+=p*delta.mean(axis=0);var+=p*p*np.var(delta,axis=0,ddof=1)/len(cohort)
        summary.append(dict(q=q,means={k:v.tolist() for k,v in means.items()},
            reference_minus_Simon_Love=difference.tolist(),paired_MC_SE=np.sqrt(var).tolist(),
            diagnostic_95_upper=(difference+1.96*np.sqrt(var)).tolist()))
    eligible=bool(np.any(np.array(summary[-1]['diagnostic_95_upper'])<0))
    result=dict(cells=6,samples=len(rows),n_per_stratum=PRICE_N,metrics=list(METRICS),rows=summary,
        pass_to_A4=eligible,status='RESOURCE_OPPORTUNITY' if eligible else 'STOP_NO_RESOURCE_ADVANTAGE_IN_REGISTERED_COHORT',
        uncertainty='stratified paired MC diagnostic; no simultaneous or long-round certificate',
        strongest_applicable_baseline='Simon--Love partial-S2 cancellation, unchanged controlled finite RTE',
        independent_PR_symmetric_control='Not substituted: selected saved SOURCE is ordinary diag(I,V). Two independent branches are verified in A2, without changing this law.',
        T=None)
    save_json(OUT/'A3_PRICES.json',result)
    return result


def main():
    if (OUT/'STOP.json').exists():raise RuntimeError('Mandatory STOP already recorded')
    init_inputs()
    stage_a1();print('A1 PASS',flush=True)
    stage_a2_synthetic();parallel(stage_a2_h4,[False,True]);print('A2 PASS',flush=True)
    prices=stage_a3();print('A3',prices['status'],flush=True)
    return prices
