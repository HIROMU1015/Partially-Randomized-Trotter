"""Invented arrays/events only; no saved molecular file or QPY is opened."""
from dataclasses import replace
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.sparse import csr_matrix
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_pr_native_structure as poc
from trottertracks import c_pr_aware_df_stage_b as b
from trottertracks import c_pr_aware_df_stage_b2 as b2
from trottertracks.c1_h4_strong_baseline import StrongBuilder
from trottertracks.c1_h6_compiler_precision import compiled_wrapper
from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
from trottertracks.pr_dictionary_validation.circuits import BasisProvider,build_event_sequence
from trottertracks.pr_dictionary_validation.prototype import MixedAtom,make_tail,ordered_event
from trottertracks.pr_dictionary_validation.f2_execution import circuit_metrics
from trotterlib import rte


def artificial(n=3):
    rng=np.random.default_rng(904+n)
    a=rng.normal(size=(5,n,n))*.02;J=a+a.transpose(0,2,1)
    U=[]
    for i in range(5):
        v=rng.normal(size=(n,n))*.3;U.append(expm(v-v.T))
    U=np.array(U);U[3,:,0]*=-1 # reflection/column gauge, not identified with U[2]
    v=np.zeros((n,n));v[0,1]=1e-8;v[1,0]=-1e-8
    U[4]=U[2]@expm(v) # near basis MUST remain a different exact key
    h=np.diag(np.linspace(-.2,.2,n));h[0,1]=h[1,0]=.03
    g=np.zeros((n,)*4)
    arm=b.prepare_arm(h,g,.31,J,U,role='proposed',source_identity='INVENTED',input_identity='INVENTED',ld=2)
    return arm,h,g


def provider(arm,builder):
    class Full(BasisProvider):
        def basis(self,atom,singleton=False):
            block=poc._block(atom,arm.plan)
            return builder._basis(arm.model,block).copy()
    return Full(arm.model.modes,arm.plan.frames)


def events_for(arm,K=0):
    tail,c,dist=b.configure(arm,Fraction(1,10),4,K)
    groups={}
    for atom in arm.plan.atoms:groups.setdefault(atom.frame_id,[]).append(atom)
    groups=list(groups.values());assert len(groups)==3
    # Repeated exact key, then different and near keys. Negative atoms included.
    chosen=[next(a for a in groups[0] if a.coefficient<0),groups[0][1],groups[1][0],groups[2][0]]
    events=[]
    for i,a in enumerate(chosen):
        products=() if K==0 else (groups[(i+1)%3][0].atom_id,chosen[0].atom_id)
        events.append(ordered_event(tail,dist,K,products,a.atom_id))
    return tuple(events),c


def full_context(arm,h,g):
    car=b.stage1.car_bilinears(arm.model.modes)
    G=[b.stage1.raw_base(b.stage1.corrected_kappa(h,g),arm.model.constant,car)]
    G.extend(b.stage1.raw_frame(J,U,car)[0] for J,U in zip(arm.J,arm.U))
    ctx=SimpleNamespace(d=len(G[0]),bilinears={(p,q):csr_matrix(car[p,q])
        for p in range(arm.model.modes) for q in range(arm.model.modes) if p%2==q%2})
    return G,b.atom_operators(ctx,arm)


def proposed_kernel(arm,events,*,relative=True,fold=False):
    # Artificial assembly only. No existing molecular runner imports this PoC.
    builder=StrongBuilder(relative_fusion=True);half=tuple((i,Fraction(1,20)) for i in range(3))
    qc=QuantumCircuit(arm.model.modes+1)
    qc.compose(builder.sequence(arm.model,half,optimized=True,scalar_time=0),inplace=True)
    qc.compose(poc.tail_sequence(events,arm.plan,builder,relative=relative,fold_central=fold),inplace=True)
    qc.compose(builder.sequence(arm.model,half[::-1],optimized=True,scalar_time=0),inplace=True)
    qc.p(-(arm.model.constant+arm.plan.extracted_identity)*.1,arm.model.modes)
    return qc


def record(name,value):
    if os.environ.get('PRT_STRUCTURE_ARTIFACT'):
        root=Path(os.environ['PRT_STRUCTURE_ARTIFACT']);root.mkdir(parents=True,exist_ok=True)
        (root/(name+'.json')).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def test_passive_import_and_no_scientific_entry():
    code="import sys;sys.path.insert(0,'src');from trottertracks import c_pr_native_structure as p;assert not any(x in sys.modules for x in ['numpy','scipy','qiskit','ffsim']);assert not hasattr(p,'run')"
    subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,check=True)


@pytest.mark.parametrize('n,K',[(2,0),(3,2)])
def test_complete_controlled_CAR_P0_adjoint_XY_scalar_sign_and_order(n,K):
    arm,h,g=artificial(n);events,config=events_for(arm,K);before=repr(events)
    old=b.build_kernel(arm,Fraction(1,10),events,config)
    new=proposed_kernel(arm,events);G,operators=full_context(arm,h,g)
    ref=b.trajectory_reference(arm,G,operators,Fraction(1,10),events,config)
    compiled=compiled_wrapper(new,CompilerProfile())
    R,target,raw_checks=b2.raw_gate(new,ref)
    C,compiled_checks=b2.compiled_gate(compiled,R,target)
    adj,adj_checks=b2.adjoint_pair_gate(new,compiled,R,C,target)
    np.testing.assert_allclose(Operator(old).data,R,rtol=0,atol=1e-10)
    assert repr(events)==before
    d=1<<arm.model.modes
    np.testing.assert_allclose(R[:d,:d],np.eye(d),rtol=0,atol=1e-10)
    shells={}
    for name in ('head','X','Y'):
        shell=b2.shell_build(arm.model.modes,name)
        shells[name],_=b2.shell_assessment(shell,compiled_wrapper(shell,CompilerProfile()),name)
    rng=np.random.default_rng(33);psi=rng.normal(size=d)+1j*rng.normal(size=d);psi/=np.linalg.norm(psi)
    xy=b2.XY_gate(R,C,ref,psi,shells)
    record(f'controlled_n{n}_K{K}',dict(raw=raw_checks,compiled=compiled_checks,adjoint=adj_checks,XY=xy,
        dimension=2*d,fixture='invented arrays; no molecular claim',event_sequence_unchanged=True))
    # Independent mutations fail without a fitted global phase.
    corrupt=new.copy();corrupt.p(.02,arm.model.modes)
    with pytest.raises(ValueError):b2.raw_gate(corrupt,ref)
    reverse=proposed_kernel(arm,events[::-1])
    assert np.max(abs(Operator(reverse).data-target))>1e-7


def test_identity_negative_products_phase_and_barrier_exactly_preserved():
    arm,h,g=artificial(2);atoms=list(arm.plan.atoms)
    identity=MixedAtom('Inegative','IDENTITY',Fraction(-1,7),('inventedI',),('I',)*4,None)
    atoms.append(identity);plan=replace(arm.plan,atoms=tuple(atoms))
    tail=make_tail(tuple(atoms),'artificial identity')
    config,dist=rte.make_rte_config(tail,evolution_time=.1,rte_steps=4,truncation_tolerance=1.,finite_taylor_order=2,seed=1)
    a=next(x for x in atoms if x.kind=='FRAME_Z' and x.coefficient<0)
    z=next(x for x in atoms if x.kind=='FRAME_Z' and x.frame_id!=a.frame_id)
    events=(ordered_event(tail,dist,2,(a.atom_id,a.atom_id),identity.atom_id),
        ordered_event(tail,dist,2,(identity.atom_id,z.atom_id),a.atom_id))
    builder=StrongBuilder(relative_fusion=True)
    old=build_event_sequence(events,atoms,provider(replace(arm,plan=plan),builder),optimized=True)
    new=poc.tail_sequence(events,plan,builder)
    folded=poc.tail_sequence(events,plan,builder,fold_central=True)
    G,ops=full_context(arm,h,g);ops[identity.atom_id]=-np.eye(16)
    ref=np.eye(16,dtype=complex)
    for e in events:ref=rte.event_unitary(e,ops)@ref
    b2.raw_gate(old,ref);b2.raw_gate(new,ref);b2.raw_gate(folded,ref)
    np.testing.assert_allclose(Operator(old).data,Operator(new).data,rtol=0,atol=1e-11)


@pytest.mark.parametrize('kind',['reflection','pi','signed_zero','complex_branch'])
def test_analytic_transition_is_legal_at_tail_and_head_tail_boundaries(kind):
    arm,_,_=artificial(2);a,c=arm.model.blocks[2:4]
    if kind=='reflection':U=np.diag([-1.,1.,-1.,1.])
    elif kind=='pi':U=np.diag([-1.,-1.,1.,1.])
    elif kind=='signed_zero':
        U=np.eye(4);U[0,1]=-0.
    else:U=np.diag(np.exp(1j*np.array([.8*np.pi,-.8*np.pi,.1,-.1])))
    a=replace(a,rotation=b.native._rotation(U));builder=StrongBuilder(relative_fusion=True)
    old=QuantumCircuit(4);old.compose(builder._basis(arm.model,a),inplace=True)
    old.compose(builder._basis(arm.model,c).inverse(),inplace=True)
    relative=builder.transition(arm.model,a,c)
    np.testing.assert_allclose(Operator(old).data,Operator(relative).data,atol=1e-12,rtol=0)
    assert builder.fusion_records[-1]['phase_method']=='analytic RZ vacuum phase; no fitting'


def test_near_basis_keys_and_invalid_metadata_are_never_merged_or_fallback():
    arm,_,_=artificial(3);events,_=events_for(arm)
    groups=poc._runs(poc._records(events,arm.plan.atoms)[0])
    assert len(groups)==3 # same F2 run + reflection F3 + near-but-distinct F4
    assert groups[0][0][0].basis_key!=groups[2][0][0].basis_key
    assert 0<np.max(abs(np.array(arm.U[2])-np.array(arm.U[4])))<1e-6
    atom=groups[0][0][0];forged=replace(atom,basis_key='near is not same')
    with pytest.raises(ValueError,match='EXACT_DECLARED'):poc._block(forged,arm.plan)
    wrong=replace(events[0].application_sequence[-1],coefficient_sign=-events[0].application_sequence[-1].coefficient_sign)
    e=replace(events[0],application_sequence=(*events[0].application_sequence[:-1],wrong))
    with pytest.raises(ValueError,match='sign'):poc.tail_sequence((e,),arm.plan,StrongBuilder())
    with pytest.raises(ValueError,match='ONE_EXACT'):poc.central_fold(QuantumCircuit(7),[groups[0][0],groups[1][0]],6)
    with pytest.raises(ValueError,match='NONEMPTY'):poc.central_fold(QuantumCircuit(7),[],6)


@pytest.mark.parametrize('n',[2,3])
def test_artificial_P0_metrics_A_and_C_not_assumed_additive(n):
    arm,_,_=artificial(n);events,config=events_for(arm)
    old=b.build_kernel(arm,Fraction(1,10),events,config)
    A=proposed_kernel(arm,events)
    compiled_old=compiled_wrapper(old,CompilerProfile());compiled_A=compiled_wrapper(A,CompilerProfile())
    np.testing.assert_allclose(Operator(compiled_A).data,Operator(compiled_old).data,rtol=0,atol=1e-10)
    C=proposed_kernel(arm,events,relative=False,fold=True)
    compiled_C=compiled_wrapper(C,CompilerProfile())
    np.testing.assert_allclose(Operator(compiled_C).data,Operator(compiled_old).data,rtol=0,atol=1e-10)
    metrics=dict(complete_kernel=dict(old=circuit_metrics(compiled_old),A=circuit_metrics(compiled_A),
        C=circuit_metrics(compiled_C)))
    # C cases fixed before measurement: adjacent duplicate products, and
    # interleaved diagonal supports. They are not added to A's science proposal.
    atoms=[a for a in arm.plan.atoms if a.frame_id==arm.plan.atoms[0].frame_id]
    a=atoms[0];z=next(x for x in atoms if x.support!=a.support)
    for name,run in [('duplicate_products',[(a,None),(a,None)]),
                     ('interleaved_rotations',[(a,.17),(z,-.11),(a,.23),(z,.11)])]:
        baseline=QuantumCircuit(arm.model.modes+1)
        for atom,angle in run:
            from trottertracks.pr_dictionary_validation.circuits import append_central
            append_central(baseline,atom.support,arm.model.modes,angle)
        folded=QuantumCircuit(arm.model.modes+1);poc.central_fold(folded,run,arm.model.modes)
        np.testing.assert_allclose(Operator(folded).data,Operator(baseline).data,rtol=0,atol=1e-13)
        c_old=compiled_wrapper(baseline,CompilerProfile());c_new=compiled_wrapper(folded,CompilerProfile())
        metrics[name]=dict(raw_old=circuit_metrics(baseline),raw_C=circuit_metrics(folded),
            P0_old=circuit_metrics(c_old),P0_C=circuit_metrics(c_new))
    # A is accepted as a legal transform independently of this fixture's price.
    record(f'artificial_P0_metrics_n{n}',dict(qualification='synthetic mechanism only; H4 prices UNKNOWN',metrics=metrics))
