"""Opt-in precision adapter tests; exclusively artificial inputs."""
from fractions import Fraction
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from trotterlib import b2_strong_pf_precision as hp
from trotterlib.b2_p1_strong_pf_h4_diagnosis import mp_copy


@pytest.fixture(autouse=True)
def no_molecular_or_circuit_work(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Synthetic inputs only')
    from qiskit import QuantumCircuit
    monkeypatch.setattr(np,'load',forbidden)
    monkeypatch.setattr(hp.legacy.p,'load_private_inputs',forbidden)
    monkeypatch.setattr(hp.legacy.p,'public_json',forbidden)
    monkeypatch.setattr(hp.legacy.saved,'fixed_json',forbidden)
    monkeypatch.setattr(QuantumCircuit,'__init__',forbidden)
    monkeypatch.setattr(hp.legacy.strong.StrongBuilder,'sequence',forbidden)


@pytest.fixture(scope='module')
def artificial():
    path=Path(__file__).resolve().parents[1]/'scripts/validate_b2_strong_pf_precision.py'
    spec=importlib.util.spec_from_file_location('synthetic_precision_benchmark',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    one,w,gs,c,indices,psi=module.artificial_input()
    prep=hp.prepare_precision(one,w,gs,c,digits=80,basis_indices=indices)
    reference_prep=hp.prepare_precision(one,w,gs,c,digits=120,basis_indices=indices)
    reference=module.IndependentReference(reference_prep.ctx,one,w,gs,c,indices)
    return module,(one,w,gs,c,indices,psi),prep,reference


@pytest.mark.parametrize('formula,s',hp.legacy.ARMS)
def test_same_schedule_and_independent_reference(artificial,formula,s):
    _,data,prep,ref=artificial
    pair=hp.precise_step(prep,formula=formula,substeps=s)
    assert pair.schedule==hp.legacy.strong_baseline_schedule(3,substeps=s,formula=formula,delta_time=.02)
    assert any(t<0 for _,t in pair.schedule)==(formula=='Yoshida4')
    target=ref.step(pair.schedule,.02)
    for a in (pair.native,pair.car):
        assert hp.maxabs(mp_copy(ref.ctx,a)-target)<hp.legacy.p.MATCH
        assert hp.unitarity(a)<hp.legacy.p.MATCH
    assert pair.emitter_correctness=='UNASSESSED'
    assert all(v<prep.ctx.mpf(hp.legacy.p.MATCH) for d in prep.diagnostics for v in d.values())


@pytest.mark.parametrize('formula,s',[('Full_S2',4),('Yoshida4',2)])
def test_scalar_signed_identity_and_occupation_semantics(formula,s):
    h=.5*np.eye(2);g=.75*np.eye(2);w=-.5;c=.125
    prep=hp.prepare_precision(h,[w],[g],c);ctx=prep.ctx
    pair=hp.precise_step(prep,formula=formula,substeps=s)
    times=[sum((t for i,t in pair.schedule if i==k),Fraction(0)) for k in range(2)]
    for x in range(4):
        n=x.bit_count()
        phase=hp.exact_real(ctx,c)*hp.exact_real(ctx,Fraction(.02))
        phase+=hp.exact_real(ctx,times[0])*hp.exact_real(ctx,.5)*n
        phase+=hp.exact_real(ctx,times[1])*hp.exact_real(ctx,w)*hp.exact_real(ctx,.75)**2*n*n
        target=ctx.exp(-ctx.j*phase)
        assert abs(pair.native[x,x]-target)<hp.legacy.p.MATCH
        assert abs(pair.car[x,x]-target)<hp.legacy.p.MATCH
    assert hp.maxabs(pair.native-ctx.diag([pair.native[i,i] for i in range(4)]))==0


def test_exact_conversion_fraction_and_no_unitary_repair(artificial):
    _,_,p,_=artificial;ctx=p.ctx
    x=np.nextafter(.1,1.);n,d=x.as_integer_ratio()
    assert hp.exact_real(ctx,x)==ctx.mpf(n)/d
    time=Fraction(.02)*Fraction(1.3512071919596578)/8
    assert hp.exact_real(ctx,time)==ctx.mpf(time.numerator)/time.denominator
    assert hp.exact_real(ctx,time)!=hp.exact_real(ctx,float(time))
    a=ctx.diag([hp.exact_real(ctx,1.+2.**-46),1])
    lifted=hp.fock_lift_hp(ctx,a,(0,1,2,3))
    assert lifted[3,3]>1 and hp.unitarity(lifted)>0


def test_minor_exact_singularity_and_tiny_nonzero_pivot(artificial):
    _,_,p,_=artificial;ctx=p.ctx
    assert hp.minor_determinant(ctx,ctx.matrix([[0,1],[0,2]]))==0
    assert hp.minor_determinant(ctx,ctx.matrix([[1,2],[2,4]]))==0
    delta=ctx.mpf(2)**-200
    assert hp.minor_determinant(ctx,ctx.matrix([[1,1],[1,1+delta]]))==delta


def test_sector_order_and_full_space_invariance(artificial):
    _,(one,w,gs,c,indices,_),prep,_=artificial
    full=hp.prepare_precision(one,w,gs,c,digits=80)
    a=hp.precise_step(prep,formula='Full_S2',substeps=1)
    b=hp.precise_step(full,formula='Full_S2',substeps=1)
    ctx=prep.ctx
    restricted=ctx.matrix([[b.native[int(i),int(j)] for j in indices] for i in indices])
    assert hp.maxabs(a.native-restricted)<hp.legacy.p.MATCH
    assert a.indices==tuple(indices)  # OF sector -> Qiskit reversal retained.
    outside=set(range(16))-set(indices)
    assert max(abs(b.native[int(i),j]) for i in indices for j in outside)==0
    with pytest.raises(ValueError,match='COMPLETE_SPIN_SECTORS'):
        hp.prepare_precision(one,w,gs,c,basis_indices=indices[:-1])


@pytest.mark.parametrize('defect',['hermitian','spin','dtype'])
def test_invalid_input_is_rejected_without_projection(artificial,defect):
    _,(one,w,gs,c,_,_),_,_=artificial
    bad=one.copy()
    if defect=='hermitian':bad[0,2]+=1e-12
    if defect=='spin':bad[0,1]=bad[1,0]=1e-30
    if defect=='dtype':bad=bad.astype(np.complex64)
    with pytest.raises(ValueError):hp.prepare_precision(bad,w,gs,c)
