"""Invented real spatial packets only; no saved H4, states or circuits."""
from dataclasses import replace
import numpy as np
import pytest
from test_o1_m2_residual_adapter import (
    adapter,fixture,prepare,physical_car,pauli_matrix,density_core_car,
)
from trottertracks import o1_m2_hermitian_b as b


PLAN=b.RoundoffPlan(tensor_contraction_operations=512,kappa_operations=16,
    core_one_body_operations=16,mapper_operations=2048,reference_operations=2048)


@pytest.fixture(autouse=True)
def no_saved_input_or_circuit(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('No saved scientific input or circuit in artificial B tests')
    monkeypatch.setattr(np,'load',forbidden)
    import qiskit
    monkeypatch.setattr(qiskit,'QuantumCircuit',forbidden)


def matrix(packet):
    return physical_car(packet.h,packet.g,packet.nuclear_constant)


def antisymmetric_raw():
    raw=prepare()
    ah=np.array([[0.,3e-8],[-3e-8,0.]])
    ag=np.zeros((2,)*4);ag[0,1,1,0]=7e-8;ag[1,0,0,1]=-7e-8
    physical=adapter.SpatialIntegrals(raw.physical.h+ah,raw.physical.g+ag,1.7)
    core=adapter.SpatialIntegrals(raw.core.h+2*ah,raw.core.g-3*ag,1.7)
    residual=adapter.SpatialIntegrals(physical.h-core.h,physical.g-core.g,0.)
    return replace(raw,physical=physical,core=core,residual=residual)


def test_all_three_operator_projections_not_residual_only_and_raw_unchanged():
    raw=antisymmetric_raw()
    originals=[(p.h.copy(),p.g.copy()) for p in (raw.physical,raw.core,raw.residual)]
    candidate=b.derive_hermitian_b(raw)
    assert candidate.raw is raw and candidate.sector==(1,1)
    for old,new in ((raw.physical,candidate.physical),(raw.core,candidate.core)):
        operator=matrix(old)
        np.testing.assert_allclose(matrix(new),(operator+operator.conj().T)/2,rtol=0,atol=2e-15)
        np.testing.assert_array_equal(new.h,new.h.T)
        np.testing.assert_array_equal(new.g,new.g.transpose(1,0,3,2))
        assert not new.h.flags.writeable and not new.g.flags.writeable
    H,C,R=(matrix(p) for p in (candidate.physical,candidate.core,candidate.residual))
    np.testing.assert_allclose(C+R,H,rtol=0,atol=2e-15)
    assert np.max(abs(matrix(raw.physical)-matrix(raw.physical).T))>1e-8
    assert np.max(abs(matrix(raw.core)-matrix(raw.core).T))>1e-8
    assert np.max(abs(matrix(raw.core)+matrix(candidate.residual)-matrix(raw.physical)))>1e-8
    for p,(h,g) in zip((raw.physical,raw.core,raw.residual),originals):
        np.testing.assert_array_equal(p.h,h);np.testing.assert_array_equal(p.g,g)


def test_original_g_kappa_all_frames_scalar_identity_one_body_and_sector_are_kept():
    raw=prepare();candidate=b.derive_hermitian_b(raw)
    assert candidate.raw.general_core.original_outer_rank==2
    assert candidate.raw.general_core is raw.general_core
    independent=density_core_car(raw.general_core)
    # Reuse the preceding artificial general-J CAR fixture's 1e-14 bound.
    np.testing.assert_allclose(matrix(candidate.core),(independent+independent.T)/2,rtol=0,atol=1e-14)
    mapped=candidate.pauli_parts()
    for p,m in zip((candidate.physical,candidate.core,candidate.residual),mapped):
        np.testing.assert_allclose(pauli_matrix(m),matrix(p),rtol=0,atol=2e-15)
        assert m.metadata['imaginary_policy']=='reject_any_nonzero'
    assert candidate.physical.nuclear_constant==candidate.core.nuclear_constant==1.7
    assert candidate.residual.nuclear_constant==0.
    assert abs(mapped[2].constant)>.01 and np.max(abs(candidate.residual.h))>.01
    assert mapped[2].constant==pytest.approx(np.trace(matrix(candidate.residual))/16)
    sector=[s for s in range(16) if sum((s>>q)&1 for q in (0,2))==1
            and sum((s>>q)&1 for q in (1,3))==1]
    outside=np.setdiff1d(np.arange(16),sector)
    assert np.max(abs(matrix(candidate.residual)[np.ix_(outside,sector)]))==0.


@pytest.mark.parametrize('scale',[1e-10,1e-16,1e-22])
def test_residual_scale_mapping_catches_zero_sign_one_body_and_identity_losses(scale):
    h,g,c,J,U=fixture()
    candidate=b.derive_hermitian_b(prepare(h*scale,g*scale,c*scale,J*scale,U))
    ref=matrix(candidate.residual)
    packet=candidate.pauli_parts()[2]
    mapped=pauli_matrix(packet)
    good=b.assess_residual_mapper(candidate,ref,mapped,PLAN)
    assert good['status']=='PASS' and good['resolution_qualified']
    assert good['upstream_allowance_used'] is False and good['physical_model_qualified'] is False
    assert packet.terms and packet.constant!=0.
    missing_one=physical_car(np.zeros((2,2)),candidate.residual.g,0.)
    missing_I=mapped-packet.constant*np.eye(16)
    for wrong in (np.zeros_like(mapped),-mapped,missing_one,missing_I):
        assert b.assess_residual_mapper(candidate,ref,wrong,PLAN)['status']=='FAIL'


def test_raw_imaginary_part_below_old_tolerance_is_explicitly_rejected():
    import openfermion.config as config
    old_tolerance=config.EQ_TOLERANCE
    raw=adapter.SpatialIntegrals(np.array([[0.,3e-17],[-3e-17,0.]]),np.zeros((2,)*4),0.)
    coefficients=b.complex_pauli_coefficients(raw)
    assert max(abs(c.imag) for c in coefficients.values())>0.
    with pytest.raises(ValueError,match='NONZERO_IMAGINARY'):
        b.strict_pauli_map(raw,model_id='raw_anti')
    projected=b.hermitian_packet(raw)
    assert b.strict_pauli_map(projected,model_id='explicit_B').terms==()
    assert config.EQ_TOLERANCE==old_tolerance


def test_an_identically_wrong_antihermitian_reference_and_mapper_are_rejected():
    candidate=b.derive_hermitian_b(prepare())
    ref=matrix(candidate.residual)
    wrong=ref.copy();wrong[1,2]+=3e-8;wrong[2,1]-=3e-8
    result=b.assess_residual_mapper(candidate,wrong,wrong,PLAN)
    assert result['max_entry']==0. and result['status']=='FAIL'
    assert result['reference_hermiticity_defect']>1e-8


@pytest.mark.parametrize('field',['h','g','nuclear_constant'])
def test_complex_spatial_input_cannot_be_silently_cast(field):
    p=adapter.SpatialIntegrals(np.zeros((1,1)),np.zeros((1,)*4),0.)
    value=getattr(p,field)
    changed=np.asarray(value,dtype=complex)+1e-30j
    bad=replace(p,**{field:changed})
    for function in (b.hermitian_packet,b.complex_pauli_coefficients):
        with pytest.raises(ValueError,match='real'):
            function(bad)


def test_canonical_subtraction_order_is_explicit_and_other_route_is_diagnostic():
    raw=prepare()
    H=adapter.SpatialIntegrals(np.array([[0.,1e16+2],[1e16,0.]]),np.zeros((2,)*4),1.7)
    C=adapter.SpatialIntegrals(np.array([[0.,1e16],[1e16,0.]]),np.zeros((2,)*4),1.7)
    R=adapter.SpatialIntegrals(H.h-C.h,H.g-C.g,0.)
    candidate=b.derive_hermitian_b(replace(raw,physical=H,core=C,residual=R))
    np.testing.assert_array_equal(candidate.residual.h,np.zeros((2,2)))
    assert candidate.projected_raw_residual.h[0,1]==1.
    budgets=b.predeclared_budgets(candidate,PLAN)
    assert budgets['canonical_vs_projected_raw_residual_bound']>0.


def test_input_model_formation_and_mapper_qualifications_remain_separate():
    raw=antisymmetric_raw();candidate=b.derive_hermitian_b(raw)
    closure=float(np.max(abs(matrix(candidate.core)+matrix(candidate.residual)-matrix(candidate.physical))))
    unknown=b.assess_formation(candidate,PLAN,closure_error=closure,input_projection_limits=None)
    assert unknown['formation_status']=='PASS' and unknown['model_change_status']=='UNASSESSED'
    rejected=b.assess_formation(candidate,PLAN,closure_error=closure,
        input_projection_limits={'physical':0.,'core':0.,'residual':0.})
    assert rejected['model_change_status']=='FAIL'
    accepted=b.assess_formation(candidate,PLAN,closure_error=closure,
        input_projection_limits={'physical':1e-4,'core':1e-4,'residual':1e-4})
    assert accepted['model_change_status']=='PASS'
    assert accepted['physical_generation_accuracy_qualified'] is False
    failure=b.assess_formation(candidate,PLAN,closure_error=1.,input_projection_limits=None)
    assert failure['formation_status']=='FAIL'


def test_upstream_floor_is_not_used_to_hide_an_almost_exact_core_residual():
    h,g,c,J,U=fixture();seed=prepare()
    candidate=b.derive_hermitian_b(prepare(h,seed.core.g+1e-10*g,c,J,U))
    ref=matrix(candidate.residual);mapped=pauli_matrix(candidate.pauli_parts()[2])
    report=b.assess_residual_mapper(candidate,ref,mapped,PLAN)
    assert report['status']=='PASS'
    assert report['budgets']['formation_allowance']>1e6*report['tolerance']
    assert b.assess_residual_mapper(candidate,ref,np.zeros_like(ref),PLAN)['status']=='FAIL'
    coarse=replace(PLAN,guard=1e20)
    assert b.assess_residual_mapper(candidate,ref,mapped,coarse)['status']=='NOT_QUALIFIED'


def test_zero_residual_is_not_mistaken_for_an_accidentally_lost_nonzero_one():
    raw=prepare();zero=adapter.SpatialIntegrals(np.zeros((2,2)),np.zeros((2,)*4),0.)
    exact=replace(raw,physical=raw.core,residual=zero)
    candidate=b.derive_hermitian_b(exact)
    ref=matrix(candidate.residual)
    assert b.assess_residual_mapper(candidate,ref,ref,PLAN)['status']=='PASS'
    assert candidate.pauli_parts()[2].terms==()
