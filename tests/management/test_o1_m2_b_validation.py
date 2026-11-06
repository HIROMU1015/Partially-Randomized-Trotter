"""Invented packets only; forbid saved input loading and circuit construction."""
from dataclasses import replace
import importlib
from pathlib import Path
import sys

import numpy as np
import pytest
from test_o1_m2_residual_adapter import fixture,prepare,physical_car
from trottertracks import o1_m2_hermitian_b as b
from trottertracks.o1_m2_residual_adapter import SpatialIntegrals

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
runner=importlib.import_module('validate_o1_m2_hermitian_b')


@pytest.fixture(autouse=True)
def forbid_science_files_and_circuits(monkeypatch):
    def forbidden(*a,**k): pytest.fail('No saved science or circuit construction')
    monkeypatch.setattr(np,'load',forbidden)
    import qiskit
    monkeypatch.setattr(qiskit,'QuantumCircuit',forbidden)


def test_fixed_H4_plan_is_dimension_only_and_reference_reuses_existing_CAR():
    assert runner.H4_PLAN==runner.fixed_plan(4,8)
    assert runner.H4_PLAN.mapper_operations==33824
    assert runner.H4_PLAN.reference_operations==4228
    assert runner.H4_PLAN.guard==8 and runner.FORMATION_RELATIVE_LIMIT==.01
    raw=prepare();reference=runner.previous.CARReference(2)
    for p in (raw.physical,raw.core,raw.residual):
        np.testing.assert_allclose(runner.fsum_car(reference,p),physical_car(p.h,p.g,p.nuclear_constant),
                                   rtol=0,atol=2e-15)


@pytest.mark.parametrize('scale',[1.,1e-10,1e-22])
def test_synthetic_operator_checks_preserve_small_residual_and_model_unassessed(scale):
    h,g,c,J,U=fixture()
    raw=prepare(h*scale,g*scale,c*scale,J*scale,U)
    result=runner.evaluate(raw)
    assert result['operator_checks_passed']
    assert result['residual_mapper']['status']=='PASS'
    assert result['residual_mapper']['upstream_allowance_used'] is False
    assert result['model_change_status']=='UNASSESSED' and not result['formal_adoption']
    assert result['formation_resolution_qualified']
    assert result['scalar_identity_bookkeeping']
    assert result['identity']['residual']['nonidentity_terms']>0
    assert result['identity']['residual']['electronic']!=0.
    assert result['sector_dimensions']=={'full':16,'sector':4}


def test_canonical_cancellation_is_detected_even_if_zero_mapper_is_correct():
    raw=prepare()
    h=np.array([[0.,1e16+2],[1e16,0.]])
    c=np.array([[0.,1e16],[1e16,0.]])
    g=np.zeros((2,)*4)
    raw=replace(raw,physical=SpatialIntegrals(h,g,0.),core=SpatialIntegrals(c,g,0.),
                residual=SpatialIntegrals(h-c,g,0.))
    result=runner.evaluate(raw)
    # Correct mapping of canonical zero must not hide upstream cancellation.
    assert result['residual_mapper']['status']=='PASS'
    assert not result['formation_resolution_qualified']
    assert not result['operator_checks_passed']
    assert result['exact_formation']['canonical_formation_error_bound']>0
    assert result['canonical_vs_P_raw_R_max_entry']>0


@pytest.mark.parametrize('sign',[0.,-1.])
def test_zero_and_reverse_sign_mapper_fail_on_residual_own_scale(monkeypatch,sign):
    original=runner.previous.pauli_matrix
    def wrong(packet):
        value=original(packet)
        return sign*value if packet.metadata['model_id']=='M2_B_residual' else value
    monkeypatch.setattr(runner.previous,'pauli_matrix',wrong)
    result=runner.evaluate(prepare())
    assert not result['operator_checks_passed']
    assert result['residual_mapper']['status']=='FAIL'


def test_raw_antihermitian_part_is_recorded_with_independent_CAR():
    raw=prepare();ah=np.array([[0.,3e-17],[-3e-17,0.]])
    # Use zero h/core to keep invented asymmetry representable before P.
    g=np.zeros((2,)*4)
    raw=replace(raw,physical=SpatialIntegrals(ah,g,0.),core=SpatialIntegrals(2*ah,g,0.),
                residual=SpatialIntegrals(-ah,g,0.))
    result=runner.evaluate(raw)
    assert all(v>0 for v in result['raw_antihermitian_half_max_entry'].values())
    assert result['residual_mapper']['status']=='PASS'
    assert result['model_change_status']=='UNASSESSED'
    denied=runner.evaluate(raw,projection_limits=dict(physical=0.,core=0.,residual=0.))
    assert denied['model_change_status']=='FAIL' and not denied['formal_adoption']


def test_exactly_zero_H_and_core_is_distinct_from_lost_residual():
    h,g,c,J,U=fixture()
    raw=prepare(np.zeros_like(h),np.zeros_like(g),0.,np.zeros_like(J),U)
    result=runner.evaluate(raw)
    assert result['operator_checks_passed']
    assert not result['exact_formation']['exact_coefficient_nonzero']
    assert result['formation_resolution_qualified']


def test_full_H_allowance_cannot_hide_unresolved_upstream_core_when_R_is_zero(monkeypatch):
    raw=prepare();zero=SpatialIntegrals(np.zeros((2,2)),np.zeros((2,)*4),0.)
    raw=replace(raw,physical=raw.core,residual=zero)
    original=runner.previous.CARReference.density_core
    def tiny_gap(reference,model):
        value=original(reference,model).copy();value[0,0]+=1e-14
        return value
    monkeypatch.setattr(runner.previous.CARReference,'density_core',tiny_gap)
    result=runner.evaluate(raw)
    assert result['residual_mapper']['status']=='PASS'
    assert all(c['passed'] for c in result['checks'])
    assert not result['upstream_reference_resolution_qualified']
    assert not result['operator_checks_passed']
