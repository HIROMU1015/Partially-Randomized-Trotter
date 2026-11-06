"""Invented arrays only; independent existing CAR and micro-coefficient mapper."""
from dataclasses import replace
from fractions import Fraction
import importlib
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from test_o1_m2_residual_adapter import fixture,prepare,physical_car
from trottertracks import o1_m2_core_only_bound as bound
from trottertracks.o1_m2_residual_adapter import SpatialIntegrals


@pytest.fixture(autouse=True)
def no_saved_input_or_circuit(monkeypatch):
    def forbidden(*a,**k):pytest.fail('No saved H4 input or circuits in synthetic tests')
    monkeypatch.setattr(np,'load',forbidden)
    import qiskit
    monkeypatch.setattr(qiskit,'QuantumCircuit',forbidden)


@pytest.mark.parametrize('scale',[1.,1e-10,1e-22])
def test_conservative_bound_against_independent_CAR_and_eigenvalue_difference(scale):
    h,g,c,J,U=fixture();raw=prepare(h*scale,g*scale,c*scale,J*scale,U)
    result=bound.core_only_bounds(raw)
    candidate=bound.b.derive_hermitian_b(raw)
    H=physical_car(raw.physical.h,raw.physical.g,raw.physical.nuclear_constant)
    C=physical_car(raw.core.h,raw.core.g,raw.core.nuclear_constant)
    H=(H+H.T)/2;C=(C+C.T)/2
    limit=result['energy_bound_reference_H_H_vs_C_H']['upper_bound']
    assert np.linalg.norm(H-C,2)<=limit*(1+1e-12)+1e-14*scale
    assert abs(np.linalg.eigvalsh(H)[0]-np.linalg.eigvalsh(C)[0])<=limit+1e-14*scale
    packet=bound.b.strict_pauli_map(candidate.residual,model_id='invented_micro_residual')
    assert packet.terms and packet.constant!=0 and limit>0
    assert result['hh_budget_status']=='DEFERRED'
    assert result['physical_model_status']=='UNASSESSED' and result['physical_energy_bound'] is None


def test_large_common_scalar_does_not_create_absolute_tolerance_floor():
    raw=prepare();h=np.diag([1e-22,0.]);z=np.zeros((2,2));g=np.zeros((2,)*4)
    raw=replace(raw,physical=SpatialIntegrals(h,g,1e30),core=SpatialIntegrals(z,g,1e30),
                residual=SpatialIntegrals(h,g,0.))
    result=bound.core_only_bounds(raw)
    limit=result['bounds']['reference_R_H']['upper_bound']
    assert 0<limit<1e-20
    assert result['bounds']['formed_H_H_minus_C_H_minus_R_H']['upper_bound']==0


def test_exact_P_reference_and_formed_B_packet_cancellation_are_separate():
    raw=prepare();h=np.array([[0.,1e16+2],[1e16,0.]])
    c=np.array([[0.,1e16],[1e16,0.]]);g=np.zeros((2,)*4)
    raw=replace(raw,physical=SpatialIntegrals(h,g,0.),core=SpatialIntegrals(c,g,0.),
                residual=SpatialIntegrals(h-c,g,0.))
    result=bound.core_only_bounds(raw,hh_omission_budget=.1)
    assert result['bounds']['formed_R_H']['upper_bound']==0
    assert result['bounds']['reference_R_H']['upper_bound']>0
    assert result['bounds']['formed_R_H_minus_reference_R_H']['upper_bound']>0
    assert result['hh_budget_status']=='BOUND_INCONCLUSIVE'
    assert result['formal_physical_redefinition'] is False


def test_exact_zero_has_zero_bound_and_conditional_HH_budget_only():
    h,g,c,J,U=fixture();raw=prepare(np.zeros_like(h),np.zeros_like(g),0.,np.zeros_like(J),U)
    result=bound.core_only_bounds(raw,hh_omission_budget=1e-20)
    assert all(item['upper_bound']==0 for item in result['bounds'].values())
    assert result['hh_budget_status']=='WITHIN_DECLARED_B_REFERENCE_BUDGET'
    assert result['physical_target_status']=='DEFERRED' and result['physical_energy_bound'] is None


def test_raw_antihermitian_change_is_not_a_physical_energy_bound():
    raw=prepare();h=np.array([[0.,1e-22],[-1e-22,0.]]);g=np.zeros((2,)*4)
    raw=replace(raw,physical=SpatialIntegrals(h,g,raw.physical.nuclear_constant))
    result=bound.core_only_bounds(raw)
    assert result['bounds']['reference_H_H_minus_raw_H0']['upper_bound']>0
    assert not result['raw_H0_change_is_physical_energy_bound']
    assert result['physical_model_tolerance'] is None


def test_float_serialization_is_upward_from_exact_fraction():
    value=Fraction(1,3);record=bound._record(value)
    assert Fraction.from_float(record['upper_bound'])>=value
    assert Fraction(record['exact_fraction'])==value


@pytest.mark.parametrize('budget',[0.,-1.,float('nan')])
def test_invalid_budget_is_not_adopted(budget):
    with pytest.raises(ValueError,match='explicit H_H budget'):
        bound.core_only_bounds(prepare(),hh_omission_budget=budget)


def test_changed_reused_SOURCE_rejected_before_any_input_decode(tmp_path,monkeypatch):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
    runner=importlib.import_module('run_o1_m2_core_only_bound')
    fake=tmp_path/'src/trottertracks/o1_m2_hermitian_b.py';fake.parent.mkdir(parents=True)
    fake.write_text('changed artificial SOURCE')
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner.subprocess,'check_output',lambda *a,**k:b'fixed artificial SOURCE')
    from trottertracks import o1_m2_residual_adapter as adapter
    monkeypatch.setattr(adapter,'load_saved_pair',lambda *a,**k:pytest.fail('Input decoded before SOURCE check'))
    monkeypatch.setattr(sys,'argv',['runner','--core','unused-core','--input','unused-input','--output',str(tmp_path/'out.json')])
    with pytest.raises(ValueError,match='Fixed reused SOURCE differs'):runner.main()
    assert not (tmp_path/'out.json').exists()


def test_prepared_runner_records_runtime_and_leaves_budget_None(tmp_path,monkeypatch):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
    runner=importlib.import_module('run_o1_m2_core_only_bound')
    from trottertracks import o1_m2_residual_adapter as adapter
    monkeypatch.setattr(adapter,'load_saved_pair',lambda *a:object())
    def invented(raw,*,hh_omission_budget):
        assert hh_omission_budget is None
        return dict(physical_model_status='UNASSESSED',hh_budget_status='DEFERRED')
    monkeypatch.setattr(bound,'core_only_bounds',invented)
    output=tmp_path/'invented.json'
    monkeypatch.setattr(sys,'argv',['runner','--core','unused-core','--input','unused-input','--output',str(output)])
    assert runner.main()==0
    result=json.loads(output.read_text())
    assert result['source']['reused_base_commit']==runner.BASE
    assert set(result['runtime']['packages'])=={'numpy','scipy','openfermion','qiskit'}
    assert result['physical_model_status']=='UNASSESSED' and result['hh_budget_status']=='DEFERRED'
