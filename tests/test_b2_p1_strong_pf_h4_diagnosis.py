"""Artificial matrices only: exact conversion, contribution split and scope."""
from types import SimpleNamespace

import mpmath
import numpy as np
import pytest

from trotterlib import b2_p1_strong_pf_h4_diagnosis as m


@pytest.fixture(autouse=True)
def no_molecular_or_circuit_calls(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('No molecular/circuit/shot runs in synthetic tests')
    from qiskit import QuantumCircuit
    monkeypatch.setattr(np,'load',forbidden)
    monkeypatch.setattr(m.old.p,'load_private_inputs',forbidden)
    monkeypatch.setattr(m.old.saved,'fixed_json',forbidden)
    monkeypatch.setattr(QuantumCircuit,'__init__',forbidden)
    monkeypatch.setattr(m.old.strong.StrongBuilder,'sequence',forbidden)
    monkeypatch.setattr(m.old.p.connection.accounting,'_hoeffding_shots',forbidden)


def stored_records(native, car, psi, rounds):
    records={}
    for r in rounds:
        records[r]={'reproduction':dict(synthetic_only=True)}
        for name,matrix in (('N',native),('C',car)):
            power=np.linalg.matrix_power(matrix,1<<r)
            records[r][name]=dict(matrix=power,value=complex(np.vdot(psi,power@psi)))
    return records


def test_exact_binary_conversion_and_fixed_real_scope():
    ctx=mpmath.mp.clone();ctx.dps=120
    x=np.nextafter(.1,1.)
    value=m.hp_helpers().mp_matrix(ctx,np.array([[x+complex(0,x)]]))[0]
    n,d=x.as_integer_ratio()
    assert value.real==value.imag==ctx.mpf(n)/d
    assert sum(len(ms) for _,_,ms in m.SCOPE)==15
    assert [(f,s) for f,s,_ in m.SCOPE]==[('Full_S2',4),('Full_S2',8),('Yoshida4',2),('Yoshida4',4),('Yoshida4',8)]


@pytest.mark.parametrize('nonnormal',[False,True])
def test_power_split_precision_convergence_and_no_projection(nonnormal):
    native=np.diag([1.+2.**-46,-1.]).astype(complex)
    car=np.diag([1.,-1.]).astype(complex)
    if nonnormal:
        native[0,1]=.0001j
    psi=np.array([.6,.8j]) if nonnormal else np.array([1.,0.])
    rounds=(0,3,5)
    rows=m.power_diagnosis(native,car,psi,rounds,stored_records(native,car,psi,rounds))
    assert [r['m'] for r in rows]==list(rounds)
    ctx=mpmath.mp.clone();ctx.dps=120
    for r in rows:
        d=r['decomposition']
        assert ctx.mpf(d['identity_residual'])<ctx.mpf('1e-100')
        assert all(r['precision_check'][k]['precision_stable'] for k in ('N','C'))
        assert not d['qualification_changed']
        for name in ('N','C'):
            details=r['precision_check'][name]
            z=lambda v:ctx.mpc(v['real'],v['imag'])
            total=z(d['D_'+name+'_power'])
            partial=z(details['matrix_power_only_difference'])+z(details['contraction_rounding_difference'])
            assert abs(total-partial)<ctx.mpf('1e-100')
    if not nonnormal:
        assert ctx.mpf(rows[-1]['signals']['N_HP']['radius'])>1


def test_original_reproduction_rejects_even_small_changed_saved_value():
    a=np.diag([1.+2.**-46,1.]).astype(complex);b=np.eye(2,dtype=complex);psi=np.array([1.,0.])
    points=(0,2,3);records=stored_records(a,b,psi,points)
    prior=dict(rounds=[dict(m=i,status='OLD_STATUS',signals=dict(PF=m.old.p.signal_record(records[i]['N']['value'])),
        independent_PF=m.old.p.signal_record(records[i]['C']['value']),qualification=dict(independent_difference=
        m.old.p.signal_difference(records[i]['N']['value'],records[i]['C']['value']))) for i in points])
    new=m.reproduce64(a,b,psi,points,prior)
    assert all(new[i]['reproduction']['exact_saved_scalar_equality'] for i in points)
    prior['rounds'][-1]['signals']['PF']['real']+=1e-12
    with pytest.raises(ValueError,match='ORIGINAL_BINARY64_SIGNAL_NOT_REPRODUCED'):
        m.reproduce64(a,b,psi,points,prior)


@pytest.mark.parametrize('formula,s',m.old.ARMS)
def test_component_diagnostics_on_signed_artificial_generators(formula,s):
    one=np.kron(np.array([[.4,.13j],[-.13j,-.2]]),np.eye(2))
    gs=[np.kron(np.array([[1.,.17j],[-.17j,-.3]]),np.eye(2)),np.kron(np.array([[.3,.21],[.21,.6]]),np.eye(2))]
    weights=[.7,-.35]
    model,reports=m.old.strong.prepare_xdf_basis(one,weights,gs,.371,
        source_identity='toy',input_identity='toy',basis_policy=m.old.strong.SPIN)
    ref=m.old.p.car_reference();ann=ref.annihilators(4)
    generators=[ref.one_body(one,ann).toarray()]+[w*(ref.one_body(g,ann)@ref.one_body(g,ann)).toarray() for w,g in zip(weights,gs)]
    inputs=dict(ref=ref,indices=np.arange(16),ham=SimpleNamespace(one_body=one,g_matrices=gs,lambdas=weights))
    _,bases,energies,diag=m.gaussian_diagnostics(inputs,model,reports,generators)
    assert len(diag)==3 and all(x['signed_factor_preserved'] for x in diag)
    schedule=m.old.strong_baseline_schedule(3,substeps=s,formula=formula,delta_time=.02)
    d=m.schedule_diagnostics(inputs,model,schedule,bases,energies,{})
    assert all(x['primitive_vs_occupation_max_abs']<m.old.p.MATCH for x in d['unique_diagonal_groups'])
    assert all(x['homomorphism_max_abs']<m.old.p.MATCH for x in d['transitions'])
