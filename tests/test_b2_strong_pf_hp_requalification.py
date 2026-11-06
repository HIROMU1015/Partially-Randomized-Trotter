"""Artificial tests only for the fixed-five-arm qualification adapter."""
from types import SimpleNamespace
import json
import mpmath
import numpy as np
import pytest

from trotterlib import b2_strong_pf_hp_requalification as m


@pytest.fixture(autouse=True)
def no_science_loaders_or_circuits(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Synthetic-only test')
    from qiskit import QuantumCircuit
    monkeypatch.setattr(np,'load',forbidden)
    monkeypatch.setattr(m.hp.legacy.p,'load_private_inputs',forbidden)
    monkeypatch.setattr(m.hp.legacy.saved,'fixed_json',forbidden)
    monkeypatch.setattr(QuantumCircuit,'__init__',forbidden)
    monkeypatch.setattr(m.hp.legacy.p.connection.accounting,'_hoeffding_shots',forbidden)


def test_exact_private_ipc_without_float_round_trip():
    a=mpmath.mp.clone();a.dps=120;b=mpmath.mp.clone();b.dps=120
    matrix=a.matrix([[a.mpf(1)+a.mpf(2)**-300,a.j/7],[0,a.pi]])
    unpacked=m.unpack_matrix(b,m.pack_matrix(a,matrix))
    assert all(x._mpc_==y._mpc_ for x,y in zip((a.mpc(v) for v in matrix),(b.mpc(v) for v in unpacked)))


def test_input_gate_rejects_nonhermitian_without_repair():
    h=np.eye(4);g=2*np.eye(4);indices=np.array([3,6,9,12])
    ham=SimpleNamespace(one_body=h,g_matrices=(g,),n_qubits=4)
    assert m.input_gate(ham,indices)['status']=='PASS'
    h[0,2]=1e-20
    result=m.input_gate(ham,indices)
    assert result['status']=='STOP_INPUT_INELIGIBLE'
    assert result['blocks'][0]['Hermiticity_max_abs_entry']==1e-20
    assert h[2,0]==0  # No symmetrization or fallback.


def test_fixed_five_arm_scope_and_ascending_stop():
    assert m.ARMS==( ('Full_S2',4),('Full_S2',8),('Yoshida4',2),('Yoshida4',4),('Yoshida4',8))
    seen=[]
    def evaluate(i):
        seen.append(i);return dict(m=i,qualification=dict(feasible=i<3))
    rows,status=m.evaluate_in_order(evaluate,lambda rows:None)
    assert seen==[0,1,2,3] and status=='STOP_ARM_INELIGIBLE'
    rows,status=m.evaluate_in_order(lambda i:dict(m=i,qualification=dict(feasible=True)),lambda rows:None)
    assert len(rows)==18 and status=='P1_ALL18_HP_CONDITIONAL_SIGNAL_PASS'


def test_HP_checks_and_old_radius_allocation_are_both_enforced():
    ctx=mpmath.mp.clone();ctx.dps=120
    z=ctx.mpf('.9')*ctx.exp(ctx.j*ctx.mpf('.01'));z64=complex(float(z.real),float(z.imag))
    q=m.qualify_hp(0,z,z,z64,z64,{'N':True,'C':True},ctx)
    assert q['feasible'] and q['finite_RTE_complex_disk']==0
    assert q['RTE_budget_reserved_unreallocated']==.04
    assert ctx.mpf(q['HP_bounds']['PF_radius_lower'])>0
    assert q['phase_total_with_reserved_statistical_and_unused_RTE']<.4
    fail=m.qualify_hp(0,z*ctx.exp(ctx.j*ctx.mpf('.061')),z,z64,z64,{'N':True,'C':True},ctx)
    assert not fail['feasible'] and not fail['HP_tests']['PF']
    fail=m.qualify_hp(0,z,z,z64,z64,{'N':False,'C':True},ctx)
    assert not fail['feasible'] and 'HP_precision' in fail['reasons']
    assert not m.qualify_hp(0,ctx.mpc(0),z,z64,z64,{'N':True,'C':True},ctx)['feasible']


def test_full_artificial_round_connection_and_public_serialization():
    builds={}
    for digits in m.PRECISIONS:
        ctx=mpmath.mp.clone();ctx.dps=digits
        phase=ctx.exp(-ctx.j*ctx.mpf(float(.02))*ctx.mpf('.125'))
        matrix=ctx.diag([phase,ctx.conj(phase)])
        builds[digits]=dict(native=m.pack_matrix(ctx,matrix),CAR=m.pack_matrix(ctx,matrix))
    references=[]
    for index in range(18):
        z=complex(np.exp(-1j*(1<<index)*.02*.125));sig=m.hp.legacy.p.signal_record(z)
        references.append(dict(signals=dict(phys=sig,DF=sig),evidence='synthetic'))
    rows,status=m.evaluate_arm(builds,np.array([1.,0.]),references,lambda rows:None)
    assert status=='P1_ALL18_HP_CONDITIONAL_SIGNAL_PASS' and len(rows)==18
    assert all(r['qualification']['feasible'] for r in rows)
    json.dumps(rows,allow_nan=False)
