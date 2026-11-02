"""Small artificial native programs only. No molecular input or scientific replay."""
from dataclasses import replace
from fractions import Fraction as F
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import sys

import numpy as np
import pytest
from qiskit import QuantumCircuit,qpy
from qiskit.quantum_info import Operator,Statevector

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import run_h4_p3_stage3 as runner
K=runner.modules()
from stage1.rte import _make_event


def tiny(name):
    one=np.array([[.4,.12],[.12,-.23]],dtype=complex)
    t=K.native_context(2)(one).toarray()
    retained=K.retained_preflight(one,t,q=2)
    rotations=[]
    for angle in (.19,-.31):
        c,s=np.cos(angle),np.sin(angle)
        rotations.append(np.array([[c,-s],[s,c]],dtype=complex))
    bases={i:K.basis_from_saved(u) for i,u in enumerate(rotations)}
    if name=='P':
        rows=[dict(id='p0',label='YX',coefficient=-.23),dict(id='p1',label='ZY',coefficient=.17)]
    else:
        rows=[dict(id='d0',fragment=0,basis=0,support=[0],coefficient=-.21),
              dict(id='d1',fragment=0,basis=0,support=[0,1],coefficient=.13),
              dict(id='d2',fragment=1,basis=1,support=[1],coefficient=.07)]
    saved=dict(components=rows,basis=[dict(fragment=i,U_real=u.real.tolist(),U_imag=u.imag.tolist())
                                    for i,u in enumerate(rotations)])
    lam=K.fsum(abs(x['coefficient']) for x in rows);a=lam*.1
    dist=K.finite_rte_distribution(a,4)
    bundle=dict(lambda_R=lam,dimensionless_step_time=a,weights=list(dist.unnormalized_order_weights),
        order_probabilities=list(dist.order_probabilities),beta=dist.exact_finite_distribution)
    inputs=K.component_inputs(name,saved,bundle,bases,2)
    # Includes every even order, signed rotations/products, basis switches and reuse.
    orders=(0,1,2,0,2,1,0,0)
    events=tuple(_make_event(tuple((j+i)%len(rows) for j in range(dist.orders[k]+1)),
        inputs.components,dist,k) for i,k in enumerate(orders))
    return inputs,saved,retained,events,K.scalar_angle(.13,.21)


def test_gaussian_complex_orientation_actual_adjoint_and_native_no_fallback():
    angle=.29;c,s=np.cos(angle),np.sin(angle)
    u=np.array([[c,1j*s],[1j*s,c]])
    basis=K.basis_from_saved(u)
    assert all(x['status']=='PASS' for x in basis.checks.values())
    assert all(g.name!='unitary' for g,qs in basis.ops)
    np.testing.assert_allclose(basis.inverse_matrix,basis.matrix.conj().T,atol=1e-14)


@pytest.mark.parametrize('name',['P','DF'])
def test_mixed_gaussian_residual_parts_phase_and_native_oracle(name):
    inputs,saved,retained,events,angle=tiny(name)
    parts,sidecar=K.trajectory_parts(inputs,events,retained,angle)
    oracle=K.DictionaryOracle(inputs,saved,2)
    for part,body in parts.items():
        expected,error=oracle.trajectory(events,retained,angle,part=part)
        row=K.program_check(body,expected,error)
        assert row['status']=='PASS' and row['spectral_difference']<5e-14
        actual=Operator(body).data
        np.testing.assert_allclose(actual[:4,:4],np.eye(4),atol=1e-14)
        np.testing.assert_allclose(actual[:4,4:],0,atol=1e-14)
    assert len(sidecar)==2
    assert {e.taylor_order for e in events}=={0,2,4}
    shared=K.shared_parts(retained,angle)
    expected,error=oracle.trajectory((),retained,angle,part='R')
    assert K.program_check(shared['R'],expected,error)['status']=='PASS'
    assert K.program_check(shared['Empty'],np.eye(8),0.)['status']=='PASS'
    # An outer scalar mutation must not be accepted by phase alignment.
    mutated=parts['F'].copy();mutated.p(.03,2)
    expected,error=oracle.trajectory(events,retained,angle,part='F')
    assert K.program_check(mutated,expected,error)['status']=='FAIL'


@pytest.mark.parametrize('name',['P','DF'])
@pytest.mark.parametrize('axis',['cosine','sine'])
def test_wrapper_plus_im_and_four_artificial_compiler_regressions(name,axis):
    inputs,saved,retained,events,angle=tiny(name)
    parts,_=K.trajectory_parts(inputs,events,retained,angle)
    body=parts['F'];circuit=K.wrappers(body)[axis]
    u=Operator(body).data[4:,4:]
    # This fixture alone uses |00>, to check wrapper sign; no molecular state.
    expected=u[0,0].real if axis=='cosine' else u[0,0].imag
    native=Statevector.from_instruction(circuit.remove_final_measurements(inplace=False))
    probs=native.probabilities();value=float(sum(probs[:4])-sum(probs[4:]))
    assert value==pytest.approx(expected,abs=1e-13)
    output,metrics=K.price(circuit)
    assert output.num_qubits==3 and metrics['compiler']['optimization_level']==1
    before=Operator(circuit.remove_final_measurements(inplace=False)).data
    after=Operator(output.remove_final_measurements(inplace=False)).data
    # The target includes measurement. Compare its observable without phase
    # alignment; compiler global phase is preserved verbatim in the QPY.
    ancilla_z=np.diag([1.]*4+[-1.]*4)
    np.testing.assert_allclose(after.conj().T@ancilla_z@after,
        before.conj().T@ancilla_z@before,atol=1e-12,rtol=0)
    assert set(metrics['output_histogram'])<=set(('rz','sx','x','cx','measure','barrier'))


def test_saved_law_binding_seed_sampling_and_component_order():
    inputs,saved,retained,events,angle=tiny('DF')
    sampled=K.sample_rte_events(inputs.components,inputs.distribution,sample_count=8,seed=321)
    assert len(sampled)==8
    assert [c.component_id for c in inputs.components]==[x['id'] for x in saved['components']]
    row=runner.compact_event(sampled[0])
    assert all('basis_change_operations' not in x for x in row['application_sequence'])
    assert set(K.SEEDS['P']).isdisjoint(K.SEEDS['DF'])
    for e in sampled:assert e.selected_component_ids==(*e.product_component_ids,e.rotation_component_id)
    bundle=dict(lambda_R=.1,dimensionless_step_time=.01,weights=[],order_probabilities=[],beta=1.)
    with pytest.raises(ValueError,match='LAMBDA_BINDING'):K.component_inputs('DF',saved,bundle,{},2)


def test_basis_reuse_is_confined_to_residual_sequence_and_independent_oracle():
    inputs,saved,retained,events,angle=tiny('DF')
    d=inputs.distribution
    event=_make_event((0,0,0),inputs.components,d,1)
    sequence,meta=K.residual_sequence(inputs,(event,event))
    assert meta['cancelled_basis_change_pairs']>0
    oracle=K.DictionaryOracle(inputs,saved,2)
    small=NS(reference=np.eye(4),reference_error=0.,q=1)
    expected,error=oracle.trajectory((event,event),small,angle,part='E',r=2)
    assert K.program_check(sequence,expected,error)['status']=='PASS'


def test_d2_rejects_mixed_gaussian_body_instead_of_generic_fallback():
    inputs,saved,retained,events,angle=tiny('P')
    with pytest.raises(ValueError,match='D2_UNSUPPORTED'):
        K.central_control_body(retained.half)


def test_preflight_nonpass_keeps_diagnostic_and_never_phase_aligns():
    with pytest.raises(ValueError) as caught:
        K.require_pass({'bad':dict(status='FAIL',residual=.1)})
    assert caught.value.checks['bad']['residual']==.1
    assert K.classification(.5,1.)=='PASS'
    assert K.classification(2.,1.)=='UNRESOLVED'
    assert K.classification(5.,1.)=='FAIL'
    with pytest.raises(ValueError):K.classification(float('nan'),1.)


def test_gzip_qpy_is_lossless(tmp_path):
    import gzip
    c=QuantumCircuit(2);c.h(0);c.cx(0,1);c.global_phase=.37
    with patch.object(runner,'OUT',tmp_path):row=runner.save_qpy('toy.qpy.gz',c)
    with gzip.open(tmp_path/'toy.qpy.gz','rb') as f:restored=qpy.load(f)[0]
    assert K.canonical_qiskit_circuit_fingerprint(c)==K.canonical_qiskit_circuit_fingerprint(restored)
    assert row['decoded_bytes']>row['stored_bytes']
