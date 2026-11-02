"""Artificial small states only; no H6 numerical inputs or saved QPY walks."""
from fractions import Fraction as F
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import pytest
from qiskit import QuantumCircuit,transpile
from qiskit.quantum_info import SparsePauliOp,Statevector
from scipy.linalg import expm

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_h6_selected_implementation as runner
K=runner.modules()


def mat(label):return SparsePauliOp.from_list([(label,1.)]).to_matrix()


def event(products,rotation,theta,phase=1):
    applications=[NS(role='product',component_id=cid,coefficient_sign=sign) for cid,sign in products]
    applications.append(NS(role='rotation',component_id=rotation[0],coefficient_sign=rotation[1]))
    return NS(application_sequence=applications,rotation_angle=theta,phase=complex(phase))


def toy():
    events=[event([],('0',-1),.13),event([('1',-1),('2',1)],('2',-1),.07,-1),
            event([],('1',1),.13),event([],('0',1),.13)]
    spec=dict(n=2,q=2,r=2,delta=.17,retained=[dict(label='ZI',coefficient=.31),dict(label='IX',coefficient=-.23)],
              labels={'0':'XX','1':'YZ','2':'ZY'},events=events,scalar_step=.19,saved_body_phase=0.)
    raw=np.array([2.,1.j,.5,-.3j]);nu=sum((F(float(z.real))**2+F(float(z.imag))**2 for z in raw),F(0))
    return spec,raw,nu


def dense_target(spec,raw,nu):
    out=raw.copy()
    retained=[expm(-1j*float(F(x['coefficient'])*F(spec['delta'])/2)*mat(x['label'])) for x in spec['retained']]
    for step in range(spec['q']):
        for a in retained:out=a@out
        for e in spec['events'][step*spec['r']:(step+1)*spec['r']]:
            for a in e.application_sequence:
                op=a.coefficient_sign*mat(spec['labels'][a.component_id])
                out=(op if a.role=='product' else expm(-1j*e.rotation_angle*op))@out
            out*=e.phase
        for a in reversed(retained):out=a@out
        out*=np.exp(1j*spec['scalar_step'])
    return np.vdot(raw,out)/float(nu)


def wrapper(axis,theta=.31,phase=.27):
    c=QuantumCircuit(2,1);c.h(1);c.crz(2*theta,1,0);c.p(phase,1)
    if axis=='sine':c.sdg(1)
    c.h(1);c.measure(1,0)
    return c


def test_signed_pauli_mapping_and_little_endian():
    v=np.array([1.,2.j,-.7,3.1j])
    for label in ('XY','YX','IZ','ZI','YY'):
        for sign in (-1,1):np.testing.assert_array_equal(K.permute(v,K.mapping(label,sign)),sign*mat(label)@v)


def test_selected_noncommuting_products_scalar_and_nu_against_dense():
    spec,raw,nu=toy();expected=dense_target(spec,raw,nu)
    with patch.object(np.random,'Generator',side_effect=AssertionError('NO_SAMPLING')):
        result=K.selected_action(spec,raw,nu)
    actual=complex(result['real'],result['imag'])
    assert abs(actual-expected)<result['engineering_signal_uncertainty']
    assert result['operations']==dict(rotations=12,products=2,scalar_phases=2)
    assert result['norm_divisions']==1 and result['B_multiplications']==0 and not result['law_mean_used']


def test_retained_exact_operand_binding_and_scalar_only_boundary():
    c,s,u=K.rotation_coefficients(F(1,10)*F(1,3))
    assert u>0 and abs(c*c+s*s-1)<u
    spec,raw,nu=toy();spec['retained']=[]
    result=K.selected_action(spec,raw,nu)
    assert result['operations']['rotations']==4
    assert abs(complex(result['real'],result['imag'])-dense_target(spec,raw,nu))<result['engineering_signal_uncertainty']


@pytest.mark.parametrize('axis',['cosine','sine'])
def test_saved_observable_plus_im_and_full_global_phase_invariance(axis):
    raw=np.array([2.,1.j]);nu=F(5);c=wrapper(axis)
    z=np.vdot(raw,np.exp(.27j)*expm(-.31j*mat('Z'))@raw)/5
    value=K.saved_action(c,raw,nu,compiled=False)
    expected=z.real if axis=='cosine' else z.imag
    assert abs(value['signal']-expected)<value['engineering_signal_uncertainty']
    phase=c.copy();phase.global_phase=.73
    other=K.saved_action(phase,raw,nu,compiled=False)
    assert abs(value['signal']-other['signal'])<value['engineering_signal_uncertainty']+other['engineering_signal_uncertainty']
    assert value['norm_divisions']==1 and value['B_multiplications']==0


def test_artificial_compiler_pair_and_structural_rejections():
    raw=np.array([2.,1.j]);c=wrapper('sine');compiled=transpile(c,basis_gates=['rz','sx','x','cx'],optimization_level=1,seed_transpiler=17)
    before=K.saved_action(c,raw,F(5),compiled=False);after=K.saved_action(compiled,raw,F(5),compiled=True)
    assert abs(before['signal']-after['signal'])<before['engineering_signal_uncertainty']+after['engineering_signal_uncertainty']
    bad=c.copy();bad.reset(0)
    with pytest.raises(ValueError,match='TERMINAL'):K.validate_circuit(bad,compiled=False,n=1)
    with pytest.raises(ValueError,match='FAMILY'):K.validate_circuit(c,compiled=True,n=1)


def test_exact_B_once_difference_triangle_and_noncertified_local_budget():
    selected=dict(real=.2,imag=-.4,engineering_signal_uncertainty=1e-9)
    native=dict(signal=.200000001,engineering_signal_uncertainty=2e-9)
    compiled=dict(signal=.200000002,engineering_signal_uncertainty=3e-9)
    r=K.reduce_axis(selected,native,compiled,'cosine','3','1/1000')
    assert F(r['d_exact'])==3*abs(F(compiled['signal'])-F(selected['real']))
    assert F(r['d_exact'])<=F(r['eta_exact'])+F(r['delta_exact'])
    assert r['engineering_comparison']=='ENGINEERING_UPPER_WITHIN_SAVED_LOCAL_BUDGET'
    assert r['actual_compiled_rho'] is None and r['certified_upper'] is None and r['rho_upper'] is None
    r=K.reduce_axis(selected,native,dict(compiled,signal=.3),'cosine','3','1/1000')
    assert r['engineering_comparison']=='ENGINEERING_LOWER_EXCEEDS_SAVED_LOCAL_BUDGET'


def test_missing_or_large_resolution_is_not_filled_with_zero():
    a=dict(real=0.,imag=0.,engineering_signal_uncertainty=None);b=dict(signal=0.,engineering_signal_uncertainty=1.)
    r=K.reduce_axis(a,b,b,'sine','2','1/10')
    assert r['engineering_comparison']=='UNRESOLVED_MISSING_RESOLUTION' and r['engineering_intervals'] is None
    a['engineering_signal_uncertainty']=1.
    r=K.reduce_axis(a,b,b,'sine','2','1/10')
    assert r['engineering_comparison']=='UNRESOLVED_ENGINEERING'


def test_fifteen_unique_jobs_and_R64_shared_between_comparisons():
    laws={k:{} for k in runner.P2.CANDIDATES};results={k:{} for k in laws};prefixes={k:k+'/' for k in laws}
    members={p+'prices/confirmation/0000/'+a+'_'+b+'.qpy.gz':b'toy' for p in prefixes.values() for a in ('cosine','sine') for b in ('input','compiled')}
    tasks=runner.jobs(laws,results,members,prefixes)
    assert len(tasks)==len({x['key'] for x in tasks})==15
    assert sum(x['role']=='selected' for x in tasks)==3
    assert sum(x['tag']=='R64' for x in tasks)==5
