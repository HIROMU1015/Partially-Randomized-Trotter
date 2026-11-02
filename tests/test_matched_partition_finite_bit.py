"""Small artificial fixtures only; no H4 numerical data or RNG draws."""
from fractions import Fraction as F
from itertools import product
import io
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest
from scipy.linalg import expm

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_h4_p3_finite_bit as runner
K=runner.modules()
X=np.array([[0.,1.],[1.,0.]])
Z=np.diag([1.,-1.]);I=np.eye(2)


def test_cdf_ties_repeated_endpoints_and_complete_small_grid():
    base,grid,info=K.choice_law([.25,0.,.25,.5],bits=4)
    assert grid==[F(1,4),0,F(1,4),F(1,2)] and info['counts']==[4,0,4,8]
    for vector in ([.3,.7],[.1,.2,.7],[0.,1.]):
        _,grid,info=K.choice_law(vector,bits=5)
        counts=[0]*len(vector)
        for m in range(32):counts[int(np.searchsorted(info['final_cdf'],m/32,side='right'))]+=1
        assert info['counts']==counts and sum(grid)==1
    with pytest.raises(ValueError):K.choice_law([0.,0.])
    with pytest.raises(ValueError):K.grid_masses([.7,.5,1.])


def test_scalar_rotation_intervals_and_exact_binary64_parameters():
    ideal,grid,rows=K.angle_table(.2)
    for k,a,b,row in zip(K.ORDERS,ideal,grid,rows):
        assert F(row['theta_exact'])==F(math.atan(.2/(k+1)))
        assert row['double_angle_is_exact']
        for value,interval in [(a[0],row['ideal_cos_interval_exact']),(b[0],row['fixed_angle_cos_interval_exact'])]:
            lo,hi=map(F,interval)
            assert abs(F(value)-(lo+hi)/2)<=F(max(a[2],b[2]))+(hi-lo)
        assert F(row['cos_difference_upper_exact'])>=0


def test_noncommuting_signed_order_moment_matches_all_events():
    ops=[X,-Z];p=[F(1,3),F(2,3)];pi=[F(1,2),F(1,3),F(1,6)]
    w=K.bundle(sum(float(x)*a for x,a in zip(p,ops)))
    coeff,_,_=K.angle_table(.17)
    actual=K.event_mean(w,K.moments(w),pi,coeff)
    explicit=np.zeros((2,2),complex)
    for k,mass,(c,s,*_) in zip(K.ORDERS,pi,coeff):
        for picks in product(range(2),repeat=k+1):
            result=I.copy()
            for j in picks[1:]:result=ops[j]@result
            result=(c*I-1j*s*ops[picks[0]])@result
            probability=mass*math.prod(p[j] for j in picks)
            explicit+=float(probability)*(-1)**(k//2)*result
    assert np.linalg.norm(actual.matrix-explicit)<actual.num+1e-15


def test_component_moments_reuse_saved_nonorthogonal_DF_U_without_repair():
    u=np.array([[1.,.2],[0.,1.]])
    d=dict(basis=[dict(U_real=u.tolist(),U_imag=np.zeros_like(u).tolist())],
           components=[dict(coefficient=-.3,basis=0,support=[0]),dict(coefficient=.2,basis=0,support=[0,1])])
    (w0,w1),bounds,info=K.component_moments('DF',d,[F(1,3),F(2,3)],[F(1,2),F(1,2)],2)
    assert w0.bind>0 and w1.bind>0 and bounds[1]>1
    assert info['DF_basis_count']==1 and not info['physical_operator_binding_difference_added']
    native=K.K.native_context(2)
    zs=[np.eye(4)-2*native(np.outer(u[:,i],u[:,i])).toarray() for i in range(2)]
    np.testing.assert_allclose(w0.matrix,-zs[0]/3+2*zs[0]@zs[1]/3,atol=1e-15,rtol=0)


def test_coefficient_gap_envelope_covers_nonunitary_matrix_difference():
    p0=[F(1,3),F(2,3)];p1=[F(2,5),F(3,5)];pi0=[F(1,2),F(1,3),F(1,6)];pi1=[F(1,3)]*3
    ops=[1.4*X,-1.2*Z];norms=[F(7,5),F(6,5)]
    w0=K.bundle(sum(float(p)*a for p,a in zip(p0,ops)))
    w1=K.bundle(sum(float(p)*a for p,a in zip(p1,ops)))
    c0,c1,rows=K.angle_table(.3)
    fs=[K.event_mean(w0,K.moments(w0),pi0,c0),K.event_mean(w1,K.moments(w1),pi1,c0),
        K.event_mean(w1,K.moments(w1),pi1,c1)]
    b=K.local_gap_bounds(p0,p1,pi0,pi1,norms,rows)
    assert np.linalg.norm(fs[1].matrix-fs[0].matrix,2)<=float(F(b['CDF_one_event_upper_exact']))
    assert np.linalg.norm(fs[2].matrix-fs[1].matrix,2)<=float(F(b['angle_one_event_upper_exact']))+fs[1].num+fs[2].num
    whole=K.full_gap_upper(b['CDF_one_event_upper_exact'],fs[0],fs[1],1.,0.,3.,q=2,r=2)
    direct=3*np.linalg.norm(np.linalg.matrix_power(fs[1].matrix,4)-np.linalg.matrix_power(fs[0].matrix,4),2)
    assert whole>=direct


def test_signal_plus_im_scalar_B_nu_and_separate_binding():
    phi=np.array([2.,1.j]);et=expm(-.07j*X);tail=K.bundle(expm(-.09j*Z),num=1e-14,bind=1e-8)
    phase=np.exp(-.2j)
    actual=K.corrected_signal(et,1e-14,tail,phi,F(5),phase,1e-15,3.,q=2,r=2)
    expected=3*np.vdot(phi,phase*np.linalg.matrix_power(et@tail.matrix@tail.matrix@et,2)@phi)/5
    assert abs(K.K.value(actual)-expected)<actual['u_num']
    assert actual['imag']==pytest.approx(expected.imag) and actual['u_bind']>0
    assert actual['B_multiplications']==actual['norm_divisions']==actual['scalar_phase_applications']==1


def test_direct_bias_does_not_recharge_nominal_split_or_delta_errors():
    signal=dict(real=.2,imag=.3,u_num=.01,u_bind=.02,u_ref_transfer=0.)
    s2=dict(reference=dict(physical_interval={a:dict(lower='0',upper='1/10') for a in ('real','imag')},transfer_uncertainty=.03),
            signals=dict(nom_P=dict(u_num=1000),split=dict(u_num=1000)))
    s4=dict(rows=[dict(arm=a,epsilon_sig='1',axes={axis:dict(epsilon_operand_exact='1/2') for axis in ('cosine','sine')}) for a in ('P','DF')])
    biases,rows=K.headroom_preview({a:dict(grid=signal) for a in ('P','DF')},s2,s4)
    assert F(biases['P']['axes']['cosine']['bias_upper_exact'])==F(.2)+F(.01)+F(.02)+F(.03)
    assert len(rows)==4 and all(r['actual_implemented_headroom'] is None for r in rows)
    signal['u_num']=None
    _,rows=K.headroom_preview({a:dict(grid=signal) for a in ('P','DF')},s2,s4)
    assert all(r['headroom'] is None and r['status']=='UNRESOLVED' for r in rows)


def test_six_signal_toy_pipeline_without_sampling_circuits_or_accounting():
    from qiskit import QuantumCircuit
    t=.2*Z;phi=np.array([2.,1.j]);nu=F(5);a=.3
    norm=K.K.normalization([a]);angle=.13
    et=expm(-1j*float(K.K.DELTA)/2*t)
    poly=K.K.finite_taylor_operator(a*Z,float(K.K.H),4)
    expected=np.vdot(phi,np.exp(-1j*angle)*np.linalg.matrix_power(et@np.linalg.matrix_power(poly,4)@et,8)@phi)/5
    nominal=dict(real=expected.real,imag=expected.imag,u_num=1e-9,u_bind=0.,u_ref_transfer=0.)
    s2=dict(phase=dict(binary64_angle=angle,engineering_error=1e-15),normalization={arm:norm for arm in ('P','DF')},
        signals={f'nom_{arm}':nominal for arm in ('P','DF')},
        reference=dict(physical_interval={key:dict(lower=str(F(x)),upper=str(F(x))) for key,x in [('real',expected.real),('imag',expected.imag)]},transfer_uncertainty=0.))
    descriptors={'P':dict(components=[dict(id='p',label='Z',coefficient=a)]),
        'DF':dict(components=[dict(id='p',basis=0,support=[0],coefficient=a)],basis=[dict(U_real=[[1.]],U_imag=[[0.]])])}
    laws={arm:dict(components=[dict(id='p',probability=1.)]) for arm in ('P','DF')}
    s4=dict(rows=[dict(arm=arm,epsilon_sig=str(j),axes={axis:dict(epsilon_operand_exact='1/100') for axis in ('cosine','sine')}) for j in range(5) for arm in ('P','DF')])
    with patch.object(np.random,'Generator',side_effect=AssertionError('RNG')), \
         patch.object(QuantumCircuit,'__init__',side_effect=AssertionError('CIRCUIT')):
        K.K.OPERATIONS.clear()
        result=K.evaluate(t,descriptors,laws,s2,s4,phi,nu)
    assert K.K.OPERATIONS['physical_complex_signals']==6
    assert result['status']=='COMPLETE_ENGINEERING' and len(result['headroom_preview'])==20
    assert all(row['status']=='POSITIVE_CONDITIONAL' for row in result['headroom_preview'])
    assert result['new_N_G'] is None and result['certified_gap_upper'] is None
    for arm in result['arms'].values():
        assert arm['bridge_status']=='CONSISTENT_ENGINEERING'
        assert len(arm['signals'])==3 and all(x['status']=='UNRESOLVED' for x in arm['differences']['grid']['axes'].values())


def test_runner_validates_synthetic_schema_and_native_inheritance():
    from stage1.pauli_signal_adapter import array_hash
    phi=np.zeros(256,complex);phi[0]=1.
    condition=dict(molecule='linear H4',basis='STO-3G',charge=0,multiplicity=1,system_qubits=8,
        dimension=256,distance_Angstrom=1.,T_exact=str(F(.8)),q=8,r=4,K=4,arms=['M_B','M_C'],DF='synthetic rank10')
    files={'synthetic':dict(ref='toy',sha256='toy')}
    norm={a:dict(B=1.) for a in ('P','DF')}
    data={'stage2/source_and_inputs.json':dict(condition=condition,commit='toy2',inputs=dict(files=files,state_hash=array_hash(phi),raw_norm_exact='1')),
        'stage4/source_and_inputs.json':dict(commit='toy4',scientific_lineage=dict(raw_norm_exact='1')),
        'stage2/STOP.json':dict(status='STOP_STAGE2_UNRESOLVED_COMPONENTS'),
        'run04/STOP.json':dict(status='STOP_STAGE3_PRICE_CONFIRMATION_COMPLETE_NO_AUTOMATIC_FOLLOWUP'),
        'stage4/STOP.json':dict(status='STOP_STAGE4_CONDITIONAL_ACCOUNTING_COMPLETE_NO_AUTOMATIC_FOLLOWUP'),
        'run02/preflight.json':{str(i):dict(status='PASS') for i in range(81)},
        'run02/source_and_inputs.json':{},'run02/native_bases.json':{},'run02/STOP.json':dict(status='old partial STOP'),
        'run02/law_bindings.json':{a:dict(saved_Stage2_normalization=norm[a]) for a in ('P','DF')},
        'stage2/results.json':dict(normalization=norm),
        'stage2/dictionaries.json':{a:dict(component_count=m,components=[{}]*m,basis=[{}]*10) for a,m in [('P',184),('DF',360)]},
        'stage4/results.json':dict(rows=[{}]*10),
        'state.json':dict(state=[[x.real,x.imag] for x in phi],state_hash=array_hash(phi))}
    saved={key:json.dumps(value).encode() for key,value in data.items()}
    source=dict(commit='toy3',conditions=dict(condition,partition='A'),input_files=files,
        pilot_result_commit='90bc02d458299212d3a1aa7c4bd34a0b98d1e0b5',native_SOURCE='9c4006bff3e76ea2ba7fbe4a10e679a8693c3a5c',
        saved_native_files_sha256={'artifacts/df_pauli_matched_partition_validation/2026-11-02/stage3/run02/'+key:runner.sha(saved['run02/'+key])
            for key in ('source_and_inputs.json','law_bindings.json','native_bases.json','preflight.json','STOP.json')})
    saved['run04/source_and_inputs.json']=json.dumps(source).encode()
    buffer=io.BytesIO();np.savez(buffer,T=np.zeros((256,256)))
    saved['stage2/common_operators.npz']=buffer.getvalue()
    values=runner.validate(saved,K)
    assert values[0].shape==(256,256) and values[-2]==1
    source['pilot_result_commit']='incorrect';saved['run04/source_and_inputs.json']=json.dumps(source).encode()
    with pytest.raises(ValueError,match='LINEAGE'):runner.validate(saved,K)
