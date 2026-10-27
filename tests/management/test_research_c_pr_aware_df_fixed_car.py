"""Invented J/U, events and QPY only; never decode molecular scientific input."""
from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_pr_aware_df_fixed_car as c
from trottertracks import c_pr_aware_df_fixed_car_execution as e
from trottertracks import c_pr_aware_df_fixed_certificate as v
from trottertracks import c_pr_aware_df_zsxx_enclosure as z
from trottertracks import c_pr_aware_df_stage_b as b


@pytest.fixture
def example():
    from trotterlib.rte import _make_event
    h=np.array([[.13,.04],[.04,-.2]])
    J=np.array([[[.17,.06],[.06,-.21]],[[.11,-.03],[-.03,.07]],
                [[-.3,.08],[.08,.23]],[[.16,-.07],[-.07,-.19]]])
    def R(t):return np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
    U=np.array([R(.21),R(-.37),R(.49),R(-.62)])
    arm=b.prepare_arm(h,np.zeros((2,)*4),.19,J,U,role='baseline',
                      source_identity='artificialJU',input_identity='artificialInput',ld=2)
    tail,config,dist=b.configure(arm,F(1,40),16,0)
    # Predetermined, noncommuting bases; both coefficient signs. No sampler.
    choices=[next(i for i,a in enumerate(arm.plan.atoms) if a.coefficient<0),
             next(i for i,a in enumerate(arm.plan.atoms) if a.frame_id.endswith('F3') and len(a.support)==2)]
    events=tuple(_make_event((choices[i%2],),tail.components,dist,0) for i in range(16))
    return arm,h,events,config


def build_raw(arm,events,config):
    from trottertracks.c_pr_aware_df_stage_b2_compat import SharedQiskitCalls
    lock=SharedQiskitCalls(lambda k,l,f:f())
    return lock('kernel_build','invented CAR',lambda:b.build_kernel(arm,F(1,40),events,config))


@pytest.fixture
def certificates(example):
    arm,h,events,config=example
    ref=c.build_reference(arm,h,.19,events,parallel=False)
    raw=build_raw(arm,events,config)
    record=v.record(raw)
    result=c.compare(record,ref,parallel=False)
    return record,ref,result


def test_actual_independent_CAR_native_trajectory_full_Fock_roundtrip(certificates):
    record,ref,result=certificates
    assert result['dimension']==result['columns_evaluated']==32
    assert F(result['upper_exact'])<F(1,10**10)
    assert ref['reference_independent_of_QPY'] and not ref['attenuation_applied']
    assert F(ref['bridge']['upper_exact'])>0  # approximate U never exact-unitarized silently
    assert result['whole_law_bound'] is result['physical_Hamiltonian_bound'] is None
    assert any(F(x['raw_evaluation_radius_exact'])>0 for x in result['columns'])


def test_opt_in_arithmetic_reuses_binary64_CAR_event_path(example):
    from scipy.sparse import csr_matrix
    from flint import ctx
    arm,h,events,_=example
    full=b.stage1.car_bilinears(4)
    G=[b.stage1.raw_base(h,.19,full)]+[b.stage1.raw_frame(J,U,full)[0] for J,U in zip(arm.J,arm.U)]
    ctx0=SimpleNamespace(d=16,bilinears={(p,q):csr_matrix(full[p,q]) for p in range(4) for q in range(4) if p%2==q%2})
    ops=b.atom_operators(ctx0,arm)
    numeric=b.trajectory_reference(arm,G,ops,F(1,40),events,object())
    ref=c.build_reference(arm,h,.19,events,parallel=False)
    with ctx.workprec(192):
        for block in ref['blocks']:
            idx=block['indices'];A=c.unpack(block['entries'],len(idx))
            assert F(z._norm(A-z._matrix(numeric[np.ix_(idx,idx)]))['upper_exact'])<F(1,10**12)
    # This diagnostic is not used as the enclosure of the mathematical CAR map.
    assert ref['status']=='VALIDATED_LITERAL_CAR_WITH_ANALYTIC_POLAR_BRIDGE'


@pytest.mark.parametrize('fault',['phase','order','gaussian','control'])
def test_independent_reference_detects_wrong_phase_event_order_basis_control(example,fault):
    arm,h,events,config=example
    ref=c.build_reference(arm,h,.19,events,parallel=False)
    if fault=='order':raw=build_raw(arm,tuple(reversed(events)),config)
    elif fault=='gaussian':
        U=arm.U.copy();U[2]=np.eye(2)
        # Change independent target basis only; native emission stays original.
        altered=b.prepare_arm(h,np.zeros((2,)*4),.19,arm.J,U,role='baseline',
            source_identity='artificialChanged',input_identity='artificialInput',ld=2)
        ref=c.build_reference(altered,h,.19,events,parallel=False)
        raw=build_raw(arm,events,config)
    else:raw=build_raw(arm,events,config)
    record=v.record(raw)
    if fault=='phase':record['phase_hex']=.1.hex()
    if fault=='control':record['gates'].append(dict(name='x',qubits=[4],params_hex=[]))
    result=c.compare(record,ref,parallel=False)
    assert F(result['lower_exact'])>F(1,10**8)


def test_polar_error_bound_covers_nonorthogonal_density_not_silent_projection(example):
    from flint import ctx
    arm,_,events,_=example
    arm.U[2]*=1.0001
    with ctx.workprec(192):bound=c.polar_bridge(arm,F(1,40),events)
    assert F(bound['upper_exact'])>F(1,10**7)
    assert bound['input_nonunitarity_included'] and not bound['input_U_replaced']
    arm.U[2]*=3
    with ctx.workprec(192),pytest.raises(ValueError,match='POLAR_BASIS'):c.polar_bridge(arm,F(1,40),events)


def test_full512_controlled_all_basis_and_scalar_vacuum_phase():
    from qiskit import QuantumCircuit
    h=np.zeros((4,4));J=np.zeros((3,4,4));U=np.array([np.eye(4)]*3)
    arm=b.prepare_arm(h,np.zeros((4,)*4),.125,J,U,role='baseline',source_identity='fake',input_identity='fake',ld=2)
    ref=c.build_reference(arm,h,.125,(),parallel=False)
    q=QuantumCircuit(9);q.p(-.125/40,8)
    result=c.compare(v.record(q),ref,parallel=True)
    assert result['columns_evaluated']==result['dimension']==512
    assert F(result['upper_exact'])<F(1,10**15)
    assert sorted(i for block in ref['blocks'] for i in block['indices'])==list(range(256))


def test_reference_ball_uncertainty_is_retained_once(certificates):
    from flint import ctx
    record,ref,base=certificates
    uncertain=deepcopy(ref)
    # Widen a target amplitude interval; never promote its midpoint to exact.
    pair=uncertain['blocks'][0]['entries'][0][0]
    pair[0]=str(F(pair[0])-F(1,100));pair[1]=str(F(pair[1])+F(1,100))
    result=c.compare(record,uncertain,parallel=False)
    assert F(result['upper_exact'])>F(base['upper_exact'])+F(1,200)
    assert result['reference_uncertainty'].startswith('included')
    with ctx.workprec(192),pytest.raises(ValueError,match='ENDPOINT_ORDER'):
        c.unpack([[[1,0],[0,0]]],1)


def test_three_way_triangle_requires_qualification_raw_identity_and_phase(certificates):
    record,ref,result=certificates
    pair=v.compare(record,record,parallel=False)
    triangle=c.compose(pair,result,raw_identity='a',expected_raw_identity='a')
    assert F(triangle['upper_exact'])==F(pair['upper_exact'])+F(result['upper_exact'])
    assert c.compose(pair,None,raw_identity='a',expected_raw_identity='a')['upper_exact'] is None
    with pytest.raises(ValueError,match='RAW_IDENTITY'):c.compose(pair,result,raw_identity='b',expected_raw_identity='a')
    with pytest.raises(ValueError,match='QUALIFICATION'):
        c.compose(dict(pair,status='binary64 diagnostic'),result,raw_identity='a',expected_raw_identity='a')


def artificial_payload(example):
    arm,h,events,config=example
    return dict(arm='baseline',sample=0,SOURCE_sha256='samplingsource',final_JU_sha256='export',
        cohort='B2_fixed32_per_arm_v1',microsteps=16,quantum_shots=0,reused_as_coherent_repetitions=False,
        trajectory_key=[[0,x.rotation_component_id,[]] for x in events],events=[x.to_dict() for x in events])


def test_actual_saved_event_schema_signed_negative_atoms_roundtrip(example):
    arm,h,events,_=example;p=artificial_payload(example)
    e.trajectory_gate(p,role='baseline',source_digest='samplingsource',export_digest='export')
    assert e.restore_events(p,arm)==events
    assert any(x.application_sequence[-1].coefficient_sign==-1 for x in events)
    p['events'][0]['application_sequence'][0]['coefficient_sign']*=-1
    with pytest.raises(ValueError,match='SIGNED_ATOM'):e.restore_events(p,arm)


@pytest.mark.parametrize('key,value',[('arm','proposed'),('sample',1),('SOURCE_sha256','other'),
    ('final_JU_sha256','other'),('cohort','same index does not imply same law'),('reused_as_coherent_repetitions',True)])
def test_source_and_sampling_law_mixture_rejected(example,key,value):
    p=artificial_payload(example);p[key]=value
    with pytest.raises(ValueError,match='IDENTITY'):
        e.trajectory_gate(p,role='baseline',source_digest='samplingsource',export_digest='export')


@pytest.mark.parametrize('bad',['nonhermitian','nonfinite'])
def test_invalid_generator_stops_explicitly(example,bad):
    arm,h,events,_=example
    h=h.copy();h[0,1]=.12 if bad=='nonhermitian' else np.nan
    with pytest.raises(ValueError,match='HERMITIAN'):c.build_reference(arm,h,.19,events,parallel=False)


def test_F1_synthetic_E2E_7calls_and_stop_on_compile_mismatch(tmp_path):
    from trottertracks.c1_stage3_integrated import Calls
    record=dict(v.identity(2),gates=[dict(name='rz',qubits=[0],params_hex=[.2.hex()])])
    records=dict(raw=record,compiled=record,raw_adjoint=v.adjoint(record),compiled_adjoint=v.adjoint(record))
    contract=dict(circuits={x:x for x in records},allowance_exact='1/100000000',arm='proposed')
    call=Calls(tmp_path,e.LIMITS['F1'],7)
    result=e.execute_F1(tmp_path,contract,tmp_path,call,loader=lambda p,**k:records[p.name],compare=lambda a,b:v.compare(a,b,parallel=False))
    assert result['status']=='COMPLETE' and sum(call.counts.values())==7
    fail=tmp_path/'fail';fail.mkdir();call=Calls(fail,e.LIMITS['F1'],7)
    records['compiled']=dict(record,phase_hex=.1.hex())
    with pytest.raises(ValueError,match='ALLOWANCE_FAIL'):
        e.execute_F1(tmp_path,contract,fail,call,loader=lambda p,**k:records[p.name],compare=lambda a,b:v.compare(a,b,parallel=False))
    assert sum(call.counts.values())==5 and call.stop.is_set()
    assert (fail/'compiled_raw.json').exists() and not (fail/'raw_actual_adjoint.json').exists()


def test_F2_artificial_reference_end_to_end_7calls(tmp_path,certificates):
    from trottertracks.c1_stage3_integrated import Calls
    record,ref,result=certificates
    prior=v.compare(record,record,parallel=False)
    (tmp_path/'prior.json').write_text(json.dumps(prior))
    c0=dict(experiment='F2',arm='baseline',allowance_exact='1/100000000',circuits=dict(raw='fake.qpy'),
        saved_inputs={'fake.qpy':'hash'},reuse=dict(compiled_raw='prior.json',raw_sha256='hash'))
    call=Calls(tmp_path,e.LIMITS['F2'],7)
    def prepare(root,contract,call):
        for kind in ['input_arrays','arm_replay','trajectory_load']:call(kind,'invented',lambda:None)
        return call('CAR_reference','invented',lambda:ref)
    r=e.execute_CAR(tmp_path,c0,tmp_path,call,reference_loader=prepare,
        loader=lambda *a,**k:record,compare=lambda a,b:c.compare(a,b,parallel=False))
    assert r['status']=='COMPLETE' and sum(call.counts.values())==7
    assert r['CA_achieved'] is r['whole_law_bound'] is None


def test_F2_unresolved_width_does_not_compose_or_claim_CA(tmp_path,certificates):
    from trottertracks.c1_stage3_integrated import Calls
    record,ref,result=certificates
    c0=dict(experiment='F2',arm='baseline',allowance_exact='1/100000000',circuits=dict(raw='fake.qpy'))
    call=Calls(tmp_path,e.LIMITS['F2'],7)
    r=e.execute_CAR(tmp_path,c0,tmp_path,call,reference_loader=lambda *a:ref,loader=lambda *a,**k:record,
        compare=lambda *a:dict(status='VALIDATED_FIXED_RAW_IDEAL_CAR',lower_exact='0',upper_exact='1'))
    assert r['status']=='UNRESOLVED' and 'triangle_composition' not in call.counts


def test_saved_dyadic_matrix_is_category_A_never_ideal_CAR(certificates):
    record,ref,result=certificates
    from qiskit.quantum_info import Operator
    # A saved arbitrary nonunitary matrix is accepted only as a fixed number.
    numeric=np.eye(16)*1.01
    r=c.compare_saved_numeric(record,numeric,parallel=False)
    assert r['status']=='VALIDATED_FIXED_SAVED_NUMERICAL_MATRIX_ONLY'
    assert r['ideal_CAR_bound'] is r['reference_uncertainty'] is None
    with pytest.raises(ValueError,match='QUALIFICATION'):
        c.compose(v.compare(record,record,parallel=False),r,raw_identity='a',expected_raw_identity='a')


def test_reuse_same_shell_certificates_requires_identity_and_qualified_values(tmp_path):
    from trottertracks.c1_stage1_execution import save,sha
    p=tmp_path/'certificate.json'
    save(p,dict(gate_status='VALIDATED_WITHIN_FIXED_ALLOWANCE',phase_fit=False,precision_bits=192,upper_exact='1/10000000000'))
    contract=dict(allowance_exact='1/100000000',saved_inputs={'certificate.json':sha(p)},
        reuse=dict(files=['certificate.json'],certificates={'head':'certificate.json'}))
    assert e.reuse_gate(tmp_path,contract)['recomputed']==0
    p.write_text('{}')
    with pytest.raises(ValueError,match='SOURCE_OR_INPUT'):e.reuse_gate(tmp_path,contract)


def test_F2_correctness_failure_preserves_partial_and_stops_before_triangle(tmp_path,certificates):
    from trottertracks.c1_stage3_integrated import Calls
    record,ref,result=certificates
    call=Calls(tmp_path,e.LIMITS['F2'],7)
    contract=dict(experiment='F2',arm='baseline',allowance_exact='1/100000000',circuits=dict(raw='fake.qpy'))
    with pytest.raises(ValueError,match='ALLOWANCE_FAIL'):
        e.execute_CAR(tmp_path,contract,tmp_path,call,reference_loader=lambda *a:ref,
            loader=lambda *a,**k:record,compare=lambda *a:dict(lower_exact='1/100',upper_exact='1'))
    assert call.stop.is_set() and 'triangle_composition' not in call.counts
    assert (tmp_path/'CAR_reference.json').exists() and (tmp_path/'raw_CAR.json').exists()


def test_passive_preflight_never_decodes_arrays_or_QPY(tmp_path,example,monkeypatch):
    from trottertracks.c1_stage1_execution import save,sha
    certificate=dict(gate_status='VALIDATED_WITHIN_FIXED_ALLOWANCE',phase_fit=False,
                     precision_bits=192,upper_exact='1/10000000000')
    save(tmp_path/'certificate.json',certificate)
    payload=artificial_payload(example);payload['arm']='proposed'
    for event in payload['events']:
        for key in ('rotation_component_id',):event[key]=event[key].replace('baseline','proposed')
    payload['trajectory_key']=[[0,x['rotation_component_id'],[]] for x in payload['events']]
    (tmp_path/'fake.npz').write_bytes(b'not numerical input')
    payload['final_JU_sha256']=sha(tmp_path/'fake.npz')
    save(tmp_path/'trajectory.json',payload);save(tmp_path/'SOURCE.json',dict(files={}))
    contract=dict(experiment='F1',implementation=e.VERSION,verifier=v.VERSION,car_adapter=c.VERSION,
        status='UNAPPROVED',call_limits=e.LIMITS['F1'],maximum_logical_units=7,
        retry=0,resume=0,precision_bits=192,allowance_exact='1/100000000',global_phase_fit=False,
        source_manifest='SOURCE.json',source_manifest_sha256=sha(tmp_path/'SOURCE.json'),packages={},
        arm='proposed',trajectory='trajectory.json',sampling_SOURCE_sha256='samplingsource',export='fake.npz',
        reuse=dict(files=['certificate.json'],certificates=dict(head='certificate.json')),executable=True,
        saved_inputs={x:sha(tmp_path/x) for x in ['certificate.json','fake.npz','trajectory.json']})
    save(tmp_path/'contract.json',contract)
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('no numerical array decode'))
    monkeypatch.setattr(e.old,'_read_qpy',lambda *a,**k:pytest.fail('no QPY decode'))
    assert e.preflight(tmp_path,tmp_path/'contract.json')['scientific_calls']==0
    (tmp_path/'fake.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='SOURCE_OR_INPUT'):e.preflight(tmp_path,tmp_path/'contract.json')


def test_unadopted_ideal_target_cannot_launch_or_create_output(tmp_path,monkeypatch):
    from trottertracks.c1_stage1_execution import save
    save(tmp_path/'contract.json',dict(executable=False))
    monkeypatch.setattr(e,'preflight',lambda *a:None)
    monkeypatch.setattr(e,'execute_CAR',lambda *a:pytest.fail('no unadopted scientific target'))
    with pytest.raises(ValueError,match='NOT_EXECUTABLE'):e.run(tmp_path,tmp_path/'contract.json')
    assert sorted(p.name for p in tmp_path.iterdir())==['contract.json']
