"""Artificial fixed records only: no molecular data, sampler or RNG replay."""
from copy import deepcopy
import gzip
import importlib.util
import io
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
import pytest
from qiskit import QuantumCircuit,qpy
from qiskit.quantum_info import Operator

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_h6_pauli_p2_cost as runner
K=runner.modules()
from p2_study import h4_d1_adapter as D1,h4_d2_pauli_control as D2
from p2_study.pauli_signal_adapter import append_pauli_event,append_pauli_evolution
from p2_study import rte,pauli_signal_adapter,rte_compiled_cost


def toy():
    law=dict(n_system=2,config=dict(q=2,r=1),step_time=.1,scalar_step_binary64=.037,
        retained=[dict(label='IZ',coefficient=.7),dict(label='ZI',coefficient=-.3)],
        components=[dict(label='XY',sign=-1),dict(label='ZI',sign=1),dict(label='YX',sign=-1)])
    events=[dict(order=2,products=['0','2'],rotation='1',signs=[-1,-1,1],angle=.17,phase=[-1.,0.]),
            dict(order=0,products=[],rotation='0',signs=[-1],angle=.11,phase=[1.,0.])]
    result=dict(events=events,event_count=2,body_phase=math.pi+.074)
    return law,result,K.recipe(law,result)


def full_record(record):
    ids=[*record['products'],record['rotation']];order=record['order']
    return dict(taylor_order=order,product_component_ids=record['products'],
        rotation_component_id=record['rotation'],rotation_angle=record['angle'],
        phase=dict(real=record['phase'][0],imag=record['phase'][1]),
        application_sequence=[dict(role='rotation' if i==order else 'product',
            component_id=key,coefficient_sign=record['signs'][i]) for i,key in enumerate(ids)])


def artificial_full(spec):
    body=QuantumCircuit(spec['n'])
    for step in range(spec['q']):
        for x in spec['retained']:append_pauli_evolution(body,x['label'],x['coefficient'],spec['delta']/2)
        for event in spec['events'][step*spec['r']:(step+1)*spec['r']]:append_pauli_event(body,event,spec['labels'])
        for x in reversed(spec['retained']):append_pauli_evolution(body,x['label'],x['coefficient'],spec['delta']/2)
        body.global_phase+=spec['scalar_step']
    return body


def test_both_schemas_exact_saved_applications():
    law,result,spec=toy()
    for compact,expected in zip(result['events'],spec['events']):
        full=D1.pauli_event_view(full_record(compact),law['components'])
        assert vars(full)==vars(expected)
        assert full.rotation_angle==compact['angle']
    assert [x.component_id for x in spec['events'][0].application_sequence]==['0','2','1']


@pytest.mark.parametrize('field,value',[('signs',[1,-1,1]),('phase',[1.,0.]),('angle',float('nan'))])
def test_wrong_saved_semantics_rejected(field,value):
    law,result,_=toy();record=deepcopy(result['events'][0]);record[field]=value
    with pytest.raises(ValueError):D1.pauli_event_view(record,law['components'])


def test_full_recorder_matches_original_source_and_both_qpy_wrappers():
    _,_,spec=toy();body=artificial_full(spec)
    lift=D2.central_control_body(body)
    u=Operator(body).data;expected=np.eye(8,dtype=complex);expected[4:,4:]=u
    np.testing.assert_allclose(Operator(lift.body).data,expected,atol=2e-12,rtol=0)
    for axis in ('cosine','sine'):
        wrapped=K.hadamard_wrapper(lift.body,axis,already_controlled=True)
        stream=io.BytesIO();qpy.dump(wrapped,stream);stream.seek(0)
        restored=qpy.load(stream)[0]
        check=K.compare_programs(K.semantic_program(restored),K.expected_full(spec,axis))
        assert check['Full_circuits_rebuilt']==0
        assert check['maximum_parameter_difference']<1e-12
        assert restored.data[-1].operation.name=='measure'
        assert (restored.data[-3].operation.name=='sdg')==(axis=='sine')


def test_phase_ownership_order_and_scalar_only_R_is_not_empty():
    _,_,spec=toy()
    phases={part:K.source_tokens(spec,part)[1] for part in ('F','R','E')}
    assert abs(math.remainder(phases['F']-phases['R']-phases['E'],2*math.pi))<1e-12
    assert abs(phases['E']-math.pi)<1e-12
    assert abs(phases['R']-.074)<1e-12
    spec['retained']=[]
    retained,sidecar=K.part_wrappers(spec,'R');empty,_=K.part_wrappers(spec,'0')
    assert sidecar['phase_owner']=='scalar'
    assert K.program_hash(K.semantic_program(retained['cosine']))!=K.program_hash(K.semantic_program(empty['cosine']))
    r=retained['cosine'].remove_final_measurements(inplace=False)
    z=empty['cosine'].remove_final_measurements(inplace=False)
    assert np.max(np.abs(Operator(r).data-Operator(z).data))>1e-3
    assert empty['cosine'].count_ops()=={'h':2,'measure':1}
    assert empty['sine'].count_ops()=={'h':2,'sdg':1,'measure':1}


@pytest.mark.parametrize('part',['R','E'])
def test_parts_reuse_source_under_no_sampler_or_generic_control(part):
    _,_,spec=toy()
    with patch.object(rte,'sample_rte_events',side_effect=AssertionError('SAMPLE')), \
         patch.object(rte,'_make_event',side_effect=AssertionError('ANGLE')), \
         patch.object(pauli_signal_adapter,'sample_rte_events',side_effect=AssertionError('SAMPLE')), \
         patch.object(QuantumCircuit,'to_gate',side_effect=AssertionError('GENERIC')), \
         patch.object(rte_compiled_cost,'transpile',side_effect=AssertionError('COMPILE')):
        body,sidecar=K.build_part(spec,part)
        tokens,phase=K.source_tokens(spec,part)
        assert sidecar['aggregate_phase']==phase
        recorded=D2.central_control_tokens(tokens,phase,2,circuit_factory=K.DescriptorRecorder).body
        assert K.semantic_program(body)['instructions']==recorded.data
    with pytest.raises(ValueError,match='FULL_BUILD_FORBIDDEN'):K.build_part(spec,'F')


@pytest.mark.parametrize('mutation',['angle','qubit','phase'])
def test_full_semantic_mismatch_rejected(mutation):
    _,_,spec=toy();expected=K.expected_full(spec,'sine');actual=deepcopy(expected)
    if mutation=='angle':
        next(x for x in actual['instructions'] if x['name']=='crz')['params'][0]+=.01
    elif mutation=='phase':
        next(x for x in actual['instructions'] if x['name']=='p')['params'][0]+=math.pi
    else:actual['instructions'][0]['qubits']=[0]
    with pytest.raises(ValueError):K.compare_programs(actual,expected)


def test_same_program_identity_ignores_qpy_names_and_distinguishes_axis():
    _,_,spec=toy();a,_=K.part_wrappers(spec,'R');b,_=K.part_wrappers(spec,'R')
    a['cosine'].name='one';b['cosine'].name='another'
    assert K.program_hash(K.semantic_program(a['cosine']))==K.program_hash(K.semantic_program(b['cosine']))
    assert K.program_hash(K.semantic_program(a['cosine']))!=K.program_hash(K.semantic_program(a['sine']))


def test_d1_interaction_integer_identities():
    f,r,e,z=[dict(RZ=x,CX=2*x) for x in (19,8,15,2)]
    row=D1.interaction_metrics(f,r,e,z)
    assert row['RZ']['Delta_interaction']==-2
    assert row['CX']['contextual_retained_increment']==8
    assert row['RZ']['contextual_event_increment']==11
    legacy=dict(pauli_jw_PR_full=f,pauli_jw_PR_retained=r,pauli_jw_PR_event=e,
                df_native_PR_full=f,df_native_PR_retained=r,df_native_PR_event=e,empty=z)
    assert D1.interactions(legacy)==dict(pauli_jw=row,df_native=row)


def test_worker_synthetic_compile_and_qpy_metrics():
    _,_,spec=toy();wrapper=K.part_wrappers(spec,'E')[0]['sine']
    with TemporaryDirectory() as folder,patch.object(runner,'OUT',Path(folder)):
        saved=runner.save_qpy(wrapper,'input.qpy.gz')
        name,record=runner.compile_one('toy',saved)
        restored=runner.decode((Path(folder)/record['compiled_qpy']['path']).read_bytes())
        assert record['RZ']==restored.count_ops()['rz']
        assert record['CX']==restored.count_ops()['cx']
        assert set(record['gate_histogram'])<=set(['rz','sx','x','cx','measure'])
        assert record['compiler']['qiskit_version']=='1.3.0'
        assert record['compiler']['transpiler_seed']==17
        assert restored.num_qubits==3
