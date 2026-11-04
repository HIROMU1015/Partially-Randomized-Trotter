"""Artificial/mocked integration only; never call h4_context/run_science."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as Q
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from qiskit.circuit.library import RXGate, XXPlusYYGate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE),str(ROOT/'src')]
import integration_audit as audit
import connection_pipeline as pipeline
import run_connection as runner


def definition(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def toy():
    old=definition('_connection_toy_old',
        ROOT/'docs/research/rpe_controlled_hadamard_preparation_20261104/test_circuit_semantics.py')
    exporter=definition('_connection_toy_export',
        ROOT/'docs/research/rpe_m17_rigorous_mean_enclosure_preparation_20261103/export_adapter.py')
    h,prep=old.artificial_preparation.__wrapped__()
    saved=exporter.export_preparation(prep,np.array([1.,1j]),[1,2],
                                     n_qubits=2,n_electrons=1,delta=.17)
    saved=json.loads(json.dumps(saved))
    from trotterlib.research_direction_sequence_policy import register_support_restricted_bases
    support,proofs=register_support_restricted_bases(h,prep)
    import math
    schedule=[1]*8+[2,2,4,8,8,16,16,32,32,32]
    theta={r:[math.atan(prep.exact_rte_lambda_r*(.17/r)/(n+1)).hex() for n in (0,2)]
           for r in sorted(set(schedule))}
    rows=[dict(m=m,q=1<<m,r=r,K=2,delta='.17',axes={
        a:dict(N='100',alpha='.01',epsilon='.125') for a in ('cosine','sine')})
        for m,r in enumerate(schedule)]
    axis=dict(U_cat_upper='1/10000',shot_half_width=dict(lower='1/100',upper='1/100'),
              interval=dict(lower='-1/10',upper='1/10'))
    bundle=dict(mode='ARTIFICIAL_INTEGRATION',rows=rows,theta_by_r=theta,
                support_policy='synthetic_same_policy',training_fingerprint='toy',
                saved_axes={m:{a:deepcopy(axis) for a in ('cosine','sine')} for m in range(18)},
                support_proofs=proofs)
    data=dict(preparation=prep,support=support,export=saved,library_evidence=True)
    return bundle,data


def execute(tmp_path,toy,name='run',workers=1,data=None,bundle=None):
    b,d=toy
    return pipeline.execute(tmp_path/name,b if bundle is None else bundle,
        lambda supplied,record: d if data is None else data,workers=workers)


def test_end_to_end_all18_and_parallel_cache_identity(tmp_path,toy):
    first=execute(tmp_path,toy,'serial')
    second=execute(tmp_path,toy,'parallel',2)
    assert first['status']=='COMPLETED_MANDATORY_STOP',first.get('error')
    assert first['connection_decision'].startswith('CONDITIONAL_')
    for key in ('components','boundary','rounds','counts','basis_cache','lowering'):
        assert first[key]==second[key],key
    b,d=toy
    assert first['counts']['half_primitives']==len(d['preparation'].deterministic_blocks)
    assert first['counts']['full_primitives']==1
    assert first['counts']['scalar_boundary_evaluations']==18
    assert first['counts']['axis_comparisons']==36
    assert first['counts']['component_bounds']==len(d['export']['components'])
    assert first['counts']['basis_evaluations_started']==first['basis_cache']['evaluations']
    assert first['basis_cache']['hits']>0
    assert first['lowering']['candidate_slots']<=172
    assert first['RPE_tolerance_certified'] is False
    assert first['scientific_status']=='UNDETERMINED'
    assert first['U_exec_upper'] is first['U_virtual_IID_upper'] is None
    assert all(n==0 for n in first['forbidden_calls'].values())
    assert (tmp_path/'serial/DESCRIPTORS.json').exists()
    assert len((tmp_path/'parallel/basis_enclosures.jsonl').read_text().splitlines())==first['basis_cache']['evaluations']
    assert json.loads((tmp_path/'serial/RESULT.json').read_text())==first


def test_default_never_dispatches_science(monkeypatch,capsys):
    def forbidden(*a,**kw):
        raise AssertionError('science called by default')
    monkeypatch.setattr(runner,'h4_context',forbidden)
    monkeypatch.setattr(runner,'run_science',forbidden)
    # The REAL default validator is executed; it only reads fixed bytes/literals.
    assert runner.main([])==0
    out=json.loads(capsys.readouterr().out)
    assert out['new_H4_scientific_calls']==0 and out['axes']==36
    assert out['U_exec_upper'] is out['U_virtual_IID_upper'] is None


@pytest.mark.parametrize('mutation',['sha','blob','current','commit_payload'])
def test_fixed_hash_and_blob_mismatch_on_artificial_bytes(tmp_path,monkeypatch,mutation):
    raw=b'ARTIFICIAL_SOURCE'
    (tmp_path/'fixture').write_bytes(raw)
    item=dict(commit='artificial',path='fixture',sha256=hashlib.sha256(raw).hexdigest(),
              blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest())
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'git',lambda *args:raw if mutation!='commit_payload' else b'OTHER')
    if mutation in ('sha','blob'): item[mutation if mutation=='blob' else 'sha256']='0'*40
    if mutation=='current': (tmp_path/'fixture').write_bytes(b'OTHER')
    with pytest.raises(audit.Unknown): runner.verify_bound(item,current=True)


@pytest.mark.parametrize('key,value',[('m',17),('q',3),('r',32),('K',4)])
def test_wrong_tuple_join(toy,key,value):
    rows=deepcopy(toy[0]['rows']); expected=deepcopy(rows)
    rows[0][key]=value
    with pytest.raises(audit.Unknown): runner.validate_tuple_join(rows,expected)


@pytest.mark.parametrize('mutation',['class','wire','parameter','hash'])
def test_descriptor_checks_without_operator(toy,monkeypatch,mutation):
    from qiskit.quantum_info import Operator
    d=next(iter(toy[1]['support'].values()))
    gate,wires=d.runtime_operations[0]
    runtime=list(d.runtime_operations)
    if mutation=='class': runtime[0]=(RXGate(.1),wires)
    if mutation=='wire': runtime[0]=(gate,tuple(1-i for i in wires))
    if mutation=='parameter':
        bad=gate.copy(); bad.params=[float(p)+.01 for p in gate.params]
        runtime[0]=(bad,wires)
    meta=replace(d.metadata,basis_hash='wrong') if mutation=='hash' else d.metadata
    bad=replace(d,metadata=meta,runtime_operations=tuple(runtime))
    monkeypatch.setattr(Operator,'__init__',lambda *a,**kw: (_ for _ in ()).throw(AssertionError('Operator called')))
    with pytest.raises(audit.Unknown): audit.descriptor(bad)


def test_nonadjacent_gaussian_gate_rejected():
    from trotterlib.df_rte_tail import DFBasisRegistry
    d=DFBasisRegistry().register([(XXPlusYYGate(.2,.1),(0,2))],num_system_qubits=3)
    with pytest.raises(ValueError,match='topology'): audit.descriptor(d)


@pytest.mark.parametrize('failure',['coverage','selected_hash','primitive','lowering','tau_tuple'])
def test_partial_failure_saved_and_no_later_stage(tmp_path,toy,failure):
    bundle=deepcopy(toy[0]); data=deepcopy(toy[1])
    if failure=='coverage': data['support'].pop(next(iter(data['support'])))
    if failure=='selected_hash': bundle['support_proofs'][0]['selected_basis_hash']='wrong'
    if failure=='primitive':
        data['export']['deterministic_blocks'][0]['primitives']['global_phase']='0x1p-2'
    if failure=='lowering': data['library_evidence']=False
    if failure=='tau_tuple': bundle['rows'][0]['K']=4
    result=execute(tmp_path,toy,data=data,bundle=bundle)
    assert result['status']=='UNKNOWN_MANDATORY_STOP'
    assert result['connection_decision']=='UNDETERMINED'
    assert 'error' in result and result['mandatory_STOP']
    assert json.loads((tmp_path/'run/RESULT.json').read_text())==result
    if failure!='tau_tuple':
        assert result['counts'].get('basis_evaluations_started',0)==0
        assert not result['rounds']


def test_component_failure_keeps_first_certificate(tmp_path,toy,monkeypatch):
    original=audit.transfer.component_gap
    calls=[]
    def fail_second(*args):
        calls.append(1)
        if len(calls)==2: raise audit.Unknown('artificial component error')
        return original(*args)
    monkeypatch.setattr(audit.transfer,'component_gap',fail_second)
    result=execute(tmp_path,toy)
    assert result['stage']=='component_bounds' and result['status']=='UNKNOWN_MANDATORY_STOP'
    assert len(result['components'])==1
    assert result['counts']['component_bounds']==2
    assert (tmp_path/'run/basis_enclosures.jsonl').exists()


def test_basis_failure_and_nonfinite_parameter_are_recorded(tmp_path):
    bad=('bad',3,[dict(name='xx_plus_yy',qubits=[0,2],params=['0x1p-2','0x0p+0'])])
    record=dict(counts={})
    with pytest.raises(audit.Unknown): pipeline.evaluate_bases([bad],1,tmp_path,record)
    assert record['counts']['basis_evaluations_started']==1 and record['basis_errors']
    with pytest.raises(audit.Unknown): audit.finite_hex(float('nan'))


@pytest.mark.parametrize('value',[float('inf'),float('nan'),float.fromhex('0x0.0000000000001p-1022')])
def test_lowering_rejects_finite_and_subnormal_failures(value):
    with pytest.raises(audit.Unknown): audit.controlled_template('rz',value)


def test_lowering_missing_evidence_is_not_zero(toy):
    data=toy[1]; counts={}
    halves,full,_,_=audit.boundary_context(data['preparation'],data['export'],.17,counts)
    with pytest.raises(audit.Unknown):
        audit.audit_lowering(halves,full,toy[0]['theta_by_r'],counts,False)
    assert counts.get('local_control_constructors',0)==0


def test_lowering_wrong_control_wire_template_detected(monkeypatch):
    from qiskit.circuit.library import RZZGate
    from qiskit import QuantumCircuit
    original=RZZGate.control
    def broken(self,*args,**kw):
        gate=original(self,*args,**kw)
        gate.definition=QuantumCircuit(3)
        return gate
    monkeypatch.setattr(RZZGate,'control',broken)
    with pytest.raises(audit.Unknown): audit.controlled_template('rzz',.25)


def test_exact_strict_unknown_and_no_double_pi64(toy):
    axis=toy[0]['saved_axes'][0]['cosine']
    altered={**axis,'U_cat_upper':'1/20','shot_half_width':dict(lower='1/10',upper='1/10')}
    assert audit.classify_axis(altered,'1/5',Q(1,20))['decision']=='UNDETERMINED'
    result=audit.classify_axis(altered,'0.2000000000000000000001',Q(1,20))
    assert result['decision']=='SUFFICIENT_CONDITION_CONFIRMED'
    assert result['total_upper']=='1/5' and not result['pi64_added_again']
    assert audit.classify_axis(altered,'1/5',None)['decision']=='UNKNOWN'
    with pytest.raises(audit.Unknown): audit.classify_axis(altered,'1/5',Q(-1))


def test_existing_pi64_bound_is_not_called_again(tmp_path,toy,monkeypatch):
    _,mean=audit.transfer.modules()
    def forbidden(*args): raise AssertionError('double-counted pi64')
    monkeypatch.setattr(mean,'builder_phase_bound',forbidden)
    result=execute(tmp_path,toy)
    assert result['status']=='COMPLETED_MANDATORY_STOP'
    assert not result['pi64_added_again']


def test_existing_output_never_overwritten(tmp_path,toy):
    out=tmp_path/'run';out.mkdir();(out/'RESULT.json').write_text('old')
    with pytest.raises(FileExistsError): execute(tmp_path,toy)
    assert (out/'RESULT.json').read_text()=='old'


def test_strict_failure_keeps_status_and_does_not_retry(tmp_path,toy):
    bundle=deepcopy(toy[0])
    bundle['rows'][0]['axes']['sine']['epsilon']='0.000001'
    result=execute(tmp_path,toy,bundle=bundle)
    assert result['status']=='COMPLETED_MANDATORY_STOP'
    assert result['connection_decision']=='UNDETERMINED'
    assert result['rounds'][0]['axes']['sine']['decision']=='UNDETERMINED'
    assert result['counts']['round_transfers']==18
    assert result['scientific_status']=='UNDETERMINED'


def test_pipeline_scalar_order_cancellation_detected(tmp_path,toy):
    from types import SimpleNamespace
    bundle=deepcopy(toy[0]);data=deepcopy(toy[1])
    # Mock scalar coefficients only; the production dataclass correctly forbids
    # inconsistent extraction metadata. This fixture is not an H4 preparation.
    prep=SimpleNamespace(**vars(data['preparation']))
    prep.constant_coefficient=1.e16
    prep.extracted_identity_coefficient=-1.e16+2
    delta=float(bundle['rows'][0]['delta'])
    a,b=-delta*prep.constant_coefficient,-delta*prep.extracted_identity_coefficient
    data['preparation']=prep
    data['export']['step_phase_angles']=[a.hex(),b.hex()]
    result=execute(tmp_path,toy,data=data,bundle=bundle)
    assert result['status']=='COMPLETED_MANDATORY_STOP',result.get('error')
    first=result['boundary'][0]
    optimized=-delta*(prep.constant_coefficient+prep.extracted_identity_coefficient)
    assert first['optimized_scalar']==optimized.hex()
    assert optimized != a+b
    assert Q(first['scalar_gap'])==abs(Q.from_float(optimized)-Q.from_float(a)-Q.from_float(b))>0


def test_invalid_saved_width_and_negative_terms_stay_unknown(toy):
    axis=deepcopy(toy[0]['saved_axes'][0]['cosine'])
    axis['shot_half_width']['lower']='1'
    with pytest.raises(audit.Unknown): audit.classify_axis(axis,'.125',Q(0))
    with pytest.raises(ValueError):
        audit.transfer.total_transfer(1,1,boundary=0,component=Q(-1),lowering=0)
    assert audit.transfer.total_transfer(1,1,boundary=0,component=0,lowering=None) is None
