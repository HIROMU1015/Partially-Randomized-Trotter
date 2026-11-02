"""Numerical/discrete separation, small JW gates, binding and history fixtures."""
import json
from pathlib import Path
import pytest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZGate,XXPlusYYGate
from qiskit.quantum_info import Operator
from trottertracks.resource_applicability import h6_roundoff_preparation_v1 as r
from trottertracks.resource_applicability import h6_parallel_contract_v1 as c
from trottertracks.resource_applicability.ax2a_preparation import digest


@pytest.mark.parametrize('value',[.1,.1+1e-15,.1-1e-15])
def test_small_numeric_roundoff_allowed(value):
    old={'coefficient':.1,'seed':42,'sign':-1,'ids':['a','b']}
    new=dict(old,coefficient=value)
    r.compare(old,new)


@pytest.mark.parametrize('change',['large_float','sign','seed','ids','type','nan','inf'])
def test_scientific_discrete_or_large_change_rejected(change):
    old={'coefficient':.1,'seed':42,'sign':-1,'ids':['a','b']};new=dict(old)
    if change=='large_float':new['coefficient']+=.001
    if change=='sign':new['sign']=1
    if change=='seed':new['seed']=43
    if change=='ids':new['ids']=['b','a']
    if change=='type':new['seed']=42.
    if change=='nan':new['coefficient']=float('nan')
    if change=='inf':new['coefficient']=float('inf')
    with pytest.raises(ValueError):r.compare(old,new)


def test_single_particle_projection_matches_full_small_operator():
    ops=[(RZGate(.2),(0,)),(XXPlusYYGate(.7,.3),(1,0)),(RZGate(-.1),(2,)),(XXPlusYYGate(.8,-.2),(2,1))]
    circuit=QuantumCircuit(3)
    for gate,positions in ops:circuit.append(gate,positions)
    full=Operator(circuit).data
    expected=full[np.ix_([1,2,4],[1,2,4])]/full[0,0]
    assert np.max(np.abs(r.single_particle(ops,3)-expected))<1e-14


def test_nonadjacent_jw_gate_is_rejected():
    with pytest.raises(ValueError):r.single_particle([(XXPlusYYGate(.1,.2),(0,2))],3)


@pytest.mark.parametrize('case',['valid','archive_changed','choice','policy','source','input','operator','nan','original_digest'])
def test_roundoff_receipt_is_bound_and_fail_closed(tmp_path,case):
    archive=tmp_path/'archive.json';archive.write_text('ARTIFICIAL_BOUND_ARCHIVE')
    receipt=tmp_path/'receipt.json'
    source_paths=['src/trottertracks/resource_applicability/h6_roundoff_preparation_v1.py',
        'src/trottertracks/resource_applicability/h6_roundoff_cost_port_v1.py',
        'src/trottertracks/resource_applicability/h6_frozen_preparation_v1.py']
    m={'source_hashes':{p:'a'*64 for p in source_paths},'base_manifest':{
        'input_identity':{'hamiltonian_hash':'b'*64},'reuse_registry':{'missing':[{'index':450,'partial_event_digest':'c'*64}]}}}
    proof={'status':'PASS_USER_APPROVED_ROUNDOFF','policy':r.policy(),
        'deterministic_preparation_unchanged':True,'original_hamiltonian_hash':'b'*64,
        'diagnostic_calls':{'trajectory':1,'occurrence':8},'compile':0,'signal':0,'input_generation':0,
        'partial_450':{'exact_event_choices':True,'only_numeric_differences':True,'original_event_digest':'c'*64},
        'diagnostic_source_hashes':m['source_hashes'].copy(),
        'fragment_checks':[{'index':i,'native_input_weighted_bound':0.,'aligned_weighted_difference_bound':0.,'unitarity':0.} for i in range(19)]}
    if case=='choice':proof['partial_450']['exact_event_choices']=False
    if case=='policy':proof['policy']['absolute_tolerance']=1.
    if case=='source':proof['diagnostic_source_hashes'][source_paths[0]]='e'*64
    if case=='input':proof['original_hamiltonian_hash']='e'*64
    if case=='operator':proof['fragment_checks'][0]['aligned_weighted_difference_bound']=.01
    if case=='nan':proof['fragment_checks'][0]['unitarity']=float('nan')
    if case=='original_digest':proof['partial_450']['original_event_digest']='e'*64
    receipt.write_text(json.dumps(proof))
    m['roundoff_preparation_binding']={'archive_path':str(archive),'archive_bytes':archive.stat().st_size,
        'archive_sha256':c.file_hash(archive),'receipt_path':str(receipt),'receipt_sha256':c.file_hash(receipt)}
    if case=='archive_changed':archive.write_text('ALTERED')
    if case=='valid':assert c.roundoff_binding(tmp_path,m)
    else:
        with pytest.raises(ValueError):c.roundoff_binding(tmp_path,m)


def test_prior_failed_totals_stay_unknown_with_separate_reservation(tmp_path):
    path=tmp_path/c.PRIOR_STOP;path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'actual_call_totals':None,'budget_not_reset':True,
        'attempted_trajectory_lower_bound':4,'attempted_occurrence_lower_bound':32}))
    history=c.prior_stop_accounting(tmp_path)
    assert history['actual_call_totals'] is None and history['exact_total_unknown']
    assert history['conservative_reservation']=={'trajectory':4,'occurrence':32}
    assert history['reservation_is_not_an_exact_total']
    assert not history['failed_calls_refunded'] and not history['budget_reset']


def test_new_cost_port_keeps_frozen_logic_except_numeric_partial_gate():
    import inspect
    from trottertracks.resource_applicability import h6_matched_cost_port_v3 as old,h6_roundoff_cost_port_v1 as new
    before=inspect.getsource(old.cost_task)
    replacement="""    if expected is not None and expected != event_hash:
        from .h6_roundoff_preparation_v1 import compare_partial
        comparison=compare_partial(root,events,distribution)
        if comparison['original_event_digest']!=expected:
            raise ValueError('PARTIAL_EVENT_ORIGIN_CHANGED')
        writer.write('roundoff_partial_event_comparison.json',comparison)"""
    expected=before.replace("    if expected is not None and expected != event_hash:\n        raise ValueError('PARTIAL_EVENT_REGENERATION_MISMATCH')",replacement)
    assert inspect.getsource(new.cost_task)==expected
