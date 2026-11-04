"""B1 invented circuits/bounds and passive metadata only; no new H4 work."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from trottertracks import c_p0_uniform_method as m

ROOT=Path(__file__).resolve().parents[2]


def argument(**kw):
    return dict(dict(family='all216_atom_words_length32',final_output='whole_kernel_P0',
        dimension=512,phase_sensitive=True,actual_inverse=True,shells=True,
        compiler_success_all_words=True,kind='supplied_structural_uniform_argument',
        provenance='Invented exact finite grammar argument;not an H4 proof'),**kw)


@pytest.mark.parametrize('oneq,cx',[(0,0),(1,0),(4,2),(1200,9)])
def test_strict_shortening_budget_covers_adversarial_rewrites(oneq,cx):
    # Worst case accepts length2->1 one at a time, with CX deletion exposing
    # another run; each accepted replacement decreases total1Q by >=1.
    q=oneq;accepted=0
    while q>=2:q-=1;accepted+=1
    b=m.rewrite_budget(oneq,cx)
    assert accepted<=b['accepted_1q_rewrites_max']
    assert b['candidate_matrix_products_max']>=b['successful_output_iterations_max']*oneq
    assert not b['proves_all_words_compile_successfully']
    assert b['uniform_error_bound'] is None


@pytest.mark.parametrize('value',[-1,1.5,True])
def test_invalid_gate_budget(value):
    with pytest.raises(ValueError):m.rewrite_budget(value,3)


def test_symbolic_translation_definitions_and_whole_word_syntax():
    inv=m.primitive_translation_inventory();g=m.grammar_budget(inv)
    # Includes both literal zero-time head scalar P nodes. Counts are upper
    # counts before shortening, not counts measured on an H4 word.
    assert g['primitive_multiplicities']['p']==10
    assert g['rewrite_budget']['initial_one_qubit']==100914
    assert g['rewrite_budget']['initial_cx']==6608
    assert g['H4_circuits_built']==0 and g['uniform_error_bound'] is None
    assert not inv['phase_error_enclosed']


def test_cross_event_runs_are_whole_P0_not_sum_of_event_compiles():
    from qiskit import QuantumCircuit
    from qiskit.converters import circuit_to_dag
    from qiskit.quantum_info import Operator
    a=QuantumCircuit(2);a.rz(.21,0);a.rz(.13,0)
    b=QuantumCircuit(2);b.rz(-.13,0);b.rz(-.21,0)
    raw=a.compose(b);raw.p(.19,1);raw.global_phase=.27
    runs=list(circuit_to_dag(raw).collect_1q_runs())
    assert max(map(len,runs))>=4
    compiled,report=m.diagnose_artificial_circuit(raw,Operator(raw).data)
    assert compiled.size()<a.size()+b.size()+1
    assert report['segment']['compiled_reference_frobenius']<1e-12
    assert report['coverage']=='finite_artificial_family_only'
    assert not report['all_law_qualified']


@pytest.mark.parametrize('missing',['node_matrix_error','model','zsxx','translation','phase'])
def test_unknown_component_is_never_zero_or_qualified(missing):
    from trottertracks import c_pr_aware_df_zsxx_uniform as old
    def proof(scope,**kw):return dict(kind='uniform_proof_input',scope=scope,norm='operator_2',
        phase_sensitive=True,provenance='invented proof inputs',upper_exact='1/1000000000000',**kw)
    inputs=dict(node_matrix_error='0',model=old.ieee_model(),exact_unitary_targets=True,
        zsxx=proof('M_to_V_all_reachable_D_N',compiler_interval_image_enclosed=True,covers_D_N=5,all_branches=True),
        translation=proof('all_initial_translation_paths'),phase=proof('all_phase_accumulation_and_normalizations'))
    inputs[missing]=None
    r=m.conditional_compiler_bound(dict(rewrite_budget=m.rewrite_budget(3,2)),**inputs)
    assert r['whole_P0_bound_exact'] is None and r['actual_H4_bound'] is None


def test_known_unitary_perturbations_inverse_and_coherent_signal():
    theta=1e-7
    k=dict(operator_bound_pilot=4*abs(np.exp(1j*theta)-1)+1e-10,norm_upper_pilot=1.)
    U=np.diag([1.,np.exp(4j*theta)]).astype(complex)
    assert np.linalg.norm(U-np.eye(2),2)<=k['operator_bound_pilot']
    shells={x:dict(operator_bound_pilot=0.,norm_upper_pilot=1.) for x in ('head','X','Y')}
    r=m.conditional_rows(k,shells,argument(),expected_identity='toy',observed_identity='toy')
    assert len(r['rows'])==13
    for row in r['rows']:
        assert abs(np.exp(4j*(1<<row['m'])*theta)-1)<=row['bound']
    assert r['actual_H4_qualification'] is None and not r['proof_inputs_verified_by_schema']
    assert r['actual_CA_achieved'] is None and not r['PCG64_joint_stream_qualified']


@pytest.mark.parametrize('change',[dict(family='finite_diagnostic_words'),dict(dimension=72),
    dict(final_output='eventwise_P0'),dict(phase_sensitive=False),dict(actual_inverse=False),
    dict(shells=False),dict(compiler_success_all_words=False)])
def test_incomplete_scope_is_refused(change):
    k=dict(operator_bound_pilot=0.,norm_upper_pilot=1.)
    with pytest.raises(ValueError):m.conditional_rows(k,{x:k for x in ('head','X','Y')},argument(**change),
        expected_identity='toy',observed_identity='toy')


def test_loose_bound_is_not_RPE_GO_and_missing_shell_stays_unknown():
    k=dict(operator_bound_pilot=.1,norm_upper_pilot=1.)
    shells={x:dict(operator_bound_pilot=0.,norm_upper_pilot=1.) for x in ('head','X','Y')}
    r=m.conditional_rows(k,shells,argument(),expected_identity='toy',observed_identity='toy')
    assert r['status']=='INFEASIBLE_SYNTHESIS_ONLY_BOUND' and r['actual_CA_achieved'] is None
    shells['Y']['operator_bound_pilot']=None
    assert m.conditional_rows(k,shells,argument(),expected_identity='toy',observed_identity='toy')['rows'] is None
    with pytest.raises(ValueError):m.conditional_rows(k,shells,argument(),expected_identity='old',observed_identity='new')


def test_full512_columns_detect_unseen_phase_and_leakage():
    from trottertracks.c1_subspace_synthesis_bound import ColumnStream
    # A perturbation only on a non-sector column is invisible to a sector
    # projection. Keep the complete 512-column/full-output difference instead.
    I=np.eye(512,dtype=complex);C=I.copy();C[0,0]=np.exp(1e-7j)
    stream=ColumnStream(512,512);stream.add(list(range(512)),C.T.copy(),I.T.copy())
    r=stream.finish()
    assert r['restricted_operator_bound_pilot']>9.99e-8
    assert r['covered_columns']==512 and r['rigorous_operator_bound'] is None
    leaking=I.copy();a=1e-6
    leaking[np.ix_([0,1],[0,1])]=[[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]]
    stream=ColumnStream(1,512);stream.add([0],leaking[:,0].reshape(1,512),I[:,0].reshape(1,512))
    assert stream.finish()['represented_difference_frobenius']>9.99e-7


def metadata_fixture(tmp_path):
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    compiler=dict(optimization_level=1,seed_transpiler=17)
    (tmp_path/'code').write_text('fixed source');(tmp_path/'input').write_bytes(b'no H4 arrays')
    (tmp_path/'STOP').write_text(json.dumps(dict(failed=False,scientific_units=38,retry=0,resume=0,mandatory_stop=True)))
    (tmp_path/'Acontract').write_text(json.dumps(dict(candidate_sha256=m.CANDIDATE,compiler=compiler)))
    source=dict(files={'code':sha(tmp_path/'code')});(tmp_path/'SOURCE').write_text(json.dumps(source))
    c=m.fixed();c.update(source_manifest='SOURCE',source_manifest_sha256=sha(tmp_path/'SOURCE'),
        saved_inputs={p:sha(tmp_path/p) for p in ['input','STOP','Acontract']},packages={},installed_compiler_sources={},
        compiler=compiler,stage_A=dict(STOP='STOP',contract='Acontract'))
    p=tmp_path/'contract';p.write_text(json.dumps(c));return c,p


@pytest.mark.parametrize('fault',['none','source','input','compiler','fresh','retry','resume'])
def test_passive_drift_and_fresh_gates(tmp_path,fault,monkeypatch):
    c,p=metadata_fixture(tmp_path)
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('NO_ARRAY_DECODE'))
    if fault in ('source','input'):(tmp_path/('code' if fault=='source' else 'input')).write_text('changed')
    if fault=='compiler':c['compiler']['seed_transpiler']=18
    if fault=='fresh':(tmp_path/c['fresh_output']).mkdir(parents=True)
    if fault in ('retry','resume'):c[fault]=1
    p.write_text(json.dumps(c))
    if fault=='none':assert m.preflight(tmp_path,p)['scientific_calls']==0
    else:
        with pytest.raises(ValueError):m.preflight(tmp_path,p)


def test_draft_never_runs_H4_even_with_execute_flag():
    with pytest.raises(PermissionError):m.run_science()
    with pytest.raises(NotImplementedError,match='P0_UNIFORM_NOT_ESTABLISHED'):m.run_science(execute_science=True)


def test_consumed_stage_A_is_refused_and_STOP_preserved():
    from trottertracks import c_iid_grid_bridge as a
    stop=ROOT/a.FRESH/'STOP.json';before=stop.read_bytes()
    with pytest.raises(ValueError,match='FRESH_NO_RETRY_RESUME'):a.preflight(ROOT,ROOT/a.CONTRACT)
    assert stop.read_bytes()==before and json.loads(before)['scientific_units']==38


def test_real_metadata_passive_without_numerical_imports():
    code='''
import sys
from pathlib import Path
class Block:
 def find_spec(self,fullname,*args):
  if fullname.split('.')[0] in ('numpy','scipy','qiskit','pyscf','openfermion','ffsim','jax'):
   raise AssertionError('NO_NUMERICAL_IMPORT:'+fullname)
sys.meta_path.insert(0,Block())
from trottertracks import c_p0_uniform_method as m
r=m.preflight(Path.cwd(),Path(m.CONTRACT))
assert r['scientific_calls']==0 and not r['H4_execution_ready']
'''
    subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,check=True)
