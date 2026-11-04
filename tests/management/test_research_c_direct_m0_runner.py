"""Invented symmetric matrices, physical-sector indices and gate records only."""
from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from trottertracks import c_direct_m0_oracle as oracle
from trottertracks import c_direct_m0_runner as m
from trottertracks import c_direct_round_signal as direct
from trottertracks import c_fresh_iid_calibration as fresh
from trottertracks import c_pr_aware_df_fixed_certificate as fixed
from trottertracks import c_pr_aware_df_zsxx_enclosure as z
from trottertracks.c_direct_round_artificial import fixture

ROOT=Path(__file__).resolve().parents[2]


def test_exact_inertia_has_one_ground_below_shift():
    A=[['0','1/3'],['1/3','2']]
    r=oracle.inertia_below(A,'1')
    assert r['negative']==r['positive']==1 and r['zero']==0
    assert r['pivots_exact']==['-1','10/9']
    assert not r['floating_point_inertia_used']


def test_known_ground_state_distance_without_normalizing_input():
    A=[[0,0],[0,2]];v=[['999/1000','0'],['1/1000','0']]
    p=oracle.propose_isolation(A);r=oracle.qualify_ground(A,v,p)
    # Exact ground is (1,0); exact centre distance squared is 2/10^6.
    eta=F(r['vector2_distance_upper_exact'])
    assert eta*eta>=F(2,10**6) and eta<F(1,100)
    assert F(r['normalization_distance_upper_exact'])>0
    assert F(r['excited_weight_upper_exact'])>0
    assert not r['input_normalized_or_projected']
    assert not r['existing_own_DF_target_identity_established']


def test_exact_eigenvector_and_complex_global_phase_need_no_state_error():
    A=[[0,0],[0,2]];p=oracle.propose_isolation(A)
    r=oracle.qualify_ground(A,[['0','1'],['0','0']],p)
    assert r['vector2_distance_upper_exact']=='0'
    assert r['normalized_residual_squared_exact']=='0'


def test_nondiagonal_known_rational_ground_vector():
    A=[['32/25','-24/25'],['-24/25','18/25']]
    r=oracle.qualify_ground(A,[['3/5','0'],['4/5','0']],oracle.propose_isolation(A))
    assert r['rayleigh_exact']=='0' and r['vector2_distance_upper_exact']=='0'
    assert r['inertia']['negative']==1


@pytest.mark.parametrize('matrix,match',[
    ([[0,1],[0,2]],'SYMMETRY'),([[0,1j],[-1j,2]],'REAL_H'),
    ([[0,0],[0,float('nan')]],'FINITE'),([[0,0]],'SQUARE')])
def test_invalid_hamiltonians_are_not_symmetrized(matrix,match):
    with pytest.raises(ValueError,match=match):oracle.propose_isolation(matrix)


def test_zero_pivot_and_degeneracy_stop_without_regularization():
    with pytest.raises(ValueError,match='ZERO_LDL'):
        oracle.inertia_below([[0,1],[1,0]],0)
    with pytest.raises(ValueError,match='PROPOSAL'):
        oracle.propose_isolation([[0,0],[0,0]])
    A=[[0,0,0],[0,0,0],[0,0,2]]
    p=dict(H_id=oracle.matrix_id(oracle.symmetric_matrix(A)),shift_exact='1')
    with pytest.raises(ValueError,match='UNIQUE_GROUND'):oracle.qualify_ground(A,[['1','0'],['0','0'],['0','0']],p)


def test_proposal_and_wrong_state_cannot_create_oracle_certificate():
    A=[[0,0],[0,2]];p=oracle.propose_isolation(A)
    assert 'proposal_only' in p['qualification']
    bad=dict(p,H_id='0'*64)
    with pytest.raises(ValueError,match='TARGET_IDENTITY'):oracle.qualify_ground(A,[['1','0'],['0','0']],bad)
    with pytest.raises(ValueError,match='UNIQUE_GROUND'):oracle.qualify_ground(A,[['0','0'],['1','0']],p)
    with pytest.raises(ValueError,match='NONZERO'):oracle.qualify_ground(A,[['0','0'],['0','0']],p)


def toy():
    args=fixture(n=9);part=args['occurrences'][0]
    J=np.zeros((8,4,4));U=np.array([np.eye(4)]*8)
    def bits(a):
        import hashlib
        return dict(shape=list(a.shape),dtype=a.dtype.str,SHA256=hashlib.sha256(a.tobytes()).hexdigest())
    target=dict(args['target'],J=bits(J),U=bits(U))
    ref=deepcopy(part['reference']);ref['binding'].update(target=target,target_id=fresh.digest(target))
    C=part['compiled']['record'];R=part['raw']['record'];I=part['actual_inverse']['record']
    bundle=dict(raw=R,compiled=C,actual_inverse=I,literal=ref,
        per_word=dict(binding=deepcopy(ref['binding']),compiled_id=fresh.digest(C),raw_id=fresh.digest(R)),
        inverse_binding=dict(same_realized_forward_record=fresh.digest(C)))
    ix=[i for i in range(256) if sum((i>>q)&1 for q in (0,2,4,6))==2
        and sum((i>>q)&1 for q in (1,3,5,7))==2]
    indices=np.array([int(f'{i:08b}'[::-1],2) for i in ix])
    v=np.zeros(36,complex);v[0]=1
    data=dict(candidate=dict(J=J,U=U),sector=dict(J=J.copy(),U=U.copy(),H=np.diag(np.arange(36)*2.)),
        basis=dict(sector_basis_indices=indices),ground=dict(energy=np.array(0.),sector_state=v))
    c=dict(literal_target_id=ref['binding']['target_id'],word=ref['binding']['word'],
        identity=args['context']['identity'],source_manifest_sha256='e'*64)
    return bundle,data,c


def evaluate_toy(tmp_path, *, change=None, shell_loader=None):
    from trottertracks.c1_stage3_integrated import Calls
    bundle,data,c=toy()
    if change:change(bundle,data,c)
    call=Calls(tmp_path,m.LIMITS,m.MAXIMUM)
    call('metadata_load','INVENTED',lambda:None);call('arrays_load','INVENTED',lambda:None)
    result=m.evaluate(bundle,data,c,tmp_path,call,
        shell_loader=shell_loader or (lambda *a:direct.ideal_shells(9)))
    return result,call


def test_artificial_complete_fifteen_unit_m0_pipeline(tmp_path):
    r,call=evaluate_toy(tmp_path)
    assert sum(call.counts.values())==m.MAXIMUM==15 and call.first_failure is None
    assert all(value==1 for value in call.counts.values())
    assert r['m']==0 and r['occurrences']==1 and not r['fresh_IID_sample']
    assert not r['calibration_eligible'] and not r['full_operator_error_certified']
    assert r['statistical_confidence'] is None
    for axis in ('X','Y'):
        lo,hi=map(F,r[axis]);assert lo<=0<=hi
    assert (tmp_path/'oracle_certificate.json').exists()
    assert (tmp_path/'compiled_forward_state.json').exists()
    assert json.loads((tmp_path/'embedding.json').read_text())['sector_dimension']==36


def test_original_saved_state_norm_error_is_propagated(tmp_path):
    def change(b,d,c):d['ground']['sector_state'][0]=.999;d['ground']['sector_state'][1]=.001
    r,_=evaluate_toy(tmp_path,change=change)
    state=json.loads((tmp_path/'initial_state.json').read_text())
    assert state['centre_norm_squared_exact']!='1' and F(state['uncertainty_exact'])>0
    assert all(F(r[a][0])<0<F(r[a][1]) for a in ('X','Y'))


@pytest.mark.parametrize('fault',['candidate','word','phase','inverse_parent','basis','symmetry','reference_precision'])
def test_identity_mismatches_stop_before_later_calls(tmp_path,fault):
    def change(b,d,c):
        if fault=='candidate':d['sector']['J'][0,0,0]=1
        if fault=='word':c['word']=[]
        if fault=='phase':b['compiled']['phase_hex']=.5.hex()
        if fault=='inverse_parent':b['inverse_binding']['same_realized_forward_record']='0'*64
        if fault=='basis':d['basis']['sector_basis_indices'][0]=d['basis']['sector_basis_indices'][1]
        if fault=='symmetry':d['sector']['H'][0,1]=.25
        if fault=='reference_precision':b['literal']['precision_bits']=160
    with pytest.raises(ValueError):evaluate_toy(tmp_path,change=change)
    assert not (tmp_path/'compiled_forward_state.json').exists()


def test_saved_word_is_rejected_by_fresh_calibration_adapter(tmp_path):
    r,_=evaluate_toy(tmp_path)
    assert r['scope']!= 'paired_round_raw_complex_signal'
    assert r['fresh_IID_sample'] is False and not r['Stage_C_ready']


def test_exact_factor_reduction_keeps_inverse_state_witness(tmp_path):
    from flint import ctx,acb_mat
    r=fixed.identity(2);r['gates']=[dict(name='sx',qubits=[0],params_hex=[]),dict(name='rz',qubits=[1],params_hex=[.25.hex()])]
    actual=fixed.adjoint(r);actual['phase_hex']=.125.hex()
    with z._BALL_LOCK,ctx.workprec(192):
        state=(acb_mat([[1],[0],[0],[0]]),F())
        reduced=m.inverse_witness(actual,r,state)
        a,ea=direct.native_action(actual,*state);b,eb=direct.native_action(fixed.adjoint(r),*state)
        raw=F(z._norm(a-b)['upper_exact'])+ea+eb
        assert F(reduced['upper_exact'])>0 and raw>0
        assert abs(F(reduced['upper_exact'])-raw)<F(1,10**50)
        assert reduced['exact_unitary_factor_reduction']['prefix_gates']==2


def test_ancilla_shell_decode_and_system_gate_rejection():
    from qiskit import QuantumCircuit
    q=QuantumCircuit(9,1);q.sdg(8);q.h(8);q.measure(8,0)
    r=m.shell_record(q)
    assert r['qubits']==9 and [g['qubits'] for g in r['gates']]==[[8],[8]]
    q=QuantumCircuit(9);q.h(0)
    with pytest.raises(ValueError,match='ANCILLA'):m.shell_record(q)


def test_passive_draft_and_no_permission_entry_never_decode_scientific_inputs():
    code='''
import sys
from pathlib import Path
class Block:
 def find_spec(self,name,*a):
  if name.split('.')[0] in ('numpy','scipy','flint','qiskit','ffsim','pyscf'):
   raise AssertionError('NO_NUMERICAL_IMPORT:'+name)
sys.meta_path.insert(0,Block())
from trottertracks import c_direct_m0_runner as m
r=m.preflight(Path.cwd())
assert not r['execution_ready'] and r['scientific_calls']==0 and r['arrays_decoded']==0
assert r['maximum_logical_units']==15 and r['mandatory_stop']
try:m.run(Path.cwd())
except PermissionError:pass
else:raise AssertionError('NO_SCIENCE_PERMISSION')
try:m.run(Path.cwd(),execute_science=True)
except ValueError as e:assert 'NOT_READY' in str(e)
else:raise AssertionError('DRAFT_MUST_REMAIN_NONEXECUTABLE')
'''
    subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,check=True)


def test_missing_oracle_and_source_are_not_filled_by_preflight():
    r=m.preflight(ROOT)
    assert set(r['missing'])=={'nonexecutable_draft','scientific_SOURCE_to_be_frozen',
        'saved_H_vs_existing_own_DF_oracle_target_identity'}


@pytest.mark.parametrize('error',[RuntimeError('INVENTED callback'),KeyboardInterrupt('INVENTED interruption')])
def test_runner_failure_always_preserves_partial_and_stop(tmp_path,monkeypatch,error):
    c=dict(source_manifest='source.json',fresh_output='new-output')
    (tmp_path/'contract.json').write_text(json.dumps(c));(tmp_path/'source.json').write_text('{}')
    monkeypatch.setattr(m,'preflight',lambda *a:dict(execution_ready=True))
    monkeypatch.setattr(z,'runtime_model',lambda:None)
    def fail(root,c,out,call):
        m.io.save(out/'completed_step.json',dict(kind='INVENTED'))
        raise error
    monkeypatch.setattr(m,'execute',fail)
    with pytest.raises(type(error)):m.run(tmp_path,tmp_path/'contract.json',execute_science=True)
    out=tmp_path/'new-output'
    assert (out/'completed_step.json').exists() and (out/'partial_failure.json').exists()
    stop=json.loads((out/'STOP.json').read_text())
    assert stop['failed'] is not None and stop['retry']==stop['resume']==0 and stop['mandatory_stop']
