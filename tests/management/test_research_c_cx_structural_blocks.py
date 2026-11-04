"""Invented records only. No saved molecular operator evaluation."""
from copy import deepcopy
from fractions import Fraction as F
import pytest
from trottertracks import c_cx_structural_blocks as b
from trottertracks import c_compiled_raw_local as local
from trottertracks import c_arb_column_batch as columns
from trottertracks import c_pr_aware_df_fixed_certificate as fixed


def g(name,qs,*values):return dict(name=name,qubits=qs,params_hex=[float(x).hex() for x in values])
def r(n,gates=(),phase=0.):return dict(fixed.identity(n),gates=list(gates),phase_hex=float(phase).hex())


def checked(a,c,**policy):
    result=b.compare(a,c,parallel=False,**policy)
    old=columns.compare(a,c,parallel=False)
    assert F(result['upper_exact'])>=F(old['lower_exact'])
    assert b.check_binding(result,a,c)
    return float(F(result['upper_exact']))


@pytest.mark.parametrize('name',['sx','s'])
@pytest.mark.parametrize('i',range(4))
@pytest.mark.parametrize('j',range(4))
def test_exact_C4_phase_preserving_table(name,i,j):
    names=('id','sx','x','sxdg') if name=='sx' else ('id','s','z','sdg')
    a=r(1,[g(names[i],[0]),g(names[j],[0])]);rewritten,proof=b.exact_rewrite(a)
    assert proof['global_phase_unchanged']
    assert checked(a,rewritten)<1e-50


@pytest.mark.parametrize('n',[2,3,9])
def test_same_skeleton_and_full_register_embedding(n):
    a=r(n,[g('rz',[0],.125),g('cx',[0,1])]);c=r(n,[g('rz',[0],.125+2**-40),g('cx',[0,1])])
    out=b.compare(a,c,parallel=False)
    assert out['dimension']==1<<n and out['columns_evaluated']==0
    assert 0<float(F(out['upper_exact']))<1e-11 and b.check_binding(out,a,c)


def test_unmatched_CX_pairs_commuting_rotation_and_compensated_roots():
    a=r(2,[g('cx',[0,1]),g('rz',[0],.25),g('cx',[0,1])]);c=r(2,[g('rz',[0],.25)])
    assert local.compare(a,c,parallel=False)['upper_exact'] is None
    assert checked(a,c)<1e-50
    out=b.compare(a,c,parallel=False)
    assert any(x.get('root_index',0)!=0 for x in out['block_witnesses'].values())
    assert F(out['global_delta_pi_exact'])%2==0


def test_noncommuting_rotation_is_detected():
    a=r(2,[g('cx',[0,1]),g('rz',[1],.25),g('cx',[0,1])]);c=r(2,[g('rz',[1],.25)])
    assert .2<checked(a,c)<1


def test_Clifford_conjugation_reversed_control_direction():
    hh=[g('h',[0]),g('h',[1])]
    assert checked(r(2,hh+[g('cx',[0,1])]+hh),r(2,[g('cx',[1,0])]))<1e-50


def test_multiple_CX_different_order_exact_linear_map():
    a=r(3,[g('cx',[0,1]),g('cx',[1,2])])
    c=r(3,[g('cx',[1,2]),g('cx',[0,2]),g('cx',[0,1])])
    assert checked(a,c)<1e-50


def test_disjoint_order_and_inverse_cancellation():
    a=r(4,[g('cx',[0,1]),g('rz',[3],.125),g('cx',[0,1]),g('cx',[2,3])])
    c=r(4,[g('rz',[3],.125),g('cx',[2,3])])
    out,proof=b.exact_rewrite(a)
    assert proof['counts']['CX_inverse_pairs']==1 and checked(a,c)<1e-50


@pytest.mark.parametrize('n',[3,4])
def test_internal_component_full_local_dimension(n):
    a=r(n,[g('cx',[q,q+1]) for q in range(n-1)]+[g('rz',[n-1],.125)])
    c=r(n)
    result=b.compare(a,c,parallel=False)
    assert result['maximum_support']==n
    assert any(x.get('support_dimension')==1<<n for x in result['block_witnesses'].values())
    assert checked(a,c)>0


def test_support_growth_no_dense_fallback_or_false_full512():
    a=r(9,[g('cx',[q,q+1]) for q in range(4)]);c=r(9)
    result=b.compare(a,c,parallel=False)
    assert result['status']=='NOT_APPLICABLE_BLOCK_SUPPORT' and result['upper_exact'] is None
    assert not result['full_operator_coverage'] and not result['dense_fallback_used']


@pytest.mark.parametrize('phase',[.125,-.375])
def test_global_phase_and_controlled_relative_phase(phase):
    assert checked(r(2),r(2,phase=phase))>abs(phase)*.9
    assert checked(r(2),r(2,[g('p',[1],phase)]))>abs(phase)*.9


def test_internal_error_cancellation_and_coarse_telescope_distinguished():
    # Three-CX bridge changes skeleton. Keeping one adjacent shared anchor pools
    # two opposite small perturbations; splitting it loses their cancellation.
    eps=2**-25
    a=r(3,[g('rz',[0],eps),g('cx',[0,1]),g('cx',[1,2]),g('rz',[0],-eps)])
    c=r(3,[g('cx',[1,2]),g('cx',[0,2]),g('cx',[0,1])])
    tight=b.compare(a,c,parallel=False,guard=1)
    coarse=b.compare(a,c,parallel=False,guard=0)
    assert F(tight['upper_exact'])>=0 and F(coarse['upper_exact'])>=0
    assert checked(a,c)<1e-5
    # Exact cancellation inside one local matrix, with no rewrite across CX.
    pair=dict(kind='block',support=[0,1],left=[g('rz',[0],eps),g('cx',[0,1]),g('rz',[0],-eps)],right=[g('cx',[0,1])])
    assert float(F(b._internal_job(pair)['upper_exact']))<1e-50
    # Separate local bounds necessarily discard this cancellation.
    single=local._block_job(dict(left=[g('rz',[0],eps)],right=[]))
    assert 2*F(single['upper_exact'])>F('1e-9')


def test_accumulation_and_near_Clifford_not_exact_rewrite():
    a=r(2,[g('rz',[0],.125)]*60+[g('cx',[0,1])]);c=r(2,[g('rz',[0],7.5+2**-40),g('cx',[0,1])])
    assert 0<checked(a,c)<1e-11
    near=r(1,[g('rz',[0],3.141592653589793)])
    assert b.exact_rewrite(near)[0]==near


@pytest.mark.parametrize('fault',['literal','nonunitary','phase','angle','gate','register'])
def test_invalid_input_explicit_FAIL(fault):
    a=r(2,[g('rz',[0],.125)])
    if fault=='literal':a['target']='literal_saved_atom'
    if fault=='nonunitary':a['unitary']=False
    if fault=='phase':a['phase_hex']='inf'
    if fault=='angle':a['gates'][0]['params_hex']=['nan']
    if fault=='gate':a['gates'][0]['name']='unknown'
    if fault=='register':a['qubits']=True
    with pytest.raises((ValueError,KeyError)):b.compare(a,r(2),parallel=False)


@pytest.mark.parametrize('fault',['record','expansion','plan','false_dimension','false_columns','calibration','all_law'])
def test_binding_partition_and_false_qualification(fault):
    a,c=r(3,[g('cx',[0,1])]),r(3,[g('cx',[1,0])])
    saved,p=b.make_plan(a,c)
    if fault=='record':
        a['phase_hex']=.125.hex()
        with pytest.raises(ValueError):b.verify_saved_plan(saved,a,c)
    elif fault=='expansion':
        saved['expansions'][0]['record']['gates'][0]['qubits']=[0,2]
        with pytest.raises(ValueError):b.verify_saved_plan(saved,a,c)
    elif fault=='plan':
        p['factors']=[];p['plan_id']=local._digest({k:v for k,v in p.items() if k!='plan_id'})
        with pytest.raises(ValueError):b.evaluate(saved,p,parallel=False)
    else:
        result=b.evaluate(saved,p,parallel=False)
        key={'false_dimension':'dimension','false_columns':'columns_evaluated','calibration':'calibration_ready','all_law':'all_word_uniform'}[fault]
        result[key]=512 if fault in ('false_dimension','false_columns') else True
        with pytest.raises(ValueError):b.check_binding(result,a,c)


def test_saved_definition_replay_and_original_primitive_bridge():
    a=r(3,[g('crzz',[2,0,1],.125)]);c=r(3,[g('crz',[2,0],.125)])
    saved,p=b.make_plan(a,c)
    assert b.verify_saved_plan(saved,a,c)
    result=b.evaluate(saved,p,parallel=False)
    assert result['unique_definitions']==2 and F(result['translation_upper_exact'])>0
    assert checked(a,c)>0


def test_artificial_actual_P0_MPFR_path():
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import XXPlusYYGate
    from trottertracks.c1_h4_native_fair import compile_checked,CompilerProfile
    q=QuantumCircuit(3);q.append(XXPlusYYGate(.375,.25),[0,1]);q.crz(.125,1,0);q.cx(1,2);q.rz(.25,2)
    a,c=fixed.record(compile_checked(q,CompilerProfile())),fixed.record(q)
    assert checked(a,c)<1e-12


def test_nonfinite_numerical_enclosure_rejected(monkeypatch):
    from flint import acb_mat
    monkeypatch.setattr(fixed,'_local',lambda *args:acb_mat([['nan',0],[0,1]]))
    with pytest.raises(ValueError):b.compare(r(2,[g('ry',[0],.25)]),r(2),parallel=False)


def test_parallel_matches_serial_invented_unique_tasks():
    a,c=r(3),r(3)
    for j in range(70):
        a['gates'] += [g('ry',[0],.1+j*.001),g('cx',[0,1])]
        c['gates'] += [g('ry',[0],.1+j*.001+2**-45),g('cx',[0,1])]
    x=b.compare(a,c,parallel=False);y=b.compare(a,c,parallel=True)
    assert x['upper_exact']==y['upper_exact'] and x['block_witnesses']==y['block_witnesses']


@pytest.mark.parametrize('guard',[0,1])
def test_complete_partition_reconstructs_both_native_operators(guard):
    a=r(4,[g('h',[0]),g('cx',[0,1]),g('rz',[2],.25),g('cx',[1,2]),g('h',[1]),g('cx',[2,3]),g('p',[3],.5)])
    c=r(4,[g('sx',[3]),g('cx',[1,0]),g('cx',[2,3]),g('ry',[1],.125),g('cx',[0,2]),g('z',[0])])
    saved,p=b.make_plan(a,c,guard=guard)
    for arm,original in (('left',a),('right',c)):
        gates=[]
        for f in p['factors']:
            if f['kind']=='shared_CX':gates.append(f['gate']);continue
            gates.extend(dict(item,qubits=[f['support'][q] for q in item['qubits']]) for item in f[arm])
        out=columns.compare(original,r(4,gates),parallel=False)
        assert float(F(out['upper_exact']))<1e-50


def test_Gram_bound_is_operator_bound_not_Frobenius_only():
    from flint import acb_mat,ctx
    from trottertracks import c_pr_aware_df_zsxx_enclosure as z
    with ctx.workprec(192):
        M=acb_mat([[int(i==j)/8 for j in range(16)] for i in range(16)])
        assert b._operator_upper(M)==F(1,8)
        assert F(z._norm(M)['upper_exact'])>=F(1,2)


def test_invented_dense_operator_sanity_independent_of_partition():
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    rng=np.random.default_rng(20261105)
    for n in (2,3,4):
        q=QuantumCircuit(n);t=QuantumCircuit(n)
        for _ in range(7):
            control,target=rng.choice(n,2,replace=False);theta=float(rng.uniform(-.5,.5))
            q.cx(int(control),int(target));q.rz(theta,int(target))
            t.rz(theta,int(target));t.cx(int(control),int(target))
        a,c=fixed.record(q),fixed.record(t)
        out=b.compare(a,c,parallel=False)
        actual=np.linalg.norm(Operator(q).data-Operator(t).data,2)
        assert float(F(out['upper_exact']))>=actual-1e-12


def test_existing_column_only_admission_rejects_new_proof(monkeypatch):
    from test_research_c_compiled_raw_local import test_existing_column_only_integration_does_not_promote_local_proof
    monkeypatch.setattr(local,'compare',b.compare)
    test_existing_column_only_integration_does_not_promote_local_proof()
