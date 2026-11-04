"""Artificial-only normal forms and rigorous relative-operator tests."""
from copy import deepcopy
from fractions import Fraction as F
import math
import pytest
from trottertracks import c_relative_pauli_enclosure as p,c_cx_structural_blocks as blocks
from trottertracks import c_compiled_raw_local as local,c_arb_column_batch as columns
from trottertracks import c_pr_aware_df_fixed_certificate as fixed


def g(name,qs,*values):return dict(name=name,qubits=qs,params_hex=[float(x).hex() for x in values])
def r(n,gates=(),phase=0.):return dict(fixed.identity(n),gates=list(gates),phase_hex=float(phase).hex())


def checked(a,b,**policy):
    value=p.compare(a,b,parallel=False,**policy)
    assert p.check_binding(value,a,b)
    old=columns.compare(a,b,parallel=False)
    assert F(value['upper_exact'])>=F(old['lower_exact'])
    return value


def dense(record):
    import numpy as np
    from flint import ctx
    with ctx.workprec(192):
        cols=[fixed._column(record,i,{})[0] for i in range(1<<record['qubits'])]
        return np.array([[complex(float(col[i,0].real.mid()),float(col[i,0].imag.mid()))
            for col in cols] for i in range(len(cols))])


@pytest.mark.parametrize('name',['rz','ry','p'])
@pytest.mark.parametrize('theta',[.125,math.pi/2,-math.pi,3*math.pi/2,.5+2*math.pi])
def test_symbolic_normal_form_reconstructs_declared_operator(name,theta):
    import numpy as np
    from qiskit.quantum_info import Pauli
    a=r(2,[g('h',[0]),g('cx',[0,1]),g(name,[1],theta),g('sx',[1]),g('s',[0])],phase=.125)
    saved=local.expand(a);form=p.normal_form(saved);N=4;matrix=np.eye(N,dtype=complex)
    for item in form['rotations']:
        x=np.array([bool(item['x']>>i&1) for i in range(2)]);z=np.array([bool(item['z']>>i&1) for i in range(2)])
        P=Pauli((z,x)).to_matrix();angle=float(F(item['angle']['pi']))*math.pi+float(F(item['angle']['rad']))
        matrix=(math.cos(angle/2)*np.eye(N)-1j*math.sin(angle/2)*P)@matrix
    phase=float(F(form['phase']['pi']))*math.pi+float(F(form['phase']['rad']))
    matrix=np.exp(1j*phase)*dense(form['clifford_record'])@matrix
    assert np.linalg.norm(matrix-dense(a),2)<2e-14


@pytest.mark.parametrize('n',[2,3,9])
def test_cross_boundary_cancellation_preserves_full_register(n):
    a=r(n,[g('rz',[0],.25),g('cx',[0,1]),g('rz',[0],-.25)]);b=r(n,[g('cx',[0,1])])
    coarse=blocks.compare(a,b,parallel=False);new=checked(a,b)
    assert float(F(coarse['upper_exact']))>.2
    assert float(F(new['upper_exact']))<1e-50
    assert new['dimension']==1<<n and new['scalar_columns_evaluated']==1 and new['complete_basis_columns_evaluated']==0


def test_identical_operator_with_old_global_phase_sqrt2_is_representation_loss():
    hh=[g('h',[0]),g('h',[1])]
    a,b=r(2,hh+[g('cx',[0,1])]+hh),r(2,[g('cx',[1,0])])
    coarse=blocks.compare(a,b,guard=0,parallel=False)
    assert float(F(coarse['global_phase_upper_exact']))>1
    assert float(F(coarse['upper_exact']))>1
    assert float(F(checked(a,b)['upper_exact']))<1e-50


def test_near_Clifford_roundoff_is_charged_not_erased():
    a=r(1,[g('rz',[0],math.pi/2)],phase=math.pi/4);b=r(1,[g('s',[0])])
    result=checked(a,b)
    assert F(result['rotation_upper_exact'])>0
    assert 0<float(F(result['upper_exact']))<1e-15
    saved,plan=p.make_plan(a,b)
    assert plan['relative_reduction']['peeled']
    assert any(F(x['angle']['pi'])!=0 for x in plan['relative_reduction']['peeled'])


def test_many_noncommuting_rotations_with_charged_roundoff_cancel():
    gates=[]
    for j in range(20):gates += [g('rz',[0],.125+j/256),g('ry',[0],.25-j/512),g('cx',[0,1])]
    a=r(2,gates);b=deepcopy(a)
    for item in b['gates']:
        if item['params_hex']:item['params_hex']=[(float.fromhex(item['params_hex'][0])+2**-48).hex()]
    new=checked(a,b)
    assert new['remaining_rotations']==0 and new['peeled_rotations']>0
    assert 0<float(F(new['upper_exact']))<1e-11
    assert new['rotation_upper_exact']!='0'


def test_phase_and_SU2_center_are_not_fitted_away():
    a,b=r(2),r(2,phase=.125)
    assert float(F(checked(a,b)['upper_exact']))>.12
    a=r(2,[g('x',[0]),g('z',[0]),g('x',[0]),g('z',[0])])
    assert checked(a,r(2))['upper_exact']=='2'
    assert float(F(checked(r(1,[g('sx',[0])]*3),r(1,[g('sxdg',[0])]))['upper_exact']))<1e-50


def test_controlled_relative_phase_and_no_cancellation():
    value=checked(r(3),r(3,[g('p',[2],.25)]))
    assert float(F(value['upper_exact']))>.24
    value=checked(r(2,[g('rz',[0],.125)]),r(2,[g('ry',[0],.125)]))
    assert value['remaining_rotations']>=2 and float(F(value['upper_exact']))>.08


def test_non_scalar_Clifford_NOT_APPLICABLE_no_probe_fallback():
    value=p.compare(r(9,[g('h',[8])]),r(9),parallel=False)
    assert value['status']=='NOT_APPLICABLE_PROJECTIVE_CLIFFORD' and value['upper_exact'] is None
    assert not value['full_operator_coverage'] and not value['dense_fallback_used']


@pytest.mark.parametrize('fault',['nonunitary','literal','angle','phase','register'])
def test_invalid_targets_FAIL(fault):
    a=r(2,[g('rz',[0],.125)])
    if fault=='nonunitary':a['unitary']=False
    if fault=='literal':a['target']='literal_saved_atom'
    if fault=='angle':a['gates'][0]['params_hex']=['nan']
    if fault=='phase':a['phase_hex']='inf'
    if fault=='register':a['qubits']=True
    with pytest.raises((ValueError,KeyError)):p.compare(a,r(2),parallel=False)


@pytest.mark.parametrize('fault',['source','plan','dimension','finite_probe','lemma','calibration'])
def test_binding_and_false_promotion_refused(fault):
    a,b=r(3,[g('rz',[0],.125)]),r(3,[g('rz',[0],.125)])
    saved,plan=p.make_plan(a,b);cert=p.evaluate(a,b,saved=saved,plan=plan,parallel=False)
    if fault=='source':
        a['phase_hex']=.125.hex()
        with pytest.raises(ValueError):p.check_binding(cert,a,b)
    elif fault=='plan':
        plan['relative_reduction']['peeled'].append(dict(x=1,z=0,angle=p._angle(rad=F(1,100))))
        with pytest.raises(ValueError):p.evaluate(a,b,saved=saved,plan=plan,parallel=False)
    else:
        key={'dimension':'dimension','finite_probe':'finite_probe_operator_promotion','lemma':'exact_projective_scalar_lemma','calibration':'calibration_ready'}[fault]
        cert[key]=512 if fault=='dimension' else fault!='lemma'
        with pytest.raises(ValueError):p.check_binding(cert,a,b)


def test_artificial_raw_primitive_and_existing_P0():
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import XXPlusYYGate,RZZGate
    from trottertracks.c1_h4_native_fair import compile_checked,CompilerProfile
    q=QuantumCircuit(3);q.append(XXPlusYYGate(.375,.25),[0,1]);q.crz(.125,1,0);q.append(RZZGate(.25).control(1,annotated=False),[2,0,1])
    a,b=fixed.record(compile_checked(q,CompilerProfile())),fixed.record(q)
    result=p.compare(a,b,parallel=False)
    # This alternative's exact scalar lemma is restrictive for generic P0.
    # Refuse qualification rather than promoting one state column to an operator.
    assert result['status']=='NOT_APPLICABLE_PROJECTIVE_CLIFFORD'
    assert result['upper_exact'] is None and not result['full_operator_coverage']


def test_threshold_changes_only_charged_approximation_not_target():
    a,b=r(2,[g('rz',[0],.25),g('ry',[0],2**-45),g('rz',[0],-.25)]),r(2)
    exact=checked(a,b,threshold=F(0));peel=checked(a,b)
    assert peel['peeled_rotations']>0 and peel['remaining_rotations']==0
    assert float(F(peel['upper_exact']))<float(F(exact['upper_exact']))
    assert F(peel['rotation_upper_exact'])>0


def test_invented_dense_spectral_sanity():
    import numpy as np
    for n in (2,3):
        a=r(n,[g('rz',[0],.125),g('cx',[0,1]),g('ry',[1],.25)])
        b=r(n,[g('ry',[1],.25),g('cx',[0,1]),g('rz',[0],.125)])
        value=checked(a,b);actual=np.linalg.norm(dense(a)-dense(b),2)
        assert float(F(value['upper_exact']))>=actual-1e-13
