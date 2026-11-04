"""Artificial bounds, small circuits and passive hash gates; no real H4 math."""
from copy import deepcopy
from fractions import Fraction
import math
from pathlib import Path
import subprocess
import sys

import pytest

from trottertracks import c_all_law_signal_ca as m

ROOT=Path(__file__).resolve().parents[2]


def coverage(**kw):
    return dict(dict(scope=m.COVERAGE,compiler_path='whole_kernel_P0',
        final_transform_covered=True,phase_sensitive=True,actual_adjoint_covered=True,
        dimension=512,provenance='synthetic supplied uniform argument',kind='all_law_numerical_bound'),**kw)


def premises(**kw):
    return dict(dict(coverage=m.COVERAGE,actual_joint_stream_covered=True,
        independent_shots_justified=True,independent_coherent_occurrences=True,
        final_whole_kernel_P0_covered=True,phase_sensitive=True,actual_adjoint_covered=True,
        controlled_dimension=512,reference_bridge_target='saved_CAR_raw_finite_RTE',
        synthesis_target='exact_saved_atom_map_under_same_actual_stream',
        provenance='invented complete scientific inputs;not an H4 proof'),**kw)


def segment(epsilon=0.):
    return dict(operator_bound_pilot=epsilon,norm_upper_pilot=1.)


def parts(kind='numerical_bound',value='1/10000000'):
    return {name:[dict(m=i,quantity=name,kind=kind,bound=value,norm='raw_complex_signal',
        provenance='artificial exact input',rounding_enclosed=kind=='rigorous_bound',
        **({'reference_uncertainty_included':False} if name=='reference_bridge' else {})) for i in range(13)] for name in m.PARTS}


def packet():
    from test_research_c_pr_aware_df_stage_c import packet as old
    return old('toy',A=.8)


def attach(comp=None,pre=None):
    identity=dict(candidate='toy',SOURCE='toy_source',compiler='toy_P0',law='toy_joint_law')
    return m.attach_evidence(packet(),expected_identity=identity,observed_identity=identity,
        components=parts() if comp is None else comp,premises=premises() if pre is None else pre)


@pytest.mark.parametrize('epsilon',[0.,1e-8,1e-5])
def test_uniform_numerical_propagation_and_full_norm_growth(epsilon):
    from trottertracks.c1_h4_full_synthesis import propagate
    k=segment(epsilon);shells={a:segment(0.) for a in ('head','X','Y')}
    rows=m.numerical_synthesis_rows(k,shells,coverage())
    assert [r['m'] for r in rows]==list(range(13))
    for row in rows:
        ref=propagate(k,shells,s=1,m=row['m'],saved_pf_error=0.)
        assert row['bound']==ref['complex_signal_bound_pilot']
        if epsilon==0:assert row['bound']==0
        else:assert row['bound']>=2*(1<<row['m'])*epsilon
        assert row['rounding_enclosed'] is False


@pytest.mark.parametrize('change',[dict(scope='native32_samples'),dict(compiler_path='eventwise_P0'),
    dict(final_transform_covered=False),dict(phase_sensitive=False),dict(actual_adjoint_covered=False),
    dict(dimension=72),dict(kind='rigorous_all_law_bound')])
def test_sample_sector_eventwise_phase_and_adjoint_do_not_cover_whole_P0(change):
    with pytest.raises(ValueError):m.numerical_synthesis_rows(segment(),{x:segment() for x in ('head','X','Y')},coverage(**change))


def test_exact_unitary_preview_remains_conditional():
    def proof(e='1/10000'):
        return dict(kind='analytic_unitary_bound',norm='full_controlled_operator_2',
            phase_sensitive=True,reference_error='0',provenance='toy identity argument',
            compiler_path='whole_kernel_P0',coverage='all_final_whole_kernel_outputs',epsilon=e)
    v=m.exact_unitary_preview([proof()],{x:proof('0') for x in ('head','X','Y')},2)
    assert Fraction(v['candidate_raw_complex_bound'])==Fraction(16,10000)
    assert v['actual_H4_qualification'] is None and v['proof_inputs_are_not_verified_by_schema']


def test_shells_and_coherent_repetition_against_known_phase():
    import numpy as np
    theta=1e-6
    # Exact toy controlled relative phase, not an H4 circuit.
    C=np.diag([1.,np.exp(1j*theta)]);I=np.eye(2)
    eps=float(np.linalg.norm(C-I,2))
    k=segment(eps);shells={a:segment(eps) for a in ('head','X','Y')}
    values=m.numerical_synthesis_rows(k,shells,coverage())
    for row in values:
        n=1<<row['m'];difference=abs(np.exp(1j*(n+2)*theta)-1)
        assert difference<=row['bound']
    assert values[-1]['bound']>values[0]['bound']


def test_controlled_relative_phase_global_phase_and_actual_adjoint():
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from trottertracks.c1_stage2_native import actual_adjoint
    from trottertracks.c_pr_aware_df_stage_b2_compat import SharedQiskitCalls
    q=QuantumCircuit(2);q.h(0);q.cx(1,0);q.rz(.13,0);q.p(.17,1);q.global_phase=.23
    U=Operator(q).data
    wrong=q.copy();wrong.data.pop()
    assert np.max(abs(Operator(wrong).data-U))>.1
    phased=np.exp(.2j)*U
    assert np.linalg.norm(phased-U)>0  # do not phase-fit a full controlled target
    # A common outer global phase leaves an expectation unchanged; a controlled
    # base-unitary phase becomes a relative phase and cannot be dropped.
    plus=np.array([1.,1.])/math.sqrt(2)
    assert abs(np.vdot(plus,np.diag([1.,np.exp(.2j)])@plus)-1)>0
    lock=SharedQiskitCalls(lambda k,l,fn:fn())
    with lock.adjoint_builds():inverse=actual_adjoint(q)
    assert np.max(abs(Operator(inverse).data@U-np.eye(4)))<1e-14


@pytest.mark.parametrize('probabilities',[(.25,.75),(.251,.749)])
def test_categorical_TV_signed_complex_signal_and_r32_s1(probabilities):
    from trottertracks import c_pr_aware_df_stage_c2 as old
    table=old.analyze_table([1,3],probabilities)
    eta=Fraction(table['eta_exact'])
    for i in (0,12):
        v=m.categorical_preview(table,i)
        assert v['events']==32*(1<<i)
        assert Fraction(v['conditional_raw_bound_exact'])==min(2,2*v['events']*eta)
        assert v['actual_implemented_law_error'] is None
    q=list(map(Fraction,table['grid']['probabilities_exact']))
    # signed complex event signals, each modulus <=1
    delta=abs(complex(float(q[0]-Fraction(1,4))*1j-float(q[1]-Fraction(3,4))))
    assert delta<=float(2*eta)


@pytest.mark.parametrize('p',[[-.1,1.1],[.4,.4],[float('nan'),1.],[float('inf'),0.]])
def test_invalid_probability_inputs(p):
    from trottertracks.c_pr_aware_df_stage_c2 import choice_grid
    with pytest.raises(ValueError):choice_grid(p)


def test_event_order_and_reused_coherent_occurrence_are_distinct():
    import numpy as np
    X=np.array([[0.,1.],[1.,0.]]);Z=np.diag([1.,-1.]);mean=(X+Z)/2
    assert not np.allclose(X@Z,Z@X)
    assert not np.allclose(mean@mean,(X@X+Z@Z)/2)
    with pytest.raises(ValueError,match='OCCURRENCE'):m.categorical_preview({},0,independent_occurrences=False)


def test_disjoint_components_existing_evidence_schema_and_no_attenuation_twice():
    p,q=attach()
    assert q['kind']=='all_law_numerical_bound' and q['all_law_bound_qualified']
    assert q['round_errors'][0]['synthesis']==pytest.approx(1e-7)
    assert q['round_errors'][0]['law']==pytest.approx(4e-7)
    assert q['actual_CA_achieved'] is None and q['guaranteed_CA'] is None
    from trottertracks import c_pr_aware_df_stage_c as c1
    target=c1.accuracy.AccuracyTarget(**c1.TARGET)
    value=c1.assess_round(target,p,q,p['rounds'][0])
    assert value['raw_signal_error']==pytest.approx(.8*(.002+.003)+5e-7)


@pytest.mark.parametrize('part',m.PARTS)
def test_unknown_components_remain_null_and_stop_accuracy(part):
    comp=parts();comp[part]=None;p,q=attach(comp)
    assert q['kind']=='UNQUALIFIED' and p['error_evidence'] is None and part in q['missing']
    from trottertracks import c_pr_aware_df_stage_c as c1
    out=m.assess_ready_packet(c1.accuracy.AccuracyTarget(**c1.TARGET),p,q)
    assert out['rounds']==[] and out['energy'] is None and out['actual_CA_achieved'] is None


@pytest.mark.parametrize('kind',['observed_residual','sample_max','assumed_bound'])
def test_limited_or_assumed_evidence_never_promoted(kind):
    comp=parts();comp['synthesis'][0]['kind']=kind
    assert attach(comp)[1]['kind']=='UNQUALIFIED'


@pytest.mark.parametrize('field',['actual_joint_stream_covered','independent_shots_justified',
    'independent_coherent_occurrences','final_whole_kernel_P0_covered','phase_sensitive','actual_adjoint_covered'])
def test_counterfactual_IID_or_incomplete_coverage_not_actual_law(field):
    assert attach(pre=premises(**{field:False}))[1]['kind']=='UNQUALIFIED'


@pytest.mark.parametrize('fault',['missing_round','wrong_units','double_reference','PF_in_law','negative','duplicate_round'])
def test_bad_bounds_or_double_count_rejected(fault):
    c=parts()
    if fault=='missing_round':c['synthesis'].pop()
    elif fault=='wrong_units':c['synthesis'][0]['norm']='energy_Ha'
    elif fault=='double_reference':c['reference_bridge'][0]['reference_uncertainty_included']=True
    elif fault=='PF_in_law':c['categorical'][0]['quantity']='PF'
    elif fault=='negative':c['synthesis'][0]['bound']='-1/2'
    elif fault=='duplicate_round':c['categorical'][0]['m']=1
    with pytest.raises(ValueError):attach(c)


@pytest.mark.parametrize('field',['candidate','SOURCE','compiler','law'])
def test_wrong_identity(field):
    expected={k:'toy' for k in ('candidate','SOURCE','compiler','law')};observed=dict(expected,**{field:'wrong'})
    with pytest.raises(ValueError,match='IDENTITY'):
        m.attach_evidence(packet(),expected_identity=expected,observed_identity=observed,components=parts(),premises=premises())


@pytest.mark.parametrize('kind',['numerical_bound','rigorous_bound'])
def test_ready_synthetic_16_calls_never_promotes_actual_or_guaranteed_CA(tmp_path,kind):
    from trottertracks.c1_stage3_integrated import Calls
    from trottertracks import c_pr_aware_df_stage_c as c1
    p,q=attach(parts(kind));call=Calls(tmp_path,m.ACCURACY_LIMITS,16)
    out=m.assess_ready_packet(c1.accuracy.AccuracyTarget(**c1.TARGET),p,q,call)
    assert dict(call.counts)==m.ACCURACY_LIMITS
    assert out['status']=='NUMERICAL_IMPLEMENTED_LAW_SUFFICIENT_CONDITION'
    assert out['energy']['numerical_pilot'] is True
    assert out['actual_CA_achieved'] is None and out['guaranteed_CA'] is None and out['quantum_shots_drawn']==0
    assert q['kind']==('rigorous_all_law_bound' if kind=='rigorous_bound' else 'all_law_numerical_bound')
    with pytest.raises(ValueError,match='TOTAL_CALL_LIMIT'):call('qualification_summary','extra',lambda:None)


def test_phase_room_exceeded_stops_later_rounds_without_rescue(tmp_path):
    from trottertracks.c1_stage3_integrated import Calls
    from trottertracks import c_pr_aware_df_stage_c as c1
    c=parts();c['synthesis'][0]['bound']='1';p,q=attach(c)
    call=Calls(tmp_path,m.ACCURACY_LIMITS,16)
    out=m.assess_ready_packet(c1.accuracy.AccuracyTarget(**c1.TARGET),p,q,call)
    assert len(out['rounds'])==1 and out['rounds'][0]['shots'] is None and out['energy'] is None
    assert sum(call.counts.values())==3 and 'energy_assessment' not in call.counts


def test_correctness_FAIL_existing_journal_partial_and_STOP(tmp_path):
    from trottertracks.c1_stage3_integrated import Calls,Stopped
    from trottertracks import c_pr_aware_df_stage_c as c1
    p,q=attach();p['error_evidence']['round_errors'][0]['synthesis']=-1
    call=Calls(tmp_path,m.ACCURACY_LIMITS,16)
    with pytest.raises(ValueError):m.assess_ready_packet(c1.accuracy.AccuracyTarget(**c1.TARGET),p,q,call)
    assert call.stop.is_set() and call.first_failure['kind']=='round_accuracy'
    with pytest.raises(Stopped):call('round_accuracy','dependent',lambda:None)
    m.io.save(tmp_path/'STOP.json',dict(failed=True,scientific_units=sum(call.counts.values()),retry=0,resume=0,mandatory_stop=True))
    assert 'FAIL' in (tmp_path/'science_calls.jsonl').read_text()
    assert m.io.read(tmp_path/'STOP.json')['scientific_units']==2


def test_nonexecutable_design_and_permission_touch_no_inputs(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'preflight',lambda *a:pytest.fail('no input should be touched'))
    with pytest.raises(PermissionError):m.run(tmp_path,'missing')
    with pytest.raises(RuntimeError,match='DESIGN_NOT_EXECUTABLE'):m.run(tmp_path,'missing',execute_science=True)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('fault',['none','source','input','candidate','compiler','SOURCE','law','STOP','fresh','retry','resume'])
def test_real_metadata_preflight_only_no_H4_numeric_work(monkeypatch,fault):
    read,sha=m.io.read,m.io.sha
    cp=ROOT/m.CONTRACT;pr=ROOT/m.records()['PR_result'];oldc=ROOT/m.records()['PR_contract']
    stop=ROOT/m.records()['PR_STOP'];law=ROOT/m.records()['PR_law']
    def modified_read(path):
        p=Path(path);v=deepcopy(read(p))
        if fault in ('retry','resume') and p==cp:v[fault]=1
        if fault=='candidate' and p==pr:v['arms']['PR']['candidate']='wrong'
        if fault=='compiler' and p==oldc:v['compiler']['seed_transpiler']=42
        if fault=='SOURCE' and p==oldc:v['source_manifest_sha256']='wrong'
        if fault=='STOP' and p==stop:v['failed']=True
        if fault=='law' and p==law:v['atoms']=215
        return v
    monkeypatch.setattr(m.io,'read',modified_read)
    bad=ROOT/('src/trottertracks/c_all_law_signal_ca.py' if fault=='source' else m.records()['PR_result'])
    if fault in ('source','input'):monkeypatch.setattr(m.io,'sha',lambda p:'wrong' if Path(p)==bad else sha(p))
    if fault=='fresh':
        exists=Path.exists
        monkeypatch.setattr(Path,'exists',lambda p:True if p==ROOT/m.FRESH else exists(p))
    if fault=='none':
        c,v=m.preflight(ROOT,cp)
        assert v['scientific_calls']==0 and v['scientific_execution_ready'] is False
        assert c['executable'] is False and c['maximum_logical_units'] is None
    else:
        with pytest.raises(ValueError):m.preflight(ROOT,cp)


def test_passive_preflight_with_numerical_imports_forbidden():
    code="""import importlib.abc,sys
from pathlib import Path
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'numpy','scipy','qiskit','ffsim','jax','pyscf','openfermion','flint'}:raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
from trottertracks import c_all_law_signal_ca as m
_,s=m.preflight(Path('.'),m.CONTRACT)
assert s['scientific_calls']==0 and not s['scientific_execution_ready']
"""
    subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,check=True)
