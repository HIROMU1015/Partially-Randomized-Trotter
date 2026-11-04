"""Invented states, literal blocks and explicit integer streams; no H4 load."""
from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path
import subprocess
import sys

import pytest

from trottertracks import c_direct_round_signal as m
from trottertracks import c_fresh_iid_calibration as fresh
from trottertracks import c_iid_calibration_integration as integration
from trottertracks import c_pr_aware_df_fixed_certificate as fixed
from trottertracks import c_pr_aware_df_zsxx_enclosure as z


from trottertracks.c_direct_round_artificial import fixture


def contains(pair,value):
    from flint import arb
    lo,hi = map(F,pair)
    if isinstance(value,(int,F)):
        assert lo <= value <= hi
    else:
        # Independent Arb384 analytic value is contained in the Arb192 result.
        # Rational tests below additionally have exact known answers.
        assert z._real(lo) <= value.lower() and value.upper() <= z._real(hi)


def test_analytic_phase_axes_complex_inner_product_and_rounding():
    from flint import ctx,acb,acb_mat
    args = fixture(); partial={}; result = m.evaluate_round(**args,partial=partial)
    with z._BALL_LOCK,ctx.workprec(384):
        contains(result['raw_signals']['compiled']['real'],z._real(.25).cos())
        contains(result['raw_signals']['compiled']['imag'],z._real(.25).sin())
        v=acb_mat([[1],[acb(0,1)]])
        assert m.inner_product(v,v) == 2
    contains(result['X'],0); contains(result['Y'],0)
    for a in partial['axes'].values():
        for state in a.values(): contains(state['norm_squared'],1)
    assert result['precision_bits']==192 and result['rounding_enclosed']
    assert result['statistical_confidence'] is None and not result['full_operator_error_certified']
    assert F(result['raw_signals']['compiled']['imag'][0]) > 0


def test_artificial_controlled512_state_scope_is_not_operator_scope():
    r=m.evaluate_round(**fixture(n=9))
    assert r['system_dimension']==256 and r['controlled_dimension']==512
    assert r['input_states_evaluated']==1 and r['full_output_amplitudes_evaluated']
    assert not r['full_operator_error_certified'] and not r['Stage_C_ready']
    contains(r['X'],0);contains(r['Y'],0)


def test_exact_small_matrix_state_propagation_and_control_order():
    from flint import ctx,acb_mat
    r=fixed.identity(2)
    r['gates']=[dict(name='sx',qubits=[0],params_hex=[]),dict(name='cx',qubits=[0,1],params_hex=[])]
    with z._BALL_LOCK,ctx.workprec(192):
        v,e=m.native_action(r,acb_mat([[1],[0],[0],[0]]))
        expected=[(F(1,2),F(1,2)),(0,0),(0,0),(F(1,2),F(-1,2))]
        for i,(re,im) in enumerate(expected):
            contains(m.interval(v[i,0].real,e),re);contains(m.interval(v[i,0].imag,e),im)
        assert e==0


def test_coherent_occurrence_product_and_same_xy_trajectory():
    from flint import ctx
    args=fixture(m_round=2);part={};r=m.evaluate_round(**args,partial=part)
    with z._BALL_LOCK,ctx.workprec(384):
        contains(r['raw_signals']['compiled']['real'],z._real(1).cos())
        contains(r['raw_signals']['compiled']['imag'],z._real(1).sin())
    assert [x['occurrence'] for x in part['occurrences']]==list(range(4))
    assert r['occurrences']==4 and r['round_word_id']==args['sample']['round_word_id']
    assert len({x['compiled_id'] for x in part['occurrences']})==1
    assert len({fresh.digest(x['binding']) for x in part['occurrences']})==4


def test_round_replay_is_linear_not_once_per_operator(monkeypatch):
    args=fixture(m_round=2);original=fresh.validate_sample;calls=[]
    def count(*a,**kw):calls.append(1);return original(*a,**kw)
    monkeypatch.setattr(fresh,'validate_sample',count)
    m.evaluate_round(**args)
    assert len(calls)==1


def test_global_phase_cancels_but_control_relative_phase_does_not():
    args=fixture();a=m.evaluate_round(**args)
    for i,p in enumerate(args['occurrences']):
        r=deepcopy(p['compiled']['record']);r['phase_hex']=.5.hex()
        p['raw']=integration.bind_operator(r,args['sample'],args['context'],i,role='raw')
        p['compiled']=integration.bind_operator(r,args['sample'],args['context'],i,role='compiled',raw_id=p['raw']['record_id'])
        p['actual_inverse']=m.bind_inverse(fixed.adjoint(r),p['compiled'],args['sample'],args['context'],i)
    b=m.evaluate_round(**args)
    for axis in ('X','Y'):
        assert max(map(abs,map(F,b[axis]))) < F(1,10**50)
        assert max(map(abs,map(F,a[axis]))) < F(1,10**50)
    assert F(b['raw_signals']['compiled']['imag'][0])>0


def test_actual_inverse_is_evaluated_not_replaced_by_an_ideal_adjoint():
    args=fixture();part={}
    args['occurrences'][0]['actual_inverse']=m.bind_inverse(fixed.identity(2),args['occurrences'][0]['compiled'],
        args['sample'],args['context'],0)
    m.evaluate_round(**args,partial=part)
    assert F(part['occurrences'][0]['actual_inverse_vs_adjoint_state_upper_exact'])>F(1,100)
    assert 'not_operator' in part['occurrences'][0]['inverse_qualification']
    args['occurrences'][0]['actual_inverse']['forward_id']='0'*64
    with pytest.raises(ValueError,match='SAME_FORWARD'): m.evaluate_round(**args)


def test_nonunitary_literal_signal_and_norm_are_not_clipped_or_normalized():
    args=fixture(scale=4,angle=0.);part={};r=m.evaluate_round(**args,partial=part)
    contains(r['raw_signals']['literal_atom']['real'],4)
    contains(r['X'],-3)
    contains(part['axes']['X']['literal_atom']['norm_squared'],F(17,2))
    assert F(r['X'][1]) < -2
    assert F(part['occurrences'][0]['literal_norm_upper_exact'])>=4


def test_nontrivial_literal_block_matrix_and_norm_growth():
    from flint import ctx,acb_mat
    from trottertracks.c_pr_aware_df_fixed_car import pack
    args=fixture(m_round=1,angle=0.)
    with z._BALL_LOCK,ctx.workprec(192):
        for p,B in zip(args['occurrences'],([[1,1],[0,1]],[[1,0],[1,1]])):
            p['reference']['blocks']=[dict(indices=[0,1],entries=pack(acb_mat(B)))]
    args['state']=m.bind_state([['3/5','0'],['4/5','0']],source_id='d'*64,
        target_id=fresh.digest(args['target']),provenance='INVENTED normalized rational state')
    part={};r=m.evaluate_round(**args,partial=part)
    contains(r['raw_signals']['literal_atom']['real'],F(13,5))
    contains(r['X'],F(-8,5))
    contains(part['axes']['X']['literal_atom']['norm_squared'],F(39,10))
    assert F(part['occurrences'][0]['literal_norm_upper_exact'])>1


def test_literal_radius_outward_dyadic_reenclosure_prevents_fraction_growth():
    from flint import ctx,acb_mat
    with z._BALL_LOCK,ctx.workprec(192):
        prepared=([(list(range(2)),acb_mat([[1,0],[0,1]]))],F(1001,1000))
        state=acb_mat([[1],[0],[0],[0]]);radius=F(1,10**50)
        for _ in range(128):state,radius=m.literal_action(prepared,state,radius)
        assert radius >= F(1001,1000)**128*F(1,10**50)
        assert radius.denominator & (radius.denominator-1)==0
        assert len(str(radius))<250


def test_initial_state_binding_normalization_and_error_propagation():
    args=fixture();r=m.evaluate_round(**args)
    with pytest.raises(ValueError,match='UNIT_STATE'):
        m.bind_state([['2','0'],['0','0']],source_id='d'*64,target_id=fresh.digest(args['target']),provenance='invented')
    args['state']=m.bind_state([['1','0'],['0','0']],source_id='d'*64,target_id=fresh.digest(args['target']),
        uncertainty='1/100',provenance='INVENTED supplied vector2 state premise')
    wide=m.evaluate_round(**args)
    for a in ('X','Y'):
        assert F(wide[a][0]) < F(r[a][0]) and F(wide[a][1]) > F(r[a][1])
    assert wide['state_qualification'].startswith('conditional')


@pytest.mark.parametrize('fault',['source','candidate','word','angle','phase','register','target','precision','partition',
    'missing_occurrence','compile_failed','compile_incomplete','state','shell','inverse_word'])
def test_wrong_or_unqualified_inputs_rejected(fault):
    args=fixture();p=args['occurrences'][0]
    if fault=='source': p['raw']['calibration_binding']['source_id']='0'*64
    if fault=='candidate': p['compiled']['calibration_binding']['candidate']='wrong'
    if fault=='word': p['raw']['word']=[]
    if fault=='angle': p['reference']['binding']['word'][0][1]=.5.hex()
    if fault=='phase': p['compiled']['record']['phase_hex']=.5.hex()
    if fault=='register': p['raw']['record']['qubits']=3
    if fault=='target': p['reference']['binding']['target_id']='0'*64
    if fault=='precision': p['reference']['precision_bits']=160
    if fault=='partition': p['reference']['blocks'].append(deepcopy(p['reference']['blocks'][0]))
    if fault=='missing_occurrence': args['occurrences']=[]
    if fault.startswith('compile_'): p['compile_status']='FAILED' if fault=='compile_failed' else None
    if fault=='state': args['state']['source_id']='0'*64
    if fault=='shell': args['shells']['Y']['gates'][0]['qubits']=[0]
    if fault=='inverse_word': p['actual_inverse']['calibration_binding']['word_id']='0'*64
    with pytest.raises((ValueError,KeyError)): m.evaluate_round(**args)


def test_head_and_y_shell_change_is_evaluated():
    args=fixture(angle=0.)
    args['shells']['head']=fixed.identity(2)
    r=m.evaluate_round(**args)
    # Without preparation there is no interference: X/Y are zero, literal X=1.
    contains(r['raw_signals']['compiled']['real'],0);contains(r['X'],-1)
    args=fixture();args['shells']['Y']=deepcopy(args['shells']['X'])
    r=m.evaluate_round(**args)
    assert r['raw_signals']['compiled']['real']==r['raw_signals']['compiled']['imag']
    assert F(r['Y'][0])>F(1,2)


def test_partial_exception_preserved_and_mandatory_stop(tmp_path,monkeypatch):
    args=fixture(m_round=1);path=fresh.open_batch(tmp_path/'batch')
    original=m.literal_action;calls=[]
    def fail_second(*a,**kw):
        calls.append(1)
        if len(calls)==2: raise ArithmeticError('INVENTED second occurrence failure')
        return original(*a,**kw)
    monkeypatch.setattr(m,'literal_action',fail_second)
    r=m.process_round(path,**args)
    assert r['status']=='NUMERICAL_UNEVALUABLE' and r['evidence']['round_enclosure'] is None
    assert len(r['evidence']['partial']['occurrences'])==1
    assert len(r['evidence']['partial']['input_occurrences'])==2
    assert fresh.read_batch(path)==[r]
    assert json.loads((path/'STOP.json').read_text())['mandatory_stop']
    with pytest.raises(ValueError,match='STOP'): m.process_round(path,**fixture(index=1))


def test_interruption_keeps_original_sample_and_partial(tmp_path,monkeypatch):
    path=fresh.open_batch(tmp_path/'batch')
    def interrupt(*a,**kw):raise KeyboardInterrupt('INVENTED interruption')
    monkeypatch.setattr(m,'literal_action',interrupt)
    with pytest.raises(KeyboardInterrupt):m.process_round(path,**fixture())
    assert fresh.read_batch(path)[0]['status']=='INCOMPLETE' and (path/'STOP.json').exists()


@pytest.mark.parametrize('status,outcome',[('FAILED','P0_COMPILE_FAILED'),(None,'INCOMPLETE')])
def test_failed_and_unfinished_compiler_attempts_retained(tmp_path,status,outcome):
    args=fixture();args['occurrences'][0]['compile_status']=status
    path=fresh.open_batch(tmp_path/'batch');row=m.process_round(path,**args)
    assert row['status']==outcome and (path/'STOP.json').exists()
    assert row['evidence']['partial']['input_occurrences'][0]['compile_status']==status


def payload(rows,args,**kwargs):
    support=dict(qualification='deterministic_population_bound',m=0,target_id=fresh.digest(args['target']),
        state_id=args['state']['state_id'],lower_exact='-3',upper_exact='3',provenance='INVENTED support premise')
    options=dict(context=args['context'],planned_ids=[r['sample']['sample_id'] for r in rows],
        state_id=args['state']['state_id'],target_id=fresh.digest(args['target']),axis_support=support,
        delta_cal='1/20',failure_policy='record_all_fixed_N',iid_assumed=True,
        compiler_domain=dict(kind='structural_totality_argument',scope='all_legal_q_words',
            gates='exact_ideal_P0_unitaries',provenance='INVENTED premise'))
    options.update(kwargs)
    return m.calibration_payload(rows,**options)


def test_fresh_schema_to_unchanged_statistics_without_full512_certificates(tmp_path):
    path=fresh.open_batch(tmp_path/'batch');rows=[]
    for i in range(2):
        args=fixture(index=i);rows.append(m.process_round(path,**args))
    fresh.finish_batch(path)
    p=payload(rows,args)
    assert p['status']=='CONDITIONAL_MATHEMATICAL_IID_INPUTS'
    result=m.evaluate_calibration(p)
    assert result['bound'] is not None and result['confidence']['calibration_success_lower']=='19/20'
    assert result['all_word_uniform'] is False and not result['Stage_C_ready']
    assert rows[0]['evidence']['round_enclosure']['statistical_confidence'] is None
    assert 'per_word_enclosures' not in rows[0]['evidence']
    assert len(rows[0]['evidence']['round_enclosure']['raw_signals']['compiled']['real'])==2
    # Old adapter cannot interpret this new qualification or interval schema.
    old=fresh.calibration_payload(rows,context=args['context'],planned_ids=p['policy']['sample_ids'],
        method='direct_round',delta_cal='1/20',failure_policy='record_all_fixed_N',iid_assumed=True)
    assert old['status']=='NOT_READY'


@pytest.mark.parametrize('fault',['missing','failed','kernel','confidence','full_operator','state','interval','planned',
    'domain','delta','policy','iid'])
def test_statistics_fail_closed_without_missing_zero_or_qualification_upgrade(tmp_path,fault):
    args=fixture();row=m.process_round(fresh.open_batch(tmp_path/'batch'),**args)
    e=row['evidence']['round_enclosure'];opts={}
    if fault=='missing':e['Y']=None
    if fault=='failed':row['status']='INCOMPLETE'
    if fault=='kernel':row['sample']['unit']='kernel'
    if fault=='confidence':e['statistical_confidence']='95%'
    if fault=='full_operator':e['full_operator_error_certified']=True
    if fault=='state':e['state_id']='0'*64
    if fault=='interval':e['X']=['0','0']
    if fault=='planned':opts['planned_ids']=[row['sample']['sample_id'],'missing']
    if fault=='domain':opts['compiler_domain']=None
    if fault=='delta':opts['delta_cal']=None
    if fault=='policy':opts['failure_policy']=None
    if fault=='iid':opts['iid_assumed']=False
    try:p=payload([row],args,**opts)
    except (ValueError,KeyError):return
    assert p['status']=='NOT_READY' and m.evaluate_calibration(p)['bound'] is None


def test_science_entry_and_stage_c_are_not_authorized():
    for kw in ({},{'execute_science':True}):
        with pytest.raises(PermissionError,match='MANDATORY_STOP'):m.run_science(**kw)
    from trottertracks import c_iid_grid_expected_bias as expected
    assert expected.assess_if_adopted(None,None,dict(eligible=False),policy_adopted=False)['status']=='NOT_READY_NO_ACCURACY_CALLS'


def test_passive_entry_does_not_import_numerical_libraries():
    code='''
import sys
from pathlib import Path
class Block:
 def find_spec(self,name,*args):
  if name.split('.')[0] in ('numpy','scipy','flint','qiskit','ffsim','pyscf'):
   raise AssertionError('NO_NUMERICAL_IMPORT:'+name)
sys.meta_path.insert(0,Block())
from trottertracks.c_direct_round_signal import preflight
r=preflight(Path.cwd())
assert r['scientific_calls']==0 and not r['execution_ready'] and r['mandatory_stop']
'''
    subprocess.run([sys.executable,'-B','-c',code],cwd=Path(__file__).resolve().parents[2],check=True)
