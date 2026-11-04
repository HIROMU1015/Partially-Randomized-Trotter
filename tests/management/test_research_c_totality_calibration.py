"""Invented gates/trials only; real H4 arrays/records/QPY are never evaluated."""
from copy import deepcopy
from fractions import Fraction as F
import json,subprocess,sys
from pathlib import Path
import pytest
from trottertracks import c_arb_column_batch as fast
from trottertracks import c_pr_aware_df_fixed_certificate as old
from trottertracks import c_totality_calibration as m


def analytic_phase(delta):
    from flint import acb,ctx
    from trottertracks import c_pr_aware_df_zsxx_enclosure as z
    with ctx.workprec(192):
        a=abs((acb(0,1)*z._real(delta)).exp()-1)
        return z._fraction(a.lower()),z._fraction(a.upper())


@pytest.mark.parametrize('n,batch',[(2,1),(2,3),(3,8),(9,16)])
@pytest.mark.parametrize('cancel',[False,True])
def test_batch_full_basis_encloses_same_mathematical_phase(n,batch,cancel):
    a=old.identity(n);a['gates']=[dict(name='h',qubits=[0],params_hex=[]),dict(name='cx',qubits=[0,1],params_hex=[]),dict(name='rz',qubits=[1],params_hex=[(.3).hex()])]
    b=deepcopy(a);b['phase_hex']=(.125).hex()
    value=fast.compare(a,b,batch_size=batch,cancel_common=cancel,parallel=(n==9))
    lo,hi=analytic_phase(F(1,8))
    assert F(value['upper_exact'])>=lo and F(value['lower_exact'])<=hi
    assert value['columns_evaluated']==1<<n and len(value['columns'])==1<<n
    assert value['precision_bits']==192 and not value['phase_fit'] and value['original_full_operator_coverage']
    assert b['phase_hex']==(.125).hex() and len(a['gates'])==3
    if cancel:assert value['factor_reduction']['prefix_gates']==3


def test_common_suffix_and_prefix_keep_operator_norm_and_global_phase():
    a=old.identity(2);p=dict(name='cx',qubits=[0,1],params_hex=[]);q=dict(name='h',qubits=[1],params_hex=[])
    a['gates']=[p,dict(name='rz',qubits=[0],params_hex=[(.125).hex()]),q]
    b=deepcopy(a);b['gates'][1]['params_hex']=[(.25).hex()];b['phase_hex']=(.25).hex()
    aa,bb,proof=fast.reduce_pair(a,b)
    assert proof['prefix_gates']==proof['suffix_gates']==1
    assert aa['phase_hex']==a['phase_hex'] and bb['phase_hex']==b['phase_hex']
    x=fast.compare(a,b,parallel=False);ref=old.compare(a,b,parallel=False)
    assert F(x['lower_exact'])<=F(ref['upper_exact']) and F(ref['lower_exact'])<=F(x['upper_exact'])


@pytest.mark.parametrize('fault',['unknown','nonfinite','register','batch','literal'])
def test_unsupported_or_nonunitary_target_is_not_silently_substituted(fault):
    a=old.identity(2);b=deepcopy(a);kw={}
    if fault=='unknown':a['gates']=[dict(name='opaque',qubits=[0],params_hex=[])]
    if fault=='nonfinite':a['phase_hex']='inf'
    if fault=='register':b=old.identity(3)
    if fault=='batch':kw['batch_size']=True
    if fault=='literal':b=dict(target='literal CAR',matrix=[[1,2],[0,1]])
    with pytest.raises((ValueError,KeyError)):fast.compare(a,b,parallel=False,**kw)


def pop():
    return dict(status='VALIDATED_LITERAL_POPULATION_NORM',D_norm_upper_exact='10001/10000',
                d_range_upper_exact='20001/10000',all_q_words_compile_success_qualified=False,
                accepted_P0_C_norm_exact='1',no_sample_maximum_used=True)


def test_fixed_word_is_not_population_or_accuracy_qualification():
    finite=dict(status='STAGE_P_MISSING_OUTPUTS_COMPLETE',missing_outputs_obtained=True,original_41_call_run_complete=False)
    value=m.calibration_gate(pop(),fixed_word_record=finite)
    assert value['graph']['fixed_word_operator_enclosure']
    assert not value['graph']['all_q_words_compile_success_qualified']
    assert not value['graph']['IID_population_expected_bias_qualified']
    assert value['existing_range_gate']['support'] is None
    assert value['existing_accuracy_gate']['status']=='NOT_READY_NO_ACCURACY_CALLS'
    assert value['expected_bias_upper'] is None and value['Stage_C_calls']==0
    assert F(value['graph']['operator_error_range_exact'])>2
    assert not m.evidence_graph(pop())['fixed_word_operator_enclosure']


def trials(*outcomes):
    return [dict(draw_id=i,word_id='duplicate-word-allowed',outcome=x,retry=0,resume=0,resampled=False) for i,x in enumerate(outcomes)]


@pytest.mark.parametrize('kind',['COMPILE_EXCEPTION','CONTROLLER_GUARD_FAILURE','INVALID_NATIVE_OUTPUT','RAW_BUILD_FAILURE'])
def test_failure_is_retained_and_zero_success_is_not_totality(kind):
    rows=trials('SUCCESS',kind,'SUCCESS')
    r=m.failure_rate(rows,planned_N=3,delta='1/20')
    assert r['failures']==1 and F(r['failure_upper'])>0 and not r['all_q_words_compile_success_qualified']
    stop=m.failure_rate(rows,planned_N=3,delta='1/20',experiment='stop_on_failure')
    assert stop['failure_upper'] is None
    zero=m.failure_rate(trials(*['SUCCESS']*20),planned_N=20,delta='1/20')
    assert 0<F(zero['failure_upper'])<1


@pytest.mark.parametrize('kind',['UNKNOWN','NONTERMINATION_UNRESOLVED','EXECUTION_INTERRUPTION'])
def test_unknown_and_nontermination_cannot_complete_confidence(kind):
    v=m.failure_rate(trials('SUCCESS',kind),planned_N=2,delta='1/20')
    assert v['failure_upper'] is None


def test_discard_replacement_retry_and_stopped_prefix_are_rejected():
    rows=trials('SUCCESS','COMPILE_EXCEPTION','SUCCESS')
    with pytest.raises(ValueError):m.failure_rate([rows[0],rows[2]],planned_N=3,delta='.05')
    for key,value in [('retry',1),('resume',1),('resampled',True)]:
        bad=deepcopy(rows);bad[1][key]=value
        with pytest.raises(ValueError):m.failure_rate(bad,planned_N=3,delta='.05')
    assert m.failure_rate(rows[:1],planned_N=3,delta='.05')['failure_upper'] is None


def test_raw_and_conditional_compile_denominators_are_distinct():
    rows=trials('RAW_BUILD_FAILURE','COMPILE_EXCEPTION','SUCCESS')
    p=m.failure_rate(rows,planned_N=3,delta='.05',event='pipeline')
    c=m.failure_rate(rows,planned_N=3,delta='.05',event='compile_given_raw_success')
    assert (p['denominator'],p['failures'])==(3,2) and (c['denominator'],c['failures'])==(2,1)


def test_requests_cache_and_same_realized_inverse_keep_draws():
    a=m.compile_requests([1]*13,[2]*13)
    assert a['word_draws']==3*8191 and a['actual_inverse_new_word_draws']==0
    b=m.compile_requests([1]*13,[2]*13,cache='exact_word_source_environment',deterministic_word_map=True)
    assert b['word_draws']==a['word_draws'] and b['actual_compile_calls_interval']==[1,3*8191]
    with pytest.raises(ValueError):m.compile_requests([1]*13,[2]*13,cache='exact_word_source_environment')
    assert m.compile_requests(None,None)['word_draws'] is None


def test_confidence_events_not_rounds_and_abort_is_not_delta_compile():
    rows=[dict(id='kernel',kind='calibration',delta='1/1000')]*13+[dict(id='compile',kind='compile_confidence',delta='1/2000')]
    v=m.confidence(rows,p_fail_upper='1/10000',requests=10)
    assert v['unique_confidence_events']==2 and v['delta_cal']=='1/1000'
    assert v['delta_compile']=='1/2000' and v['operational_abort_upper']=='1/1000'
    assert v['measurement_alpha']=='1/100' and not v['joint_001_policy_adopted']
    assert v['unconditional_failure_upper'] is None
    assert F(v['candidate_union_under_RPE_scope_premise'])==F('1/100')+F('1/1000')+F('1/2000')+F('1/1000')
    assert m.confidence(rows)['operational_abort_upper'] is None
    with pytest.raises(ValueError):m.confidence(rows+[dict(id='kernel',kind='calibration',delta='.02')])


def test_discarding_failed_word_changes_law_in_invented_example():
    # Equally likely words a,b;only a compiles. The retained mean of D is0,
    # while its original-law mean is1/2. Neither filtering nor cache fixes it.
    assert F(0)!= (F(0)+F(1))/2
    with pytest.raises(ValueError):m.compiler.require_complete_sample(['SUCCESS','COMPILE_EXCEPTION'],2)
    with pytest.raises(ValueError):m.compiler.require_complete_sample(['SUCCESS'],2)


def test_calibration_entry_remains_nonexecutable():
    with pytest.raises(PermissionError):m.run_science()
    with pytest.raises(NotImplementedError):m.run_science(execute_science=True)


def test_Codex_A_adapter_compiles_only_invented_grammar_once(monkeypatch):
    from qiskit import QuantumCircuit
    from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
    from trottertracks.c_pr_aware_df_stage_b2_compat import SharedQiskitCalls
    q=QuantumCircuit(3);q.rz(.125,0);q.p(.25,2);q.cx(0,1)
    calls=[]
    def call(kind,label,fn):calls.append(kind);return fn()
    result=m.compiler.compile_once(q,CompilerProfile(),call=SharedQiskitCalls(call))
    assert calls==['compile'];m.compiler.check_circuit(result,qubits=3,native=True)
    from trottertracks import c1_h4_native_fair as native
    def fail(*args):raise ValueError('invented compiler failure')
    monkeypatch.setattr(native,'compile_checked',fail);calls.clear()
    with pytest.raises(ValueError,match='invented compiler failure'):
        m.compiler.compile_once(q,CompilerProfile(),call=SharedQiskitCalls(call))
    assert calls==['compile']


def test_existing_tail_variance_and_direct_adapters_on_invented_intervals():
    from trottertracks.c_iid_grid_bridge import MODEL
    n=1000
    policy=dict(model=MODEL,PCG64_joint_stream_qualified=False,origin='fresh_IID_grid_calibration',
        pool_old_samples=False,adaptive=False,fixed_samples=n,reuse='calibration_only_not_measurement_shots',
        iid_assumed=True,sample_ids=list(range(n)))
    rows=[dict(kind='rigorous_enclosure',lower_exact='0',upper_exact='1/1000',provenance='invented certified interval') for _ in range(n)]
    support=dict(qualification='deterministic_population_bound',upper_exact='3',provenance='invented bound,not sample maximum')
    tail=m.population.threshold_tail(rows,support=support,threshold='1/1000',delta='1/20',policy=policy)
    var=m.population.variance_sensitive(rows,support=support,delta='1/20',policy=policy)
    assert tail['good']==n and F(tail['tail_probability_upper'])>0
    assert F(tail['bound'])>F('1/1000') and F(var['bound'])>F('1/1000')
    assert var['numerical_interval_uncertainty_included'] and not var['all_word_uniform']
    rows[0]['upper_exact']='1/100'
    straddle=m.population.threshold_tail(rows,support=support,threshold='1/1000',delta='1/20',policy=policy)
    assert straddle['straddle']==1 and straddle['good']==n-1
    rows[0]['upper_exact']=None
    assert m.population.variance_sensitive(rows,support=support,delta='1/20',policy=policy)['bound'] is None
    signed=[dict(kind='rigorous_enclosure',provenance='invented paired interval',X=['-1/100','1/100'],Y=['-1/100','1/100']) for _ in range(n)]
    direct=m.expected.paired_round_upper(signed,axis_support=dict(qualification='deterministic_population_bound',lower_exact='-3',upper_exact='3',provenance='invented signed support'),delta_round='1/20',policy=policy)
    assert F(direct['bound'])>0 and not direct['all_word_uniform']


def test_passive_contracts_no_molecular_import_or_scientific_execution():
    code='''
import sys
from pathlib import Path
class Block:
 def find_spec(self,name,*args):
  if name.split('.')[0] in ('numpy','scipy','flint','qiskit','ffsim','pyscf','openfermion'):
   raise AssertionError('NO_MOLECULAR_IMPORT:'+name)
sys.meta_path.insert(0,Block())
from trottertracks import c_totality_calibration as m
from trottertracks import c_batch_pair_pilot as p
a=m.preflight(Path.cwd(),m.CONTRACT);b=p.preflight(Path.cwd(),p.CONTRACT)
assert a['scientific_calls']==b['scientific_calls']==0 and a['maximum_logical_units'] is None
assert b['maximum_logical_units']==7 and not a['calibration_ready']
'''
    subprocess.run([sys.executable,'-B','-c',code],cwd=Path(__file__).resolve().parents[2],check=True)


@pytest.mark.parametrize('target,fault',[('design','N'),('design','delta'),('design','executable'),('design','input'),('pilot','N'),('pilot','word'),('pilot','model'),('pilot','precision'),('pilot','retry')])
def test_unadopted_or_wrong_contract_is_rejected_without_decoding(tmp_path,target,fault):
    from trottertracks import c_batch_pair_pilot as pilot
    root=Path(__file__).resolve().parents[2];module=m if target=='design' else pilot
    c=json.loads((root/module.CONTRACT).read_text())
    if fault=='N':c['samples']=10
    if fault=='delta':c['delta_cal']='.05'
    if fault=='executable':c['executable']=True
    if fault=='input':c['saved_inputs'][next(iter(c['saved_inputs']))]='0'*64
    if fault=='word':c['word']=['CDF_proposed:F6:0']*32
    if fault=='model':c['identity']['model']['independent_events']=1
    if fault=='precision':c['precision_bits']=160
    if fault=='retry':c['retry']=1
    p=tmp_path/'contract.json';p.write_text(json.dumps(c))
    with pytest.raises(ValueError):module.preflight(root,p)
