"""Invented integer streams, scalar literal blocks, circuits and failures only."""
from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path
import subprocess
import sys

import pytest
from trottertracks import c_iid_calibration_integration as m
from trottertracks import c_fresh_iid_calibration as fresh
from trottertracks import c_pr_aware_df_fixed_certificate as fixed
from trottertracks import c_iid_grid_expected_bias as expected
from trottertracks.c_b1_b2_scope_integration import identity
from trottertracks.c_pr_aware_df_stage_c2 import choice_grid

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def context():
    return fresh.make_context(source_id='3'*64, identity=identity(), atom_ids=['invented:atom'],
        table=choice_grid([1.]), event_angle_bits={'invented:atom': 0.0.hex()})


def sample(c, i=0, m=None):
    return fresh.sample_from_integers(c, dataset_id='integration-INVENTED', sample_index=i,
        integers=[[0]*64 for _ in range(1 if m is None else 1 << m)],
        unit='kernel' if m is None else 'round', m=m, axis='kernel' if m is None else 'paired_XY')


def toy(c, s, n=3, scale=1):
    """Scalar blocks describe diag(I,scale*I), without loading a molecule."""
    from flint import acb_mat
    from trottertracks.c_pr_aware_df_fixed_car import pack
    target = dict(target=m.population.TARGET, candidate=c['identity']['candidate'],
        invented_system_dimension=1 << (n-1), literal_scale=scale)
    def raw(k, i):
        return m.bind_operator(fixed.identity(n), s, c, i, role='raw')
    def compile_raw(r, i):
        return m.bind_operator(r['record'], s, c, i, role='compiled', raw_id=r['record_id'])
    def reference(k, i):
        return dict(status='VALIDATED_LITERAL_REFERENCE_BLOCKS', target=m.population.TARGET,
            binding=dict(target=target, target_id=fresh.digest(target), word=k['word'], word_id=k['word_id']),
            system_dimension=1 << (n-1), precision_bits=192,
            blocks=[dict(indices=[j], entries=pack(acb_mat([[scale]]))) for j in range(1 << (n-1))])
    return dict(build_raw=raw, compile_raw=compile_raw, make_reference=reference, target=target, parallel=False)


def run(tmp_path, c, s=None, **changes):
    s = sample(c) if s is None else s
    args = toy(c, s); args.update(changes)
    path = fresh.open_batch(tmp_path/'fresh')
    return m.process_sample(path, s, c, **args), path, args


def population(c, target):
    return dict(status='VALIDATED_LITERAL_POPULATION_NORM', D_norm_upper_exact='1',
        d_range_upper_exact='2', accepted_P0_C_norm_exact='1', no_sample_maximum_used=True,
        target_id=fresh.digest(target), event_angle_bits=c['law']['event_angle_bits'])


def payload(rows, c, target, **overrides):
    args = dict(context=c, target=target, planned_ids=[r['sample']['sample_id'] for r in rows],
        method='threshold_tail', delta_cal='1/20', failure_policy='record_all_fixed_N',
        iid_assumed=True, population=population(c, target),
        compiler_domain=dict(kind='structural_totality_argument', scope='all_legal_q_words',
            gates='exact_ideal_P0_unitaries', provenance='INVENTED premise, not a scientific proof'))
    args.update(overrides)
    return m.calibration_payload(rows, **args)


@pytest.fixture(scope='module')
def full_rows(tmp_path_factory):
    c = fresh.make_context(source_id='3'*64, identity=identity(), atom_ids=['invented:atom'],
        table=choice_grid([1.]), event_angle_bits={'invented:atom': 0.0.hex()})
    path = fresh.open_batch(tmp_path_factory.mktemp('full512')/'batch')
    rows = []
    for i in range(2):
        s = sample(c, i); args = toy(c, s, n=9)
        rows.append(m.process_sample(path, s, c, **args))
    fresh.finish_batch(path)
    assert all(r['status'] == 'SUCCESS' for r in rows)
    return c, rows, args['target']


@pytest.mark.parametrize('method', ['threshold_tail', 'variance_sensitive'])
def test_full512_fresh_to_fast_certificate_to_old_statistics(full_rows, method):
    c, rows, target = full_rows
    p = payload(rows, c, target, method=method)
    assert p['status'] == 'CONDITIONAL_MATHEMATICAL_IID_INPUTS'
    cert = rows[0]['evidence']['per_word_enclosures'][0]
    assert cert['compiled_raw']['columns_evaluated'] == cert['raw_literal']['columns_evaluated'] == 512
    result = fresh.evaluate_calibration(p, threshold='1/1000')
    assert result['bound'] is not None
    assert result['confidence'] == expected.failure_accounting('1/20')
    assert result['Stage_C_ready'] is result['PCG64_joint_stream_qualified'] is False
    assert rows[0]['sample']['kernels'][0]['word_id'] == rows[1]['sample']['kernels'][0]['word_id']
    assert rows[0]['sample']['sample_id'] != rows[1]['sample']['sample_id']


@pytest.mark.parametrize('fault', ['source_id', 'candidate', 'law_id', 'word', 'angle', 'phase', 'round', 'axis', 'occurrence'])
def test_binding_faults_are_preserved_and_stop_before_compile(tmp_path, context, fault):
    s = sample(context); args = toy(context, s); original = args['build_raw']; called = []
    def bad(k, i):
        r = original(k, i)
        if fault in ('source_id', 'candidate', 'law_id'):
            r['calibration_binding'][fault] = 'wrong'
        if fault == 'word': r['word'] = list(reversed(r['word']))[:-1]
        if fault == 'angle': r['word'][0][1] = .125.hex()
        if fault == 'phase': r['record']['phase_hex'] = .125.hex()
        if fault == 'round': r['m'] = 1
        if fault == 'axis': r['axis'] = 'Y'
        if fault == 'occurrence': r['calibration_binding']['occurrence'] = 1
        return r
    args.update(build_raw=bad, compile_raw=lambda *a: called.append(a))
    row, path, _ = run(tmp_path, context, **args)
    assert row['status'] == 'RAW_BUILD_FAILED' and called == []
    assert row['evidence']['occurrences'][0]['raw'] is not None
    assert fresh.read_batch(path) == [row] and (path/'STOP.json').exists()


@pytest.mark.parametrize('where,status', [('build_raw','RAW_BUILD_FAILED'), ('compile_raw','P0_COMPILE_FAILED'),
    ('make_reference','PER_WORD_CERTIFICATE_FAILED'), ('certifier','PER_WORD_CERTIFICATE_FAILED'),
    ('numeric','NUMERICAL_UNEVALUABLE'), ('missing_compile','INCOMPLETE'),
    ('missing_cert','PER_WORD_CERTIFICATE_FAILED'), ('missing_reference','INCOMPLETE'),
    ('native','NATIVE_INELIGIBLE')])
def test_failure_states_partial_evidence_and_no_retry(tmp_path, context, where, status):
    s = sample(context); args = toy(context, s); calls = []
    def fail(*a, **kw):
        calls.append(where)
        if where == 'numeric': raise ArithmeticError('INVENTED numerical failure')
        raise RuntimeError('INVENTED callback failure')
    if where == 'numeric': args['certifier'] = fail
    elif where == 'missing_compile': args['compile_raw'] = lambda *a: None
    elif where == 'missing_reference': args['make_reference'] = lambda *a: None
    elif where == 'missing_cert': args['certifier'] = lambda *a, **kw: None
    elif where == 'native':
        def ineligible(r, i):
            v = deepcopy(r['record']); v['gates'] = [dict(name='h', qubits=[0], params_hex=[])]
            return m.bind_operator(v, s, context, i, role='compiled', raw_id=r['record_id'])
        args['compile_raw'] = ineligible
    else: args[where] = fail
    row, path, _ = run(tmp_path, context, **args)
    assert row['status'] == status and len(calls) <= 1
    assert fresh.read_batch(path) == [row]
    assert json.loads((path/'STOP.json').read_text())['retry'] == 0
    with pytest.raises(ValueError, match='STOP'):
        m.process_sample(path, sample(context, 1), context, **toy(context, sample(context, 1)))
    result = payload([row], context, args['target'])
    assert result['status'] == 'NOT_READY' and fresh.evaluate_calibration(result)['bound'] is None


def test_interruption_retains_incomplete_and_closes_batch(tmp_path, context):
    path = fresh.open_batch(tmp_path/'batch'); s = sample(context)
    def interrupted(*a): raise KeyboardInterrupt('INVENTED interruption')
    with pytest.raises(KeyboardInterrupt):
        m.process_sample(path, s, context, **dict(toy(context,s), compile_raw=interrupted))
    assert fresh.read_batch(path)[0]['status'] == 'INCOMPLETE'
    assert (path/'STOP.json').exists()


def test_actual_p0_adapter_on_artificial_circuit_once(tmp_path, context):
    from qiskit import QuantumCircuit
    from trottertracks.c_p0_compiler_totality import compile_once
    from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
    q = QuantumCircuit(3); q.p(.125, 2); q.cx(0, 1)
    s = sample(context); args = toy(context, s); calls = []
    def call(kind, label, fn): calls.append(kind); return fn()
    def build(k, i): return m.bind_operator(fixed.record(q), s, context, i, role='raw')
    def compile_raw(r, i):
        v = compile_once(q, CompilerProfile(), call=call)
        return m.bind_operator(fixed.record(v), s, context, i, role='compiled', raw_id=r['record_id'])
    row, _, _ = run(tmp_path, context, **dict(args, build_raw=build, compile_raw=compile_raw))
    assert calls == ['compile'] and row['status'] == 'SUCCESS'
    assert F(row['evidence']['per_word_enclosures'][0]['compiled_raw']['upper_exact']) < F(1,10**12)


@pytest.mark.parametrize('fault', ['target', 'word', 'angle', 'precision'])
def test_literal_target_mismatch_stops_without_certificate(tmp_path, context, fault):
    s = sample(context); args = toy(context,s); original = args['make_reference']
    def wrong(k, i):
        r = original(k,i)
        if fault == 'target': r['binding']['target']['candidate'] = 'wrong'
        if fault == 'word': r['binding']['word_id'] = 'wrong'
        if fault == 'angle': r['binding']['word'][0][1] = .125.hex()
        if fault == 'precision': r['precision_bits'] = 160
        return r
    row, _, _ = run(tmp_path,context,**dict(args,make_reference=wrong))
    assert row['status'] == 'PER_WORD_CERTIFICATE_FAILED'


def test_nonunitary_literal_never_uses_native_reduction_or_cap2(tmp_path,context,monkeypatch):
    from trottertracks import c_arb_column_batch as fast
    original = fast.compare; pairs = []
    def native_only(a,b,**kw):
        assert 'gates' in a and 'gates' in b
        pairs.append((a,b)); return original(a,b,**kw)
    monkeypatch.setattr(fast,'compare',native_only)
    s = sample(context); row, _, _ = run(tmp_path,context,**toy(context,s,scale=4))
    cert = row['evidence']['per_word_enclosures'][0]
    assert row['status'] == 'SUCCESS' and len(pairs) == 1
    assert F(cert['raw_literal']['lower_exact']) >= 3 and F(cert['upper_exact']) >= 3
    with pytest.raises((ValueError,KeyError)):
        original(fixed.identity(3),dict(target='nonunitary literal',matrix=[[4]]),parallel=False)


def test_second_leg_failure_keeps_completed_first_leg(tmp_path,context,monkeypatch):
    def numerical_failure(*a,**kw): raise ArithmeticError('INVENTED literal evaluation failure')
    monkeypatch.setattr(m.population,'compare_literal',numerical_failure)
    row,path,_=run(tmp_path,context)
    assert row['status']=='NUMERICAL_UNEVALUABLE'
    part=row['evidence']['occurrences'][0]
    assert part['compiled_raw']['status']=='VALIDATED_BATCHED_NATIVE_PAIR'
    assert part['reference']['binding']['word']==row['sample']['kernels'][0]['word']
    assert row['evidence']['per_word_enclosures']==[] and (path/'STOP.json').exists()


@pytest.mark.parametrize('fault',['parent','phase','source'])
def test_compiled_record_binding_cannot_change_input_or_phase(tmp_path,context,fault):
    s=sample(context);args=toy(context,s);original=args['compile_raw']
    def bad(r,i):
        v=original(r,i)
        if fault=='parent':v['raw_id']='0'*64
        if fault=='phase':v['record']['phase_hex']=.125.hex()
        if fault=='source':v['calibration_binding']['source_id']='0'*64
        return v
    row,_,_=run(tmp_path,context,**dict(args,compile_raw=bad))
    assert row['status']=='NATIVE_INELIGIBLE'


@pytest.mark.parametrize('fault', ['sample', 'phase', 'missing_leg', 'precision', 'coverage', 'triangle', 'qualifier'])
def test_persisted_certificate_cannot_be_reused_or_unqualified(full_rows,fault):
    c, original, target = full_rows; rows = deepcopy(original)
    cert = rows[0]['evidence']['per_word_enclosures'][0]
    if fault == 'sample': rows[0]['evidence']['per_word_enclosures'][0] = deepcopy(rows[1]['evidence']['per_word_enclosures'][0])
    if fault == 'phase': rows[0]['evidence']['occurrences'][0]['compiled']['record']['phase_hex'] = .125.hex()
    if fault == 'missing_leg': del cert['raw_literal']
    if fault == 'precision': cert['compiled_raw']['precision_bits'] = 160
    if fault == 'coverage': cert['compiled_raw']['columns'].pop()
    if fault == 'triangle': cert['upper_exact'] = '1'
    if fault == 'qualifier': cert['rounding_enclosed'] = False
    with pytest.raises((ValueError,KeyError)): payload(rows,c,target)


def test_duplicate_sample_missing_cert_and_failed_subset_are_rejected(full_rows):
    c, rows, target = full_rows
    with pytest.raises(ValueError,match='DUPLICATE'): payload([rows[0],rows[0]],c,target)
    planned = [r['sample']['sample_id'] for r in rows]
    assert payload(rows[:1],c,target,planned_ids=planned)['status'] == 'NOT_READY'
    missing = deepcopy(rows); missing[0]['evidence']['per_word_enclosures'] = []
    assert payload(missing,c,target)['samples'] is None
    old = deepcopy(rows); old[0]['sample']['origin'] = 'Stage_P_saved_fixed_word'
    with pytest.raises(ValueError,match='NO_FIXED'): payload(old,c,target)


@pytest.mark.parametrize('key,value', [('delta_cal',None),('failure_policy',None),('method',None),
    ('iid_assumed',False),('compiler_domain',None)])
def test_unadopted_scientific_premises_remain_not_ready(full_rows,key,value):
    c, rows, target = full_rows
    p = payload(rows,c,target,**{key:value})
    assert p['status'] == 'NOT_READY' and fresh.evaluate_calibration(p)['bound'] is None
    assert not p['Stage_C_ready']


def test_pcg64_replay_is_not_iid_qualification(context):
    s = fresh.sample_from_pcg64(context,dataset_id='INVENTED replay',sample_index=0,master=17)
    assert fresh.validate_sample(s,context) and not s['rng']['PCG64_joint_stream_qualified']
    s['rng']['IID_guarantee'] = True
    with pytest.raises(ValueError,match='IID'): fresh.validate_sample(s,context)


def test_round_occurrences_and_paired_xy_connect_to_direct_statistics(tmp_path,context):
    path = fresh.open_batch(tmp_path/'rounds'); rows=[]
    def round_eval(s,e):
        return dict(kind='rigorous_enclosure',scope='paired_round_raw_complex_signal',m=s['m'],
            round_word_id=s['round_word_id'],sample_id=s['sample_id'],source_id=s['source_id'],law_id=s['law_id'],
            target_id=fresh.digest(target),phase_fit=False,rounding_enclosed=True,shells_included=True,
            included_error_terms=[expected.TERMS[0]],X=['0','0'],Y=['0','0'],
            raw_signals={k:dict(real='1',imag='0') for k in ('compiled','literal_atom')},
            provenance='INVENTED paired full-round enclosure, not a real signal evaluator')
    for i in range(2):
        s=sample(context,i,m=1); args=toy(context,s,n=9); target=args['target']
        rows.append(m.process_sample(path,s,context,**args,round_evaluator=round_eval))
    fresh.finish_batch(path)
    support=dict(qualification='deterministic_population_bound',lower_exact='-2',upper_exact='2',
        m=1,target_id=fresh.digest(target),provenance='INVENTED deterministic round support')
    p=payload(rows,context,target,method='direct_round',axis_support=support)
    assert p['status']=='CONDITIONAL_MATHEMATICAL_IID_INPUTS'
    assert fresh.evaluate_calibration(p)['bound'] is not None
    assert [x['occurrence'] for x in rows[0]['evidence']['occurrences']]==[0,1]
    bad=deepcopy(rows);bad[0]['sample']['axis']='Y'
    with pytest.raises(ValueError):payload(bad,context,target,method='direct_round',axis_support=support)
    bad=deepcopy(rows);bad[0]['evidence']['round_enclosure']['shells_included']=False
    assert payload(bad,context,target,method='direct_round',axis_support=support)['status']=='NOT_READY'


def test_compiler_failure_mapping_retains_fixed_plan_and_separates_certificate_failure(tmp_path,context):
    good, _, args = run(tmp_path,context)
    certfail=deepcopy(good);certfail['status']='PER_WORD_CERTIFICATE_FAILED'
    certfail['evidence']['per_word_enclosures']=[]
    kw=dict(context=context,planned_ids=[good['sample']['sample_id']],delta_compile='1/20',
        failure_policy='record_all_fixed_N',iid_assumed=True)
    a=m.compiler_failure_summary([certfail],**kw)
    assert a['failures']==0 and a['denominator']==1 and F(a['failure_upper'])>0
    assert not a['all_q_words_compile_success_qualified']
    b=deepcopy(good);part=b['evidence']['occurrences'][0];part['compile_status']='FAILED'
    assert m.compiler_failure_summary([b],**kw)['failures']==1
    assert m.compiler_failure_summary([b],**dict(kw,failure_policy='stop_on_failure'))['failure_upper'] is None
    part['raw_status']='FAILED';part['compile_status']='UNKNOWN'
    assert m.compiler_failure_summary([b],**kw)['failures']==1
    assert m.compiler_failure_summary([b],**dict(kw,event='compile_given_raw_success'))['failure_upper'] is None
    part['raw_status']='UNKNOWN'
    assert m.compiler_failure_summary([b],**kw)['failure_upper'] is None
    for key in ('delta_compile','failure_policy'):
        assert m.compiler_failure_summary([good],**dict(kw,**{key:None}))['failure_upper'] is None
    with pytest.raises(ValueError,match='DISCARD'):
        m.compiler_failure_summary([good],**dict(kw,planned_ids=['missing',good['sample']['sample_id']]))
    assert m.compiler_failure_summary([good],**dict(kw,planned_ids=[good['sample']['sample_id'],'missing']))['failure_upper'] is None


def test_unqualified_stage_C_remains_rejected(full_rows):
    c,rows,target=full_rows
    p=payload(rows,c,target)
    result=expected.assess_if_adopted(None,None,dict(eligible=False),policy_adopted=False)
    assert not p['Stage_C_ready'] and result['status']=='NOT_READY_NO_ACCURACY_CALLS'


def test_passive_preflight_and_cli_never_import_numerical_libraries():
    code='''
import sys
from pathlib import Path
class Block:
 def find_spec(self,name,*args):
  if name.split('.')[0] in ('numpy','scipy','flint','qiskit','ffsim','pyscf','openfermion'):
   raise AssertionError('NO_SCIENTIFIC_IMPORT:'+name)
sys.meta_path.insert(0,Block())
from trottertracks import c_iid_calibration_integration as m
r=m.preflight(Path.cwd())
assert r['scientific_calls']==0 and not r['execution_ready']
assert not r['fixed_7_call_pilot_executed'] and r['mandatory_stop']
'''
    subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,check=True)
    result=subprocess.run([sys.executable,'-B','scripts/prepare_research_c_iid_calibration_integration.py'],
        cwd=ROOT,check=True,capture_output=True,text=True)
    assert json.loads(result.stdout)['scientific_calls']==0


@pytest.mark.parametrize('fault',['N','executable','retry','source','contract','base'])
def test_connection_contract_and_independent_source_are_fixed(tmp_path,fault):
    c=json.loads((ROOT/m.CONTRACT).read_text())
    if fault=='N':c['N']=2
    if fault=='executable':c['executable']=True
    if fault=='retry':c['retry']=True
    if fault=='source':c['source_manifest_sha256']='0'*64
    if fault=='contract':c['readonly_contracts'][fresh.CONTRACT]='0'*64
    if fault=='base':c['base_commit']=m.A
    p=tmp_path/'contract.json';p.write_text(json.dumps(c))
    with pytest.raises(ValueError):m.preflight(ROOT,p)
