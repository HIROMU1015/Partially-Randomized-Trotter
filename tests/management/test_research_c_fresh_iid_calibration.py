"""Synthetic bit streams, scalar enclosures and mock failures only; no H4."""
from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from trottertracks import c_fresh_iid_calibration as a
from trottertracks import c_iid_grid_expected_bias as expected
from trottertracks import c_per_word_population_range as population_adapter
from trottertracks.c_b1_b2_scope_integration import identity
from trottertracks.c_pr_aware_df_stage_c2 import choice_grid

ROOT = Path(__file__).resolve().parents[2]


def context():
    return a.make_context(source_id='1'*64, identity=identity(), atom_ids=['invented:A', 'invented:B', 'invented:C'],
        table=choice_grid([.25, .25, .5]), event_angle_bits={k: .1.hex() for k in ('invented:A', 'invented:B', 'invented:C')})


def sample(index=0, *, c=None, m=None):
    return a.sample_from_integers(c or context(), dataset_id='synthetic-only', sample_index=index,
        integers=[[0]*64 for _ in range(1 if m is None else 1 << m)],
        unit='kernel' if m is None else 'round', m=m, axis='kernel' if m is None else 'paired_XY')


def compiler():
    return dict(kind='structural_totality_argument', scope='all_legal_q_words',
        gates='exact_ideal_P0_unitaries', provenance='INVENTED premise; never a real totality proof')


def population(c):
    return dict(status='VALIDATED_LITERAL_POPULATION_NORM', D_norm_upper_exact='1', d_range_upper_exact='2',
        accepted_P0_C_norm_exact='1', no_sample_maximum_used=True, target_id='INVENTED target',
        event_angle_bits=c['law']['event_angle_bits'])


def success(s):
    certs = []
    for k in s['kernels']:
        certs.append(dict(kind='rigorous_enclosure', scope='per_word_full512',
            status='VALIDATED_PER_WORD_COMPILED_LITERAL_OPERATOR', target=population_adapter.TARGET,
            lower_exact='0', upper_exact='1/1000000',
            binding=dict(target_id='INVENTED target', word=k['word'], word_id=k['word_id']),
            calibration_binding=dict(sample_id=s['sample_id'], source_id=s['source_id'], candidate=s['candidate'],
                law_id=s['law_id'], occurrence=k['coherent_occurrence'], word_id=k['word_id']),
            phase_fit=False, rounding_enclosed=True, provenance='INVENTED two-leg per-word enclosure'))
    e = dict(per_word_enclosures=certs, partial_raw='INVENTED raw artifact')
    if s['unit'] == 'round':
        e['round_enclosure'] = dict(kind='rigorous_enclosure', scope='paired_round_raw_complex_signal',
            m=s['m'], round_word_id=s['round_word_id'], sample_id=s['sample_id'], source_id=s['source_id'],
            law_id=s['law_id'], target_id='INVENTED target', phase_fit=False, rounding_enclosed=True,
            shells_included=True, included_error_terms=[expected.TERMS[0]],
            X=['-1/1000', '1/1000'], Y=['-1/1000', '1/1000'],
            raw_signals={k: dict(real='1', imag='0') for k in ('compiled', 'literal_atom')},
            provenance='INVENTED all-occurrence phase-sensitive paired enclosure')
    return a.outcome(s, status='SUCCESS', evidence=e)


def payload(rows=None, *, method='threshold_tail', **kwargs):
    c = context(); rows = rows if rows is not None else [success(sample(i, c=c)) for i in range(2)]
    args = dict(context=c, planned_ids=[r['sample']['sample_id'] for r in rows], method=method,
        delta_cal='1/20', failure_policy='record_all_fixed_N', iid_assumed=True,
        population=population(c), compiler_domain=compiler())
    args.update(kwargs)
    return a.calibration_payload(rows, **args)


def test_exact_boundary_order_and_atom_consumption_including_zero_bins():
    c = context(); t = choice_grid([0., .5, .5])
    c = a.make_context(source_id=c['source_id'], identity=c['identity'], atom_ids=c['law']['atom_ids'],
        table=t, event_angle_bits=c['law']['event_angle_bits'])
    boundary = 1 << 52
    atoms = [0, 1, boundary-1, boundary, boundary+1, (1 << 53)-1]
    bits = [x for i in range(32) for x in ((1 << 53)-1, atoms[i % len(atoms)])]
    s = a.sample_from_integers(c, dataset_id='boundaries', sample_index=0, integers=[bits])
    for i, e in enumerate(s['kernels'][0]['events']):
        assert e['event_index'] == i and e['order_index'] == 0
        assert e['atom_index'] == np.searchsorted(t['cdf'], atoms[i % len(atoms)]/(1 << 53), side='right')
        assert e['order_draw'] == dict(offset=2*i, uint53=(1 << 53)-1)
        assert e['atom_draw']['offset'] == 2*i+1
    assert a.validate_sample(s, c)


@pytest.mark.parametrize('bad', [-1, 1 << 53, True, 1.0])
@pytest.mark.parametrize('offset', [0, 1])
def test_rejects_bad_order_or_atom_uint53(bad, offset):
    bits = [0]*64; bits[offset] = bad
    with pytest.raises(ValueError, match='53_BIT'):
        a.sample_from_integers(context(), dataset_id='bad', sample_index=0, integers=[bits])


@pytest.mark.parametrize('fault', ['cdf', 'counts', 'denominator', 'bits', 'mapping', 'atom_order', 'angle', 'candidate', 'SOURCE', 'model'])
def test_context_rejects_wrong_fixed_identity_and_grid_law(fault):
    c = context(); kw = dict(source_id=c['source_id'], identity=c['identity'], atom_ids=c['law']['atom_ids'],
        table=c['law']['table'], event_angle_bits=c['law']['event_angle_bits'])
    if fault == 'cdf': kw['table']['cdf'] = [.8, .5, 1.]
    if fault == 'counts': kw['table']['counts'][0] += 1
    if fault == 'denominator': kw['table']['denominator'] = 1 << 52
    if fault == 'bits': kw['table']['bits'] = 52
    if fault == 'mapping': kw['table']['mapping'] = 'searchsorted_left'
    if fault == 'atom_order': kw['atom_ids'][0] = kw['atom_ids'][1]
    if fault == 'angle': kw['event_angle_bits']['invented:A'] = 'nan'
    if fault == 'candidate': kw['identity']['candidate'] = 'wrong'
    if fault == 'SOURCE': kw['identity']['PR_scientific_SOURCE'] = 'wrong'
    if fault == 'model': kw['identity']['model']['id'] = 'PCG64_is_IID'
    with pytest.raises(ValueError): a.make_context(**kw)


def test_sample_source_replay_and_occurrence_identity_without_word_deduplication():
    c = context(); s = sample(c=c, m=2)
    assert s == sample(c=c, m=2) and a.validate_sample(json.loads(json.dumps(s)), c)
    assert len(s['kernels']) == 4
    assert len({k['word_id'] for k in s['kernels']}) == 1  # IID may repeat a word
    assert len({e['event_id'] for k in s['kernels'] for e in k['events']}) == 128
    assert s['axis'] == 'paired_XY' and [k['coherent_occurrence'] for k in s['kernels']] == list(range(4))
    other = dict(c); other['source_id'] = '2'*64
    other = a.make_context(source_id=other['source_id'], identity=c['identity'], atom_ids=c['law']['atom_ids'],
        table=c['law']['table'], event_angle_bits=c['law']['event_angle_bits'])
    assert sample(c=other, m=2)['sample_id'] != s['sample_id']
    with pytest.raises(ValueError): a.validate_sample(s, other)


@pytest.mark.parametrize('fault', ['origin', 'source_id', 'candidate', 'law_id', 'event_order', 'IID', 'PCG64'])
def test_wrong_sample_or_old_fixed_pool_is_rejected(fault):
    c = context(); s = sample(c=c)
    if fault == 'origin': s['origin'] = 'old_fixed4_or_native32'
    elif fault == 'event_order': s['kernels'][0]['events'].reverse()
    elif fault == 'IID': s['rng']['IID_guarantee'] = True
    elif fault == 'PCG64': s['rng']['PCG64_joint_stream_qualified'] = True
    else: s[fault] = 'wrong'
    with pytest.raises(ValueError): a.validate_sample(s, c)


@pytest.mark.parametrize('m', [None, 1])
def test_pcg64_matches_existing_independent_occurrence_including_order_consumption(m):
    from trotterlib.rte import RTEComponent, finite_rte_distribution, _make_event
    from trottertracks import c_pr_aware_df_stage_b as b
    components = tuple(RTEComponent(k, p, p, 1) for k, p in zip(('invented:A', 'invented:B', 'invented:C'), (.25, .25, .5)))
    dist = finite_rte_distribution(.1, 0)
    angles = {c.component_id: float(_make_event((i,), components, dist, 0).rotation_angle).hex() for i, c in enumerate(components)}
    c = a.make_context(source_id='1'*64, identity=identity(), atom_ids=list(angles), table=choice_grid([.25,.25,.5]), event_angle_bits=angles)
    args = dict(dataset_id='synthetic-pcg64', sample_index=3, master=17,
        unit='kernel' if m is None else 'round', m=m, axis='kernel' if m is None else 'paired_XY')
    s = a.sample_from_pcg64(c, **args)
    assert s == a.sample_from_pcg64(c, **args)
    assert a.validate_sample(s, c) and s['rng']['PCG64_joint_stream_qualified'] is False
    tail = NS(components=components); arm = NS(role='proposed', model=NS(source_identity=c['identity']['candidate']))
    events = a.rte_events(s, c, components=components, distribution=dist)
    for i, occurrence in enumerate(events):
        old = b.independent_occurrence(arm, tail, dist, NS(rte_steps=32), master=tuple(s['rng']['domain_master']),
            round_index=-1 if m is None else m, axis=args['axis'], shot=3, occurrence=i)
        assert occurrence == old
        rng = np.random.Generator(np.random.PCG64(s['rng']['streams'][i]['seed']))
        for e in s['kernels'][i]['events']:
            assert int(rng.random()*(1 << 53)) == e['order_draw']['uint53']
            assert int(rng.random()*(1 << 53)) == e['atom_draw']['uint53']
        assert rng.bit_generator.state == s['rng']['streams'][i]['after']
    bad = deepcopy(s); bad['rng']['streams'][0]['raw64_consumed'] = 32
    with pytest.raises(ValueError): a.validate_sample(bad, c)
    with pytest.raises(ValueError): a.rte_events(s, c, components=components, distribution=finite_rte_distribution(.1, 2))


@pytest.mark.parametrize('status', list(a.FAILURE_STAGES)+['INCOMPLETE'])
def test_append_only_failed_word_partial_evidence_and_mandatory_stop(tmp_path, status):
    c = context(); s = sample(c=c); path = a.open_batch(tmp_path/'fresh')
    a.save_sample(path, s, c)
    assert a.read_batch(path)[0]['status'] == 'INCOMPLETE'
    with pytest.raises(FileExistsError): a.save_sample(path, s, c)  # no replacement extraction
    r = a.outcome(s, status=status, stage=a.FAILURE_STAGES.get(status), error=RuntimeError('invented failure'),
        evidence=dict(raw_record='invented partial raw', native=None))
    a.save_outcome(path, r)
    assert a.read_batch(path) == [r] and a.read_batch(path)[0]['sample']['kernels'] == s['kernels']
    with pytest.raises(FileExistsError): a.save_outcome(path, success(s))
    stop = a.finish_batch(path)
    assert stop['mandatory_stop'] and stop['outcomes'][status] == 1 and stop['failure_policy_adopted'] is None
    with pytest.raises(ValueError, match='STOP'): a.save_sample(path, sample(1, c=c), c)
    with pytest.raises(ValueError, match='STOP'): a.finish_batch(path)
    with pytest.raises(FileExistsError): a.open_batch(path)


def test_duplicate_sample_detection_allows_identical_words_from_distinct_draws():
    s0, s1 = sample(0), sample(1)
    assert s0['kernels'][0]['word_id'] == s1['kernels'][0]['word_id']
    assert payload([success(s0), success(s1)])['status'] == 'CONDITIONAL_MATHEMATICAL_IID_INPUTS'
    with pytest.raises(ValueError, match='DUPLICATE'): payload([success(s0), success(s0)])


@pytest.mark.parametrize('method', ['threshold_tail', 'variance_sensitive'])
def test_existing_kernel_calibration_connection_and_confidence_are_conditional(method):
    p = payload(method=method)
    assert p['status'] == 'CONDITIONAL_MATHEMATICAL_IID_INPUTS'
    result = a.evaluate_calibration(p, threshold='1/1000')
    old = getattr(population_adapter, method)
    args = dict(support=p['support'], delta=p['delta_cal'], policy=p['policy'])
    if method == 'threshold_tail': args['threshold'] = '1/1000'
    assert result['bound'] == old(p['samples'], **args)['bound']
    assert result['confidence']['unconditional_failure_upper'] == '3/50'
    assert result['PCG64_joint_stream_qualified'] is result['Stage_C_ready'] is False


def test_direct_round_connects_paired_enclosure_and_preserves_raw_complex_signals():
    rows = [success(sample(i, m=1)) for i in range(2)]
    support = dict(qualification='deterministic_population_bound', lower_exact='-2', upper_exact='2',
        m=1, target_id='INVENTED target', provenance='INVENTED full round population bound')
    p = payload(rows, method='direct_round', axis_support=support)
    result = a.evaluate_calibration(p)
    assert result['bound'] == expected.paired_round_upper(p['samples'], axis_support=support,
        delta_round=p['delta_cal'], policy=p['policy'])['bound']
    assert p['records'][0]['evidence']['round_enclosure']['raw_signals']['compiled'] == dict(real='1', imag='0')
    rows[0]['evidence']['round_enclosure']['shells_included'] = False
    q = payload(rows, method='direct_round', axis_support=support)
    assert q['status'] == 'NOT_READY' and a.evaluate_calibration(q)['bound'] is None
    with pytest.raises(ValueError, match='UNIT'): payload(rows)


@pytest.mark.parametrize('fault', ['N', 'method', 'delta', 'IID', 'failure_policy', 'totality', 'population',
    'missing_enclosure', 'unqualified_enclosure', 'sector_enclosure', 'unfinished', 'failed', 'empty'])
def test_not_ready_never_filters_or_qualifies_missing_information(fault):
    rows = [success(sample(i)) for i in range(2)]; kw = {}
    if fault == 'N': kw['planned_ids'] = [r['sample']['sample_id'] for r in rows]+['missing']
    if fault == 'method': kw['method'] = None
    if fault == 'delta': kw['delta_cal'] = None
    if fault == 'IID': kw['iid_assumed'] = False
    if fault == 'failure_policy': kw['failure_policy'] = None
    if fault == 'totality': kw['compiler_domain'] = None
    if fault == 'population': kw['population'] = None
    if fault == 'missing_enclosure': rows[0]['evidence']['per_word_enclosures'] = []
    if fault == 'unqualified_enclosure': rows[0]['evidence']['per_word_enclosures'][0]['kind'] = 'observed_residual'
    if fault == 'sector_enclosure': rows[0]['evidence']['per_word_enclosures'][0]['scope'] = 'sector36'
    if fault == 'unfinished': rows[0] = a.outcome(rows[0]['sample'], status='INCOMPLETE')
    if fault == 'failed': rows[0] = a.outcome(rows[0]['sample'], status='P0_COMPILE_FAILED', stage='P0_compile', error=RuntimeError('mock'))
    if fault == 'empty': rows = []
    p = payload(rows, **kw)
    assert p['status'] == 'NOT_READY' and p['missing']
    assert len(p['records']) == len(rows) and p['calibration_confidence'] is None
    assert a.evaluate_calibration(p)['bound'] is None


@pytest.mark.parametrize('policy', ['stop_on_failure', 'record_all_fixed_N'])
def test_success_only_subset_cannot_estimate_original_fixed_population(policy):
    rows = [success(sample(i)) for i in range(2)]
    ids = [r['sample']['sample_id'] for r in rows]
    p = payload(rows[1:], planned_ids=ids, failure_policy=policy)
    assert 'all_fixed_N_attempts' in p['missing'] and p['status'] == 'NOT_READY'


@pytest.mark.parametrize('fault', ['source', 'candidate', 'law', 'old_certificate', 'wrong_word', 'wrong_target'])
def test_certificates_require_new_sample_binding_and_literal_target(fault):
    rows = [success(sample(i)) for i in range(2)]; cert = rows[0]['evidence']['per_word_enclosures'][0]
    if fault == 'source': cert['calibration_binding']['source_id'] = 'wrong'
    if fault == 'candidate': cert['calibration_binding']['candidate'] = 'wrong'
    if fault == 'law': cert['calibration_binding']['law_id'] = 'wrong'
    if fault == 'old_certificate': cert.pop('calibration_binding')
    if fault == 'wrong_word': cert['binding']['word_id'] = 'wrong'
    if fault == 'wrong_target': cert['binding']['target_id'] = 'wrong'
    with pytest.raises(ValueError): payload(rows)


def test_no_scientific_runner_is_enabled():
    with pytest.raises(PermissionError): a.run_science()
    with pytest.raises(NotImplementedError): a.run_science(execute_science=True)


def test_passive_actual_contract_with_all_scientific_imports_blocked():
    code = """
import sys
class Block:
 def find_spec(self, fullname, *args):
  if fullname.split('.')[0] in ('numpy','scipy','qiskit','pyscf','ffsim','openfermion','flint'):
   raise AssertionError('PASSIVE_IMPORT_ONLY:'+fullname)
sys.meta_path.insert(0,Block())
from trottertracks import c_fresh_iid_calibration as a
from pathlib import Path
p=a.preflight(Path.cwd(),a.CONTRACT)
assert not p['execution_ready'] and p['scientific_calls']==p['real_H4_samples']==p['arrays_decoded']==0
"""
    subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True)


@pytest.mark.parametrize('fault', ['retry', 'resume', 'stop', 'method', 'N', 'delta', 'candidate', 'law', 'source', 'fresh'])
def test_passive_rejects_contract_source_changes_and_retry_resume(tmp_path, fault):
    source = tmp_path/'SOURCE.json'; tiny = tmp_path/'tiny.txt'; tiny.write_text('invented technical source')
    import hashlib
    source.write_text(json.dumps(dict(identity=identity(), files={'tiny.txt': hashlib.sha256(tiny.read_bytes()).hexdigest()})))
    c = dict(scope=a.SCOPE, base_commit=a.BASE, executable=False, maximum_logical_units=None,
        N=None, N_by_round=None, delta_cal=None, delta_compile=None, method=None,
        compiler_failure_policy=None, retry=0, resume=0, mandatory_stop=True, identity=identity(),
        sampling_model=identity()['model'], stream_policy=a.STREAM, source_manifest='SOURCE.json',
        source_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), saved_input_bytes={}, fresh_output='run01')
    if fault in ('retry', 'resume'): c[fault] = 1
    if fault == 'stop': c['mandatory_stop'] = False
    if fault == 'method': c['method'] = 'threshold_tail'
    if fault == 'N': c['N'] = 1
    if fault == 'delta': c['delta_cal'] = '1/20'
    if fault == 'candidate': c['identity']['candidate'] = 'wrong'
    if fault == 'law': c['sampling_model']['id'] = 'wrong'
    if fault == 'source': tiny.write_text('changed')
    if fault == 'fresh': (tmp_path/'run01').mkdir()
    path = tmp_path/'contract.json'; path.write_text(json.dumps(c))
    with pytest.raises(ValueError): a.preflight(tmp_path, path)
