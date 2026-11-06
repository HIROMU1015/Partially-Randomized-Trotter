"""Invented complex signals only; never load stored H4 signals/private arrays."""
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np
import pytest

from trotterlib import b2_p1_strong_pf_branch_shots as m
from trotterlib import b2_p1_strong_pf_validation as signal
from trotterlib.o1_m1_resource_bridge import _hoeffding_shots


@pytest.fixture(autouse=True)
def synthetic_only(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('No stored scientific inputs, PF signals, circuits or prices in synthetic tests')
    monkeypatch.setattr(np, 'load', forbidden)
    monkeypatch.setattr(m.branch, 'git', forbidden)
    monkeypatch.setattr(m.branch, 'fixed_json', forbidden)
    monkeypatch.setattr(signal, 'native_mean_step', forbidden)
    monkeypatch.setattr(signal.p, 'load_private_inputs', forbidden)
    monkeypatch.setattr(signal, '_ordered_exponentials', forbidden)
    from qiskit import QuantumCircuit
    monkeypatch.setattr(QuantumCircuit, '__init__', forbidden)


def identity():
    return dict(original_hamiltonian='H_phys:S0_array_hash:bab58c17220230196aff8a7c7d1e1264c36e101f0fe6bf367eacba4538ae10b2',
        original_state='S0_array_hash:1a2df11b16743e68fb27a3f4b39e78e145897cb49a8cc931b080e44e7400c1db',
        df_hamiltonian='de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424',
        sector='Nalpha=2,Nbeta=2;OF_basis_S0_hash:665f72f65a4300043a32f3aabc76b7deafbd2563ad3947228dd937bc6d521dd7',
        spin_orbital_order='2p=alpha,2p+1=beta;Qiskit full index=sum(n_q*2**q)',
        state_role='original_H_ground_state', state_preparation_included=False, DF_state_used=False)


def fixture_arm(name='Full_S2_s1', formula='Full_S2', substeps=1):
    rows = []
    for index in range(18):
        Q = 1 << index
        phys = np.exp(1j*Q*.12)
        PF = .8*np.exp(1j*Q*(.12+1e-8))
        qualification = signal.qualify(index, phys, phys, PF, PF)
        rows.append(dict(m=index, Q=Q, T_Ha_inverse=Q*.02,
            signals={'PF':signal.p.signal_record(PF)}, qualification=qualification,
            status=qualification['status']))
    return dict(arm_id=name, formula=formula, substeps=substeps, rounds=rows,
        status='P1_ALL18_CONDITIONAL_SIGNAL_PASS', state_preparation_included=False,
        finite_RTE=False, attenuation=1., normalization=1., execution_SOURCE=m.SIGNAL_SOURCE,
        scientific_context=identity())


def bundle():
    arms = [fixture_arm(*entry) for entry in m.ARMS]
    return dict(schema_version='b2_p1_strong_pf_eight_arms_v1', status='FIXED_EIGHT_ARM_SCREEN_COMPLETE',
        execution_SOURCE=m.SIGNAL_SOURCE, shared_input_status='PASS', P1_conditions=m.expected_conditions(),
        scientific_context=identity(), input_identity=dict(identity(), status='PASS'),
        qualified_arms=[a[0] for a in m.ARMS], arms=arms)


def reference():
    return {'validation':{'energy_Ha':-6.}}, {'commit':'synthetic_energy_after_decoding'}


def test_shared_context_selects_only_fixed_three_arms_and_never_reads_stop_rounds():
    value = bundle()
    # Other arms carry deliberately invalid/unreadable signal payloads.
    value['arms'].append(dict(arm_id='Full_S2_s4', rounds=object(), status='STOP'))
    arms, context, conditions = m.validate_shared(value)
    assert [a['arm_id'] for a in arms] == [a[0] for a in m.ARMS]
    assert context == identity() and conditions == m.expected_conditions()


@pytest.mark.parametrize('mutation', ['SOURCE','conditions','identity','state','duplicate','arm_SOURCE','substeps'])
def test_shared_mismatch_rejected(mutation):
    value = bundle()
    if mutation == 'SOURCE': value['execution_SOURCE'] = 'other'
    if mutation == 'conditions': value['P1_conditions']['allocation']['beta_stat_budget'] = .28
    if mutation == 'identity': value['scientific_context']['df_hamiltonian'] = 'other'
    if mutation == 'state': value['input_identity']['DF_state_used'] = True
    if mutation == 'duplicate': value['arms'].append(copy.deepcopy(value['arms'][0]))
    if mutation == 'arm_SOURCE': value['arms'][0]['execution_SOURCE'] = 'other'
    if mutation == 'substeps': value['arms'][0]['substeps'] = 4
    with pytest.raises(ValueError): m.validate_shared(value)


@pytest.mark.parametrize('mutation', ['missing','duplicate','time','radius','phase','A','disk','RTE_reserve','PF_cap','match','status'])
def test_arm_inventory_signal_and_unchanged_budget_rejection(mutation):
    arm = fixture_arm(); row = arm['rounds'][0]; qualification = row['qualification']
    if mutation == 'missing': arm['rounds'].pop()
    if mutation == 'duplicate': arm['rounds'][2]['m'] = 1
    if mutation == 'time': row['T_Ha_inverse'] = .03
    if mutation == 'radius': qualification['bounds']['reference_radius_lower_bound'] = 0.
    if mutation == 'phase': row['signals']['PF']['phase_rad'] += .1
    if mutation == 'A': qualification['attenuation'] = .8
    if mutation == 'disk': qualification['bounds']['corrected_signal_error_bound'] = 1e-8
    if mutation == 'RTE_reserve': qualification['RTE_budget_reserved_unreallocated'] = 0.
    if mutation == 'PF_cap': qualification['bounds']['pf_phase_bound'] = .061
    if mutation == 'match': qualification['independent_difference']['complex_absolute'] = 1.01e-9
    if mutation == 'status': arm['status'] = 'STOP_ARM_INELIGIBLE'
    with pytest.raises(ValueError): m.validate_arm(arm)


def test_every_decoder_finishes_before_energy_and_existing_shot_formula_used(monkeypatch):
    arms = bundle()['arms']; items = m.decode_arms(arms)
    assert all(i['status'] == 'DECODED_AWAITING_POSTHOC_REFERENCE' for i in items)
    original_choices = [[r['selected_branch'] for r in i['reconstruction']['rounds']] for i in items]
    calls = []
    actual = m.shots.account_radius_round
    def spy(**kwargs):
        calls.append(kwargs)
        assert kwargs['config'] is None and kwargs['cost'] is None
        assert kwargs['corrected_signal_error_bound'] == 0.
        assert kwargs['allocation'] == m.shots.ALLOCATION
        return actual(**kwargs)
    monkeypatch.setattr(m.shots, 'account_radius_round', spy)
    def after_decoding():
        assert all(i['reconstruction']['all_decoding_complete'] for i in items)
        assert not calls
        return reference()
    m.complete_arms(items, after_decoding, context_hash='toy_context')
    assert len(calls) == 54
    assert all(i['status'] == m.PASS for i in items)
    assert all(i['correctness']['shots_prices_total_cost'].startswith('Conditional sufficient shots') for i in items)
    assert original_choices == [[r['selected_branch'] for r in i['reconstruction']['rounds']] for i in items]
    for item in items:
        row = item['rounds'][0]
        expected = _hoeffding_shots(attenuation=1., reference_radius=.8-m.branch.U,
            signal_error_bound=0., beta_stat_budget=.24, alpha_axis=.05/36)
        assert row['N_X'] == row['N_Y'] == expected
        assert row['raw_radius_lower'] == row['PF_radius_lower']
        assert row['complex_raw_signal'] == row['complex_PF_signal']
        assert row['price'] is None and row['total_circuit_cost'] is None
        assert item['totals']['N_all_rounds'] == sum(r['N_round'] for r in item['rounds'])
        assert item['totals']['exact_failure_budget_sum'] == '1/20'
        assert row['fixed_phase_caps_rad']['RTE'] == .04
        assert row['fixed_allocation'] == asdict(m.shots.ALLOCATION)


def test_one_arm_stops_remaining_two_continue_without_loading_reference_early():
    arms = bundle()['arms']; arms[0]['rounds'].pop()
    items = m.decode_arms(arms)
    m.complete_arms(items, reference, context_hash='toy')
    assert items[0]['status'] == 'STOP_ARM_BRANCH_OR_SIGNAL'
    assert [item['status'] for item in items[1:]] == [m.PASS, m.PASS]
    assert 'totals' not in items[0]


def test_branch_ambiguity_stops_only_affected_arm():
    arms = bundle()['arms']; row = arms[0]['rounds'][1]
    row['signals']['PF'] = signal.p.signal_record(-.8+0j)
    row['qualification'] = signal.qualify(1, -.8+0j, -.8+0j, -.8+0j, -.8+0j)
    # m0=0 makes the m1 phase pi nearest-branch tie exact.
    row0 = arms[0]['rounds'][0]
    row0['signals']['PF'] = signal.p.signal_record(.8+0j)
    row0['qualification'] = signal.qualify(0, .8+0j, .8+0j, .8+0j, .8+0j)
    items = m.decode_arms(arms)
    assert items[0]['status'] == 'STOP_ARM_BRANCH_OR_SIGNAL'
    m.complete_arms(items, reference, context_hash='toy')
    assert all(item['status'] == m.PASS for item in items[1:])


def test_wrong_known_energy_cannot_repair_selected_branches_or_trigger_shots(monkeypatch):
    items = m.decode_arms(bundle()['arms'])
    before = [[r['selected_branch'] for r in i['reconstruction']['rounds']] for i in items]
    def forbidden(*a, **k): raise AssertionError('Failed energy arm must not receive shots')
    monkeypatch.setattr(m.shots, 'account_saved_round', forbidden)
    m.complete_arms(items, lambda:({'validation':{'energy_Ha':40.}}, {}), context_hash='toy')
    assert all(i['status'] == 'STOP_ARM_ENERGY_OR_SHOTS' for i in items)
    assert before == [[r['selected_branch'] for r in i['reconstruction']['rounds']] for i in items]


def test_no_reference_opened_when_every_arm_has_already_stopped():
    items = [{'status':'STOP_ARM_BRANCH_OR_SIGNAL'} for _ in range(3)]
    def forbidden(): raise AssertionError('Energy reference not needed')
    assert m.complete_arms(items, forbidden, context_hash='toy') is None


def test_per_arm_json_csv_integer_totals_and_branch_records(tmp_path):
    items = m.decode_arms(bundle()['arms'])
    m.complete_arms(items, reference, context_hash='toy')
    summary = m.write_arm(tmp_path/'toy_arm', items[0])
    data = json.loads((tmp_path/'toy_arm/result.json').read_text())
    assert len(data['rounds']) == 18
    assert data['serialization_check']['JSON_CSV_integer_match']
    assert 'reconstruction' not in summary
    assert all('branch_index' in r and r['fixed_phase_caps_rad']['statistical'] == .24
               for r in data['rounds'])
    assert all(type(r['N_X']) is int for r in data['rounds'])


def test_shared_B2_metadata_mismatch_before_shots(monkeypatch):
    value = bundle(); refs = []
    def fixed(commit, path):
        refs.append((commit, path))
        return dict(status='P1_CONDITIONAL_BRANCH_ENERGY_PASS' if commit == m.shots.BRANCH_COMMIT else
            'ALL18_CONDITIONAL_SUFFICIENT_SHOTS', scientific_context={'wrong':True},
            fixed_conditions=m.expected_conditions()), {}
    monkeypatch.setattr(m.branch, 'fixed_json', fixed)
    with pytest.raises(ValueError, match='SHARED_B2_STRONG_P1_CONTEXT'):
        m.check_b2_context(identity(), m.expected_conditions(), value)
    assert len(refs) == 2


def mocked_runner_inputs(tmp_path, monkeypatch, *, mismatch=False):
    value = bundle(); value['SOURCE_SHA256'] = {}
    bm, sm = {'blob':'synthetic_B2_branch'}, {'blob':'synthetic_B2_shots'}
    value['saved_inputs'] = [bm, sm]
    source = 'synthetic_committed_adapter'
    paths = ['src/trotterlib/b2_p1_strong_pf_branch_shots.py',
        'scripts/run_b2_p1_strong_pf_branch_shots.py', 'src/trotterlib/b2_p1_branch_validation.py',
        'src/trotterlib/b2_p1_shot_accounting.py', 'src/trotterlib/rpe_four_round_phase_validation.py',
        'src/trotterlib/o1_m1_resource_bridge.py', 'src/trotterlib/rpe_resource_accounting.py',
        'src/trotterlib/finite_rte_signal_validation.py']
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    for path in paths:
        p = tmp_path/path; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b'artificial source')
    def fake_git(*args):
        if args == ('rev-parse', 'HEAD'): return source.encode()
        if args[0] == 'rev-parse': return b'synthetic_P1_blob'
        if args[0] == 'show' and args[1].startswith(source+':'): return b'artificial source'
        if args[0] == 'show' and args[1] == m.P1_COMMIT+':'+m.P1_PATH: return b'artificial P1 document'
        raise AssertionError(args)
    monkeypatch.setattr(m.branch, 'git', fake_git)
    events = []
    actual_decode = m.branch.decode_raw_means
    def decode(*args, **kwargs):
        events.append('decode')
        return actual_decode(*args, **kwargs)
    monkeypatch.setattr(m.branch, 'decode_raw_means', decode)
    def fixed(commit, path):
        if (commit, path) == (m.SIGNAL_COMMIT, m.SIGNAL_PATH): return value, {'blob':'synthetic_signal'}
        assert events.count('decode') == 3
        if (commit, path) == (m.branch.STATE_COMMIT, m.branch.STATE_RESULT):
            events.append('energy_reference')
            return reference()
        context = identity() if not mismatch else {'mismatched':True}
        common = dict(scientific_context=context, fixed_conditions=m.expected_conditions(),
                      execution_SOURCE='synthetic_B2_SOURCE', SOURCE_SHA256={})
        if (commit, path) == (m.shots.BRANCH_COMMIT, m.shots.BRANCH_PATH):
            return dict(common, status='P1_CONDITIONAL_BRANCH_ENERGY_PASS'), bm
        if (commit, path) == (m.B2_SHOT_COMMIT, m.B2_SHOT_PATH):
            return dict(common, status='ALL18_CONDITIONAL_SUFFICIENT_SHOTS', saved_inputs=[bm],
                        totals={'N_all_rounds':777}), sm
        raise AssertionError((commit, path))
    monkeypatch.setattr(m.branch, 'fixed_json', fixed)
    return events


def test_full_runner_on_artificial_bundle_only(tmp_path, monkeypatch):
    events = mocked_runner_inputs(tmp_path/'source', monkeypatch)
    out = tmp_path/'results'
    result = m.run(out)
    assert result['status'] == 'THREE_ARMS_CONDITIONAL_BRANCH_SHOTS_PASS'
    assert events == ['decode']*3+['energy_reference']
    assert result['B2_reference_quotation']['total_shots'] == 777
    assert all(arm['totals']['N_all_rounds'] != 777 for arm in result['arms'])
    for arm in result['arms']:
        raw = json.loads((out/arm['result_path']).read_text())
        assert raw['execution_SOURCE'] == result['execution_SOURCE']
        assert raw['serialization_check']['JSON_CSV_integer_match']
    assert result['forbidden_work']['new_PF_signal_computation'] == 0


def test_shared_failure_stops_full_runner_before_reference_and_shots(tmp_path, monkeypatch):
    events = mocked_runner_inputs(tmp_path/'source', monkeypatch, mismatch=True)
    def forbidden(*a, **kw): raise AssertionError('No shot call after shared mismatch')
    monkeypatch.setattr(m.shots, 'account_saved_round', forbidden)
    with pytest.raises(ValueError, match='SHARED_B2_STRONG_P1_CONTEXT'):
        m.run(tmp_path/'results')
    raw = json.loads((tmp_path/'results/result.json').read_text())
    assert raw['status'] == 'STOP_SHARED_INPUT_SOURCE_OR_EXECUTION' and raw['mandatory_stop']
    assert events == ['decode']*3
