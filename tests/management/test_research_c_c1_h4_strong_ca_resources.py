"""Artificial B2/B3 scalar evidence; no real saved-value evaluation."""
import json
import math
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock, patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks import c1_h4_strong_ca_resources as m
from test_research_c_c1_h4_ca_resources import artificial as legacy_artificial, metrics


def segment(label, epsilon=1e-10, alpha=1.000000000001):
    return dict(label=label, input_dimension=512, output_dimension=512,
        coverage='all full-Fock controlled input columns and all output components',
        evidence_level=m.LEVEL, floating_point_rounding_enclosed=False,
        rigorous_operator_bound=None, global_phase_fit=False,
        **{k: dict(max_entry=1e-12) for k in ('raw_reference', 'compiler', 'compiled_reference')},
        operator_bound_pilot=epsilon, compiled_reference_frobenius=epsilon,
        norm_upper_pilot=alpha, norm_guards={k: dict(reference_norm_upper_pilot=alpha)
            for k in ('raw', 'compiled', 'reference')})


@pytest.fixture
def artificial(tmp_path, monkeypatch, legacy_artificial):
    def create(fault=None):
        legacy_path, b = legacy_artificial()
        old = m.io.read(legacy_path)
        b['native_contract']['ground_states'] = {'XDF': 'artificial_ground.npz'}
        b['native_contract']['original_input'] = {'sha256': 'synthetic'}
        b['native']['shells'] = {k: segment('shell:' + k, 1e-12) for k in ('head', 'X', 'Y')}
        for k in ('native', 'native_contract'):
            (tmp_path / old['records'][k]).write_text(json.dumps(b[k]))
        monkeypatch.setattr(m.prior, 'NATIVE_CONTRACT_SHA', m.io.sha(tmp_path / old['records']['native_contract']))
        old['native_contract_sha256'] = m.prior.NATIVE_CONTRACT_SHA
        old['saved_inputs'] = {p: m.io.sha(tmp_path / p) for p in old['saved_inputs']}
        legacy_path.write_text(json.dumps(old))
        c = dict(scope=m.SCOPE, status='UNAPPROVED_SCIENCE_CONTRACT_DRAFT',
            conditions=m.CONDITIONS, accuracy_target=m.TARGET, candidate_order=list(m.ROUTES.values()),
            cost_scope=m.COST_SCOPE, synthesis=m.synthesis.VERSION, call_limits=m.LIMITS,
            maximum_logical_units=m.MAX_CALLS, compiler=m.prior.prior.profile_dict(),
            gaussian=m.prior.prior.GAUSSIAN, compiler_phase=m.prior.prior.PHASE,
            source_manifest='SOURCE.json', source_manifest_sha256=m.io.sha(tmp_path / 'SOURCE.json'),
            fresh_output=m.FAMILY + '/<execution-date>/run01', legacy_contract=legacy_path.name,
            evaluation_packages={}, shells={k: {'raw': k + '.raw', 'compiled': k + '.compiled'}
                for k in ('head', 'X', 'Y')}, **{k: 0 for k in m.FORBIDDEN})
        source = m.io.read(tmp_path / 'SOURCE.json')
        def record(arm):
            route = m.ROUTES[arm]
            return dict(route=route, arm=arm, s=1, segment=segment(route),
                metrics=metrics(8 if arm == 'CDF' else 9), XY={'compiled_reference': dict(error=1e-12)})
        b.update(strong=dict(status='H4_STRONG_BASELINE_S1_NUMERICAL_COMPLETE',
                     candidates={m.ROUTES['CDF']: record('CDF')}),
                 B3=dict(record('XDF'), status='H4_B3_S1_NUMERICAL_COMPLETE'))
        for key, scope, limits in [('strong', m.b3.strong.SCOPE, m.b3.strong.LIMITS),
                                   ('B3', m.b3.SCOPE, m.b3.LIMITS)]:
            b[key + '_SOURCE'] = dict(source, execution_commit=m.OLD_SOURCE[key])
            b[key + '_STOP'] = dict(failed=False, mandatory_stop=True, retry=0, resume=0,
                scientific_units=sum(limits.values()), call_counts=limits)
            b[key + '_contract'] = dict(scope=scope, source_manifest='SOURCE.json',
                conditions=dict(m.CONDITIONS, route=('B0 saved; B1 XDF spin; B2 both legacy bases relative fusion'
                    if key == 'strong' else m.b3.ROUTE)), compiler=c['compiler'], gaussian=c['gaussian'],
                compiler_phase=c['compiler_phase'], allowances=m.b3.strong.ALLOWANCES,
                shells=c['shells'], original_input={'sha256': 'synthetic'}, references={'XDF': 'synthetic-reference'})
        c['records'] = {k: 'new_' + k + '.json' for k in m.RECORDS}
        b['B3_contract'].update(previous_contract=c['records']['strong_contract'],
            previous_scientific_SOURCE=m.OLD_SOURCE['strong'], reference='synthetic-reference',
            ground='artificial_ground.npz')
        s = b['strong']['candidates'][m.ROUTES['CDF']]['segment']
        if fault == 'source': b['B3_SOURCE']['execution_commit'] = 'wrong'
        if fault == 'candidate': b['B3']['route'] = 'B2_XDF_relative'
        if fault == 'coverage': s['input_dimension'] = 36
        if fault == 'missing': s['operator_bound_pilot'] = None
        if fault == 'missing_guard': s['norm_guards'] = None
        if fault == 'guard': s['norm_upper_pilot'] = 1.
        if fault == 'nonfinite': s['norm_upper_pilot'] = float('nan')
        if fault == 'infeasible':
            s['norm_upper_pilot'] = 1.01
            for x in s['norm_guards'].values(): x['reference_norm_upper_pilot'] = 1.01
        if fault == 'overflow':
            s['norm_upper_pilot'] = 2.
            for x in s['norm_guards'].values(): x['reference_norm_upper_pilot'] = 2.
        if fault == 'shell_binding': b['B3_contract']['shells'] = {}
        if fault == 'condition': b['B3_contract']['conditions']['substeps'] = [4]
        if fault == 'PF':
            pfpath = tmp_path / old['records']['CDF_PF_s1']
            pf = m.io.read(pfpath); pf['rounds'][0]['pf_operator_error'] = .2; pfpath.write_text(json.dumps(pf))
            old['saved_inputs'][str(pfpath.relative_to(tmp_path))] = m.io.sha(pfpath)
            legacy_path.write_text(json.dumps(old))
        for k, p in c['records'].items():
            # JSON allows a nonfinite fixture so the runtime validator is tested.
            (tmp_path / p).write_text(json.dumps(b[k]))
        c['saved_inputs'] = {p: m.io.sha(tmp_path / p) for p in [*c['records'].values(), legacy_path.name]}
        path = tmp_path / 'strong_contract.json'; m.io.save(path, c)
        return path, b
    return create


@pytest.mark.parametrize('fault', [None, 'infeasible', 'missing', 'missing_guard', 'overflow',
    'source', 'candidate', 'coverage', 'guard', 'nonfinite', 'shell_binding', 'condition', 'PF'])
def test_end_to_end(tmp_path, artificial, fault):
    path, b = artificial(fault)
    out = tmp_path / m.FAMILY / '2026-10-26/run01'
    blocked = Mock(side_effect=AssertionError('NO_SCIENTIFIC_BUILD_FIT_OR_OLD_RUN'))
    bad = fault in ('source', 'candidate', 'coverage', 'guard', 'nonfinite', 'shell_binding', 'condition', 'PF')
    with patch.object(m.prior, 'run', blocked), patch.object(m.b3, 'run', blocked), \
            patch.object(m.b3.strong, 'run', blocked), patch.object(m.prior, 'source_commit', return_value='synthetic'):
        if bad:
            with pytest.raises(ValueError): m.run(tmp_path, path, out)
        else: m.run(tmp_path, path, out)
    blocked.assert_not_called()
    stop = m.io.read(out / 'STOP.json')
    assert stop['retry'] == stop['resume'] == 0 and stop['scientific_units'] <= 86
    journal = [json.loads(x) for x in (out / 'science_calls.jsonl').read_text().splitlines()]
    if bad:
        assert stop['failed'] and (out / 'failure.json').exists() and not (out / 'result.json').exists()
        first = next(i for i, x in enumerate(journal) if x['event'] == 'FAIL')
        assert not any(x['event'] == 'attempt' for x in journal[first + 1:])
    else:
        result = m.io.read(out / 'result.json'); selected = result['candidates'][m.ROUTES['CDF']]
        assert not stop['failed'] and list(result['candidates']) == list(m.ROUTES.values())
        assert len(selected['synthesis_rounds']) == len(selected['rounds']) == 13
        if fault:
            expected = 'INFEASIBLE_REGISTERED_SUFFICIENT_CONDITION' if fault == 'infeasible' else 'UNRESOLVED'
            assert selected['status'] == expected
            assert selected['cost']['totals'] is None and selected['per_shot_costs'] == []
            assert stop['scientific_units'] == 73
            assert result['comparison']['verdict'] == 'INDETERMINATE_MISSING_QUALIFICATION'
            assert result['candidates'][m.ROUTES['XDF']]['status'] == 'NUMERICAL_PILOT_CA_PASS'
        else:
            assert stop['scientific_units'] == 86 and stop['call_counts'] == m.LIMITS
            for value in result['candidates'].values():
                assert value['status'] == 'NUMERICAL_PILOT_CA_PASS'
                costs = {(x['m'], x['axis']): x for x in value['per_shot_costs']}
                for row, accuracy in zip(value['synthesis_rounds'], value['rounds'], strict=True):
                    assert row['occurrences'] == 1 << row['m']
                    assert row['combined_complex_error_pilot'] == math.fsum([
                        row['saved_PF_operator_error'], row['complex_signal_bound_pilot']])
                    assert accuracy['raw_signal_error'] == row['combined_complex_error_pilot']
                    assert type(accuracy['shots']) is int and accuracy['statistical_margin'] >= 0
                n = 8 if value['arm'] == 'CDF' else 9
                assert costs[12, 'X']['mean']['rz_count'] == 4096 * n + 3
                assert costs[12, 'X']['kernel_occurrences'] == 4096
                assert costs[12, 'X']['depth_semantics'] == 'scheduled_serial_sum'
                assert value['cost']['totals']['rz_count'] == sum(
                    p['shots'] * sum(costs[p['m'], a]['mean']['rz_count'] for a in ('X', 'Y')) for p in value['rounds'])
                expected = (1e-5 if value['arm'] == 'CDF' else 2e-5) + m.TARGET['beta_total'] / (.1 * 4096)
                assert value['numerical_energy_assessment']['energy_error_bound_ha'] == pytest.approx(expected)
                assert value['CA_achieved'] is value['rigorous_enclosure'] is value['guaranteed_actual_shots'] is None
                assert not value['guaranteed_total_resource_superiority']
    before = (out / 'STOP.json').read_bytes()
    with pytest.raises(ValueError, match='FRESH'): m.run(tmp_path, path, out)
    assert (out / 'STOP.json').read_bytes() == before


def test_exact_propagate_dispatch_and_no_old_combined(artificial):
    _, b = artificial()
    packet = m.attach_evidence(b, 'CDF'); pf = packet['PF_rounds'][3]
    with patch.object(m.synthesis, 'propagate', wraps=m.synthesis.propagate) as call:
        r = m.propagate_round(packet, pf)
    call.assert_called_once_with(packet['segment'], packet['shells'], s=1, m=3,
                                saved_pf_error=pf['pf_operator_error'])
    assert r['alpha'] >= packet['segment']['norm_upper_pilot'] and r['norm_growth_pilot'] >= 1
    assert r['saved_PF_operator_error'] != b['native']['candidates']['CDF_s1']['rounds'][3]['combined_complex_error_pilot']


@pytest.mark.parametrize('fault', ['source', 'saved_bytes', 'calls', 's4', 'candidate_order', 'policy'])
def test_passive_preflight_failures(tmp_path, artificial, fault):
    path, _ = artificial(); c = m.io.read(path)
    if fault == 'source': (tmp_path / 'artificial_source.py').write_text('changed')
    if fault == 'saved_bytes': (tmp_path / c['records']['B3']).write_text('{}')
    if fault == 'calls': c['maximum_logical_units'] += 1
    if fault == 's4': c['conditions'] = dict(c['conditions'], substeps=[4])
    if fault == 'candidate_order': c['candidate_order'].reverse()
    if fault == 'policy': c['cost_scope'] = 'optimized_whole_round'
    path.write_text(json.dumps(c))
    with patch.object(m, 'load_bundle', side_effect=AssertionError('NO_NUMERIC_JOIN')):
        with pytest.raises(ValueError): m.preflight(tmp_path, path)


def test_import_is_passive():
    code = "import sys;sys.path.insert(0,'src');from trottertracks import c1_h4_strong_ca_resources;assert not any(k in sys.modules for k in ('numpy','scipy','qiskit','ffsim'))"
    subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, check=True)
    assert m.MAX_CALLS == 86 and m.TARGET == m.prior.TARGET


def test_missing_PF_is_unresolved(artificial):
    _, b = artificial()
    packet = m.attach_evidence(b, 'CDF')
    row = m.propagate_round(packet, dict(m=0, pf_operator_error=None))
    assert row['status'] == 'UNRESOLVED_SYNTHESIS'
    assert row['saved_PF_operator_error'] is row['complex_signal_bound_pilot'] is None


def test_fixed_N_failure_is_unresolved_without_cost(tmp_path, artificial):
    path, _ = artificial(); out = tmp_path / m.FAMILY / '2026-10-26/run01'
    with patch.object(m.reused.det, 'assess_fixed_round', return_value=dict(status='INFEASIBLE_FIXED_SHOTS')):
        r = m.run(tmp_path, path, out)
    assert all(x['status'] == 'UNRESOLVED' and x['cost']['totals'] is None for x in r['candidates'].values())
    assert r['scientific_units'] == 60
    assert all(x['per_shot_costs'] == [] for x in r['candidates'].values())
