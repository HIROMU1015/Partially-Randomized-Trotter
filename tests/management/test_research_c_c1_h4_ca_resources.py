"""Artificial scalar evidence only. No saved molecule joins or circuits."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock, patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks import c1_h4_ca_resources as m


def metrics(n, measurement=0):
    return {k: n for k in m.accuracy.METRICS} | dict(measurement_count=measurement)


@pytest.fixture
def artificial(tmp_path, monkeypatch):
    """Fake H4-shaped metadata, invented scalar errors/counts, no real values."""
    def create(fault=None):
        native = dict(status='H4_FAIR_NATIVE_FULL512_PILOT_COMPLETE', local_reports=[{}] * 18,
            candidates={}, shells={k: {} for k in ('head', 'X', 'Y')},
            shell_metrics={k: metrics(1 if k == 'head' else 2, k != 'head') for k in ('head', 'X', 'Y')})
        # Measurement count is an integer, not bool.
        for k in native['shell_metrics']:
            native['shell_metrics'][k]['measurement_count'] = int(k != 'head')
        old = dict(scope=m.prior.SCOPE, conditions=copy.deepcopy(m.prior.CONDITIONS),
            allowances=m.prior.ALLOWANCES, compiler=m.prior.profile_dict(),
            gaussian=m.prior.GAUSSIAN, compiler_phase=m.prior.PHASE, synthesis=m.prior.synthesis.VERSION)
        source = dict(files={'artificial_source.py': 'not-a-real-science-source'})
        b = dict(native=native, native_contract=old,
            native_STOP=dict(failed=False, mandatory_stop=True, scientific_units=m.prior.MAX_CALLS,
                call_counts=m.prior.LIMITS, retry=0, resume=0),
            native_SOURCE=dict(source, execution_commit=m.NATIVE_SOURCE),
            identity=dict(status='PASS', DF_bounds_Ha=dict(CDF=1e-5, XDF=2e-5)),
            CDF_prepared=dict(state_preparation='ORACLE_EXCLUDED', df_error_bounds_ha=dict(CDF=1e-5)),
            CDF_STOP=dict(failed=False, mandatory_stop=True), XDF_STOP=dict(failed=False, mandatory_stop=True),
            XDF=dict(status='XDF8_STAGE1_COMPLETE', representation='PASS', ground=dict(status='PASS'),
                saved_DF=dict(status='PASS', saved_norm_plus_allowance_Ha=2e-5), candidates={}))
        for arm, s in m.CANDIDATES:
            label = f'{arm}_s{s}'
            rows, pfrows = [], []
            for i in range(13):
                pf = (.015 if arm == 'CDF' else .025) * (1 << i) / 4096
                syn = 1e-8 * (1 << i)
                if fault == 'infeasible' and label == 'CDF_s1' and i == 12:
                    syn = .3
                combined = math.fsum([pf, syn])
                row = dict(m=i, s=s, occurrences=s * (1 << i), evidence_level=m.LEVEL,
                    saved_PF_operator_error=pf, complex_signal_bound_pilot=syn,
                    combined_complex_error_pilot=combined, phase_bias_bound_pilot=math.asin(combined),
                    registered_phase_room=math.pi / 12, rigorous_signal_error_bound=None,
                    floating_point_rounding_enclosed=False)
                if fault == 'unresolved' and label == 'CDF_s1' and i == 0:
                    row.update(complex_signal_bound_pilot=None, combined_complex_error_pilot=None,
                        phase_bias_bound_pilot=None)
                if fault == 'signal' and label == 'CDF_s1' and i == 3:
                    row['combined_complex_error_pilot'] += .01
                rows.append(row)
                pfrows.append(dict(m=i, attenuation=1., rte_corrected_operator_error=0., pf_operator_error=pf))
            native['candidates'][label] = dict(arm=arm, s=s, rounds=rows,
                metrics=metrics(8 if arm == 'CDF' else 12), segment=dict(label=label,
                    input_dimension=512, output_dimension=512, evidence_level=m.LEVEL,
                    rigorous_operator_bound=None, floating_point_rounding_enclosed=False,
                    coverage='all full-Fock controlled input columns and all output components'))
            if arm == 'CDF':
                b[f'CDF_PF_s{s}'] = dict(candidate=dict(arm='CDF', family='D', s=s, r=0, K=0),
                    status='POTENTIALLY_FEASIBLE', rounds=pfrows)
            else:
                b['XDF']['candidates'][str(s)] = dict(summary=dict(status='POTENTIALLY_FEASIBLE'), rounds=pfrows)
        if fault == 'identity': b['native_SOURCE']['execution_commit'] = 'changed'
        if fault == 'alias': b['XDF']['ground']['status'] = 'FAIL'
        if fault == 'metrics': native['candidates']['CDF_s1']['metrics']['rz_count'] = -1
        if fault == 'label': native['candidates']['CDF_s1']['arm'] = 'XDF'
        if fault == 'coverage': native['candidates']['CDF_s1']['segment']['input_dimension'] = 72
        if fault == 'DF': b['identity']['DF_bounds_Ha']['CDF'] = b['CDF_prepared']['df_error_bounds_ha']['CDF'] = .001
        if fault == 'conditions': old['conditions']['CDF_rank'] = 10
        if fault == 'PF': b['CDF_PF_s1']['rounds'][0]['pf_operator_error'] += .01
        records = {}
        for k, v in b.items():
            records[k] = k + '.json'
            m.io.save(tmp_path / records[k], v)
        m.io.save(tmp_path / 'native_manifest.json', source)
        (tmp_path / 'artificial_source.py').write_text('# invented fixture source\n')
        m.io.save(tmp_path / 'SOURCE.json', dict(files={'artificial_source.py': m.io.sha(tmp_path / 'artificial_source.py')}))
        # The real passive preflight is also tested separately against frozen bytes.
        monkeypatch.setattr(m, 'NATIVE_CONTRACT_SHA', m.io.sha(tmp_path / records['native_contract']))
        c = dict(scope=m.SCOPE, status='UNAPPROVED_SCIENCE_CONTRACT', conditions=m.prior.CONDITIONS,
            compiler=m.prior.profile_dict(), gaussian=m.prior.GAUSSIAN, compiler_phase=m.prior.PHASE,
            accuracy_target=m.TARGET, cost_scope=m.COST_SCOPE, call_limits=m.LIMITS,
            maximum_logical_units=m.MAX_CALLS, candidate_order=[f'{a}:s{s}' for a, s in m.CANDIDATES],
            fresh_output=m.FAMILY + '/<execution-date>/run01', native_SOURCE_commit=m.NATIVE_SOURCE,
            native_contract_sha256=m.NATIVE_CONTRACT_SHA, source_manifest='SOURCE.json',
            source_manifest_sha256=m.io.sha(tmp_path / 'SOURCE.json'), native_source_manifest='native_manifest.json',
            records=records, evaluation_packages={},
            saved_inputs={p: m.io.sha(tmp_path / p) for p in [*records.values(), 'native_manifest.json']},
            **{k: 0 for k in m.FORBIDDEN})
        path = tmp_path / 'contract.json'
        m.io.save(path, c)
        return path, b
    return create


@pytest.mark.parametrize('fault', [None, 'infeasible', 'unresolved', 'identity', 'alias',
    'metrics', 'label', 'coverage', 'DF', 'conditions', 'PF', 'signal'])
def test_artificial_end_to_end_and_STOP(tmp_path, artificial, fault):
    path, b = artificial(fault)
    out = tmp_path / m.FAMILY / '2026-10-26/run01'
    forbidden = Mock(side_effect=AssertionError('NO_MOLECULE_OPERATOR_BUILD_COMPILE_OR_OLD_RUN'))
    with patch.object(m.prior, 'run', forbidden), patch.object(m.reused, 'run', forbidden), \
            patch.object(m, 'source_commit', return_value='artificial-source'):
        if fault not in (None, 'infeasible', 'unresolved'):
            with pytest.raises(ValueError): m.run(tmp_path, path, out)
        else:
            m.run(tmp_path, path, out)
    forbidden.assert_not_called()
    stop = m.io.read(out / 'STOP.json')
    assert stop['retry'] == stop['resume'] == 0 and stop['scientific_units'] <= 119
    journal = [json.loads(x) for x in (out / 'science_calls.jsonl').read_text().splitlines()]
    if fault not in (None, 'infeasible', 'unresolved'):
        assert stop['failed'] and (out / 'failure.json').exists() and not (out / 'result.json').exists()
        fail = next(i for i, x in enumerate(journal) if x['event'] == 'FAIL')
        assert not any(x['event'] == 'attempt' for x in journal[fail + 1:])
    else:
        r = m.io.read(out / 'result.json')
        assert not stop['failed'] and list(r['candidates']) == [f'{a}:s{s}' for a, s in m.CANDIDATES]
        selected = r['candidates']['CDF:s1']
        if fault:
            assert selected['status'] == ('UNRESOLVED' if fault == 'unresolved' else 'INFEASIBLE_REGISTERED_SUFFICIENT_CONDITION')
            assert selected['cost']['totals'] is None and selected['per_shot_costs'] == []
            assert stop['scientific_units'] == 106
            assert r['comparisons']['1']['comparison']['verdict'] == 'INDETERMINATE_MISSING_QUALIFICATION'
            assert r['candidates']['XDF:s1']['status'] == 'NUMERICAL_PILOT_CA_PASS'
        else:
            assert stop['scientific_units'] == 119 and stop['call_counts'] == m.LIMITS
            assert all(x['status'] == 'NUMERICAL_PILOT_CA_PASS' for x in r['candidates'].values())
            for key, v in r['candidates'].items():
                assert v['numerical_energy_assessment']['energy_error_bound_ha'] == pytest.approx(
                    b['identity']['DF_bounds_Ha'][v['arm']] + m.TARGET['beta_total'] / (.1 * 4096))
                assert v['numerical_energy_assessment']['statistical_failure_upper'] <= .01
                assert len(v['per_shot_costs']) == 26 and len(v['rounds']) == 13
                assert v['cost']['shots'] == v['cost']['executions'] == v['cost']['measurement_count']
                assert v['CA_achieved'] is v['guaranteed_actual_shots'] is v['rigorous_enclosure'] is None
                assert not v['guaranteed_total_resource_superiority']
                for row in v['rounds']:
                    assert row['shots'] > 0 and row['statistical_margin'] >= 0
                    assert row['raw_signal_error'] == row['source_combined_complex_error']
                costs = { (x['m'], x['axis']): x for x in v['per_shot_costs'] }
                assert costs[12, 'X']['kernel_occurrences'] == v['s'] * 4096
                n = v['s'] * 4096
                count = 8 if v['arm'] == 'CDF' else 12
                assert costs[12, 'X']['mean']['rz_count'] == n * count + 3
                assert costs[12, 'X']['depth_semantics'] == 'scheduled_serial_sum'
                assert v['cost']['totals']['rz_count'] == sum(
                    p['shots'] * sum(costs[p['m'], a]['mean']['rz_count'] for a in ('X', 'Y')) for p in v['rounds'])
    before = (out / 'STOP.json').read_bytes()
    with pytest.raises(ValueError, match='FRESH'): m.run(tmp_path, path, out)
    assert (out / 'STOP.json').read_bytes() == before


@pytest.mark.parametrize('fault', ['source', 'input', 'budget', 'calls', 'fresh', 'candidate', 'package'])
def test_passive_preflight_failures(tmp_path, artificial, fault):
    path, _ = artificial()
    c = m.io.read(path)
    if fault == 'source': (tmp_path / 'artificial_source.py').write_text('changed')
    if fault == 'input': (tmp_path / 'native.json').write_text('{}')
    if fault == 'budget': c['accuracy_target'] = dict(c['accuracy_target'], alpha_total=.05)
    if fault == 'calls': c['maximum_logical_units'] += 1
    if fault == 'fresh': c['fresh_output'] = 'old-run'
    if fault == 'candidate': c['candidate_order'].reverse()
    if fault == 'package': c['evaluation_packages'] = {'numpy': 'not-a-version'}
    path.write_text(json.dumps(c))
    with patch.object(m, 'load_bundle', side_effect=AssertionError('NO_NUMERIC_JOIN')):
        with pytest.raises(ValueError): m.preflight(tmp_path, path)


def test_unknown_bound_and_numeric_limits_reuse(tmp_path, artificial):
    _, b = artificial()
    packet = m.attach_evidence(b, 'CDF', 1)
    row = copy.deepcopy(packet['rounds'][0])
    target = m.accuracy.AccuracyTarget(**m.TARGET)
    row['evidence_level'] = 'A_FINITE_PROBES'
    assert m.reused.assess_saved_round(target, packet, row)['status'].startswith('UNRESOLVED')
    row = packet['rounds'][0]
    with patch.object(m.reused.det, 'assess_fixed_round', return_value=dict(status='INFEASIBLE_FIXED_SHOTS')):
        assert m.reused.assess_saved_round(target, packet, row)['status'] == 'UNRESOLVED_INTEGER_SHOT_MARGIN'
    with patch.object(m.accuracy, 'aggregate_totals', return_value=dict(
            status='CONDITIONAL_TOTAL_AVAILABLE', totals={k: float(1 << 60) for k in m.accuracy.METRICS})):
        result = m.reused.aggregate_candidate(packet, [], dict(same_accuracy=True), [])
    assert result['status'] == 'UNRESOLVED' and result['cost']['totals'] is None


def test_passive_import_and_fixed_conditions():
    code = "import sys;from pathlib import Path;sys.path.insert(0,'src');from trottertracks import c1_h4_ca_resources as m;m.preflight(Path.cwd(),Path('docs/research/research_c_c1_h4_ca_resources_contract_unapproved_20261026.json'));assert not any(x in sys.modules for x in ['numpy','scipy','qiskit','ffsim'])"
    subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True)
    assert m.TARGET == m.reused.TARGET
    assert m.prior.CONDITIONS['candidate_failure'] == m.TARGET['alpha_total']
    assert m.prior.CONDITIONS['axis_failure'] == m.TARGET['alpha_axis']
    assert m.MAX_CALLS == 119 and m.CANDIDATES == (('CDF', 1), ('CDF', 4), ('XDF', 1), ('XDF', 4))
