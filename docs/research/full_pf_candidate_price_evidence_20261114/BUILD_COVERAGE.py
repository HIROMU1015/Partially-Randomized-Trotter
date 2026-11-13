"""Passive A -> D receipt projection; no S/C values, fit or predictions.

Reuse C's canonical metadata adapter and D's completed receipt trace. Do not
repeat D's native-result audit or load Hamiltonian/private circuit artifacts.
"""
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks import baseline_signal_pipeline as C

SOURCES = dict(
    A=dict(commit='3ae055fa7a696b13320e00ec79a5ba15c4ed182e',
           path='docs/research/h4_full_pf_heff_scan_results_20261114/C_HANDOFF.json'),
    C=dict(commit='3ae2e605dd6efcc63ed3742fe3076a8be3e6afae',
           path='docs/research/rpe_price_evidence_validation_20261113/REPORT.md'),
    D=dict(commit='e7dca5f7301c8633b17a59b6a7161331303aef2d',
           path='docs/research/d_full_pf_full_h_v2_final_receipts_20261113/COVERAGE_V2.json'),
    D_TRACE=dict(commit='e7dca5f7301c8633b17a59b6a7161331303aef2d',
           path='docs/research/d_full_pf_full_h_v2_final_receipts_20261113/RECEIPTS.json'),
    D_INPUT=dict(commit='af284d853890155bacd19e61850c196bd1d5c660',
           path='artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/scientific_provenance.json'))
NATIVE = 'M5_TOL_ONLY_DF_EXPLICIT_GAUSSIAN_DIRECTIONAL_SCALAR_ONCE_V1'


def read(descriptor):
    return json.loads(C.blob(ROOT, descriptor['commit'], descriptor['path']))


def step_source(locator):
    if ':' in locator:
        commit, path = locator.split(':', 1)
        return dict(commit=commit, path=path)
    return dict(commit=SOURCES['A']['commit'],
                path='docs/research/h4_full_pf_heff_scan_results_20261114/' + locator)


def price_metadata(price, axis):
    """Select provenance only, deliberately omit saved RZ and numerical gates."""
    out = {k: deepcopy(price[k]) for k in ('kind', 'cohort', 'n', 'carrier', 'compiler')}
    for k in ('cost_policy', 'reuse_provenance', 'result_sha256', 'trial_files'):
        if k in price:
            out[k] = deepcopy(price[k])
    trace = price.get('receipt_trace', {})
    out['receipt_evidence'] = [dict(kind=e['kind'], source=deepcopy(e['source']),
        native_SOURCE=e['native_SOURCE'], preparation_hash=e['preparation_hash'],
        fingerprint=e['axes'][axis]['fingerprint'], axis_status=e['axes'][axis]['status'])
        for e in trace.get('evidence', []) if axis in e['axes']]
    out['status'] = price['axes'][axis]['status']
    return out


def build():
    a, d, trace, native = (read(SOURCES[k]) for k in ('A', 'D', 'D_TRACE', 'D_INPUT'))
    api = C.published_api(str(ROOT))
    physical_binding = native['inputs']['H4']['binding']
    assert all(C.p.same(v, a['shared']['binding'].get(k)) for k, v in physical_binding.items())
    profile = a['shared']['profile']
    assert profile['compiler'] == native['compiler']['hash']
    assert profile['basis_gates'] == native['compiler']['basis']
    assert (profile['optimization_level'], profile['seed_transpiler'], profile['qiskit_version']) == (
        native['compiler']['optimization_level'], native['compiler']['seed_transpiler'], native['runtime']['qiskit'])
    assert (profile['control'], profile['wrapper'], profile['P']) == (
        'symmetric_directional', 'full_Hadamard_with_measurement', 0)
    assert native['cost_policy'] == 'FULL_COSINE_SINE_HADAMARD_WRAPPER_MEASUREMENT_INCLUDED_P0'
    assert trace['source_checks']['status'] == 'EXACT_METADATA_MATCH'
    assert trace['source_checks']['items']['long_time_schedule_and_scalar'] is True
    weights_path = 'src/trotterlib/h46_m5_plan.py'
    assert C.blob(ROOT, a['scientific_source'], weights_path) == C.blob(ROOT, SOURCES['D']['commit'], weights_path)
    by_id = {r['physical_candidate_id']: (i, r) for i, r in enumerate(d['rows'])}
    assert len(by_id) == len(d['rows'])
    candidates, cells = [], []
    for index, candidate in enumerate(a['shortlist']):
        ctor = candidate['constructor']
        base = read(ctor['base'])
        for key in ctor['base_selector']:
            base = base[key]
        spec = api.D.full_pf_spec(base, formula=ctor['formula'], delta_pf=ctor['delta_pf'],
                                 M=ctor['M'], rank=ctor['rank'])
        config = C.configure(ROOT, spec, api=api)
        assert config['id'] == candidate['configuration_id']
        requests = C.prepare(ROOT, config, target=a['shared']['target'], api=api)['requests']
        assert C.p.same(config['spec']['binding'], a['shared']['binding'])
        assert C.p.same(config['spec']['profile'], profile)
        label = candidate['label']
        candidates.append(dict(label=label, configuration_id=config['id'], constructor=deepcopy(ctor),
            candidate_source=dict(SOURCES['A'], selector=['shortlist', index]),
            interpretation='CURRENT_COMPARISON_CANDIDATE_NOT_COST_WINNER',
            law=deepcopy(config['spec']['law']),
            science_source=a['scientific_source'], preparation_source=a['preparation']))
        for req, row in zip(requests, candidate['rounds'], strict=True):
            physical = row['physical_candidate']; m = row['m']
            assert req['m'] == m and C.p.same(req['physical_candidate'], physical)
            canonical = config['rounds'][m]
            assert canonical['candidate']['id'] == row['canonical_candidate_id']
            assert C.p.same(canonical['actual_delta_pf'], row['actual_delta_pf'])
            assert row['q_PF'] == canonical['q_PF'] == physical['q']
            match = by_id.get(physical['id'])
            if match is not None:
                di, saved = match
                assert C.p.same(saved['physical_candidate'], physical)
                assert saved['binding']['status'] == 'EXACT_METADATA_MATCH'
                locator = dict(SOURCES['D'], selector=['rows', di])
                assert C.p.same(locator, row['price_coverage']['source'])
            for axis in ('X', 'Y'):
                status = match[1][axis] if match else 'OUTSIDE_CHECKED_CATALOG'
                assert status == row['price_coverage'][axis]
                evidence = []
                if match:
                    for pi, price in enumerate(match[1]['prices']):
                        assert price['compiler'] == profile['compiler']
                        if price['axes'][axis]['status'] == 'EXACT_METADATA_MATCH':
                            evidence.append(dict(price_metadata(price, axis),
                                source=dict(SOURCES['D'], selector=['rows', match[0], 'prices', pi])))
                cells.append(dict(label=label, configuration_id=config['id'], m=m, axis=axis,
                    canonical_candidate_id=row['canonical_candidate_id'],
                    identity_and_law_source=dict(generator='C.configure+D.full_pf_spec; canonical B round',
                        constructor_ref=['candidates', index, 'constructor'], round_index=m),
                    physical_candidate=deepcopy(physical),
                    T=deepcopy(row['T']), q_PF=row['q_PF'], q_RPE=row['q_RPE'],
                    requested_delta=deepcopy(ctor['delta_pf']), actual_delta=deepcopy(row['actual_delta_pf']),
                    structure=deepcopy(req['structure']),
                    candidate_source=dict(SOURCES['A'], selector=['shortlist', index, 'rounds', m]),
                    numerical_step_source=step_source(row['eigensystem_source']),
                    receipt_catalog_status=status, direct_evidence=evidence,
                    reuse=('SAVED_NATIVE_DIRECT_AVAILABLE_CONSUMER_LOWERING_NOT_ADOPTED' if evidence else
                           'MISSING_IN_CATALOG' if match else 'OUTSIDE_CHECKED_CATALOG'),
                    saved_lowering=NATIVE, consumer_lowering='NOT_EXPLICIT_IN_A_PROFILE',
                    scalar_policy='ORIGINAL_FULL_H_CONSTANT_ONCE_AT_TOTAL_T_NO_TAIL_IDENTITY_ADDON',
                    expectation='DETERMINISTIC_FULL_WRAPPER' if evidence else 'UNAVAILABLE',
                    model_status='NO_QUALIFIED_FULL_PF_Q_MODEL', model_prediction=None,
                    consumer_price_record_created=False, mean_C=None, qualified_for_full_RPE=False,
                    numerical_model_error=None, N=None, G=None))
    assert len(candidates) == 3 and len(cells) == 78
    assert sum(bool(r['direct_evidence']) for r in cells) == 10
    assert not any(k in sys.modules for k in ('numpy', 'scipy', 'qiskit', 'pyscf'))
    return dict(status='STATIC_METADATA_CONNECTION_NO_PRICE_ARITHMETIC', sources=SOURCES,
        physical_binding=physical_binding, profile=profile, native_cost_policy=native['cost_policy'],
        D_receipt_source_checks_reused=trace['source_checks'],
        D_preparation_trace_reused=trace['private_preparation_check']['H4'],
        A_D_weights_module_bytes_equal=True, candidates=candidates, coverage=cells,
        saved_price_values_copied=False, new_price_calculations=0, new_science_calls=0,
        model_predictions=None, G=None, winner=None, mandatory_stop=True)


if __name__ == '__main__':
    result = build()
    directory = Path(__file__).parent
    with (directory / 'COVERAGE.json').open('x') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    fields = ['label', 'm', 'axis', 'configuration_id', 'physical_id', 'T', 'q_PF',
              'requested_delta_exact', 'actual_delta_exact', 'catalog_status', 'reuse', 'C', 'N', 'G']
    with (directory / 'COVERAGE.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for r in result['coverage']:
            def exact(v):
                return str(v['numerator']) + '/' + str(v['denominator'])
            writer.writerow(dict(label=r['label'], m=r['m'], axis=r['axis'], configuration_id=r['configuration_id'],
                physical_id=r['physical_candidate']['id'], T=r['physical_candidate']['T'], q_PF=r['q_PF'],
                requested_delta_exact=exact(r['requested_delta']), actual_delta_exact=exact(r['actual_delta']),
                catalog_status=r['receipt_catalog_status'], reuse=r['reuse'], C='', N='', G=''))
    print('PASS: 3 configurations, 39 physical rounds / 78 axis cells; 10 saved direct-axis locators; no S/C arithmetic')
