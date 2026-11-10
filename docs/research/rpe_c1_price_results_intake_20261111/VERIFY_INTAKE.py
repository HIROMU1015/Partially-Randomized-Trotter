"""Technical checks of the C1 price intake. Reads Git JSON and saved outputs only.

No sampling, compile, S count, fit, prediction, residual, N or G evaluation.
Run: <B venv python> -B docs/research/rpe_c1_price_results_intake_20261111/VERIFY_INTAKE.py
"""
from collections import Counter
from contextlib import ExitStack
import json
import math
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'src')]
from trottertracks import rpe_r_price_intake as ci  # noqa: E402
from trottertracks import h4_df10_price_provider as h4, h6_df19_price_provider as h6  # noqa: E402

p, r, b = ci.p, ci.r, ci.b
PUB = '2e6a0cdf5a647058cc7afeeb5741bfafa24c974b'
FROZEN = '7eeabf0ae482ee610d667e28470692d209f2bb21'
RUN = 'docs/research/r_price_phase1_results_20261111/run01'
C_SOURCE_FILES = ('src/trottertracks/r_price_phase1.py', 'src/trottertracks/r_price_phase1_runner.py',
                  'src/trottertracks/r_price_phase1_science.py', 'scripts/run_r_price_phase1.py')


def blob(commit, path):
    try:
        return subprocess.check_output(['git', 'show', commit+':'+path], cwd=ROOT, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        return None


def load(name):
    return json.loads((HERE/name).read_text())


def main():
    checks = {}
    intake_doc, coverage, readiness = load('INTAKE.json'), load('COVERAGE.json'), load('READINESS.json')
    prices = intake_doc['prices']

    # Publication, freeze and SOURCE provenance.
    subprocess.check_call(['git', 'merge-base', '--is-ancestor', FROZEN, PUB], cwd=ROOT)
    assert blob(FROZEN, RUN+'/VALIDATION.json') is None and blob(FROZEN, RUN+'/RARE_DIAGNOSTICS.json') is None
    for name in ('FROZEN_MODELS.json', 'PLAN.json', 'CALIBRATION.json', 'CALIBRATION_EVALUATION.json'):
        assert blob(FROZEN, RUN+'/'+name) == blob(PUB, RUN+'/'+name), name
    assert blob(PUB, RUN+'/PLAN.json') == blob(ci.C_PLAN_COMMIT, ci.C_PLAN['path'])
    for name in C_SOURCE_FILES:
        assert blob(PUB, name) == blob(ci.C_SOURCE, name), name
    frozen = json.loads(blob(PUB, RUN+'/FROZEN_MODELS.json'))
    validation = json.loads(blob(PUB, RUN+'/VALIDATION.json'))
    calibration = json.loads(blob(PUB, RUN+'/CALIBRATION_EVALUATION.json'))
    stop = json.loads(blob(PUB, RUN+'/STOP.json'))
    assert validation['frozen_model_commit'] == FROZEN and intake_doc['frozen_model_commit'] == FROZEN
    assert frozen['validation_prices_observed'] is False and frozen['validation_refit'] is False
    assert validation['validation_refit'] is False and stop['status'] == 'PHASE1_COMPLETE_MANDATORY_STOP'
    checks['freeze_before_validation'] = dict(frozen_commit=FROZEN, publication=PUB, frozen_is_ancestor=True,
        validation_absent_at_freeze=True, frozen_plan_calibration_bytes_unchanged=True, refit=False)
    checks['source'] = dict(plan_bytes_equal_fixed_plan=True, execution_source=ci.C_SOURCE,
        science_files_equal_execution_source=list(C_SOURCE_FILES))

    # 32 records, roles and scenario kinds.
    assert intake_doc['status'] == 'C1_PRICE_RECORDS_ONLY' and intake_doc['completion'] == 'PHASE1_COMPLETE_RECORDED'
    assert len(prices) == 32 == len(calibration['prices']) + len(validation['prices'])
    kinds = Counter(v['c1']['price_kind'] for v in prices)
    assert kinds == {'SAVED_CALIBRATION_DIRECT': 8, 'NEW_CALIBRATION_DIRECT': 4, 'INDEPENDENT_VALIDATION_DIRECT': 4,
                     'MODEL_PREDICTION_AT_CALIBRATION_POINT': 12, 'MODEL_PREDICTION_AT_VALIDATION_POINT': 4}
    stage = Counter((v['c1']['stage'], v['c1']['role']) for v in prices)
    assert stage == {('CALIBRATION', 'SAVED_CALIBRATION'): 16, ('CALIBRATION', 'NEW_CALIBRATION'): 8,
                     ('VALIDATION', 'INDEPENDENT_VALIDATION'): 8}
    originals = {(v['identity']['candidate']['id'], v['axis'], v['classification']): v
                 for v in calibration['prices'] + validation['prices']}
    plan_cells = {c['id']: c for c in json.loads(blob(PUB, RUN+'/PLAN.json'))['cells']}
    times, allocations = [], {}
    for v in prices:
        cell = plan_cells[v['c1']['cell_id']]
        c = v['identity']['candidate']
        o = originals[c['id'], v['axis'], v['classification']]
        # Copy fidelity: every C field except the documented B binding/label additions.
        assert all(p.same(v[k], o[k]) for k in o if k not in ('law_binding', 'cohort'))
        assert v['c1_original_law_binding'] == o['law_binding'] and v['cohort'] == 'C1:'+o['cohort']
        assert c['id'] == cell['candidate_id'] and v['n_paths'] == cell['n'] and v['source_cohort'].endswith(cell['cohort'])
        assert v['identity']['profile'] == p.PROFILE and v['outcome_convention'] == 'P0_minus_P1'
        assert v['expectation'] == 'FINITE_COHORT_ONLY' and v['normalization_is_price_multiplier'] is False
        assert v['law_correspondence']['ideal_sampler_distance_qualified'] is False
        assert v['law_binding']['definition'] == b.LAW and v['c1_original_law_binding']['definition'] == ci.C_LAW
        assert v['c1']['qualified_for_candidate'] is False and v['c1']['classical_paths_are_quantum_shots'] is False
        assert all(v['c1']['uncertainty'][k] is None for k in ci.UNQUALIFIED)
        if v['classification'] == 'MODEL_PREDICTION':
            assert v['model']['qualified_for_candidate'] is False and v['model']['domain'] == c
        # Actual evolution time is candidate.T; binding.T is the old task reference.
        m = v['identity']['m']; base = v['identity']['configuration']['base_time']
        assert c['T'] == math.ldexp(base, m) and c['delta'] == c['T']/c['q'] and v['identity']['q_PF'] == c['q']
        times.append((v['c1']['cell_id'], c['T'], c['q'], c['delta'], v['identity']['binding'].get('T')))
        s = v['identity']['structure']
        assert sum(s['allocation']) == c['R_outer'] and s['event_count'] == c['q']*c['R_outer']
        if c['system'] == 'H6':
            assert len(s['allocation']) == 11 and min(s['allocation']) >= 1
        allocations[v['c1']['cell_id']] = s['allocation']
    for ev in calibration['evaluations'] + validation['evaluations']:
        for v in prices:
            if v['c1']['cell_id'] == ev['cell_id'] and v['c1']['stage'] == 'VALIDATION':
                assert v['c1']['uncertainty']['validation_summaries'] == ev['summaries']
    assert allocations['H6_q1_R16'] == [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]
    assert allocations['H6_q1_R32'] == [3, 3, 3, 3, 2, 5, 2, 3, 2, 3, 3]
    stored_rmse = {e['cell_id']: sorted({s['metrics']['relative_RMSE'] for s in e['summaries']}) for e in validation['evaluations']}
    checks['records'] = dict(count=32, price_kinds=dict(kinds), finite_cohort_only=32, model_qualified=0,
        copy_fidelity='all C fields equal except B law_binding and C1: cohort label',
        time_fields=sorted(set(times)), H6_allocations={k: v for k, v in allocations.items() if k.startswith('H6')},
        stored_validation_relative_RMSE_transcribed=stored_rmse)
    assert len(intake_doc['rare_diagnostics']) == 4 and not any('FORCED' in v['cohort'] for v in prices)
    assert all(v['price_record'] is False and v['pooled_with_cohort'] is False for v in intake_doc['rare_diagnostics'])

    # Same result when re-read with every numerical entry disabled.
    with ExitStack() as stack:
        intake = ci.C1PriceIntake(ROOT)
        for obj, name in [(p, 'evaluate'), (r, 'evaluate_menu'), (p.accounting, 'aggregate_supplied_axes'),
                          (b.nmodel, 'analyze'), (h6, '_affine'), (h6.H6AffinePriceProvider, 'predict_affine'),
                          (h4.H4PriceProvider, 'predict_path')] + [(intake.c, n) for n in
                          ('calibrate', 'evaluate_cell', 'price_records', 'saved_cell', 'helpers', 'make_plan')]:
            stack.enter_context(patch.object(obj, name, side_effect=AssertionError('numerical entry: '+name)))
        again = intake.read_published(PUB, RUN)
    assert json.loads(json.dumps(again)) == intake_doc and 'trottertracks.r_price_phase1_science' not in sys.modules
    assert readiness == json.loads(json.dumps(intake.readiness()))
    checks['rerun_with_numerical_entries_disabled'] = 'IDENTICAL_INTAKE'

    # Published-fixture mismatch detection: edited in-memory copies must be rejected.
    files = {k: (None if blob(PUB, RUN+'/'+f) is None else json.loads(blob(PUB, RUN+'/'+f))) for k, f in ci.FILES.items()}
    publication = dict(commit=PUB, run_dir=RUN)
    edits = {
        'law_definition': lambda f: f['CALIBRATION']['prices'][0]['law_binding']['definition'].update(sampling='IDEAL_FINITE_RTE_MEAN'),
        'compiler': lambda f: f['CALIBRATION']['prices'][0]['identity']['profile'].update(compiler='OTHER'),
        'candidate_R': lambda f: f['VALIDATION']['prices'][0]['identity']['candidate'].update(R_outer=16),
        'cohort': lambda f: f['VALIDATION']['prices'][0].update(cohort='DIRECT:OTHER'),
        'n_paths': lambda f: f['VALIDATION']['prices'][0].update(n_paths=16),
        'value': lambda f: f['VALIDATION']['prices'][0]['mean'].update(rz_count=1.0),
        'validation_in_calibration': lambda f: f['CALIBRATION']['prices'].append(json.loads(json.dumps(f['VALIDATION']['prices'][0]))),
        'model_qualified': lambda f: next(x for x in f['VALIDATION']['prices'] if x['model'])['model'].update(qualified_for_candidate=True),
        'execution_source': lambda f: f['VALIDATION']['prices'][0]['source'].update(execution_source='0'*40),
        'frozen_commit': lambda f: f['VALIDATION'].update(frozen_model_commit='0'*40),
    }
    rejected = {}
    for name, edit in edits.items():
        copy = json.loads(json.dumps(files)); edit(copy)
        try:
            intake.accept(copy, publication=publication, frozen_committed=files['frozen'])
        except ValueError as exc:
            rejected[name] = str(exc)
    assert set(rejected) == set(edits), set(edits) - set(rejected)
    assert intake.accept(files, publication=publication, frozen_committed=files['frozen'])['completion'] == 'PHASE1_COMPLETE_RECORDED'
    checks['published_fixture_mismatch_rejected'] = rejected

    # Coverage: only exact C1 rows get C1 scenarios; N/A0 unchanged; G unevaluated.
    a0 = p.SavedArtifacts(ROOT).read(dict(commit='d5d9823e17ecf78a82216fc02b7fcc13a21bb2cb',
        path='docs/research/rpe_normalization_results_intake_20261111/INTAKE.json'))
    a0status = {v['candidate_id']: v['normalization_status'] for v in a0['rows']}
    c1ids = {c['candidate_id']: c for c in readiness['cells']}
    rows = [v for s in coverage['systems'].values() for v in s['rows']]
    for v in rows:
        c1s = [x for x in v['price_scenarios'] if x['cohort'].startswith('C1:')]
        assert v['G'] == 'NOT_EVALUATED' and v['normalization']['used_as_N'] is False
        assert v['normalization']['status'] == a0status.get(v['candidate_id'])
        if v['candidate_id'] in c1ids:
            assert len(c1s) == 4 and all(x['value_available'] for x in c1s)
        else:
            assert not c1s
    new = [v for v in rows if v['candidate_id'] in c1ids and c1ids[v['candidate_id']]['role'] in ('NEW_CALIBRATION', 'INDEPENDENT_VALIDATION')]
    assert len(new) == 4 and all(v['N_status'] == 'MISSING' and 'MISSING_N' in v['reasons'] for v in new)
    long = {(v['candidate_id']) for v in rows if v['m'] == 12 and v['R_outer'] in (8192, 2048)}
    assert all(v['C_status'] == 'NOT_ACQUIRED' for v in rows if v['candidate_id'] in long)
    assert sum(v['normalization']['in_A0'] for v in rows) == 24 and coverage['costs'] is None
    a_ids = set()
    for s, v in coverage['systems'].items():
        a_ids |= {x['candidate_id'] for x in p.SavedArtifacts(ROOT).read(v['A_menu'])['rows']}
    assert len(a_ids) == 86 and a_ids <= {v['candidate_id'] for v in rows}
    inputs = b.saved.load_inputs(ROOT, p.SavedArtifacts(ROOT))
    hand = p.SavedArtifacts(ROOT).read(b.saved.H4_PUBLISHED_N)
    m10 = next(n for n in hand['shots'] if n['identity']['m'] == 10 and n['identity']['candidate']['method'] == 'PR')
    assert str(m10['plan']['N_X']) == '1070910265913026585642'
    assert next(n for n in inputs['H4']['shots'] if p.key(n['identity']) == p.key(m10['identity']))['plan'] == m10['plan']
    checks['coverage'] = dict(rows=len(rows), C1_rows_with_4_scenarios=sum(v['candidate_id'] in c1ids for v in rows),
        C1_new_R_rows_missing_N=len(new), long_q_large_R_rows_without_price=len(long), A0_rows=24,
        saved_N_records={s: v['saved_N_records'] for s, v in coverage['systems'].items()},
        R_upper_bound={s: v['R_upper_bound'] for s, v in coverage['systems'].items()},
        range_complete={s: v['range_complete'] for s, v in coverage['systems'].items()},
        H4_m10_PR_N_X_unchanged='1070910265913026585642', G='NOT_EVALUATED', costs=None)
    print(json.dumps(dict(status='PASS', checks=checks, new_scientific_computation=0), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
