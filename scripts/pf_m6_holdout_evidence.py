"""Frozen m6 metadata gate and exact comparison of supplied, published receipts.

No circuit generation, compilation, fit, prediction or stage progression.
Acquisition remains in the existing deterministic PF stage callers.
"""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import argparse
import csv
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import pf_price_stages as S
import run_full_pf_shortlist_prices as F
import run_truncated_p7_prices as T

FROZEN = dict(commit='12372fa6b53c878dc5dd124e036572ba0cd82061',
    path='docs/research/pf_stage1_model_freeze_20261115/run01/FROZEN_MODELS.json')
FROZEN_SHA256 = '845c705082565a2b44d198bddb3034497620d991a00604454394ae5acda9651a'
INDEX = dict(commit='b190bd815c886446291fac8b281c3c7df7343372',
    path='docs/research/d_pf_stage1_calibration_20261115/C_HANDOFF_INDEX.json')
ROUTES = dict(FULL_PF=F, TRUNCATED_PF=T)


def exact(value):
    if (not isinstance(value, dict) or type(value.get('numerator')) is not int
            or type(value.get('denominator')) is not int or value['denominator'] <= 0):
        raise ValueError('Exact integer numerator and positive denominator required')
    return Fraction(value['numerator'], value['denominator'])


def enc(value):
    return dict(numerator=value.numerator, denominator=value.denominator)


def validate_predictions(models, plan):
    expected = {S.key(r['model_group']): (i, r) for i, r in enumerate(plan) if r['m'] == 6}
    if len(expected) != 12 or set(models) != set(expected):
        raise ValueError('Exactly twelve fixed m6 model groups required')
    rows = []
    for group_id, (i, cell) in sorted(expected.items()):
        model = models[group_id]; pred = model['m6_prediction']
        if (model.get('model_group_id') != group_id or model.get('model_group') != cell['model_group']
                or model.get('formula') != cell['formula'] or model.get('status') != 'CALIBRATED_UNVALIDATED'
                or model.get('qualified_for_candidate') is not False
                or model['domain']['route'] != cell['route'] or model['domain']['axis'] != cell['axis']
                or pred['configuration_id'] != cell['configuration_id']
                or pred['physical_candidate'] != cell['physical_candidate'] or pred['axis'] != cell['axis']
                or pred['q'] != cell['physical_candidate']['q'] or pred['features'] != cell['features']
                or pred['m'] != 6 or pred['observed_RZ'] is not None or pred['holdout_pass'] is not None
                or pred['independent_holdout_status'] != 'NOT_EVALUATED'
                or pred['cell_source'] != dict(S.P7_CELL_PLAN, selector=[i])):
            raise ValueError('Frozen m6 identity/actual width/q/axis/SOURCE differs: ' + group_id)
        if exact(pred['predicted_RZ']) <= 0:
            raise ValueError('Positive frozen prediction required')
        rows.append(dict(model_group=group_id, route=cell['route'], configuration_id=cell['configuration_id'],
            physical_candidate=deepcopy(cell['physical_candidate']), width_over_h=cell['width_over_h'],
            axis=cell['axis'], q=pred['q'], predicted_RZ=deepcopy(pred['predicted_RZ']),
            prediction_source=dict(FROZEN, selector=['models', group_id, 'm6_prediction', 'predicted_RZ'])))
    return rows


def preopen(root=ROOT):
    """No m6 observations are read here. Call before any duplicate inventory or acquisition."""
    data = S.read_pinned(root, FROZEN)
    if hashlib.sha256(data).hexdigest() != FROZEN_SHA256:
        raise ValueError('Frozen JSON SHA-256 differs')
    doc = json.loads(data)
    if doc['index_source'] != INDEX or doc['comparison_policy'] != S.COMPARISON_POLICY or doc['holdout_observation_reads'] != 0:
        raise ValueError('Frozen Stage1 input/policy/blinding metadata differs')
    rows = validate_predictions(doc['models'], S.plan_rows(root))
    index = json.loads(S.read_pinned(root, INDEX)); readers = {}; snapshots = {}
    for route, module in ROUTES.items():
        spec = module.stage_spec(); cal, _ = S.paths(spec, 'CALIBRATION')
        snapshots[route] = S.committed(root, cal, S.STAGE_FILES + S.BOUND_FILES, commit=INDEX['commit'])
        inputs = json.loads((Path(root) / cal / 'C_INPUTS.json').read_text())
        for key, group in inputs.items():
            loc = index['groups'][key]['C_inputs_source']
            pinned = json.loads(S.read_pinned(root, loc))
            for selector in loc['selector']: pinned = pinned[selector]
            if group != pinned: raise ValueError('Stage1 C_INPUTS differ from the fixed input index')
        readers[route] = S.frozen_models(root, FROZEN, inputs)
    return dict(status='FROZEN_PREDICTIONS_BOUND_BEFORE_M6_OBSERVATION', frozen=FROZEN,
        sha256=FROZEN_SHA256, input_index=INDEX, policy=deepcopy(S.COMPARISON_POLICY),
        reader_checks=readers, calibration_snapshots=snapshots, rows=rows, m6_observation_reads=0)


def judge_row(row, observation=None, *, problem=None):
    result = dict(deepcopy(row), observed_RZ=None, relative_error=None, decision='UNKNOWN', reason=problem, receipt=None)
    if problem or observation is None:
        result['reason'] = problem or 'No admissible independent m6 receipt'
        return result
    count = observation['observed_RZ']; result['receipt'] = deepcopy(observation.get('receipt'))
    if type(count) is not int or count <= 0:
        result['reason'] = 'Positive integer compiled RZ count required'
        return result
    relative = abs(exact(row['predicted_RZ']) - count) / count
    result.update(observed_RZ=count, relative_error=enc(relative),
                  decision='PASS' if relative <= Fraction(1, 50) else 'FAIL', reason=None)
    return result


def supplied_receipts(root, route, publication, models):
    """Existing stage publication only. C guard is an implementation gate, not the model verdict."""
    spec = ROUTES[route].stage_spec(); rel, _ = S.paths(spec, 'HOLDOUT')
    S.committed(root, rel, S.STAGE_FILES + S.BOUND_FILES, commit=publication)
    folder = Path(root) / rel; read = lambda name: json.loads((folder / name).read_text())
    result, guard, prices = read('RESULT.json'), read('GUARD_CHECK.json'), read('PRICES.json')
    if (result['status'] != 'HOLDOUT_PRICES_COMPLETE' or result['stage'] != 'HOLDOUT'
            or result['route'] != route or result['calibration_after'] != result['holdout_gate']['calibration']
            or result['holdout_gate']['frozen_models']['sha256'] != FROZEN_SHA256
            or len(result['results']) != 3 or any(not r['ok'] for r in result['results'])
            or len(prices) != 6 or sum(len(c['guard']['checked']) for c in guard['checks']) != 6
            or guard['comparison_policy'] != S.COMPARISON_POLICY):
        raise ValueError('Incomplete/inconsistent independent holdout stage/guard: ' + route)
    observations = {}
    for i, price in enumerate(prices):
        physical = price['physical_context']['physical_candidate']
        keys = [k for k, m in models.items() if m['domain']['route'] == route
                and m['m6_prediction']['configuration_id'] == price['configuration_id']
                and m['m6_prediction']['axis'] == price['axis']]
        if len(keys) != 1: raise ValueError('Ambiguous route/configuration/axis receipt')
        key = keys[0]; model = models[key]
        if (key in observations or price['m'] != 6 or physical != model['m6_prediction']['physical_candidate']
                or price['price_profile'] != model['domain']['profile']
                or price['classification'] != 'DIRECT_COMPILED_DETERMINISTIC' or price['expectation'] != 'DETERMINISTIC'
                or price['diagnostic']['stage'] != 'HOLDOUT' or price['cohort'] != spec['cohort']):
            raise ValueError('Receipt physical/axis/scalar/profile/domain differs: ' + key)
        observations[key] = dict(observed_RZ=price['mean']['rz_count'], receipt=dict(commit=publication,
            path=rel + '/PRICES.json', selector=[i], physical_id=physical['id'], axis=price['axis'],
            execution_source=result['source'], sidecars=dict(commit=publication, path=rel + '/SIDECARS.json'),
            guard=dict(commit=publication, path=rel + '/GUARD_CHECK.json')))
    return observations


def evaluate(root, publications):
    gate = preopen(root); models = json.loads(S.read_pinned(root, FROZEN))['models']
    observations = {}; problems = {}
    for route in ROUTES:
        if route not in publications:
            problems[route] = 'Stage not acquired/published; no substitution'
            continue
        try: observations.update(supplied_receipts(root, route, publications[route], models))
        except Exception as exc: problems[route] = type(exc).__name__ + ': ' + str(exc)
    rows = [judge_row(row, observations.get(row['model_group']), problem=problems.get(row['route'])) for row in gate['rows']]
    counts = {kind: sum(r['decision'] == kind for r in rows) for kind in ('PASS', 'FAIL', 'UNKNOWN')}
    verdict = 'FAIL' if counts['FAIL'] else 'UNKNOWN' if counts['UNKNOWN'] else 'PASS'
    return dict(status='M6_INDIVIDUAL_HOLDOUT_JUDGED', frozen=FROZEN, sha256=FROZEN_SHA256,
        policy=gate['policy'], rows=rows, counts=counts, campaign=verdict,
        campaign_pass=None if verdict == 'UNKNOWN' else verdict == 'PASS', publications=publications,
        analysis_source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        fit_calls=0, prediction_recalculations=0, long_q_extrapolation_qualified=False, N=None, G=None, mandatory_stop=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preopen-output', type=Path)
    p.add_argument('--evaluate-publications', type=Path, help='JSON route -> published holdout commit; no acquisition')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.preopen_output:
        with args.preopen_output.open('x') as f: json.dump(preopen(), f, indent=2); f.write('\n')
        return
    if not args.evaluate_publications or not args.output: p.error('preopen-output or evaluate-publications/output required')
    result = evaluate(ROOT, json.loads(args.evaluate_publications.read_text()))
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'RESULT.json').open('x') as f: json.dump(result, f, indent=2); f.write('\n')
    with (args.output / 'HOLDOUT.csv').open('x', newline='') as f:
        writer = csv.writer(f);writer.writerow(['model_group','physical_id','axis','q','predicted_numerator','predicted_denominator',
            'observed_RZ','relative_error_numerator','relative_error_denominator','decision','reason','receipt'])
        for r in result['rows']:
            e=r['relative_error'] or {};writer.writerow([r['model_group'],r['physical_candidate']['id'],r['axis'],r['q'],
                r['predicted_RZ']['numerator'],r['predicted_RZ']['denominator'],r['observed_RZ'],e.get('numerator'),
                e.get('denominator'),r['decision'],r['reason'],json.dumps(r['receipt'])])
    print(json.dumps(dict(campaign=result['campaign'],counts=result['counts'],output=str(args.output))))


if __name__ == '__main__': main()
