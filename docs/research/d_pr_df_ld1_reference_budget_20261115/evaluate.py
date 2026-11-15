"""Fixed LD1 reference prices and TWO signed PF boundaries; no actual-law claims.

prepare: metadata only. execute: committed inputs/source, create-only run01.
Reuse fixed predict/_exact function bodies, never fit/validate or H4 libraries.
"""
import argparse
import ast
import csv
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import lru_cache
import json
import math
from pathlib import Path
import resource
import subprocess
import time

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
REL = BASE.relative_to(ROOT)
PLAN = '44f4e40f24a027346e686d7044d102e707292374'
SHOTS = [64, 60, 56, 52, 47, 43, 38, 34, 30, 25, 21, 17, 12]
HELPER = dict(commit='d83737b6d5241962ce39c81d39c56af3506c0cdd',
              path='src/trottertracks/price_model_validation.py')
PF = dict(commit='1002f1dfea34de439ca1b704a0192bca473c5387',
          path='docs/research/pf6_allround_cost_20261115/RESULT.json')
COMPARATORS = [('B0I_P7_S2_16h_over_7', 593555352), ('FULL_PF_S2_16h_over_7', 838693580)]


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


@lru_cache(None)
def blob(commit, path):
    return git('show', commit + ':' + path)


@lru_cache(None)
def document(commit, path):
    return json.loads(blob(commit, path))


def selected(ref):
    value = document(ref['commit'], ref['path'])
    for key in ref.get('selector', []):
        value = value[key]
    return value


def enc(value):
    return dict(numerator=value.numerator, denominator=value.denominator)


def frac(value):
    return Fraction(value['numerator'], value['denominator'])


def display(value):
    with localcontext() as ctx:
        ctx.prec = 60
        return format(Decimal(value.numerator) / Decimal(value.denominator), '.9f')


def predictor():
    """Execute unchanged named function ASTs from the fixed Git blob only."""
    tree = ast.parse(blob(HELPER['commit'], HELPER['path']))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('_exact', 'predict')]
    require([n.name for n in nodes] == ['_exact', 'predict'], 'PINNED_PREDICT_FUNCTIONS')
    namespace = {'Fraction': Fraction}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), HELPER['path'], 'exec'), namespace)
    return namespace['predict']


def inputs():
    return dict(schema='LD1_REFERENCE_TOTAL_FIXED_INPUTS_V1',
        plan=dict(commit=PLAN, path='docs/research/d_pr_df_ld1_expectation_construction_20261115/REPORT.md'),
        axes=dict(commit=PLAN, path='docs/research/d_pr_df_ld1_expectation_construction_20261115/AXIS_JOIN.json'),
        predictor=HELPER, PF=PF, shots_per_axis=SHOTS, comparators=[list(v) for v in COMPARATORS],
        output=str(REL / 'run01'),
        scope='26 reference prices, exact weighted total, two signed PF boundaries ONLY',
        source_plan_difference='Second PF comparator added; K_IJ and numerical-band propagation not executed under current narrower instruction.',
        law_target='SAVED_SOURCE_IDEAL_FINITE_RTE_LAW_I',
        formula='C_ref = saved_E_I_S + frozen_residual_reference; reuse m4 full prediction',
        signed_boundary='D_PF=T_PF-T_ref; Delta=T_LD1-T_ref remains UNKNOWN',
        m4_role='INDEPENDENT_HOLDOUT_NOT_REFIT_OR_REPLACED_BY_OBSERVATION',
        frozen_inputs=dict(model='3cac53da1a4a8b2228245593c74c1ff9ec1f5212',
                           structural='9d95d3ea544e1fa1d0e4710c1ca24d8c5776469f',
                           Q3='a82b0f0ec7f0f11d9981ddb2343fcd5378a87503'),
        new_paths=0, H4_builds=0, compiles=0, fit=0, CI=None, formal_E_C=None, G=None, winner=None)


def structural_identity(s, row):
    native_axis = {'X': 'cosine', 'Y': 'sine'}[row['axis']]
    require((s['m'], s['axis'], s['q_PF'], s['R_outer']) ==
            (row['m'], native_axis, row['q'], row['R_outer']), 'STRUCTURAL_IDENTITY')


def check_inputs(data):
    require(data == inputs(), 'FIXED_INPUT_REFERENCE_MISMATCH')
    axes = selected(data['axes'])
    require(axes['shots_per_axis'] == SHOTS and len(axes['rows']) == 26, '26_AXES_FIXED_SHOTS')
    require(axes['target_law'] == 'ideal_finite_RTE_law', 'TARGET_LAW')
    for i, row in enumerate(axes['rows']):
        m, a = divmod(i, 2)
        require((row['m'], row['axis'], row['N_fixed']) == (m, ('X', 'Y')[a], SHOTS[m]), 'AXIS_OR_SHOT')
        prior, q3, s, model = [selected(row[k]) for k in ('prior_axis', 'Q3_row', 'structural', 'model')]
        require(all((x['m'], x['axis'], x['physical_id']) == (m, row['axis'], row['physical_id']) for x in (prior, q3)), 'PHYSICAL_IDENTITY')
        structural_identity(s, row)
        require(prior['original_canonical_id'] == q3['original_candidate_id'] and prior['total_candidate_id'] == q3['total_candidate_id'], 'CANONICAL_IDENTITY')
        require(q3['N'] == SHOTS[m], 'Q3_SHOTS')
        require(s['quantity'] == 'FULL_WRAPPER_MEAN_STRUCTURAL_RZ_COUNT' and s['expectation'] == 'IDEAL_FINITE_K_SOURCE_LAW', 'STRUCTURAL_QUANTITY_LAW')
        require(s['law_binding']['definition']['model'] == 'ideal_finite_RTE_law', 'LAW_BINDING')
        require(type(s['mean_S']) is float and math.isfinite(s['mean_S']), 'FINITE_SAVED_BINARY64')
        require(model['axis'] == row['axis'] and model['features'] == ['one', 'q', 'qR'] and model['calibration_rounds'] == [0, 3, 6], 'FROZEN_MODEL')
        require(row['model']['commit'] == data['frozen_inputs']['model'], 'FROZEN_SOURCE')
        require(row['finite_role'] == ('INDEPENDENT_HOLDOUT_ONLY' if m == 4 else 'CALIBRATION' if m in (0, 3, 6) else 'NO_MATCHED_COMPILED_COHORT'), 'FINITE_ROLE')
    pf = selected(data['PF'])
    for label, expected in COMPARATORS:
        matches = [i for i, r in enumerate(pf['rows']) if r['label'] == label]
        require(len(matches) == 1 and pf['rows'][matches[0]]['total'] == expected, 'PF_COMPARATOR_SOURCE')
    return axes


def price(row, s, model, frozen, predict):
    structural = Fraction.from_float(s['mean_S'])
    m, axis = row['m'], row['axis']
    require(model['axis'] == axis, 'MODEL_AXIS')
    if m == 4:
        p = model['m4_prediction']
        require((p['m'], p['axis'], p['physical_id'], p['q'], p['R_outer']) ==
                (m, axis, row['physical_id'], row['q'], row['R_outer']), 'M4_PREDICTION_IDENTITY')
        require(p['ideal_structural_source'] == row['structural'] and
                p['ideal_structural']['binary64_hex'] == s['mean_S'].hex() and
                frac(p['ideal_structural']['exact_binary64_value']) == structural, 'M4_STRUCTURAL_SOURCE')
        residual, result = frac(p['residual_prediction']), frac(p['predicted_RZ'])
        require(result == structural + residual, 'M4_FROZEN_DECOMPOSITION')
        reuse = 'FROZEN_M4_FULL_PREDICTION'
    elif m in (0, 3, 6):
        means = [x for x in frozen['cell_means'] if (x['m'], x['axis']) == (m, axis)]
        require(len(means) == 1, 'CALIBRATION_MEAN')
        require((means[0]['q'], means[0]['R'], means[0]['n_paths']) == (row['q'], row['R_outer'], 4), 'CALIBRATION_IDENTITY')
        residual = frac(means[0]['mean_residual'])
        result, reuse = structural + residual, 'SAVED_CALIBRATION_RESIDUAL_REFERENCE'
    else:
        residual = predict(model, dict(one=1, q=row['q'], qR=row['q'] * row['R_outer']))
        result, reuse = structural + residual, 'FROZEN_FEATURE_EVALUATION'
    require(result >= 0, 'NEGATIVE_REFERENCE_PRICE_NO_CLAMP')
    return structural, residual, result, reuse


def aggregate(rows, comparators):
    total = sum((frac(r['weighted_term']) for r in rows), Fraction())
    boundaries = [dict(label=label, T_PF=value, T_ref=enc(total), D_PF=enc(Fraction(value)-total),
                       D_PF_display=display(Fraction(value)-total), Delta=None,
                       rank_status='UNRESOLVED_DELTA_NOT_BOUNDED',
                       condition='LD1 cheaper iff Delta<D_PF; tie iff equal; PF cheaper iff Delta>D_PF')
                  for label, value in comparators]
    return total, boundaries


def calculate(data, rows):
    axes = check_inputs(data)
    predict = predictor()
    for i, row in enumerate(axes['rows']):
        model, s = selected(row['model']), selected(row['structural'])
        frozen = document(row['model']['commit'], row['model']['path'])
        structural, residual, result, reused = price(row, s, model, frozen, predict)
        refs = {k: row[k] for k in ('Q3_row', 'prior_axis', 'structural', 'model', 'finite_evidence')}
        refs['axis_join'] = dict(data['axes'], selector=['rows', i])
        refs['residual'] = dict(row['model'], selector=['cell_means', next(k for k, v in enumerate(frozen['cell_means']) if (v['m'], v['axis']) == (row['m'], row['axis']))]) if row['m'] in (0, 3, 6) else dict(row['model'], selector=['models', row['axis'], 'm4_prediction', 'residual_prediction'] if row['m'] == 4 else ['models', row['axis'], 'coefficients'])
        rows.append(dict(m=row['m'], axis=row['axis'], physical_id=row['physical_id'], q=row['q'], R_outer=row['R_outer'],
                         N=row['N_fixed'], saved_S=s['mean_S'], saved_S_hex=s['mean_S'].hex(),
                         structural_exact=enc(structural), residual_reference=enc(residual), C_ref=enc(result),
                         C_ref_display=display(result), weighted_term=enc(row['N_fixed'] * result),
                         weighted_display=display(row['N_fixed'] * result), reuse=reused, finite_role=row['finite_role'],
                         model_application='CALIBRATION' if row['m'] in (0, 3, 6) else 'INDEPENDENT_M4_FORECAST' if row['m'] == 4 else 'UNVALIDATED_OUTSIDE_CALIBRATION',
                         source=refs, qualified_E_C=None))
    total, boundaries = aggregate(rows, data['comparators'])
    return dict(schema='LD1_FIXED_SHOT_MODEL_REFERENCE_AND_SIGNED_BOUNDARIES_V1',
                status='REFERENCE_AND_BOUNDARIES_COMPLETE_RANK_UNRESOLVED', rows=rows,
                T_ref=enc(total), T_ref_display=display(total), boundaries=boundaries,
                quantity='MODEL_REFERENCE_TOTAL_RZ_AT_FIXED_Q3_SHOTS_NOT_FORMAL_EXPECTED_G',
                target_law=data['law_target'], definition='C_ref=saved_E_I_S+frozen_residual_reference',
                counts=dict(axis_references=26,full_prices_reused=2,new_full_prices=24,
                            residual_references_reused=8,new_residual_feature_evaluations=18,
                            weighted_terms=26,new_total=1,new_signed_boundaries=2,PF_totals_reused=2),
                Delta=None, Delta_bound=None, ranking=None, formal_E_C=None, CI=None, G=None, winner=None,
                fit_calls=0, new_paths=0, H4_builds=0, compiles=0, F0_reruns=0, F0_budget_evaluated=False,
                structural_enclosure_propagated=False, model_verdict='NOT_ASSESSED_NO_ADOPTED_TOLERANCE', mandatory_stop=True)


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def execute():
    source = git('rev-parse', 'HEAD').decode().strip()
    for path in (REL / 'evaluate.py', REL / 'INPUTS.json'):
        require((ROOT / path).read_bytes() == blob(source, str(path)), 'COMMITTED_EXECUTION_SOURCE_INPUT')
    data = json.loads((BASE / 'INPUTS.json').read_text())
    out = ROOT / data['output']
    out.mkdir(exist_ok=False)
    start, rows = time.perf_counter(), []
    print('BEGIN fixed saved-input join / 26 reference prices / two PF boundaries', flush=True)
    try:
        result = calculate(data, rows)
        result.update(execution_source=source, inputs=dict(commit=source, path=str(REL / 'INPUTS.json')),
                      predictor=HELPER, PF_source=PF)
        print('END exact arithmetic: 26 weighted terms, one total, two signed boundaries', flush=True)
        write(out / 'RESULT.json', result)
        with (out / 'AXIS_TERMS.csv').open('x', newline='') as stream:
            columns = ['m', 'axis', 'physical_id', 'N', 'C_ref_numerator', 'C_ref_denominator', 'weighted_numerator', 'weighted_denominator', 'C_ref_display', 'weighted_display', 'reuse']
            writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
            for r in rows:
                writer.writerow({**{k: r[k] for k in ['m', 'axis', 'physical_id', 'N', 'C_ref_display', 'weighted_display', 'reuse']},
                                 'C_ref_numerator': r['C_ref']['numerator'], 'C_ref_denominator': r['C_ref']['denominator'],
                                 'weighted_numerator': r['weighted_term']['numerator'], 'weighted_denominator': r['weighted_term']['denominator']})
        usage = resource.getrusage(resource.RUSAGE_SELF)
        write(out / 'EXECUTION.json', dict(execution_source=source, wall_seconds=time.perf_counter()-start,
              user_seconds=usage.ru_utime, system_seconds=usage.ru_stime, max_RSS_KiB=usage.ru_maxrss,
              status='COMPLETE', exit_code=0, science_retry=0, mandatory_stop=True))
        print(json.dumps(dict(T_ref=result['T_ref'], T_ref_display=result['T_ref_display'],
                              boundaries=result['boundaries'], mandatory_stop=True)), flush=True)
    except Exception as error:
        write(out / 'STOP.json', dict(execution_source=source, error=repr(error), partial_rows=rows,
              wall_seconds=time.perf_counter()-start, science_retry=0, mandatory_stop=True))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'execute'])
    args = parser.parse_args()
    if args.mode == 'prepare':
        data = inputs(); check_inputs(data)
        write(BASE / 'INPUTS.json', data)
        print('26 metadata rows accepted; no prediction, weighted sum or boundary calculated.')
    else:
        execute()
