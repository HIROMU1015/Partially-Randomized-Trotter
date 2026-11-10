"""Approved fixed H4 DF10 m3 RZ connection check, three saved price cases only."""
import argparse
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks.h4_df10_rpe_cost_connection import account_subset

BASE = '305439c01b6e9d8ec264045585ad77b36630275c'
INPUTS = ('docs/research/h4_df10_rpe_cost_connection_20261110/PREPARED_INPUTS.json',
          'docs/research/h4_df10_rpe_d1_20261110/result.json',
          'docs/research/h4_df10_rpe_preparation_20261110/SAVED_POINTS.json')
MODULES = ('src/trottertracks/h4_df10_rpe_cost_connection.py',
           'src/trottertracks/h4_df10_rpe_connection.py',
           'src/trottertracks/c1_stage3_accuracy.py')
CASES = (('49709717a6c3d405c60d', 'DETERMINISTIC', 705),
         ('14bd81a375400c9b0305', 'EXPLORATION', 977),
         ('14bd81a375400c9b0305', 'CONFIRMATION', 977))


def blob(ref, path):
    return subprocess.check_output(['git', 'show', ref+':'+path], cwd=ROOT)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(); output = Path(args.output)
    if output.exists():
        raise FileExistsError('Preserve existing result; use no overwrite')
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    script = Path(__file__).relative_to(ROOT).as_posix()
    if blob(source, script) != Path(__file__).read_bytes():
        raise ValueError('Runner must be committed before numerical execution')
    for path in MODULES:
        if (ROOT/path).read_bytes() != blob(BASE, path):
            raise ValueError('Fixed accounting SOURCE differs')
    raw = {path: blob(BASE, path) for path in INPUTS}
    prepared, d1, points = (json.loads(raw[path]) for path in INPUTS)
    if prepared['saved_source'] != 'ffe8a71da98bec330ce1af109a9606f14e1fc598':
        raise ValueError('Fixed D1 source differs')
    for path in INPUTS[1:]:
        if digest(raw[path]) != prepared['inputs'][path]:
            raise ValueError('D1/preparation input differs')
    if len(prepared['cases']) != 3:
        raise ValueError('Exactly three cases required')
    config = prepared['configuration']
    if config != d1['configuration'] or (config['base_time'],config['beta_total'],config['beta_stat'],config['alpha_total_per_method']) != (.1,.4,.2,.05):
        raise ValueError('Fixed configuration differs')
    rows = []
    for case, (cid, cohort, shots) in zip(prepared['cases'], CASES):
        plan = case['plan']; prices = case['prices']
        if (case['candidate_id'],case['cohort'],plan['m'],plan['candidate']['T'],plan['N_X'],plan['N_Y']) != (cid,cohort,3,.8,shots,shots):
            raise ValueError('Fixed candidate/cohort/N differs')
        saved = next(row for row in d1['rows'] if row['candidate_id'] == cid)
        original_plan = deepcopy(plan); original_plan.pop('law_context')
        if original_plan != saved['ideal_numerical_pilot']:
            raise ValueError('Saved D1 shot plan differs')
        point = next(p for p in points['points'] if p['candidate']['id'] == cid)
        original_price = next(p for p in point['prices'] if p['cohort'] == cohort)
        for price in prices:
            axis = price['measurement_axis']
            if (price['cohort'] != cohort or price['n'] != original_price['n']
                    or price['source'] != original_price['metrics_source']
                    or price['mean']['rz_count'] != original_price['mean_metrics'][axis]['rz_count']):
                raise ValueError('Saved price/axis/cohort differs')
        # Exactly one primary accounting call per fixed case, RZ only.
        result = account_subset([plan], prices, selected_rounds=[3], expected_rounds=range(13),
            compiler=original_price['compiler'], alpha_total=.05, metrics=('rz_count',))
        if (result['status'] != 'CONDITIONAL_SELECTED_ROUND_SUBTOTAL'
                or result['covered_rounds'] != [3] or result['missing_rounds'] != [0,1,2,4,5,6,7,8,9,10,11,12]
                or result['full_rpe_resources'] is not None
                or result['total_covariance'] is not None or result['total_confidence_interval'] is not None):
            raise ValueError('Partial coverage/qualification differs')
        # Independent exact-rational check of the supplied binary64 C values.
        exact = sum((Fraction(plan['N_'+p['axis']])*Fraction(p['mean']['rz_count']) for p in prices),Fraction(0))
        total = result['subset_resources']['rz_count']
        if abs(Fraction(total)-exact) > Fraction(math.ulp(total)):
            raise ValueError('Axis product sum differs from exact supplied-input arithmetic')
        rows.append(dict(candidate_id=cid,cohort=cohort,
            inputs=[dict(axis=p['axis'],measurement_axis=p['measurement_axis'],N=plan['N_'+p['axis']],
                C_RZ=p['mean']['rz_count'],classical_n=p['n'],classification=p['classification'],
                source=p['source']) for p in prices],
            binding=plan['binding'],law_context=plan['law_context'],price_assumptions=prices[0]['price_assumptions'],
            accounting=result,axis_sum_check='PASS_EXACT_RATIONAL_SUPPLIED_INPUTS_WITHIN_ONE_ULP',
            actual_N=None,overall_qualification='CONDITIONAL_SAVED_N_FINITE_COHORT_PRICE_PROXY'))
    payload = dict(status='THREE_SAVED_M3_RZ_SUBTOTALS_COMPLETED',scientific_source=source,
        accounting_source=BASE,D1_scientific_source=d1['scientific_source'],configuration=config,
        input_sha256={p:digest(b) for p,b in raw.items()},
        source_sha256={p:digest((ROOT/p).read_bytes()) for p in (*MODULES,script)},rows=rows,
        scope=dict(account_subset_calls=3,exact_axis_sum_checks=3,new_N_or_C_estimates=0,
            new_prior_or_horizon=0,new_signals=0,trajectories=0,builds_compiles=0,model_fits_bootstraps=0,
            other_round_evaluations=0,ratios_rankings_sensitivity_residuals=0,full_RPE_or_H6_evaluations=0),
        full_rpe_resources=None,actual_success_qualified=False,mandatory_stop=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:
        json.dump(payload,f,ensure_ascii=False,indent=2); f.write('\n')
    print('Completed three fixed m3 RZ subtotals; no full RPE cost or ranking.')


if __name__ == '__main__':
    main()
