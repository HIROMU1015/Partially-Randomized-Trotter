"""Fixed public metadata projection only: no saved S/C/N values or science."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.bv2_mean_price import B_RELEASE

C1='2e6a0cdf5a647058cc7afeeb5741bfafa24c974b'
PURE='7a8bc7f132e166390c943e6b32c92cc6abe1baf4'
B_PATH='docs/research/pr_rpe_method_coverage_20261113/COVERAGE.json'
PURE_PATH='artifacts/resource_applicability/df_full_h_pure_rte_t20_opt/2026-11-09/'


def read(root,commit,path):
    return json.loads(subprocess.check_output(['git','show',commit+':'+path],cwd=root))


def inventory(root):
    d=read(root,B_RELEASE,B_PATH)
    if len(d['recipes'])!=118 or len(d['rows'])!=236:raise ValueError('FIXED_B_COVERAGE_SHAPE')
    if any(r['eligible_for_RMSE_comparison'] or r['G'] is not None for r in d['rows']):raise ValueError('NOT_STATIC_UNQUALIFIED_MENU')
    recipes=deepcopy(d['recipes']);rows=[]
    for r in d['rows']:
        price=r['C'];scenarios=price['scenarios']
        direct=[x for x in scenarios if x['value_available'] and x['classification'].startswith('DIRECT_COMPILED_')]
        model=[x for x in scenarios if x['value_available'] and x['classification']=='MODEL_PREDICTION']
        rows.append(dict(system=r['system'],method=r['method'],candidate_id=r['candidate_id'],m=r['m'],axis=r['axis'],
            saved_price_presence=price['availability'],direct_cohorts=deepcopy(direct),model_cohorts=deepcopy(model),
            RMSE_qualification=price['RMSE_qualification'],existing_provider_domain=price['existing_provider_domain'],
            C1_metadata=deepcopy(price['C1_sources']),source=deepcopy(price['reference']),
            current_common_kappa_V2_identity=None,qualified_all_law_mean=False,eligible_for_RMSE_comparison=False))
    plan=read(root,C1,'docs/research/r_price_phase1_results_20261111/run01/PLAN.json')
    cells=[dict(cell_id=c['id'],candidate_id=c['candidate_id'],role=c['role'],cohort=c['cohort'],n_paths=c['n'],
        identity=deepcopy(c['request']['identity']),law_binding=deepcopy(c['request']['law_binding']),
        present_in_B_historical_menu=c['candidate_id'] in recipes,new_B_V2_identity=None,
        price_unit='FINITE_COHORT_ONLY',model_qualified_for_candidate=False) for c in plan['cells']]
    pure=read(root,PURE,PURE_PATH+'SOURCE_INDEX.json')
    return dict(status='STATIC_PRICE_METADATA_ONLY_NO_VALUE_REANALYSIS',B_commit=B_RELEASE,B_path=B_PATH,
        recipes=recipes,rows=rows,contexts=deepcopy(d['contexts']),missing_routes=deepcopy(d['missing_routes']),
        summary=dict(historical_candidates=len(recipes),measurement_axes=len(rows),
            methods=dict(Counter(r['method'] for r in recipes.values())),
            direct_axis_presence=dict(Counter(r['method'] for r in rows if r['direct_cohorts'])),
            model_axis_presence=dict(Counter(r['method'] for r in rows if r['model_cohorts'])),
            current_qualified_common_kappa_schedules=0),
        C1_cells=cells,C1_source=dict(commit=C1,path='docs/research/r_price_phase1_results_20261111/run01/PLAN.json'),
        pure_RTE_historical_metadata=dict(source=dict(commit=PURE,path=PURE_PATH+'SOURCE_INDEX.json'),
            evaluation_T=pure['evaluation_T'],price_groups=deepcopy(pure['price_groups']),
            confirmation_groups=deepcopy(pure['confirmation_groups']),compiler=deepcopy(pure['compiler']),
            current_RMSE_price_route='NOT_CONNECTED',scope='FULL_H_INCLUDING_ONE_BODY_NOT_B3',
            kernel_SOURCE=pure['kernel_SOURCE'],provider_SOURCE=pure['provider_SOURCE']),
        S0a_reference=dict(commit='a11efd130e9dfe7086b88386570594ebe54d46af',
            path='docs/research/s0a_saved_structure_results_20261113/run02/REPORT.md',
            evidence='8_OF_8_SAME_PATH_STRUCTURAL_PARITY_NOT_MEAN_COMPILED_PRICE'),
        actual_S_C_N_values_copied=False,new_means_predictions_fit_sampling_compile_N_G=0,
        total_CI=None,joint_covariance=None,full_RPE_qualified=False,mandatory_stop=True)


if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('Usage: inventory_bv2_mean_prices.py <fresh-output.json>')
    out=Path(sys.argv[1])
    with out.open('x') as f:json.dump(inventory(ROOT),f,ensure_ascii=False,indent=2);f.write('\n')
