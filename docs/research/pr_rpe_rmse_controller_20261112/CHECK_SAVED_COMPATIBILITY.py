"""Pinned metadata checks only. No N reinterpretation, cost sums or prediction."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks import pf_pr_rpe_pipeline as p
from trottertracks import rpe_r_selection as r


def check():
    store = p.SavedArtifacts(ROOT)
    ref = dict(commit='71ba49bb28b295aa28e51101bc3b1bf07d500767',
        path='docs/research/pr_r_detailed_n_p1_results_20261111/run01/N_HANDOFF.json')
    saved = store.read(ref)
    assert len(saved['shots']) == 12 and saved['source_preserved'] and saved['input_preserved']
    for n in saved['shots']:
        menu = saved['menus'][n['identity']['binding']['system']]
        row = next(v for v in menu['candidates'] if v['candidate']['id'] == n['identity']['candidate']['id'])
        request = r.request_context(menu, row)
        assert p.same(request['identity'], n['identity']) and p.same(request['law_binding'], n['law_binding'])
        assert n['plan']['status'] == 'CONDITIONAL_ROUND_PASS'
        assert n['N_comparison_policy'] == 'R_AWARE_EMPIRICAL_ACTION_COUNT_V1'
        assert 'alpha_X' in n['plan'] and 'alpha_Y' in n['plan']
        assert all(type(n['plan']['N_' + axis]) is int for axis in p.AXES)
    cref = dict(commit='7dd61438b64a4b015ce0b7a7db81187d9867d8a2',
        path='docs/research/rpe_c1_price_results_intake_20261111/INTAKE.json')
    c1 = store.read(cref)
    models = [v for v in c1['prices'] if v['classification'] == 'MODEL_PREDICTION']
    assert len(c1['prices']) == 32 and len(models) == 16
    assert all(v['model']['qualified_for_candidate'] is False for v in models)
    assert all(v['expectation'] == 'FINITE_COHORT_ONLY' for v in c1['prices'])
    assert saved['full_RPE_qualified'] is False and saved['actual_N_qualified'] is False
    assert saved['total_resources'] is None
    return dict(status='SAVED_METADATA_COMPATIBILITY_PASS', P1_source=ref, P1_records=12,
        original_N_policy='HIGH_PROBABILITY_RPE_WITH_FIXED_ALPHA_BETA',
        original_comparison_policy=saved['comparison_policy'], N_values_recomputed=False,
        reinterpreted_as_RMSE=False, RMSE_N_records_produced=0,
        C1_source=cref, C1_records=32, C1_models=16, C1_models_qualified_for_candidate=False,
        C1_qualification_flags_unchanged=True,
        C1_finite_cohort_only=True, C1_identity_relabelled=False,
        numerical_N_C_G_or_prediction=0, scientific_policy_adopted=False, mandatory_stop=True)


if __name__ == '__main__':
    print(json.dumps(check(), ensure_ascii=False, indent=2))
