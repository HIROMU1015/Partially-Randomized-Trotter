"""Invented scalar evidence only; no molecular input or circuit generation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from trottertracks import track_d_r16_validation as v


def test_all13_pass_is_conditional_original_h_not_actual():
    refs = [dict(attenuation=.9, pf_operator_error=.01,
                 rte_corrected_operator_error=.02) for _ in range(13)]
    r = v.assess(refs, df_bound=1e-9, source='synthetic', law_hash='invented')
    assert r['S2_allowed'] and r['energy']['same_accuracy']
    assert len(r['round_plans']) == 13
    assert not r['actual_implemented_success_qualified']
    assert not r['own_DF_signal_or_N_used']
    assert all(p['shots'] > 0 for p in r['round_plans'])


def test_one_failing_round_prevents_s2_without_budget_change():
    refs = [dict(attenuation=.9, pf_operator_error=.01,
                 rte_corrected_operator_error=.02) for _ in range(13)]
    refs[12]['rte_corrected_operator_error'] = .4
    r = v.assess(refs, df_bound=1e-9, source='synthetic', law_hash='invented')
    assert not r['S2_allowed']
    assert r['round_plans'][12]['shots'] is None
    assert r['round_plans'][0]['shots'] > 0


def test_seed_is_fixed_distinct_and_scoped_to_one_r16_q4_path():
    a, b = v.seed_plan(), v.seed_plan()
    assert a == b and a['q'] == 4 and a['r'] == 16
    assert len(set(a['seeds'])) == 4 and a['cut_after_complete_step'] == 2
