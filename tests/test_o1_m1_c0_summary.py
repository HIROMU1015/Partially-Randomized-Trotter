"""Invented price records only; no scientific data or code entry."""
import importlib.util
from pathlib import Path
import pytest

path=Path(__file__).resolve().parents[1]/'scripts/summarize_o1_m1_c0_pilot.py'
spec=importlib.util.spec_from_file_location('c0_report_synthetic',path)
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)


def fixture():
    rows=[]
    for arm in report.ARMS:
        for m in range(7):
            rows.append(dict(arm=arm,round_index=m,
                account=dict(feasible=True,shots=dict(cosine=2,sine=3)),
                cost=dict(cosine_expected_cost={k:1. for k in report.METRICS},
                          sine_expected_cost={k:4. for k in report.METRICS},
                          classical_sample_count=32),
                CX_relative_sample_SE=dict(cosine=.01,sine=.02)))
    return dict(status='C0_CONDITIONAL_PILOT_COMPLETE',priced_rounds=rows,provenance={'invented':True})


def test_same_axis_N_and_MC_not_multiplier_across_all_metrics():
    p=fixture();result=report.summarize(p)
    assert result['total_mean_cost']=={arm:{metric:98. for metric in report.METRICS} for arm in report.ARMS}
    assert result['price_confidence_interval'] is None
    assert result['guarantee_status']=='not_certified'


@pytest.mark.parametrize('failure',['incomplete','duplicate','unfinished','infeasible','price_precision'])
def test_no_full_horizon_value_for_unqualified_or_missing_data(failure):
    p=fixture()
    if failure=='incomplete': p['priced_rounds'].pop()
    if failure=='duplicate': p['priced_rounds'][-1]=p['priced_rounds'][0]
    if failure=='unfinished': p['status']='RUNNING'
    if failure=='infeasible': p['priced_rounds'][0]['account']['feasible']=False
    if failure=='price_precision': p['priced_rounds'][0]['CX_relative_sample_SE']['cosine']=.11
    result=report.summarize(p)
    assert result['total_mean_cost'] is None and result['status']=='UNASSESSED'
