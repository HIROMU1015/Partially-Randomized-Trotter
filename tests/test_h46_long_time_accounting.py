"""Shot-weighted paired uncertainty must retain within-trajectory covariance."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h46_long_time_report as report
from h246_saved_cost_evaluation import paired


def test_paired_cost_uncertainty_includes_covariance():
    ep=dict(candidate=dict(method='PR'),N_cos=10,N_sin=20)
    result=report.cost_for(ep,dict(unit_statistics=paired([(2,2),(4,4)]),cohort='ARTIFICIAL'))
    assert result['G']==90 and result['G_exact']=='90'
    assert result['G_SE']==30


def test_deterministic_integer_accounting_has_no_sampling_error():
    ep=dict(candidate=dict(method='FULL'),N_cos=10**14+1,N_sin=10**14+3)
    result=report.cost_for(ep,dict(cosine=1201,sine=1203))
    assert result['G_exact']==str((10**14+1)*1201+(10**14+3)*1203)
    assert result['G_SE'] is None
