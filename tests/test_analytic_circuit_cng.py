"""Non-mirroring checks: covariance, absent N, theoretical endpoints, bad input."""
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('cng',Path(__file__).parents[1]/'src/trotterlib/analytic_circuit_cng.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_anticorrelated_axes_cancel_in_weighted_total():
    cosine=m.summarize([0,4],1.0);sine=m.summarize([4,0],1.0)
    total=m.summarize([4,4],1.0)
    assert cosine['SE']>0 and sine['SE']>0 and total['SE']==0


def test_theory_interval_and_nominal_interval_stay_distinct():
    r=m.ratio_interval('5',1,'8','10')
    assert r==dict(lower=.5,upper=.625,nominal95_lower=.4,nominal95_upper=.75)


def fixture():
    v=dict(theory={a:dict(lower='8',upper='10') for a in m.AXES},
           shots=dict(cosine=2,sine=3),compiled=dict(n=2,cosine_mean='2',sine_mean='2',half95=0),preparation_hash='x')
    return dict(PR=v,system='H4',epsilon='0.05',PR_candidate_id='fixed',PR_L_D=1,PR_q=2,PR_r=4,PR_K=2,
        DF_rank=10,cohort='artificial',status='COMPLETE_PAIRED_WITHIN_COHORT',executed_technical_checks_complete=True,
        G_PR_theory_lower='40',G_PR_theory_upper='50',G_PR_compiled='10')


def test_no_invented_circuit_shots_or_G11():
    d=m.account(fixture(),'PR',[dict(cosine=2,sine=2)]*2,1)
    assert d['G01']['mean_exact']=='10' and d['G00']==dict(lower='40',upper='50')
    assert d['G11'] is None and d['G10'] is None and d['delta_G_N'] is None
    assert all(v['N_circuit'] is None and v['N_ratio'] is None for v in d['axes'].values())


def test_source_total_mismatch_is_rejected():
    import pytest
    r=fixture();r['G_PR_compiled']='11'
    with pytest.raises(ValueError,match='G01'):m.account(r,'PR',[dict(cosine=2,sine=2)]*2,1)
