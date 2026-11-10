"""Synthetic statistical/input planning checks; no H4 data or circuits."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pytest
from trottertracks import track_d_r16_calibration as p


def invented():
    rows=[]
    for row in p.sampling_plan()['rows']:
        for m in p.METHODS:
            for a in p.AXES:
                values={g:10*row['q']+i+row['sample']*(i+1) for i,g in enumerate(p.METRICS)}
                if m=='reference':values={g:v-2 for g,v in values.items()}
                rows.append(dict(q=row['q'],sample=row['sample'],method=m,axis=a,metrics=values))
    return rows


def test_fixed_plan_reuses_one_path_and_has_exact_nonoverlapping_seed_scope():
    x=p.sampling_plan();assert x==p.sampling_plan()
    new=[r for r in x['rows'] if not r['reused']]
    seeds=[s for r in new for s in r['seeds']]
    assert len(new)==11 and len(seeds)==len(set(seeds))==52
    assert sum(r['q']*16 for r in new)==832
    assert len(x['rows'])==12 and x['new_compile_limit']==44
    assert all(r['cut_after_complete_step']==r['q']//2 for r in x['rows'])


def test_paired_statistics_use_path_units_and_keep_correlated_XY():
    x=p.paired_statistics(invented())
    for q in p.QS:
        s=x['by_q'][str(q)]
        assert s['n_path_units']==4 and s['covariance_rank_limit']==3
        assert len(s['sample_covariance'])==28 and len(s['paired_sample_covariance'])==14
        assert all(r['paired_delta']['mean']==-2 and r['paired_delta']['sample_SD']==0 for r in s['rows'])
        assert s['sample_covariance'][0][0]==s['sample_covariance'][0][7]
        assert all(c is None for row in s['paired_sample_correlation'] for c in row)
        assert s['sample_correlation'][0][7]==pytest.approx(1.)
    assert not x['population_confidence_guaranteed'] and x['confidence_interval'] is None


@pytest.mark.parametrize('change',['missing','duplicate'])
def test_incomplete_or_duplicate_prices_never_become_complete_case_mean(change):
    rows=invented();rows=rows[:-1] if change=='missing' else rows+[rows[0]]
    with pytest.raises(ValueError,match='COMPLETE_FIXED_CAMPAIGN'):p.paired_statistics(rows)
