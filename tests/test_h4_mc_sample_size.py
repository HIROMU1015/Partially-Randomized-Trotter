"""Synthetic checks only: paired covariance, draw sizes, ranking and one-shot scope."""
import importlib.util
from pathlib import Path
import sys
from fractions import Fraction

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('mc',ROOT/'scripts/analyze_h4_mc_sample_size.py')
mc=importlib.util.module_from_spec(spec);spec.loader.exec_module(mc)


def fixture():
    rows=[]
    for i,(cos,sin,s) in enumerate([(1,9,'S0_ONLY'),(9,1,'S2_SINGLE')]):
        rows.append(dict(cell_id='fake',replica=i,cohort='INDEPENDENT_H4_CONFIRMATION',stratum=s,
            axes={a:dict(axis=a,RZ=v,compiler_hash='fake',fingerprint='a'*64) for a,v in [('cosine',cos),('sine',sin)]}))
    ep=dict(n=2,epsilon=.05,cell={'id':'fake'},saved_shots={'cosine':1,'sine':1},
        PF_G_exact='11',expected_direction=1,strata={'S0_ONLY':1,'S2_SINGLE':1,'S2_MULTI':0,'S4_PRESENT':0})
    ep['result']=mc.c.evaluate_pairs(rows,ep['saved_shots'],'11',1)
    return rows,ep


def test_covariance_preserved():
    rows,ep=fixture();rs,values,stats=mc.restore_pairs(rows,ep,'fake')
    assert values==[10,10] and stats['paired_SE']==0
    checkpoints,samples,_=mc.bootstrap(values,[r['stratum'] for r in rows],4,0,'11',Fraction(0),np)
    assert checkpoints['bootstrap_SE']==0 and checkpoints['PR_below_PF_fraction']==1
    assert all(x['mean_G']==10 and x['within_sample_SE']==0 for x in samples)


def test_prefixes_pair_cost_and_order():
    rows,ep=fixture();rs,values,_=mc.restore_pairs(rows[::-1],ep,'fake')
    out=mc.prefixes(rs,values,ep)
    assert out[0]['replica']==0 and out[0]['cumulative_SE'] is None
    assert out[-1]['cumulative_mean_exact']=='10' and out[-1]['cumulative_SE']==0


@pytest.mark.parametrize('n',[4,8,12,16])
def test_reproducibility_counts_and_exact_comparison(n):
    args=([0,10],['S0_ONLY','S2_SINGLE'],n,1,'5',Fraction(50),np)
    a,ss,gg=mc.bootstrap(*args); b,tt,hh=mc.bootstrap(*args)
    assert (a,ss,gg)==(b,tt,hh)
    assert a['PR_below_PF_count']+a['PR_equal_PF_count']+a['PR_above_PF_count']==2000
    assert sum(x['replicates'] for x in gg)==2000
    for x in ss:
        assert x['sum_G_exact']==10*x['S2_draws'] and x['mean_G']==x['sum_G_exact']/n
    assert abs(a['bootstrap_SE']-a['conditional_empirical_SE'])/a['conditional_empirical_SE']<.06


def test_replica_and_saved_stat_mismatch_rejected():
    rows,ep=fixture();rows[1]['replica']=0
    with pytest.raises(ValueError,match='EXACT_REPLICA_COVERAGE'):mc.restore_pairs(rows,ep,'fake')
    rows,ep=fixture();ep['result']['G_exact']='999'
    with pytest.raises(ValueError,match='REPRODUCE_SAVED'):mc.restore_pairs(rows,ep,'fake')


def test_overflow_and_noninteger_pf_rejected():
    with pytest.raises(ValueError,match='INT64'):mc.bootstrap([2**62,2**62],['S0_ONLY']*2,4,0,'1',0,np)
    with pytest.raises(ValueError,match='INTEGER_PF'):mc.bootstrap([1,2],['S0_ONLY']*2,4,0,'3/2',1,np)


def test_one_shot_not_refunded_or_overwritten(tmp_path):
    mc.consume_once('a'*64,tmp_path)
    original=(tmp_path/'ONE_SHOT.json').read_bytes()
    with pytest.raises(FileExistsError):mc.consume_once('b'*64,tmp_path)
    assert (tmp_path/'ONE_SHOT.json').read_bytes()==original
