"""Artificial checks for the new selection and shared-cell accounting adapter."""
import numpy as np
from trottertracks.track_c_closed_p3_joint_opt import weight_map, shrink
from trottertracks.track_c_closed_p3_resources import covariance


def test_shared_cell_weights_combine_before_covariance():
    p={'selected':[{'signal':{'steps':3,'recipe':[{'time':.02,'r':2}]}},{'signal':{'steps':5,'recipe':[{'time':.02,'r':2}]}}], 'trials':[[2,3],[4,6]]}
    w=weight_map(p);assert list(w.values())==[65]
    key=next(iter(w));values=np.array([[10.,20.,30.,40.],[14.,24.,34.,44.]])
    rows=[{'canonical':v,'closed':v-1,'stratum_probability':{'canonical':.3,'closed':.25}} for v in values]
    got=covariance({key:{'True':rows}},[('canonical',w),('closed',{key:-40})])
    exact=np.cov(65*.3*values-40*.25*(values-1),rowvar=False,ddof=1)/2
    np.testing.assert_allclose(got,exact)


def test_source_disk_consumes_both_signal_and_phase_budget():
    q={'rho_lower':.9,'beta_stat_lower':.3,'closed':True}
    r=shrink(q,.01);assert r['closed'] and r['rho_lower']<q['rho_lower'] and r['beta_stat_lower']<q['beta_stat_lower']
    assert not shrink(q,1.)['closed']


def test_integer_alpha_refinement_preserves_bounds_and_improves_or_keeps_cost():
    from fractions import Fraction as F
    from trottertracks.track_c_bottleneck import allocate,rigorous_shots
    from trottertracks.track_c_closed_p3_joint_opt_alpha import discrete_allocate
    points=[dict(qualification=dict(rho_lower=.7+.01*(m%3),beta_stat_lower=.31),mean=np.full((2,4),2.**m)) for m in range(18)]
    _,old=allocate(points,0);a,n=discrete_allocate(points)
    assert sum(F.from_float(float(x)) for x in a.flat)<=F(1,20)
    assert sum(p['mean'][b,0]*n[m,b] for m,p in enumerate(points) for b in range(2))<=sum(p['mean'][b,0]*old[m,b] for m,p in enumerate(points) for b in range(2))
    assert all(n[m,b]>=rigorous_shots(p['qualification']['rho_lower'],p['qualification']['beta_stat_lower'],a[m,b]) for m,p in enumerate(points) for b in range(2))
