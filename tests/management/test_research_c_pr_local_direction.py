"""Analytic inner/reduced references and invented DF only; no saved NPZ."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from trottertracks import c_pr_local_direction as m
from test_research_c_separated_J_U import invented


class AnalyticInner:
    """F=.5(j-a*u)^2+phi(u); exact reduced phi and controlled J errors.

    Two refinements can agree at a biased inner solution. Values/partials below
    are genuine analytic F/partials, not fictitious optimizer responses.
    """
    def __init__(self,*,F0=1.,g=1.,L=2.,cubic=0.,coupling=0.,error=0.,noise=0.,unqualified=None):
        self.F0,self.g,self.L,self.cubic=F0,g,L,cubic
        self.a,self.error,self.noise=coupling,error,noise
        self.unqualified=unqualified;self.queries=[]
    def phi(self,u):return self.F0+self.g*u+.5*self.L*u*u+self.cubic*u**3
    def evaluate(self,x):
        u=float(x[0]);self.queries.append(u)
        if self.unqualified=='initial' or self.unqualified=='one_side' and len(self.queries)==2:
            raise m.stable.Unqualified('REDUCED_GRADIENT_UNQUALIFIED')
        if self.unqualified=='correctness' and len(self.queries)==2:raise ValueError('ARTIFICIAL_CORRECTNESS_FAIL')
        e=self.error+(np.sqrt(2*self.noise) if u>0 else 0.)
        value=self.phi(u)+.5*e*e
        partial=self.g+self.L*u+3*self.cubic*u*u-self.a*e
        return dict(label=str(len(self.queries)-1),value=value,eta=x.copy(),gradient=np.array([partial]),
            state=dict(components={'objective':value},value_same_array={'maxabs':0.},binary64_AD_value_same_array={'maxabs':0.},
                J_stationarity=m.stable.old.gradient_gate(abs(e),1e-5,m.DEFAULT)),
            qualification=dict(status='NUMERICAL_PILOT_ADMITTED',objective_refinement={'before':value,'after':value},
                latest_refinement_U_shift_l2=0.))


def diagnose(toy):
    experiment=m.collect(toy.evaluate,np.zeros(1));count=len(toy.queries)
    report=m.analyse(experiment)
    assert len(toy.queries)==count  # finite-difference assembly has no queries
    return experiment,report


def test_nonnegative_scale_is_not_a_curvature_bound():
    good=AnalyticInner();_,r=diagnose(good)
    assert all(good.phi(-a)>good.phi(0.) for a in (1.9,3.8,7.6))
    assert all(x['forward_observation']=='EMPIRICAL_DECREASE' for x in r['rows'])
    bad=AnalyticInner(L=1e6);_,s=diagnose(bad)
    assert all(x['forward_observation']=='EMPIRICAL_INCREASE' for x in s['rows'])
    assert r['accepted_U_updates']==s['accepted_U_updates']==0


def test_stationarity_and_refinement_agreement_can_hide_wrong_reduced_direction():
    o=AnalyticInner(coupling=1e6,error=2e-6);exp,r=diagnose(o)
    assert exp['initial']['state']['J_stationarity']['status']=='MET'
    assert exp['initial']['qualification']['latest_refinement_U_shift_l2']==0.
    assert r['reported_slope']==pytest.approx(-1.)
    assert all(x['central_FD']==pytest.approx(1.,abs=2e-12) for x in r['rows'])
    assert all(x['forward_observation']=='EMPIRICAL_INCREASE' for x in r['rows'])
    # Fixed-J finite differences agree with the partial gradient, while the
    # independently minimized reduced objective has the opposite derivative.
    h=1e-5;e=o.error
    fixed=lambda u:o.phi(u)+.5*(e-o.a*u)**2
    fixed_fd=(fixed(h)-fixed(-h))/(2*h)
    true_reduced=(o.phi(h)-o.phi(-h))/(2*h)
    assert fixed_fd==pytest.approx(-1.,abs=1e-8) and true_reduced==pytest.approx(1.,abs=1e-10)
    assert r['true_reduced_gradient_qualification'] is None


def test_curvature_truncation_is_not_misclassified_as_gradient_failure():
    _,r=diagnose(AnalyticInner(cubic=1.))
    errors=[abs(x['FD_minus_reported_slope']) for x in r['rows']]
    assert errors[0]/errors[1]==pytest.approx(16.) and errors[1]/errors[2]==pytest.approx(16.)
    assert r['true_reduced_gradient_qualification'] is None
    assert all(x['truncation_error_bound'] is None for x in r['rows'])


def test_toy_inner_accuracy_changes_directional_bias_without_relaxing_J_gate():
    _,bad=diagnose(AnalyticInner(coupling=1e6,error=2e-6))
    _,good=diagnose(AnalyticInner(coupling=1e6,error=2e-16))
    assert abs(bad['rows'][-1]['FD_minus_reported_slope'])>1.
    assert abs(good['rows'][-1]['FD_minus_reported_slope'])<1e-8
    assert good['true_reduced_gradient_qualification'] is None


def test_value_floor_and_inner_error_over_h_remain_unresolved():
    o=AnalyticInner(F0=1e-12,g=1e-6,L=1.,noise=2e-11);_,r=diagnose(o)
    assert all(not x['FD_signal_above_comparison_scale'] for x in r['rows'])
    errors=[abs(x['FD_minus_reported_slope']) for x in r['rows']]
    assert errors[2]>errors[1]>errors[0]
    assert all(x['forward_observation']=='UNRESOLVED_VALUE_SCALE' for x in r['rows'])
    assert all(x['unknown_inner_solution_error'] is None for x in r['rows'])


@pytest.mark.parametrize('mode',['initial','one_side'])
def test_missing_qualified_point_is_not_replaced_by_zero(mode):
    o=AnalyticInner(unqualified=mode);_,r=diagnose(o)
    if mode=='initial':assert r['status']=='INITIAL_UNQUALIFIED_NO_LOCAL_DIAGNOSTICS' and len(o.queries)==1
    else:
        assert r['status']=='LOCAL_DIAGNOSTICS_PARTIAL' and len(o.queries)==7
        assert r['rows'][0]['central_FD'] is None and r['rows'][0]['FD_sensitivity_scale'] is None


def test_correctness_FAIL_no_remaining_probe_or_optimizer():
    o=AnalyticInner(unqualified='correctness')
    with pytest.raises(ValueError,match='ARTIFICIAL_CORRECTNESS_FAIL'):m.collect(o.evaluate,np.zeros(1))
    assert len(o.queries)==2


def test_zero_scale_stops_and_nonfinite_slope_is_correctness_failure():
    zero=AnalyticInner(g=0.);r=m.collect(zero.evaluate,np.zeros(1))
    assert r['status']=='NO_NONZERO_INITIAL_DIRECTION' and len(zero.queries)==1
    value=AnalyticInner(F0=0.);r=m.collect(value.evaluate,np.zeros(1))
    assert r['status']=='NONPOSITIVE_INITIAL_F_NO_SCALE' and len(value.queries)==1
    huge=AnalyticInner(g=1e308)
    with np.errstate(over='ignore'):
        with pytest.raises(ValueError,match='NONFINITE_LOCAL_DIRECTIONAL_SLOPE'):m.collect(huge.evaluate,np.zeros(1))
    assert len(huge.queries)==1


def test_real_invented_DF_resolves_J_at_each_signed_point_and_preserves_cache():
    g,J,U,s=invented();counts={};records=[]
    def call(k,l,f):counts[k]=counts.get(k,0)+1;return f()
    oracle=m.previous.Oracle(g,J,U,s,m.DEFAULT,call=call,record=lambda *x:records.append(x))
    exp=m.collect(oracle.evaluate,np.zeros(3));r=m.analyse(exp)
    assert len(oracle.points)==7 and counts==dict(inner_J_solve=21,precision_state=14,reduced_admission=7)
    assert r['status'] in ('LOCAL_DIRECTIONAL_DIAGNOSTICS_AVAILABLE','LOCAL_DIAGNOSTICS_PARTIAL')
    assert oracle.evaluate(np.zeros(3)) is exp['initial'] and counts['inner_J_solve']==21
    assert all(i['method']==m.previous.previous.inner.VERSION for p in oracle.points for i in p['inner'])


def study():
    rows=[]
    for name,kwargs in [('curvature_local',{}),('still_too_large',{'L':1e6}),
        ('biased_inner',{'coupling':1e6,'error':2e-6}),('accurate_inner',{'coupling':1e6,'error':2e-16}),('truncation',{'cubic':1.}),
        ('value_floor',{'F0':1e-12,'g':1e-6,'L':1.,'noise':2e-11})]:
        toy=AnalyticInner(**kwargs);experiment,r=diagnose(toy)
        rows.append(dict(case=name,exact_true_phi_derivative_at0=toy.g,
            exact_true_directional_derivative=toy.g*(-float(experiment['initial']['gradient'][0])),
            parameters=kwargs,queries=toy.queries,diagnostics=r))
    return dict(artificial_only=True,real_molecular_array_decodes=0,new_molecular_objective_queries=0,
        rows=rows,interpretation='Analytic inner/reduced references,not measured H4 gradient errors;engineering scales are not enclosures')


def test_artificial_study_is_finite_JSON_including_value_floor_case():
    import json
    result=json.loads(json.dumps(study(),allow_nan=False))
    assert len(result['rows'])==6 and result['real_molecular_array_decodes']==0


if __name__=='__main__':
    import json
    with Path(sys.argv[1]).open('x') as f:json.dump(study(),f,indent=2,allow_nan=False)
