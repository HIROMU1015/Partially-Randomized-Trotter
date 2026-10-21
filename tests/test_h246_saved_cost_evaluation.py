"""Artificial accounting and authorization checks, without molecular inputs."""
import copy
from fractions import Fraction
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import h246_saved_cost_evaluation as s
import run_h246_saved_cost_evaluation as runner


def record(method='PR', q=1, total=.001, pf=.002, u=1e-9, logB=.02):
    candidate = dict(system='H2', id='ARTIFICIAL_'+method+str(q), method=method,
                     formula='2nd', prefix=1, q=q)
    total_bias = dict(real=total, imag=-total/2)
    outer = dict(real=pf, imag=-pf/2)
    finite = {a:total_bias[a]-outer[a] for a in ('real','imag')}
    rows = []
    for eps in s.EPS:
        for axis in ('real', 'imag'):
            h = eps/math.sqrt(2)-abs(total_bias[axis])-u
            status = 'INELIGIBLE' if h <= 0 else 'UNDETERMINED' if u/h>.01 else 'ELIGIBLE'
            rows.append(dict(epsilon=eps,axis=axis,bias=total_bias[axis],u=u,
                log_B=logB,log_B_margin=1e-12,status_under_proposed_empirical_policy=status))
    return dict(candidate=candidate,signal=dict(total_signed_bias=total_bias,
        outer_PF_or_discard_bias=outer,finite_RTE_bias=finite,
        corrected=total_bias,reference=dict(real=0.,imag=0.),log_B=logB),
        empirical_eligibility=dict(u=u,log_B_margin=1e-12,rows=rows))


class SavedEvaluationTests(unittest.TestCase):
    def test_exact_known_moments(self):
        m=s.moments([1,2,3]);self.assertEqual(s.frac(m['mean_exact']),2)
        self.assertEqual(s.frac(m['variance_exact']),1)
        self.assertAlmostEqual(m['SE'],1/math.sqrt(3))

    def test_large_integer_costs_are_not_rounded_before_statistics(self):
        n=10**22;m=s.moments([n,n+1,n+2])
        self.assertEqual(s.frac(m['mean_exact']),n+1)
        self.assertEqual(s.frac(m['variance_exact']),1)

    def test_anticorrelation_cancels_variance(self):
        p=[(1,3),(2,2),(3,1)];stats=s.paired(p)
        self.assertEqual(stats['covariance'],-1)
        cost=s.weighted(p,stats,2,2)
        self.assertEqual(cost['mean'],8);self.assertEqual(cost['variance'],0)

    def test_unequal_axis_weights_keep_covariance(self):
        p=[(1,3),(2,2),(3,1)];g=s.weighted(p,s.paired(p),2,3)
        self.assertEqual(g['variance'],1)
        self.assertEqual(g['mean'],10)

    def test_single_observation_does_not_create_zero_variance(self):
        self.assertIsNone(s.moments([7])['variance'])

    def test_price_type_and_missing_rejected(self):
        for values in ([],[None],[1.5],[True],[-1]):
            with self.assertRaises(ValueError):s.moments(values)

    def test_non_cancellation_is_not_signed_bias(self):
        r=record(total=.001,pf=.1)
        self.assertEqual(s.signal_bias(r,'USUAL_SIGNED')[0]['real'],.001)
        self.assertGreater(s.signal_bias(r,'NONCANCELLATION')[0]['real'],.19)
        self.assertEqual(s.shot_record(r,.05,'NONCANCELLATION')['status'],'INELIGIBLE')

    def test_truncated_pf_missing_components_remain_null(self):
        out=s.shot_record(record('B0'),.05,'NONCANCELLATION')
        self.assertEqual(out['status'],'MISSING_COMPONENTS');self.assertIsNone(out['N_cos'])

    def test_full_pf_has_unambiguous_component(self):
        out=s.shot_record(record('FULL'),.05,'NONCANCELLATION')
        self.assertEqual(out['status'],'ELIGIBLE')

    def test_empirical_guard_and_ineligible(self):
        self.assertEqual(s.shot_record(record(total=.1),.05,'USUAL_SIGNED')['status'],'INELIGIBLE')
        r=record(total=.001,u=.001)
        self.assertEqual(s.shot_record(r,.01,'USUAL_SIGNED')['status'],'UNDETERMINED')

    def test_log_domain_no_fabricated_shot(self):
        r=record(total=0.,pf=0.,logB=30.)
        out=s.shot_record(r,.05,'USUAL_SIGNED')
        self.assertEqual(out['status'],'LOG_DOMAIN_ONLY');self.assertIsNone(out['N_total'])

    def test_frozen_and_decimal_shots_and_saved_eligibility(self):
        r=record()
        for eps in (.05,.005):
            out=s.shot_record(r,eps,'USUAL_SIGNED')
            self.assertEqual(out['status'],'ELIGIBLE')
            self.assertGreater(out['N_total'],0)

    def test_bad_component_closure_rejected(self):
        r=record();r['signal']['finite_RTE_bias']['real']+=.01
        with self.assertRaisesRegex(ValueError,'CLOSURE'):s.signal_bias(r,'NONCANCELLATION')

    def test_exact_decomposition_with_axis_weights(self):
        base=dict(total_RZ_exact=s.rational(Fraction(100)),shot=dict(N_total=10),total_RZ_SE=1.)
        other=dict(total_RZ_exact=s.rational(Fraction(60)),shot=dict(N_total=20),total_RZ_SE=2.)
        d=s.comparison(base,other)
        self.assertEqual(d['improvement_percent'],40.)
        self.assertEqual(d['shot_total_ratio'],2.)
        self.assertEqual(d['effective_unit_ratio'],.3)

    def test_unknown_candidate_prevents_fixed_set_best(self):
        rows=[dict(status='MISSING_COMPILED_COST'),dict(status='INELIGIBLE')]
        self.assertIsNone(s.select(rows)['best'])

    def test_no_eligible_is_distinct_from_unknown(self):
        self.assertEqual(s.select([dict(status='INELIGIBLE')])['status'],'NO_ELIGIBLE_CANDIDATE')

    def test_selection_uses_exact_rational_and_retains_ties(self):
        n=10**22
        rows=[dict(status='ELIGIBLE',candidate_id='a',q=1,total_RZ_exact=s.rational(Fraction(n+1))),
              dict(status='ELIGIBLE',candidate_id='b',q=2,total_RZ_exact=s.rational(Fraction(n)))]
        self.assertEqual(s.select(rows)['best']['candidate_id'],'b')
        rows[0]['total_RZ_exact']=s.rational(Fraction(n))
        self.assertEqual(s.select(rows)['tied_candidate_ids'],['a','b'])

    def test_unobserved_rare_stratum_not_zero_imputation(self):
        value=s.evaluate_candidate(record(),[(10,10),(11,11)],['S0','S0'])
        self.assertIsNone(value['strata']['S4']['conditional_statistics'])
        self.assertIsNone(value['strata']['S4']['population_cost_bound'])

    def test_invalid_grant_rejected_before_inputs(self):
        with self.assertRaisesRegex(ValueError,'EXPLICIT_NUMERICAL'):runner.verify({'authorized':False})
        with self.assertRaisesRegex(ValueError,'FIXED_GRANT'):
            runner.verify(dict(authorized=True,grant={'valid':False},plan={}))

    def test_consumed_root_cannot_resume(self):
        with tempfile.TemporaryDirectory() as td:
            with patch.object(runner,'verify',return_value=None):
                with self.assertRaisesRegex(ValueError,'UNUSED_ONE_SHOT_ROOT'):
                    runner.execute(dict(run_root=td))

    def test_native_pair_identity_and_cohort_reject_tampering(self):
        task=dict(id='ARTIFICIAL',stage='H2_PR_EXPLORATION',candidate=dict(q=1))
        checks=[dict(PASS=True,phase_alignment=False,strictly_below=1e-9,error=0.,check='check'+str(i)) for i in range(8)]
        axes=[dict(axis=a,compiler_settings_hash=s.COMPILER,native_policy=s.NATIVE,
            preparation_hash='P',event_digest='E',metrics=dict(rz_count=7)) for a in ('cosine','sine')]
        value=dict(task=task,SOURCE='S',plan_sha256='Q',stage=task['stage'],seed_cohort='EXPLORATION',
            paired_trajectory=True,status='UNIT_COST_PASS_NOT_ACCOUNTING',native_scientific_PASS=True,
            byte_restoration_PASS=True,conditional_sampling=False,cohort_pooling=False,
            shot_accounting=False,cost_ranking=False,input_binding={'id':'I'},preparation_hash='P',
            log_B=.1,checks=checks,axis_costs=axes,event_digest='E',events=dict(N2=0,N4=0,stratum='S0'))
        self.assertEqual(s.load_price(value,task,{'id':'I'},'P',.1,'S','Q'),((7,7),'S0'))
        for field,bad in [('seed_cohort','DESIGN_PILOT'),('SOURCE','OTHER'),
                          ('preparation_hash','OTHER'),('paired_trajectory',False),('cohort_pooling',True)]:
            broken=copy.deepcopy(value);broken[field]=bad
            with self.assertRaises(ValueError):s.load_price(broken,task,{'id':'I'},'P',.1,'S','Q')

    def test_missing_direct_price_is_not_zero_cost(self):
        value=s.evaluate_candidate(record('FULL'),[],None)
        row=next(r for r in value['rows'] if r['epsilon']==.05 and r['mode']=='USUAL_SIGNED')
        self.assertEqual(row['status'],'MISSING_COMPILED_COST')
        self.assertIsNone(row['total_RZ'])


if __name__=='__main__':unittest.main()
