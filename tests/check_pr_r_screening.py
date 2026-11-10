"""Artificial normalization and mock interfaces only; no saved tail numerics."""
from copy import deepcopy
import hashlib,json,math,os,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from trottertracks import pr_r_screening as s

LAW=dict(model='ideal_finite_RTE_law',sampling='ARTIFICIAL_REFERENCE_LAW',source=dict(fixture='TOY_LAW'))


class ScreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.base={system:s.A.saved_task(ROOT,system) for system in ('H4','H6')}

    def plan(self,system='H4',rounds=None,**kwargs):
        return s.metadata_plan(ROOT,self.base[system],rounds or ([0,6,12] if system=='H4' else [4,8,12]),
            factors=['1/2','1','2'],sampling_law=LAW,source=dict(fixture='METADATA_ONLY'),**kwargs)

    def tail(self,plan,lam=.5):return dict(binding=plan['menu']['binding'],prefix=1 if plan['menu']['binding']['system']=='H4' else 7,lambda_R=lam,source=dict(fixture='ARTIFICIAL_LAMBDA_NOT_MOLECULAR'))

    def score(self,plan,**kwargs):return s.screen_normalization(ROOT,plan,self.tail(plan),shot_inflation_budget=2.,budget_source=dict(fixture='ARTIFICIAL_SENSITIVITY_NOT_ACCEPTANCE'),**kwargs)

    def test_metadata_planning_is_passive_and_has_no_tail_inputs_or_N_C_G(self):
        code="import sys;from pathlib import Path;from trottertracks import pr_r_screening as s;b=s.A.saved_task(Path.cwd(),'H4');o=s.metadata_plan(Path.cwd(),b,[0,12],factors=['1','2'],sampling_law={'model':'ideal_finite_RTE_law','sampling':'MOCK','source':{'fixture':'MOCK'}},source={'fixture':'MOCK'});assert o['science_calls']==0;assert not any(x in sys.modules for x in ('numpy','scipy','qiskit'))"
        env=os.environ.copy();env['PYTHONPATH']='src:scripts';subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,env=env,check=True)
        with patch.object(s.A.scalar,'analyze_bound_point',side_effect=AssertionError('No N')):
            plan=self.plan()
        self.assertIsNone(plan['R_upper_bound']);self.assertFalse(plan['range_complete'])
        self.assertTrue(all(r['normalization']['log_B'] is None and r['N'] is None and r['C'] is None and r['G'] is None for r in plan['rows']))

    def test_time_scaled_finite_candidates_are_not_a_global_R_cap(self):
        plan=self.plan();values={m:[r['R_outer'] for r in plan['rows'] if r['m']==m] for m in (0,6,12)}
        self.assertEqual(values[0],[2,4,8]);self.assertEqual(values[12],[4,8192,16384,32768])
        self.assertNotEqual(values[0],values[6]);self.assertTrue(all(4 in v for v in values.values()))
        bigger=s.metadata_plan(ROOT,self.base['H4'],[12],factors=['4'],sampling_law=LAW,source={'fixture':'MOCK'})
        self.assertIn(65536,[r['R_outer'] for r in bigger['rows']]);self.assertIsNone(bigger['R_upper_bound'])

    def test_H6_early_rounds_and_signed_allocation_are_preserved(self):
        plan=self.plan('H6');self.assertEqual(plan['menu']['candidates'][:8],self.base['H6']['candidates'][:8])
        for row in plan['rows']:
            self.assertEqual((len(row['structure']['allocation']),sum(row['structure']['allocation'])),(11,row['R_outer']))
            self.assertGreaterEqual(min(row['structure']['allocation']),1);self.assertTrue(any(w<0 for w in row['structure']['signed_weights']))
            if row['m']==12:self.assertEqual((row['q_RPE'],row['q_PF']),(4096,256))
        with self.assertRaises(ValueError):self.plan('H6',[3])

    def test_B_body_and_requests_are_reused_without_cost_evaluator(self):
        B=s.b_metadata(ROOT);raw=subprocess.check_output(['git','show',s.B_COMMIT+':'+s.B_PATH],cwd=ROOT)
        self.assertEqual(B.source['sha256'],hashlib.sha256(raw).hexdigest());self.assertFalse(hasattr(B,'evaluate_menu'))
        plan=self.plan(reference_R={12:[8,32]});row=plan['rows'][-1]
        request=s.A_request(ROOT,plan,row['candidate_id'],dict(fixture='MOCK_SOURCE'))
        self.assertEqual(request['B_request'],row['request']);self.assertTrue(request['law_echo_requires_matching_scientific_evidence'])
        self.assertEqual(s.p.identity(request['A_proposal']['task'],request['A_row']),row['request']['identity'])
        inventory=B.coverage(plan['menu'],[],[],cohort_selection={})
        self.assertTrue(all('MISSING_N' in r['reasons'] for r in inventory['rows']));self.assertIsNone(inventory['costs'])

    def test_existing_K2_finite_formula_matches_independent_artificial_expression(self):
        # lambda=.5 is an artificial scalar; no saved molecular lambda is read.
        from trotterlib.rte import finite_rte_distribution
        for system in ('H4','H6'):
            plan=self.plan(system);out=self.score(plan,finite_distribution=finite_rte_distribution)
            for row in out['rows']:
                terms=[];sensitivities=[]
                for w,r in zip(row['structure']['signed_weights'],row['structure']['allocation']):
                    tau=.5*row['delta']*w/r
                    b=math.sqrt(1+tau*tau)+(tau*tau/2)*math.sqrt(1+(tau/3)**2)
                    terms.append(r*math.log(b))
                    actual_b=finite_rte_distribution(tau,2).exact_finite_distribution
                    self.assertLessEqual(abs(actual_b-b),2*math.ulp(max(actual_b,b)))
                    sensitivities.append(r*(math.log(math.nextafter(actual_b,math.inf))-math.log(math.nextafter(actual_b,0.))))
                self.assertLessEqual(abs(row['normalization']['log_B']-row['q_PF']*math.fsum(terms)),row['q_PF']*math.fsum(sensitivities))
                self.assertFalse(row['normalization']['leading_term_is_used_to_accept'])
                self.assertEqual(row['scientific_status'],'NOT_EVALUATED');self.assertIsNone(row['N'])

    def test_negative_piece_time_and_whole_outer_event_count_reach_finite_kernel(self):
        plan=self.plan('H6',[12]);times=[]
        def finite(tau,K):times.append(tau);return SimpleNamespace(exact_finite_distribution=1+tau*tau)
        out=self.score(plan,finite_distribution=finite)
        self.assertTrue(any(t<0 for t in times))
        for row in out['rows']:self.assertEqual(row['structure']['event_count'],row['q_PF']*row['R_outer'])

    def test_extreme_attenuation_only_defers_and_baseline_is_kept(self):
        plan=self.plan('H4',[12],reference_R={12:[1]})
        out=self.score(plan,finite_distribution=lambda t,K:SimpleNamespace(exact_finite_distribution=1+t*t))
        row=next(r for r in out['rows'] if r['R_outer']==1);baseline=next(r for r in out['rows'] if r['baseline'])
        self.assertFalse(row['priority']);self.assertTrue(row['detailed_N_required']);self.assertIsNone(row['N'])
        self.assertTrue(baseline['priority']);self.assertEqual(len(out['menu']['candidates']),len(plan['menu']['candidates']))
        self.assertIn(row['candidate_id'],out['deferred_ids']);self.assertFalse(out['N_C_optimum_known'])
        self.assertTrue(out['needs_range_extension_review']);self.assertTrue(any(r['upper_R_frontier_open'] for r in out['rows']))
        self.assertIsNone(out['R_upper_bound']);self.assertFalse(out['screening_policy']['scientific_acceptance_threshold'])

    def test_missing_or_unsourced_budget_never_creates_a_default_threshold(self):
        plan=self.plan()
        for budget,source in [(True,{'fixture':'MOCK'}),(1.,{'fixture':'MOCK'}),(float('inf'),{'fixture':'MOCK'}),(2.,{})]:
            with self.assertRaises(ValueError):s.screen_normalization(ROOT,plan,self.tail(plan),shot_inflation_budget=budget,budget_source=source)
        scores=s.screen_normalization(ROOT,plan,self.tail(plan),shot_inflation_budget=None,budget_source=None,finite_distribution=lambda t,K:SimpleNamespace(exact_finite_distribution=1+t*t))
        self.assertEqual(scores['deferred_ids'],[])
        self.assertTrue(all(r['normalization']['relation_to_explicit_budget']=='NOT_CLASSIFIED_NO_BUDGET' for r in scores['rows']))
        for factors in ([],['0'],['1','1'],[.5]):
            with self.assertRaises(ValueError):s.metadata_plan(ROOT,self.base['H4'],[12],factors=factors,sampling_law=LAW,source={'fixture':'MOCK'})

    def test_numerical_normalizer_loss_never_returns_B1_as_a_success(self):
        plan=self.plan('H4',[12]);out=self.score(plan,finite_distribution=lambda t,K:SimpleNamespace(exact_finite_distribution=1.))
        self.assertTrue(all(r['normalization']['status']=='NUMERICAL_NORMALIZATION_UNRESOLVED' for r in out['rows']))
        self.assertTrue(all(r['normalization']['log_B'] is None and r['detailed_N_required'] for r in out['rows']))
        self.assertTrue(all(r['scientific_status']=='NOT_EVALUATED' for r in out['rows']))

    def test_numerical_budget_boundary_is_retained_not_scientific_PASS(self):
        plan=self.plan('H4',[0]);finite=lambda t,K:SimpleNamespace(exact_finite_distribution=1+t*t)
        scores=self.score(plan,finite_distribution=finite);row=next(r for r in scores['rows'] if not r['baseline'])
        F=math.exp(2*row['normalization']['log_B'])
        out=s.screen_normalization(ROOT,plan,self.tail(plan),shot_inflation_budget=F,budget_source={'fixture':'TOY_BOUNDARY'},finite_distribution=finite)
        same=next(r for r in out['rows'] if r['candidate_id']==row['candidate_id'])
        self.assertTrue(same['priority']);self.assertEqual(same['normalization']['relation_to_explicit_budget'],'NUMERIC_BOUNDARY_RETAIN')
        self.assertEqual(same['scientific_status'],'NOT_EVALUATED')

    def test_foreign_tail_law_recipe_or_q_rejected_before_numeric_callback(self):
        plan=self.plan();finite=lambda t,K:(_ for _ in ()).throw(AssertionError('Preflight first'))
        tail=self.tail(plan);tail['binding']=dict(state='FOREIGN')
        with self.assertRaises(ValueError):s.screen_normalization(ROOT,plan,tail,shot_inflation_budget=2,budget_source={'fixture':'MOCK'},finite_distribution=finite)
        for change in [lambda x:x['rows'][0]['request']['law_binding'].update(K=0),lambda x:x['rows'][0].update(q_PF=4),lambda x:x['rows'][0]['structure']['allocation'].__setitem__(0,999)]:
            bad=deepcopy(plan);change(bad)
            with self.assertRaises(ValueError):self.score(bad,finite_distribution=finite)

    def test_A_and_baseline_scientific_sources_are_unchanged(self):
        for path in ['src/trottertracks/pr_r_dependent_n.py','src/trottertracks/pf_pr_rpe_pipeline.py','src/trotterlib/rte.py','src/trottertracks/h6_df19_rpe_d2.py','src/trotterlib/h46_m5_plan.py']:
            self.assertEqual((ROOT/path).read_bytes(),subprocess.check_output(['git','show','5ca2dce3234f3e4a971f4a8c852da676f050e315:'+path],cwd=ROOT))


if __name__=='__main__':unittest.main()
