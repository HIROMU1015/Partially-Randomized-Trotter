import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h46_pr_pilot_evaluation as e

def observations(constant=False):
    return [dict(trajectory_id=str(i),candidate_id='TEST',campaign_id='TOY',cohort=e.design.COHORT,
      conditional=False,paired=True,cosine_RZ=100 if constant else 100+i,
      sine_RZ=200 if constant else 200+2*i,stratum='S0') for i in range(33)]
def law(p=(1.,0.,0.),steps=32):return dict(orders=[0,2,4],K=4,probabilities=list(p),steps=steps)

class EvaluationTests(unittest.TestCase):
    def test_mixture_enumeration_two_steps(self):
        m=e.mixture([law((.5,.25,.25),2)],33)
        self.assertEqual(m['probabilities'],dict(S0=.25,S2_single=.25,S2_multi=.0625,S4=.4375))
        self.assertFalse(m['guarantee']);self.assertIsNone(m['population_cost_bound'])
    def test_tiny_rare_probability_not_zeroed(self):
        m=e.mixture([law((1.,0.,1e-30),32)],33)
        self.assertGreater(m['probabilities']['S4'],0)
        self.assertAlmostEqual(m['probabilities']['S4']/32e-30,1.)
        self.assertGreater(m['ideal_probability_at_least_one_in_n']['S4'],0)
    def test_bad_saved_laws_rejected(self):
        for value in [law((.5,.5,.1)),law((1.,-.1,.1)),law((float('nan'),0.,0.)),law(steps=0),dict(law(),K=2)]:
            with self.assertRaises(ValueError):e.mixture([value],33)
    def test_paired_covariance_and_direct_variance(self):
        v=e.describe(observations(),[law()])
        self.assertEqual(v['axis_means'],dict(cosine=116,sine=232))
        self.assertAlmostEqual(v['covariance'][0][1],2*v['covariance'][0][0])
        self.assertAlmostEqual(v['pair_unit_RZ_variance'],9*v['covariance'][0][0])
        self.assertEqual(len(v['n_E_sensitivity']),9)
    def test_negative_covariance_retained(self):
        rows=observations()
        for i,row in enumerate(rows):row['sine_RZ']=200-i
        v=e.describe(rows,[law()]);self.assertEqual(v['pair_unit_RZ_variance'],0)
        self.assertLess(v['covariance'][0][1],0)
    def test_unobserved_null_and_zero_variance_not_adopted(self):
        v=e.describe(observations(True),[law((.999,.0009,.0001))])
        self.assertIsNone(v['strata']['S4']['observed_conditional_pair_mean'])
        self.assertIsNone(v['strata']['S4']['conditional_variance'])
        self.assertTrue(all(s['n_E_nominal_only'] is None and not s['adopted'] for s in v['n_E_sensitivity']))
        self.assertGreater(v['ideal_unobserved_strata_probability_mass'],0)
        self.assertFalse(v['population_rank_guaranteed'])
    def test_bad_cohort_pairing_and_partial_data_rejected(self):
        rows=observations()
        with self.assertRaises(ValueError):e.describe(rows[:-1],[law()])
        for key,value in [('cohort','EXPLORATION'),('paired',False),('conditional',True),('trajectory_id','1')]:
            values=copy.deepcopy(rows);values[0][key]=value
            with self.assertRaises(ValueError):e.describe(values,[law()])
    def test_family_does_not_drop_unresolved_q(self):
        groups=[]
        for candidate in e.design.groups():
            if candidate['system']=='H4':groups.append(dict(candidate=candidate,
              statistics=e.describe(observations(candidate['q']==1),[law()])))
        values=e.family_sensitivity(groups)
        self.assertTrue(all(s['common_n_E_nominal_only'] is None for g in values for s in g['scenarios']))
    def test_invalid_grant_refuses_before_inputs(self):
        with patch.object(e,'read',side_effect=AssertionError('input')):
            with self.assertRaisesRegex(ValueError,'EXPLICIT_SAVED_PILOT'):e.authorize({})
    def make_task(self,root):
        candidate=next(v for v in e.design.groups() if v['system']=='H4' and v['formula']=='2nd' and v['q']==1)
        task=dict(id='TOY',candidate=candidate,sample=0)
        names=['saved_normalization_log_B','native_low','native_leakage','independent_occupation_vector',
          'hadamard_cosine','compiled_vector_cosine','hadamard_sine','compiled_vector_sine']
        result=dict(task=task,SOURCE='TOY',plan_sha256='PLAN',status='UNIT_COST_PASS_NOT_ACCOUNTING',
          native_scientific_PASS=True,byte_restoration_PASS=True,paired_trajectory=True,seed_cohort=e.design.COHORT,
          conditional_sampling=False,cohort_pooling=False,shot_accounting=False,cost_ranking=False,
          input_binding={'synthetic':True},preparation_hash='PREP',event_digest='EVENT',
          events=dict(N2=0,N4=0,stratum='S0'),
          checks=[dict(check=n,PASS=True,error=0.,strictly_below=1e-7 if n.startswith('compiled_vector') else 1e-9,phase_alignment=False) for n in names],
          axis_costs=[dict(axis=a,compiler_settings_hash=e.design.COMPILER_HASH,native_policy=e.design.NATIVE_POLICY,event_digest='EVENT',metrics=dict(rz_count=v)) for a,v in [('cosine',100),('sine',200)]])
        p=root/'result.json';p.write_text(json.dumps(result));lp=root/'law.json'
        lp.write_text(json.dumps(dict(piece_request_index=0,cohort=e.design.COHORT,conditioned=False,event_count=32,rte_steps=32,signed_step_time=.8,
          law=dict(orders=[0,2,4],order_probabilities=[1.,0.,0.],finite_taylor_order=4,dimensionless_step_time=.01,exact_finite_distribution=1.))))
        index=dict(result_path=str(p),result_sha256=e.sha(p),task=task,SOURCE='TOY',plan_sha256='PLAN',
          input_binding={'synthetic':True},preparation_hash='PREP',law_paths=[str(lp)])
        return index,result
    def test_saved_reader_and_input_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            index,result=self.make_task(Path(tmp));r,l=e.load_task(index,'TOY')
            self.assertEqual((r['cosine_RZ'],r['sine_RZ']),(100,200));self.assertEqual(len(l),1)
            p=Path(index['result_path']);result['SOURCE']='WRONG';p.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError,'SHA_IDENTITY'):e.load_task(index,'TOY')
    def test_reader_rejects_wrong_pair_and_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            index,original=self.make_task(Path(tmp));p=Path(index['result_path'])
            for change in ('axis','threshold','cohort'):
                r=copy.deepcopy(original)
                if change=='axis':r['axis_costs'][1]['axis']='cosine'
                if change=='threshold':r['checks'][0]['strictly_below']=1e-6
                if change=='cohort':r['seed_cohort']='EXPLORATION'
                p.write_text(json.dumps(r));index['result_sha256']=e.sha(p)
                with self.assertRaises(ValueError):e.load_task(index,'TOY')
    def test_stage_one_shot_synthetic_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'out';candidates=[v for v in e.design.groups() if v['system']=='H4']
            cfg=dict(run_root=str(root),SOURCE='TOY',stage=e.STAGES[0],plan_sha256='PLAN',grant={},
              plan=dict(groups=[dict(candidate=v) for v in candidates],campaign_id='TOY'))
            def fake(g,campaign):return dict(candidate=g['candidate'],statistics=e.describe(observations(),[law()]))
            with patch.object(e,'authorize'),patch.object(e,'verify'),patch.object(e,'evaluate_group',side_effect=fake):
                answer=e.execute_stage(cfg)
                self.assertIn('COMPLETE',answer['status'])
                with self.assertRaisesRegex(ValueError,'ONE_SHOT'):e.execute_stage(cfg)
            out=e.read(root/'evaluation.json');self.assertFalse(out['ranking']);self.assertFalse(out['n_E_adopted'])

if __name__=='__main__':unittest.main()
