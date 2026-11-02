"""Deterministic artificial tests; no random draws or physical signal generation."""
import inspect
import math
from pathlib import Path
import subprocess
import os
import unittest
import numpy as np
from run_saved_v2 import FIXED,selected_functions,select_point,physical_state,axis_tail_cdf_sf

class FixedFunctionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f={}
        env=dict(os.environ,GIT_OPTIONAL_LOCKS='0',GIT_NO_LAZY_FETCH='1')
        repo=str(Path(__file__).resolve().parent)
        for key,names in [('RPE4SRC',('_wrap_phase','_phase_distance','reconstruct_rpe_phase','_vectorized_reconstruction','_oracle_branch')),
                          ('FAILURESRC',('_axis_distribution','_round_exact_phase_probability'))]:
            commit,path,blob=FIXED[key]
            raw=subprocess.check_output(['git','-C',repo,'cat-file','blob',blob],env=env).decode()
            fun,record=selected_functions(raw,names);cls.f.update(fun)
            assert all(r['unchanged_ast'] for r in record.values())

    def test_small_q_scalar_vectorized_match_existing_rule(self):
        q=(1,2,4,8);refs=np.asarray([.22,.72,-.61])
        phases=[np.angle(np.exp(1j*x*refs)) for x in q]
        vector,branches,ambiguous=self.f['_vectorized_reconstruction'](phases,q)
        self.assertFalse(np.any(ambiguous))
        for i,r in enumerate(refs):
            scalar=self.f['reconstruct_rpe_phase']([v[i] for v in phases],q)
            self.assertAlmostEqual(vector[i],scalar['final_phase_estimate'])
            self.assertAlmostEqual(vector[i],r)
            self.assertEqual([x['selected_branch'] for x in scalar['rounds']],[int(b[i]) for b in branches])

    def test_phase_wrap_both_sides(self):
        q=(1,2,4,8)
        for target in (math.pi-.01,-math.pi+.01):
            phases=[float(np.angle(np.exp(1j*x*target))) for x in q]
            result=self.f['reconstruct_rpe_phase'](phases,q)
            self.assertLess(self.f['_phase_distance'](result['final_phase_estimate'],target),1e-12)

    def test_tie_is_reported_ambiguous(self):
        phases=[np.asarray([0.]),np.asarray([math.pi])]
        _v,_b,ambiguous=self.f['_vectorized_reconstruction'](phases,(1,2))
        self.assertTrue(ambiguous[0])
        scalar=self.f['reconstruct_rpe_phase']([0.,math.pi],(1,2))
        row=scalar['rounds'][-1]
        self.assertTrue(np.isclose(row['distance_from_previous'],row['next_smallest_branch_distance'],rtol=0,atol=1e-12))

    def test_invalid_q_sequence_rejected(self):
        with self.assertRaises(ValueError):self.f['reconstruct_rpe_phase']([0.,0.],(1,4))
        with self.assertRaises(ValueError):self.f['reconstruct_rpe_phase']([0.],(2,))

    def test_oracle_is_separate_from_actual_reconstructor(self):
        self.assertEqual(tuple(inspect.signature(self.f['reconstruct_rpe_phase']).parameters),('measured_round_phases','q_values'))
        phases=[.2,.4,.8];q=[1,2,4]
        before=self.f['reconstruct_rpe_phase'](phases,q)
        self.f['_oracle_branch'](np.asarray(.8),4,-2.5)
        self.assertEqual(before,self.f['reconstruct_rpe_phase'](phases,q))

    def test_axis_pmf_tail_known_small_example_and_cdf_sf(self):
        d=self.f['_axis_distribution'](0.,4,.25)
        self.assertEqual(d['shots'],4);self.assertEqual(d['epsilon_coordinate'],.25)
        self.assertAlmostEqual(d['exact_coordinate_failure_probability'],10/16)
        self.assertAlmostEqual(math.fsum(d['_probabilities']),1)
        self.assertAlmostEqual(axis_tail_cdf_sf(d),d['exact_coordinate_failure_probability'])

    def test_round_enumeration_matches_independent_small_product_sum(self):
        co=self.f['_axis_distribution'](.2,3,.3);si=self.f['_axis_distribution'](-.1,2,.3)
        signal=.2-.1j;beta=.4
        expected=0.
        for x,px in zip(co['_sample_means'],co['_probabilities']):
            for y,py in zip(si['_sample_means'],si['_probabilities']):
                z=complex(x,y)
                if z==0 or abs(np.angle(z*signal.conjugate()))>beta:expected+=px*py
        got,_=self.f['_round_exact_phase_probability'](co,si,signal,beta)
        self.assertAlmostEqual(got,expected)

    def test_zero_estimate_counted_as_failure(self):
        axis=self.f['_axis_distribution'](0.,2,.3)
        probability,_=self.f['_round_exact_phase_probability'](axis,axis,1.+0j,math.pi)
        self.assertAlmostEqual(probability,.25)

    def test_mean_range_validation_and_original_roundoff_clipping(self):
        with self.assertRaises(ValueError):self.f['_axis_distribution'](1.01,2,.1)
        rounded=self.f['_axis_distribution'](1.+1e-13,2,.1)
        self.assertEqual(rounded['probability_plus_one'],1.)
        self.assertEqual(rounded['shots'],2)

    def test_tuple_and_state_selection_are_exact(self):
        target={'q_m':'8','r_m':'2','K_m':'2'}
        for k in target:
            with self.assertRaises(ValueError):select_point([{**target,k:'wrong'}],target)
        physical={'state_label':'physical_df_ground_state'};control={'state_label':'effective_partial_s2_eigenstate_control'}
        self.assertIs(physical_state({'state_results':[control,physical]})[1],physical)
        with self.assertRaises(ValueError):physical_state({'state_results':[control]})

if __name__=='__main__':unittest.main()
