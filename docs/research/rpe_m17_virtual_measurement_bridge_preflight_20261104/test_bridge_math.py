"""Artificial scalars/operators only; no real-data numerical calls or RNG."""
from fractions import Fraction as Q
import contextlib
import io
import itertools
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import bridge_math as bridge

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'docs/research/rpe_fresh_iid_measurement_a_preparation_20261103'))
import fresh_iid_adapter as legacy
import run_fresh_iid_a_v2 as old_runner


class BridgeTests(unittest.TestCase):
    def test_raw_vector_vs_normalized_ray(self):
        state=np.array([2.,0.],complex)
        op=np.diag([1j,1.])
        raw=complex(np.vdot(state,op@state))
        normalized=raw/np.vdot(state,state)
        self.assertEqual(raw,4j);self.assertEqual(normalized,1j)
        with self.assertRaises(ValueError):legacy.conditional_probability(raw,'sine')

    def test_sine_sign_and_accepted_clipping(self):
        raw,used,p,changed=legacy.conditional_probability(.3-.2j,'sine')
        self.assertEqual(raw,-.2);self.assertEqual(used,-.2)
        self.assertEqual(p,(1+used)/2);self.assertFalse(changed)
        raw,used,p,changed=legacy.conditional_probability(1+2e-13,'cosine')
        self.assertTrue(changed);self.assertEqual(used,1.);self.assertEqual(p,1.)

    def test_guards_before_clipping_define_abort(self):
        # Inside MATRIX_ATOL but outside coordinate allowance: no unconditional clip.
        with self.assertRaises(ValueError):legacy.conditional_probability(1+2e-12,'cosine')
        # Each coordinate is in [-1,1], yet complex magnitude guard rejects.
        with self.assertRaises(ValueError):legacy.conditional_probability(.8+.8j,'cosine')

    def test_projection_nonexpansive_against_true_bounded_mean(self):
        for raw,true in itertools.product([Q(-3,2),Q(-1),Q(1,4),Q(1),Q(3,2)],[Q(-1),Q(0),Q(1)]):
            clipped=min(Q(1),max(Q(-1),raw))
            self.assertLessEqual(abs(clipped-true),abs(raw-true))

    def test_accepted_probability_rounding_bound(self):
        values=[-1., math.nextafter(-1.,0.),-.75,-.2,0.,2.**-100,.1,.5,math.nextafter(1.,0.),1.]
        u=Q(1,1<<53)
        for value in values:
            p=(1+value)/2
            ideal=(1+Q.from_float(value))/2
            self.assertLessEqual(2*abs(Q.from_float(p)-ideal),u)
            self.assertLessEqual(abs(bridge.bernoulli_grid_mean(p)-Q.from_float(value)),bridge.postprocess_rounding_bound())

    def test_strict_bernoulli_threshold_and_grid_endpoints(self):
        for p in [Q(0),Q(1,16),Q(1,3),Q(1)]:
            enumerated=sum((1 if Q(k,16)<p else -1) for k in range(16))/Q(16)
            self.assertEqual(bridge.bernoulli_grid_mean(p,bits=4),enumerated)
        self.assertEqual(bridge.bernoulli_grid_mean(.5),0)
        # A <= threshold would count the zero grid word at p=0.
        self.assertNotEqual(bridge.bernoulli_grid_mean(0,bits=4),Q(-1)+Q(2,16))

    def test_two_distinct_rounding_contributions(self):
        t=.1;p=(1+t)/2
        probability_contribution=(2*Q.from_float(p)-1)-Q.from_float(t)
        grid_contribution=bridge.bernoulli_grid_mean(p)-(2*Q.from_float(p)-1)
        self.assertEqual(probability_contribution+grid_contribution,bridge.bernoulli_grid_mean(p)-Q.from_float(t))
        self.assertGreaterEqual(grid_contribution,0)

    def test_fixed_word_is_not_an_iid_law(self):
        word=Q(1,4);p=.5
        fixed_outcome=1 if word<Q.from_float(p) else -1
        self.assertEqual(fixed_outcome,1)
        self.assertEqual(bridge.bernoulli_grid_mean(p),0)
        self.assertNotEqual(fixed_outcome,bridge.bernoulli_grid_mean(p))

    def test_nonlinear_float_program_cannot_average_before_rounding(self):
        large=float(1<<53)
        f=lambda x:(large+x)-large
        averaged_outputs=(f(0.)+f(2.))/2
        output_of_average=f((0.+2.)/2)
        self.assertNotEqual(averaged_outputs,output_of_average)

    def test_order_averaged_growth_majorant(self):
        forward,reverse=Q(101,100),Q(102,100)
        g,mass=[Q(101,100),Q(103,100)],[Q(3,4),Q(1,4)]
        # r=2, q=2: enumerate artificial order words only, no physical events.
        explicit=Q(0)
        for word in itertools.product([0,1],repeat=4):
            probability=math.prod(mass[i] for i in word)
            gain=(forward*reverse)**2*math.prod(g[i] for i in word)
            explicit+=probability*gain
        self.assertEqual(bridge.average_gain(forward,reverse,g,mass,r=2,q=2),explicit)

    def test_unknown_bridge_or_abort_domain_stays_null(self):
        for arithmetic,completion in [(None,True),(Q(1,100),False),(Q(1,100),None)]:
            self.assertIsNone(bridge.combine_virtual_scope(Q(1,10),arithmetic,completion_proved=completion)['U_virtual_IID'])
        result=bridge.combine_virtual_scope(Q(1,10),Q(1,100),completion_proved=True)
        self.assertIsNotNone(result['U_virtual_IID'])
        self.assertFalse(result['fixed_PCG64_IID_certified'])

    def test_strict_comparison_and_not_a_failure_proof(self):
        self.assertEqual(bridge.strict_budget(Q(1,10),Q(1,10),Q(1,5)),'NOT_CERTIFIED_BY_THIS_BOUND')
        self.assertEqual(bridge.strict_budget(Q(1,10),Q(1,10),Q(1,4)),'SUFFICIENT_CONDITION_CONFIRMED')
        self.assertEqual(bridge.strict_budget(None,Q(1,10),Q(1,4)),'UNDETERMINED')

    def test_legacy_action_order_and_exact_event_sign_on_toy(self):
        functions=legacy.legacy_functions()
        x=np.array([[0,1],[1,0]],complex);z=np.diag([-1.,1.]).astype(complex)
        event=SimpleNamespace(product_component_ids=('x',),rotation_component_id='z',rotation_angle=0.,phase=-1.)
        state=np.array([1.,0.],complex)
        result=functions['_apply_event_to_state'](event,{'x':x,'z':z},state)
        np.testing.assert_array_equal(result,-x@state)
        f=np.array([[1,.25],[0,1]],complex);r=np.array([[1,0],[.5,1]],complex)
        got=functions['_trajectory_signal'](initial_state=state,forward=f,reverse=r,events=[event],operators={'x':x,'z':z},q_m=1,rte_steps=1)
        self.assertEqual(got,complex(np.vdot(state,r@(-x)@f@state)))

    def test_default_legacy_cli_cannot_start_science(self):
        with patch.object(old_runner,'prepare_context',side_effect=AssertionError('No H4')),patch.object(old_runner,'execute',side_effect=AssertionError('No science')),patch.object(np,'load',side_effect=AssertionError('No arrays')),patch.object(np.random,'PCG64',side_effect=AssertionError('No RNG')),contextlib.redirect_stdout(io.StringIO()):
            old_runner.main([])

    def test_invalid_pure_helper_inputs(self):
        for p in [-.1,1.1,float('nan'),True]:
            with self.assertRaises(ValueError):bridge.bernoulli_grid_mean(p)
        with self.assertRaises(ValueError):bridge.average_gain(1,1,[1],[Q(1,2)],r=1,q=1)
        with self.assertRaises(ValueError):bridge.strict_budget(-1,1,2)


if __name__=='__main__':
    unittest.main()
