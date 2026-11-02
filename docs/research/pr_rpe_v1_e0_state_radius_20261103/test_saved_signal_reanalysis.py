"""Artificial scalar/metadata tests; no scientific runner imports or execution."""
import cmath
import math
import unittest
from reanalyze_saved_signals import (
    CONTROL,PHYSICAL,Evidence,comparison,decode,finite,phase_delta,saved_complex,
    select_point,select_state,signal_record,tuple_key,
)

class SavedSignalTests(unittest.TestCase):
    def test_negative_time_phase_and_reversed_comparison_sign(self):
        self.assertAlmostEqual(phase_delta(-1j,1),-math.pi/2)
        self.assertAlmostEqual(phase_delta(1,-1j),math.pi/2)
        self.assertAlmostEqual(abs(phase_delta(-1j,1)),math.pi/2)

    def test_principal_phase_is_circular_and_does_not_unwrap(self):
        angle=0.37
        self.assertAlmostEqual(phase_delta(cmath.exp(1j*(angle+2*math.pi)),1),angle)
        self.assertGreater(phase_delta(-1+1e-12j,1),0)
        self.assertLess(phase_delta(-1-1e-12j,1),0)

    def test_positive_normalization_changes_radius_not_phase(self):
        state={'state_label':PHYSICAL,'pf_signal':{'real':'0.6','imag':'0.8'},
               'attenuated_event_mean_signal':{'real':'0.3','imag':'0.4'}}
        pf=signal_record(state,'pf_signal',1)
        mean=signal_record(state,'attenuated_event_mean_signal',1)
        self.assertAlmostEqual(pf['reanalyzed_radius'],1)
        self.assertAlmostEqual(mean['reanalyzed_radius'],0.5)
        self.assertAlmostEqual(pf['signed_principal_residual_to_common_E0'],mean['signed_principal_residual_to_common_E0'])

    def test_tuple_join_rejects_each_changed_parameter_and_duplicates(self):
        target={'q_m':'8','r_m':'2','K_m':'2'}
        for field in target:
            with self.assertRaises(ValueError):select_point([{**target,field:'different'}],target)
        with self.assertRaises(ValueError):select_point([target,target],target)
        self.assertEqual(select_point([target],target),(0,target))
        self.assertEqual(tuple_key(target),('8','2','2'))

    def test_model_LD_context_mismatch_is_rejected_before_signal_evaluation(self):
        evidence=Evidence.__new__(Evidence)
        evidence.data={'ABLGRID':{},'ABLJSON':{},'OPT2JSON':{'coherent_reoptimization':{'best_by_ld':{'3':{'ld':'12','delta_time':'0.02'}}}}}
        with self.assertRaisesRegex(ValueError,'LD/delta mismatch'):
            evidence.round('3',0)

    def test_state_join_is_label_based_not_array_position(self):
        p={'state_label':PHYSICAL};c={'state_label':CONTROL}
        point={'state_results':[c,p]}
        self.assertIs(select_state(point,PHYSICAL)[1],p)
        self.assertIs(select_state(point,CONTROL)[1],c)
        with self.assertRaises(ValueError):select_state(point,'invented_state')
        with self.assertRaises(ValueError):select_state({'state_results':[p,p]},PHYSICAL)

    def test_missing_or_confused_signal_kind_never_falls_back(self):
        state={'state_label':PHYSICAL,'pf_signal':{'real':'1','imag':'0'}}
        for kind in ('corrected_finite_rte_signal','signal','UNKNOWN'):
            with self.assertRaises(ValueError):signal_record(state,kind,1)
        with self.assertRaises(ValueError):signal_record({**state,'state_label':'UNKNOWN'},'pf_signal',1)
        with self.assertRaises(ValueError):saved_complex({'real':'1','imaginary':'0'})

    def test_zero_missing_and_nonfinite_values_remain_unresolved(self):
        self.assertIsNone(phase_delta(0,1));self.assertIsNone(phase_delta(1,0))
        self.assertEqual(comparison(None,'0.1',1e-10)['status'],'UNRESOLVED')
        for v in (None,float('nan'),float('inf'),True):
            with self.assertRaises(ValueError):finite(v)
        self.assertIsNone(decode('{"x":null}')['x'])
        self.assertEqual(decode('{"x":0.02}')['x'],'0.02')

    def test_budget_and_allowance_are_reported_separately(self):
        c=comparison(0.10000000001,'0.1',1e-10)
        self.assertFalse(c['within_without_allowance'])
        self.assertEqual(c['status'],'NUMERICALLY_CONSISTENT')
        self.assertGreater(c['excess'],0)
        self.assertEqual(comparison(0.2,'0.1',1e-10)['status'],'INCONSISTENT')
        self.assertEqual(comparison(0.05,'0.1',1e-10)['status'],'NUMERICALLY_CONSISTENT')

if __name__=='__main__':unittest.main()
