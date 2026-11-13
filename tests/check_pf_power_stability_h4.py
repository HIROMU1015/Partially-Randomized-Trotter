"""Artificial checks for the fixed two-path runner, never a molecular run."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('h4_two_paths',ROOT/'scripts/run_pf_power_stability_h4.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def test_same_candidate_and_scalar_once(self):
        c=dict(T=409.6,q=1024);state=np.array([1.,0.],complex)
        record,_,_=runner.observe_route(np.eye(2),c,state,.17)
        value=complex(record['mean']['real'],record['mean']['imag'])
        self.assertLess(abs(value-np.exp(-1j*c['T']*.17)),1e-14)
        self.assertTrue(record['scalar_once']);self.assertTrue(record['numerical']['gate_pass'])

    def test_failed_route_is_saved_as_values_then_second_route_can_run(self):
        c=dict(T=409.6,q=1024);state=np.array([1.,0.],complex)
        bad,_,_=runner.observe_route(np.diag([1+1e-10,1.]),c,state,0.)
        good,_,_=runner.observe_route(np.eye(2),c,state,0.)
        self.assertEqual(bad['status'],'LOCAL_NUMERICAL_GATE_FAIL')
        self.assertFalse(bad['numerical']['repaired'])
        self.assertGreater(bad['numerical']['power_unitarity_error'],1e-9)
        self.assertEqual(good['status'],'LOCAL_NUMERICAL_GATE_PASS')

    def test_nonfinite_stops_without_fallback(self):
        with self.assertRaises(ValueError):runner.observe_route(np.array([[np.nan]]),dict(q=1,T=.1),np.ones(1),0.)

    def test_fixed_scope_selection_metadata_only(self):
        planned=runner.screen.plan(ROOT);row=runner.selected_cell(planned)
        self.assertEqual(row['candidate']['id'],'4f0aca817edccc7fbe17')
        row['candidate']['q']=4096
        with self.assertRaises(ValueError):runner.selected_cell(dict(fresh=[row]))


if __name__=='__main__':unittest.main()
