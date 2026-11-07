"""Synthetic-only tests of reanalysis definitions and native numerical adapter."""
import unittest
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_track_d_h4_reference as bootstrap
bootstrap.bootstrap(os.environ['TRACK_D_ORIGIN'])
import numpy as np
from trottertracks import track_d_c7_4_models as m


class Models(unittest.TestCase):
    def test_original_pilot_completion_schema_is_distinct(self):
        r=dict(status='PILOT_COMPLETE_MANDATORY_STOP',planned_ids=['x'],completed_ids=['x'],missing_or_failed_ids=[])
        s=dict(status=r['status'],mandatory_stop=True)
        self.assertTrue(m.completed_result(r,s,calibration=True))
        self.assertFalse(m.completed_result(r,s))
        r['missing_or_failed_ids']=['y']
        self.assertFalse(m.completed_result(r,s,calibration=True))

    def test_equal_q_fit_and_rank(self):
        q=np.array([2]*16+[4]*16+[8]*32+[16]*16+[32]*8)
        x=np.column_stack((np.ones(len(q)),q));y=np.column_stack((3+4*q,9-7*q))
        np.testing.assert_allclose(m.fit(x,y,q),[[3,9],[4,-7]],atol=1e-12)
        with self.assertRaises(ValueError):m.fit(np.column_stack((x,q-1)),y,q)

    def test_feature_max_is_pathwise(self):
        app=lambda b:dict(basis_id=b,diagonal_pauli_support=[0],coefficient_sign=1)
        event=lambda b:dict(rotation_angle=.1,application_sequence=[app(b)])
        r=dict(q=4,sample_id='synthetic',ordered_events=[[event(0),event(1)],[event(0),event(0)],
            [event(1),event(2)],[event(0),event(1)]])
        f=m.event_features(r)
        self.assertEqual((f['basis_runs'],f['lane_max_runs'],f['lane_imbalance']),(7,4,1))

    def test_designs_and_output_labels(self):
        fs=[dict(q=q,basis_runs=3*q+q%3,lane_max_runs=2*q+q%5) for q in [2,4,8,16,32]]
        self.assertEqual(len(m.LABELS),36)
        for name in m.CANDIDATES:self.assertEqual(len(m.design(fs,name)),5)
        self.assertEqual(len(m.design(fs,'affine')[0]),2)

    def test_axis_dependence_preserved(self):
        x=np.arange(16.)
        cov=np.cov(np.column_stack((x,x,2*x)),rowvar=False,ddof=1)
        self.assertEqual(np.linalg.matrix_rank(cov),1)
        self.assertEqual(cov[0,0],cov[0,1])


class Native(unittest.TestCase):
    def test_full_matrix_and_actual_inverse(self):
        from qiskit.quantum_info import Operator
        from trottertracks.track_d_c7_4_stability import artificial_block,NativeSmallUnitary,residual
        c=artificial_block(3);engine=NativeSmallUnitary(1)
        self.assertTrue(residual(engine(c),Operator(c).data)['component_PASS'])
        self.assertTrue(residual(engine(c.inverse()),Operator(c).data.conj().T)['component_PASS'])

    def test_reject_large_and_nonunitary(self):
        from qiskit import QuantumCircuit
        from trottertracks.track_d_c7_4_stability import NativeSmallUnitary
        e=NativeSmallUnitary(1)
        with self.assertRaises(ValueError):e(QuantumCircuit(17))
        c=QuantumCircuit(3);c.h(0)
        with self.assertRaises(ValueError):e(c)


if __name__=='__main__':unittest.main()
