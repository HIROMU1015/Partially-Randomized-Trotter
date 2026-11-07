"""Artificial backend checks; no molecular inputs, raw build or compile."""
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import run_track_d_h4_reference as boot
boot.bootstrap(Path("/tmp/track-d-paired32-20261106/origin"))
from trottertracks import track_d_c8_1_nofusion as repair
from trottertracks import track_d_c7_3_acceleration as fast
from trottertracks import c1_h6_state_action as primitive
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from threadpoolctl import threadpool_limits


class NoFusionTests(unittest.TestCase):
    def circuit(self):
        qc=QuantumCircuit(3)
        for i in range(50):
            qc.rz(.17,0);qc.sx(1);qc.cx(1,2);qc.rz(-.23,2);qc.x(0)
        qc.global_phase=.123
        return qc

    def install(self):
        return repair.install(1,existing_install=fast.install)

    def test_both_official_backends_disable_only_simulation_fusion(self):
        engine=self.install()
        self.assertFalse(engine.backend.options.fusion_enable)
        self.assertIs(primitive.StateActions,repair.UnfusedStateActions)
        states=primitive.StateActions(1)
        self.assertFalse(states.backend.options.fusion_enable)
        self.assertEqual(states.backend.options.precision,"double")
        self.assertEqual(engine.backend.options.precision,"double")

    def test_full_small_unitary_and_declared_phase(self):
        engine=self.install();qc=self.circuit();before=qc.copy()
        with threadpool_limits(limits=1):actual=engine(qc);expected=Operator(qc).data
        np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)
        self.assertEqual(qc,before)

    def test_complete_state_actions_match_full_unitary(self):
        qc=self.circuit()
        with threadpool_limits(limits=1):
            actual=repair.UnfusedStateActions(2).apply(qc,np.eye(8,dtype=complex))
            expected=Operator(qc).data.T
        np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)
        self.assertEqual(actual.shape,(8,8))

    def test_large_full_operator_remains_forbidden(self):
        with self.assertRaisesRegex(ValueError,"FULL_REFERENCE_OPERATOR_FORBIDDEN"):
            self.install()(QuantumCircuit(17))


if __name__=="__main__":unittest.main()
