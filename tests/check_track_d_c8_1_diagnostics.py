"""Failure persistence and historical acceptance, with artificial arrays only."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import run_track_d_h4_reference as boot
boot.bootstrap(Path("/tmp/track-d-paired32-20261106/origin"))
from trottertracks import track_d_c8_1_diagnostics as d
from trottertracks import c1_stage3_molecular as molecular


class DiagnosticsTests(unittest.TestCase):
    def fixture(self,error=0.,mode="observable"):
        target=np.eye(4,dtype=complex)
        m=SimpleNamespace(ATOL=1e-10,SIGNAL_ATOL=2e-10,
            artificial=SimpleNamespace(expected_observable=lambda v,axis:target),
            observable_from_columns=lambda columns,dim:target+error*np.ones((4,4)),
            small_operator=lambda circuit:target)
        class Checker:
            def check_compiled(self,*args):
                m.small_operator(None)
                if mode=="component":molecular.component_check(target+error,target,m.ATOL,label="component")
                obs=m.observable_from_columns(target,2)
                row=molecular.component_check(obs,target,m.ATOL,label="CAR")
                if row["frobenius"]>m.SIGNAL_ATOL:raise ValueError("CAR:observable_signal_allowance")
                return obs,dict(status="PASS")
        return m,Checker(),dict(V=target,raw_symmetric_directional_X=target)

    def run_fixture(self,error,mode="observable"):
        m,checker,arrays=self.fixture(error,mode)
        before=(molecular.component_check,m.small_operator,m.observable_from_columns)
        with tempfile.TemporaryDirectory() as t:
            output=Path(t)/"new"
            report=d.diagnose(checker,None,"symmetric_directional","X",arrays,m,1,output)
            self.assertEqual(before,(molecular.component_check,m.small_operator,m.observable_from_columns))
            self.assertTrue((output/"partial_checks.json").exists())
            self.assertTrue((output/"small_matrix.npz").exists())
            if mode=="observable":
                self.assertTrue((output/"observable.npz").exists())
                self.assertEqual(len(report["observable_residuals"]),2)
            return report

    def test_pass_unchanged(self):
        self.assertEqual(self.run_fixture(1e-12)["status"],"PASS")

    def test_signal_failure_preserves_exact_values(self):
        report=self.run_fixture(8e-11)
        self.assertEqual(report["status"],"FAIL")
        self.assertGreater(report["observable_residuals"][0]["frobenius"],2e-10)
        self.assertLess(report["observable_residuals"][0]["max_entry"],1e-10)
        self.assertEqual(report["signal_atol"],2e-10)
        self.assertFalse(report["acceptance_changed"])

    def test_component_failure_retained_and_restored(self):
        report=self.run_fixture(3e-10,"component")
        self.assertEqual(report["status"],"FAIL")
        self.assertEqual(report["checks"][0]["status"],"FAIL")

    def test_nonfinite_does_not_pass(self):
        self.assertEqual(d.residual(np.array([[np.nan]]),np.eye(1),1e-10,"bad")["status"],"FAIL")

    def test_shape_mismatch_does_not_pass(self):
        self.assertEqual(d.residual(np.eye(2),np.eye(1),1e-10,"bad")["status"],"FAIL")


if __name__=="__main__":unittest.main()
