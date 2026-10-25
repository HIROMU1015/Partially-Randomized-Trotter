"""New production runner exercised only with supplied artificial input."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("toy_pr_runner",root/"scripts/run_h4_pauli_pr_search.py")
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)


class ExecutionTests(unittest.TestCase):
    def test_actual_price_job_saves_paired_events_and_QPY(self):
        from pr_search_execution.pauli_signal_adapter import PauliHamiltonian
        h=PauliHamiltonian.from_terms([("II",.125),("ZI",.5),("XI",-.25),("YI",.125)])
        c=dict(L_D=1,q=2,r=2,K=4)
        plan=dict(jobs=[dict(candidate=c,cohort="confirmation",index=0)])
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)
            with patch.object(R,"bind",return_value=(plan,run)),patch.object(R,"hamiltonian",return_value=({},h)):
                R.price_job("TOY",0)
            p=run/"prices/p001_q2_r2_K4/confirmation/0000"
            row=R.read(p/"trajectory.json")
            self.assertEqual(len(row["events"]),4)
            self.assertEqual(row["sampler_calls"],1)
            self.assertEqual(row["top_level_compiles"],2)
            self.assertEqual(set(row["metrics"]),{"cosine","sine"})
            self.assertEqual(len(list(p.glob("*.qpy.gz"))),4)
            self.assertEqual(row["seed"],R.S.trajectory_seed(1,2,2,4,"confirmation",0))
            with patch.object(R,"bind",return_value=(plan,run)):
                with self.assertRaises(FileExistsError):R.price_job("TOY",0)


if __name__=="__main__":unittest.main()
