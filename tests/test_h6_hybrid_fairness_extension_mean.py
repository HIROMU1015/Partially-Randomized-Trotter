"""Arb-vs-tight-L2 containment on four artificial Hybrid inputs only."""
import importlib,json,sys,types,unittest
from pathlib import Path
from fractions import Fraction as Q
ROOT=Path(__file__).resolve().parents[1]
p=types.ModuleType('hybrid_mean_test');p.__path__=[str(ROOT/'src/trotterlib')];sys.modules[p.__name__]=p
E=importlib.import_module(p.__name__+'.h6_hybrid_fairness_extension')
M=importlib.import_module(p.__name__+'.h4_pauli_pr_mean')
FIXTURE=ROOT/'analysis/h6_hybrid_fairness_readiness_20261027/artificial_mean_inputs.json'

class Checks(unittest.TestCase):
    def test_retained_event_phase_exact_norm_and_B_once_enclose_oracle(self):
        for f in json.loads(FIXTURE.read_text()):
            self.assertEqual(f['artificial_qubits'],2)
            law,model,ref=f['law'],f['model'],f['reference']
            got=E.artificial_mean(model,ref,law)
            oracle=M.artificial_mean(law,model['state'],ref,128)
            self.assertEqual(got['raw_norm_exact'],oracle['raw_norm_exact'])
            self.assertEqual(got['B_applied_count'],1)
            self.assertFalse(got['state_renormalized'])
            for key in ('real','imag'):
                a=got['physical_corrected_mean'][key];b=oracle['physical_corrected_mean'][key]
                self.assertLessEqual(Q(a['lower']),Q(b['lower']))
                self.assertGreaterEqual(Q(a['upper']),Q(b['upper']))
    def test_real_dimension_refused_by_artificial_entrypoint_before_evaluation(self):
        with self.assertRaises(ValueError):E.artificial_mean(dict(n_qubits=12),{},dict(n_system=12))

if __name__=='__main__':unittest.main()
