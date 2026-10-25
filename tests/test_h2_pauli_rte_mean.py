"""Only synthetic 2qubit extended-law oracle checks."""
import importlib,sys,types,json,unittest
from fractions import Fraction as Q
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=types.ModuleType('h2_mean_test');p.__path__=[str(ROOT/'src/trotterlib')];sys.modules[p.__name__]=p
V=importlib.import_module(p.__name__+'.pauli_vector_tight_enclosure');M=importlib.import_module(p.__name__+'.h4_pauli_pr_mean')
class Checks(unittest.TestCase):
    def test_extended_effective_law_contains_original_Arb_oracle(self):
        fs=json.loads((ROOT/'artifacts/h2_pauli_rte_validation/2026-10-26/artificial_mean_inputs.json').read_text())
        for f in fs:
            assert f['artificial_qubits']==2 and f['model']['n_qubits']==2
            law,model=f['law'],f['model'];ref=V.reference_enclosure(dict(model,energy=.2,Hamiltonian_hash='toy'))
            got=V.evaluate(model,ref,law['config'],law);oracle=M.artificial_mean(law,model['state'],ref,128)
            self.assertEqual(got['B_applied_count'],1);self.assertFalse(got['state_renormalized']);self.assertEqual(got['raw_norm_exact'],oracle['raw_norm_exact'])
            for k in ['real','imag']:
                self.assertLessEqual(Q(got['physical_corrected_mean'][k]['lower']),Q(oracle['physical_corrected_mean'][k]['lower']))
                self.assertGreaterEqual(Q(got['physical_corrected_mean'][k]['upper']),Q(oracle['physical_corrected_mean'][k]['upper']))
if __name__=='__main__':unittest.main()
