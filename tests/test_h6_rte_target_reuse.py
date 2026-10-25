"""Artificial check of the reused tight engine against the old Arb oracle."""
import importlib, json, sys, types, unittest
from fractions import Fraction as Q
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=types.ModuleType('fair_target_test');p.__path__=[str(ROOT/'src/trotterlib')];sys.modules[p.__name__]=p
T=importlib.import_module(p.__name__+'.pauli_vector_tight_enclosure')
M=importlib.import_module(p.__name__+'.h4_pauli_pr_mean')
class Checks(unittest.TestCase):
    def test_tight_effective_law_contains_Arb_oracle(self):
        fixtures=json.loads((ROOT/'artifacts/pauli_h2_h6_comparison/2026-10-25/artificial/pr_fixtures.json').read_text())
        for f in fixtures:
            law=f['law'];ref=dict(f['reference'],method='ARTIFICIAL_REFERENCE')
            model=dict(state=f['raw'],n_qubits=law['n_system'],hamiltonian=dict(constant=0.,terms=[(x['label'],x['coefficient']) for x in law['retained']]))
            got=T.evaluate(model,ref,law['config'],law);oracle=M.artificial_mean(law,f['raw'],ref,96)
            self.assertEqual(got['B_applied_count'],1)
            self.assertFalse(got['state_renormalized'])
            for key in ['real','imag']:
                a=got['physical_corrected_mean'][key];b=oracle['physical_corrected_mean'][key]
                self.assertLessEqual(Q(a['lower']),Q(b['lower']))
                self.assertGreaterEqual(Q(a['upper']),Q(b['upper']))
if __name__=='__main__':unittest.main()
