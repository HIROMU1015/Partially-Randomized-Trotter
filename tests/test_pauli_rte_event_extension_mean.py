"""Arb oracle comparisons on synthetic raw rays only."""
import importlib,json,sys,types,unittest
from pathlib import Path
from fractions import Fraction as Q
ROOT=Path(__file__).resolve().parents[1]
p=types.ModuleType('event_mean_test');p.__path__=[str(ROOT/'src/trotterlib')];sys.modules[p.__name__]=p
E=importlib.import_module(p.__name__+'.pauli_rte_event_extension')
M=importlib.import_module(p.__name__+'.h4_pauli_pr_mean')
class Checks(unittest.TestCase):
    def test_both_numeric_bridges_enclose_original_Arb_oracle(self):
        fixtures=json.loads((ROOT/'artifacts/pauli_rte_event_extension_preparation/2026-10-26/artificial_mean_inputs.json').read_text())
        for f in fixtures:
            assert f['artificial_qubits']==2
            law,model,reference=f['law'],f['model'],f['reference'];raw=model['state']
            oracle=M.artificial_mean(law,raw,reference,128)
            for got in [E.h4_mean(law,raw,reference,64),E.h6_mean(model,reference,law)]:
                self.assertEqual(got['B_applied_count'],1);self.assertFalse(got['state_renormalized'])
                self.assertEqual(got['raw_norm_exact'],oracle['raw_norm_exact'])
                for key in ['real','imag']:
                    a=got['physical_corrected_mean'][key];b=oracle['physical_corrected_mean'][key]
                    self.assertLessEqual(Q(a['lower']),Q(b['lower']));self.assertGreaterEqual(Q(a['upper']),Q(b['upper']))
if __name__=='__main__':unittest.main()
