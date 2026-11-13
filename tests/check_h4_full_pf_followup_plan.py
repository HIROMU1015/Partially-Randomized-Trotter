"""Follow-up parameter metadata and artificial log-window checks only."""
from fractions import Fraction
import math
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests'),str(ROOT/'src')]
import prepare_h4_full_pf_followup as P


class FollowupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.plan=P.build()

    def test_exact_rounds_and_finite_first_batch(self):
        p=self.plan
        self.assertEqual(p['metadata_counts']['first_batch_new_actual_steps'],5)
        self.assertEqual(p['metadata_counts']['held_new_actual_steps'],1)
        self.assertEqual(p['actual_scientific_calls'],0);self.assertEqual(p['saved_numerical_reanalysis'],0)
        expected={'S2_16h_over_7':1792,'S2_8h_over_3':1536,'Y4_16h_over_3':768,'Y4_32h_over_5':640,'M5_16h':256,'M5_32h':128}
        for c in p['configurations']:
            self.assertEqual(c['rounds'][12]['q_PF'],expected[c['label']])
            for r in c['rounds']:
                self.assertEqual(P.previous.exact(r['T']),r['q_PF']*P.previous.exact(r['actual_delta_pf']))
        self.assertTrue(all(not c['qualified'] and c['delta_E_ha'] is None for c in p['step_cells']))

    def test_h_relative_request_is_not_replaced_by_display_float(self):
        c=next(c for c in self.plan['configurations'] if c['label']=='Y4_32h_over_5')
        t=P.previous.exact(c['rounds'][5]['T']);width=P.previous.exact(c['requested_delta_pf'])
        self.assertEqual(math.ceil(t/width),5)
        self.assertEqual(math.ceil(t/Fraction(.64)),6)
        early=c['rounds'][4]
        self.assertEqual(early['q_PF'],3);self.assertEqual(P.previous.exact(early['actual_delta_pf']),16*P.previous.exact(self.plan['shared']['h'])/3)

    def test_metadata_build_has_no_scientific_import_or_leaf(self):
        code="import sys;sys.path.insert(0,'scripts');import prepare_h4_full_pf_followup as p;p.build();assert not any(x in sys.modules for x in ('numpy','scipy','qiskit','pyscf'))"
        subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,check=True)

    def test_proposed_window_on_artificial_wrapped_spectrum_keeps_reference_independent(self):
        import numpy as np
        from scipy.linalg import expm
        from check_pf_effective_hamiltonian import evaluate,matrix,SOURCE,REF
        h=np.diag([-2.2,.7]);delta=1.6
        branch=dict(mode='ENERGY_WINDOW',purpose='EFFECTIVE_SPECTRUM_WINDOW',window=[-2.5,1.],source=SOURCE)
        principal=evaluate(expm(-1j*delta*h),delta)
        lifted=evaluate(expm(-1j*delta*h),delta,branch=branch)
        self.assertEqual(lifted['status'],'EFFECTIVE_HAMILTONIAN_DIAGNOSTIC_AVAILABLE')
        self.assertGreater(principal['spectrum']['energies_ha'][0],0.)
        np.testing.assert_allclose(matrix(lifted['H_eff']),h,atol=1e-13)
        other=evaluate(expm(-1j*delta*h),delta,branch=branch,reference=dict(REF,energy=900.))
        self.assertEqual(lifted['H_eff'],other['H_eff']);self.assertFalse(lifted['physical_alias_qualified'])
        invalid=evaluate(expm(-3.2j*h),3.2,branch=branch)
        self.assertEqual(invalid['status'],'UNKNOWN');self.assertIsNone(invalid['H_eff'])


if __name__=='__main__':unittest.main()
