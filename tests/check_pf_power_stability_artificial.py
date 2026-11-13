"""Artificial stability regressions; no molecular load or saved reanalysis."""
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from trotterlib.h46_m5_plan import weights
from trottertracks import pf_power_stability as D
from trottertracks.h4_four_method_screen_mean import deterministic_step
from trottertracks.pr_rpe_power_mean import kernels


class StabilityTests(unittest.TestCase):
    def test_failed_gate_retains_values_without_repair_or_mutation(self):
        step=np.diag([1+1e-10,1]).astype(complex);before=step.copy();state=np.array([1.,0.])
        powered,row=D.power_diagnostics(step,64,state)
        self.assertFalse(row['gate_pass']);self.assertGreater(row['norm_error'],1e-9)
        self.assertGreater(row['power_unitarity_error'],1e-9)
        self.assertFalse(row['repaired']);self.assertFalse(row['certified'])
        np.testing.assert_array_equal(step,before)
        self.assertGreater(abs(powered[0,0]),1.)

    def test_zero_eigenspace_reconstruction_gate_can_miss_gram_error(self):
        n=36;zero=np.zeros((n,n),complex);v=np.eye(n,dtype=complex);v[0,1]=3e-12
        k=kernels(ROOT)
        # Deliberately injected output; this is not a claimed LAPACK failure.
        with patch.dict(k.spectrum.__globals__,eigh=lambda a,**kw:(np.zeros(n),v)):
            spec=k.spectrum(zero)
        quality=D.spectral_metrics(zero,spec)
        self.assertEqual(quality['reconstruction_fro'],0.)
        self.assertGreater(quality['gram_defect_fro'],1e-12)
        state=np.zeros(n,complex);state[:2]=1/np.sqrt(2)
        ctx=SimpleNamespace(state=state,specs=(spec,),cache={})
        step=deterministic_step(ctx,dict(formula='2nd',prefix=0,T=0.,q=1),weights('2nd'))
        _,early=D.power_diagnostics(step,1,state);_,late=D.power_diagnostics(step,4096,state)
        self.assertTrue(early['gate_pass']);self.assertFalse(late['gate_pass'])

    def test_explicit_driver_is_not_automatic_fallback(self):
        matrices,_=D.artificial_blocks('clustered_zero',34,n=12,count=2)
        for driver in ('evr','evd','ev'):
            spec=D.explicit_spectrum(matrices[0],driver)
            quality=D.spectral_metrics(matrices[0],spec)
            self.assertLess(quality['reconstruction_fro'],1e-12)
        with self.assertRaises(ValueError):D.explicit_spectrum(matrices[0],'auto_retry')
        bad=matrices[0].copy();bad[0,1]+=1j
        with self.assertRaises(ValueError):D.explicit_spectrum(bad,'evd')

    def test_same_signed_order_matches_independent_expm(self):
        matrices,state=D.artificial_blocks('regular',19,n=8,count=3)
        specs=tuple(D.explicit_spectrum(a,'evd') for a in matrices)
        for formula in ('2nd','4th','4th(m5_best)'):
            self.assertEqual(tuple(weights(formula)),tuple(reversed(weights(formula))))
            for delta in (.1,.4):
                ctx=SimpleNamespace(state=state,specs=specs,cache={})
                step=deterministic_step(ctx,dict(formula=formula,prefix=2,T=delta,q=1),weights(formula))
                ref=D.expm_step(matrices,weights(formula),delta)
                a,row=D.power_diagnostics(step,64,state);b,_=D.power_diagnostics(ref,64,state)
                self.assertLess(np.linalg.norm(a-b,'fro'),1e-10)
                self.assertTrue(row['gate_pass'])

    def test_nonfinite_input_and_invalid_counts_are_not_repaired(self):
        step=np.eye(2,dtype=complex);state=np.array([1.,0.])
        for q in (0,1.5):
            with self.assertRaises(ValueError):D.power_diagnostics(step,q,state)
        with self.assertRaises(ValueError):D.power_diagnostics(step,2,state*2)
        step[0,0]=np.nan
        with self.assertRaises(ValueError):D.power_diagnostics(step,2,state)

    def test_artificial_grid_has_no_scientific_qualification(self):
        path=ROOT/'docs/research/pf_power_stability_artificial_20261113/RESULT.json'
        r=json.loads(path.read_text());self.assertEqual(len(r['records']),192)
        keys={(v['fixture'],v['backend'],v['formula'],v['delta'],v['q']) for v in r['records']}
        self.assertEqual(len(keys),192)
        self.assertEqual(r['real_H4_calls'],0);self.assertEqual(r['saved_numerical_reanalysis'],0)
        self.assertFalse(r['production_changed']);self.assertIsNone(r['scientific_qualification'])
        self.assertTrue(all(v['threshold']==1e-9 and not v['repaired'] and not v['certified'] for v in r['records']))

    def test_cli_rejects_real_input_and_preserves_existing_output(self):
        script='scripts/check_pf_power_stability_artificial.py'
        with tempfile.TemporaryDirectory(prefix='pf-stability-cli-') as tmp:
            path=Path(tmp)/'existing.json';path.write_text('KEEP')
            for extra in ([],['--hamiltonian','real.npz']):
                p=subprocess.run([sys.executable,'-B',script,'--out',str(path),*extra],cwd=ROOT,capture_output=True)
                self.assertNotEqual(p.returncode,0)
            self.assertEqual(path.read_text(),'KEEP')

    def test_frozen_screen_and_old_runs_are_preserved(self):
        files=('src/trottertracks/h4_four_method_screen.py','src/trottertracks/h4_four_method_screen_mean.py')
        for file in files:
            expected=subprocess.check_output(['git','show','256caad3414a80de02fb68eca89c13c957ed6354:'+file],cwd=ROOT)
            self.assertEqual((ROOT/file).read_bytes(),expected)
        base='docs/research/h4_four_method_signal_screen_results_20261113/'
        for run,commit in (('run01','37e30194b0fca129a35f651ee260c66618bbb7e4'),('run02','699178305cec241b83220f6c307c65e0539a7856')):
            for path in (ROOT/base/run).glob('*.json'):
                expected=subprocess.check_output(['git','show',commit+':'+str(path.relative_to(ROOT))],cwd=ROOT)
                self.assertEqual(path.read_bytes(),expected)


if __name__=='__main__':unittest.main()
