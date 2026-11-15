"""Metadata and toy two-dimensional tests; no real H4 evaluations or Q3."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_p7_y4_coarse_signal_q3 as R


class CoarseWiring(unittest.TestCase):
    def test_fixed_metadata_and_source_bound_seven(self):
        prepared,inputs,saved=R.signal_plan(dict(fixture='METADATA_ONLY'))
        self.assertEqual(prepared['configuration']['id'],R.CID)
        self.assertEqual([r['m'] for r in inputs['new']],list(range(6,12)))
        self.assertEqual([r['q'] for r in inputs['new']],[10,20,40,80,160,320])
        self.assertEqual(len(saved),7);self.assertEqual(len(inputs['steps']),1)
        self.assertTrue(all(s['numerical']['gates_passed'] for s in saved))
        self.assertEqual(inputs['configuration']['spec']['L_D'],7)

    def test_original_numerical_scalar_once_on_toy_two_dimensional_step(self):
        import numpy as np
        F,_=R.signal_modules()
        step=np.diag(np.exp(-1j*np.array([.07,-.11])))
        state=np.array([1.,0.],dtype=complex)
        context=SimpleNamespace(root='TOY',state=state,constant=.23,metadata=dict(source='TOY'),
            pf_step=lambda c,w:step,tail_scalar=lambda prefix:.19)
        reference=dict(source={'TOY':1},binding={'TOY':1},T=.8,
            reference={'real':1.,'imag':0.},independent_reference={'real':1.,'imag':0.},signal_difference=0.)
        request=dict(domain=dict(reference={k:deepcopy(reference[k]) for k in ('source','binding','T')}),
            physical_candidate=dict(T=.8,q=4,method='B0I',prefix=7),comparison_method='TRUNCATED_PF',
            structure=dict(signed_weights=[1.,-1.,1.]))
        def power(a,q,v):return np.linalg.matrix_power(a,q),dict(gate_pass=True,norm_gate_pass=True,unitarity_gate_pass=True,repaired=False)
        with patch.object(F.N,'kernels',return_value=SimpleNamespace(power_diagnostics=power)):
            result=F.N.evaluate(context,request,reference,source={'fixture':'TOY_TWO_DIMENSIONAL'})
        actual=complex(result['signal']['raw']['real'],result['signal']['raw']['imag'])
        self.assertEqual(result['status'],'NUMERICAL_MEAN_AVAILABLE')
        self.assertLess(abs(actual-np.exp(-1j*(.07*4+.8*(.23+.19)))),1e-14)
        self.assertTrue(result['numerical']['scalar_once'])

    def test_q3_runtime_entry_in_fresh_process_is_pinned_and_passive(self):
        code=('import sys;sys.path.insert(0,"scripts");import run_p7_y4_coarse_signal_q3 as R;'
              'F=R.q3_runtime();F.check_source();'
              'print("PINNED_Q3_RUNTIME_READY_NO_EVALUATION",F.count_bounds([dict(m=m,N_X=n,N_Y=n) for m,n in enumerate(R.VECTOR)]))')
        result=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('PINNED_Q3_RUNTIME_READY_NO_EVALUATION',result.stdout)

    def test_bound_view_retains_values_gates_and_numerical_origins(self):
        folder=ROOT/R.STUDY
        original=json.loads((folder/'signal_run01/B_SIGNAL_INPUT.json').read_text())
        view=json.loads((folder/'BOUND_SIGNAL_INPUT.json').read_text())
        provenance=json.loads((folder/'BINDING_PROVENANCE.json').read_text())
        self.assertEqual(provenance['new_signal_evaluations'],0)
        for old,new,origin in zip(original['means'],view['means'],provenance['records'],strict=True):
            for field in ('raw','corrected','reference','normalization','bias','uncertainty'):
                self.assertEqual(old[field],new[field])
            self.assertEqual(new['source'],view['source'])
            self.assertEqual(new['provider_binding']['input_source']['original_numerical_source'],old['source'])
            self.assertEqual(origin['original_numerical_source'],old['source'])


if __name__=='__main__':unittest.main()
