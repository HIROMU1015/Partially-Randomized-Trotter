"""Two-mode artificial full wrapper: original RTE, scalar/control and compiler."""
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src/trotterlib')]
import pr_cost_contract as c
import pr_cost_native as n


class ArtificialNative(unittest.TestCase):
    def test_two_mode_paired_native_compile_and_event_control(self):
        import numpy as np
        import qiskit
        from trotterlib.df_hamiltonian import DFHamiltonian
        from trotterlib.pr2_matched_accuracy_m1_execution import _prepare
        from trotterlib.rte_compiled_cost import compiler_settings_hash
        from trottertracks.resource_applicability import h6_matched_port_v1 as m
        from trottertracks.resource_applicability import ax2b_native_df_v5 as native
        ham=DFHamiltonian(.17,np.array([[.2,.04],[.04,-.3]]),np.array([.4,-.2]),
            (np.array([[.6,.1],[.1,-.2]]),np.array([[.3,.07],[.07,.5]])),{})
        prep=_prepare(ham,1)
        compiler=m.CompilerSettings(('rz','sx','x','cx'),None,None,1,None,None,17,qiskit.__version__)
        binding=dict(compiler=compiler_settings_hash(compiler),system='ARTIFICIAL')
        context=SimpleNamespace(m=m,native=native,qstate=np.array([0,1,0,0],dtype=complex),
            preparations={1:prep},compiler=compiler,binding=binding)
        cell=dict(id='B2_p1_q2_2nd_r4_K4',method='B2',prefix=1,q=2,r=4,K=4,R=8,formula='2nd')
        _,law=n.steps(prep,cell,123)
        task=dict(cell=cell,replica=0,seed=123,preparation_hash=prep.preparation_hash,
            saved_log_B=8*math.log(law.exact_finite_distribution),log_B_tolerance=1e-12,control_tolerance=1e-9)
        with tempfile.TemporaryDirectory() as d:
            result=n.compile_task(context,task,Path(d),synthetic=True)
            self.assertEqual(set(result['axes']),{'cosine','sine'})
            self.assertTrue(all(a['RZ']>0 for a in result['axes'].values()))
            self.assertEqual(result['calls']['wrapper_compile'],2)
            self.assertEqual(c.read(Path(d)/'control.json')['status'],'PASS')
            self.assertEqual(result['axes']['cosine']['event_digest'],result['axes']['sine']['event_digest'])


if __name__=='__main__':unittest.main()
