"""Artificial H8 scope/scalar/grant tests and a two-qubit QPY/native fixture."""
from fractions import Fraction
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h8_dba_contract as c
from h8_dba_direct import scalar_for


class Contract(unittest.TestCase):
    def test_fixed_four_axes_without_H6_reexecution(self):
        p=c.plan()
        self.assertEqual([x['id'] for x in p['D']],['B0_p15_q8_2nd','B0I_p16_q4_4th'])
        self.assertEqual(p['calls_per_case']['compile'],2)
        self.assertFalse(p['cost_reaccounting']);self.assertFalse(p['retry'])

    def test_exact_two_percent_stop(self):
        self.assertFalse(c.axis_check('102',100)['PASS'])
        self.assertTrue(c.axis_check('10199/100',100)['PASS'])

    def test_own_saved_scalar_added_once_with_rounding(self):
        scalar=dict(policy='EXACT_STORED_BINARY64_TAIL_IDENTITY',numerator='1',denominator='10',C=.1)
        self.assertEqual(scalar_for(dict(method='B0I'),(.2).hex(),dict(scalar_receipt=scalar)),.2+.1)
        self.assertEqual(scalar_for(dict(method='B0'),(.2).hex(),dict(scalar_receipt=None)),.2)

    def test_scalar_receipt_drift_refused(self):
        scalar=dict(policy='EXACT_STORED_BINARY64_TAIL_IDENTITY',numerator='1',denominator='10',C=.2)
        with self.assertRaises(ValueError):scalar_for(dict(method='B0I'),(.2).hex(),dict(scalar_receipt=scalar))
        with self.assertRaises(ValueError):scalar_for(dict(method='B0'),(.2).hex(),dict(scalar_receipt=scalar))

    def test_unapproved_refused_before_any_inputs(self):
        with self.assertRaisesRegex(ValueError,'EXPLICIT_SCOPED'):
            c.verify({},dict(scope=c.SCOPE,stage='D',approved_by_user=False))

    def test_H6_or_other_stage_grant_is_not_transferred(self):
        for grant in [dict(scope='H6_DBA_20261018',stage='D'),dict(scope=c.SCOPE,stage='B')]:
            with self.assertRaisesRegex(ValueError,'H8_D_SCOPE'):c.verify({},grant)


class NativeFixture(unittest.TestCase):
    def test_parallel_control_qpy_and_hash_reuse_keep_metrics(self):
        import numpy as np
        import qiskit
        from qiskit import qpy
        from trotterlib.df_hamiltonian import DFHamiltonian
        from trotterlib.pr2_matched_accuracy_m1_execution import _prepare
        from trottertracks.resource_applicability import h6_matched_port_v1 as m
        from trottertracks.resource_applicability import ax2b_native_df_v5 as native
        from trottertracks.resource_applicability.h6_hash_reuse_v1 import use_fingerprint_reuse
        import h8_key_compile_controls as controls
        # The fixture is synthetic two-qubit data, not a molecular H8 calculation.
        ham=DFHamiltonian(.17,np.array([[.2,.04],[.04,-.3]]),np.array([.4]),
            (np.array([[.6,.1],[.1,-.2]]),),{})
        prep=_prepare(ham,1);state=np.array([0,1,0,0],dtype=complex)
        high=np.r_[np.zeros_like(state),state]
        ordinary=native.build_deterministic_native(prep.deterministic_blocks,num_system_qubits=2,T=.8,q=1,
            formula='4th',scalar=ham.constant,control_policy='ordinary',max_instructions=100000)
        z=np.vdot(state,m.simulate_statevector(ordinary.circuit,high)[4:])
        target=dict(corrected=dict(real=float(z.real),imag=float(z.imag)),
                    independent_forward=dict(real=float(z.real),imag=float(z.imag)))
        cp=m.CompilerSettings(('rz','sx','x','cx'),None,None,1,None,None,17,qiskit.__version__)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            class Permit:
                output=root;config={};grant={'worker_cpus':[0,1]}
                def require(self,*args):pass
                def take(self,*args):pass
            def serial_fixture(tasks,callback,**kw):
                rows=[]
                for task in tasks:
                    d=kw['directory']/task['id'];(d/'scratch').mkdir(parents=True)
                    value=callback(task,d);c.save(d/'task_result.json',value)
                    rows.append(dict(task_id=task['id'],directory=str(d),exit_code=0,reaped=True,
                                     result_sha256=c.sha(d/'task_result.json')))
                return dict(status='PASS',all_reaped=True,cleanup_errors=[],workers=rows)
            with use_fingerprint_reuse():
                built=controls.build_case(dict(id='toy',q=1,formula='4th'),prep.deterministic_blocks,
                    ham.constant,state,target,Permit(),native,m,n=2,cap=100000,map_owned=serial_fixture)
            for axis in ['cosine','sine']:
                wrapper=controls.load_wrapper(built['spools'][axis],root,qpy)
                fingerprint=m.canonical_qiskit_circuit_fingerprint(wrapper)
                self.assertEqual(fingerprint,built['fingerprints'][axis])
                first=m.transpile_and_measure_cost(wrapper,cp,actual_circuit_fingerprint=fingerprint)
                with use_fingerprint_reuse():
                    second=m.transpile_and_measure_cost(wrapper,cp,actual_circuit_fingerprint=fingerprint)
                self.assertEqual(first.rz_count,second.rz_count)
                self.assertEqual(first.cx_count,second.cx_count)
                self.assertEqual(first.compiler_settings_hash,second.compiler_settings_hash)


if __name__=='__main__':unittest.main()
