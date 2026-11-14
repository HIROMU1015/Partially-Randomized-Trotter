"""Artificial four-mode DF only: exact hashes, ordering, vectors and compiled metrics."""
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import sys
import time
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests')]
import h46_long_time_signal as signal
signal.install()
import check_truncated_p7_prices_circuits as TT
import pf_native_price_fast as P
import h46_b3_fast_fingerprint as fast
import h46_long_time_native as native
from trottertracks.resource_applicability import ax2b_native_df_v5 as api
from trottertracks.resource_applicability.h6_hash_reuse_v1 import use_fingerprint_reuse
from trotterlib.df_trotter.circuit import simulate_statevector
from trotterlib.rte import CompilerSettings
from trotterlib.rte_compiled_cost import transpile_and_measure_cost


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.ctx=TT.toy()

    def pair(self, formula, method='B0I', prefix=2, T=.4, q=3):
        prep=self.ctx.preparations[prefix];row=TT.row(method,prefix,formula,q=q,T=T)
        scalar=self.ctx.ham.constant+(prep.extracted_identity_coefficient if method=='B0I' else 0.)
        start=time.monotonic()
        with use_fingerprint_reuse() as fp:old=native.compose(row,prep,[],scalar,fp.fingerprint)
        old_time=time.monotonic()-start;stats={};builder=api.build_deterministic_native
        start=time.monotonic()
        with fast.use_fast_fingerprint() as fp:new=P.compose_reusing_parts(native.compose,row,prep,[],scalar,fp.fingerprint,stats=stats)
        fast_time=time.monotonic()-start
        self.assertIs(api.build_deterministic_native,builder)
        self.assertEqual(old.circuit_fingerprint,new.circuit_fingerprint)
        self.assertEqual(old.instruction_upper_bound,new.instruction_upper_bound)
        self.assertEqual(old.circuit.global_phase,new.circuit.global_phase)
        self.assertEqual(len(old.circuit.data),len(new.circuit.data))
        self.assertGreater(stats['reused_parts'],0)
        low=np.r_[self.ctx.qstate,np.zeros_like(self.ctx.qstate)]
        self.assertLess(np.linalg.norm(simulate_statevector(old.circuit,low)-simulate_statevector(new.circuit,low)),1e-12)
        high=np.r_[np.zeros_like(self.ctx.qstate),self.ctx.qstate]
        self.assertLess(np.linalg.norm(simulate_statevector(old.circuit,high)-simulate_statevector(new.circuit,high)),1e-12)
        return old,new,stats,old_time,fast_time

    def test_signed_formulas_and_full_scalar(self):
        for formula in ('2nd','4th','4th(m5_best)'):
            with self.subTest(formula=formula):
                a,b,stats,ot,nt=self.pair(formula)
                print('ARTIFICIAL_COMPOSE',formula,'baseline_s',ot,'fast_s',nt,'builder_counts',stats)
        self.pair('4th',method='FULL',prefix=3,T=-.4)

    def test_both_measured_axes_compiler_metrics_and_vectors(self):
        old,new,*_=self.pair('4th(m5_best)',q=2)
        compiler=CompilerSettings(('rz','sx','x','cx'),None,None,1,None,None,17,'1.3.0')
        low=np.r_[self.ctx.qstate,np.zeros_like(self.ctx.qstate)]
        for axis in ('cosine','sine'):
            a=api.build_native_hadamard_wrapper(old,axis=axis,include_measurement=True,max_instructions=len(old.circuit.data)+4)
            b=api.build_native_hadamard_wrapper(new,axis=axis,include_measurement=True,max_instructions=len(new.circuit.data)+4)
            fp=fast.legacy.canonical_qiskit_circuit_fingerprint(a)
            self.assertEqual(fp,fast.FingerprintSession().fingerprint(b))
            ca=transpile_and_measure_cost(a,compiler,actual_circuit_fingerprint=fp)
            cb=transpile_and_measure_cost(b,compiler,actual_circuit_fingerprint=fp)
            self.assertEqual(ca.compiler_settings_hash,cb.compiler_settings_hash)
            for metric in native.METRICS:self.assertEqual(getattr(ca,metric),getattr(cb,metric))
            before=simulate_statevector(b.remove_final_measurements(inplace=False),low)
            after=simulate_statevector(cb.transpiled_circuit.remove_final_measurements(inplace=False),low)
            self.assertLess(np.linalg.norm(after-before),1e-7)
            self.assertEqual(fast.legacy.canonical_qiskit_circuit_fingerprint(ca.transpiled_circuit),fast.legacy.canonical_qiskit_circuit_fingerprint(cb.transpiled_circuit))

    def test_no_rte_or_cross_call_reuse_and_restore_after_error(self):
        row=TT.row('B0I',2,'2nd');prep=self.ctx.preparations[2];builder=api.build_deterministic_native
        with self.assertRaises(ValueError):P.compose_reusing_parts(native.compose,dict(row,method='PR'),prep,[],.1,lambda c:'unused')
        def failure(*a):raise RuntimeError('synthetic compose exception')
        with self.assertRaises(RuntimeError):P.compose_reusing_parts(failure,row,prep,[],.1,lambda c:'unused')
        self.assertIs(api.build_deterministic_native,builder)
        a=self.pair('2nd',T=.4)[1];b=self.pair('2nd',T=.41)[1]
        self.assertNotEqual(a.circuit_fingerprint,b.circuit_fingerprint)
        original=native.compose
        try:
            with P.accelerated_native(native):
                self.assertIsNot(native.compose,original)
                raise RuntimeError('synthetic context exception')
        except RuntimeError:pass
        self.assertIs(native.compose,original)

    def test_original_cost_trial_gates_with_runtime_fast_context(self):
        with TemporaryDirectory(prefix='pf_fast_toy_') as tmp:
            old_private,old_public,old_compile=native.PRIVATE,native.PUBLIC,native.compile_cached
            native.PRIVATE=Path(tmp);native.PUBLIC=Path(tmp)/'public';native.CONTEXT['TOY']=self.ctx
            def fake(wrapper, fingerprint, representative):
                return dict(metrics={k:1 for k in native.METRICS}),wrapper.copy(),False
            native.compile_cached=fake
            try:
                with P.accelerated_native(native):
                    result,facts,ok=TT.T3.checked_trial(native,self.ctx,TT.row('B0I',2,'2nd',q=3),'TOY_RUNTIME_FAST')
                self.assertTrue(ok)
                self.assertEqual(len(result['checks']),7)
                self.assertEqual(result['technical_acceleration'][0]['distinct_parts'],1)
                self.assertEqual(result['technical_acceleration'][0]['reused_parts'],2)
            finally:
                native.PRIVATE,native.PUBLIC,native.compile_cached=old_private,old_public,old_compile


if __name__=='__main__':unittest.main(verbosity=2)
