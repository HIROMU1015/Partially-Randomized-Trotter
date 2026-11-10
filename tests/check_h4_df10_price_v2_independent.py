"""Artificial checks for the independent-q adapter; no H4 input or sampling."""
from pathlib import Path
import sys
import importlib.util
from functools import partial
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from trottertracks import h4_df10_price_v1 as old
from trottertracks import h4_df10_price_v2_independent as m


def event():
    return dict(phase=dict(real=1.,imag=0.),rotation_angle=.2,taylor_order=0,
        application_sequence=[dict(application_index=0,is_identity=False,role='rotation',
            coefficient_sign=1,basis_id='synthetic',basis_hash='synthetic',
            basis_change_operations=[],diagonal_pauli_support=[0])])


class IndependentTests(unittest.TestCase):
    def test_scope_extension_leaves_accounting_body_identical(self):
        blocks=[dict(basis=[],diagonal=SimpleNamespace(rz=[(0,.07)],rzz=[],global_phase=.01))]
        for q,T,batches in [(8,.8,None),(16,1.6,None),(8,.8,[[event()]*4 for _ in range(8)])]:
            for axis in ('cosine','sine'):
                self.assertEqual(old.predict(blocks,q,.17,T,axis,batches),m.predict(blocks,q,.17,T,axis,batches))
        self.assertGreater(m.predict(blocks,32,.17,3.2,'cosine')['structural_rz'],0)
        self.assertGreater(m.predict(blocks,16,.17,1.6,'cosine',[[event()]*4 for _ in range(16)])['structural_rz'],0)
        with self.assertRaises(ValueError):m.predict(blocks,64,.17,6.4,'cosine')
        with self.assertRaises(ValueError):m.predict(blocks,16,.17,1.6,'cosine',[[event()]*3 for _ in range(16)])

    def test_cache_requires_fingerprint_compiler_and_price(self):
        value=dict(compiler=m.COMPILER,fingerprint='synthetic',metrics={'rz_count':19})
        self.assertEqual(m.validate_cache(value,'synthetic'),value)
        for v in [dict(value,compiler='other'),dict(value,fingerprint='other'),dict(value,metrics={'rz_count':-1})]:
            with self.assertRaises(ValueError):m.validate_cache(v,'synthetic')

    def test_frozen_correction_is_not_affected_by_observed_price(self):
        prediction=dict(method='PR',candidate_id='synthetic',q=16,T=1.6,path_index=0,
                        axis='cosine',structural_rz=123)
        def trial(actual):
            return [dict(candidate={'id':'synthetic'},index=0,prices={'cosine':{'rz_count':actual}})]
        a=m.residual_rows([prediction],trial(3000));b=m.residual_rows([prediction],trial(4000))
        self.assertEqual([x['prediction_rz'] for x in a],[x['prediction_rz'] for x in b])
        self.assertEqual(a[1]['prediction_rz'],123+16*176.88888888888894)
        self.assertEqual(b[1]['error_rz']-a[1]['error_rz'],1000)

    def test_original_compose_and_compiler_on_artificial_fixture(self):
        root=Path(__file__).resolve().parents[1]
        sys.path.insert(0,str(root/'scripts'))
        import h46_long_time_native as native
        cfg=native.study.install()
        import trottertracks
        trottertracks.__path__.insert(0,str(Path(cfg['provider_root'])/'src/trottertracks'))
        spec=importlib.util.spec_from_file_location('synthetic_native_fixture',root/'tests/test_h46_new2_native.py')
        fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
        fixture.NativeTests.setUpClass();f=fixture.NativeTests
        import numpy as np
        import qiskit
        import trotterlib.rte_compiled_cost as costs
        from trotterlib.rte import CompilerSettings
        from trotterlib.df_trotter.circuit import simulate_statevector
        from trottertracks.resource_applicability import ax2b_native_df_v5 as provider
        from trottertracks.resource_applicability.h6_hash_reuse_v1 import use_fingerprint_reuse
        costs.transpile=partial(qiskit.transpile,num_processes=1)
        row=dict(system='ARTIFICIAL',method='PR',formula='2nd',id='synthetic',q=2,T=.2,R_outer=4,K=2,prefix=2)
        requests,_,_=native.sample_requests(row,f.prep,'ARTIFICIAL_ONLY',0)
        scalar=f.ham.constant+f.prep.extracted_identity_coefficient
        state=np.zeros(16,complex);state[f.oracle.qindices]=f.oracle.initial
        low=np.r_[state,np.zeros_like(state)]
        with use_fingerprint_reuse() as fp:
            evolution=native.compose(row,f.prep,requests,scalar,fp.fingerprint)
            for axis in ('cosine','sine'):
                wrapper=provider.build_native_hadamard_wrapper(evolution,axis=axis,include_measurement=True,
                    max_instructions=len(evolution.circuit.data)+4)
                cost=costs.transpile_and_measure_cost(wrapper,
                    CompilerSettings(('rz','sx','x','cx'),None,None,1,None,None,17,'1.3.0'),
                    actual_circuit_fingerprint=fp.fingerprint(wrapper))
                self.assertEqual(cost.compiler_settings_hash,m.COMPILER)
                before=simulate_statevector(wrapper.remove_final_measurements(inplace=False),low)
                after=simulate_statevector(cost.transpiled_circuit.remove_final_measurements(inplace=False),low)
                self.assertLess(np.linalg.norm(after-before),1e-7)


if __name__=='__main__':unittest.main()
