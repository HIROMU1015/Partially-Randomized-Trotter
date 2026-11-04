"""Synthetic state/circuit fixtures only; no molecular preparation or run()."""
import ast
from concurrent.futures import ProcessPoolExecutor
import math
import multiprocessing
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from qiskit import QuantumCircuit

import run_fixed_request_b as b


def artificial_job(theta=.31):
    psi=np.array([math.sqrt(.3),1j*math.sqrt(.7)],dtype=complex)
    evolution=QuantumCircuit(2)
    evolution.p(-theta,1)
    wrappers={}
    for axis in ('cosine','sine'):
        circuit=QuantumCircuit(2,1)
        circuit.h(1);circuit.compose(evolution,inplace=True)
        if axis=='sine':circuit.sdg(1)
        circuit.h(1);circuit.measure(1,0)
        wrappers[axis]=circuit
    return {'id':'artificial_only','ld':3,'q':1,'profile':'artificial',
            'system_state':psi,'target_system_state':np.exp(-1j*theta)*psi,
            'sector_indices':np.array([0,1]),'signal':np.exp(-1j*theta),
            'evolution':evolution,'wrappers':wrappers}


class FixedRequestBTests(unittest.TestCase):
    def test_exact_six_profile_specification(self):
        self.assertEqual(b.SPECS,((3,1,'all_order0'),(3,1,'first_order2'),
                                 (3,8,'all_order0'),(3,8,'first_order2'),
                                 (12,1,'deterministic'),(12,8,'deterministic')))

    def test_component_selector_uses_ID_order_and_original_indices(self):
        components=[SimpleNamespace(component_id='b',basis_id='B'),
                    SimpleNamespace(component_id='a',basis_id='A'),
                    SimpleNamespace(component_id='c',basis_id='A')]
        self.assertEqual(b.component_indices(components),(1,0))
        with self.assertRaises(ValueError):b.component_indices(components[1:])

    def test_profiles_are_explicit_not_random_samples(self):
        zero,two=object(),object()
        self.assertEqual(b.event_profile(8,'all_order0',zero,two),(zero,)*8)
        self.assertEqual(b.event_profile(8,'first_order2',zero,two),(two,)+(zero,)*7)
        with self.assertRaises(ValueError):b.event_profile(16,'all_order0',zero,two)

    def test_embedding_matches_sector_matrix_permutation(self):
        state=np.array([.2,.3j,.4]);indices=np.array([1,2,4]);perm=np.array([2,0,1])
        embedded=b.embed_sector(state,indices,perm,3)
        np.testing.assert_array_equal(embedded[indices[perm]],state)
        self.assertEqual(np.count_nonzero(embedded),3)

    def test_vector_companion_matches_unchanged_legacy_scalar_on_toy(self):
        source=(b.ROOT/'src/trotterlib/rpe_hadamard_failure_validation.py').read_text()
        names={'_apply_event_to_state','_trajectory_signal'}
        nodes=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in names]
        tree=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*nodes],type_ignores=[])
        ast.fix_missing_locations(tree);ns={'np':np,'math':math}
        exec(compile(tree,'<unchanged-legacy-toy-functions>','exec'),ns)
        operators={'x':np.array([[0,1],[1,0]],complex),'z':np.diag([1.,-1.])}
        events=[SimpleNamespace(product_component_ids=('x','z'),rotation_component_id='x',rotation_angle=.13,phase=-1.),
                SimpleNamespace(product_component_ids=(),rotation_component_id='z',rotation_angle=.27,phase=1.)]
        psi=np.array([math.sqrt(.3),1j*math.sqrt(.7)])
        forward=np.diag([np.exp(.12j),np.exp(-.12j)])
        reverse=np.array([[math.cos(.2),-math.sin(.2)],[math.sin(.2),math.cos(.2)]],complex)
        result=b.trajectory_state(psi,forward,reverse,events,operators,2,1,ns['_apply_event_to_state'])
        scalar=ns['_trajectory_signal'](initial_state=psi,forward=forward,reverse=reverse,
            events=events,operators=operators,q_m=2,rte_steps=1)
        self.assertAlmostEqual(complex(np.vdot(psi,result)),scalar)
        with self.assertRaises(ValueError):b.trajectory_state(psi,forward,reverse,events,operators,1,1,ns['_apply_event_to_state'])

    def test_ideal_wrapper_and_control_actions_preserve_negative_sine(self):
        result=b.evaluate_job(artificial_job())
        self.assertTrue(result['pass'])
        self.assertAlmostEqual(result['axes']['sine']['actual_Z_mean'],-math.sin(.31))
        self.assertTrue(all(v<=1e-12 for v in result['residuals'].values()))

    def test_relative_phase_error_is_not_hidden_by_global_alignment(self):
        job=artificial_job();job['target_system_state']*=1j
        result=b.evaluate_job(job)
        self.assertFalse(result['pass'])
        self.assertGreater(result['residuals']['control_one_branch_max_abs'],.1)

    def test_wrong_sine_sign_is_detected(self):
        job=artificial_job();job['signal']=job['signal'].conjugate()
        result=b.evaluate_job(job)
        self.assertFalse(result['pass'])
        self.assertGreater(result['axes']['sine']['absolute_residual'],.1)

    def test_process_path_matches_serial_on_artificial_circuit(self):
        job=artificial_job();serial=b.evaluate_job(job)
        with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
            outputs=list(pool.map(b.evaluate_job,[job,job]))
        for output in outputs:
            self.assertEqual(output['axes'],serial['axes'])
            self.assertEqual(output['residuals'],serial['residuals'])

    def test_nonfinite_or_large_residual_is_not_accepted(self):
        for value in (math.nan,math.inf,b.ATOL*2):
            with self.assertRaises(ValueError):b.require_small([value],'artificial')

    def test_no_sampler_compile_or_decoder_calls_in_new_runner(self):
        tree=ast.parse(Path(b.__file__).read_text())
        calls={n.func.id if isinstance(n.func,ast.Name) else n.func.attr for n in ast.walk(tree)
               if isinstance(n,ast.Call) and isinstance(n.func,(ast.Name,ast.Attribute))}
        self.assertFalse(calls & {'sample_rte_events','make_df_partial_s2_repeated_request',
            'default_rng','PCG64','transpile','transpile_and_measure_cost','reconstruct_rpe_phase'})


if __name__=='__main__':unittest.main()
