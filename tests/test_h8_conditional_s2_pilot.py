"""Artificial law/events/circuits and owned minimal children; no real H8 loads."""
import copy
import itertools
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import h8_conditional_s2_sampling as s
import h8_conditional_s2_contract as c
import h8_key_compile_parallel as pool


class Mathematics(unittest.TestCase):
    def test_exact_enumerated_conditional_distribution(self):
        R = 4
        p0, p2, p4 = .5, .3, .2
        law = s.conditional_law(R, (p0, p2, p4))
        masses = {}
        for vector in itertools.product((0, 2), repeat=R):
            m = vector.count(2)
            if m:
                target = p2**m*p0**(R-m)/law['pi2']
                implemented = law['count_probabilities'][m-1]/math.comb(R, m)
                self.assertAlmostEqual(target, implemented, places=14)
                masses[vector] = implemented
        self.assertAlmostEqual(sum(masses.values()), 1.)
        self.assertAlmostEqual(law['pi0']+law['pi2']+law['pi4'], 1.)
        self.assertGreater(law['count_probabilities'][1], 0)

    def test_tiny_probability_stability(self):
        law = s.conditional_law(32, (1., 1e-250, 0.))
        self.assertGreater(law['pi2'], 0)
        self.assertAlmostEqual(law['count_probabilities'][0], 1.)
        self.assertTrue(all(math.isfinite(v) for v in law['count_probabilities']))

    def test_uniform_position_combinatorial_identity(self):
        for R in (16, 32):
            for m in range(1, R+1):
                self.assertAlmostEqual(math.comb(R-1, m-1)/math.comb(R, m), m/R)

    def test_bad_laws_rejected(self):
        for R, p in ((0, (.5, .4, .1)), (True, (.5, .4, .1)), (16, (1., 0., 0.)),
                     (16, (.6, .6, 0.)), (16, (.9, float('nan'), .1))):
            with self.assertRaises(ValueError):
                s.conditional_law(R, p)

    def test_incomplete_and_wrong_strata_rejected(self):
        for orders, q in (((0,)*16, 4), ((2,)*15, 4), ((2,)*16, 8), ((4,)+(0,)*15, 4)):
            with self.assertRaises(ValueError):
                s.validate_orders(orders, q, 4)

    def test_seed_namespaces_and_old_collision(self):
        original = set(c.evidence()['original_seeds'])
        new = [t['seed'] for q in (4, 8) for t in c.tasks(q)]
        self.assertEqual(len(set(new)), 16)
        self.assertFalse(original&set(new))
        self.assertTrue(all(s.order_seed(n) != n for n in new))
        altered = copy.deepcopy(c.evidence())
        altered['original_seeds'].append(new[0])
        with patch.object(c, 'evidence', return_value=altered), self.assertRaisesRegex(ValueError, 'SEED_COLLISION'):
            c.plan()


class ArtificialEvents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        from trotterlib.rte import RTEComponent, finite_rte_distribution
        cls.np = np
        cls.components = (RTEComponent('toy-a', .25, .5, 1), RTEComponent('toy-b', .75, 1.5, -1))
        cls.distribution = finite_rte_distribution(.12, 4)

    def test_draw_reproducibility_multiple_events_q4_q8(self):
        for q in (4, 8):
            law = s.conditional_law(q*4, (.5, .4, .1))
            first = s.draw_orders(law, self.np.random.default_rng(123))
            second = s.draw_orders(law, self.np.random.default_rng(123))
            self.assertEqual(first, second)
            s.validate_orders(first, q, 4)
            self.assertGreater(first.count(2), 1)
            self.assertNotIn(4, first)

    def test_component_counts_probabilities_and_exact_event_constructor(self):
        from trotterlib.rte import _make_event
        class RNG:
            def __init__(self):
                self.calls = []
            def choice(self, n, *, size, p):
                self.calls.append((n, size, p))
                return self.np_array(size)
            @staticmethod
            def np_array(size):
                return [i % 2 for i in range(size)]
        orders = (2, 0, 2, 0)+(0,)*12
        rngs = [RNG() for _ in range(4)]
        ev = s.events_from_orders(orders, self.components, self.distribution, rngs, 4)
        for k, event in zip(orders, ev):
            expected = _make_event([i % 2 for i in range(k+1)], self.components, self.distribution,
                                   self.distribution.orders.index(k))
            self.assertEqual(event.to_dict(), expected.to_dict())
        calls = [x for r in rngs for x in r.calls]
        self.assertEqual([x[1] for x in calls], [k+1 for k in orders])
        self.assertTrue(all(x[2] == [.25, .75] for x in calls))
        self.assertEqual(len(ev), 16)

    def test_component_draw_repeatable_and_k2_rejected(self):
        orders = (2,)+(0,)*15
        def draw():
            return s.events_from_orders(orders, self.components, self.distribution,
                [self.np.random.default_rng(v) for v in s.step_seeds(99, 4)], 4)
        self.assertEqual([e.to_dict() for e in draw()], [e.to_dict() for e in draw()])
        from trotterlib.rte import finite_rte_distribution
        with self.assertRaises(ValueError):
            s.events_from_orders(orders, self.components, finite_rte_distribution(.12, 2), [None]*4, 4)

    def test_synthetic_whole_wrapper_and_metrics(self):
        from qiskit import QuantumCircuit
        from trottertracks.resource_applicability.ax2b_native_df_v5 import NativeEvolution, build_native_hadamard_wrapper
        from trottertracks.resource_applicability.ax2b_stream_fingerprint_v5 import canonical_qiskit_circuit_fingerprint as fp
        from trottertracks.resource_applicability.h6_matched_memory_v3 import take_wrapper
        # NativeEvolution fields are supplied from the pinned dataclass; these
        # artificial 2-system gates never load chemistry or an H8 trajectory.
        from dataclasses import fields
        circuit = QuantumCircuit(3)
        circuit.crz(.2, 2, 0)
        circuit.cx(0, 1)
        values = dict(circuit=circuit, num_system_qubits=2, circuit_fingerprint=fp(circuit))
        for f in fields(NativeEvolution):
            if f.name not in values:
                values[f.name] = {} if 'receipt' in f.name else 'artificial'
        evolution = NativeEvolution(**values)
        for axis in ('cosine', 'sine'):
            from dataclasses import replace
            reference = build_native_hadamard_wrapper(evolution, axis=axis, include_measurement=True, max_instructions=100)
            observed = take_wrapper([replace(evolution, circuit=circuit.copy())], axis=axis, max_instructions=100)
            self.assertEqual(fp(reference), fp(observed))
            self.assertEqual(observed.num_clbits, 1)

    def test_artificial_sixteen_mode_conditional_native_connection(self):
        from trotterlib.df_hamiltonian import DFHamiltonian
        from trotterlib.pr2_matched_accuracy_m1_execution import _prepare
        from trotterlib.rte import finite_rte_distribution, CompilerSettings
        from trotterlib.rte_compiled_cost import transpile_and_measure_cost
        from trottertracks.resource_applicability.ax2b_native_df_v5 import partial_native_from_step_requests, build_native_hadamard_wrapper
        from trottertracks.resource_applicability.h8_resource_map_optimization_v1 import ancilla_diagonal
        from trottertracks.resource_applicability.ax2b_stream_fingerprint_v5 import canonical_qiskit_circuit_fingerprint as fp
        np = self.np
        # Artificial diagonal DF, not H8 saved scientific input.
        sparse_diagonal = np.array([.1, -.02]+[0.]*14)
        ham = DFHamiltonian(.1, np.zeros((16, 16), dtype=complex),
            np.array([.001]*8), tuple(np.diag(sparse_diagonal).astype(complex) for _ in range(8)), {'artificial':True})
        prep = _prepare(ham, 7)
        import qiskit
        compiler = CompilerSettings(('rz', 'sx', 'x', 'cx'), None, None, 1, None, None, 17, qiskit.__version__)
        for q in (4, 8):
            expected = finite_rte_distribution(prep.exact_rte_lambda_r*(.8/q)/4, 4).to_dict()
            steps, receipt = s.make_steps(prep, q=q, master=123, expected=expected)
            again, repeat = s.make_steps(prep, q=q, master=123, expected=expected)
            self.assertEqual(receipt, repeat)
            self.assertEqual([[e.to_dict() for e in x.rte_occurrence.events] for x in steps],
                             [[e.to_dict() for e in x.rte_occurrence.events] for x in again])
            evolution = partial_native_from_step_requests(steps, control_policy='symmetric_directional', max_instructions=4000000)
            self.assertIs(type(ancilla_diagonal(evolution.circuit, 16)), bool)
            self.assertEqual(evolution.num_system_qubits, 16)
            for axis in ('cosine', 'sine'):
                wrapper = build_native_hadamard_wrapper(evolution, axis=axis, include_measurement=True, max_instructions=4000000)
                if q == 4:
                    measured = transpile_and_measure_cost(wrapper, compiler, actual_circuit_fingerprint=fp(wrapper))
                    counts = measured.transpiled_circuit.count_ops()
                    self.assertEqual(measured.compiler_settings_hash, c.COMPILER)
                    self.assertEqual(measured.rz_count, counts.get('rz', 0))
                    self.assertEqual(measured.cx_count, counts.get('cx', 0))
                    self.assertEqual(measured.total_depth, measured.transpiled_circuit.depth())
                    self.assertEqual(measured.circuit_size, measured.transpiled_circuit.size())
            with self.assertRaisesRegex(ValueError, 'FINITE_LAW_CHANGED'):
                s.make_steps(prep, q=q, master=123, expected={})

    def test_inconclusive_control_structure_uses_four_probe_fallback(self):
        from qiskit import QuantumCircuit
        from dataclasses import fields
        from trottertracks.resource_applicability.ax2b_native_df_v5 import NativeEvolution
        from trottertracks.resource_applicability.h8_resource_map_optimization_v1 import validate_control
        from trottertracks.resource_applicability.ax2b_stream_fingerprint_v5 import canonical_qiskit_circuit_fingerprint as fp
        from trottertracks.resource_applicability.ax2b_molecular_ports_v3 import hadamard_expectation
        from trotterlib.df_trotter.circuit import simulate_statevector
        inner = QuantumCircuit(3)
        inner.crz(.12, 2, 0)
        circuit = QuantumCircuit(3)
        circuit.append(inner.to_gate(), [0, 1, 2])
        values = dict(circuit=circuit, num_system_qubits=2, circuit_fingerprint=fp(circuit))
        for f in fields(NativeEvolution):
            if f.name not in values:
                values[f.name] = {} if 'receipt' in f.name else 'artificial'
        class Calls:
            used = 0
            def take(self, key, n=1):
                self.used += n
        calls = Calls()
        state = self.np.array([1, 0, 0, 0], dtype=complex)
        proof, branch = validate_control(NativeEvolution(**values), state, calls=calls,
            simulate=simulate_statevector, expectation=hadamard_expectation)
        self.assertEqual(calls.used, 4)
        self.assertEqual(proof['algorithm'], 'two-branch fallback')
        self.assertLess(proof['max_error'], 1e-8)


class AuthorizationAndAccounting(unittest.TestCase):
    def test_unapproved_grant_rejected_before_imports_and_lock(self):
        with self.assertRaisesRegex(ValueError, 'GRANT_REQUIRED'):
            c.authorize({'execution_sealed': False}, {'approved': False})

    def test_budget_preserves_carry_and_failed_reservations(self):
        b = c.budget()
        self.assertEqual(b['prior_held_by_attempt']['original']['compile'], 22)
        self.assertEqual(b['prior_held_by_attempt']['conditional_run01']['compile'], 16)
        self.assertEqual(b['prior_held']['compile'], 38)
        self.assertEqual(b['maximum_after']['compile'], 1642)
        self.assertEqual(b['maximum_after']['trajectory'], 803)
        self.assertEqual(b['maximum_after']['occurrence'], 3195)
        self.assertEqual(b['maximum_after']['control_probe'], 376)
        self.assertFalse(b['reset'])
        e = copy.deepcopy(c.evidence())
        e['old_caps']['compile'] = 1641
        with patch.object(c, 'evidence', return_value=e), self.assertRaises(ValueError):
            c.budget()

    def test_exact_reservations_duplicate_and_overflow(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = c.Ledger(directory)
            for task in c.tasks(4)+c.tasks(8):
                ledger.reserve(task)
            self.assertEqual(ledger.used, c.CAPS)
            with self.assertRaises(ValueError):
                ledger.reserve(c.tasks(4)[0])
            foreign = dict(c.tasks(4)[0], id='other')
            with self.assertRaises(ValueError):
                ledger.reserve(foreign)

    def test_one_shot_and_output_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'lock'
            c.save(p, {'consumed': True})
            with self.assertRaises(FileExistsError):
                c.save(p, {'consumed': True})

    def test_source_and_input_mutation_rejected(self):
        # Real input bytes are neither read nor decoded by this fixture.
        config = dict(plan_sha256=c.digest(c.plan()), provider_source=c.PROVIDER,
            expected_profile_digest=c.PROFILE, provider_hashes={}, runtime_file_hashes={}, source_hashes={})
        with self.assertRaisesRegex(ValueError, 'PINNED_PROVIDER_RUNTIME_MAPS'):
            c.verify(config, git=False)
        config['plan_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'PLAN_PROFILE_CHANGED'):
            c.verify(config, git=False)

    def test_actual_byte_tamper_and_runtime_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, data = root/'toy.py', root/'toy.json'
            source.write_text('# toy\n')
            data.write_text('{}\n')
            spec = dict(logical_name='toy', bytes=data.stat().st_size, sha256=c.sha(data))
            e = dict(provider_map_sha256=c.digest({}), runtime_map_sha256=c.digest({}), inputs=[spec])
            cfg = dict(plan_sha256=c.digest({}), provider_source=c.PROVIDER, expected_profile_digest=c.PROFILE,
                provider_hashes={}, runtime_file_hashes={}, source_hashes={'toy.py':c.sha(source)},
                files=[dict(spec, path=str(data))], python=sys.executable, python_sha256=c.sha(sys.executable))
            with patch.object(c, 'ROOT', root), patch.object(c, 'SOURCES', ('toy.py',)), patch.object(c, 'evidence', return_value=e), patch.object(c, 'plan', return_value={}):
                c.verify(cfg, git=False)
                source.write_text('# mutation\n')
                with self.assertRaisesRegex(ValueError, 'SOURCE_CHANGED'):
                    c.verify(cfg, git=False)
                source.write_text('# toy\n')
                data.write_text('[]\n')
                with self.assertRaisesRegex(ValueError, 'INPUT_BYTE_SHA'):
                    c.verify(cfg, git=False)

    def test_descriptive_pair_statistics_not_old_n32(self):
        rows = [dict(rows=[dict(metrics={m:i+1 for m in c.plan()['metrics']}),
                          dict(metrics={m:2*(i+1) for m in c.plan()['metrics']})]) for i in range(8)]
        summary = c.summary(rows)
        self.assertEqual(summary['conditional_s2_descriptive']['RZ']['paired_sum']['mean'], 13.5)
        self.assertFalse(summary['original_n32_mixed'])
        self.assertIsNone(summary['s4_cost'])


class MinimalOwnedChildren(unittest.TestCase):
    def test_two_minimal_children_exit_and_reap(self):
        cpu = min(os.sched_getaffinity(0))
        with tempfile.TemporaryDirectory() as directory:
            receipt = pool.map_owned([dict(id='toy0'), dict(id='toy1')], lambda t, d: dict(id=t['id']),
                cpus=[cpu], limits={'max_workers':1}, directory=Path(directory)/'wave', deadline=math.inf)
            self.assertEqual(receipt['status'], 'PASS')
            self.assertTrue(receipt['all_reaped'])
            self.assertTrue(all(h['exit_code'] == 0 and h['reaped'] and pool.identity(h['pid']) is None for h in receipt['workers']))

    def test_first_failure_cleanup_no_retry(self):
        cpu = min(os.sched_getaffinity(0))
        def fail(task, path):
            raise ValueError('TOY_FIRST_REASON')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'wave'
            with self.assertRaisesRegex(RuntimeError, 'TOY_FIRST_REASON'):
                pool.map_owned([dict(id='bad'), dict(id='undispatched')], fail, cpus=[cpu],
                    limits={'max_workers':1}, directory=path, deadline=math.inf)
            receipt = c.read(path/'parallel_receipt.json')
            self.assertEqual(len(receipt['workers']), 1)
            self.assertTrue(receipt['all_reaped'])
            self.assertIn('TOY_FIRST_REASON', receipt['first_error'])

    def test_identity_mismatch_never_signals_and_esrch_is_exit(self):
        h = dict(pid=99, identity=(99, 1, 1, 1), reaped=False)
        with patch.object(pool, 'identity', return_value=(99, 2, 1, 1)), patch.object(pool.signal, 'pidfd_send_signal') as send:
            with self.assertRaisesRegex(RuntimeError, 'IDENTITY_CHANGED'):
                pool.send_owned(h, signal.SIGTERM)
            send.assert_not_called()
        with patch.object(pool, 'identity', return_value=h['identity']), patch.object(pool.os, 'pidfd_open', side_effect=ProcessLookupError):
            pool.send_owned(h, signal.SIGTERM)


class SavedPreparationReceiptRegression(unittest.TestCase):
    def setUp(self):
        from types import SimpleNamespace
        self.saved = copy.deepcopy(c.evidence()['preparation'])
        self.prepared = SimpleNamespace(preparation_hash=self.saved['preparation_hash'], partition_hash='a'*64,
            exact_rte_lambda_r=self.saved['lambda_r'], extracted_identity_coefficient=self.saved['extracted_identity'],
            constant_coefficient=self.saved['constant'], coefficient_atol=0., threshold_dropped_component_count=0,
            deterministic_fragment_indices=tuple(range(7)))

    def test_exact_h8_receipt_without_standalone_partition(self):
        self.assertNotIn('partition_hash', self.saved)
        proof = c.verify_saved_preparation(self.prepared, self.saved)
        self.assertFalse(proof['historical_standalone_partition_hash_present'])

    def test_existing_partition_field_must_match(self):
        self.saved['partition_hash'] = 'b'*64
        with self.assertRaisesRegex(ValueError, 'PARTITION_CHANGED'):
            c.verify_saved_preparation(self.prepared, self.saved)

    def test_saved_hash_drift_is_still_stop(self):
        self.prepared.preparation_hash = 'c'*64
        with self.assertRaisesRegex(ValueError, 'PREPARATION_CHANGED'):
            c.verify_saved_preparation(self.prepared, self.saved)

    def test_missing_required_field_stops_metadata_preflight(self):
        del self.saved['lambda_r']
        with self.assertRaisesRegex(ValueError, 'MISSING_FIELD'):
            c.preparation_receipt_schema(self.saved)


if __name__ == '__main__':
    unittest.main()
