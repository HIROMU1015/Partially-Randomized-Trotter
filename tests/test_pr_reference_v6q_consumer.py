"""Artificial outer-consumer tests; no cache unpickle or quantum runtime."""
from functools import lru_cache
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import pr_reference_v6q_consumer as consumer
from trottertracks import pr_reference_v6q_schedule as caller
from trottertracks import pr_reference_v6q as provider

spec = importlib.util.spec_from_file_location('existing_caller_mocks', ROOT/'tests/test_pr_reference_v6q_schedule.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


class FakeCircuit:
    def __init__(self, nq, nc, log):
        self.num_qubits, self.num_clbits = nq, nc
        self.operations, self.log = [], log
        log.append(('allocate', nq, nc))

    def reset(self, wires):
        self.operations.append(('reset', tuple(wires)))
        self.log.append(self.operations[-1])

    def compose(self, native, *, inplace):
        assert inplace is True
        self.operations.append(('compose', native))
        self.log.append(('compose', native.axis))


class ConsumerTests(unittest.TestCase):
    def fixture(self, m=17):
        f = old.ScheduleCallerTests().fixture(m)
        prep = NS(ld=3, num_system_qubits=8, product_formula='2nd', exact_rte_lambda_r=.5,
            rte_preparation=NS(symbolic_tail=NS(tail_id='artificial-tail', tail_hash='artificial-hash')))
        f['cfg'].tail_id, f['cfg'].tail_hash, f['cfg'].lambda_r = 'artificial-tail', 'artificial-hash', .5
        data = dict(preps={3:prep})
        context = consumer.FixedInputs(Path('/artificial/unused-cache'), consumer.CACHE_SHA256, data, prep)
        batch = consumer.DeclaredCanonicalEvents(f['meta'], prep, f['cfg'], f['dist'], f['events'],
            consumer.CANONICAL_LAW, 'artificial-producer-only')
        log = []

        def compile_axis(axis_circuit):
            axis, core = axis_circuit
            log.append(('compile', axis, core))
            return NS(num_qubits=17, num_clbits=1, axis=axis, original=axis_circuit)

        runtime = NS(common=Mock(return_value=data), compile_reference=Mock(side_effect=compile_axis),
            circuit=Mock(side_effect=lambda nq,nc:FakeCircuit(nq,nc,log)))
        f.update(context=context, batch=batch, source=Mock(return_value=batch), runtime=runtime, log=log)
        return f

    def invoke(self, f, mode='exact_binary_m17', **overrides):
        m = f['meta']
        args = dict(q=m.q, r=m.r, bits=m.bits, canonical_source=f['source'], mode=mode)
        args.update(overrides)
        with patch.object(consumer, '_runtime', return_value=f['runtime']), \
             patch.object(caller, '_legacy_modules', return_value=(f['pilot'], f['transfer'])):
            return consumer.consume_saved_round(f['context'], m.m, **args)

    def test_full_artificial_pipeline_phase_compiler_then_reset(self):
        f = self.fixture()
        with patch.object(caller, 'phase_for_round', wraps=provider.phase_for_round) as phase:
            result = self.invoke(f)
        phase.assert_called_once()
        f['source'].assert_called_once_with(f['meta'], f['context'].prep)
        f['transfer'].assemble.assert_called_once_with(f['meta'].q, f['meta'].r,
            f['cfg'], f['dist'], f['events'])
        f['pilot'].reference_core.assert_called_once_with(f['assembly']['F'], f['assembly']['G'], result['phase'])
        self.assertEqual(f['runtime'].compile_reference.call_count, 2)
        for index, axis in enumerate(('X','Y')):
            wrapped = result['axes'][axis]
            self.assertIs(wrapped[1], result['core'])
            self.assertIs(f['runtime'].compile_reference.call_args_list[index].args[0], wrapped)
            output = result['native_axes'][axis]
            self.assertEqual(output.operations[0], ('reset', tuple(range(8,16))))
            self.assertIs(output.operations[1][1].original, wrapped)
            begin = index*4
            self.assertEqual([x[0] for x in f['log'][begin:begin+4]], ['compile','allocate','reset','compose'])
        self.assertEqual(result['input_provenance']['producer_qualification'], 'NOT_ESTABLISHED_BY_THIS_INTERFACE')

    def test_default_and_non_m17_remain_legacy(self):
        for m, mode in ((17,'legacy'), (0,'exact_binary_m17'), (16,'exact_binary_m17')):
            f = self.fixture(m)
            with self.subTest(m=m), patch.object(caller, 'phase_for_round', side_effect=AssertionError('Wrong opt-in')):
                result = self.invoke(f, mode)
            self.assertEqual(result['phase_mode'], 'legacy')
            self.assertEqual(result['phase'], f['assembly']['phase'])

    def test_schedule_mismatch_stops_before_supplier(self):
        f = self.fixture()
        with self.assertRaises(ValueError):
            self.invoke(f, bits=87)
        f['source'].assert_not_called()
        f['transfer'].assemble.assert_not_called()

    def test_no_default_sampler_or_cohort_fallback(self):
        f = self.fixture()
        with patch.object(consumer, '_runtime', side_effect=AssertionError('Unexpected runtime')), self.assertRaises(NotImplementedError):
            consumer.consume_saved_round(f['context'], 17, q=131072, r=32, bits=88)

    def test_cache_or_active_common_mismatch_rejected(self):
        for defect in ('sha', 'active-common'):
            f = self.fixture()
            if defect == 'sha':
                c = f['context']
                f['context'] = consumer.FixedInputs(c.cache_path, 'wrong', c.common_data, c.prep)
            else:
                f['runtime'].common.return_value = dict(preps={3:object()})
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                self.invoke(f)
            f['source'].assert_not_called()

    def test_bad_supplier_provenance_and_config_rejected(self):
        for defect in ('prep', 'law', 'source', 'tail', 'lambda', 'length'):
            f = self.fixture()
            b = f['batch']
            if defect in ('prep','law','source'):
                from dataclasses import replace
                changes = {'prep':object()} if defect == 'prep' else (
                    {'law':'numpy_conditional_cohort'} if defect == 'law' else {'producer_source':''})
                f['source'].return_value = replace(b, **changes)
            elif defect == 'tail':
                f['cfg'].tail_hash = 'other-tail'
            elif defect == 'lambda':
                f['cfg'].lambda_r = .75
            else:
                f['events'].length = 1
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                self.invoke(f)
            f['transfer'].assemble.assert_not_called()
            f['runtime'].compile_reference.assert_not_called()

    def test_native_shape_rejected_without_reset(self):
        f = self.fixture()
        f['runtime'].compile_reference.side_effect = None
        f['runtime'].compile_reference.return_value = NS(num_qubits=9, num_clbits=1)
        with self.assertRaises(ValueError):
            self.invoke(f)
        f['runtime'].circuit.assert_not_called()

    def test_fixed_cache_binding_loader_reuse_and_environment_restore(self):
        f = self.fixture()
        seen = []

        @lru_cache(None)
        def loader():
            seen.append(os.environ['TRACK_C_H4_COMMON_CACHE'])
            return f['context'].common_data

        runtime = NS(common=loader)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'artificial.bin'
            path.write_bytes(b'artificial fixture, not a scientific pickle')
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(consumer, '_runtime', return_value=runtime), patch.object(consumer, 'CACHE_SHA256', expected), \
                 patch.dict(os.environ, {'TRACK_C_H4_COMMON_CACHE':'preserved-artificial-setting'}):
                context = consumer.load_fixed_inputs(path)
                self.assertIs(context.prep, f['context'].prep)
                self.assertEqual(seen, [str(path.resolve())])
                self.assertEqual(os.environ['TRACK_C_H4_COMMON_CACHE'], 'preserved-artificial-setting')
                with self.assertRaises(ValueError):
                    consumer.load_fixed_inputs(path)  # Does not clear or relabel the existing loader.
                path.write_bytes(b'wrong bytes')
                with self.assertRaises(ValueError):
                    consumer.load_fixed_inputs(path)
        self.assertEqual(loader.cache_info().misses, 1)

    def test_no_quantum_or_science_runtime_imported(self):
        self.assertFalse(any(n == 'qiskit' or n.startswith(('qiskit.', 'numpy', 'trotterlib')) for n in sys.modules))


if __name__ == '__main__':
    unittest.main()
