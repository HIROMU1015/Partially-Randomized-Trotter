"""Artificial complete-pipeline fixtures; real H4 loaders are never invoked."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from flint import ctx

import bridge_pipeline as p
import kernel_bounds as k
import run_scientific_bridge as run
import runtime_checks as runtime
import runtime_policy as policy


def toy():
    export = {'n_qubits': 2, 'n_electrons': 1, 'ordered_sector': [1, 2],
              'bases': {'u': {'operations': []}},
              'components': [{'id': 'P', 'basis_id': 'u', 'support': [0], 'sign': 1}]}
    half = p.bridge.b.encode_matrix_ball(p.bridge.b.unit(p.bridge.b.identity(2)))
    ep = lambda value: dict(lower=value, upper=value)
    # theta=0, P^2=I, exact event phase: 3/4 I - 1/4 I = 1/2 I.
    saved = {'state': {'saved_vector_norm_squared': ep('1')},
             'logical_signed_mean_interval': dict(real=ep('1/2'), imag=ep('0')),
             'axes': {a: {'shot_half_width': ep('1/100')} for a in ('cosine', 'sine')}}
    literals = dict(q='1', r='1', saved_mean=dict(real='.5', imag='0'),
                    axes={a: dict(N='100', alpha='.1', epsilon='.1') for a in ('cosine', 'sine')})
    return dict(export=export, psi=np.array([1., 0.], complex), forward=np.eye(2, dtype=complex),
                reverse=np.eye(2, dtype=complex), angles=[0., 0.], halves=dict(forward=half, reverse=half),
                laws={'order': {'grid_masses': ['3/4', '1/4']}}, literals=literals, saved_enclosure=saved)


def fake_probe(checkpoint):
    checkpoint('probe_start')
    return checkpoint('probe_complete')


class CheckpointFixture:
    def __init__(self, output): self.phases = []
    def __call__(self, phase):
        self.phases.append(phase)
        return {'observed_mxcsr': [0x1f80, 0x1fa2]}


def observed_artificial_fixture(output):
    """For a live GDB test: keep real observer, replace only real-data entrypoints."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    run.result(output, {'SOURCE': {}, 'mode': 'SYNTHETIC_FIXTURE', 'real_H4_calls': 0,
        'scientific_calls': dict(saved_input_load_started=0, numeric_context_started=0,
                                 bound_pass_started=0, bound_pass_completed=0), **run.STATUS})
    with (patch.object(run, 'source_metadata', return_value={}),
          patch.object(run.prior, 'metadata', return_value=[]), patch.object(p, 'load_fixed_data', side_effect=toy),
          patch.object(np, 'load', side_effect=AssertionError('No real arrays')),
          patch.object(np.random, 'PCG64', side_effect=AssertionError('No RNG'))):
        run.worker('science', output)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.prec, self.threads = ctx.prec, ctx.threads
        ctx.prec, ctx.threads = 96, 1

    def tearDown(self):
        ctx.prec, ctx.threads = self.prec, self.threads

    def evaluate(self, data):
        return p.evaluate(data, p.bridge.numeric_components(data['export']), [(1., 0.), (1., 0.)])

    def test_complete_artificial_bound_and_strict_classification(self):
        out = self.evaluate(toy())
        self.assertTrue(out['finite_intermediates_proved'])
        self.assertTrue(out['all_support_completion_proved'])
        self.assertEqual(out['local_model_II_status'], 'CONDITIONAL_VIRTUAL_IID_SUFFICIENT_CONDITION_CONFIRMED')
        self.assertIsNotNone(out['axes']['cosine']['U_virtual_IID_upper'])
        self.assertIsNone(out['U_exec_upper'])
        self.assertFalse(out['fixed_PCG64_IID_certified'])

    def test_strict_epsilon_equality_does_not_pass_integrated_pipeline(self):
        data = toy()
        first = self.evaluate(data)
        for axis in ('cosine', 'sine'):
            value = p.bridge.b.rational(first['axes'][axis]['U_virtual_IID_upper'])+p.bridge.b.rational('1/100')
            data['literals']['axes'][axis]['epsilon'] = str(value)
        out = self.evaluate(data)
        self.assertEqual(out['local_model_II_status'], 'NOT_CERTIFIED_BY_THIS_BOUND')

    def test_guard_unknown_keeps_model_mean_null(self):
        data = toy(); data['forward'] *= 1+2.**-20
        out = self.evaluate(data)
        self.assertFalse(out['all_support_completion_proved'])
        self.assertEqual(out['local_model_II_status'], 'ARITHMETIC_BOUND_OBTAINED_BUT_COMPLETION_UNRESOLVED')
        for axis in out['axes'].values(): self.assertIsNone(axis['U_virtual_IID_upper'])

    def test_finite_envelope_failure_is_not_silently_assumed(self):
        data = toy(); data['forward'] *= 2.**201
        out = self.evaluate(data)
        self.assertFalse(out['finite_intermediates_proved'])
        self.assertEqual(out['local_model_II_status'], 'UNDETERMINED_FINITE_ENVELOPE')
        self.assertIsNone(out['axes']['cosine']['U_virtual_IID_upper'])

    def test_actual_array_layout_and_nonfinite_rejections(self):
        for array in [np.eye(2, dtype=np.float64), np.asfortranarray(np.eye(2, dtype=complex)),
                      np.full((2, 2), np.nan, complex), np.eye(4, dtype=complex)[::2, ::2]]:
            with self.assertRaises(ValueError): runtime.validate_array(array, (2, 2))
        runtime.validate_array(np.eye(2, dtype=complex), (2, 2))

    def test_saved_loader_wiring_on_fully_artificial_seventy_dimensional_metadata(self):
        # Only the schema/scope literals are shared; no physical array is read.
        psi = np.zeros(70, complex); psi[0] = 1
        export = {'state': [[float(z.real).hex(), float(z.imag).hex()] for z in psi],
                  'n_qubits': 8, 'n_electrons': 4, 'identity_policy': 'extract_identity_phase',
                  'coefficient_atol': 1e-12, 'components': [{'probability64': float(i == 0)} for i in range(218)],
                  'bases': dict.fromkeys(range(11)), 'deterministic_blocks': [None]*4}
        laws = {'component': {'grid_bits': 53, 'input_probability64_hex': [float(i == 0).hex() for i in range(218)]},
                'order': {'grid_bits': 53}}
        saved = {'original_fixed_literals': dict(ld='3', m='17', q='131072', r='32', K='2', delta='0.02',
                  axes={a: {'N': '411'} for a in ('cosine', 'sine')}), 'enclosures': [{}], 'fixed_theta64_hex': ['0x0p0']*2}
        class Arrays(dict):
            def __enter__(self): return self
            def __exit__(self, *args): pass
        arrays = [Arrays(state=psi), Arrays(forward=np.eye(70, dtype=complex), reverse=np.eye(70, dtype=complex))]
        documents = {'RESULT.json': saved, 'source_replay.json': export, 'categorical_laws.json': laws}
        with (patch.object(Path, 'read_text', lambda path: json.dumps(documents[path.name])),
              patch.object(np, 'load', side_effect=arrays), patch.object(p.zipfile, 'ZipFile') as zipclass):
            archive = zipclass.return_value.__enter__.return_value
            archive.namelist.return_value = ['96.json']; archive.read.return_value = '{}'
            data = p.load_fixed_data()
            self.assertEqual(data['psi'].tobytes(), psi.tobytes())
            self.assertEqual(data['angles'], [0., 0.])
        bad_export = copy.deepcopy(export); bad_export['state'][0][1] = '-0x0p0'
        documents['source_replay.json'] = bad_export
        with (patch.object(Path, 'read_text', lambda path: json.dumps(documents[path.name])),
              patch.object(np, 'load', side_effect=arrays), patch.object(p.zipfile, 'ZipFile') as zipclass):
            archive = zipclass.return_value.__enter__.return_value
            archive.namelist.return_value = ['96.json']; archive.read.return_value = '{}'
            with self.assertRaisesRegex(ValueError, 'byte identity'): p.load_fixed_data()

    def test_probe_requires_all_dispatches_and_full_output_partition(self):
        counts = dict.fromkeys(policy.REQUIRED, 1)
        policy.validate_probe(counts, 70)
        with self.assertRaises(ValueError): policy.validate_probe(counts, 69)
        counts.pop('hypot')
        with self.assertRaises(ValueError): policy.validate_probe(counts, 70)
        for value in [0x1f80 | 0x40, 0x1f80 | 0x8000, 0x1f80 | 0x2000]:
            self.assertFalse(policy.mode_supported(value))
            self.assertEqual(policy.mode_supported(value), k.fp_mode_supported(value))

    def test_checkpoint_rejects_missing_or_stale_observer_ack(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(policy.os, 'kill'):
            checkpoint = policy.Checkpoints(directory)
            with self.assertRaises(FileNotFoundError): checkpoint('probe_start')
            policy.write_json(Path(directory)/'runtime.json', dict(sequence=1, phase='probe_start', status='RUNTIME_CONDITIONS_CONFIRMED'))
            with self.assertRaises(ValueError): checkpoint('probe_complete')

    def test_full_worker_orchestration_uses_only_artificial_context(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run.result(output, {'SOURCE': {}, 'scientific_calls': dict(saved_input_load_started=0,
                numeric_context_started=0, bound_pass_started=0, bound_pass_completed=0), **run.STATUS})
            with (patch.object(run, 'source_metadata', return_value={}),
                  patch.object(run, 'Checkpoints', CheckpointFixture),
                  patch.object(runtime, 'binary_profile', return_value=k.PROFILE),
                  patch.object(runtime, 'probe', side_effect=fake_probe),
                  patch.object(run.prior, 'metadata', return_value=[]),
                  patch.object(p, 'load_fixed_data', side_effect=toy),
                  patch.object(np, 'load', side_effect=AssertionError('No real arrays')),
                  patch.object(np.random, 'PCG64', side_effect=AssertionError('No RNG'))):
                run.worker('science', output)
            record = json.loads((output/'RESULT.json').read_text())
            self.assertEqual(record['execution_status'], 'COMPLETED_MANDATORY_STOP')
            self.assertEqual(record['scientific_calls']['bound_pass_started'], 1)
            self.assertEqual(record['scientific_calls']['bound_pass_completed'], 1)
            self.assertTrue((output/'numeric_context.npz').exists())
            self.assertIsNone(record['U_exec_upper'])
            self.assertFalse(record['RPE_tolerance_certified'])

    def test_worker_bad_profile_stops_before_numerical_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run.result(output, {'SOURCE': {}, 'scientific_calls': dict(saved_input_load_started=0)})
            with (patch.object(run, 'source_metadata', return_value={}),
                  patch.object(runtime, 'binary_profile', return_value={}),
                  patch.object(p, 'load_fixed_data', side_effect=AssertionError('No H4'))):
                with self.assertRaises(ValueError): run.worker('science', output)
            record = json.loads((output/'RESULT.json').read_text())
            self.assertEqual(record['execution_status'], 'INTERRUPTED_OR_INVALID_MANDATORY_STOP')
            self.assertEqual(record['scientific_calls']['saved_input_load_started'], 0)

    def test_late_runtime_failure_prevents_publishing_a_candidate_pass(self):
        class BadAfterBound(CheckpointFixture):
            def __call__(self, phase):
                if phase == 'after_bound': raise ValueError('Artificial unconfirmed thread')
                return super().__call__(phase)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run.result(output, {'SOURCE': {}, 'scientific_calls': dict(saved_input_load_started=0,
                numeric_context_started=0, bound_pass_started=0, bound_pass_completed=0)})
            with (patch.object(run, 'source_metadata', return_value={}), patch.object(run, 'Checkpoints', BadAfterBound),
                  patch.object(runtime, 'binary_profile', return_value=k.PROFILE),
                  patch.object(runtime, 'probe', side_effect=fake_probe), patch.object(run.prior, 'metadata', return_value=[]),
                  patch.object(p, 'load_fixed_data', side_effect=toy), patch.object(np, 'load', side_effect=AssertionError('No real arrays'))):
                with self.assertRaises(ValueError): run.worker('science', output)
            record = json.loads((output/'RESULT.json').read_text())
            self.assertNotIn('bound', record)
            self.assertEqual(record['scientific_calls']['bound_pass_completed'], 0)
            self.assertTrue((output/'numeric_context.npz').exists())

    def test_fresh_output_and_missing_observer_stop_preserve_record(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'fresh'
            with patch.object(run, 'source_metadata', return_value={}), patch.object(run.shutil, 'which', return_value=None):
                with self.assertRaises(ValueError): run.execute('runtime', output)
            original = (output/'RESULT.json').read_bytes()
            with self.assertRaises(FileExistsError): run.execute('runtime', output)
            self.assertEqual((output/'RESULT.json').read_bytes(), original)
            self.assertEqual(json.loads(original)['scientific_calls']['bound_pass_started'], 0)

    def test_default_and_source_mismatch_cannot_launch_science(self):
        with (patch.object(np, 'load', side_effect=AssertionError('No numerical load')),
              patch.object(p, 'load_fixed_data', side_effect=AssertionError('No H4')),
              patch.object(run.subprocess, 'Popen', side_effect=AssertionError('No child')),
              contextlib.redirect_stdout(io.StringIO()) as output):
            # Metadata uses check_output internally; mock only its byte inventory.
            with patch.object(run.prior, 'metadata', return_value=[]): run.main([])
            self.assertEqual(json.loads(output.getvalue())['real_numerical_calls'], 0)
        with patch.object(run.prior, 'fixed_bytes', return_value=b'changed'):
            with self.assertRaisesRegex(ValueError, 'Freeze numerical SOURCE'): run.source_metadata()


if __name__ == '__main__': unittest.main()
