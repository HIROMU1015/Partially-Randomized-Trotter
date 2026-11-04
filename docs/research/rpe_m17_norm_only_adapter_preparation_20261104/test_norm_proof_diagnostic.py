"""Artificial vectors/ZIP payloads and exact structural constants only."""
import contextlib
from fractions import Fraction as Q
from io import BytesIO, StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from flint import acb_mat, ctx

import run_norm_proof_diagnostic as run
import virtual_bridge_adapter as old


def fixture_zip(psi=None):
    psi = np.eye(1, 70, dtype=np.complex128)[0] if psi is None else psi
    serialized, archive = BytesIO(), BytesIO()
    np.save(serialized, psi, allow_pickle=False)
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('psi.npy', serialized.getvalue())
        # Deliberately invalid matrix payloads prove no matrices are decoded.
        for name in [f'P{i:03d}' for i in range(218)]+['forward', 'reverse']:
            z.writestr(name+'.npy', b'unread artificial matrix')
    return archive.getvalue()


def fixture_inputs():
    result = dict(SOURCE=dict(commit='c3a5cc387b8962b63782dc71ab175147f2d87e33'),
                  execution_status='COMPLETED_MANDATORY_STOP',
                  fixed_literals=dict(ld='3', m='17', q='131072', r='32', K='2', delta='0.02',
                                      axes={a: dict(N='411', alpha='unchanged', epsilon='unchanged')
                                            for a in ['cosine', 'sine']}, saved_betas='unchanged'))
    return {'numeric_context.npz': fixture_zip(), 'RESULT.json': json.dumps(result).encode()}, []


class NormProofDiagnosticTests(unittest.TestCase):
    def test_structural_floor_uses_existing_coefficients_and_both_guards(self):
        with patch.object(np, 'load', side_effect=AssertionError('No real arrays')):
            factor = run.proof_floor()
            bounds = run.kernel.path_bounds(70, 70)
            self.assertEqual(factor, (1+bounds.dot_rel)*(1+262144*8*bounds.mv_rel))
            self.assertGreater(factor, run.COORDINATE_LIMIT)
            self.assertGreater(factor, run.MAGNITUDE_LIMIT)
            self.assertGreater(bounds.mv_abs, 0)  # Discarded only in a lower bound on a majorant.
        self.assertEqual(run.COORDINATE_LIMIT, Q.from_float(1.+1e-12))
        self.assertEqual(run.MAGNITUDE_LIMIT, Q.from_float(1.+1e-10))

    def test_nonunitary_toy_matrices_obey_universal_floor(self):
        previous = ctx.prec
        ctx.prec = 96
        try:
            kappa, identity = old.b.real(Q(1, 16)), old.b.identity(2)
            for matrix in [acb_mat([[0, 0], [0, 0]]), identity, -identity,
                           identity*old.b.real(Q(1, 2)), identity*2,
                           acb_mat([[1, old.b.real(Q(1, 8))], [0, 1]])]:
                rho = old.b.frobenius(old.b.dagger(matrix)*matrix-identity)
                g = old.b.upper((1+rho).sqrt()+kappa*old.b.frobenius(matrix))
                self.assertGreaterEqual(g, 1+kappa)  # isqrt(2)=1
        finally:
            ctx.prec = previous

    def test_floor_boundary_is_strict_and_never_certifies_measurement(self):
        critical = run.COORDINATE_LIMIT/run.proof_floor()
        equal = run.assess_norm_squared(critical)
        self.assertEqual(equal['method_status'], 'FLOOR_INCONCLUSIVE')
        self.assertFalse(equal['coordinate_bound_obstructed'])
        self.assertTrue(run.assess_norm_squared(critical+Q(1, 1 << 100))['coordinate_bound_obstructed'])
        for value in [critical, Q(1, 2), Q(1)]:
            result = run.assess_norm_squared(value)
            self.assertIsNone(result['U_virtual_IID_upper'])
            self.assertIsNone(result['U_exec_upper'])
            self.assertEqual(result['scientific_status'], 'UNDETERMINED')
            self.assertFalse(result['all_support_completion_proved'])

    def test_long_chain_floor_without_constructing_a_long_rational_power(self):
        short, long = run.proof_floor(q=1), run.proof_floor(q=10**12)
        self.assertLess(short, long)
        self.assertLess(short, run.COORDINATE_LIMIT)
        self.assertGreater(long, run.COORDINATE_LIMIT)
        # Subtract the same dot contribution: exact linear Bernoulli lower bound.
        dot = run.kernel.path_bounds(70, 70).dot_rel
        self.assertEqual(long/(1+dot)-1, (short/(1+dot)-1)*10**12)

    def test_exact_dyadic_norm_retains_subnormal_squares(self):
        self.assertEqual(run.exact_norm_squared(np.array([.5+.25j, .125-.5j], complex)), Q(37, 64))
        tiny = float.fromhex('0x0.0000000000001p-1022')
        self.assertEqual(run.exact_norm_squared(np.array([tiny+0j], complex)), Q(1, 1 << 2148))

    def test_state_only_load_does_not_decode_other_npz_members(self):
        state = run.load_state(fixture_zip())
        self.assertEqual(state.shape, (70,))
        self.assertEqual(run.exact_norm_squared(state), 1)
        with self.assertRaises(ValueError): run.load_state(fixture_zip(np.zeros(2, complex)))

    def test_invalid_vectors_and_scope_stop(self):
        for vector in [np.zeros(2, np.float32), np.zeros((2, 2), complex),
                       np.zeros(8, complex)[::2], np.array([complex(float('nan'), 0)])]:
            with self.assertRaises(ValueError): run.exact_norm_squared(vector)
        for n2 in [0, -1, True, 1.0]:
            with self.assertRaises(ValueError): run.assess_norm_squared(n2)
        for q in [0, -1, True]:
            with self.assertRaises(ValueError): run.proof_floor(q=q)

    def test_default_description_never_enters_scientific_loader(self):
        with (patch.object(run, 'source_metadata', return_value={}),
              patch.object(run, 'checked_inputs', return_value=({}, [])),
              patch.object(np, 'load', side_effect=AssertionError('No arrays')),
              patch.object(run, 'exact_norm_squared', side_effect=AssertionError('No norm')),
              contextlib.redirect_stdout(StringIO()) as capture):
            self.assertEqual(run.main([]), 0)
            record = json.loads(capture.getvalue())
            self.assertEqual(record['new_real_numerical_calls'], 0)
            self.assertFalse(record['virtual_bridge_execution_ready'])

    def test_artificial_execution_preserves_status_and_stops_after_one_pass(self):
        with (tempfile.TemporaryDirectory() as temporary, patch.object(run, 'ROOT', Path(temporary)),
              patch.object(run, 'source_metadata', return_value={'artificial': True}),
              patch.object(run, 'checked_inputs', side_effect=fixture_inputs)):
            output = Path(temporary)/'artifacts/rpe_m17_norm_proof_feasibility/toy'
            record = run.execute(output)
            self.assertEqual(record['execution_status'], 'COMPLETED_MANDATORY_STOP')
            self.assertEqual(record['diagnostic']['method_status'], 'NORM_PROOF_OBSTRUCTION_CONFIRMED')
            self.assertEqual(record['scientific_calls'],
                             dict(saved_context_load_started=1, norm_pass_started=1, norm_pass_completed=1))
            self.assertEqual(record['fixed_literals']['saved_betas'], 'unchanged')
            with self.assertRaises(FileExistsError): run.execute(output)
            self.assertEqual(json.loads((output/'RESULT.json').read_text()), record)

    def test_frozen_source_or_input_failure_stops_before_loading(self):
        for failure in ['source', 'input']:
            with (self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary,
                  patch.object(run, 'ROOT', Path(temporary)),
                  patch.object(run, 'source_metadata', side_effect=ValueError('SOURCE') if failure == 'source' else None,
                               return_value={}),
                  patch.object(run, 'checked_inputs', side_effect=ValueError('INPUT')),
                  patch.object(np, 'load', side_effect=AssertionError('No arrays'))):
                output = Path(temporary)/'artifacts/rpe_m17_norm_proof_feasibility/toy'
                with self.assertRaises(ValueError): run.execute(output)
                record = json.loads((output/'RESULT.json').read_text())
                self.assertEqual(record['execution_status'], 'INTERRUPTED_OR_INVALID_MANDATORY_STOP')
                self.assertEqual(record['scientific_calls']['saved_context_load_started'], 0)

    def test_interruption_is_recorded_without_retry(self):
        with (tempfile.TemporaryDirectory() as temporary, patch.object(run, 'ROOT', Path(temporary)),
              patch.object(run, 'source_metadata', return_value={}),
              patch.object(run, 'checked_inputs', side_effect=fixture_inputs),
              patch.object(run, 'exact_norm_squared', side_effect=KeyboardInterrupt) as norm):
            output = Path(temporary)/'artifacts/rpe_m17_norm_proof_feasibility/toy'
            with self.assertRaises(KeyboardInterrupt): run.execute(output)
            record = json.loads((output/'RESULT.json').read_text())
            norm.assert_called_once()
            self.assertEqual(record['scientific_calls']['norm_pass_completed'], 0)
            self.assertEqual(record['execution_status'], 'INTERRUPTED_OR_INVALID_MANDATORY_STOP')


if __name__ == '__main__':
    unittest.main()
