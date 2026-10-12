"""Synthetic matrix contracts only; never diagnoses the two S1 instances."""
from concurrent.futures import ProcessPoolExecutor
import copy
from fractions import Fraction as F
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_n3_old_family_v1 import BRANCH, RUN, validate_authorization, worker
from trottertracks.representation_exploration.n3_old_family_contract_v1 import (
    Arithmetic, CAPS, decode, encode, fingerprint,
)
from trottertracks.representation_exploration.n3_old_family_check_v1 import verify
from trottertracks.representation_exploration.n3_old_family_v1 import diagnose, mixture_weight, schur_PSD


def fixture(name, A, v, nu=1, U=F(0), c=F(1), delta=F(1, 10)):
    return encode({'id': 'synthetic_' + name, 'h_A': [[F(x) for x in r] for r in A],
                   'v': [F(x) for x in v], 'nu': nu, 'U': F(U), 'c': F(c), 'delta': F(delta)})


PRIMAL = fixture('interior_primal', [[1, 0], [0, -1]], [F(2, 5), 0])
DUAL = fixture('mixed_dual', [[1, 0], [0, -1]], [F(1, 2), 0])
RAY = fixture('ray_dual', [[-1, 0], [0, -1]], [1, 0])


class OldFamilyTests(unittest.TestCase):
    def test_interior_primal_exact_threshold_and_original_matrix(self):
        record = diagnose(PRIMAL)
        self.assertEqual(record['status'], 'OLD_FAMILY_RECOVERS_SAVED_OCC_WIDTH')
        self.assertEqual(record['resources']['counts']['numeric_eigh_calls'], 0)
        self.assertFalse(record['NORM_at_target'])
        audit = verify(PRIMAL, record)
        self.assertEqual(audit['mathematical_classification'], record['status'])
        k, mu = F(*record['certificate']['kappa']), F(*record['certificate']['mu'])
        A, v, nu, U, c, delta = decode(PRIMAL); D = c - U + delta
        self.assertEqual(mu, (k * U - delta * D) / nu)
        self.assertTrue(0 <= k <= D)
        self.assertGreaterEqual(k - v[0] ** 2 - mu, 0)
        self.assertGreaterEqual(-k - mu, 0)

    def test_mixed_dual_covers_entire_interval(self):
        record = diagnose(DUAL); audit = verify(DUAL, record)
        self.assertEqual(record['status'], 'CERTIFIED_NORM_AND_OLD_FAMILY_SEPARATION')
        self.assertEqual(record['resources']['counts']['numeric_eigh_calls'], 0)
        self.assertGreater(len(record['certificate']['origins']), 1)
        traces = [F(*x) for x in record['certificate']['endpoint_traces']]
        for j in range(11): self.assertLess((1 - F(j, 10)) * traces[0] + F(j, 10) * traces[1], 0)
        self.assertEqual(audit['PSD_checks'], 1)

    def test_rank_one_analytic_separator(self):
        record = diagnose(RAY)
        self.assertEqual(len(record['certificate']['origins']), 1)
        self.assertEqual(verify(RAY, record)['status'], 'INDEPENDENT_EXACT_OLD_FAMILY_CERTIFICATE_PASS')

    def test_norm_recovers_does_not_become_union_separation(self):
        case = fixture('norm_only', [[-1, 0], [0, -1]], [F(1, 2), 0], nu=2, c=1, delta=F(1, 4))
        record = diagnose(case)
        self.assertTrue(record['NORM_at_target'])
        self.assertEqual(record['status'], 'OLD_FAMILY_SEPARATED_NORM_RECOVERS_TARGET')
        verify(case, record)

    def test_non_dyadic_singleton_proposals_remain_unknown(self):
        # Exact feasible k=1/3, missed by the fixed grid and dyadic probes.
        case = fixture('singleton', [[F(3, 2), 0], [0, F(-3, 2)]], [1, 0], c=F(1, 2), delta=F(1, 2))
        record = diagnose(case)
        self.assertEqual(record['status'], 'UNDETERMINED_FINITE_PROPOSALS')
        self.assertIsNone(record['certificate'])
        self.assertGreater(record['resources']['counts']['numeric_eigh_calls'], 0)
        self.assertLessEqual(record['resources']['counts']['numeric_eigh_calls'], CAPS['numeric_eigh_calls'])
        self.assertEqual(verify(case, record)['mathematical_classification'], 'UNDETERMINED')

    def test_numeric_eigenvector_becomes_exact_dual_certificate(self):
        case = fixture('numeric_dual', [[1, 2], [2, 1]], [1, 0])
        record = diagnose(case)
        self.assertEqual(record['status'], 'CERTIFIED_NORM_AND_OLD_FAMILY_SEPARATION')
        self.assertGreater(record['resources']['counts']['numeric_eigh_calls'], 0)
        self.assertEqual(verify(case, record)['status'], 'INDEPENDENT_EXACT_OLD_FAMILY_CERTIFICATE_PASS')

    def test_numeric_abscissa_becomes_exact_primal_certificate(self):
        case = fixture('numeric_primal', [[F(3, 2), 0], [0, F(-3, 2)]], [1, 0],
                       c=F(1, 2), delta=F(5001, 10000))
        record = diagnose(case)
        self.assertEqual(record['status'], 'OLD_FAMILY_RECOVERS_SAVED_OCC_WIDTH')
        self.assertGreater(record['resources']['counts']['numeric_eigh_calls'], 0)
        self.assertEqual(verify(case, record)['status'], 'INDEPENDENT_EXACT_OLD_FAMILY_CERTIFICATE_PASS')

    def test_exact_singular_PSD_and_indefinite_zero_pivot(self):
        for M, valid in [([[0, 0], [0, 1]], True), ([[0, 1], [1, 0]], False),
                         ([[1, 1], [1, 1]], True), ([[1, 2], [2, 1]], False)]:
            self.assertEqual(schur_PSD([[F(x) for x in r] for r in M], Arithmetic()), valid)

    def test_mixture_strict_endpoint_inequalities(self):
        a = Arithmetic()
        t = mixture_weight((F(-1), F(2)), (F(2), F(-5)), a)
        self.assertIsNotNone(t)
        self.assertLess(t * -1 + (1 - t) * 2, 0)
        self.assertLess(t * 2 + (1 - t) * -5, 0)
        self.assertIsNone(mixture_weight((F(0), F(-1)), (F(0), F(-2)), Arithmetic()))

    def test_energy_translation_preserves_affine_problem(self):
        case = copy.deepcopy(DUAL); A, v, nu, U, c, delta = decode(case); shift = F(7, 3)
        for i in range(len(A)): A[i][i] += shift
        case.update(h_A=encode(A), U=encode(U + nu * shift), c=encode(c + nu * shift))
        first, moved = diagnose(DUAL), diagnose(case)
        self.assertEqual(moved['status'], first['status'])
        self.assertEqual(moved['certificate']['endpoint_traces'], first['certificate']['endpoint_traces'])
        verify(case, moved)

    def test_energy_scaling_preserves_classification(self):
        A, v, nu, U, c, delta = decode(PRIMAL); scale = F(3, 2)
        case = fixture('scaled', [[x * scale for x in r] for r in A], [x * scale for x in v], nu,
                       U * scale, c * scale, delta * scale)
        record = diagnose(case); self.assertEqual(record['status'], diagnose(PRIMAL)['status']); verify(case, record)

    def test_malformed_or_oracle_inputs_rejected(self):
        variants = []
        for key, value in [('true_gap', [1, 1]), ('nu', True), ('delta', [0, 1]), ('U', [False, 1]), ('c', [1., 1])]:
            x = copy.deepcopy(PRIMAL); x[key] = value; variants.append(x)
        x = copy.deepcopy(PRIMAL); x['h_A'][0][1] = [1, 1]; variants.append(x)
        x = copy.deepcopy(PRIMAL); x['v'].pop(); variants.append(x)
        for x in variants:
            with self.assertRaises(ValueError): decode(x)

    def test_arithmetic_caps_rejected(self):
        with self.assertRaises(ArithmeticError): Arithmetic().q(F(1 << 4097))
        a = Arithmetic(); a.operations = CAPS['arithmetic_operations']
        with self.assertRaises(ArithmeticError): a.q(1)

    def test_primal_bad_kappa_mu_and_identity_rejected(self):
        good = diagnose(PRIMAL)
        for field, value in [('kappa', [-1, 1]), ('mu', [0, 1]), ('kappa', [10, 1])]:
            record = copy.deepcopy(good); record['certificate'][field] = value
            with self.assertRaises(ValueError): verify(PRIMAL, record)
        wrong = copy.deepcopy(good); wrong['instance_sha256'] = '0' * 64
        with self.assertRaises(ValueError): verify(PRIMAL, wrong)

    def test_dual_false_PSD_zero_trace_and_endpoints_rejected(self):
        good = diagnose(DUAL)
        for field, value in [('W', [[[1, 1], [2, 1]], [[2, 1], [1, 1]]]),
                             ('W', [[[0, 1], [0, 1]], [[0, 1], [0, 1]]]),
                             ('endpoint_traces', [[0, 1], [-1, 1]]), ('trace_W', [0, 1])]:
            wrong = copy.deepcopy(good); wrong['certificate'][field] = value
            with self.assertRaises(ValueError): verify(DUAL, wrong)

    def test_false_status_and_extra_truth_fields_rejected(self):
        record = diagnose(PRIMAL); record['status'] = 'CERTIFIED_NORM_AND_OLD_FAMILY_SEPARATION'
        with self.assertRaises(ValueError): verify(PRIMAL, record)
        record = diagnose(PRIMAL); record['true_ground'] = [0, 1]
        with self.assertRaises(ValueError): verify(PRIMAL, record)

    def test_resource_and_stop_tampering_rejected(self):
        for change in ('counts', 'events', 'stop'):
            record = diagnose(DUAL)
            if change == 'counts': record['resources']['counts']['numeric_eigh_calls'] = 65
            elif change == 'events': record['events'] = []
            else: record['next_stage_authorized'] = True
            with self.assertRaises(ValueError): verify(DUAL, record)

    def test_two_spawn_workers_and_independent_verifier(self):
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
            records = list(pool.map(worker, [PRIMAL, DUAL]))
        for case, record in zip((PRIMAL, DUAL), records): verify(case, record)
        self.assertEqual([r['status'] for r in records], ['OLD_FAMILY_RECOVERS_SAVED_OCC_WIDTH', 'CERTIFIED_NORM_AND_OLD_FAMILY_SEPARATION'])

    def test_checker_import_does_not_load_producer_or_numpy(self):
        code = "import sys; import trottertracks.representation_exploration.n3_old_family_check_v1; assert 'numpy' not in sys.modules; assert 'trottertracks.representation_exploration.n3_old_family_v1' not in sys.modules"
        subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True)

    def test_missing_execution_authorization_rejected_before_plan_load(self):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/run_n3_old_family_v1.py'), '--diagnose'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('separate explicit', result.stderr)
        self.assertFalse((ROOT / RUN).exists())

    def test_new_scope_binding_rejects_old_or_changed_authorization(self):
        plan = {'source_commit': 'a' * 40, 'source_fingerprint': 'b' * 64, 'input_fingerprint': 'c' * 64,
                'original_payload_commit': 'd' * 40, 'ledger_sha256': 'e' * 64}
        auth = dict(plan, scope='n3_old_family_v1_two_S1_one_diagnostic', diagnostic_authorized=True,
                    plan_sha256='f' * 64, attempts=1, retries=0, execution_branch=BRANCH,
                    output_path=RUN, next_stage_authorized=False, central_theme=None)
        validate_authorization(auth, plan, 'f' * 64)
        for key, value in [('scope', 'n3_saved_width_v1_one_posthoc_evaluation'), ('diagnostic_authorized', False),
                           ('source_commit', '0' * 40), ('attempts', True), ('retries', 1), ('plan_sha256', '0' * 64),
                           ('execution_branch', 'other'), ('output_path', 'other'), ('next_stage_authorized', True)]:
            wrong = dict(auth); wrong[key] = value
            with self.assertRaises(ValueError): validate_authorization(wrong, plan, 'f' * 64)

    def test_nine_dimensional_exact_checker_reference(self):
        # Dimension matches planned workload, coefficients are synthetic only.
        A = [[F(1 if i == j else 0) for j in range(9)] for i in range(9)]
        case = fixture('dimension9', A, [F(1, 20)] + [0] * 8, nu=4)
        record = diagnose(case); self.assertEqual(record['status'], 'OLD_FAMILY_RECOVERS_SAVED_OCC_WIDTH')
        verify(case, record)

    def test_dense_nine_dimensional_reference_checker(self):
        z = [F(i + 1, 10) for i in range(9)]
        A = [[F(2 if i == j else 0) + z[i] * z[j] for j in range(9)] for i in range(9)]
        case = fixture('dense_dimension9', A, [F(1, 10)] * 9, nu=4, delta=F(1, 1000))
        record = diagnose(case)
        self.assertEqual(record['status'], 'OLD_FAMILY_RECOVERS_SAVED_OCC_WIDTH')
        self.assertGreater(record['resources']['counts']['primal_PSD_probes'], 1)
        self.assertEqual(verify(case, record)['status'], 'INDEPENDENT_EXACT_OLD_FAMILY_CERTIFICATE_PASS')


if __name__ == '__main__': unittest.main()
