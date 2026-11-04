"""Artificial inputs only; no saved H4 arrays or numerical claims are evaluated."""
import contextlib
import io
import json
from fractions import Fraction as Q
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from flint import acb, acb_mat, arb, ctx
import alpha_adapter as a
from common import D, ball, real
import decoder_finite as b
import reference_adapter as c
import run_three_gaps as run


class ArbTests(unittest.TestCase):
    def setUp(self):
        self.prec = ctx.prec
        ctx.prec = 160

    def tearDown(self):
        ctx.prec = self.prec


class AlphaTests(ArbTests):
    def test_exponential_contains_independent_higher_precision_value(self):
        out = a.failure_upper(32, '1/2', 0)
        with ctx.workprec(320):
            self.assertGreaterEqual(real(out['probability_upper']['upper']), 2*real(-4).exp())
        self.assertEqual(a.aggregate([out])['decision'], 'DIRECT_HOEFFDING_95_SUFFICIENT')

    def test_mean_error_consumes_the_same_coordinate_budget(self):
        clean = a.failure_upper(32, '1/2', 0)
        biased = a.failure_upper(32, '1/2', '1/4')
        self.assertGreater(Q(biased['probability_upper']['upper']), Q(clean['probability_upper']['upper']))

    def test_zero_negative_or_uncertain_t_stays_unknown(self):
        for u in ('1/2', '3/4', arb('.5', '.01')):
            self.assertEqual(a.failure_upper(32, '.5', u)['decision'], 'UNKNOWN/STOP')
        bad = a.failure_upper(32, '.5', '.5')
        self.assertIsNone(a.aggregate([bad])['failure_upper'])

    def test_closed_target_equality_and_strict_excess(self):
        item = dict(probability_upper={'lower':'1/64', 'upper':'1/64'})
        self.assertEqual(a.aggregate([item, item], '1/32')['decision'], 'DIRECT_HOEFFDING_95_SUFFICIENT')
        self.assertEqual(a.aggregate([item, item], '0.031249999999999999')['decision'], 'UNKNOWN/STOP')

    def test_outward_sum_of_dyadic_probability_endpoints(self):
        bounds = [a.failure_upper(n, '.5', '.01') for n in (200, 250, 300)]
        exact = sum((Q(x['probability_upper']['upper']) for x in bounds), Q())
        out = a.aggregate(bounds)
        self.assertEqual(Q(out['failure_upper_exact']), exact)
        self.assertGreaterEqual(Q(out['failure_upper']['upper']), exact)

    def test_no_float_underflow_to_probability_zero(self):
        out = a.failure_upper(2000, 1, 0)
        endpoint = Q(out['probability_upper']['upper'])
        self.assertGreater(endpoint, 0)
        self.assertLess(endpoint, Q(1, 10**400))

    def test_invalid_unproved_inputs_stop(self):
        for n, eps, u in ((0, '.1', 0), (True, '.1', 0), (10, 0, 0), (10, '.1', -1), (10, '.1', float('nan'))):
            with self.assertRaisesRegex(ValueError, 'UNKNOWN/STOP'):
                a.failure_upper(n, eps, u)
        with self.assertRaisesRegex(ValueError, 'UNKNOWN/STOP'):
            a.aggregate([dict(probability_upper={'upper':'-1/100'})])


class ReferenceTests(ArbTests):
    def test_ground_identified_by_positive_shift_and_rayleigh(self):
        h = c.hermitian_lower(acb_mat([[-2, 0], [0, 1]]))
        out = c.ground_bracket(h, [2, 0], -2, '1/10')
        self.assertEqual(out['decision'], 'REPRESENTED_DF_SECTOR_GROUND_BRACKET')
        self.assertLessEqual(Q(out['eigenvalue_interval']['lower']), -2)
        self.assertGreaterEqual(Q(out['eigenvalue_interval']['upper']), -2)
        self.assertGreaterEqual(Q(out['eta_ref_upper']['upper']), Q(1, 10))

    def test_zero_residual_excited_eigenstate_does_not_identify_ground(self):
        h = c.hermitian_lower(acb_mat([[-2, 0], [0, 1]]))
        out = c.ground_bracket(h, [0, 1], 1, '1/10')
        self.assertEqual(out['decision'], 'UNKNOWN/STOP')
        self.assertIsNone(out['eta_ref_upper'])

    def test_shift_on_eigenvalue_is_not_strict_positive(self):
        h = c.hermitian_lower(acb_mat([[0, 0], [0, 1]]))
        proof, _ = c.positive_shift(h, 0)
        self.assertEqual(proof['decision'], 'UNKNOWN/STOP')

    def test_interval_pivot_crossing_zero_stays_unknown(self):
        h = c.hermitian_lower(acb_mat([[arb(0, '.1'), 0], [0, 1]]))
        proof, _ = c.positive_shift(h, 0)
        self.assertEqual(proof['decision'], 'UNKNOWN/STOP')

    def test_complex_hermitian_and_unnormalized_ray(self):
        h = c.hermitian_lower(acb_mat([[2, 1j], [-1j, 2]]))
        out = c.ground_bracket(h, [3, 3j], 1, '1/100')
        self.assertEqual(out['decision'], 'REPRESENTED_DF_SECTOR_GROUND_BRACKET')
        self.assertTrue(D.interval(out['rayleigh_interval']).contains(1))
        self.assertTrue(D.interval(out['norm_squared']).contains(18))

    def test_degenerate_ground_needs_no_gap_assumption(self):
        h = c.hermitian_lower(acb_mat([[0, 0], [0, 0]]))
        out = c.ground_bracket(h, [1, 1], 0, '.01')
        self.assertEqual(out['positive_shift']['positive_pivots'], 2)

    def test_nonhermitian_coefficients_or_raw_target_stop(self):
        with self.assertRaisesRegex(ValueError, 'UNKNOWN/STOP'):
            c.coefficient_matrix([[1, 2], [3, 4]])
        with self.assertRaisesRegex(ValueError, 'UNKNOWN/STOP'):
            c.ground_bracket(acb_mat([[1, 2], [3, 4]]), [1, 0], 0, '.1')
        with self.assertRaisesRegex(ValueError, 'UNKNOWN/STOP'):
            c.coefficient_matrix([[float('nan')]])

    def test_zero_state_and_incomplete_sector_stop(self):
        h = c.hermitian_lower(acb_mat([[1]]))
        with self.assertRaisesRegex(ValueError, 'UNKNOWN/STOP'):
            c.ground_bracket(h, [0], 0, '.1')
        with self.assertRaisesRegex(ValueError, 'UNKNOWN/STOP'):
            c.complete_number_sector(2, 1, [1])

    def test_reused_excitation_map_and_DF_square_match_known_toy(self):
        sector = c.complete_number_sector(2, 1, [1, 2])
        h, records = c.represented_df(-.1, [[1, .25], [.25, 2]], [.5], [[[1, 0], [0, -1]]], sector)
        self.assertTrue(h.matrix[0, 0].real.contains(real(-.1)+real(2)+real(.5)))
        self.assertTrue(h.matrix[1, 1].real.contains(real(-.1)+real(1)+real(.5)))
        self.assertTrue(h.matrix[0, 1].real.contains(real('.25')))
        self.assertTrue(all(x['unchanged_ast'] for x in records.values()))

    def test_fermionic_offdiagonal_sign_is_preserved(self):
        basis = [x for x in range(16) if x.bit_count() == 2]
        sector = c.complete_number_sector(4, 2, basis)
        h = np.zeros((4, 4), dtype=complex)
        h[3, 0] = h[0, 3] = 1
        out, _ = c.represented_df(0, h, [], [], sector)
        self.assertTrue(out.matrix[basis.index(3), basis.index(10)].contains(-1))


def toy_round(q):
    return dict(q=q, N_cosine=4, N_sine=4, mu_cosine='1', mu_sine='0',
                eps_cosine='3/5', eps_sine='3/5')


class DecoderTests(ArbTests):
    def test_entire_supplied_prefix_not_selected_examples(self):
        result = b.finite_prefix_check([toy_round(1), toy_round(2)], 0)
        self.assertEqual(result['feasible_pairs_per_round'], [6, 6])
        self.assertEqual(result['traces'], 36)
        self.assertEqual(result['decision'], 'EXHAUSTIVE_SUPPLIED_FINITE_PREFIX_ONLY')
        self.assertEqual(result['H4_uniform_decoder_decision'], 'UNKNOWN/STOP')

    def test_wrong_branch_is_detected_on_supplied_domain(self):
        f, _ = D.existing_functions()
        def wrong(phases, qs):
            out = f['reconstruct_rpe_phase'](phases, qs)
            out['rounds'][-1]['selected_branch'] ^= 1
            return out
        result = b.finite_prefix_check([toy_round(1), toy_round(2)], 0, decoder=wrong)
        self.assertEqual(result['decision'], 'COUNTEREXAMPLE_ON_SUPPLIED_PREFIX')

    def test_mean_conversion_paths_cannot_be_silently_identified(self):
        self.assertNotEqual(b.represented_mean(1, 3, 'count_grid'), b.represented_mean(1, 3, 'integer_sum'))
        self.assertEqual(Q(2, 3)-1, Q(-1, 3))

    def test_origin_realization_stays_unknown(self):
        row = dict(toy_round(1), mu_cosine='0', eps_cosine='3/5')
        self.assertEqual(b.finite_prefix_check([row], 0)['reason'], 'zero-signal realization')

    def test_principal_cut_tie_and_nonfinite_source_input(self):
        self.assertGreaterEqual(b.distance_interval(arb.pi(), 0).upper(), arb.pi())
        with self.assertRaisesRegex(ValueError, 'UNKNOWN/STOP'):
            b.represented_mean(1, 3, 'unfixed')
        with self.assertRaises(ValueError):
            D.branch_conditions([float('inf')], [1])
        self.assertEqual(D.branch_conditions(['1/10']*2, [1, 2])['implementation_decision'], 'UNKNOWN/STOP')

    def test_exact_oracle_tie_is_not_resolved_by_argmin(self):
        rows = [dict(toy_round(q), eps_cosine='1/4', eps_sine='1/4') for q in (1, 2)]
        self.assertEqual(b.finite_prefix_check(rows, arb.pi()/2)['reason'], 'interval oracle tie')

    def test_subnormal_and_signed_zero_cases_do_not_create_uniform_proof(self):
        smallest = float.fromhex('0x0.0000000000001p-1022')
        for z in (complex(-1, smallest), complex(-1, -smallest), complex(0.0, -0.0)):
            self.assertTrue(np.isfinite(np.angle(z)))
        self.assertIsNone(D.branch_conditions(['.1'], [1])['implementation_final_phase_upper'])


def synthetic_reference_inputs():
    # Explicit artificial constant Hamiltonian in a generic complete number
    # sector. This creates no molecular data, and no saved H4 data is read.
    hbuf, sbuf = io.BytesIO(), io.BytesIO()
    basis = np.asarray([x for x in range(256) if x.bit_count() == 4], dtype=np.int64)
    state = np.zeros(70, complex); state[0] = 1
    np.savez(hbuf, hamiltonian_hash='toy', constant=-1., one_body=np.zeros((8,8), complex),
             lambdas=np.zeros(12), g_matrices=np.zeros((12,8,8), complex))
    np.savez(sbuf, hamiltonian_hash='toy', fingerprint='toy-state', n_qubits=8,
             n_electrons=4, basis_indices=basis, state=state)
    manifest = dict(hamiltonian_hash='toy', state_fingerprint='toy-state', reference_lower_offset='.001',
                    precision_bits={'A':160,'C':256}, files={},
                    constants=dict(energy_reference_binary64_decimal='-1', target_energy_precision_ha='.01'))
    prior = dict(energy=dict(literal_energy_upper=dict(upper='.001'), boundary_margin=dict(lower='3')),
                 delta_exact='1/8')
    return manifest, dict(H=hbuf.getvalue(), STATE=sbuf.getvalue()), prior


class RunnerTests(ArbTests):
    def test_artificial_C_integration_includes_reference_once(self):
        manifest, raw, prior = synthetic_reference_inputs()
        out = run.reference_task(manifest, raw, prior)
        self.assertEqual(out['decision'], 'REPRESENTED_DF_SECTOR_GROUND_BRACKET')
        self.assertEqual(out['ideal_saved_DF_target_decision'], 'SUFFICIENT')
        self.assertIsNone(out['implementation_energy_upper'])
        self.assertEqual(out['real_molecular_energy_decision'], 'UNKNOWN/STOP')

    def test_A_unresolved_does_not_block_independent_C_or_trigger_retry(self):
        manifest, raw, prior = synthetic_reference_inputs()
        axes = [dict(m='0',axis='cosine',N='32',epsilon='.25',U_cat_upper='.25',alpha='.01')]
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(run, 'frozen_source', return_value={'commit':'artificial'}), \
             patch.object(run, 'metadata', return_value=(manifest,raw,axes,prior)):
            out = run.execute(['A','C'], Path(temporary)/'result')
        self.assertEqual(out['A']['decision'], 'UNKNOWN/STOP')
        self.assertEqual(out['C']['decision'], 'REPRESENTED_DF_SECTOR_GROUND_BRACKET')
        self.assertFalse(out['RPE_tolerance_certified'])

    def test_default_mode_never_enters_science(self):
        with patch.object(run, 'metadata', return_value=({'precision_bits':{'A':160,'C':256}}, {}, [{}]*36, {})), \
             patch.object(run, 'alpha_task', side_effect=AssertionError('science forbidden')), \
             patch.object(run, 'reference_task', side_effect=AssertionError('science forbidden')), \
             patch('sys.argv', ['run_three_gaps.py']), contextlib.redirect_stdout(io.StringIO()) as out:
            run.main()
        self.assertEqual(json.loads(out.getvalue())['new_H4_scientific_calls'], 0)

    def test_no_H4_B_entry_and_no_existing_output_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, 'UNKNOWN/STOP'):
                run.execute(['A'], Path(temporary))
            with self.assertRaisesRegex(ValueError, 'UNKNOWN/STOP'):
                run.execute(['B'], Path(temporary)/'new')


if __name__ == '__main__':
    unittest.main()
