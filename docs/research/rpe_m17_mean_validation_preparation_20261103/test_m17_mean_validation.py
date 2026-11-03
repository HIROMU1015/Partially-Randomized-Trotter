"""Synthetic scalars/2x2 operators or literal metadata only; never H4 science."""
from contextlib import redirect_stdout
from decimal import Decimal
import io
import itertools
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import run_m17_mean_validation as runner

sys.path.insert(0, str(runner.ROOT / "src/trotterlib"))
import rte
sys.path.insert(0, str(runner.ROOT / "docs/research/rpe_mean_tolerance_validation_selection_20261103"))
from test_design_identities import functions

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.diag([1., -1.]).astype(complex)
H = np.array([[1., 1.], [1., -1.]])/math.sqrt(2.)


def toy_literals():
    return {"axes": {a: {"N": "100", "alpha": "0.05", "epsilon": "0.5"}
                     for a in ("cosine", "sine")}, "saved_mean": {"real": "0", "imag": "0"}}


def toy_context():
    tail = rte.normalize_involutory_tail("synthetic-m17-preparation-only", (
        rte.InvolutoryTailTerm("x", .7, X), rte.InvolutoryTailTerm("z", -.3, Z)))
    config, distribution = rte.make_rte_config(tail, evolution_time=.23, rte_steps=2,
                    finite_taylor_order=0, truncation_tolerance=1.)
    f = math.cos(.31)*I-1j*math.sin(.31)*X
    reverse = math.cos(.19)*I-1j*math.sin(.19)*Z
    context = {"psi": np.array([1., 1j])/math.sqrt(2.), "forward": f, "reverse": reverse,
               "A_probability": tail.normalized_hamiltonian, "A_coefficient": tail.normalized_hamiltonian,
               "distribution": distribution,
               "reference_tail": rte.finite_rte_operator_moments(tail.normalized_hamiltonian, config).attenuated_event_mean_operator}
    return context, tail


class MeanPreparationTests(unittest.TestCase):
    def test_grouping_keeps_weight_sources_and_shares_basis_transform(self):
        probability = [SimpleNamespace(component_id=name, basis_id=name, probability=p,
            coefficient_sign=sign, diagonal_pauli_support=(0,))
            for name, p, sign in (("x", .7, 1), ("z", .3, -1))]
        coefficient = [SimpleNamespace(component_id=name, basis_id=name, coefficient=c,
            diagonal_pauli_support=(0,)) for name, c in (("x", .7), ("z", -.3))]
        basis = Mock(side_effect=lambda name: {"x": H, "z": I}[name])
        ap, ac, total = runner.grouped_averages(probability, coefficient, 1., np.array([0, 1]), basis)
        np.testing.assert_allclose(ap, .7*X-.3*Z, atol=1e-14)
        np.testing.assert_allclose(ac, ap, atol=1e-14)
        self.assertEqual(basis.call_count, 2)
        probability[0].probability, probability[1].probability = .6, .4
        changed, original, _ = runner.grouped_averages(probability, coefficient, 1., np.array([0, 1]), basis)
        self.assertGreater(np.linalg.norm(changed-original), .1)

    def test_literal_law_matches_full_small_support_including_order2_and_negative_time(self):
        _, tail = toy_context()
        operators = {c.component_id: op for c, op in zip(tail.components, tail.operators)}
        for tau in (.17, -.17):
            distribution = rte.finite_rte_distribution(tau, 2)
            events = rte.enumerate_rte_events(tail.components, distribution)
            expected = rte.exact_enumerated_event_mean_operator(events, operators)
            actual = runner.literal_law_mean(tail.normalized_hamiltonian, distribution)
            np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=0)

    def test_r_lives_inside_q_and_scalar_normalization_is_applied_once(self):
        context, tail = toy_context()
        operators = {c.component_id: op for c, op in zip(tail.components, tail.operators)}
        events = rte.enumerate_rte_events(tail.components, context["distribution"])
        legacy = functions("src/trotterlib/rpe_hadamard_failure_validation.py", ("_apply_event_to_state", "_trajectory_signal"))
        expected = 0j
        for trajectory in itertools.product(events, repeat=4):
            expected += math.prod(e.event_probability for e in trajectory)*legacy['_trajectory_signal'](
                initial_state=context['psi'], forward=context['forward'], reverse=context['reverse'],
                events=trajectory, operators=operators, q_m=2, rte_steps=2)
        evaluation, arrays = runner.evaluate_context(context, expected, q=2, r=2)
        self.assertEqual(evaluation['numeric_diagnostic_status'], 'NUMERICALLY_CONSISTENT')
        self.assertEqual(evaluation['scientific_status'], 'UNDETERMINED')
        self.assertLess(evaluation['comparisons']['law_vs_saved']['complex_abs'], 1e-13)
        self.assertEqual(len(arrays), 7)

    def test_a_reference_mismatch_is_reported_without_tolerance_inflation(self):
        context, _ = toy_context()
        context['reference_tail'] *= .9
        evaluation, _ = runner.evaluate_context(context, 0j, q=2, r=2)
        self.assertEqual(evaluation['numeric_diagnostic_status'], 'INCONSISTENT')
        self.assertEqual(evaluation['diagnostic_atol'], 1e-10)

    def test_scalar_reserve_uses_saved_n_alpha_epsilon_on_synthetic_values(self):
        raw = toy_literals()['axes']
        reserve = runner.statistical_reserve(raw)
        expected = .5-math.sqrt(2*math.log(2/.05)/100)
        self.assertAlmostEqual(float(reserve['minimum_reserve']), expected, places=14)
        self.assertEqual(reserve['axes']['cosine']['saved'], raw['cosine'])
        comparisons = {'law_vs_saved': {'cosine_abs': .01, 'sine_abs': .02}}
        interpreted = runner.reserve_interpretation(reserve, comparisons)
        self.assertTrue(interpreted['reserve_positive'])
        self.assertIsNone(interpreted['u_num_bound'])
        self.assertFalse(interpreted['RPE_tolerance_certified'])

    def test_invalid_scalar_inputs_and_nonpositive_reserve_do_not_become_passes(self):
        for name, value in (('N', '0'), ('N', '1.5'), ('alpha', '0'), ('alpha', '1'), ('epsilon', 'NaN')):
            raw = toy_literals()['axes']
            raw['cosine'][name] = value
            with self.assertRaises(ValueError): runner.statistical_reserve(raw)
        reserve = {'axes': {a: {'reserve': '-0.1'} for a in ('cosine','sine')}, 'minimum_reserve': '-0.1'}
        interpreted = runner.reserve_interpretation(reserve, {'law_vs_saved': {'cosine_abs': 0., 'sine_abs': 0.}})
        self.assertEqual(interpreted['status'], 'UNDETERMINED_NONPOSITIVE_RESERVE')

    def test_default_cli_cannot_enter_science_or_open_outputs(self):
        with patch.object(runner, 'fixed_literals', return_value=toy_literals()), \
             patch.object(runner, 'statistical_reserve', side_effect=AssertionError('science')), \
             patch.object(runner, 'prepare_context', side_effect=AssertionError('science')), \
             patch.object(runner, 'load_numerical_api', side_effect=AssertionError('molecular import')), \
             patch.object(runner.np, 'load', side_effect=AssertionError('NPZ decode')), \
             patch.object(runner.np.random, 'default_rng', side_effect=AssertionError('RNG')), \
             redirect_stdout(io.StringIO()) as stream:
            runner.main([])
        self.assertEqual(json.loads(stream.getvalue())['status'], 'PREPARED_NOT_SCIENTIFICALLY_EXECUTED')

    def test_actual_fixed_row_only_transfers_literals(self):
        row = runner.fixed_literals()
        self.assertEqual((row['m'], row['q'], row['r'], row['K']), ('17','131072','32','2'))
        self.assertEqual(row['axes']['cosine']['N'], '411')
        self.assertEqual(row['axes']['sine']['epsilon'], '0.15244325851620388')
        self.assertIsInstance(row['saved_mean']['real'], str)

    def context_probes(self, preparation_hash='expected'):
        class ReachedConfig(Exception): pass
        p = SimpleNamespace(H_PATH='synthetic-H', H_SHA='synthetic-H-hash', STATE_PATH='synthetic-state',
                            STATE_FP='synthetic-fingerprint', PREPARATION_SHA='expected',
                            source_and_input_identity=lambda: [])
        prep = SimpleNamespace(preparation_hash=preparation_hash,
            rte_preparation=SimpleNamespace(symbolic_tail=SimpleNamespace(components=tuple(object() for _ in range(218)))),
            deterministic_blocks=())
        make = Mock(return_value=prep)
        config = Mock(side_effect=ReachedConfig)
        api = {'load_connected_cluster_hamiltonian_snapshot': lambda path: SimpleNamespace(n_qubits=8,n_blocks=12),
               'df_hamiltonian_hash': lambda h: p.H_SHA,
               'PhysicalSector': SimpleNamespace(number_sector=lambda **kw: SimpleNamespace(basis_indices=np.array([0,1]))),
               '_array_fingerprint': lambda state:p.STATE_FP, 'prepare_df_partial_s2':make,
               'split_df_hamiltonian_by_ld':lambda h,ld:'synthetic-partition', '_finite_inputs':config,
               '_validate_components':lambda c:1.}
        fixture = {'state':np.array([1.+0j,0]),'basis_indices':np.array([0,1]),'hamiltonian_hash':np.array(p.H_SHA)}
        return p,api,fixture,make,config,ReachedConfig

    def test_context_passes_threshold_and_r32_without_r1_fallback(self):
        p,api,fixture,make,config,Reached = self.context_probes()
        provenance = {}
        with patch.object(runner,'prior_runner',return_value=p), \
             patch.object(runner.importlib.metadata,'version',return_value='1.3.0'), patch.object(runner.np,'load') as load:
            load.return_value.__enter__.return_value=fixture
            with self.assertRaises(Reached): runner.prepare_context(provenance,api=api)
        self.assertEqual(make.call_args.kwargs,{'identity_policy':'extract_identity_phase','coefficient_atol':1e-12})
        self.assertEqual(config.call_args.kwargs,{'delta_time':.02,'rte_steps':32,'finite_taylor_order':2})
        self.assertEqual(provenance['preparation']['component_count'],218)

    def test_context_inventory_failure_retains_actual_values(self):
        p,api,fixture,make,config,Reached=self.context_probes('synthetic-wrong-hash')
        provenance={}
        with patch.object(runner,'prior_runner',return_value=p), \
             patch.object(runner.importlib.metadata,'version',return_value='1.3.0'), patch.object(runner.np,'load') as load:
            load.return_value.__enter__.return_value=fixture
            with self.assertRaisesRegex(ValueError,'synthetic-wrong-hash'):runner.prepare_context(provenance,api=api)
        config.assert_not_called()
        self.assertEqual(provenance['preparation']['hash'],'synthetic-wrong-hash')

    def test_complete_context_schema_with_synthetic_components_and_cached_basis(self):
        p,api,fixture,make,config,Reached=self.context_probes()
        tail=rte.normalize_involutory_tail('synthetic-context-218',tuple(
            rte.InvolutoryTailTerm(str(i),1./218,Z,basis_id='synthetic-basis',diagonal_pauli_support=(0,))
            for i in range(218)))
        prep=make.return_value
        prep.rte_preparation.symbolic_tail=tail
        coefficients=[SimpleNamespace(component_id=c.component_id,basis_id=c.basis_id,
            coefficient=c.coefficient_abs*c.coefficient_sign,diagonal_pauli_support=c.diagonal_pauli_support)
            for c in tail.components]
        definition=SimpleNamespace(metadata=SimpleNamespace(basis_hash='synthetic-basis-hash'),runtime_operations=())
        prep.tail_extraction=SimpleNamespace(components=coefficients,tail_hash='synthetic-tail',
                                            basis_definition=lambda name:definition)
        prep.exact_rte_lambda_r=tail.lambda_r
        prep.constant_coefficient=prep.extracted_identity_coefficient=0.
        p.numeric_half_operators=functions(
            'docs/research/rpe_fresh_iid_measurement_a_preparation_20261103/run_fresh_iid_a_v2.py',
            ('numeric_half_operators',))['numeric_half_operators']
        p.restricted_basis=Mock(return_value=(I,0.))
        api['_finite_inputs']=lambda preparation,**kw:rte.make_rte_config(tail,
            evolution_time=kw['delta_time'],rte_steps=kw['rte_steps'],finite_taylor_order=kw['finite_taylor_order'],
            truncation_tolerance=1.)
        api.update({'_validate_components':rte._validate_components,
            '_qiskit_to_openfermion_sector_permutation':lambda sector:np.array([0,1]),
            'finite_rte_operator_moments':rte.finite_rte_operator_moments})
        provenance={}
        with patch.object(runner,'prior_runner',return_value=p), \
             patch.object(runner.importlib.metadata,'version',return_value='1.3.0'), patch.object(runner.np,'load') as load:
            load.return_value.__enter__.return_value=fixture
            context=runner.prepare_context(provenance,api=api)
        np.testing.assert_allclose(context['A_probability'],Z,atol=1e-13)
        np.testing.assert_allclose(context['A_coefficient'],Z,atol=1e-13)
        self.assertEqual(p.restricted_basis.call_count,1)
        self.assertEqual(provenance['config']['rte_steps'],32)
        self.assertEqual(context['reference_tail'].shape,(2,2))

    def test_interruption_retains_scalar_stage_and_error_without_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'synthetic-only'
            with patch.object(runner,'source_metadata',return_value={'commit':'synthetic'}), \
                 patch.object(runner,'fixed_literals',return_value=toy_literals()), \
                 patch.object(runner,'prepare_context',side_effect=RuntimeError(str(runner.ROOT)+'/synthetic interruption')):
                with self.assertRaises(RuntimeError):runner.execute(output)
                record=json.loads((output/'RESULT.json').read_text())
                self.assertEqual(record['execution_status'],'INTERRUPTED_OR_INVALID_MANDATORY_STOP')
                self.assertEqual(record['stage'],'numerical_context')
                self.assertIn('statistical_reserve',record)
                self.assertIn('<repository>',record['error']['message'])
                original=(output/'RESULT.json').read_bytes()
                with self.assertRaises(FileExistsError):runner.execute(output)
                self.assertEqual((output/'RESULT.json').read_bytes(),original)

    def test_synthetic_completion_saves_two_means_and_seven_operators_without_certification(self):
        context,_=toy_context()
        raw=toy_literals()
        unpatched=runner.evaluate_context
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'synthetic-only'
            with patch.object(runner,'source_metadata',return_value={'commit':'synthetic'}), \
                 patch.object(runner,'fixed_literals',return_value=raw), \
                 patch.object(runner,'prepare_context',return_value=context), \
                 patch.object(runner,'evaluate_context',side_effect=lambda c,z:unpatched(c,z,q=2,r=2)):
                runner.execute(output)
            record=json.loads((output/'RESULT.json').read_text())
            self.assertEqual(record['execution_status'],'COMPLETED_MANDATORY_STOP')
            self.assertEqual(record['completed_complex_mean_paths'],2)
            self.assertEqual(record['scientific_status'],'UNDETERMINED')
            self.assertIsNone(record['reserve_interpretation']['u_num_bound'])
            with np.load(output/'operators.npz',allow_pickle=False) as saved:self.assertEqual(len(saved.files),7)


if __name__=='__main__':
    unittest.main()
