"""Synthetic-only H8 A/B scope, exact reconstruction and native-event checks."""
import itertools
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import h8_ab_contract as c
import h8_ab_sampling as sampling


class Contract(unittest.TestCase):
    def test_targeted_registry_and_stage_boundaries(self):
        p = c.plan(); cells = p['A']['cells']
        self.assertEqual(len(cells), 16)
        self.assertEqual(len({x['id'] for x in cells}), 16)
        self.assertTrue(all(x['R'] == x['q'] * x['r'] for x in cells))
        self.assertEqual([(x['prefix'], x['q'], x['r'], x['K']) for x in cells if x['q'] == 16], [(7, 16, 4, 4)])
        self.assertEqual(len(p['B']['groups']), 4)
        self.assertEqual(p['B']['compile'], 64)
        self.assertEqual(p['B']['control_probe'], 16)
        self.assertEqual(p['A']['compile'], 0)
        self.assertFalse(p['A']['automatic_cost_stage'])
        self.assertFalse(p['cost_reaccounting'])

    def test_other_system_or_stage_grant_rejected(self):
        for stage, grant in [('B', dict(scope='H6_DBA_20261018', stage='B')),
                             ('A', dict(scope=c.SCOPE, stage='B'))]:
            with self.assertRaisesRegex(ValueError, 'EXACT_H8_AB_SCOPE'): c.verify({}, grant, stage)

    def test_unapproved_refused_before_input(self):
        with self.assertRaisesRegex(ValueError, 'EXPLICIT_SCOPED'):
            c.verify({}, dict(scope=c.SCOPE, stage='A', approved_by_user=False), 'A')

    def test_other_stage_config_cannot_use_valid_approval(self):
        grant = dict(scope=c.SCOPE, stage='A', approved_by_user=True, one_shot=True,
                     retry=False, next_stage_authorized=False)
        with self.assertRaisesRegex(ValueError, 'STAGE_SPECIFIC_CONFIG'):
            c.verify(dict(stage='B'), grant, 'A')

    def test_real_CLI_invalid_grant_rejected_before_output_or_science(self):
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); config = root / 'config'; grant = root / 'grant'
            c.save(config, {}); c.save(grant, dict(scope=c.SCOPE, stage='A', approved_by_user=False))
            result = subprocess.run([sys.executable, '-B', str(c.ROOT / 'scripts/run_h8_ab.py'),
                '--stage', 'A', '--config', str(config), '--grant', str(grant)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('EXPLICIT_SCOPED_EXECUTION_APPROVAL_REQUIRED', result.stderr)
            self.assertEqual(sorted(p.name for p in root.iterdir()), ['config', 'grant'])

    def test_seed_registration_and_collision(self):
        from h8_ab_science import tasks
        config = dict(old_seed_values=[1, 2, 3], binding=dict(input='synthetic'))
        values = tasks(config)
        self.assertEqual(len(values), 32)
        all_seeds = [s for row in values for s in [row['seed'], *row['derived_seeds']]]
        self.assertEqual(len(all_seeds), len(set(all_seeds)))
        self.assertEqual(values, tasks(config))
        config['old_seed_values'].append(values[0]['seed'])
        with self.assertRaisesRegex(ValueError, 'COLLISION'): tasks(config)

    def test_before_call_budget_is_exclusive(self):
        from h8_ab_science import Budget
        with tempfile.TemporaryDirectory() as directory:
            b = Budget(c.Writer(directory), dict(compile=2))
            b.take('compile'); b.take('compile')
            self.assertEqual(len(list(Path(directory).iterdir())), 2)
            with self.assertRaisesRegex(ValueError, 'CALL_BUDGET'): b.take('compile')

    def test_full_saved_physical_time_coverage_required(self):
        from h8_ab_signal import saved_primitive_coverage
        with tempfile.TemporaryDirectory() as directory:
            mapping = {}
            for q, time in ((2, .2), (4, .1), (8, .05)):
                path = Path(directory) / str(q)
                c.save(path, dict(all_registered_times=True, probe_count=3, records=[
                    dict(block='one' if i == 0 else str(i - 1), time=t,
                         native_spin_errors=[1e-14]*3, spin_CSR_errors=[1e-14]*3)
                    for i in range(9) for t in (time, -time)]))
                mapping['primitive_q' + str(q)] = str(path)
            self.assertEqual(saved_primitive_coverage(dict(mapping=mapping))['block_time_pairs'], 54)
            value = c.read(mapping['primitive_q8']); value['records'].pop()
            Path(mapping['primitive_q8']).write_text(__import__('json').dumps(value))
            with self.assertRaisesRegex(ValueError, 'COVERAGE'): saved_primitive_coverage(dict(mapping=mapping))


class Counts(unittest.TestCase):
    def test_multinomial_matches_all_order_sequences(self):
        for stratum in ('s2_multi', 's4'):
            law = sampling.cached_law(4, (.6, .3, .1), stratum)
            expected = {}
            for orders in itertools.product((0, 2, 4), repeat=4):
                key = orders.count(2), orders.count(4)
                if sampling.counts.in_stratum(*key, stratum):
                    expected[key] = expected.get(key, 0.) + math.prod({0:.6, 2:.3, 4:.1}[x] for x in orders)
            mass = math.fsum(expected.values())
            self.assertAlmostEqual(math.exp(law.log_stratum_mass), mass, places=14)
            for key, prob in zip(law.counts, law.probabilities):
                self.assertAlmostEqual(prob, expected[key] / mass, places=14)
            self.assertIs(law, sampling.cached_law(4, (.6, .3, .1), stratum))

    def test_all_supported_count_vectors_retained(self):
        law = sampling.cached_law(8, (.6, .3, .1), 's4')
        self.assertIn((1, 2), law.counts)
        multi = sampling.cached_law(8, (.6, .3, .1), 's2_multi')
        self.assertGreater(max(n2 for n2, n4 in multi.counts), 2)
        self.assertTrue(all(n2 >= 2 and n4 == 0 for n2, n4 in multi.counts))


class NativeFixture(unittest.TestCase):
    @staticmethod
    def fixture():
        import numpy as np
        from trotterlib.df_hamiltonian import DFHamiltonian
        return DFHamiltonian(.17, np.array([[.2, .04], [.04, -.3]]), np.array([.4, -.2]),
            (np.array([[.6, .1], [.1, -.2]]), np.array([[.3, .07], [.07, .5]])), {})

    def test_exact_preparation_reconstruction_hash_and_export(self):
        import numpy as np
        from trotterlib.pr2_matched_accuracy_m1_execution import _prepare
        from trotterlib.df_partial_s2 import df_hamiltonian_hash
        from h8_ab_native import prefix_preparation, preparation_export
        ham = self.fixture(); whole = _prepare(ham, 2); expected = _prepare(ham, 1)
        observed = prefix_preparation(ham, whole.deterministic_blocks, 1,
                                      SimpleNamespace(np=np, df_hamiltonian_hash=df_hamiltonian_hash))
        self.assertEqual(observed.preparation_hash, expected.preparation_hash)
        self.assertEqual(preparation_export(observed), preparation_export(expected))
        self.assertEqual(observed.extracted_identity_coefficient, expected.extracted_identity_coefficient)

    def test_native_K4_multi_s4_control_matches_full_wrappers(self):
        import numpy as np
        from trotterlib.pr2_matched_accuracy_m1_execution import _prepare
        from trottertracks.resource_applicability import h6_matched_port_v1 as m
        from trottertracks.resource_applicability import ax2b_native_df_v5 as n
        from trottertracks.resource_applicability.h8_control_reuse_v1 import validate_control
        from h6_dba_science import event_oracle
        prep = _prepare(self.fixture(), 1); state = np.array([0, 1, 0, 0], dtype=complex)
        for stratum in ('s2_multi', 's4'):
            steps, receipt = sampling.steps(prep, c.cell(1, 2, 2, 4), stratum, 5678)
            self.assertEqual(receipt, sampling.steps(prep, c.cell(1, 2, 2, 4), stratum, 5678)[1])
            self.assertFalse(receipt['quantum_weights_changed'])
            self.assertFalse(receipt['normalization_changed'])
            evolution = n.partial_native_from_step_requests(steps, control_policy='symmetric_directional', max_instructions=100000)
            budget = SimpleNamespace(used=0)
            def take(name, count=1): budget.used += count
            budget.take = take
            evidence, high = validate_control(evolution, state, calls=budget,
                simulate=m.simulate_statevector, expectation=m.hadamard_expectation)
            self.assertEqual(budget.used, 4)
            self.assertLess(evidence['max_error'], 1e-12)
            z = complex(**dict(real=evidence['signal']['real'], imag=evidence['signal']['imag']))
            self.assertLess(abs(z - event_oracle(steps, m, n, state)), 1e-12)
            low = np.r_[state, np.zeros_like(state)]
            for axis, value in (('cosine', z.real), ('sine', z.imag)):
                wrapper = n.build_native_hadamard_wrapper(evolution, axis=axis, include_measurement=False, max_instructions=100000)
                self.assertLess(abs(m.hadamard_expectation(m.simulate_statevector(wrapper, low), 2) - value), 1e-12)

    def test_four_mode_spin_CSR_forward_and_production_horner(self):
        import numpy as np
        from trotterlib.df_hamiltonian import DFHamiltonian, PhysicalSector, df_linear_operator
        from trotterlib.pr2_matched_accuracy_m1_execution import _prepare
        from trottertracks.resource_applicability.h8_sparse_reference_v1 import OccupationCSR, SpinPrimitive, df_operator
        from trottertracks.resource_applicability.h6_matched_port_v1 import compact_signal
        from trottertracks.resource_applicability.ax2a_state_action import ActionBudget
        from h6_dba_signal import forward_finite
        toy = self.fixture()
        ham = DFHamiltonian(toy.constant, np.kron(toy.one_body, np.eye(2)), toy.lambdas,
                            tuple(np.kron(g, np.eye(2)) for g in toy.g_matrices), {})
        sector = PhysicalSector.spin_sector(n_qubits=4, nelec_alpha=1, nelec_beta=1)
        state = np.array([1, 1j, .3, -.2j], dtype=complex); state /= np.linalg.norm(state)
        prep = _prepare(ham, 1); lam = prep.exact_rte_lambda_r
        topology = OccupationCSR(sector.basis_indices, 4)
        budget = SimpleNamespace(take=lambda *args: None)
        oracle, trace = df_operator(-prep.extracted_identity_coefficient / lam, topology.one(ham.one_body) * 0,
                                    ham.lambdas[1:] / lam, (topology.one(ham.g_matrices[1]),), budget)
        production = df_linear_operator(DFHamiltonian(-prep.extracted_identity_coefficient,
            np.zeros_like(ham.one_body), ham.lambdas[1:], ham.g_matrices[1:], {}), sector, num_threads=1)[0]
        actions = [SpinPrimitive(ham.one_body, sector), SpinPrimitive(ham.g_matrices[0], sector, weight=ham.lambdas[0])]
        for K in (2, 4):
            cell = c.cell(1, 2, 3, K); scalar = prep.constant_coefficient + prep.extracted_identity_coefficient
            actual = compact_signal(state, actions, lambda v: (production @ v) / lam,
                cell=cell, T=.8, scalar=scalar, lambda_r=lam, budget=ActionBudget(1000, 1000))
            expected = forward_finite(state, actions, oracle, cell, scalar, lam, np)
            for path in ('corrected', 'raw'):
                self.assertLess(np.linalg.norm(actual['vectors'][path] - expected['vectors'][path]), 1e-12)


if __name__ == '__main__': unittest.main()
