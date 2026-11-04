"""Artificial scalar/2x2 and mock-only tests: no H8 inputs or real circuits."""
import copy
import ast
from contextlib import contextmanager
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import h8_long_time_adapter as a
import h8_long_time_runtime as rt


class MetadataTests(unittest.TestCase):
    def test_matched_q_is_one_explicit_eligible_S2_diagnostic_only(self):
        diag = a.candidate('PR', '4th(m5_best)', 8, 16, 32, 4, matched_q_diagnostic=True)
        s2 = a.candidate('PR', '2nd', 8, 16, 32, 4)
        config = dict(binding={'H': 'fixture'}, source_files={}, SOURCE='a' * 40)
        scope = dict(stage='A', T=1.6, epsilon=.0005, state_preparation_cost=0, binding=config['binding'],
                     workers=1, source_files={}, SOURCE=config['SOURCE'], candidates=[diag],
                     diagnostic_S2_endpoint=dict(candidate=s2, status='ELIGIBLE'))
        rt.validate_scope('A', scope, config)
        scope['diagnostic_S2_endpoint']['status'] = 'UNDETERMINED'
        with self.assertRaisesRegex(ValueError, 'ONE_SELECTED_ELIGIBLE_S2'):
            rt.validate_scope('A', scope, config)
        with self.assertRaises(ValueError):
            a.candidate('PR', '4th(m5_best)', 8, 16, 32, 4)

    def test_import_has_no_scientific_dependencies(self):
        program = '''
import importlib.abc,sys
sys.path.insert(0, 'scripts')
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in ('numpy','scipy','qiskit','trotterlib','trottertracks','h8_ab_native'):
   raise RuntimeError('scientific import: '+fullname)
sys.meta_path.insert(0,Block())
import h8_long_time_adapter,h8_long_time_runtime
'''
        subprocess.run([sys.executable, '-B', '-c', program], cwd=ROOT, check=True, capture_output=True)

    def test_signed_schedule_exact_and_outer_allocation(self):
        row = a.candidate('PR', '4th(m5_best)', 8, 2, 16, 4)
        schedule = a.schedule(row)
        self.assertEqual(schedule, a.core.schedule(row['formula'], 2, 'ABS_TIME_PER_OUTER', 16, T=1.6))
        self.assertEqual(len(schedule), 22)
        self.assertTrue(any(s['signed_time'] < 0 for s in schedule))
        for outer in range(2):
            steps = [s for s in schedule if s['outer'] == outer]
            self.assertEqual(sum(s['rte_steps'] for s in steps), 16)
            self.assertGreaterEqual(min(s['rte_steps'] for s in steps), 1)
            self.assertAlmostEqual(math.fsum(s['signed_time'] for s in steps), .8)

    def test_s2_R_is_per_outer(self):
        row = a.candidate('PR', '2nd', 7, 16, 4, 4)
        self.assertEqual(sum(s['rte_steps'] for s in a.schedule(row)), 64)

    def test_domain_and_candidate_identity(self):
        for args in [('PR', '4th', 7, 2, 16, 4), ('PR', '2nd', 7, 1, 4, 4),
                     ('PR', '4th(m5_best)', 8, 2, 4, 4), ('B0', '2nd', 7, 8, None, None)]:
            with self.assertRaises(ValueError):
                a.candidate(*args)
        row = a.candidate('FULL', '4th', 28, 2)
        row['T'] = .8
        with self.assertRaisesRegex(ValueError, 'NEW_TIME'):
            a.validate_candidate(row)
        row = a.candidate('FULL', '4th', 28, 2)
        row['q'] = 4
        with self.assertRaisesRegex(ValueError, 'CANDIDATE_ID'):
            a.validate_candidate(row)

    def test_fragment_split_preserves_one_body_and_order(self):
        blocks = [NS(original_fragment_index=None)] + [NS(original_fragment_index=i) for i in range(28)]
        for p in (7, 8, 15, 16, 28):
            prefix, tail = a.split_blocks(blocks, p)
            self.assertIs(prefix[0], blocks[0])
            self.assertEqual(tuple(b.original_fragment_index for b in tail), tuple(range(p, 28)))
        blocks[2], blocks[3] = blocks[3], blocks[2]
        with self.assertRaisesRegex(ValueError, 'FRAGMENT_ORDER'):
            a.split_blocks(blocks, 7)

    def test_reference_rejects_old_time_wrong_identity_and_surrogate(self):
        ref = dict(T=1.6, kind='NEW_FULL_H_SPARSE_REFERENCE', binding={'H': 'fixture'},
                   signal=dict(real=.2, imag=.3), certified=False, phase_alignment=False)
        self.assertEqual(a.check_reference(ref, {'H': 'fixture'}), .2 + .3j)
        for changed in [dict(T=.8), dict(binding={'H': 'other'}), dict(kind='SQUARED_OLD_SIGNAL'), dict(phase_alignment=True)]:
            with self.assertRaises(ValueError):
                a.check_reference(dict(ref, **changed), {'H': 'fixture'})

    def test_input_hash_mismatch_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'fixture'; path.write_text('artificial')
            row = dict(path=str(path), role='fixture', bytes=path.stat().st_size, sha256=a.sha(path))
            a.check_files([row])
            path.write_text('alteration')
            with self.assertRaisesRegex(ValueError, 'INPUT_SOURCE_BYTES'):
                a.check_files([row])

    def test_stage_scope_requires_source_and_keeps_confirmation_separate(self):
        config = dict(binding={'H': 'fixture'}, source_files={'adapter': 'fixture'}, SOURCE='a' * 40)
        scope = dict(stage='A', T=1.6, epsilon=.0005, state_preparation_cost=0,
                     binding=config['binding'], workers=1, source_files=config['source_files'], SOURCE=config['SOURCE'],
                     candidates=[a.candidate('FULL', '2nd', 28, 1)])
        rt.validate_scope('A', scope, config)
        with self.assertRaisesRegex(ValueError, 'SOURCE_COMMIT'):
            rt.validate_scope('A', dict(scope, SOURCE='b' * 40), config)
        with self.assertRaisesRegex(ValueError, 'STAGE_COHORT'):
            rt.validate_scope('C', dict(scope, stage='C', n=9, cohort='EXPLORATION'), config)

    def test_exclusive_evidence_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'evidence.json'
            a.save(path, {'fixture': True})
            with self.assertRaises(FileExistsError):
                a.save(path, {'fixture': False})


class ArtificialNumerics(unittest.TestCase):
    def test_finite_cutoff_K_has_degree_K_plus_one_signed(self):
        for K in (2, 4):
            for t in (.17, -.23):
                x = -.4 + .2j
                target = sum((-1j * t * .7)**d / math.factorial(d) for d in range(K + 2)) * x
                for horner in (False, True):
                    got = a.finite_piece(x, lambda v: .7 * v, t, K, 1.1, raw=False, horner=horner)
                    self.assertAlmostEqual(abs(got - target), 0., places=14)
                    raw = a.finite_piece(x, lambda v: .7 * v, t, K, 1.1, raw=True, horner=horner)
                    self.assertAlmostEqual(abs(raw * 1.1 - got), 0., places=14)

    def test_scalar_policy(self):
        self.assertEqual(a.effective_scalar('B0', 2., .3), 2.)
        self.assertEqual(a.effective_scalar('FULL', 2., .3), 2.)
        self.assertEqual(a.effective_scalar('B0I', 2., .3), 2.3)
        self.assertEqual(a.effective_scalar('PR', 2., .3), 2.3)

    def test_runtime_evolve_artificial_2x2_all_paths(self):
        import numpy as np
        def law(tau, K):
            return NS(exact_finite_distribution=sum(abs(tau)**k / math.factorial(k) * math.hypot(1., tau / (k + 1))
                                                  for k in range(0, K + 1, 2)))
        class Action:
            phases = {}
            def __call__(self, v, t):
                return np.exp(-1j * np.array([.2, -.3]) * t) * v
        state = np.array([1., 1j]) / math.sqrt(2)
        prep = NS(exact_rte_lambda_r=.4, rte_preparation=NS(symbolic_tail=NS(lambda_r=.4)))
        ctx = NS(np=np, state=state, actions=[Action()] * 9, preps={7: prep, 8: prep},
                 tails={7: (np.diag([.1, -.2]), 0.), 8: (np.diag([.1, -.2]), 0.)}, scalar=lambda row: .3)
        ctx.sparse = NS(exponential_action=lambda op, tr, v, t: np.exp(-1j * t * np.diag(op)) * v)
        for formula, q, R in [('2nd', 8, 4), ('4th(m5_best)', 2, 16)]:
            row = a.candidate('PR', formula, 7, q, R, 4)
            with patch.dict(sys.modules, {'trotterlib.rte': NS(finite_rte_distribution=law)}):
                corrected, _, logb = rt.Context.evolve(ctx, row, path='corrected')
                raw, _, _ = rt.Context.evolve(ctx, row, path='raw')
                horner, _, _ = rt.Context.evolve(ctx, row, path='corrected', horner=True)
                exact, _, _ = rt.Context.evolve(ctx, row, path='exact_tail')
            target = state * np.exp(-1j * 1.6 * (8 * np.array([.2, -.3]) + np.array([.1, -.2]) + .3))
            np.testing.assert_allclose(exact, target, atol=2e-14, rtol=0)
            np.testing.assert_allclose(raw, corrected * math.exp(-logb), atol=2e-14, rtol=0)
            np.testing.assert_allclose(horner, corrected, atol=2e-14, rtol=0)

    def test_arbitrary_T_trajectory_scalar_once_with_mock_actions(self):
        import numpy as np
        class Native:
            @staticmethod
            def block_instruction_bound(block, mode):
                assert mode == 'UNCONTROLLED'
                return 23
            @staticmethod
            def make_native_block_action(block, **kw):
                assert type(kw['max_instructions']) is int and kw['max_instructions'] == 23
                return lambda v, t: v * np.exp(-1j * block * t)
        row = a.candidate('PR', '4th(m5_best)', 7, 2, 16, 4)
        prep = NS(deterministic_blocks=(.2,), rte_preparation=NS(basis_registry='mock'))
        ctx = NS(np=np, qstate=np.array([1., 0.], complex), preps={7: prep}, native=Native(), scalar=lambda row: .3)
        requests = [NS(rte_occurrence=NS(events=())) for s in a.schedule(row)]
        module = 'trottertracks.resource_applicability.ax2b_molecular_ports_v3'
        with patch.dict(sys.modules, {module: NS(explicit_event_action=lambda v, event, registry: v)}):
            v = rt.independent_trajectory(ctx, row, requests)
        np.testing.assert_allclose(v, ctx.qstate * np.exp(-1j * (.2 + .3) * 1.6), atol=1e-14)


class PairedAccounting(unittest.TestCase):
    def test_confirmation_precision_failure_preserves_unresolved_no_topup(self):
        summaries = dict(pr=dict(cohort='CONFIRMATION', mean_G_exact='80', sample_SE_G=3.),
                         pf=dict(cohort='CONFIRMATION', mean_G_exact='100', sample_SE_G=None))
        value = a.confirmation_comparison(summaries, dict(PR='pr', PF='pf'), 'PR_LOWER', 1.)
        self.assertTrue(value['matches_fixed_direction'])
        self.assertFalse(value['descriptive_precision_met'])
        self.assertFalse(value['population_guarantee'])
        self.assertFalse(value['automatic_top_up'])
        summaries['pr']['cohort'] = 'EXPLORATION'
        with self.assertRaisesRegex(ValueError, 'CONFIRMATION_ONLY'):
            a.confirmation_comparison(summaries, dict(PR='pr', PF='pf'), 'PR_LOWER', 1.)

    def fixture(self):
        row = a.candidate('PR', '2nd', 7, 8, 4, 4)
        ep = dict(status='ELIGIBLE', N_cos=2, N_sin=5, candidate=row)
        rows = [dict(status='PASS', cohort='EXPLORATION', conditional=False, candidate_id=row['id'], index=i,
                     endpoint_digest=a.digest(ep), prices=dict(cosine=dict(rz_count=x), sine=dict(rz_count=y)))
                for i, (x, y) in enumerate([(3, 8), (4, 7), (7, 5)])]
        return rows, ep

    def test_weighted_G_includes_paired_covariance(self):
        rows, ep = self.fixture()
        stats = a.paired_G(rows, ep, 'EXPLORATION')
        self.assertEqual(Fraction(stats['mean_G_exact']), Fraction(46 + 43 + 39, 3))
        expected = sum((Fraction(x) - Fraction(128, 3))**2 for x in (46, 43, 39)) / 2
        self.assertEqual(Fraction(stats['sample_variance_G_exact']), expected)
        self.assertFalse(stats['pooled'])
        self.assertEqual(a.confirmation_n(float(expected), .5), math.ceil(float(expected) / .25))

    def test_pool_conditional_duplicate_candidate_endpoint_rejected(self):
        rows, ep = self.fixture()
        changes = [dict(cohort='CONFIRMATION'), dict(conditional=True), dict(index=1),
                   dict(candidate_id='wrong'), dict(endpoint_digest='wrong')]
        for change in changes:
            bad = copy.deepcopy(rows); bad[0].update(change)
            with self.assertRaises(ValueError):
                a.paired_G(bad, ep, 'EXPLORATION')

    def test_missing_price_not_zero_and_single_PF_has_no_fake_SE(self):
        rows, ep = self.fixture()
        self.assertIsNone(a.paired_G(rows[:1], ep, 'EXPLORATION')['sample_SE_G'])
        rows[0]['prices']['cosine']['rz_count'] = 0
        with self.assertRaisesRegex(ValueError, 'MEASURED_POSITIVE'):
            a.paired_G(rows, ep, 'EXPLORATION')
        with self.assertRaisesRegex(ValueError, 'ELIGIBLE_MEASURED'):
            a.paired_G([], ep, 'EXPLORATION')


class MockNativeComposition(unittest.TestCase):
    def test_price_job_full_wrapper_pair_with_mock_sampler_compiler(self):
        import numpy as np
        import h46_long_time_native as native
        row = a.candidate('PR', '2nd', 7, 8, 4, 4)
        endpoint = dict(candidate=row, status='ELIGIBLE', N_cos=2, N_sin=5, log_B=0., log_B_margin=1e-12)
        z = .6 + .8j
        class Circuit:
            data = ('mock',)
            def __init__(self, kind, axis=None): self.kind, self.axis = kind, axis
            def remove_final_measurements(self, **kwargs): return self
        def simulate(qc, vector):
            if qc.kind == 'evolution':
                return vector * np.array([1, 1, z, z])
            value = z.real if qc.axis == 'cosine' else z.imag
            return np.array([math.sqrt((1 + value) / 2), 0., math.sqrt((1 - value) / 2), 0.], complex)
        @contextmanager
        def reuse():
            yield NS(fingerprint=lambda qc: 'mock-' + str(qc.axis))
        wrapped = []
        def wrapper(evo, **kwargs):
            wrapped.append(kwargs)
            return Circuit('wrapper', kwargs['axis'])
        prep = NS(preparation_hash='mock-preparation')
        requests = [NS(rte_occurrence=NS(events=()), rte_distribution=NS(exact_finite_distribution=1.)) for _ in a.schedule(row)]
        ctx = NS(np=np, qstate=np.array([1., 0.], complex), preps={7: prep}, compiler='mock',
                 scalar=lambda row: .3, native=NS(build_native_hadamard_wrapper=wrapper))
        def compile_mock(wrapper, fp, representative):
            value = 10 if wrapper.axis == 'cosine' else 20
            return dict(metrics=dict(rz_count=value), compiler=a.COMPILER), wrapper, False
        names = {'trottertracks.resource_applicability.h6_hash_reuse_v1': NS(use_fingerprint_reuse=reuse),
                 'trotterlib.df_trotter.circuit': NS(simulate_statevector=simulate)}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, names), patch.object(rt, 'CONTEXT', ctx), \
             patch.object(native, 'sample_requests', return_value=(requests, ['artificial-seed'], [])) as sampling, \
             patch.object(native, 'compose', return_value=NS(circuit=Circuit('evolution'))), \
             patch.object(native, 'compile_cached', side_effect=compile_mock) as compiling, \
             patch.object(rt, 'independent_trajectory', return_value=z * ctx.qstate):
            result = rt.price_job((row, endpoint, 'EXPLORATION', 0, str(Path(tmp) / 'new-trial')))
        self.assertEqual(sampling.call_count, 1)
        self.assertEqual(compiling.call_count, 2)
        self.assertTrue(all(w['include_measurement'] for w in wrapped))
        self.assertEqual([w['axis'] for w in wrapped], ['cosine', 'sine'])
        self.assertEqual(a.paired_G([result], endpoint, 'EXPLORATION')['mean_G_exact'], '120')
        self.assertEqual(len(result['checks']), 7)

    def test_saved_basis_uses_reversed_actual_gate_adjoint(self):
        # Extract this small method rather than import the circuit package.
        tree = ast.parse((ROOT / 'src/trotterlib/df_rte_qiskit.py').read_text())
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_append_basis')
        node.decorator_list = []
        scope = {}
        code = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
        exec(compile(ast.fix_missing_locations(code), '<saved basis method>', 'exec'), scope)
        class Gate:
            def __init__(self, name): self.name = name
            def inverse(self): return 'actual-adjoint-' + self.name
        calls = []
        definition = NS(runtime_operations=[(Gate('a'), (0,)), (Gate('b'), (1,))])
        scope['_append_basis'](NS(append=lambda gate, qubits: calls.append((gate, qubits))), definition, inverse=True)
        self.assertEqual(calls, [('actual-adjoint-b', [1]), ('actual-adjoint-a', [0])])

    def test_reused_compose_preserves_signed_order_control_and_scalar_once(self):
        import h46_long_time_native as native
        calls = []
        class Circuit:
            def __init__(self, n): self.n, self.data = n, []
            def compose(self, qc, inplace): self.data.extend(qc.data)
            def p(self, angle, qubit): self.data.append(('phase', angle, qubit))
        def build(blocks, **kw):
            calls.append(kw)
            qc = Circuit(3); qc.data.append(('piece', kw['T']))
            return NS(circuit=qc, instruction_upper_bound=1)
        backend = NS(block_instruction_bound=lambda b, mode: 1, build_deterministic_native=build,
                     build_partial_native=build, NativeEvolution=lambda *args: args)
        row = a.candidate('FULL', '4th(m5_best)', 28, 2)
        prep = NS(num_system_qubits=2, deterministic_blocks=('saved-mock',))
        names = {'qiskit': NS(QuantumCircuit=Circuit),
                 'trottertracks.resource_applicability': NS(ax2b_native_df_v5=backend),
                 'trotterlib.df_rte_qiskit': NS(QiskitDFRTEEventCircuitBuilder=object)}
        with patch.dict(sys.modules, names):
            out = native.compose(row, prep, [], .3, lambda qc: 'mock-fingerprint')
        self.assertEqual([c['T'] for c in calls], [s['signed_time'] for s in a.schedule(row)])
        self.assertTrue(all(c['scalar'] == 0 and c['control_policy'] == 'symmetric_directional' for c in calls))
        self.assertEqual([op for op in out[0].data if op[0] == 'phase'], [('phase', -.3 * 1.6, 2)])


if __name__ == '__main__':
    unittest.main()
