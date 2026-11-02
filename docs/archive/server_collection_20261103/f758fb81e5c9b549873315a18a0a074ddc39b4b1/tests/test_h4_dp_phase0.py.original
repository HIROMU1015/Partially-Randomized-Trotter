"""Artificial-only checks: never open H4 raw trajectories or load Qiskit/NumPy."""
import ast
from copy import deepcopy
import importlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import h4_dp_contract as c
import h4_dp_science as science
import run_h4_dp_verification as runner


def signal(cid, eligible):
    a = dict(cost_acquisition_eligible=eligible, eligibility_under_declared_allowance='ELIGIBLE' if eligible else 'INELIGIBLE',
             axes={axis: dict(shots=1 if eligible else None) for axis in ('real', 'imag')})
    return dict(cell=dict(id=cid), **{field: {str(eps): {'1.0': deepcopy(a)} for eps in c.EPS}
                                    for field in c.FIELDS.values()})


def pair(cosine, sine, seed=1):
    return dict(seed=seed, event_digest='e'*64, axes={a: dict(RZ=v, seed=seed,
        event_digest='e'*64, compiler_hash='c'*64, fingerprint='f'*64)
        for a, v in [('cosine', cosine), ('sine', sine)]})


class MetadataAndStatistics(unittest.TestCase):
    def test_saved_eligibility_is_the_only_exclusion(self):
        out = c.deterministic_tasks([signal('a', True), signal('b', False), signal('c', True)], ['a'])
        self.assertEqual(out['compile_ids'], ['c'])
        self.assertEqual(out['excluded'], ['b'])

    def test_unknown_is_not_ineligible(self):
        r = signal('a', False)
        r['precision_rows']['0.05']['1.0']['eligibility_under_declared_allowance'] = 'UNRESOLVED'
        with self.assertRaisesRegex(ValueError, 'UNRESOLVED'):
            c.deterministic_tasks([r], [])

    def test_missing_shots_rejected(self):
        r = signal('a', True); r['precision_rows']['0.05']['1.0']['axes']['real']['shots'] = None
        with self.assertRaisesRegex(ValueError, 'MISSING_SAVED_SHOTS'):
            c.saved_endpoint(r, 'USUAL_SIGNED', .05)

    def test_duplicate_domain_rejected(self):
        with self.assertRaisesRegex(ValueError, 'DUPLICATE'):
            c.deterministic_tasks([signal('a', True), signal('a', True)], [])

    def test_measured_ids_cannot_expand_domain(self):
        with self.assertRaisesRegex(ValueError, 'OUTSIDE_DOMAIN'):
            c.deterministic_tasks([signal('a', True)], ['other'])

    def test_missing_axis_rejected(self):
        r = pair(1, 2); del r['axes']['sine']
        with self.assertRaisesRegex(ValueError, 'MISSING_PAIRED_AXIS'):
            c.check_pair(r, 'c'*64)

    def test_unpaired_event_rejected(self):
        r = pair(1, 2); r['axes']['sine']['event_digest'] = 'x'*64
        with self.assertRaisesRegex(ValueError, 'UNPAIRED'):
            c.check_pair(r, 'c'*64)

    def test_zero_cost_is_not_a_missing_value(self):
        r = pair(0, 2)
        with self.assertRaisesRegex(ValueError, 'INVALID_COMPILED_RZ'):
            c.check_pair(r, 'c'*64)

    def test_same_trajectory_covariance_is_preserved(self):
        out = c.evaluate_pairs([pair(10, 20), pair(20, 10)], dict(cosine=1, sine=1), '31', 1)
        self.assertEqual(out['G_exact'], '30'); self.assertEqual(out['paired_SE'], 0)
        self.assertEqual(out['status'], 'REPRODUCED')

    def test_bad_direction_and_exact_tie_are_distinct(self):
        pairs = [pair(10, 20), pair(20, 10)]
        self.assertEqual(c.evaluate_pairs(pairs, dict(cosine=1, sine=1), '29', 1)['status'], 'NOT_REPRODUCED')
        self.assertEqual(c.evaluate_pairs(pairs, dict(cosine=1, sine=1), '30', 1)['status'], 'INCONCLUSIVE')

    def test_expected_pf_advantage_can_be_reproduced(self):
        out = c.evaluate_pairs([pair(10, 20), pair(20, 10)], dict(cosine=1, sine=1), '29', -1)
        self.assertEqual(out['status'], 'REPRODUCED'); self.assertEqual(out['point_direction'], -1)

    def test_cross_epsilon_covariance(self):
        out = c.within_candidate_covariance([pair(10, 20), pair(20, 10)],
            [dict(cosine=1, sine=0), dict(cosine=0, sine=1)])
        self.assertEqual(out, [['50', '-50'], ['-50', '50']])

    def test_normal_planning_formula_synthetic(self):
        self.assertEqual(c.normal_n_formula(2., 1., 2., 1.), 36)
        with self.assertRaisesRegex(ValueError, 'INVALID'):
            c.normal_n_formula(float('nan'), 1., 2., 1.)

    def test_seed_namespace_candidate_and_replica(self):
        values = [c.independent_seed('H4_PR_INDEPENDENT_CONFIRMATION_V1', 'a'*40, 'b'*64, cid, rep)
                  for cid in ('synthetic_a', 'synthetic_b') for rep in range(3)]
        self.assertEqual(len(set(values)), 6)
        with self.assertRaisesRegex(ValueError, 'NAMESPACE'):
            c.independent_seed('OLD', 'a'*40, 'b'*64, 'x', 0)

    def test_master_and_step_collision(self):
        old = set(c.seed_hierarchy(123, 4))
        with self.assertRaisesRegex(ValueError, 'COLLISION'):
            c.reserve_seed(old, 123, 4)

    def test_rare_event_strata(self):
        for orders, label in [([0, 0], 'S0_ONLY'), ([0, 2], 'S2_SINGLE'),
                              ([2, 2], 'S2_MULTI'), ([2, 4], 'S4_PRESENT')]:
            self.assertEqual(c.event_stratum([[dict(taylor_order=x) for x in orders]]), label)
        with self.assertRaisesRegex(ValueError, 'UNREGISTERED'):
            c.event_stratum([[dict(taylor_order=6)]])

    def test_preparation_ast_does_not_call_scientific_functions(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'scripts/prepare_h4_dp_phase0.py').read_text())
        prohibited = {'evaluate_pairs', 'normal_n_formula', 'within_candidate_covariance', 'transpile',
                      'compile_task', 'bootstrap', 'stdev', 'variance', 'mean', 'load_context'}
        calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertFalse(calls & prohibited)


class GatesAndFutureAccounting(unittest.TestCase):
    def make_artificial_authorized_metadata(self, tmp, stage='D'):
        root = Path(tmp)
        receipt = root/'approval'; receipt.write_text('artificial authorization fixture')
        source = root/'code'; source.write_text('artificial source')
        data = root/'input'; data.write_text('artificial input')
        spec = dict(status='SEALED_FOR_EXECUTION', technical_sealed=True, stage=stage, SOURCE='a'*40,
            source_hashes={'code': c.sha(source)}, inputs=[dict(role='synthetic', path=str(data), sha256=c.sha(data))],
            runtime_hashes={str(source): c.sha(source)}, provider_hashes={str(source): c.sha(source)},
            python_sha256=c.sha(sys.executable), output=str(root/'output'), lock=str(root/'lock'), workers=1)
        grant = dict(approved_by_user=True, runtime_authorization=True, stage=stage, SOURCE=spec['SOURCE'],
            one_shot=True, retry=False, next_stage_authorized=False, spec_digest=c.digest(spec),
            approval_receipt=str(receipt), approval_receipt_sha256=c.sha(receipt))
        return spec, grant

    def test_approved_metadata_still_rejects_changed_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec, grant = self.make_artificial_authorized_metadata(tmp)
            c.authorize(spec, grant, tmp)
            Path(spec['inputs'][0]['path']).write_text('changed artificial input')
            with self.assertRaisesRegex(ValueError, 'INPUT_CHANGED'):
                c.authorize(spec, grant, tmp)

    def test_p_requires_its_pending_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec, grant = self.make_artificial_authorized_metadata(tmp, 'P')
            with self.assertRaisesRegex(ValueError, 'P_PENDING_DEPENDENCIES'):
                c.authorize(spec, grant, tmp)

    def test_draft_runner_rejects_before_any_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'DRAFT_UNAUTHORIZED'):
                runner.execute(dict(status='DRAFT_UNAUTHORIZED', output=tmp+'/output'), {})
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_invalid_grant_rejected(self):
        with self.assertRaisesRegex(ValueError, 'EXPLICIT'):
            c.authorize(dict(status='SEALED_FOR_EXECUTION', technical_sealed=True),
                        dict(approved_by_user=False, runtime_authorization=False), '.')

    def test_cross_stage_approval_rejected(self):
        with self.assertRaisesRegex(ValueError, 'STAGE_BINDING'):
            c.authorize(dict(status='SEALED_FOR_EXECUTION', technical_sealed=True, stage='P'),
                        dict(approved_by_user=True, runtime_authorization=True, stage='D'), '.')

    def test_d_task_scope_and_compile_budget_are_fixed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'domain'; c.save(path, dict(compile_ids=['a']))
            spec = dict(stage='D', tasks=[dict(id='a', cell=dict(id='a'))],
                        registered_tasks=str(path), calls=dict(wrapper_compile=2, control_probe=4))
            c.validate_tasks(spec)
            spec['tasks'].append(dict(id='extra', cell=dict(id='extra')))
            with self.assertRaisesRegex(ValueError, 'EXACT_D_COMPILE_SCOPE'):
                c.validate_tasks(spec)

    def test_p_seed_and_fixed_replicas_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            cell = dict(id='synthetic', q=1, prefix=2)
            registration = Path(tmp)/'registration'; exclusions = Path(tmp)/'exclusions'
            shots = dict(cosine=1, sine=1)
            c.save(registration, dict(candidates=[cell], assignments=[dict(cell_id='synthetic', epsilon=.05)],
                saved_statistics=[dict(cell_id='synthetic', epsilon=.05, saved_shots=shots)]))
            c.save(exclusions, dict(values=[]))
            bindings, preparations = Path(tmp)/'bindings', Path(tmp)/'preparations'
            c.save(bindings, dict(synthetic=dict(log_B=0., log_B_tolerance=1e-11)))
            c.save(preparations, dict(prefixes={'2': dict(preparation_hash='synthetic_prep')}))
            spec = dict(stage='P', SOURCE='a'*40, seed_plan_digest='b'*64, registration=str(registration),
                seed_exclusions=str(exclusions), n_fixed=dict(synthetic=2),
                calls=dict(wrapper_compile=4), evaluation_endpoints=[dict(cell_id='synthetic', epsilon=.05,
                    saved_shots=shots, PF_id='artificial_PF', PF_G_exact='10')])
            d_result = Path(tmp)/'D'; c.save(d_result, dict(winners=[dict(epsilon=.05, bias_mode='USUAL_SIGNED',
                best=dict(cell_id='artificial_PF', G_exact='10'))]))
            spec.update(D_result_path=str(d_result), D_result_sha256=c.sha(d_result))
            spec.update(signal_bindings=str(bindings), saved_preparation_identities=str(preparations))
            spec['tasks'] = [dict(id='synthetic_'+str(i), cell=cell, replica=i,
                saved_log_B=0., log_B_tolerance=1e-11, preparation_hash='synthetic_prep',
                seed=c.independent_seed('H4_PR_INDEPENDENT_CONFIRMATION_V1', spec['SOURCE'],
                    spec['seed_plan_digest'], 'synthetic', i)) for i in range(2)]
            c.validate_tasks(spec)
            spec['tasks'][1]['seed'] = spec['tasks'][0]['seed']
            with self.assertRaisesRegex(ValueError, 'SEALED_CONFIRMATION_SEED'):
                c.validate_tasks(spec)

    def test_compile_axes_checks_fingerprints_and_budget(self):
        result = SimpleNamespace(compiler_settings_hash='c'*64, actual_circuit_fingerprint='f'*64,
            rz_count=10, cx_count=2, rz_depth=3, cx_depth=1, total_depth=5, circuit_size=12)
        fake = SimpleNamespace(build_native_hadamard_wrapper=lambda *a, **k: object(),
            canonical_qiskit_circuit_fingerprint=lambda x: 'f'*64,
            transpile_and_measure_cost=lambda *a, **k: result)
        context = SimpleNamespace(m=fake, compiler=object(), binding=dict(compiler='c'*64))
        with tempfile.TemporaryDirectory() as tmp:
            budget = science.Budget(tmp, dict(wrapper_compile=2))
            rows = science.compile_axes(context, object(), science.Writer(tmp), budget)
            self.assertEqual(set(rows), set(c.AXES)); self.assertEqual(budget.used['wrapper_compile'], 2)
            with self.assertRaisesRegex(ValueError, 'BUDGET'):
                budget.take('wrapper_compile')
            result.actual_circuit_fingerprint = 'x'*64
            with self.assertRaisesRegex(ValueError, 'COMPILE_IDENTITY'):
                science.compile_axes(context, object(), science.Writer(tmp), science.Budget(tmp, dict(wrapper_compile=2)))

    def test_missing_eligible_pf_never_becomes_zero(self):
        records = [dict(cell=dict(id='a'))]
        endpoints = [dict(cell_id='a', bias_mode='USUAL_SIGNED', epsilon=.05,
                         saved_accounting=dict(cost_acquisition_eligible=True, axes={
                             'real': dict(shots=1), 'imag': dict(shots=1)}))]
        with self.assertRaisesRegex(ValueError, 'MISSING_ELIGIBLE_PF_COST'):
            science.deterministic_account(records, endpoints, {})

    def test_future_pf_exact_accounting_synthetic(self):
        records = [dict(cell=dict(id='a', method='B0', prefix=2, q=1, formula='2nd')),
                   dict(cell=dict(id='b', method='B0', prefix=3, q=1, formula='2nd'))]
        endpoints = [dict(cell_id=cid, bias_mode=mode, epsilon=eps,
            saved_accounting=dict(cost_acquisition_eligible=True, axes={'real': dict(shots=2), 'imag': dict(shots=3)}))
            for mode in c.FIELDS for eps in c.EPS for cid in ('a', 'b')]
        out = science.deterministic_account(records, endpoints,
            {cid: {axis: dict(RZ=cost) for axis in c.AXES} for cid, cost in [('a', 3), ('b', 4)]})
        self.assertEqual(out['winners'][0]['best']['G_exact'], '15')
        self.assertEqual(out['winners'][0]['best']['cell_id'], 'a')
        self.assertFalse(out['global_optimum_certified'])

    def test_boundary_is_formula_specific(self):
        cell = dict(method='B0', prefix=4, q=1, formula='4th')
        cells = [cell, dict(method='B0', prefix=2, q=1, formula='2nd'),
                 dict(method='B0', prefix=6, q=1, formula='4th')]
        self.assertEqual(c.boundary_attributes(cell, cells)['prefix']['position'], 'LOWER')

    def test_new_preparation_preserves_old_artifact_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = Path(tmp) / 'old'; old.write_bytes(b'artificial STOP receipt')
            before = c.sha(old)
            with self.assertRaisesRegex(ValueError, 'DRAFT_UNAUTHORIZED'):
                runner.execute(dict(status='DRAFT_UNAUTHORIZED'), dict())
            self.assertEqual(c.sha(old), before)


if __name__ == '__main__':
    unittest.main()
