"""Artificial costs/seeds/providers only. Never load or aggregate H6 science data."""
from copy import deepcopy
from fractions import Fraction
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import nullcontext

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import h6_pr_confirmation as h
import launch_h6_pr_confirmation as launcher
import h8_pr_confirmation_parallel as parallel
import prepare_h6_pr_confirmation as preparation


def target(index=0):
    return dict(deepcopy(h.TARGETS[index]), fixed_PF_G_exact='301')


def pairs(n=3, index=0):
    return [dict(replica=i, seed=i+100, event_digest=str(i), cohort=h.COHORT,
        cell_id=h.TARGETS[index]['cell']['id'], binding=dict(compiler='artificial'),
        axes={a: dict(RZ=100+i if a == 'cosine' else 200-i, seed=i+100, event_digest=str(i))
              for a in h.c.core.AXES}) for i in range(n)]


class ArtificialH6Confirmation(unittest.TestCase):
    def account(self, values=None, candidate=None):
        return h.point_account(values if values is not None else pairs(), dict(cosine=1, sine=1),
                               candidate if candidate is not None else target(), 3, dict(compiler='artificial'))

    def test_fixed_H6_cells_and_both_expected_signs(self):
        for t in h.TARGETS:
            h.check_target(t)
        self.assertEqual([t['expected_direction'] for t in h.TARGETS], [-1, 1])

    def test_wrong_candidate_q_r_K_R_prefix_rejected(self):
        for key, value in [('id', 'B2_p8_q2_2nd_r4_K4'), ('q', 16), ('r', 8),
                           ('K', 4), ('R', 32), ('prefix', 5), ('formula', '4th')]:
            t = target(); t['cell'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                h.check_target(t)

    def test_wrong_PF_epsilon_direction_rejected(self):
        for key, value in [('fixed_PF_id', 'OTHER_PF'), ('epsilon', .01),
                           ('expected_direction', 1), ('scope', 'H8_CONFIRM_P8_Q2')]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'FIXED_H6'):
                h.check_target(dict(target(), **{key: value}))
        with self.assertRaises(ValueError):
            h.check_target(dict(target(1), expected_direction=True))

    def test_PR_more_expensive_can_reproduce_expected_negative_sign(self):
        result = self.account(candidate=dict(target(), fixed_PF_G_exact='299'))
        self.assertEqual(result['reproduction_status'], 'REPRODUCED')
        self.assertFalse(result['PR_CHEAPER'])
        self.assertEqual(result['point_direction'], -1)

    def test_PR_cheaper_does_not_reproduce_expected_negative_sign(self):
        result = self.account()
        self.assertEqual(result['reproduction_status'], 'NOT_REPRODUCED')
        self.assertTrue(result['PR_CHEAPER'])

    def test_PR_cheaper_reproduces_expected_positive_sign(self):
        result = self.account(pairs(index=1), target(1))
        self.assertEqual(result['reproduction_status'], 'REPRODUCED')
        self.assertTrue(result['PR_CHEAPER'])

    def test_PR_more_expensive_does_not_reproduce_expected_positive_sign(self):
        result = self.account(pairs(index=1), dict(target(1), fixed_PF_G_exact='299'))
        self.assertEqual(result['reproduction_status'], 'NOT_REPRODUCED')
        self.assertFalse(result['PR_CHEAPER'])

    def test_exact_tie_is_INCONCLUSIVE(self):
        for index in (0, 1):
            result = self.account(pairs(index=index), dict(target(index), fixed_PF_G_exact='300'))
            self.assertEqual(result['reproduction_status'], 'INCONCLUSIVE')
            self.assertIsNone(result['PR_CHEAPER'])

    def test_paired_covariance_and_no_population_guarantee(self):
        result = self.account()
        self.assertEqual(result['mean_exact'], '300')
        self.assertEqual(result['paired_SE'], 0.)
        self.assertIsNone(result['confidence_interval'])
        self.assertIsNone(result['p_value'])
        self.assertFalse(result['old_samples_pooled'])

    def test_weighted_mean_SD_SE_share_original_H8_body(self):
        result = h.point_account(pairs(), dict(cosine=2, sine=4), target(), 3, dict(compiler='artificial'))
        self.assertEqual(Fraction(result['mean_exact']), 998)
        self.assertEqual(result['sample_SD'], 2.)
        self.assertAlmostEqual(result['paired_SE'], math.sqrt(4/3))
        self.assertEqual(h.shared.N, 48)
        self.assertEqual(h.shared.COHORT, 'INDEPENDENT_UNCONDITIONAL_CONFIRMATION_N48')

    def test_cohort_pooling_duplicate_missing_and_wrong_N_rejected(self):
        for cohort in ['EXPLORATION_N8', 'CONFIRMATION_N32', h.shared.COHORT, 'CONDITIONAL_N8']:
            values = pairs(); values[0]['cohort'] = cohort
            with self.subTest(cohort=cohort), self.assertRaises(ValueError):
                self.account(values)
        for values in [pairs()[:-1], pairs()+[pairs()[0]], [pairs()[0]]*3]:
            with self.assertRaises(ValueError): self.account(values)

    def test_candidate_input_and_axis_mismatches_rejected(self):
        for key, value in [('cell_id', h.TARGETS[1]['cell']['id']), ('binding', dict(compiler='different'))]:
            values = pairs(); values[0][key] = value
            with self.assertRaisesRegex(ValueError, 'CANDIDATE_INPUT_MIXING'): self.account(values)
        for key, value in [('seed', 1000), ('event_digest', 'wrong'), ('RZ', 0), ('RZ', float('nan'))]:
            values = pairs(); values[0]['axes']['sine'][key] = value
            with self.assertRaises(ValueError): self.account(values)

    def test_artificial_seed_registry_variable_N_and_no_H8_mutation(self):
        ns = dict(H6_CONFIRM_P4_Q2=2, H6_CONFIRM_P4_Q8=3)
        tasks = h.derive_registry(h.TARGETS, 'a'*40, 'b'*64, [], ns)
        values = [s for t in tasks for s in h.shared.hierarchy(t['seed'], t['cell']['q'])]
        self.assertEqual(len(tasks), 5)
        self.assertEqual(len(values), len(set(values)))
        self.assertEqual(h.shared.N, 48)
        for source, digest in [('c'*40, 'b'*64), ('a'*40, 'd'*64)]:
            self.assertNotEqual(tasks, h.derive_registry(h.TARGETS, source, digest, [], ns))

    def test_master_and_derived_seed_collisions_rejected_without_redraw(self):
        ns = {scope: 2 for scope in h.SCOPES}
        first = h.derive_registry(h.TARGETS, 'a'*40, 'b'*64, [], ns)[0]
        for value in h.shared.hierarchy(first['seed'], first['cell']['q']):
            with self.assertRaisesRegex(ValueError, 'COLLISION_NO_REDRAW'):
                h.derive_registry(h.TARGETS, 'a'*40, 'b'*64, [value], ns)

    def test_seed_registry_rejects_H8_target_and_unfixed_N(self):
        ts = deepcopy(h.TARGETS); ts[0]['cell']['K'] = 4
        with self.assertRaises(ValueError):
            h.derive_registry(ts, 'a'*40, 'b'*64, [], {s:2 for s in h.SCOPES})
        for n in (None, True, 1, 2.5):
            with self.assertRaises(ValueError):
                h.derive_registry(h.TARGETS, 'a'*40, 'b'*64, [], {s:n for s in h.SCOPES})

    def test_old_H6_and_A128_seed_inventory_cannot_be_omitted(self):
        tasks = [dict(seed=1000+i, cell=dict(q=2)) for i in range(128)]
        excluded = {1,2,3}
        for task in tasks: excluded.update(h.shared.hierarchy(task['seed'], 2))
        payload = dict(new_seed_registry=dict(tasks=tasks, old_seed_exclusions='artificial'))
        registry = dict(excluded_inventory_complete=True, excluded=sorted(excluded))
        with patch.object(h.c, 'read', return_value=dict(values=[1,2,3])):
            h.validate_prior_seeds(registry, payload)
            with self.assertRaisesRegex(ValueError, 'MISSING_FROM_FINAL_EXCLUSIONS'):
                h.validate_prior_seeds(dict(registry, excluded=sorted(excluded-{2})), payload)
            with self.assertRaisesRegex(ValueError, 'COMPLETE_SEED_INVENTORY_REQUIRED'):
                h.validate_prior_seeds(dict(registry, excluded_inventory_complete=False), payload)
            with self.assertRaisesRegex(ValueError, 'A128'):
                h.validate_prior_seeds(registry, dict(new_seed_registry=dict(tasks=tasks[:-1],old_seed_exclusions='artificial')))

    def test_symbolic_budget_becomes_exact_only_for_artificial_fixed_N(self):
        self.assertEqual(h.calls_for(target(), 3), dict(trajectory=3, occurrence=6,
            RTE_event=24, native_evolution=3, wrapper_compile=6, control_probe=4, independent_event_oracle=1))
        with self.assertRaises(ValueError): h.calls_for(target(), None)

    def test_saved_payload_input_shots_and_signal_mismatches_rejected(self):
        t = target()
        saved = dict(cost_acquisition_eligible=True, axes=dict(real=dict(shots=2), imag=dict(shots=3)))
        record = dict(cell=t['cell'], precision_rows={'0.05': {'1.0': saved}})
        t.update(A_signal_record_sha256=h.c.digest(record), accounting_sha256=h.c.digest(saved))
        spec = dict(target=t, binding=dict(compiler='artificial'), shots=dict(cosine=2, sine=3))
        payload = dict(native_adapter=dict(binding=spec['binding']))
        h.check_payload(spec, payload, record)
        with self.assertRaisesRegex(ValueError, 'NATIVE_INPUT_BINDING'):
            h.check_payload(spec, dict(native_adapter=dict(binding=dict(compiler='wrong'))), record)
        with self.assertRaisesRegex(ValueError, 'SAVED_SHOTS'):
            h.check_payload(dict(spec, shots=dict(cosine=3, sine=2)), payload, record)
        changed = deepcopy(record); changed['cell']['K'] = 4
        with self.assertRaises(ValueError): h.check_payload(spec, payload, changed)

    def test_invalid_grant_rejected_before_input_loading_or_provider_import(self):
        with patch.object(h.c, 'read', side_effect=AssertionError('NO_INPUT_READ')):
            with self.assertRaisesRegex(ValueError, 'SCIENCE_APPROVAL'):
                h.run({}, dict(approved=False))
            with self.assertRaisesRegex(ValueError, 'SCIENCE_APPROVAL'):
                launcher.launch({}, dict(approved=False), Path('/unused'))

    def test_even_true_grant_cannot_launch_phase0_proposal(self):
        grant = dict(approved=True, approved_by_user=True, runtime_authorization=True, one_shot=True)
        with patch.object(h.c, 'read', side_effect=AssertionError('NO_INPUT_READ')):
            with self.assertRaisesRegex(ValueError, 'FINAL_H6_EXECUTION_SEAL_REQUIRED'):
                h.run(dict(schema='H6_CONFIRMATION_PHASE0_V1', technical_sealed=False), grant)

    def test_specialization_preserves_keyword_defaults_and_original_globals(self):
        def fn(*, value=3): return value
        self.assertEqual(h.specialize(fn)(), 3)
        self.assertEqual(h.shared.N, 48)

    def test_phase0_schema_rejects_wrong_binding_and_science_design_outputs(self):
        binding = {k:'artificial' for k in h.c.core.IDENTITY}
        binding.update(system='H6', rank=19, rank_policy='tol_only_1e-8')
        plan = dict(schema='H6_CONFIRMATION_PHASE0_V1', cohort=h.COHORT,
            targets=[dict(t, n=None, law=dict(finite_taylor_order=2, orders=[0,2])) for t in h.TARGETS],
            binding=binding, metric='saved_shots_times_compiled_RZ_cosine_plus_sine',
            bias_mode='USUAL_SIGNED', allowance_factor=1, P=0, approved=False,
            runtime_authorization=False, technical_sealed=False, retry=False, next_stage_authorized=False,
            seed_registry=None, calls=None, SOURCE=None,
            numerical_design=dict(approved=False, n_results=None, power_results=None, visibility_results=None))
        with patch.object(h.c, 'read', return_value=dict(systems=dict(H6=dict(binding=binding)))):
            self.assertFalse(h.verify_proposal(plan)['launch_ready'])
            for key, value in [('cohort', h.shared.COHORT), ('SOURCE', 'a'*40),
                               ('approved', True), ('calls', dict(trajectory=48))]:
                with self.subTest(key=key), self.assertRaises(ValueError):
                    h.verify_proposal(dict(plan, **{key:value}))
            changed = deepcopy(plan); changed['binding']['hamiltonian'] = 'changed'
            with self.assertRaisesRegex(ValueError, 'BINDING'): h.verify_proposal(changed)
            changed = deepcopy(plan); changed['numerical_design']['n_results'] = [48,48]
            with self.assertRaisesRegex(ValueError, 'NO_NUMERIC_DESIGN'): h.verify_proposal(changed)

    def test_one_shot_budget_and_first_STOP_reused(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as directory:
            path = Path(directory)
            h.c.claim_attempt(path/'marker', 'a'*40, 'ARTIFICIAL')
            with self.assertRaises(FileExistsError): h.c.claim_attempt(path/'marker', 'a'*40, 'ARTIFICIAL')
            budget = h.c.Budget(path, dict(wrapper_compile=2))
            budget.take('wrapper_compile', 2)
            with self.assertRaises(ValueError): budget.take('wrapper_compile')
            h.c.first_stop(path, 'FIRST'); h.c.first_stop(path, 'SECOND')
            self.assertEqual(h.c.read(path/'STOP.json')['reason'], 'FIRST')

    def test_runtime_SHA_cache_and_changed_leaf_rejection_on_artificial_files(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as directory:
            path = Path(directory); leaf = path/'fake_runtime'; leaf.write_text('artificial runtime')
            h.c.save(path/'old_receipt', dict(expected={str(leaf):h.c.sha(leaf)}))
            origin = dict(runtime_receipt=str(path/'old_receipt'), runtime_receipt_sha256=h.c.sha(path/'old_receipt'))
            (path/'first').mkdir(); (path/'second').mkdir(); (path/'third').mkdir()
            first = preparation.runtime_audit(origin, path/'first')
            second = preparation.runtime_audit(origin, path/'second', Path(first['path']))
            self.assertEqual(h.c.read(second['path'])['SHA_reused_on_unchanged_stat'], 1)
            leaf.write_text('corrupted runtime')
            with self.assertRaisesRegex(ValueError, 'RUNTIME_SHA_MISMATCH'):
                preparation.runtime_audit(origin, path/'third', Path(second['path']))

    def test_changed_input_and_SOURCE_bytes_rejected_before_dispatch(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as directory:
            path = Path(directory); leaf = path/'fake_input'; leaf.write_text('saved')
            row = preparation.byte_row(leaf, 'artificial_input')
            h.c.verify_byte_rows([row])
            leaf.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'INPUT_BYTES_CHANGED'): h.c.verify_byte_rows([row])
            with self.assertRaisesRegex(ValueError, 'SOURCE_BYTES_CHANGED'):
                h.c.verify_source(path, 'a'*40, {'fake_input':row['sha256']})

    def fixture_parallel(self, index=0):
        # Reuse existing H8 mock provider/owned fork fixture, change only H6 metadata.
        fixtures = h.shared.load_own('h6_shared_H8_artificial_fixtures', 'tests/test_h8_pr_confirmation.py')
        task, _, _, native, context, pool = fixtures.ArtificialConfirmation().parallel_fixture()
        t = target(index)
        task['cell'] = t['cell']; task['id'] = t['cell']['id']+'_0'
        law = dict(finite_taylor_order=2, orders=[0,2], order_probabilities=[.98,.02])
        t.update(preparation_sha256='prep', original_law_sha256=h.c.digest(law))
        requests = [SimpleNamespace(seed=s) for s in h.shared.hierarchy(task['seed'], t['cell']['q'])[1:]]
        events = [[dict(taylor_order=0) for _ in range(t['cell']['r'])] for _ in range(t['cell']['q'])]
        native.steps = lambda *args: (requests, SimpleNamespace(to_dict=lambda: law, exact_finite_distribution=1.))
        native.event_values = lambda _: events
        context.preparations = {4: SimpleNamespace(preparation_hash='prep')}
        return task, law, t, native, context, pool, events

    def test_K4_law_and_K4_event_rejected_before_native_build(self):
        task, law, t, native, context, pool, events = self.fixture_parallel()
        adapter = h.checked_native(native)
        adapter.steps(None, task['cell'], task['seed'])
        events[0][0]['taylor_order'] = 4
        with self.assertRaisesRegex(ValueError, 'K2_EVENT_SUPPORT'):
            adapter.steps(None, task['cell'], task['seed'])
        events[0][0]['taylor_order'] = 0; law['orders'] = [0,2,4]
        with self.assertRaisesRegex(ValueError, 'K2_SUPPORT_ONLY'):
            adapter.steps(None, task['cell'], task['seed'])

    def test_mock_H6_parallel_control_compile_acceptance_and_reap_both_candidates(self):
        for index in (0, 1):
            task, law, t, native, context, pool, events = self.fixture_parallel(index)
            modules = {'trottertracks.resource_applicability.h6_hash_reuse_v1':
                       SimpleNamespace(use_fingerprint_reuse=nullcontext),
                       'h6_dba_science': SimpleNamespace(event_oracle=lambda *args: 1+0j)}
            with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as directory:
                with patch.dict(sys.modules, modules):
                    result = parallel.compile_one(context, task, directory, 'a'*40, t, law, 0., 1e-10,
                        h.checked_native(native), pool, sorted(os.sched_getaffinity(0))[:2], None, h.COHORT)
                self.assertEqual(result['cohort'], h.COHORT)
                parallel.verify_descendants(directory, result)
                path = Path(directory)
                h.c.save(path/'task_result.json', result)
                identity = (1337, 20, 1000, os.getuid())
                h.c.save(path/'started.json', dict(task=task['id'], identity=identity))
                handle = dict(directory=directory, reaped=True, exit_code=0,
                    result_sha256=h.c.sha(path/'task_result.json'), pid=1337, identity=identity)
                spec = dict(SOURCE='a'*40, binding=context.binding, law=law, target=t)
                self.assertEqual(h.accept(task, handle, spec)[0], result)
                handle['exit_code'] = 1
                with self.assertRaisesRegex(ValueError, 'NATIVE_EXIT'):
                    h.accept(task, handle, spec)

    def test_mock_complete_H6_runner_scheduler_accounting_and_all_reaped(self):
        task, law, t, native, context, pool, events = self.fixture_parallel()
        t.update(saved_log_B=0., log_B_margin=1e-10)
        def steps(prep, cell, master):
            requests = [SimpleNamespace(seed=s) for s in h.shared.hierarchy(master, cell['q'])[1:]]
            return requests, SimpleNamespace(to_dict=lambda: law, exact_finite_distribution=1.)
        native.steps = steps
        modules = {'pr_cost_native': native, 'h8_key_compile_parallel': pool,
            'pr_cost_batch_science': SimpleNamespace(native_context=lambda *args:context),
            'trottertracks.resource_applicability.h6_hash_reuse_v1': SimpleNamespace(use_fingerprint_reuse=nullcontext),
            'h6_dba_science': SimpleNamespace(event_oracle=lambda *args: 1+0j)}
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as directory:
            path = Path(directory); output = path/'output'; output.mkdir()
            h.c.save(path/'payload', dict(native_adapter=dict(font_seed=None)))
            tasks = h.derive_registry(h.TARGETS, 'a'*40, 'b'*64, [], {s:2 for s in h.SCOPES})
            h.c.save(path/'registry', dict(tasks=tasks))
            spec = dict(payload=str(path/'payload'), registry=str(path/'registry'), target=t,
                n=2, scope=t['scope'], SOURCE='a'*40, law=law, binding=context.binding,
                calls=h.calls_for(t,2), shots=dict(cosine=1,sine=1))
            with patch.dict(sys.modules, modules), patch.object(h.shared, 'effective_fresh', return_value=dict(artificial=True)):
                result = h.execute(spec, dict(allowed_cpus=sorted(os.sched_getaffinity(0))[:2]), output)
            self.assertEqual(result['status'], 'COMPLETE_MANDATORY_STOP')
            self.assertEqual(result['calls'], spec['calls'])
            self.assertEqual(result['primary']['mean_exact'], '20')
            self.assertEqual(result['primary']['reproduction_status'], 'NOT_REPRODUCED')
            self.assertTrue(result['primary']['PR_CHEAPER'])
            self.assertFalse(result['old_samples_pooled'])
            self.assertTrue(h.c.read(output/'native_receipt.json')['all_reaped'])

    def launch_artificial_child(self, directory, exit_code):
        path = Path(directory); output = path/'output'; output.mkdir()
        h.c.save(output/'result.json', dict(artificial=True))
        h.c.save(output/'terminal.json', dict(status='COMPLETE_MANDATORY_STOP', SOURCE='a'*40,
                                             result_sha256=h.c.sha(output/'result.json')))
        spec = dict(SOURCE='a'*40, scope='H6_CONFIRM_P4_Q2', python=sys.executable, output=str(output))
        grant = dict(spec_path='ARTIFICIAL_SPEC', grant_path='ARTIFICIAL_GRANT')
        popen = subprocess.Popen
        def tiny_child(*args, **kwargs):
            return popen([sys.executable, '-P', '-B', '-c', 'raise SystemExit('+str(exit_code)+')'], **kwargs)
        with patch.object(h, 'authorize'), patch.object(h, 'verify'), patch.object(h.shared, 'fresh_gate'), \
             patch.object(launcher.subprocess, 'Popen', side_effect=tiny_child):
            launcher.launch(spec, grant, path/'witness')

    def test_native_exit_witness_reaps_artificial_child_and_saves_audit(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as directory:
            self.launch_artificial_child(directory, 0)
            receipt = h.c.read(Path(directory)/'witness/completion_audit.json')
            self.assertTrue(receipt['native']['reaped'])
            self.assertEqual(receipt['native']['exit_code'], 0)
            self.assertFalse(receipt['next_stage_authorized'])

    def test_native_exit_failure_preserved_as_STOP(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as directory:
            with self.assertRaisesRegex(ValueError, 'H6_DRIVER_NATIVE_EXIT0_REQUIRED'):
                self.launch_artificial_child(directory, 3)
            path = Path(directory)/'witness'
            self.assertEqual(h.c.read(path/'native_completion.json')['exit_code'], 3)
            self.assertTrue(h.c.read(path/'native_completion.json')['reaped'])
            self.assertTrue((path/'STOP.json').exists())

    def test_cli_without_grant_never_dispatches(self):
        p = subprocess.run([sys.executable, '-P', '-B', str(h.ROOT/'scripts/h6_pr_confirmation.py')],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn('EXPLICIT_H6_SCIENCE_APPROVAL_REQUIRED', p.stderr)


if __name__ == '__main__':
    unittest.main()
