"""Artificial boundary tests. No molecular restore, sampling or compile."""
import copy
import importlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import h46_standard_y4_contract as c


def candidate(system='H4'):
    g = c.GROUPS[system]
    return dict(id=g['Y4'], system=system, method='PR', formula='4th',
        q=4, prefix=g['prefix'], K=4, r_policy='ABS_TIME_PER_OUTER', r_value=32)


class MetadataTests(unittest.TestCase):
    def test_fixed_counts_and_axes(self):
        for s, n in [('H4', 45), ('H6', 32)]:
            tasks = c.tasks_for(s, candidate(s))
            self.assertEqual(len(tasks), n)
            self.assertEqual(len({t['id'] for t in tasks}), n)
            self.assertTrue(all(t['axes'] == ['cosine', 'sine'] for t in tasks))

    def test_wrong_scientific_candidate_fields(self):
        for key, value in [('formula', '4th(m5_best)'), ('q', 8), ('prefix', 5),
                ('K', 2), ('r_value', 128), ('method', 'FULL'), ('system', 'H6'),
                ('id', c.GROUPS['H4']['S2']), ('r_policy', 'PER_PIECE')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                c.tasks_for('H4', dict(candidate(), **{key: value}))

    def test_no_draft_approval(self):
        with self.assertRaisesRegex(ValueError, 'EXPLICIT_STANDARD'):
            c.authorize(dict(authorized=False, grant=dict(valid=False)))

    def test_default_scope_no_numerical_import(self):
        code = 'import run_h46_standard_y4 as r,sys;r.main();assert "numpy" not in sys.modules;assert "qiskit" not in sys.modules'
        p = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT / 'scripts',
            capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('SCOPE_ONLY', p.stdout)

    def test_cli_unapproved_real_like_input_refused_early(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'cfg.json'
            p.write_text(json.dumps(dict(authorized=False, grant=dict(valid=False),
                assets='/missing/scientific/assets')))
            out = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/run_h46_standard_y4.py'),
                '--execute', '--config', str(p)], capture_output=True, text=True)
            self.assertEqual(out.returncode, 2)
            self.assertIn('EXPLICIT_STANDARD_Y4_SCIENCE_APPROVAL_REQUIRED', out.stdout)
            self.assertFalse((Path(d) / 'CONSUMED.json').exists())

    def test_create_only_preserves_old_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'receipt.json'
            c.save(p, dict(a=1))
            old = p.read_bytes()
            with self.assertRaises(FileExistsError):
                c.save(p, dict(a=2))
            self.assertEqual(p.read_bytes(), old)

    def test_changed_sealed_input(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'input'
            p.write_bytes(b'old')
            r = c.seal(p, 'mock_input')
            p.write_bytes(b'new')
            with self.assertRaisesRegex(ValueError, 'SEALED_SHA'):
                c.check_files([r])

    def test_policy_no_pooling_or_old_epsilon(self):
        scope = c.analysis_scope()
        self.assertEqual(scope['epsilon'], .001)
        self.assertEqual(scope['family_size'], 2)
        self.assertFalse(scope['pooling'])
        self.assertFalse(scope['adaptive_top_up'])
        self.assertFalse(scope['rare_cost_imputation'])
        self.assertFalse(scope['population_guarantee'])

    def test_consumed_grant_must_match(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'CONSUMED.json'
            c.save(p, dict(state='CONSUMED_NO_RETRY_NO_REFUND', grant=dict(id='old')))
            with self.assertRaisesRegex(ValueError, 'CONSUMED'):
                c.consumed(dict(marker=str(p), grant=dict(id='new')))

    def test_approval_needs_exact_hash(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'approval.json'
            c.save(p, dict(role='user', execution_authorized=True, package_sha256='wrong'))
            with patch.object(c, 'verify_package', return_value={}):
                with self.assertRaisesRegex(ValueError, 'EXPLICIT_EXACT'):
                    c.authorize_package(p, p)


class LegacyBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assets = os.environ.get('H46_STANDARD_Y4_ASSETS')
        if not assets:
            raise unittest.SkipTest('Set H46_STANDARD_Y4_ASSETS to the recovered, SHA-verified c028 closure')
        c.install_paths(assets)
        cls.assets = Path(assets)

    def test_exact_calls_and_no_new_s2(self):
        total = {}
        for s in c.GROUPS:
            scope = c.law.call_scope(c.tasks_for(s, candidate(s)), 'PR')
            for k, v in scope.items():
                if type(v) is int:
                    total[k] = total.get(k, 0) + v
        self.assertEqual(total['trajectories'], 77)
        self.assertEqual(total['compile_axes'], 154)
        self.assertEqual(total['request_sampling_calls'], 924)
        self.assertEqual(total['rte_events'], 9856)
        self.assertEqual(total['shot_calculation'], 0)

    def test_registry_reproducible_and_cohort_disjoint(self):
        tasks = c.tasks_for('H4', candidate())[:2]
        x = c.law.registry(tasks, 'a' * 40, 'new-A', 'CONFIRMATION', [])
        y = c.law.registry(tasks, 'a' * 40, 'new-B', 'CONFIRMATION', [])
        self.assertEqual(x, c.law.registry(tasks, 'a' * 40, 'new-A', 'CONFIRMATION', []))
        self.assertFalse({v for r in x.values() for v in r.values()} &
                         {v for r in y.values() for v in r.values()})

    def test_known_seed_collision_rejected_without_rehash(self):
        tasks = c.tasks_for('H4', candidate())[:1]
        reg = c.law.registry(tasks, 'a' * 40, 'fixed', 'CONFIRMATION', [])
        existing = next(iter(next(iter(reg.values())).values()))
        with self.assertRaisesRegex(ValueError, 'COLLISION'):
            c.law.registry(tasks, 'a' * 40, 'fixed', 'CONFIRMATION', [existing])

    def test_reserve_exact_no_retry(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            t = c.tasks_for('H4', candidate())[0]
            cfg = dict(marker=str(root / 'CONSUMED.json'), grant={'id': 'synthetic'},
                tasks=[t], ledger=str(root), SOURCE='a' * 40, stage='H4_CONFIRMATION')
            c.save(cfg['marker'], dict(state='CONSUMED_NO_RETRY_NO_REFUND', grant=cfg['grant']))
            c.reserve(cfg, t, 'compile_axes', 0)
            with self.assertRaises(FileExistsError):
                c.reserve(cfg, t, 'compile_axes', 0)
            with self.assertRaisesRegex(ValueError, 'EXACT_TASK_CALL'):
                c.reserve(cfg, t, 'compile_axes', 2)

    def test_c028_cohort_rebinding_changes_labels_only(self):
        import h246_confirmation_runtime as old
        import h46_pr_pilot_native as native
        rebound, ns = old.rebind(native, dict(c=SimpleNamespace()))
        self.assertIn('CONFIRMATION', rebound.run_task.__code__.co_consts)
        self.assertNotIn('DESIGN_PILOT', rebound.run_task.__code__.co_consts)
        self.assertEqual(rebound.run_task.__code__.co_code, native.run_task.__code__.co_code)

    def test_worker_is_importable_process_pool_target(self):
        import h46_standard_y4_runtime as r
        self.assertIs(pickle.loads(pickle.dumps(r.worker)), r.worker)

    def test_frozen_runner_rebinding_and_worker_default(self):
        import h46_standard_y4_runtime as r
        import run_h46_pr_design_pilot as old
        cfg = dict(stage='H4_CONFIRMATION', signal_path='/toy/signal.json')
        with patch.object(c, 'authorize'), patch.object(r, 'RT', None), \
                patch.object(r, 'ACTIVE_RUNTIME', None):
            rt = r.runtime(cfg)
            self.assertEqual(rt.runner.execute.__code__.co_code, old.execute.__code__.co_code)
            self.assertIs(rt.runner.execute.__kwdefaults__['target'], r.worker)
            self.assertIs(rt.runner_namespace['worker'], r.worker)
            self.assertIs(rt.runner_namespace['repair'].prewarm, r.prewarm)
            self.assertEqual(rt.runner.supervise.__globals__['__file__'],
                str(c.ROOT / 'scripts/run_h46_standard_y4.py'))
            self.assertEqual(rt.runner.supervise.__globals__['c'].design.COHORT, 'CONFIRMATION')

    def test_no_worker_context_miss_allowed(self):
        import h46_standard_y4_runtime as r
        with patch.object(r, 'WARM', False):
            with self.assertRaisesRegex(ValueError, 'PARENT_PREWARM'):
                r.guarded_context({}, 'H4', 3)

    def test_actual_build_dispatch_counted_before_wrapper_call(self):
        import h46_standard_y4_runtime as r
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            task = c.tasks_for('H4', candidate())[0]
            out = root / 'output' / task['id']
            out.mkdir(parents=True)
            ledger = root / 'ledger'
            ledger.mkdir()
            cfg = dict(output=str(root/'output'),ledger=str(ledger),marker=str(root/'CONSUMED.json'),
                grant={'toy':True},tasks=[task],SOURCE='a'*40,stage=task['stage'],plan_sha256='b'*64)
            c.save(cfg['marker'],dict(state='CONSUMED_NO_RETRY_NO_REFUND',grant=cfg['grant']))
            api = ModuleType('toy_native_api')
            api.build_partial_native = lambda *args,**kw: 'toy partial'
            def original_wrapper():
                self.assertTrue(list(ledger.glob('*__wrapper_build__*.json')))
                return 'toy wrapper'
            api.build_native_hadamard_wrapper = original_wrapper
            parent = ModuleType('trottertracks.resource_applicability')
            parent.ax2b_native_df_v5 = api
            def fake_worker(task):
                for _ in range(12):api.build_partial_native()
                for _ in range(2):api.build_native_hadamard_wrapper()
                return {'toy':'complete'}
            runtime = SimpleNamespace(worker=fake_worker,RUNTIME=SimpleNamespace(
                runner_namespace={'WORKER_CFG':cfg}))
            with patch.dict(sys.modules,{'trottertracks.resource_applicability':parent}), \
                    patch.object(r,'RT',runtime), patch.object(r,'ACTIVE_RUNTIME',runtime.RUNTIME):
                self.assertEqual(r.worker(task),{'toy':'complete'})
            self.assertIs(api.build_native_hadamard_wrapper,original_wrapper)
            self.assertEqual(c.read(out/'build_dispatch.json')['counts'],
                dict(partial_native_build=12,hadamard_wrapper_build=2))
            self.assertEqual(len(list(ledger.iterdir())),2)
            self.assertEqual(sum(c.task_limits(task).values()),38)

    def test_real_lazy_runtime_is_retained_through_fork_pool_initializer(self):
        """Exercise actual rebind/get_runtime and runner initialize, no molecule."""
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        import h46_standard_y4_runtime as r
        import run_h46_pr_design_pilot as old_runner
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = c.tasks_for('H4', candidate())[0]
            (root / 'output' / task['id']).mkdir(parents=True)
            (root / 'ledger').mkdir()
            cfg = dict(stage='H4_CONFIRMATION',signal_path='/synthetic/signal',
                SOURCE='a'*40,plan_sha256='b'*64,tasks=[task],
                systems={'H4':dict(inputs={'synthetic':True})},
                output=str(root/'output'),ledger=str(root/'ledger'),
                marker=str(root/'CONSUMED.json'),grant={'synthetic':True})
            c.save(cfg['marker'],dict(state='CONSUMED_NO_RETRY_NO_REFUND',grant=cfg['grant']))
            api=ModuleType('synthetic_native_api')
            api.build_partial_native=lambda *a,**k:None
            api.build_native_hadamard_wrapper=lambda *a,**k:None
            parent=ModuleType('trottertracks.resource_applicability')
            parent.ax2b_native_df_v5=api
            with patch.object(c,'authorize'),patch.object(r,'RT',None), \
                    patch.object(r,'ACTIVE_RUNTIME',None),patch.object(r,'WARM',True), \
                    patch.dict(sys.modules,{'trottertracks.resource_applicability':parent}):
                live=r.runtime(cfg)
                self.assertIsNone(r.RT.RUNTIME)
                self.assertIs(r.RT.get_runtime.__globals__['RUNTIME'],live)
                self.assertIs(r.ACTIVE_RUNTIME,live)
                context=SimpleNamespace(synthetic=True)
                key=(cfg['SOURCE'],'H4',c.digest(cfg['systems']['H4']['inputs']))
                live.native_namespace['CACHE'][key]=dict(prefixes={3:context})
                live.h46_context=lambda *args:context
                self.assertIs(r.guarded_context(cfg,'H4',3),context)
                self.assertEqual(live.runner_namespace['initialize'].__code__.co_code,
                    old_runner.initialize.__code__.co_code)
                def fake_task(actual_cfg,actual_task,stop):
                    assert actual_cfg is cfg and actual_task==task and not stop.is_set()
                    assert r.guarded_context(actual_cfg,'H4',3) is context
                    for _ in range(12):api.build_partial_native()
                    for _ in range(2):api.build_native_hadamard_wrapper()
                    return dict(id=actual_task['id'],synthetic=True,pid=os.getpid())
                live.native.run_task=fake_task
                mp=multiprocessing.get_context('fork')
                stop=mp.Event()
                with ProcessPoolExecutor(max_workers=1,mp_context=mp,
                        initializer=live.runner_namespace['initialize'],initargs=(cfg,stop)) as pool:
                    result=pool.submit(r.worker,task).result(timeout=30)
                self.assertNotEqual(result['pid'],os.getpid())
                self.assertTrue(result['synthetic'])
                self.assertEqual(len(list((root/'ledger').glob('*.json'))),2)
                self.assertTrue(c.read(root/'output'/task['id']/'build_dispatch.json')['completed'])

    def test_paired_statistics_uses_covariance(self):
        from h246_confirmation_evaluation import paired_statistics
        x = paired_statistics([[10, 30], [20, 20], [30, 10]], 2, 2)
        self.assertEqual(x['mean'], 80)
        self.assertEqual(x['variance'], 0)
        self.assertLess(x['covariance'], 0)
        self.assertTrue(x['independent_weighted_covariance_match'])
        self.assertIsNone(x['rare_tail_bound'])

    def test_pair_validation_missing_or_wrong_cohort(self):
        from h46_standard_y4_evaluation import validate_pairs
        row = dict(system='H4', candidate_id='toy', SOURCE='toySOURCE', cohort='CONFIRMATION',
            conditional=False, paired=True, axes=['cosine', 'sine'], stratum='S0', pair=[10, 20])
        rows = [dict(row, sample=i) for i in range(2)]
        args = dict(system='H4', candidate='toy', n=2, source='toySOURCE')
        self.assertEqual(validate_pairs(rows, **args), [[10, 20], [10, 20]])
        for changes in [dict(cohort='EXPLORATION'), dict(conditional=True),
                dict(axes=['sine', 'cosine']), dict(system='H6'), dict(candidate_id='m5_best')]:
            bad = copy.deepcopy(rows)
            bad[0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_pairs(bad, **args)
        with self.assertRaises(ValueError):
            validate_pairs(rows[:1], **args)
        rows[1]['sample'] = 0
        with self.assertRaises(ValueError):
            validate_pairs(rows, **args)

    def test_direction_and_inconclusive_separate_from_point(self):
        from h46_standard_y4_evaluation import compare_statistics
        def st(mean, se):
            return dict(mean=mean, SE=se, zero_observed_variance=False)
        uncertain = compare_statistics(st(100, 20), st(90, 20), z=2)
        self.assertEqual(uncertain['point_status'], 'REPRODUCED')
        self.assertEqual(uncertain['decision'], 'INCONCLUSIVE')
        self.assertFalse(uncertain['interval_is_rigorous'])
        self.assertEqual(compare_statistics(st(100, 1), st(80, 1), z=2)['decision'], 'REPRODUCED_APPROXIMATE')
        self.assertEqual(compare_statistics(st(80, 1), st(100, 1), z=2)['decision'], 'REVERSED_APPROXIMATE')

    def test_preparation_exact_budget_with_synthetic_functions(self):
        import h46_standard_y4_runtime as r
        with tempfile.TemporaryDirectory() as d:
            fake = {name: SimpleNamespace() for name in ('h46_m5_signal_adapter',
                'h46_m5_tol_native', 'h46_new2_native', 'h46_new2_core')}
            for name, fn in [('h46_m5_signal_adapter', 'base_context'),
                    ('h46_m5_tol_native', 'restore'), ('h46_new2_native', 'OccupationOracle'),
                    ('h46_new2_native', 'prepare'), ('h46_new2_core', 'matrix_only_tail_coefficients'),
                    ('h46_new2_core', 'spectrum')]:
                setattr(fake[name], fn, lambda *args: None)
            cfg = dict(run_root=d, stage='H4_CONFIRMATION', SOURCE='a' * 40,
                plan_sha256='b' * 64, context_scope=c.context_scope(), preparation_hash='toy')
            def context(*args):
                fake['h46_m5_signal_adapter'].base_context()
                fake['h46_m5_tol_native'].restore()
                fake['h46_new2_native'].OccupationOracle()
                fake['h46_new2_native'].prepare()
                fake['h46_new2_native'].prepare()
                fake['h46_new2_core'].matrix_only_tail_coefficients()
                fake['h46_new2_core'].spectrum()
                return SimpleNamespace(cut=SimpleNamespace(preparation_hash='toy'))
            with patch.dict(sys.modules, fake), patch.object(c, 'authorize'), patch.object(c, 'consumed'), \
                    patch.object(r, 'RT', SimpleNamespace(context=context)), patch.object(r, 'WARM', False):
                receipt = r.prewarm(cfg, None)
                self.assertEqual(receipt['counted_entry_points'], c.context_scope())
                self.assertEqual(len(list((Path(d) / 'context_ledger').glob('*.json'))), 9)
                with self.assertRaisesRegex(ValueError, 'SECOND_PREPARATION'):
                    r.prewarm(cfg, None)

    def test_synthetic_complete_evaluation_nine_events_and_used_root_refusal(self):
        from h46_standard_y4_evaluation import evaluate
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            assets = root / 'assets'
            saved = assets / 'fixed/fixed_evaluation'
            saved.mkdir(parents=True)
            fake_groups = {s: dict(n=3, n_s2=2, S2=s+'S2toy', Y4=s+'Y4toy',
                N_S2=[2, 3], N_Y4=[2, 2]) for s in ['H4', 'H6']}
            sample_lines = ['system,candidate_id,sample,cosine_RZ,sine_RZ,stratum,result_sha256,SOURCE']
            configs = []
            source = 'a' * 40
            package_path = root / 'package.json'
            c.save(package_path, {'toy': True})
            for s, g in fake_groups.items():
                for i in range(2):
                    sample_lines.append(','.join([s,g['S2'],str(i),str(10+i),str(11+i),
                        'S0','b'*64,c.OLD_SOURCE]))
                run = root / s
                output = run / 'output'
                output.mkdir(parents=True)
                tasks = [dict(id=s+'toy'+str(i), sample=i) for i in range(3)]
                cfg = dict(stage=s+'_CONFIRMATION', SOURCE=source, plan_sha256='c'*64,
                    package_path=str(package_path), run_root=str(run), output=str(output),
                    tasks=tasks, marker=str(run/'CONSUMED.json'), grant={'toy':s},
                    approval_receipt_sha256='d'*64)
                c.save(cfg['marker'], dict(state='CONSUMED_NO_RETRY_NO_REFUND',grant=cfg['grant']))
                records = []
                for i,t in enumerate(tasks):
                    out = output / t['id']
                    out.mkdir()
                    r = dict(task=t,SOURCE=source,plan_sha256=cfg['plan_sha256'],
                        status='UNIT_COST_PASS_NOT_ACCOUNTING',seed_cohort='CONFIRMATION',
                        paired_trajectory=True,conditional_sampling=False,cohort_pooling=False,
                        events=dict(stratum='S0'),axis_costs=[dict(axis=axis,
                            metrics=dict(rz_count=8+i)) for axis in ['cosine','sine']])
                    c.save(out/'result.json',r)
                    records.append(dict(id=t['id'],result_sha256=c.sha(out/'result.json')))
                c.save(run/'end_audit.json',dict(status='ACQUISITION_AND_END_AUDIT_PASS_MANDATORY_STOP',
                    errors={},SOURCE=source,plan_sha256=cfg['plan_sha256'],completed_trajectories=3,
                    accepted_compile_axes=6,results=records))
                c.save(run/'context_end_audit.json',dict(status='CONTEXT_AND_ACQUISITION_AUDIT_PASS',
                    errors={},end_audit_sha256=c.sha(run/'end_audit.json')))
                cp = root/(s+'.json')
                c.save(cp,cfg)
                configs.append(str(cp))
            (saved/'paired_samples_118.csv').write_text('\n'.join(sample_lines)+'\n')
            endpoint = root/'artifacts/resource_applicability/track_a_h246_reversal_batch_run01/2026-10-24'
            endpoint.mkdir(parents=True)
            (endpoint/'primary_endpoints_60.csv').write_text('system,epsilon,G\nH4,0.001,123\nH6,0.001,456\n')
            package = dict(assets=str(assets),evaluation_root=str(root/'evaluation'),SOURCE=source,
                input_manifest_sha256='e'*64)
            with patch.object(c,'verify_package',return_value=package), patch.object(c,'authorize'), \
                    patch.object(c,'GROUPS',fake_groups),patch.object(c,'ROOT',root):
                result = evaluate(package_path,configs)
                self.assertEqual(len(list((root/'evaluation/ledger').glob('*.json'))),9)
                self.assertEqual(result['results']['H4']['groups']['S2']['statistics']['mean'],55.5)
                self.assertEqual(result['results']['H4']['groups']['Y4']['statistics']['mean'],36)
                self.assertTrue((root/'evaluation/STOP.json').exists())
                self.assertFalse(result['cohort_pooling'])
                with self.assertRaisesRegex(ValueError,'NO_EVALUATION_RETRY_OR_USED_GRANT'):
                    evaluate(package_path,configs)


if __name__ == '__main__':
    unittest.main()
