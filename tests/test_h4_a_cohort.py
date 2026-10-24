"""Artificial authority, dispatch, failure and finite-cohort statistics tests."""
from contextlib import ExitStack
import copy
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_h4_a_gate import A, fixture
from d1_private import h4_a_cohort as D
from qiskit import QuantumCircuit
import numpy as np
C = D.C


def active():
    return dict(kind=D.KIND, fixed=D.fixed(), execution_authorized=True, grant_valid=True, dispatch_enabled=True,
                new_indices=list(range(2, 256)), reuse_indices=[0, 1], call_limits=D.LIMITS,
                axis='cosine', per_trajectory_compile_max=1, mandatory_STOP=True,
                automatic_retry=False, automatic_resume=False, automatic_next_stage=False,
                job_environment=D.JOB_ENV, worker_count=3, science_SOURCE='TOY', approval_sha256='TOY')


def task_files(base):
    ns = base / 'run01'
    (ns / 'trajectories').mkdir(parents=True)
    folder = ns / 'native_tasks/compile_0002'
    folder.mkdir(parents=True)
    C.save(folder / 'reservation.json', dict(kind='compile', case='0002', grant_id='TOY'))
    return ns


def synthetic_cohort(root):
    """Integer price fixtures, not saved H4 inputs or observed science."""
    ns = root / 'run'
    ns.mkdir()
    old = root / 'old'
    d2 = root / 'd2'
    rows, reused = [], []
    (root / 'joins').mkdir()
    for i in range(256):
        row = dict(trajectory_index=i, trajectory_id='ARTIFICIAL_' + str(i), event_record_hash='E' + str(i),
                   trajectory_record_hash='T' + str(i), price_join='joins/' + str(i) + '.json')
        rows.append(row)
        C.save(root / row['price_join'], dict(compiled_RZ=dict(cosine=9321 + (i < 26)), events=[dict(taylor_order=2 if i % 4 == 0 else 0)]))
        metrics = dict(RZ=1000+i % 2, CX=20, depth=30, rz_depth=10, cx_depth=15, SOURCE='TOY',
                       compiler=D.fixed()['compiler'], target=row)
        if i < 2:
            reused.append(dict(metrics=metrics, folder='REUSED_ARTIFICIAL'))
        else:
            folder = ns / 'trajectories' / f'{i:04d}'
            folder.mkdir(parents=True)
            for name in ('input.qpy', 'output.qpy'):
                (folder / name).write_bytes(b'ARTIFICIAL_NOT_QPY')
            metrics.update(input_qpy_sha256=C.sha(folder / 'input.qpy'), output_qpy_sha256=C.sha(folder / 'output.qpy'))
            C.save(folder / 'metrics.json', metrics)
            native = ns / 'native_tasks' / ('compile_' + f'{i:04d}')
            native.mkdir(parents=True)
            C.save(native / 'native_exit.json', dict(exit_code=0, reaped=True, native_wait=True))
    for case, mean in [('pauli_jw_PR', 9321.1015625), ('df_native_PF', 135355), ('df_native_PR', 23678.44140625)]:
        folder = old / case
        folder.mkdir(parents=True)
        C.save(folder / 'result.json', dict(circuit_cost=dict(axes=dict(cosine=dict(mean=mean, mean_exact=str(Fraction(mean)))))))
    folder = d2 / 'run01/variants/pauli_jw_PF'
    folder.mkdir(parents=True)
    C.save(folder / 'metrics.json', dict(RZ=15028))
    return ns, old, d2, rows, reused


class CohortTests(unittest.TestCase):
    def test_a_no_authority_no_quantum_import(self):
        code = """import sys,types,tempfile
from pathlib import Path
p=types.ModuleType('d1_private');p.__path__=[str(Path.cwd()/'src/trotterlib')];sys.modules['d1_private']=p
from d1_private import h4_a_cohort as D
with tempfile.TemporaryDirectory() as t:
 D.BASE=Path(t)
 try:D.authority(consume=True)
 except FileNotFoundError:pass
 else:raise AssertionError('authorized')
assert not any(k=='qiskit' or k.startswith('qiskit.') for k in sys.modules)
"""
        self.assertEqual(subprocess.run([sys.executable, '-B', '-c', code], cwd=D.ROOT).returncode, 0)

    def test_b_scope_blocks_sine_reused_index_more_calls_and_auto_stage(self):
        D.scope(active())
        for key, value in [('axis', 'sine'), ('new_indices', list(range(256))), ('reuse_indices', []),
                           ('call_limits', {}), ('automatic_resume', True), ('automatic_next_stage', True), ('worker_count', 0)]:
            c = active()
            c[key] = value
            with self.assertRaises(ValueError):
                D.scope(c)

    def test_c_unique_complete_cohort_required(self):
        rows = [dict(trajectory_index=i, trajectory_id=str(i)) for i in range(256)]
        with patch.object(C, 'read', return_value=dict(records=rows)):
            self.assertEqual(len(D.records()), 256)
            rows[2]['trajectory_id'] = '1'
            with self.assertRaisesRegex(ValueError, 'UNIQUE'):
                D.records()
            rows.pop()
            with self.assertRaisesRegex(ValueError, 'INDICES'):
                D.records()

    def test_d_new_grant_binding_consumption_and_STOP(self):
        with tempfile.TemporaryDirectory() as td, patch.object(D, 'BASE', Path(td)), patch.object(D, 'verify_seal'):
            b = Path(td)
            ns = b / 'run01'
            ns.mkdir()
            C.save(b / 'seal.json', dict(SOURCE='TOY', approval_sha256='TOY'))
            c = active()
            c['seal_sha256'] = C.sha(b / 'seal.json')
            C.save(b / 'active_contract.json', c)
            g = dict(kind='H4_A_COHORT_COS_ONE_SHOT', SOURCE='TOY', namespace=str(ns), call_limits=D.LIMITS,
                     contract_sha256=C.sha(b / 'active_contract.json'), seal_sha256=c['seal_sha256'],
                     approval_sha256='TOY', one_shot=True, automatic_retry=False)
            g['grant_id'] = C.digest(g)
            C.save(ns / 'grant.json', g)
            D.authority(consume=True)
            with self.assertRaisesRegex(ValueError, 'CONSUMED'):
                D.authority(consume=True)
            C.save(ns / 'ENGINE_STOP.json', dict(status='TOY_STOP'))
            with self.assertRaisesRegex(ValueError, 'STOPPED'):
                D.authority()

    def test_e_budget_last_reservation_nonrefundable_and_zero_scopes(self):
        with tempfile.TemporaryDirectory() as td:
            ns = Path(td)
            used = dict.fromkeys(D.LIMITS, 0)
            used['cost_transpile'] = 253
            C.save(ns / 'call_ledger.json', dict(limits=D.LIMITS, used=used, events=[]))
            budget = C.CallBudget(ns, D.LIMITS)
            budget.take('cost_transpile')
            for name in ('cost_transpile', 'sampling', 'sine_compile', 'N_G'):
                with self.assertRaises(ValueError):
                    budget.take(name)
            budget.take('cost_statistics_batch')
            with self.assertRaises(ValueError):
                budget.take('cost_statistics_batch')

    def test_f_reused_dispatcher_reserves_before_each_child_and_reaps(self):
        commands = []
        with tempfile.TemporaryDirectory() as td:
            ns = Path(td)
            C.save(ns / 'grant.json', dict(grant_id='TOY'))
            class Child:
                pid = os.getpid()
                def __init__(self, command, **kw):
                    commands.append(command)
                    case = command[-1]
                    assert C.read(ns / 'native_tasks' / ('compile_' + case) / 'reservation.json')['grant_id'] == 'TOY'
                def poll(self): return 0
                def wait(self): return 0
            with patch.object(D2 := D.D2, 'subprocess', SimpleNamespace(Popen=Child)):
                D.dispatch(ns, ('0002', '0003'))
            self.assertEqual([x[-1] for x in commands], ['0002', '0003'])
            self.assertEqual(C.read(ns / 'call_ledger.json')['used']['cost_transpile'], 2)
            self.assertEqual(len(list((ns / 'native_tasks').glob('*/native_exit.json'))), 2)

    def test_g_worker_failure_stops_future_waves_and_statistics(self):
        with tempfile.TemporaryDirectory() as td, patch.object(D, 'BASE', Path(td)):
            ns = Path(td) / 'run01'
            ns.mkdir()
            with (patch.object(D, 'authority', return_value=(active(), {}, {})), patch.object(D, 'reuse_results', return_value=[]),
                  patch.object(D, 'dispatch', side_effect=ValueError('ARTIFICIAL_FAILURE')) as dispatcher,
                  patch.object(D, 'statistics_batch') as stats):
                with self.assertRaises(ValueError):
                    D.worker()
                self.assertEqual(dispatcher.call_count, 1)
                stats.assert_not_called()
            self.assertFalse(C.read(ns / 'ENGINE_STOP.json')['exact_256_mean_complete'])

    def test_h_index_and_reservation_gate_before_builder(self):
        with tempfile.TemporaryDirectory() as td, patch.object(D, 'BASE', Path(td)):
            task_files(Path(td))
            with patch.object(D, 'authority', return_value=(active(), {}, dict(grant_id='OTHER'))), patch.object(A, 'api') as api:
                with self.assertRaisesRegex(ValueError, 'RESERVATION'):
                    D.task_run('0002')
                with self.assertRaisesRegex(ValueError, 'INDEX'):
                    D.task_run('0000')
                api.assert_not_called()

    def test_i_source_mismatch_stops_control_wrapper_and_compiler(self):
        with tempfile.TemporaryDirectory() as td, patch.object(D, 'BASE', Path(td)):
            ns = task_files(Path(td))
            model, join = fixture()
            body = QuantumCircuit(8)
            body.x(0)
            inspector = SimpleNamespace(pauli_recipe=lambda *args: (['different'], []), primitive_token=lambda *args: 'actual')
            original_read = C.read
            def read(path):
                return model if Path(path) == D.ROOT / D.G.MODEL else original_read(path)
            with (patch.object(D, 'authority', return_value=(active(), dict(SOURCE='TOY'), dict(grant_id='TOY'))),
                  patch.object(C, 'read', side_effect=read), patch.object(D, 'validate_join', return_value=join),
                  patch.object(D.G, 'body_only', return_value=SimpleNamespace(body=body, sidecar={})),
                  patch.object(D.D2.S, 'inspector', return_value=inspector),
                  patch('d1_private.h4_d2_pauli_control.central_control_body') as central,
                  patch.object(A, 'compile_saved') as compiler):
                with self.assertRaisesRegex(ValueError, 'SOURCE_TOKENS'):
                    D.task_run('0002')
                central.assert_not_called()
                compiler.assert_not_called()
            used = C.read(ns / 'call_ledger.json')['used']
            self.assertEqual(used['body_replay'], 1)
            self.assertEqual(used['central_control_build'], 0)
            with patch.object(D, 'authority', return_value=(active(), {}, dict(grant_id='TOY'))):
                with self.assertRaises(FileExistsError):
                    D.task_run('0002')

    def test_j_statistics_known_integer_oracle_and_nominal_t_interval(self):
        from d1_private.h4_conditional_cng import extract
        stats = extract('src/trotterlib/fixed_signal_evaluation.py', '_statistics', dict(np=np, math=math, Fraction=Fraction))
        value = D.summarize([0, 0, 0, 4], stats, np)
        self.assertEqual(value['mean_exact'], '1')
        self.assertEqual(value['sample_standard_deviation'], 2)
        self.assertEqual(value['standard_error'], 1)
        self.assertEqual(value['quantiles']['0.5'], 0)
        self.assertAlmostEqual(value['nominal95_halfwidth'], 3.1824463053, places=8)

    def test_k_artificial_256_same_ID_paired_summary_and_absent_stratum(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ns, old, d2, rows, reused = synthetic_cohort(root)
            with patch.object(D, 'ROOT', root), patch.object(D, 'OLD_RUN', old), patch.object(D.D2, 'BASE', d2), patch.object(D, 'records', return_value=rows), patch.object(D, 'plot_costs'):
                # Preserve the real fixed profile only as inert compiler metadata.
                with patch.object(D, 'fixed', return_value=active_fixed):
                    result = D.statistics_batch(ns, reused)
            self.assertEqual(result['central_cosine_RZ']['mean_exact'], '2001/2')
            self.assertEqual(result['new_successes'], 254)
            self.assertFalse(result['cross_representation_pairing'])
            self.assertEqual(result['order_groups']['contains_order4']['n'], 0)
            self.assertIsNone(result['order_groups']['contains_order4']['central_RZ'])
            self.assertEqual(len(C.read(ns / 'cohort_costs.json')['rows']), 256)

    def test_l_missing_or_failed_native_result_prevents_exact_cohort_mean(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ns, old, d2, rows, reused = synthetic_cohort(root)
            native = ns / 'native_tasks/compile_0002/native_exit.json'
            C.save(native, dict(exit_code=1, reaped=True, native_wait=True), exclusive=False)
            with patch.object(D, 'ROOT', root), patch.object(D, 'OLD_RUN', old), patch.object(D.D2, 'BASE', d2), patch.object(D, 'records', return_value=rows), patch.object(D, 'fixed', return_value=active_fixed), patch.object(D, 'plot_costs'):
                with self.assertRaisesRegex(ValueError, 'STATS_NATIVE'):
                    D.statistics_batch(ns, reused)
            self.assertFalse((ns / 'statistics.json').exists())


active_fixed = D.fixed()
if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CohortTests))
    if len(sys.argv) == 2:
        import resource
        Path(sys.argv[1]).write_text(json.dumps(dict(tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, scope='ARTIFICIAL_ONLY_NO_REAL_H4'), indent=2) + '\n')
    sys.exit(not result.wasSuccessful())
