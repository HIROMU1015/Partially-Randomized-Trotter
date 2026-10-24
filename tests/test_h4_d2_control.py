"""Ten artificial authority/pipeline tests. No real H4 scientific calls."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_h4_d1_adapter import A, toy
from d1_private import h4_d2_control as D
C = D.C


def active():
    c = C.read(D.PREP/'contract_DRAFT.json')
    c.update(kind=D.KIND, execution_authorized=True, dispatch_enabled=True, grant_valid=True,
             call_limits=D.LIMITS, science_SOURCE='TOY_SOURCE', seal_sha256='TOY', approval_sha256='TOY_APPROVAL')
    return c


class ControlTests(unittest.TestCase):
    def test_a_authority_failure_precedes_quantum_import(self):
        code = """import sys,types
from pathlib import Path
p=types.ModuleType('d1_private');p.__path__=[str(Path.cwd()/'src/trotterlib')];sys.modules['d1_private']=p
from d1_private import h4_d2_control as D
import tempfile
with tempfile.TemporaryDirectory() as t:
 D.BASE=Path(t)
 try:D.authority(consume=True)
 except FileNotFoundError:pass
 else:raise AssertionError('authority accepted')
assert not any(k=='qiskit' or k.startswith('qiskit.') for k in sys.modules)
"""
        self.assertEqual(subprocess.run([sys.executable, '-B', '-c', code], cwd=D.ROOT).returncode, 0)

    def test_b_scope_refuses_flag_budget_and_target_changes(self):
        D.scope(active())
        for name, value in [('execution_authorized', False), ('automatic_D3', True), ('call_limits', {}), ('targets', [])]:
            c = active();c[name] = value
            with self.assertRaises(ValueError):D.scope(c)

    def fixture(self, base):
        ns = base/'run01';ns.mkdir()
        s = dict(SOURCE='TOY_SOURCE', approval_sha256='TOY_APPROVAL')
        C.save(base/'seal.json', s)
        c = active();c['seal_sha256'] = C.sha(base/'seal.json')
        C.save(base/'active_contract.json', c)
        g = dict(kind='H4_D2_SCIENCE_ONE_SHOT', SOURCE='TOY_SOURCE', namespace=str(ns),
            contract_sha256=C.sha(base/'active_contract.json'), seal_sha256=C.sha(base/'seal.json'),
            approval_sha256='TOY_APPROVAL', one_shot=True, automatic_retry=False, call_limits=D.LIMITS)
        g['grant_id'] = C.digest(g)
        C.save(ns/'grant.json', g)
        return ns, c, s, g

    def test_c_single_use_grant_binding_and_STOP(self):
        with tempfile.TemporaryDirectory() as td, patch.object(D, 'BASE', Path(td)), patch.object(D, 'verify_seal'):
            ns, c, s, g = self.fixture(Path(td))
            with self.assertRaises(FileNotFoundError):D.authority(task=True)
            D.authority(consume=True)
            with self.assertRaisesRegex(ValueError, 'CONSUMED'):D.authority(consume=True)
            C.save(ns/'LAUNCH_CLAIM.json', {})
            D.authority(task=True)
            C.save(ns/'ENGINE_STOP.json', dict(status='TOY_STOP'))
            with self.assertRaisesRegex(ValueError, 'STOPPED'):D.authority(task=True)
        with tempfile.TemporaryDirectory() as td, patch.object(D, 'BASE', Path(td)), patch.object(D, 'verify_seal'):
            ns, c, s, g = self.fixture(Path(td))
            g['SOURCE'] = 'TAMPERED'
            C.save(ns/'grant.json', g, exclusive=False)
            with self.assertRaisesRegex(ValueError, 'DIGEST'):D.authority()

    def test_d_call_budget_is_nonrefundable(self):
        with tempfile.TemporaryDirectory() as td:
            b = C.CallBudget(td, D.LIMITS)
            b.take('cost_transpile');b.take('cost_transpile')
            with self.assertRaisesRegex(ValueError, 'EXHAUSTED'):b.take('cost_transpile')
            with self.assertRaises(ValueError):b.take('sampling')
            self.assertEqual(C.read(Path(td)/'call_ledger.json')['used']['cost_transpile'], 2)

    def test_e_saved_body_SHA_failure_never_loads_QPY(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td)/'toy.qpy';source.write_bytes(b'TOY')
            api = unittest.mock.Mock()
            with self.assertRaisesRegex(ValueError, 'SAVED_BODY_SHA'):
                D.validate_body(dict(body_qpy=str(source), body_qpy_sha256='0'*64), Path(td), api)
            api.qpy.load.assert_not_called()

    def pipeline(self, fail=None):
        with tempfile.TemporaryDirectory() as td, patch.object(D, 'BASE', Path(td)):
            ns = Path(td)/'run01';ns.mkdir();calls=[]
            def batch(namespace, kind, cases):
                calls.append((kind, list(cases)))
                if kind == fail:raise ValueError('TOY_GATE_FAILURE')
                if kind == 'compile':
                    for case in cases:
                        folder = namespace/'variants'/case;folder.mkdir()
                        C.save(folder/'metrics.json', dict(RZ=7))
            with patch.object(D, 'authority', return_value=(active(), {}, {})), patch.object(D, 'batch', side_effect=batch), patch.object(D, 'comparisons', return_value={}):
                if fail:
                    with self.assertRaises(ValueError):D.worker()
                else:self.assertEqual(D.worker(), 0)
            return calls, C.read(ns/'ENGINE_STOP.json')

    def test_f_failed_semantic_or_build_stage_blocks_compile(self):
        for failed, expected in [('validate', ['validate']), ('build', ['validate', 'build'])]:
            calls, stop = self.pipeline(failed)
            self.assertEqual([k for k, _ in calls], expected)
            self.assertEqual(stop['status'], 'D2_FAILED_NO_RETRY')

    def test_g_both_targets_precede_each_next_stage(self):
        calls, stop = self.pipeline()
        self.assertEqual(calls, [(kind, list(D.CASES)) for kind in ('validate', 'build', 'compile')])
        self.assertEqual(stop['cost_compiles'], 2)

    def task_fixture(self, base, kind):
        ns = base/'run01';folder = ns/'variants'/'pauli_jw_PR';folder.mkdir(parents=True)
        C.save(ns/'stage_A.json', dict(status='PASS_BOTH_SAVED_BODY_SOURCE_B'))
        C.save(ns/'stage_B.json', dict(status='PASS_BOTH_CONTROLLED_INPUTS_SAVED'))
        d = ns/'native_tasks'/(kind+'_pauli_jw_PR');d.mkdir(parents=True)
        C.save(d/'reservation.json', dict(kind=kind, case='pauli_jw_PR', grant_id='TOY_GRANT'))
        return ns, folder

    def test_h_toy_build_from_saved_body_never_compiles_or_rebuilds(self):
        a = A.api();model, join = toy('pauli_jw', 'PR');source = A.build(model, 'PR', join).body
        with tempfile.TemporaryDirectory() as td, patch.object(D, 'BASE', Path(td)):
            ns, folder = self.task_fixture(Path(td), 'build')
            with (folder/'source_body.qpy').open('wb') as f:a.qpy.dump(source, f)
            C.save(folder/'validation.json', dict(body_qpy_sha256=C.sha(folder/'source_body.qpy')))
            with patch.object(D, 'authority', return_value=(active(), dict(SOURCE='TOY'), dict(grant_id='TOY_GRANT'))), patch.object(A, 'build', side_effect=AssertionError('REBUILD')), patch.object(A, 'compile_saved', side_effect=AssertionError('COMPILE')):
                self.assertEqual(D.task_run('build', 'pauli_jw_PR'), 0)
            data = C.read(folder/'input.json')
            self.assertTrue(data['new_input_need_not_equal_generic'])
            self.assertFalse(data['historical_hash_match'])
            self.assertEqual(data['sidecar']['aggregate_source_global_phase'], float(source.global_phase))

    def test_i_tampered_input_blocks_cost_compile(self):
        with tempfile.TemporaryDirectory() as td, patch.object(D, 'BASE', Path(td)):
            ns, folder = self.task_fixture(Path(td), 'compile')
            (folder/'input.qpy').write_bytes(b'TAMPERED')
            C.save(folder/'input.json', dict(input_qpy_sha256='0'*64))
            with patch.object(D, 'authority', return_value=(active(), {}, dict(grant_id='TOY_GRANT'))), patch.object(A, 'compile_saved') as compile_call:
                with self.assertRaisesRegex(ValueError, 'LOCAL_INPUT_CHANGED'):D.task_run('compile', 'pauli_jw_PR')
                compile_call.assert_not_called()

    def test_j_signed_comparison_preserves_negative_and_denominator(self):
        fields = ('RZ', 'CX', 'depth', 'rz_depth', 'cx_depth')
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'before.json';C.save(p, dict.fromkeys(fields, 10))
            rows = D.comparisons({'TOY':dict.fromkeys(fields, 7)}, [dict(case='TOY', D1_metrics=str(p))])
            self.assertEqual(rows['TOY']['RZ']['signed_central_minus_generic'], -3)
            self.assertEqual(rows['TOY']['RZ']['central_over_generic_exact'], '7/10')


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ControlTests))
    if len(sys.argv) == 2:
        Path(sys.argv[1]).write_text(json.dumps(dict(tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
            scope='ARTIFICIAL_AUTHORITY_PIPELINE_ONLY', real_H4_calls=0), indent=2)+'\n')
    sys.exit(not result.wasSuccessful())
