"""Focused off-domain source tests. Never solve the fixed eight LPs or run audit()."""
import copy
import ctypes
from fractions import Fraction as F
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'scripts/tracks/algorithm_codesign'))
from g1_decision_packet import controller as c
from g1_decision_packet import structure as structure
from g1_decision_packet.rational_symbolic import RF, determinant, parse, sign_proof, solve_many

spec = importlib.util.spec_from_file_location('g1_unchanged_fraction_verifier',
    ROOT/'scripts/tracks/algorithm_codesign/exact_backend_pilot_v2/verify.py')
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def problem(name='MOCK_0', infeasible=False):
    return {'id': name, 'c0': '2/7', 'c': ['1', '2'], 'U': ['3', '3'],
            'A': [['1', '1']] if infeasible else [['-1', '0']],
            'b': ['1/2'] if infeasible else ['0'], 'H': [['1', '1']], 'f': ['1'],
            'expected_status': 'INFEASIBLE' if infeasible else 'OPTIMAL'}


def echo(p):
    return {**copy.deepcopy({k: p[k] for k in ('c0', 'c', 'U', 'A', 'b', 'H', 'f')}),
            'lower': ['0']*len(p['c']), 'equal_lhs': p['f']}


def output(p, only_echo=False):
    result = {'echo': echo(p), 'status': 'ECHO_ONLY' if only_echo else p['expected_status']}
    if only_echo: return result
    if p['expected_status'] == 'INFEASIBLE':
        return {**result, 'farkas_available': True, 'raw_row_farkas': ['-1', '1']}
    return {**result, 'primal_available': True, 'dual_available': True, 'reduced_cost_available': True,
            'primal': ['1', '0'], 'raw_row_dual': ['0', '1'], 'raw_reduced_cost': ['0', '1'],
            'objective_with_exact_external_offset': '9/7', 'backend_objective_without_offset': '1'}


def good_record(name):
    return {'id': name, 'failure': None, 'returncode': 0, 'residual_processes': 0}


class FakeTransport:
    def __init__(self, change=None, audit_status=c.PASS_A, failure=None):
        self.events, self.change, self.audit_status, self.failure = [], change, audit_status, failure

    def check(self): pass
    def final_check(self): pass

    def audit(self, store):
        self.events.append('MOCK_AUDIT')
        return good_record('MOCK_AUDIT'), {'classification': self.audit_status}

    def acquire(self, p, echo):
        kind = 'ECHO' if echo else 'MOCK_SOLVE'
        self.events.append((kind, p['id']))
        result, record = output(p, only_echo=echo), good_record(kind+'_'+p['id'])
        if not echo and p['id'] == 'MOCK_0':
            if self.change: self.change(result)
            if self.failure: record.update(failure=self.failure, returncode=-9)
        return record, result

    def verify(self, p, result, echo):
        self.events.append(('FRACTION_ONLY_VERIFY', p['id'], echo))
        verdict = verifier.verify(p, result)
        record = good_record('verify_'+p['id'])
        if not verdict['PASS']: record.update(failure='PROCESS_EXIT_FAILURE', returncode=1)
        return record, verdict


class AlgebraKernelTests(unittest.TestCase):
    def test_cancel_generic_rational_function(self):
        z = RF.variable()
        self.assertEqual((z*z-1)/(z-1), z+1)

    def test_fraction_coefficients_and_zero(self):
        z = RF.variable()
        self.assertEqual((z/F(7, 9))*F(7, 9)-z, 0)

    def test_power_and_evaluation(self):
        z = RF.variable()
        self.assertEqual((1/(1+z))**2, RF(1)/(1+2*z+z*z))
        self.assertEqual((z*z+1).evaluate(F(3, 5)), F(34, 25))

    def test_composition(self):
        z = RF.variable()
        self.assertEqual((z*z+z).compose(1-z), (1-z)**2+(1-z))

    def test_zero_division(self):
        with self.assertRaises(ZeroDivisionError): RF([1], [0])

    def test_generic_symbolic_linear_system(self):
        z = RF.variable()
        result = solve_many([[z, 1], [1, 0]], [[1], [2]])
        self.assertEqual(result, [[RF(2)], [1-2*z]])

    def test_symbolic_minor_nonzero_sign(self):
        z = RF.variable()
        value = determinant([[z, 1, 0], [0, 2, 1], [0, 0, 3]])
        self.assertEqual(value, 6*z)
        self.assertEqual(sign_proof(value, 'positive')['sign'], 1)

    def test_rank_deficiency_is_not_unique_solution(self):
        with self.assertRaises(ArithmeticError): solve_many([[1, 1], [2, 2]], [[1], [2]])

    def test_positive_domain_sign_is_exact(self):
        z = RF.variable()
        self.assertEqual(sign_proof(z*z+z, 'positive')['sign'], 1)

    def test_bernstein_interior_sign(self):
        z = RF.variable()
        self.assertEqual(sign_proof(z*(1-z), 'unit')['sign'], 1)
        self.assertEqual(sign_proof(z-1, 'unit')['sign'], -1)

    def test_mixed_sign_is_unproved(self):
        self.assertIsNone(sign_proof(RF.variable()-F(1, 2), 'unit')['sign'])

    def test_parser_is_restricted(self):
        with self.assertRaises(ValueError): parse('__import__("os")')

    def test_parser_fraction(self):
        z = RF.variable()
        self.assertEqual(parse('F(3,5)+z**2', z=z), F(3, 5)+z*z)

    def test_phase_convention_parser_on_artificial_text(self):
        with tempfile.TemporaryDirectory(prefix='g1_phase_mock_') as folder:
            p = Path(folder)/'phase.py'
            p.write_text('if e["phase_i_power"] != (-degree)%4: pass\n')
            self.assertTrue(structure.phase_convention_present(p))
            p.write_text('if e["phase_i_power"] != degree%4: pass\n')
            self.assertFalse(structure.phase_convention_present(p))

    def test_ast_coefficients_from_artificial_text_only(self):
        with tempfile.TemporaryDirectory(prefix='g1_ast_mock_') as folder:
            path = Path(folder)/'toy_source.py'
            path.write_text('rho=x/(1+x)\nformulas={"u":(0,F(1),rho),"v":(2,x,F(0))}\n')
            z, rho, columns = structure.source_vectors(path, ['u', 'v'])
            self.assertEqual(columns[0], [RF(1), z/(1+z), RF(0), RF(0)])
            self.assertEqual(columns[1], [RF(0), RF(0), z, RF(0)])

class PayloadTests(unittest.TestCase):
    def test_valid_primal_dual(self):
        p = problem()
        self.assertEqual(c.payload_gate(p, output(p)), (None, None))
        self.assertEqual(c.verdict_gate(verifier.verify(p, output(p))), (None, None))

    def test_valid_farkas(self):
        p = problem(infeasible=True)
        self.assertEqual(c.payload_gate(p, output(p)), (None, None))
        self.assertEqual(c.verdict_gate(verifier.verify(p, output(p))), (None, None))

    def test_error_is_acquisition_failure(self):
        p = problem(); r = {'echo': echo(p), 'status': 'ERROR'}
        self.assertEqual(c.payload_gate(p, r)[0], c.ACQUIRE)

    def test_unknown_status_is_acquisition_failure(self):
        p = problem(); r = {'echo': echo(p), 'status': 'ABORTED'}
        self.assertEqual(c.payload_gate(p, r)[0], c.ACQUIRE)

    def test_missing_payload_is_not_invalid_proof(self):
        p = problem(); r = output(p); r['dual_available'] = False
        self.assertEqual(c.payload_gate(p, r)[0], c.ACQUIRE)

    def test_missing_reduced_cost_is_acquisition_failure(self):
        p = problem(); r = output(p); del r['reduced_cost_available']
        self.assertEqual(c.payload_gate(p, r)[0], c.ACQUIRE)

    def test_recognized_wrong_status(self):
        p = problem(); r = output(problem(infeasible=True)); r['echo'] = echo(p)
        self.assertEqual(c.payload_gate(p, r)[0], 'G1_FIXTURE_STATUS_INCONCLUSIVE')

    def test_wrong_readback_precedes_status(self):
        p = problem(); r = output(p); r['status'] = 'ERROR'; r['echo']['c0'] = '0'
        self.assertEqual(c.payload_gate(p, r)[0], c.TECH)

    def test_malformed_payload_is_technical(self):
        p = problem(); r = output(p); r['primal'] = ['1']
        self.assertEqual(c.payload_gate(p, r), (c.TECH, 'PAYLOAD_FORMAT'))

    def test_float_is_not_exact_payload(self):
        p = problem(); r = output(p); r['primal'] = [1.0, 0.0]
        self.assertEqual(c.payload_gate(p, r), (c.TECH, 'PAYLOAD_FORMAT'))

    def test_noncanonical_rational_is_rejected(self):
        p = problem(); r = output(p); r['primal'] = ['2/2', '0']
        self.assertEqual(c.payload_gate(p, r), (c.TECH, 'PAYLOAD_FORMAT'))

    def test_bad_certificate_gets_math_classification(self):
        p = problem(); r = output(p); r['primal'] = ['2', '0']
        self.assertEqual(c.payload_gate(p, r), (None, None))
        verdict = verifier.verify(p, r)
        self.assertEqual(c.verdict_gate(verdict)[0], 'G1_BACKEND_INVALID_CERTIFICATE')

    def test_valid_nonzero_gap_is_acquisition_not_invalid(self):
        p = problem(); r = output(p); r['raw_row_dual'] = ['0', '0']; r['raw_reduced_cost'] = ['1', '2']
        verdict = verifier.verify(p, r)
        self.assertTrue(verdict['PASS'])
        self.assertEqual(c.verdict_gate(verdict)[0], c.ACQUIRE)

    def test_finite_box_correction_kept(self):
        p = {'id': 'MOCK_BOX', 'c0': '0', 'c': ['-1'], 'U': ['3/5'], 'A': [], 'b': [], 'H': [], 'f': [], 'expected_status': 'OPTIMAL'}
        r = {'echo': echo(p), 'status': 'OPTIMAL', 'primal_available': True, 'dual_available': True, 'reduced_cost_available': True,
             'primal': ['3/5'], 'raw_row_dual': [], 'raw_reduced_cost': ['-1'], 'backend_objective_without_offset': '-3/5', 'objective_with_exact_external_offset': '-3/5'}
        v = verifier.verify(p, r)
        self.assertTrue(v['PASS']); self.assertEqual(v['dual']['box_correction'], '-3/5')

    def test_verifier_exit_one_with_verdict_not_process_unknown(self):
        record = {'failure': 'PROCESS_EXIT_FAILURE', 'returncode': 1, 'residual_processes': 0}
        self.assertEqual(c.verifier_guard_failure(record, {'PASS': False}), (None, None))
        self.assertEqual(c.verifier_guard_failure(record, None)[0], c.TECH)

    def test_guard_resource_failure_wins(self):
        record = {'failure': 'RSS_CAP', 'returncode': -9, 'residual_processes': 0}
        self.assertEqual(c.verifier_guard_failure(record, {'PASS': False})[0], 'G1_RESOURCE_INCONCLUSIVE')


class StateMachineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='g1_state_mock_')
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.contract = {'LP_order': [f'MOCK_{i}' for i in range(8)], 'LP_call_cap': 8}
        self.fixtures = {key: problem(key, infeasible=i % 3 == 2) for i, key in enumerate(self.contract['LP_order'])}
        self.store = c.Store(self.folder/'private', self.folder/'artifact')

    def run_packet(self, transport):
        return c.execute_packet(self.contract, self.fixtures, self.store, transport,
                                {'off_domain_mock_test': True, 'explicit_one_shot_instruction_bound': False})

    def test_complete_mock_order_echoes_before_solves(self):
        t = FakeTransport(); r = self.run_packet(t)
        self.assertEqual(r['classification'], c.PASS_B)
        self.assertEqual((r['structure_audit_calls'], r['echo_only_calls'], r['LP_calls'], r['verification_calls']), (1, 8, 8, 16))
        first = next(i for i, event in enumerate(t.events) if isinstance(event, tuple) and event[0] == 'MOCK_SOLVE')
        self.assertEqual(sum(isinstance(e, tuple) and e[0] == 'ECHO' for e in t.events[:first]), 8)
        self.assertTrue(self.store.marker.exists()); self.assertTrue(self.store.stop.exists())

    def test_structural_counterexample_blocks_all_backend_calls(self):
        r = self.run_packet(FakeTransport(audit_status='G1_STRUCTURE_COUNTEREXAMPLE'))
        self.assertEqual(r['classification'], 'G1_STRUCTURE_COUNTEREXAMPLE')
        self.assertEqual(r['LP_calls'], 0); self.assertEqual(r['echo_only_calls'], 0)

    def test_error_stops_no_verifier_or_suffix(self):
        t = FakeTransport(change=lambda out: out.update(status='ERROR'))
        r = self.run_packet(t)
        self.assertEqual(r['classification'], c.ACQUIRE)
        self.assertEqual(r['LP_calls'], 1); self.assertEqual(r['verification_calls'], 8)
        self.assertFalse(any(e == ('MOCK_SOLVE', 'MOCK_1') for e in t.events))

    def test_bad_proof_stops_even_when_verifier_exits_one(self):
        r = self.run_packet(FakeTransport(change=lambda out: out.update(primal=['2', '0'])))
        self.assertEqual(r['classification'], 'G1_BACKEND_INVALID_CERTIFICATE')
        self.assertEqual(r['LP_calls'], 1)

    def test_timeout_consumes_key_and_stops(self):
        r = self.run_packet(FakeTransport(failure='WALL_CAP'))
        self.assertEqual(r['classification'], 'G1_RESOURCE_INCONCLUSIVE')
        self.assertEqual(r['LP_calls'], 1); self.assertEqual(r['verification_calls'], 8)

    def test_marker_prevents_retry_and_preserves_existing_bytes(self):
        self.run_packet(FakeTransport())
        before = {p: p.read_bytes() for p in self.folder.rglob('*') if p.is_file()}
        with self.assertRaises(PermissionError): self.run_packet(FakeTransport())
        self.assertEqual(before, {p: p.read_bytes() for p in self.folder.rglob('*') if p.is_file()})

    def test_duplicate_key_rejected_before_marker(self):
        self.contract['LP_order'][-1] = 'MOCK_0'
        with self.assertRaises(ValueError): self.run_packet(FakeTransport())
        self.assertFalse(self.store.marker.exists())

    def test_partial_key_set_rejected_before_marker(self):
        del self.fixtures['MOCK_7']
        with self.assertRaises(ValueError): self.run_packet(FakeTransport())
        self.assertFalse(self.store.marker.exists())

    def test_post_marker_exception_stops_without_retry(self):
        t = FakeTransport(); t.acquire = lambda *a, **k: (_ for _ in ()).throw(OSError('mock process failure'))
        r = self.run_packet(t)
        self.assertEqual(r['classification'], c.TECH)
        self.assertEqual(r['echo_only_calls'], 1); self.assertEqual(r['LP_calls'], 0)
        self.assertTrue(self.store.stop.exists())

    def test_final_output_cap_does_not_leave_success(self):
        t = FakeTransport(); t.final_check = lambda: (_ for _ in ()).throw(c.ResourceLimit('OUTPUT_CAP'))
        r = self.run_packet(t)
        self.assertEqual(r['classification'], 'G1_RESOURCE_INCONCLUSIVE')
        self.assertEqual(json.loads((self.store.artifact/'result.json').read_text())['classification'], 'G1_RESOURCE_INCONCLUSIVE')


class LaunchFactsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='g1_source_mock_'); self.addCleanup(self.temp.cleanup)
        self.root, self.source = Path(self.temp.name), 'a'*40
        self.file = self.root/'frozen.txt'; self.file.write_text('immutable mock payload')
        manifest = self.root/c.SOURCE_MANIFEST; manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({'frozen_files': {'frozen.txt': c.sha(self.file)}, 'protected_unchanged_files': {}}))
        prep = self.root/c.PREP; prep.mkdir(parents=True)
        runtime = {'binary': {'path': str(self.file), 'sha256': c.sha(self.file)},
                   'python': {'path': sys.executable, 'sha256': c.sha(sys.executable)}, 'static_libraries': {}, 'libraries': []}
        (prep/'runtime_identity_v1.json').write_text(json.dumps(runtime))
        self.marker = self.root/'private'/'one_shot_consumed.json'
        packet = {'state': {'one_shot_marker': str(self.marker), 'stop_file': str(self.root/'private'/'STOP.json'), 'result_artifact_relative': 'result'}}
        (prep/'decision_packet_contract_v1.json').write_text(json.dumps(packet))
        self.answers = {('rev-parse', 'HEAD'): self.source, ('status', '--porcelain'): '',
                        ('remote', 'get-url', 'origin'): 'git@github.com:HIROMU1015/Partially-Randomized-Trotter.git',
                        ('branch', '--show-current'): 'track-b-mock',
                        ('ls-remote', '--heads', 'origin', 'refs/heads/track-b-mock'): self.source+'\trefs/heads/track-b-mock'}

    def call(self, text=None):
        def fake_git(args, **kw): return self.answers[tuple(args[1:])]
        with patch.object(c.subprocess, 'check_output', side_effect=fake_git):
            return c.checked_source(self.root, self.source, text)

    def test_valid_read_only_gate_creates_no_marker(self):
        self.call(); self.assertFalse(self.marker.exists())

    def test_full_sha_required(self):
        with self.assertRaises(PermissionError): c.checked_source(self.root, 'aaaa')

    def test_head_mismatch(self):
        self.answers[('rev-parse', 'HEAD')] = 'b'*40
        with self.assertRaises(PermissionError): self.call()

    def test_remote_mismatch(self):
        key = ('ls-remote', '--heads', 'origin', 'refs/heads/track-b-mock'); self.answers[key] = 'b'*40
        with self.assertRaises(PermissionError): self.call()

    def test_dirty_worktree(self):
        self.answers[('status', '--porcelain')] = ' M source.py'
        with self.assertRaises(PermissionError): self.call()

    def test_wrong_repository(self):
        self.answers[('remote', 'get-url', 'origin')] = 'git@github.com:quration/repository.git'
        with self.assertRaises(PermissionError): self.call()

    def test_frozen_hash_change(self):
        self.file.write_text('changed')
        with self.assertRaises(PermissionError): self.call()

    def test_instruction_must_match_fixed_source_and_scope(self):
        with self.assertRaises(PermissionError): self.call('go ahead')
        self.call(c.APPROVAL_SENTENCE.format(source=self.source))
        self.assertFalse(self.marker.exists())

    def test_consumed_marker_is_refused(self):
        self.marker.parent.mkdir(); self.marker.write_text('consumed')
        with self.assertRaises(PermissionError): self.call()

    def test_existing_result_is_refused(self):
        (self.root/'result').mkdir()
        with self.assertRaises(PermissionError): self.call()


class GuardIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='g1_guard_mock_'); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        manifest = self.root/c.SOURCE_MANIFEST; manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({'frozen_files': {}, 'protected_unchanged_files': {}, 'output_accounting_extra_roots': []}))
        old_guard = self.root/'scripts/tracks/algorithm_codesign/exact_backend_pilot_v2/guard.py'
        old_guard.parent.mkdir(parents=True); old_guard.symlink_to(ROOT/'scripts/tracks/algorithm_codesign/exact_backend_pilot_v2/guard.py')
        self.store = c.Store(self.root/'private', self.root/'artifact')
        self.store.start({'off_domain_mock_test': True})
        self.contract = {'resources': {'total_wall_seconds_from_exclusive_marker': 10,
                         'RSS_bytes': 268435456, 'address_space_bytes': 268435456,
                         'new_output_bytes': 67108864, 'sample_seconds': '1/40'}}
        self.transport = c.GuardedTransport(self.root, self.contract, {'python': {'path': sys.executable}}, self.store)

    def test_actual_unchanged_guard_wraps_mock_output(self):
        record, data = self.transport.launch('mock_ok', [sys.executable, '-B', '-c', 'print("{\\"mock\\":true}")'], 2)
        self.assertEqual(c.guard_failure(record), (None, None)); self.assertEqual(data, {'mock': True})
        self.assertGreaterEqual(record['CPU_seconds'], 0); self.assertEqual(record['residual_processes'], 0)

    def test_actual_guard_timeout_stops_mock_process(self):
        record, _ = self.transport.launch('mock_timeout', [sys.executable, '-B', '-c', 'import time;time.sleep(1)'], .06)
        self.assertEqual(c.guard_failure(record)[0], 'G1_RESOURCE_INCONCLUSIVE')
        self.assertEqual(record['residual_processes'], 0); self.assertTrue(self.store.stop.exists())

    def test_actual_exit_one_with_json_verdict(self):
        record, data = self.transport.launch('mock_bad_proof', [sys.executable, '-B', '-c', 'import sys;print("{\\"PASS\\":false}");sys.exit(1)'], 2)
        self.assertEqual(c.verifier_guard_failure(record, data), (None, None))
        self.assertEqual(c.verdict_gate(data)[0], 'G1_BACKEND_INVALID_CERTIFICATE')

    def test_actual_non_json_is_technical(self):
        record, data = self.transport.launch('mock_non_json', [sys.executable, '-B', '-c', 'print("mock malformed")'], 2)
        self.assertEqual(c.guard_failure(record), (None, None)); self.assertIsNone(data)

    def test_exceptional_cleanup_preserves_sibling_and_reaps_grandchild(self):
        self.assertEqual(ctypes.CDLL(None).prctl(36, 1, 0, 0, 0), 0)
        decoy = subprocess.Popen([sys.executable, '-B', '-c', 'import time;time.sleep(10)'])
        self.addCleanup(lambda: decoy.wait(timeout=3) if decoy.poll() is None else None)
        self.addCleanup(lambda: decoy.poll() is None and decoy.kill())
        baseline = c.readonly_guard_module().members(-1, os.getpid())
        source = 'import subprocess,sys,time; p=subprocess.Popen([sys.executable,"-B","-c","import time;time.sleep(10)"]);print(p.pid,flush=True);time.sleep(10)'
        proc = subprocess.Popen([sys.executable, '-B', '-c', source], stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        grandchild = int(proc.stdout.readline())
        result = c.cleanup_supervisor(proc, baseline)
        self.assertEqual(result['survivors'], [])
        self.assertIsNone(decoy.poll())
        self.assertIsNone(c.readonly_guard_module().stat(grandchild))
        decoy.kill(); decoy.wait(timeout=3)


class EntryPointTests(unittest.TestCase):
    def test_help_does_not_execute_any_stage(self):
        p = subprocess.run([sys.executable, '-B', str(ROOT/'scripts/tracks/algorithm_codesign/run_g1_decision_packet.py'), '--help'], capture_output=True, text=True, timeout=3)
        self.assertEqual(p.returncode, 0)
        self.assertIn('--execute-one-shot', p.stdout)

    def test_execution_without_instruction_is_refused_before_launch(self):
        p = subprocess.run([sys.executable, '-B', str(ROOT/'scripts/tracks/algorithm_codesign/run_g1_decision_packet.py'),
                            '--source-commit', '0'*40, '--execute-one-shot'], capture_output=True, text=True, timeout=3)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('--instruction-file required', p.stderr)

    def test_standalone_audit_cannot_bypass_controller(self):
        p = subprocess.run([sys.executable, '-B', str(ROOT/'scripts/tracks/algorithm_codesign/audit_g1_structure.py'),
                            '/tmp', '/tmp/no_g1_contract', '/tmp/no_g1_permit'], capture_output=True, text=True, timeout=3)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('wrong audit source root', p.stderr)


if __name__ == '__main__':
    output_path = Path(sys.argv.pop(1)) if len(sys.argv) > 1 else None
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    # A hard preparation barrier: no test may perform the deferred model audit.
    with patch.object(structure, 'audit', side_effect=AssertionError('full structure audit prohibited before explicit one-shot')) as forbidden:
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if forbidden.call_count: raise AssertionError('test attempted the deferred audit')
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({'schema': 'g1_source_focused_off_domain_tests_v1', 'tests_run': result.testsRun,
            'failures': len(result.failures), 'errors': len(result.errors), 'successful': result.wasSuccessful(),
            'full_structure_audit_calls': forbidden.call_count, 'actual_backend_LP_calls': 0,
            'fixed_eight_input_solves': 0, 'science_synthesis_quantum_matrix_GPU': 0,
            'mock_guards_and_Fraction_certificates_only': True}, sort_keys=True, indent=2)+'\n')
    sys.exit(0 if result.wasSuccessful() else 1)
