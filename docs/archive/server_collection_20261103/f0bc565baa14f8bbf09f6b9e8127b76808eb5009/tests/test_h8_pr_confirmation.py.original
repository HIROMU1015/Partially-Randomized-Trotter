"""Artificial metadata/cost fixtures only; no quantum provider/compile imports."""
from copy import deepcopy
from fractions import Fraction as F
import math
import os
from pathlib import Path
import sys
import tempfile
import subprocess
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import h8_pr_confirmation as h
import launch_h8_pr_confirmation as witness


def targets():
    return [dict(scope=scope, cell=dict(id=cid, q=2 if 'P8' in scope else 16))
            for scope, cid in h.SCOPES.items()]


def pairs():
    return [dict(replica=i, seed=i + 10, event_digest=str(i), cohort=h.COHORT,
        axes={axis: dict(RZ=100 + i if axis == 'cosine' else 200 - i,
                        seed=i + 10, event_digest=str(i)) for axis in h.c.core.AXES}) for i in range(48)]


class ArtificialConfirmation(unittest.TestCase):
    def test_exact_independent_96_registry(self):
        result = h.derive_registry(targets(), 'a' * 40, 'b' * 64, [1, 2, 3])
        self.assertEqual(len(result), 96)
        all_seeds = [s for t in result for s in h.hierarchy(t['seed'], t['cell']['q'])]
        self.assertEqual(len(set(all_seeds)), len(all_seeds))
        h.validate_registry(result, targets(), 'a' * 40, 'b' * 64, [1, 2, 3])

    def test_source_and_plan_change_seeds(self):
        original = h.derive_registry(targets(), 'a' * 40, 'b' * 64, [])
        self.assertNotEqual(original, h.derive_registry(targets(), 'c' * 40, 'b' * 64, []))
        self.assertNotEqual(original, h.derive_registry(targets(), 'a' * 40, 'd' * 64, []))

    def test_master_collision_rejected_without_redraw(self):
        task = h.derive_registry(targets(), 'a' * 40, 'b' * 64, [])[0]
        with self.assertRaisesRegex(ValueError, 'COLLISION'):
            h.derive_registry(targets(), 'a' * 40, 'b' * 64, [task['seed']])

    def test_derived_collision_rejected(self):
        task = h.derive_registry(targets(), 'a' * 40, 'b' * 64, [])[0]
        with self.assertRaisesRegex(ValueError, 'COLLISION'):
            h.derive_registry(targets(), 'a' * 40, 'b' * 64, [h.hierarchy(task['seed'], 2)[1]])

    def test_changed_registry_and_candidate_rejected(self):
        result = h.derive_registry(targets(), 'a' * 40, 'b' * 64, [])
        result[0]['seed'] += 1
        with self.assertRaisesRegex(ValueError, 'REGISTRY'):
            h.validate_registry(result, targets(), 'a' * 40, 'b' * 64, [])
        wrong = targets();wrong[0]['cell']['id'] = 'DIFFERENT_CANDIDATE'
        with self.assertRaisesRegex(ValueError, 'FIXED_CONFIRMATION_TARGET'):
            h.derive_registry(wrong, 'a' * 40, 'b' * 64, [])

    def test_paired_covariance_exact_cancellation(self):
        result = h.point_account(pairs(), dict(cosine=1, sine=1), 301)
        self.assertEqual(result['mean_exact'], '300')
        self.assertEqual(result['paired_SE'], 0.)
        self.assertEqual(result['point_decision'], 'POINT_REPRODUCED')
        self.assertIsNone(result['confidence_interval'])

    def test_exact_weighted_mean_and_SE(self):
        values = [1000 - 2 * i for i in range(48)]
        mean = F(sum(values), 48)
        variance = sum((F(v) - mean)**2 for v in values) / 47
        result = h.point_account(pairs(), dict(cosine=2, sine=4), 1000)
        self.assertEqual(F(result['mean_exact']), mean)
        self.assertAlmostEqual(result['paired_SE'], math.sqrt(variance / 48))

    def test_exact_tie_and_not_reproduced(self):
        self.assertEqual(h.point_account(pairs(), dict(cosine=1, sine=1), 300)['point_decision'], 'EXACT_POINT_TIE')
        self.assertEqual(h.point_account(pairs(), dict(cosine=1, sine=1), 299)['point_decision'], 'POINT_NOT_REPRODUCED')

    def test_secondary_ineligible_PF_is_not_zero_or_reselected(self):
        result = h.point_account(pairs(), dict(cosine=1, sine=1), None)
        self.assertEqual(result['point_decision'], 'FIXED_PF_NOT_ELIGIBLE_NO_COMPARISON')
        self.assertIsNone(result['PR_reduction_percent_exact'])

    def test_incomplete_duplicate_pooled_cohort_rejected(self):
        cases = [pairs()[:-1], pairs() + [pairs()[0]], pairs()]
        cases[-1][0]['replica'] = 1
        for values in cases:
            with self.assertRaisesRegex(ValueError, 'COMPLETE_FRESH_48_ONLY'):
                h.point_account(values, dict(cosine=1, sine=1), 300)
        wrong = pairs();wrong[0]['cohort'] = 'EXPLORATION_N8'
        with self.assertRaisesRegex(ValueError, 'COMPLETE_FRESH_48_ONLY'):
            h.point_account(wrong, dict(cosine=1, sine=1), 300)

    def test_axis_seed_event_and_invalid_cost_rejected(self):
        for key, value in [('seed', 10000), ('event_digest', 'not-paired'), ('RZ', 0), ('RZ', float('nan'))]:
            wrong = pairs();wrong[0]['axes']['sine'][key] = value
            with self.assertRaisesRegex(ValueError, 'PAIRED_INTEGER_COST'):
                h.point_account(wrong, dict(cosine=1, sine=1), 300)

    def test_invalid_grant_fails_before_any_input(self):
        with patch.object(h.c, 'read', side_effect=AssertionError('MUST_NOT_LOAD')):
            with self.assertRaisesRegex(ValueError, 'EXECUTION_GRANT'):
                h.authorize({}, dict(approved=False))

    def test_grant_scope_retry_source_binding(self):
        valid = dict(approved=True, approved_by_user=True, runtime_authorization=True,
            one_shot=True, SOURCE='s' * 40, scope='H8_CONFIRM_P8_Q2', technical_digest='t' * 64,
            retry=False, next_stage_authorized=False)
        spec = {key: valid[key] for key in ('SOURCE', 'scope', 'technical_digest')}
        for field, value in [('SOURCE', 'x' * 40), ('scope', 'A_COST'), ('technical_digest', 'x' * 64),
                              ('retry', True), ('next_stage_authorized', True)]:
            grant = dict(valid, **{field: value})
            with self.assertRaises(ValueError):h.authorize(spec, grant)

    def test_first_stop_preserved_and_one_shot_exclusive(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as directory:
            h.c.first_stop(directory, 'FIRST')
            h.c.first_stop(directory, 'SECOND')
            self.assertEqual(h.c.read(Path(directory) / 'STOP.json')['reason'], 'FIRST')
            path = Path(directory) / 'lock'
            h.c.claim_attempt(path, 'a' * 40, 'TEST')
            with self.assertRaises(FileExistsError):h.c.claim_attempt(path, 'a' * 40, 'TEST')

    def test_witness_records_interruption_before_launch(self):
        with patch.object(witness, 'INTERRUPTED', 15):
            with self.assertRaises(KeyboardInterrupt):witness.check_interruption()

    def test_native_wait_and_identity_checked_cleanup_two_minimal_children(self):
        child = subprocess.Popen([sys.executable, '-P', '-B', '-c', 'raise SystemExit(3)'], start_new_session=True)
        receipt = witness.reap(child, block=True)
        self.assertTrue(receipt['reaped'])
        self.assertEqual(receipt['exit_code'], 3)
        sleeper = subprocess.Popen([sys.executable, '-P', '-B', '-c', 'import time;time.sleep(60)'], start_new_session=True)
        owned = witness.identity(sleeper.pid)
        try:
            with self.assertRaisesRegex(ValueError, 'OWN_CHILD_SESSION_ONLY'):
                witness.stop(sleeper, dict(owned, starttime=owned['starttime'] + 1))
        finally:
            receipt = witness.stop(sleeper, owned)
        self.assertTrue(receipt['reaped'])
        self.assertLess(receipt['exit_code'], 0)

    def fixture_native(self):
        cell = dict(id='B2_p8_q2_2nd_r4_K4', prefix=8, q=2, r=4, K=4, R=8, method='B2', formula='2nd')
        task = dict(id=cell['id'] + '_0', cell=cell, replica=0, seed=123)
        law = dict(orders=[0, 2, 4], order_probabilities=[.98, .019, .001])
        target = dict(preparation_sha256='prep', original_law_sha256=h.c.digest(law))
        distribution = SimpleNamespace(to_dict=lambda: law, exact_finite_distribution=1.)
        requests = [SimpleNamespace(seed=s) for s in h.hierarchy(123, 2)[1:]]
        events = [[dict(taylor_order=0) for _ in range(4)] for _ in range(2)]
        class Writer:
            def __init__(self, directory):self.directory = Path(directory)
            def write(self, name, value):h.c.save(self.directory / name, value)
        def control(evolution, request, context, writer, tolerance):
            writer.write('control.json', dict(status='PASS', max_error=0., tolerance=tolerance,
                control_probes=4, independent_event_oracles=1, full_Fock_scalar_once=True,
                reference_rotated=False, ordinary_ancilla_branch=True))
        native = SimpleNamespace(Writer=Writer, steps=lambda *args: (requests, distribution),
            event_values=lambda _: events, control=control)
        measured = SimpleNamespace(rz_count=10, cx_count=20, total_depth=30, rz_depth=5,
            cx_depth=6, circuit_size=40, compiler_settings_hash='compiler', actual_circuit_fingerprint='f' * 64)
        provider = SimpleNamespace(build_native_hadamard_wrapper=lambda *args, **kwargs: 'wrapper',
            canonical_qiskit_circuit_fingerprint=lambda _: 'f' * 64,
            transpile_and_measure_cost=lambda *args, **kwargs: measured)
        context = SimpleNamespace(m=provider, compiler='settings', binding=dict(compiler='compiler'),
            preparations={8: SimpleNamespace(preparation_hash='prep')},
            native=SimpleNamespace(partial_native_from_step_requests=lambda *args, **kwargs: 'evolution'))
        return task, law, target, native, context

    def test_mock_native_pair_control_reservations_and_acceptance(self):
        task, law, target, native, context = self.fixture_native()
        reuse = SimpleNamespace(use_fingerprint_reuse=nullcontext)
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as directory:
            with patch.dict(sys.modules, {'trottertracks.resource_applicability.h6_hash_reuse_v1': reuse}):
                result = h.compile_one(context, task, directory, 's' * 40, target, law, 0., 1e-10, native)
            self.assertEqual(result['cohort'], h.COHORT)
            self.assertEqual(result['calls']['wrapper_compile'], 2)
            self.assertEqual(result['calls']['control_probe'], 4)
            path = Path(directory) / 'task_result.json';h.c.save(path, result)
            ident = [1337, 20, 1000, os.getuid()]
            h.c.save(Path(directory) / 'started.json', dict(task=task['id'], identity=ident))
            handle = dict(directory=directory, reaped=True, exit_code=0,
                          result_sha256=h.c.sha(path), pid=1337, identity=ident)
            spec = dict(SOURCE='s' * 40, binding=context.binding, law=law)
            accepted, _ = h.accept(task, handle, spec)
            self.assertEqual(accepted, result)
            handle['exit_code'] = 2
            with self.assertRaisesRegex(ValueError, 'NATIVE_EXIT'):
                h.accept(task, handle, spec)

    def test_mock_changed_original_law_STOP_before_build(self):
        task, law, target, native, context = self.fixture_native()
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as directory:
            with self.assertRaisesRegex(ValueError, 'ORIGINAL_UNCONDITIONAL_LAW'):
                h.compile_one(context, task, directory, 's' * 40, target, dict(wrong=True), 0., 1e-10, native)
            self.assertFalse((Path(directory) / 'native_evolution_1_reserved.json').exists())


if __name__ == '__main__':
    unittest.main()
