"""New readiness checks. No real G7/F3/kernel execution or bitstream generation."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from fractions import Fraction as F
from hashlib import sha256
import importlib
import json
import multiprocessing
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))
M = importlib.import_module('trottertracks.algorithm_codesign.g6_w2c_b3_readiness')
D = M.D
BASE = json.loads((ROOT/'docs/research/track_b_g6_w2c_b3_driver_preparation_20261019/execution_contract_candidate.json').read_text())


class Clock:
    def __init__(self):self.n = 0
    def __call__(self):self.n += 10;return self.n, 2*self.n


def event():
    return {'raw_word': (0, 1), 'word': (0, 1), 'child': 2, 'phase_i_power': 2,
            'circuit_time_word': (1, 0), 'ratio': F(1, 9), 'proposal': F(1, 2),
            'coefficient': F(1, 4), 'weight': F(1, 2), 'provider_calls': {0: 1, 1: 1, 2: 2}}


def plan(scope='ARTIFICIAL_SCOPE_ONE', claim='ARTIFICIAL_CLAIM_ONE'):
    return {'artificial_only': True, 'scope': scope, 'claim_id': claim,
            'angle_adopted_in_fixture': True, 'angle': deepcopy(M.ANGLE_PROFILES[0]),
            'seed_namespace': 'ARTIFICIAL_BIT_VALUES_NO_G7_STREAM', 'mock_trial_budget': 3,
            'bindings': {key: sha256(('ARTIFICIAL_'+key).encode()).hexdigest() for key in M.MOCK_BINDING_FIELDS},
            'retry': 0, 'resume': False}


def authorization(p):
    return {'fixture_only': True, 'approved_in_fixture': True, 'consumed_in_fixture': False,
            'scope': p['scope'], 'contract_digest': p['bindings']['contract_digest']}


def session(root, p=None):
    p = p or plan()
    return M.MockAtomicScope(root).admit_mock(p, deepcopy(p['bindings']), authorization(p), Clock())


def process_attempt(root):
    try:
        s = session(root);s.stop_mock('artificial process race winner')
        return True
    except M.ReadinessStop:return False


class AngleCandidates(unittest.TestCase):
    def test_three_parameter_only_sufficient_inequalities(self):
        ceilings = (20, 37, 53)
        for p, b in zip(M.ANGLE_PROFILES, ceilings):
            with self.subTest(profile=p['id']):
                r = M.angle_bound(p)
                self.assertTrue(r['sufficient']);self.assertLess(r['radius_upper'], F(1, 1 << b))
                self.assertLess(F(1, 1 << b), r['tau'])
    def test_radius_certificate_is_not_physical_error_allocation(self):
        r = M.angle_bound(M.ANGLE_PROFILES[0])
        self.assertFalse(r['native_error_certificate']);self.assertFalse(r['G7_event_evaluated'])
    def test_invalid_K_J_tau_and_physical_scope_rejected(self):
        for key, value in (('K_phi', 2), ('J', True), ('tau', '0'), ('mode', 'PHYSICAL_RZ')):
            p = deepcopy(M.ANGLE_PROFILES[0]);p[key] = value
            with self.subTest(key=key), self.assertRaises(M.ReadinessStop):M.angle_bound(p)
    def test_harsher_unproved_tolerance_is_not_accepted(self):
        p = deepcopy(M.ANGLE_PROFILES[0]);p['tau'] = '1/1000000000000000000000000000000'
        self.assertFalse(M.angle_bound(p)['sufficient'])


class ContractChecks(unittest.TestCase):
    def test_refined_draft_passes_without_adopting_or_modifying_base(self):
        before = D.identity(BASE);c = M.refine_draft(BASE)
        r = M.validate_refined(c, BASE)
        self.assertTrue(r['technical_candidate_consistent']);self.assertEqual(D.identity(BASE), before)
        self.assertFalse(c['adopted']);self.assertFalse(c['separate_execution_approval'])
        changed = deepcopy(BASE);changed['input']['m'] = 7
        with self.assertRaises(M.ReadinessStop):M.refine_draft(changed)
        with self.assertRaises(M.ReadinessStop):M.validate_refined(c, changed)
    def test_source_input_policy_N_cell_and_call_budget_mismatch_rejected(self):
        edits = (('input', 'm', 7), ('policy', 'root_bits', 128),
                 ('policy', 'sigma', True), ('call_budget', 'budget_recomputation_calls', False),
                 ('call_budget', 'total_sampling_calls', 31710849),
                 ('source_pins', 'src/trottertracks/algorithm_codesign/g7_generator.py', 'a'*64))
        for block, field, value in edits:
            c = M.refine_draft(BASE);c['fixed_science'][block][field] = value
            with self.subTest(block=block), self.assertRaises(M.ReadinessStop):M.validate_refined(c, BASE)
        for mode in ('N', 'missing_cell', 'axis'):
            c = M.refine_draft(BASE)
            if mode == 'missing_cell':c['fixed_science']['cells'].pop()
            else:c['fixed_science']['cells'][0][mode] = 0
            with self.subTest(mode=mode), self.assertRaises(M.ReadinessStop):M.validate_refined(c, BASE)
    def test_complete_input_source_contract_hash_binding(self):
        for key in ('base_contract_sha256', 'input_sha256', 'sources_sha256', 'cells_sha256', 'call_budget_sha256'):
            c = M.refine_draft(BASE);c['binding_manifest'][key] = 'f'*64
            with self.subTest(key=key), self.assertRaises(M.ReadinessStop):M.validate_refined(c, BASE)
    def test_seed_old_namespace_or_changed_pooling_policy_rejected(self):
        c = M.refine_draft(BASE);c['proposed_values']['seed']['namespace'] = 'G7-development-bitstream-v1'
        with self.assertRaises(M.ReadinessStop):M.validate_refined(c, BASE)
    def test_cache_output_limit_and_flush_changes_rejected(self):
        for key, value in (('cache', 'NEW_MEMO'), ('result_bytes', 1), ('log_bytes', 1), ('flush_rows', 1)):
            c = M.refine_draft(BASE);c['proposed_values']['schedule_and_output'][key] = value
            with self.subTest(key=key), self.assertRaises(M.ReadinessStop):M.validate_refined(c, BASE)
    def test_fixture_angle_is_not_a_science_candidate(self):
        c = M.refine_draft(BASE)
        c['proposed_values']['angle_profiles'][0].update(tau='1/16', K_phi=8, J=4)
        with self.assertRaises(M.ReadinessStop):M.validate_refined(c, BASE)
    def test_draft_flags_cannot_become_authority(self):
        for key in ('adopted', 'contract_grants_execution', 'separate_execution_approval'):
            c = M.refine_draft(BASE);c[key] = True
            with self.subTest(key=key), self.assertRaises(M.ReadinessStop):M.validate_refined(c, BASE)
    def test_science_entry_always_denies_even_forged_adoption(self):
        c = M.refine_draft(BASE);c['adopted'] = True
        with self.assertRaises(PermissionError):M.reject_science(c, authorization=True)
        with self.assertRaises(D.Stop):D.Boundary().enter(BASE)
    def test_source_snapshot_mismatch_and_escape_rejected(self):
        with tempfile.TemporaryDirectory(prefix='b3-artificial-') as d:
            p = Path(d)/'fixture';p.write_text('only artificial source')
            M.check_source_snapshot(d, {'fixture': sha256(p.read_bytes()).hexdigest()})
            with self.assertRaises(D.audit.InvalidEvidence):M.check_source_snapshot(d, {'fixture': 'a'*64})
            with self.assertRaises(D.audit.InvalidEvidence):M.check_source_snapshot(d, {'../escape': 'a'*64})
    def test_no_scientific_modules_loaded(self):
        for name in ('g7_generator', 'g6_f3_generator', 'g6_f3_precision', 'return_aggregation'):
            self.assertNotIn('trottertracks.algorithm_codesign.'+name, sys.modules)


class AtomicFixture(unittest.TestCase):
    def test_fixture_root_only_and_no_production_port_method(self):
        with self.assertRaises(M.ReadinessStop):M.MockAtomicScope(ROOT)
        with tempfile.TemporaryDirectory(prefix='b3-artificial-') as d:
            self.assertFalse(hasattr(M.MockAtomicScope(d), 'admit_once'))
    def test_unapproved_or_old_authorization_and_digest_rejected_before_claim(self):
        for field, value in (('approved_in_fixture', False), ('consumed_in_fixture', True),
                             ('contract_digest', 'a'*64), ('scope', 'ARTIFICIAL_SCOPE_WRONG')):
            with self.subTest(field=field), tempfile.TemporaryDirectory(prefix='b3-artificial-') as d:
                p = plan();a = authorization(p);a[field] = value
                with self.assertRaises(M.ReadinessStop):M.MockAtomicScope(d).admit_mock(p, p['bindings'], a, Clock())
                self.assertEqual(list(Path(d).iterdir()), [])
    def test_binding_mismatches_rejected_before_atomic_creation(self):
        for key in M.MOCK_BINDING_FIELDS:
            with self.subTest(key=key), tempfile.TemporaryDirectory(prefix='b3-artificial-') as d:
                p = plan();o = deepcopy(p['bindings']);o[key] = 'f'*64
                with self.assertRaises(M.ReadinessStop):M.MockAtomicScope(d).admit_mock(p, o, authorization(p), Clock())
                self.assertEqual(list(Path(d).iterdir()), [])
    def test_unadopted_angle_invalid_seed_and_retry_resume_rejected(self):
        for key, value in (('angle_adopted_in_fixture', False), ('seed_namespace', 'B3_FULL_N_NEW_20261019_V1'),
                           ('retry', 1), ('resume', True), ('claim_id', 'B3_NEW_REAL_FORBIDDEN')):
            with self.subTest(key=key), tempfile.TemporaryDirectory(prefix='b3-artificial-') as d:
                p = plan();p[key] = value
                with self.assertRaises(M.ReadinessStop):session(d, p)
                self.assertEqual(list(Path(d).iterdir()), [])
    def test_scope_survives_new_adapter_and_changed_contract(self):
        with tempfile.TemporaryDirectory(prefix='b3-artificial-') as d:
            s = session(d);s.stop_mock('mock stop')
            with self.assertRaises(M.ReadinessStop):session(d)
            p = plan();p['bindings']['contract_digest'] = 'b'*64
            with self.assertRaises(M.ReadinessStop):session(d, p)
    def test_old_claim_id_cannot_be_used_under_new_scope(self):
        with tempfile.TemporaryDirectory(prefix='b3-artificial-') as d:
            s = session(d);s.stop_mock()
            with self.assertRaises(M.ReadinessStop):session(d, plan(scope='ARTIFICIAL_SCOPE_TWO'))
    def test_atomic_duplicate_start_race_has_exactly_one_winner(self):
        with tempfile.TemporaryDirectory(prefix='b3-artificial-') as d:
            def attempt(i):
                try:return session(d)
                except M.ReadinessStop:return None
            with ThreadPoolExecutor() as pool:results = list(pool.map(attempt, range(32)))
            winners = [s for s in results if s is not None]
            self.assertEqual(len(winners), 1);winners[0].stop_mock()
        with tempfile.TemporaryDirectory(prefix='b3-artificial-') as d:
            with multiprocessing.get_context('fork').Pool() as pool:
                winners = pool.map(process_attempt, [d]*32)
            self.assertEqual(sum(winners), 1)
    def test_partial_scope_write_burns_scope_without_retry(self):
        with tempfile.TemporaryDirectory(prefix='b3-artificial-') as d:
            with patch.object(M.os, 'write', side_effect=OSError('artificial interrupted write')):
                with self.assertRaises(OSError):session(d)
            self.assertEqual(len(list(Path(d).glob('mock_scope_*'))), 1)
            with self.assertRaises(M.ReadinessStop):session(d)
    def test_output_collision_preserves_original_and_scope_stays_spent(self):
        with tempfile.TemporaryDirectory(prefix='b3-artificial-') as d:
            key = sha256(plan()['scope'].encode()).hexdigest();p = Path(d)/('mock_prefix_'+key+'.jsonl')
            p.write_bytes(b'preexisting artificial output')
            with self.assertRaises(FileExistsError):session(d)
            self.assertEqual(p.read_bytes(), b'preexisting artificial output')
            with self.assertRaises(M.ReadinessStop):session(d)


class PrefixAndStop(unittest.TestCase):
    def test_completed_mock_counts_phase_order_weight_and_terminal_stop(self):
        with tempfile.TemporaryDirectory(prefix='b3-artificial-') as d:
            s = session(d)
            for kind, e in (('ZERO_R', None), ('ZERO_C', None), ('EVENT', event())):s.record_mock_trial(kind, e)
            r = s.finish_mock()
            self.assertEqual(r['completed_trials'], 3);self.assertGreater(r['fake_cpu_wall_prefix_ns'][0], 0)
            e = s.completed[-1]['event'];self.assertEqual(e['phase_i_power'], 2)
            self.assertEqual(e['circuit_time_word'], (1, 0));self.assertEqual(e['weight'], F(1, 2))
            with self.assertRaises(M.ReadinessStop):s.record_mock_trial('EVENT', event())
            with self.assertRaises(M.ReadinessStop):s.finish_mock()
    def test_partial_failure_keeps_completed_prefix_and_nonzero_fake_spend(self):
        with tempfile.TemporaryDirectory(prefix='b3-artificial-') as d:
            s = session(d);s.record_mock_trial('ZERO_R')
            with self.assertRaises(D.Stop):s.record_mock_trial('TYPED_FAILURE')
            self.assertEqual(s.state, 'STOPPED');self.assertEqual(len(s.completed), 1)
            self.assertEqual(len(s.stream.path.read_text().splitlines()), 1)
            self.assertGreater(s.failure['fake_cpu_wall_prefix_ns'][0], 0)
            self.assertFalse(s.failure['is_random_zero'])
    def test_save_failure_does_not_replace_successful_prefix_or_resume(self):
        with tempfile.TemporaryDirectory(prefix='b3-artificial-') as d:
            s = session(d);s.record_mock_trial('ZERO_R');before = s.stream.path.read_bytes()
            class FailingFile:
                def write(self, data):raise OSError('artificial output failure')
                def flush(self):return original.flush()
                def close(self):return original.close()
            original = s.stream.file;s.stream.file = FailingFile()
            with self.assertRaises(OSError):s.record_mock_trial('EVENT', event())
            self.assertEqual(s.stream.path.read_bytes(), before);self.assertEqual(s.state, 'STOPPED')
            self.assertIsNotNone(s.failure['save_failure']);self.assertGreater(s.spent[0], 0)
            with self.assertRaises(M.ReadinessStop):s.record_mock_trial('ZERO_C')
    def test_budget_exhaustion_is_stop_not_extra_sampling(self):
        with tempfile.TemporaryDirectory(prefix='b3-artificial-') as d:
            s = session(d)
            for i in range(3):s.record_mock_trial('ZERO_R')
            with self.assertRaises(M.ReadinessStop):s.record_mock_trial('EVENT', event())
            self.assertEqual(len(s.completed), 3);self.assertEqual(s.state, 'STOPPED')
    def test_no_accepted_finish_stops_without_coverage_topup(self):
        with tempfile.TemporaryDirectory(prefix='b3-artificial-') as d:
            s = session(d)
            for i in range(3):s.record_mock_trial('ZERO_C')
            with self.assertRaises(M.ReadinessStop):s.finish_mock()
            self.assertEqual(len(s.completed), 3);self.assertEqual(s.state, 'STOPPED')
    def test_original_correctness_gate_stops_after_16_zero_trials(self):
        class ZeroEndpoint:
            calls = 0
            def shared_identity(self):return 'artificial fixed identity'
            def sample(self, bits, trial):self.calls += 1;return D.Outcome('ZERO_R')
        with tempfile.TemporaryDirectory(prefix='b3-artificial-') as d:
            m = D.Meter(Clock());stream = D.Stream(Path(d)/'mock.jsonl', m, byte_limit=100000, flush_rows=1)
            cells = [{'arm': 'full_return', 'layer': 'A', 'axis': axis, 'N': 16} for axis in D.AXES]
            e = ZeroEndpoint()
            with self.assertRaises(D.Stop):
                D.run_pair(e, lambda axis: None, cells, stream, m, phase='correctness',
                           pass_name='baseline', correctness_trials=16)
            stream.close();self.assertEqual(e.calls, 16)
    def test_plan_drift_is_terminal_with_previous_prefix(self):
        with tempfile.TemporaryDirectory(prefix='b3-artificial-') as d:
            s = session(d);s.record_mock_trial('ZERO_R');s.plan['mock_trial_budget'] = 9
            with self.assertRaises(M.ReadinessStop):s.record_mock_trial('ZERO_R')
            self.assertEqual(len(s.completed), 1);self.assertEqual(s.state, 'STOPPED')


if __name__ == '__main__':unittest.main()
