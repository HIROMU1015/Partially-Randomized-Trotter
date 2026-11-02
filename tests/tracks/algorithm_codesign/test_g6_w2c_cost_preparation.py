"""Artificial data/callback tests only. No frozen G7 or F3 module imported."""
from copy import deepcopy
from fractions import Fraction as F
from hashlib import sha256
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location('w2c_aux_only', ROOT/'src/trottertracks/algorithm_codesign/g6_w2c_cost_preparation.py')
M = importlib.util.module_from_spec(SPEC);sys.modules[SPEC.name] = M;SPEC.loader.exec_module(M)


def fake(mode='iid', arm='ordinary'):
    """Witness fixture, not a generated Hamiltonian or a real G7 group formula."""
    p = (F(1, 2), F(1, 2))
    def conditional(i):return ((1-i,), (F(1),), (F(1),))
    return NS(arm=arm, p=p, x=F(1, 4), m=7, H=8, K=8, eta=F(1, 8),
              rho=F(1, 8), chi=F(1, 2),
              groups=(NS(mode=mode, degree=0, square=F(1), ratio=F(1, 9), first=0),),
              roots=(NS(lo=F(1), hi=F(1)),), B=NS(lo=F(1), hi=F(1)),
              label_law=p, group_law=(F(1),), root_child=p,
              conditional=conditional, first_distribution=lambda: (p, p))


def event(raw=()):
    # An artificial algebra record; no generator/provider/IR has been called.
    word = () if raw == (0, 0) else raw
    calls = {1: 2}
    for i in word:calls[i] = calls.get(i, 0)+1
    return {'raw_word': raw, 'word': word, 'child': 1, 'phase_i_power': len(raw)%4,
            'circuit_time_word': tuple(reversed(word)), 'ratio': F(1, 9),
            'proposal': F(1, 2), 'coefficient': F(1, 4), 'weight': F(1, 2),
            'provider_calls': calls}


def request(direction='forward'):
    return {'operator_id': 'ARTIFICIAL_OPERATOR', 'label': 'fixture_label',
            'control_role': 'helper', 'direction': direction,
            'compiler_id': 'ARTIFICIAL_NO_COMPILER', 'gate_set_id': 'fixture_only',
            'precision_id': 'exact_fixture', 'phase_policy': 'STRICT_ABSOLUTE_AND_CONTROLLED_RELATIVE',
            'optimization_id': 'same_fixture_for_all_arms'}


def price(req=None):
    req = req or request()
    return {'request': req, 'key': M.price_key(req), 'status': 'KNOWN',
            'coordinates': {k: (2, 3) for k in M.COORDINATES},
            'artifact_sha256': 'a'*64, 'certificate_sha256': 'b'*64,
            'phase_retained': True, 'strict_error_upper': F(0)}


class ExactGuards(unittest.TestCase):
    def test_seal_deterministic_order(self):
        self.assertEqual(M.fingerprint({'a': 1, 'b': F(2, 3)}), M.fingerprint({'b': F(2, 3), 'a': 1}))
    def test_large_exact_integer_without_global_decimal_change(self):
        M.Seal.create({'n': 1 << 20000}).verify()
    def test_seal_tamper(self):
        s = M.Seal.create({'n': 1})
        with self.assertRaises(M.InvalidEvidence):M.Seal(s.text+' ', s.digest).verify()
    def test_float_rejected(self):
        with self.assertRaises(M.InvalidEvidence):M.Seal.create({'n': 0.1})
    def test_bool_not_integer(self):
        with self.assertRaises(M.InvalidEvidence):M.integer(True)
    def test_named_source_hash(self):
        with tempfile.TemporaryDirectory(prefix='w2c-artificial-source-') as d:
            p = Path(d)/'fixture.txt';p.write_bytes(b'artificial only')
            pins = {'fixture.txt': sha256(p.read_bytes()).hexdigest()}
            M.check_source_files(d, pins)
            p.write_bytes(b'changed artificial file')
            with self.assertRaises(M.InvalidEvidence):M.check_source_files(d, pins)
    def test_source_path_escape_rejected(self):
        with tempfile.TemporaryDirectory(prefix='w2c-artificial-path-') as d:
            with self.assertRaises(M.InvalidEvidence):M.check_source_files(d, {'../outside': 'a'*64})
    def test_law_exact_success(self):
        M.law_witness((F(1, 2), F(1, 2)), ((F(1, 2), F(1, 2)),)*2, 8, F(1, 8)).verify()
    def test_law_not_normalized(self):
        with self.assertRaises(M.InvalidEvidence):M.law_witness((F(1, 3),), ((F(1), F(1)),), 8, F(1, 8))
    def test_law_non_dyadic(self):
        with self.assertRaises(M.InvalidEvidence):M.law_witness((F(1, 3), F(2, 3)), ((F(1, 3), F(1, 3)), (F(2, 3), F(2, 3))), 8, F(1, 8))
    def test_law_relative_precision_failure(self):
        with self.assertRaises(M.InvalidEvidence):M.law_witness((F(1, 4), F(3, 4)), ((F(1, 2), F(1, 2)),)*2, 8, F(1, 8))
    def test_root_exact_and_outward(self):
        M.root_witness(F(1), (F(1), F(1)), 8, F(1, 8))
        M.root_witness(F(3, 2), (F(1), F(3, 2)), 1, F(1, 2))
    def test_root_grid_rejected(self):
        with self.assertRaises(M.InvalidEvidence):M.root_witness(F(3, 2), (F(1), F(5, 3)), 2, F(1, 2))
    def test_root_non_enclosure(self):
        with self.assertRaises(M.InvalidEvidence):M.root_witness(F(2), (F(1), F(1)), 8, F(1, 8))
    def test_root_precision_failure(self):
        with self.assertRaises(M.InvalidEvidence):M.root_witness(F(3, 2), (F(1), F(3, 2)), 1, F(1, 100))


class CanonicalWitnesses(unittest.TestCase):
    def test_ordinary_preflight(self):
        self.assertEqual(M.canonical_snapshot(fake())['conditional_witnesses'], ())
    def test_distinct_preflight_covers_first_and_all_exclusions(self):
        self.assertEqual(len(M.canonical_snapshot(fake('distinct', 'partial_return_tail'))['conditional_witnesses']), 3)
    def test_fixed_first_preflight(self):
        self.assertEqual(len(M.canonical_snapshot(fake('fixed_first', 'closed_P3_tail'))['conditional_witnesses']), 1)
    def test_collected_preflight_requires_frozen_law_builder(self):
        g = fake('root_collected', 'closed_P3_tail')
        with self.assertRaises(M.InvalidEvidence):M.canonical_snapshot(g)
        self.assertEqual(len(M.canonical_snapshot(g, lambda q, h, eta: q)['conditional_witnesses']), 1)
    def test_state_capture_does_not_rebuild_conditional_laws(self):
        g = fake('distinct', 'partial_return_tail')
        g.first_distribution = lambda: self.fail('state capture called first law')
        g.conditional = lambda i: self.fail('state capture called conditional')
        M.canonical_state(g)
    def test_preflight_rejects_tampered_B(self):
        g = fake();g.B.hi = F(2)
        with self.assertRaises(M.InvalidEvidence):M.canonical_snapshot(g)
    def test_preflight_rejects_tampered_law(self):
        g = fake();g.label_law = (F(1, 4), F(3, 4))
        with self.assertRaises(M.InvalidEvidence):M.canonical_snapshot(g)
    def test_preflight_rejects_missing_conditional_support(self):
        g = fake('distinct', 'partial_return_tail');g.conditional = lambda i: ((i,), (F(1),), (F(1),))
        with self.assertRaises(M.InvalidEvidence):M.canonical_snapshot(g)
    def test_full_not_mislabelled_canonical(self):
        with self.assertRaises(M.InvalidEvidence):M.canonical_snapshot(fake(arm='full_return'))
    def test_canonical_reduction_keeps_original_phase(self):
        M.event_witness(event((0, 0)), full=False, degree_cap=7, labels=2)
    def test_full_raw_remapping_rejected(self):
        with self.assertRaises(M.InvalidEvidence):M.event_witness(event((0, 0)), full=True, degree_cap=7, labels=2)
    def test_pi_weight_and_phase_tampering(self):
        for k, value in [('weight', F(1)), ('phase_i_power', 2), ('provider_calls', {})]:
            e = event();e[k] = value
            with self.subTest(k=k), self.assertRaises(M.InvalidEvidence):M.event_witness(e, full=True, degree_cap=7, labels=2)
    def test_same_overlay_for_all_arms(self):
        for arm in M.ARMS:
            with self.subTest(arm=arm):
                a = M.AuditParity(arm, {'fixture': 'only'}, lambda: {'law': (F(1),)}, lambda: 'fixed')
                a.invoke(lambda: {'kind': 'ARTIFICIAL_ZERO' if arm == 'full_return' else 'ARTIFICIAL_EVENT'}).verify()
    def test_identity_and_state_drift_terminal(self):
        state = {'n': 1};called = []
        a = M.AuditParity('ordinary', {}, lambda: state, lambda: 'fixed');state['n'] = 2
        with self.assertRaises(M.InvalidEvidence):a.invoke(lambda: called.append(1))
        with self.assertRaises(M.InvalidEvidence):a.invoke(lambda: called.append(1))
        self.assertEqual(called, []);self.assertIsNotNone(a.failure)
    def test_callback_failure_not_random_zero_or_retry(self):
        a = M.AuditParity('full_return', {}, lambda: {}, lambda: 'fixed')
        with self.assertRaises(ZeroDivisionError):a.invoke(lambda: 1//0)
        self.assertIn('is_random_zero', a.failure.text)
        with self.assertRaises(M.InvalidEvidence):a.invoke(lambda: None)


class CostRecords(unittest.TestCase):
    def test_fake_clocks_only_and_exclusive_stages(self):
        clock = iter(((10, 20), (15, 31)));saved = []
        l = M.StageLedger(lambda: next(clock), saved.append);l.begin('raw')
        row = l.end('ZERO_R');self.assertEqual((row['cpu_ns'], row['wall_ns']), (5, 11));self.assertEqual(len(saved), 1)
    def test_nested_stage_rejected(self):
        l = M.StageLedger(lambda: (0, 0), lambda s: None);l.begin('raw')
        with self.assertRaises(M.InvalidEvidence):l.begin('packet')
    def test_failure_prefix_preserved_and_terminal(self):
        l = M.StageLedger(lambda: (0, 0), lambda s: None);l.begin('packet');l.end('FAIL')
        self.assertEqual(len(l.records), 1)
        with self.assertRaises(M.InvalidEvidence):l.begin('retry')
    def test_failed_sink_preserves_in_memory_prefix(self):
        def fail(s):raise OSError('artificial sink failure')
        l = M.StageLedger(lambda: (0, 0), fail);l.begin('output')
        with self.assertRaises(OSError):l.end('OK')
        self.assertTrue(l.stopped);self.assertEqual(len(l.records), 1)
    def test_backwards_clock_rejected(self):
        clock = iter(((10, 20), (9, 21)));l = M.StageLedger(lambda: next(clock), lambda s: None);l.begin('raw')
        with self.assertRaises(M.InvalidEvidence):l.end('OK')
        self.assertTrue(l.stopped);self.assertEqual(l.records[0]['outcome'], 'FAIL')
    def test_valid_coin_zero_preserves_paid_packet_stage(self):
        l = M.StageLedger(lambda: (0, 0), lambda s: None)
        for stage, outcome in [('raw', 'OK'), ('packet_including_child_law', 'OK'), ('coin', 'ZERO_C')]:
            l.begin(stage);l.end(outcome)
        self.assertEqual([r['stage'] for r in l.records], ['raw', 'packet_including_child_law', 'coin'])


class ProviderGuards(unittest.TestCase):
    def test_external_JSON_price_decode(self):
        r = price();r['coordinates'] = {k: [2, 3] for k in M.COORDINATES};r['strict_error_upper'] = '0'
        self.assertEqual(M.price_row(M.decode_price_record(r), request(), error_allowance=F(0)), 'COMPLETE')
    def test_external_float_and_nonrational_error_rejected(self):
        for value in (0.1, '0.1', '1/0', '-1/2'):
            r = price();r['coordinates'] = {k: [2, 3] for k in M.COORDINATES};r['strict_error_upper'] = value
            with self.subTest(value=value), self.assertRaises(M.InvalidEvidence):M.decode_price_record(r)
    def test_complete_artificial_price(self):
        self.assertEqual(M.price_row(price(), request(), error_allowance=F(0)), 'COMPLETE')
    def test_unknown_is_not_free(self):
        r = price();r['status'] = 'UNKNOWN';r['coordinates'] = {k: None for k in M.COORDINATES}
        self.assertEqual(M.price_row(r, request(), error_allowance=F(0)), 'UNRESOLVED')
        r['coordinates']['T'] = (0, 0)
        with self.assertRaises(M.InvalidEvidence):M.price_row(r, request(), error_allowance=F(0))
    def test_partial_depth_stays_missing(self):
        r = price();r['coordinates']['serial_depth'] = None
        self.assertEqual(M.price_row(r, request(), error_allowance=F(0)), 'PARTIAL')
    def test_all_key_identity_fields_invalidate_cache(self):
        for k in M.PRICE_KEY_FIELDS:
            with self.subTest(k=k):
                q = request();q[k] += '_changed'
                with self.assertRaises(M.InvalidEvidence):M.price_row(price(), q, error_allowance=F(0))
    def test_actual_adjoint_requires_parent_and_verification(self):
        q = request('actual_adjoint');r = price(q)
        with self.assertRaises(M.InvalidEvidence):M.price_row(r, q, error_allowance=F(0))
        r.update(forward_artifact_sha256='c'*64, actual_adjoint_verified=True)
        self.assertEqual(M.price_row(r, q, error_allowance=F(0)), 'COMPLETE')
    def test_positive_CQ_error_not_silently_absorbed(self):
        r = price();r['strict_error_upper'] = F(1, 100)
        with self.assertRaises(M.InvalidEvidence):M.price_row(r, request(), error_allowance=F(0))
    def test_phase_or_certificate_missing(self):
        for k, value in [('phase_retained', False), ('artifact_sha256', 'x'), ('certificate_sha256', None)]:
            r = price();r[k] = value
            with self.subTest(k=k), self.assertRaises(M.InvalidEvidence):M.price_row(r, request(), error_allowance=F(0))
    def test_artificial_RZ_preserves_W_and_no_numerical_check(self):
        sequence = 'HTWt';r = {'key': 'artificial', 'epsilon': '1/100', 'sequence': sequence,
            'sequence_sha256': sha256(sequence.encode()).hexdigest(), 'T_count': 2,
            'Tdagger_count': 1, 'global_W_count': 1, 'one_qubit_count': 3,
            'error_pass': True, 'strict_operator_error_upper': '1/1000'}
        status = M.rz_cache_witness(r, key='artificial', epsilon=F(1, 100))
        self.assertTrue(status['W_scalar_retained']);self.assertFalse(status['independent_numerical_error_check_performed'])
        r['global_W_count'] = 0
        with self.assertRaises(M.InvalidEvidence):M.rz_cache_witness(r, key='artificial', epsilon=F(1, 100))
    def test_candidate_is_never_execution_authority(self):
        c = {'schema': 'g6_w2c_b3_c1_candidate_v1', 'arms': list(M.ARMS), 'adopted': False,
             'contract_grants_execution': False, 'decisions_required': {'provider': None}}
        self.assertEqual(len(M.execution_blockers(c)), 3)
        for adopted in (False, True):
            c['adopted'] = adopted
            with self.assertRaises(PermissionError):M.reject_real_execution(c)
    def test_candidate_forged_grant_rejected(self):
        c = {'schema': 'g6_w2c_b3_c1_candidate_v1', 'arms': list(M.ARMS), 'adopted': False,
             'contract_grants_execution': True, 'decisions_required': {}}
        with self.assertRaises(M.InvalidEvidence):M.execution_blockers(c)


if __name__ == '__main__':unittest.main()
